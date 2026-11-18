"""CPU-only EWC primitives for synthetic continual-learning experiments."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec
from ppfcl_nids.federated_learning.initial_aggregation import validate_parameters
from ppfcl_nids.models.neural_network import FlowMLP

EWC_SPEC = FlowModelSpec(BIFLOW_CONTRACT_ID)
EWC_POLICY = "online-empirical-diagonal-v1"
MAX_FISHER_SAMPLES = 256


@dataclass(frozen=True, slots=True)
class EWCConfig:
    strength: float = 1.0
    decay: float = 1.0
    fisher_samples: int = 64

    def __post_init__(self) -> None:
        if (type(self.strength) not in (float, int) or not math.isfinite(self.strength) or self.strength < 0
                or type(self.decay) not in (float, int) or not math.isfinite(self.decay) or not 0 < self.decay <= 1
                or type(self.fisher_samples) is not int or not 1 <= self.fisher_samples <= MAX_FISHER_SAMPLES):
            raise ValueError("Invalid synthetic EWC configuration")


@dataclass(frozen=True, slots=True)
class EWCState:
    theta_star: dict[str, Tensor]
    fisher: dict[str, Tensor]
    config: EWCConfig


def model_template() -> dict[str, Tensor]:
    with torch.device("meta"):
        return FlowMLP(EWC_SPEC).state_dict()


def _parameters(model: FlowMLP) -> dict[str, Tensor]:
    if not isinstance(model, FlowMLP) or model.spec != EWC_SPEC:
        raise ValueError("EWC requires the 11-feature model architecture")
    parameters = dict(model.named_parameters())
    validate_parameters(parameters, model_template())
    if not all(p.requires_grad for p in parameters.values()):
        raise ValueError("EWC requires the full trainable model")
    return parameters


def validate_state(state: EWCState, template: dict[str, Tensor]) -> None:
    if (not isinstance(state, EWCState) or not isinstance(state.config, EWCConfig)
            or not isinstance(state.theta_star, dict) or not isinstance(state.fisher, dict)):
        raise ValueError("Invalid EWC state")
    validate_parameters(state.theta_star, template)
    validate_parameters(state.fisher, template)
    if any(t.requires_grad for t in (*state.theta_star.values(), *state.fisher.values())):
        raise ValueError("EWC reference and Fisher must be detached")
    if any((value < 0).any() for value in state.fisher.values()):
        raise ValueError("EWC Fisher diagonals must be nonnegative")


def fisher_population(features: Tensor, targets: Tensor, config: EWCConfig) -> tuple[Tensor, Tensor]:
    """Select the bounded, deterministic prefix of a synthetic calibration population."""
    if (features.ndim != 2 or features.shape[1] != 11 or targets.shape != (len(features),)
            or not len(features) or any(t.device.type != "cpu" or t.dtype != torch.float32
                                        for t in (features, targets))):
        raise ValueError("Expected CPU float32 synthetic features and binary labels")
    x = features[:config.fisher_samples].detach().contiguous().clone()
    y = targets[:config.fisher_samples].detach().contiguous().clone()
    if not torch.isfinite(x).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("Nonfinite synthetic features or invalid binary labels")
    return x, y


def estimate_fisher(model: FlowMLP, features: Tensor, targets: Tensor,
                    config: EWCConfig = EWCConfig()) -> dict[str, Tensor]:
    """Average squared individual BCE gradients at the current model parameters."""
    parameters = _parameters(model)
    x, y = fisher_population(features, targets, config)
    totals = {name: torch.zeros_like(p, dtype=torch.float64) for name, p in parameters.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            for index in range(len(x)):
                logits = model(x[index:index + 1])
                loss = binary_cross_entropy_with_logits(logits, y[index:index + 1])
                gradients = torch.autograd.grad(loss, tuple(parameters.values()))
                for name, gradient in zip(parameters, gradients, strict=True):
                    totals[name].add_(gradient.detach().double().square())
    finally:
        model.train(was_training)
    result = {name: (value / len(x)).float() for name, value in totals.items()}
    validate_parameters(result, parameters)
    return result


def capture_state(model: FlowMLP, features: Tensor, targets: Tensor, *, config: EWCConfig = EWCConfig(),
                  previous: EWCState | None = None) -> EWCState:
    """Advance online EWC, accumulating previous importance at the new anchor."""
    parameters = _parameters(model)
    if previous is not None:
        validate_state(previous, parameters)
        if previous.config != config:
            raise ValueError("EWC configuration changed within the session")
    fisher = estimate_fisher(model, features, targets, config)
    if previous is not None:
        fisher = {name: current + config.decay * previous.fisher[name] for name, current in fisher.items()}
    state = EWCState({name: p.detach().clone() for name, p in parameters.items()}, fisher, config)
    validate_state(state, parameters)
    return state


def ewc_penalty(model: FlowMLP, state: EWCState) -> Tensor:
    """Return the lambda-scaled quadratic penalty for an ordinary synthetic optimizer."""
    parameters = _parameters(model)
    validate_state(state, parameters)
    if state.config.strength == 0:
        return next(iter(parameters.values())).new_zeros(())
    penalty = sum((state.fisher[name] * (parameter - state.theta_star[name]).square()).sum()
                  for name, parameter in parameters.items())
    result = penalty * (0.5 * state.config.strength)
    if not torch.isfinite(result):
        raise ValueError("Nonfinite EWC penalty")
    return result


def synthetic_task_loss(model: FlowMLP, features: Tensor, targets: Tensor,
                        state: EWCState | None = None) -> Tensor:
    """Combine binary classification and EWC in the synthetic, non-DP training path."""
    loss = binary_cross_entropy_with_logits(model(features), targets)
    return loss if state is None else loss + ewc_penalty(model, state)
