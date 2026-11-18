"""Bounded CPU continual-training kernel. Call only from an approved client-local job."""
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC, EWCState, ewc_penalty, validate_state
from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0
from ppfcl_nids.models.neural_network import FlowMLP


@dataclass(frozen=True)
class ReviewedAnchor:
    """EWC state whose anchor and Fisher have passed independent privacy review."""
    state: EWCState
    review_sha256: str


def evaluate_local(model, batches):
    """Accumulate exact binary counts locally. Undefined ratios remain absent."""
    tp = fp = tn = fn = 0
    model.eval()
    with torch.inference_mode():
        for x, y in batches:
            if (x.ndim != 2 or x.shape[1] != 11 or y.shape != (len(x),)
                    or not torch.isfinite(x).all() or not ((y == 0) | (y == 1)).all()):
                raise ValueError("Invalid local evaluation population")
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise ValueError("Invalid model evaluation output")
            predicted = logits >= 0
            positive = y == 1
            tp += int((predicted & positive).sum())
            fp += int((predicted & ~positive).sum())
            tn += int((~predicted & ~positive).sum())
            fn += int((~predicted & positive).sum())
    ratio = lambda n, d: n / d if d else None
    return {"confusion": [[tn, fp], [fn, tp]], "support": [tn + fp, tp + fn],
            "accuracy": ratio(tp + tn, tp + tn + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def train_local(checkpoint, checkpoint_sha, features, labels, config, ledger, run_id,
                *, anchor=None, on_step=None, on_reserved=None, expected_projection=None, cancelled=lambda: False):
    """Use the existing reserve-before-training Opacus kernel on fixed slots.

    Loss reporting for private data requires separate review. This interface emits
    logical steps only. An anchor is allowed only with reviewed provenance.
    """
    from opacus import PrivacyEngine
    from ppfcl_nids.differential_privacy.training import _train

    if (features.shape != (config.sample_count, 11) or labels.shape != (config.sample_count,)
            or features.dtype != torch.float32 or labels.dtype != torch.float32
            or not torch.isfinite(features).all() or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("Expected pinned fixed-slot training tensors")
    state = decode_verified_v0(checkpoint, checkpoint_sha, model_spec=EWC_SPEC)["state_dict"]
    model = FlowMLP(EWC_SPEC).cpu()
    model.load_state_dict(state, strict=True)
    if anchor is not None:
        import re
        if not isinstance(anchor, ReviewedAnchor) or not re.fullmatch(r"[a-f0-9]{64}", anchor.review_sha256):
            raise ValueError("Reviewed anchor evidence is required")
        validate_state(anchor.state, model.state_dict())
    if cancelled():
        raise InterruptedError("Cancelled before reservation")
    def hook(current, optimizer):
        if cancelled():
            raise InterruptedError("Cancelled at a logical step. Reservation remains consumed.")
        if anchor is not None and anchor.state.config.strength:
            penalty = ewc_penalty(current, anchor.state)
            gradients = torch.autograd.grad(penalty, tuple(current.parameters()))
            for parameter, gradient in zip(current.parameters(), gradients, strict=True):
                parameter.grad.add_(gradient)

    report = _train(model, TensorDataset(features, labels, torch.ones(len(labels))), config, ledger, run_id,
                    PrivacyEngine(accountant="rdp", secure_mode=False), step_callback=hook,
                    logical_progress=on_step, on_reserved=on_reserved, expected_projection=expected_projection)
    return model.eval(), report
