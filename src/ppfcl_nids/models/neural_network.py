"""Binary PyTorch MLP with explicit NetFlow contract identity."""

from torch import Tensor, nn

from ppfcl_nids.datasets.feature_contracts import FEATURE_CONTRACT_ID, FlowModelSpec, get_contract


class FlowMLP(nn.Module):
    """Return one raw logit per flow for an explicitly chosen model specification."""

    def __init__(self, spec: FlowModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_sizes = spec.hidden_sizes
        layers: list[nn.Module] = []
        width = get_contract(spec.contract_id).width
        for size in self.hidden_sizes:
            layers.extend((nn.Linear(width, size), nn.ReLU()))
            width = size
        self.network = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class ExtendedFlowMLP(FlowMLP):
    """Compatibility entry point for the preserved extended experiment."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (128, 64)) -> None:
        super().__init__(FlowModelSpec(FEATURE_CONTRACT_ID, hidden_sizes))
