"""Normalize verified NFStream biflow fields to the common NetFlow contract."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec, transform_contract_flows

if TYPE_CHECKING:
    from ppfcl_nids.models.neural_network import FlowMLP

NFSTREAM_FIELDS = {
    "IN_BYTES": "src2dst_bytes", "IN_PKTS": "src2dst_packets",
    "OUT_BYTES": "dst2src_bytes", "OUT_PKTS": "dst2src_packets",
    "FLOW_DURATION_MILLISECONDS": "bidirectional_duration_ms", "PROTOCOL": "protocol",
}


@dataclass(frozen=True)
class InferenceRequest:
    model_metadata: Mapping
    features: NDArray[np.float32]


def nfstream_request(flow, *, model_spec: FlowModelSpec, accounting_mode: int) -> InferenceRequest:
    """Map one finalized NFStream flow using IP-layer accounting."""
    if model_spec.contract_id != BIFLOW_CONTRACT_ID or type(accounting_mode) is not int or accounting_mode != 1:
        raise ValueError("NFStream inference requires the explicit biflow-11 contract and IP accounting mode 1")
    columns = {}
    for canonical, native in NFSTREAM_FIELDS.items():
        try:
            value = flow[native] if isinstance(flow, Mapping) else getattr(flow, native)
        except (KeyError, AttributeError):
            raise ValueError(f"Missing required NFStream field {native}") from None
        columns[canonical] = np.asarray([value])
    features = transform_contract_flows(columns, contract_id=model_spec.contract_id)
    return InferenceRequest(model_spec.metadata(), features)


def infer_logits(model: FlowMLP, request: InferenceRequest):
    """Reject incompatible contract or architecture metadata before inference."""
    import torch

    expected = model.spec.metadata()
    values = request.features
    if request.model_metadata != expected:
        raise ValueError("Inference model contract metadata mismatch")
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != expected["architecture"][0]:
        raise ValueError("Inference input shape or dtype mismatch")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite inference input")
    with torch.inference_mode():
        return model(torch.from_numpy(values).to(next(model.parameters()).device))
