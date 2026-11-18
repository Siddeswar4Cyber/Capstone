"""Single-round FedAvg foundation with a verified, no-overwrite checkpoint."""

import hashlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import Tensor

from ppfcl_nids.datasets.feature_contracts import FEATURE_CONTRACT_ID, get_contract
from ppfcl_nids.models.neural_network import ExtendedFlowMLP, FlowMLP, FlowModelSpec


@dataclass(frozen=True)
class InitialUpdate:
    client_id: str
    base_id: str
    sample_count: int
    parameters: Mapping[str, Tensor]
    model_metadata: Mapping | None = None


def _state_id(parameters: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_parameters(parameters: Mapping[str, Tensor], reference: Mapping[str, Tensor]) -> None:
    if parameters.keys() != reference.keys():
        raise ValueError("Initial update tensor names differ")
    for name, expected in reference.items():
        value = parameters[name]
        if (
            not isinstance(value, Tensor) or value.device.type != "cpu" or value.layout != torch.strided
            or value.shape != expected.shape or value.dtype != expected.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid initial update tensor {name}")


class InitialFedAvg:
    def __init__(self, *, seed: int, hidden_sizes: tuple[int, ...] = (128, 64),
                 contract_id: str = FEATURE_CONTRACT_ID):
        self.seed = seed
        self.spec = FlowModelSpec(contract_id, hidden_sizes)
        self.hidden_sizes = self.spec.hidden_sizes
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._base = self._new_model().cpu().state_dict()
        self.base_id = _state_id(self._base)

    def _new_model(self) -> FlowMLP:
        if self.spec.contract_id == FEATURE_CONTRACT_ID:
            return ExtendedFlowMLP(self.hidden_sizes)
        return FlowMLP(self.spec)

    def client_model(self) -> FlowMLP:
        with torch.random.fork_rng(devices=[]):
            model = self._new_model().cpu()
        model.load_state_dict(self._base, strict=True)
        return model

    def aggregate(self, updates: Sequence[InitialUpdate]) -> dict[str, Tensor]:
        if len(updates) != 2 or {u.client_id for u in updates} != {"A", "B"}:
            raise ValueError("Initial round requires exactly clients A and B")
        ordered = sorted(updates, key=lambda update: update.client_id)
        for update in ordered:
            self.validate_update(update)
        total = sum(update.sample_count for update in ordered)
        return {
            name: sum(update.parameters[name].detach().double() * (update.sample_count / total)
                      for update in ordered).to(reference.dtype)
            for name, reference in self._base.items()
        }

    def validate_update(self, update: InitialUpdate) -> None:
        self.validate_model_metadata(update.model_metadata)
        if update.base_id != self.base_id or type(update.sample_count) is not int or update.sample_count < 1:
            raise ValueError("Invalid initial update base or sample count")
        validate_parameters(update.parameters, self._base)

    def validate_model_metadata(self, metadata: Mapping | None) -> None:
        if metadata is None and self.spec.contract_id == FEATURE_CONTRACT_ID:
            return  # Historical frames remain valid only in the explicit 41-input lineage.
        if metadata != self.spec.metadata():
            raise ValueError("Update model contract metadata mismatch")

    def publish_v0(self, path: Path, updates: Sequence[InitialUpdate], *,
                   artifact_metadata: Mapping[str, str | bool] | None = None) -> str:
        payload = {
            "version": 0, "parent_initialization_sha256": self.base_id, "seed": self.seed,
            "feature_contract": self.spec.contract_id,
            "feature_order": list(get_contract(self.spec.contract_id).feature_names),
            "hidden_sizes": list(self.hidden_sizes), "fitted_parameters": None,
            "client_sample_counts": {u.client_id: u.sample_count for u in updates},
            "state_dict": self.aggregate(updates),
        }
        if self.spec.contract_id != FEATURE_CONTRACT_ID:
            payload["model_metadata"] = self.spec.metadata()
        if artifact_metadata is not None:
            payload["artifact_metadata"] = dict(artifact_metadata)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            staging = Path(temp.name)
            try:
                torch.save(payload, temp)
                temp.flush()
                os.fsync(temp.fileno())
                temp.close()
                with staging.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if os.name == "nt":
                    # Windows rename fails if the destination already exists.
                    staging.rename(path)
                else:
                    # POSIX rename would replace a destination, so publish by link.
                    os.link(staging, path)
            finally:
                staging.unlink(missing_ok=True)
        return digest


def load_verified_v0(path: Path, expected_sha256: str, *, model_spec: FlowModelSpec | None = None) -> dict:
    return decode_verified_v0(path.read_bytes(), expected_sha256, model_spec=model_spec)


def decode_verified_v0(content: bytes, expected_sha256: str, *, model_spec: FlowModelSpec | None = None) -> dict:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint fingerprint mismatch")
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location="cpu")
    contract = get_contract(FEATURE_CONTRACT_ID if model_spec is None else model_spec.contract_id)
    if (
        payload.get("version") != 0 or payload.get("feature_contract") != contract.id
        or payload.get("feature_order") != list(contract.feature_names)
        or "fitted_parameters" not in payload or payload["fitted_parameters"] is not None
    ):
        raise ValueError("Checkpoint model contract mismatch")
    spec = model_spec or FlowModelSpec(FEATURE_CONTRACT_ID, tuple(payload["hidden_sizes"]))
    metadata = payload.get("model_metadata")
    if (
        payload["hidden_sizes"] != list(spec.hidden_sizes)
        or ((metadata is not None or contract.id != FEATURE_CONTRACT_ID) and metadata != spec.metadata())
    ):
        raise ValueError("Checkpoint model contract metadata mismatch")
    with torch.random.fork_rng(devices=[]):
        validate_parameters(payload["state_dict"], FlowMLP(spec).state_dict())
    return payload
