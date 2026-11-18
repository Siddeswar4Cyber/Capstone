"""Bounded initial-training preparation. The command-line entry point is synthetic-only."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppfcl_nids.datasets.batches import iter_contract_batches
from ppfcl_nids.datasets.partitions import load_ordinal_index, ordinal_ranges
from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FEATURE_CONTRACT_ID, FlowModelSpec, get_contract
from ppfcl_nids.federated_learning.initial_aggregation import _state_id
from ppfcl_nids.models.neural_network import FlowMLP
if TYPE_CHECKING:
    from opacus import PrivacyEngine
    from ppfcl_nids.differential_privacy.accounting import PrivacyLedger
from ppfcl_nids.federated_learning.synthetic_round import exchange_update, frame, read_frame, research_metadata, tensor_state, wire_state

DISK_RESERVE = 3 * 1024**3


@dataclass(frozen=True)
class SmokeConfig:
    MAX_SLOTS: ClassVar[int] = 2048
    sample_count: int = 2048
    batch_size: int = 64
    microbatch_size: int = 16
    epochs: int = 1
    learning_rate: float = 0.01
    clip: float = 1.0
    noise: float = 1.5
    delta: float = 5e-7
    epsilon_limit: float = 2.0

    def __post_init__(self):
        if (
            not 1 <= self.microbatch_size <= self.batch_size <= self.sample_count <= self.MAX_SLOTS
            or self.sample_count % self.batch_size or self.epochs != 1
            or not 0 < self.delta < 1
            or any(not math.isfinite(v) or v <= 0 for v in (
                self.learning_rate, self.clip, self.noise, self.epsilon_limit
            ))
        ):
            raise ValueError("Invalid bounded smoke-test configuration")

    @property
    def steps(self) -> int:
        return self.sample_count // self.batch_size

    @property
    def sample_rate(self) -> float:
        return 1 / self.steps

    def mechanism(self) -> dict:
        return {"sample_rate": self.sample_rate, "noise": self.noise, "clip": self.clip, "steps": self.steps}

    def projected_epsilon(self) -> float:
        from opacus.accountants import RDPAccountant

        accountant = RDPAccountant()
        for _ in range(self.steps):
            accountant.step(noise_multiplier=self.noise, sample_rate=self.sample_rate)
        return float(accountant.get_epsilon(delta=self.delta))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_training_subset(
    source: Path, manifest_path: Path, config: SmokeConfig, *,
    dataset_id: str, manifest_sha256: str, source_sha256: str, source_bytes: int,
    contract_id: str = FEATURE_CONTRACT_ID,
) -> TensorDataset:
    """Materialize only the first fixed N verified training ordinals, on CPU."""
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("Frozen split manifest digest differs")
    manifest = json.loads(manifest_bytes)
    if (
        dataset_id not in {"A", "B"} or manifest["dataset_id"] != dataset_id
        or manifest["schema"] != "initial-chronological-group-v1"
        or manifest["source_sha256"] != source_sha256 or source.stat().st_size != source_bytes
    ):
        raise ValueError("Source or dataset differs from the approved split")
    partition = manifest["partitions"]["train"]
    index = load_ordinal_index(manifest_path, partition, "file", "<u4")
    if len(index) < config.sample_count:
        raise ValueError("Invalid or insufficient training ordinals")
    selected = index[:config.sample_count]
    if selected[-1] >= manifest["source_rows"] or np.any(selected[1:] <= selected[:-1]):
        raise ValueError("Invalid selected training ordinals")
    ranges = ordinal_ranges(selected, manifest["source_rows"])
    features = np.empty((config.sample_count, get_contract(contract_id).width), dtype=np.float32)
    labels = np.empty(config.sample_count, dtype=np.float32)
    offset = 0
    for batch in iter_contract_batches(source, ranges, contract_id=contract_id, batch_size=config.batch_size):
        stop = offset + len(batch.targets)
        features[offset:stop], labels[offset:stop] = batch
        offset = stop
    if offset != config.sample_count:
        raise ValueError("Selected training subset was incomplete")
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels), torch.ones(config.sample_count))


def _train(model, dataset, config: SmokeConfig, ledger: PrivacyLedger, run_id: str, engine: PrivacyEngine,
           *, positive_class_weight: float | None = None, step_callback=None, logical_progress=None,
           on_reserved=None, expected_projection=None) -> dict:
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator

    width = get_contract(model.spec.contract_id).width
    if dataset.tensors[0].shape != (config.sample_count, width) or model.hidden_sizes != (128, 64):
        raise ValueError("Expected fixed-size subset and matching smoke-test architecture")
    if any(p.device.type != "cpu" for p in model.parameters()) or any(t.device.type != "cpu" for t in dataset.tensors):
        raise ValueError("Smoke training is CPU-only")
    pos_weight = None
    if positive_class_weight is not None:
        if (model.spec.contract_id != BIFLOW_CONTRACT_ID
                or not 0 < positive_class_weight <= torch.finfo(torch.float32).max):
            raise ValueError("Expected a finite positive class weight for the 11-feature experiment")
        pos_weight = torch.tensor(positive_class_weight, dtype=torch.float32)
        if pos_weight.item() == 0:
            raise ValueError("Positive class weight underflows float32")
    torch.set_num_threads(2)
    model.train()
    ModuleValidator.validate(model, strict=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
    private, optimizer, poisson = engine.make_private(
        module=model, optimizer=torch.optim.SGD(model.parameters(), lr=config.learning_rate),
        data_loader=loader, noise_multiplier=config.noise, max_grad_norm=config.clip,
        poisson_sampling=True, clipping="flat", loss_reduction="mean",
    )
    try:
        if (poisson.sample_rate != config.sample_rate or len(poisson) != config.steps
                or optimizer.expected_batch_size != config.batch_size):
            raise ValueError("Opacus sampling schedule differs from the reservation")
        approval = {"expected_projection": expected_projection} if expected_projection is not None else {}
        ledger.reserve(run_id, **config.mechanism(), **approval)
        if on_reserved:
            on_reserved()
        if step_callback is not None:
            account = optimizer.step_hook
            def after_privatization(private_optimizer):
                account(private_optimizer)
                step_callback(model, private_optimizer)
            optimizer.attach_step_hook(after_privatization)
        largest = 0
        with BatchMemoryManager(data_loader=poisson, max_physical_batch_size=config.microbatch_size,
                                optimizer=optimizer) as batches:
            for inputs, labels, active in batches:
                largest = max(largest, len(inputs))
                optimizer.zero_grad()
                logits = private(inputs)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none", pos_weight=pos_weight,
                ) * active
                (losses.mean() if len(labels) else losses.sum()).backward()
                previous = sum(steps for _, _, steps in engine.accountant.history) if logical_progress else 0
                optimizer.step()
                if logical_progress:
                    completed = sum(steps for _, _, steps in engine.accountant.history)
                    if completed > previous:
                        logical_progress(completed, config.steps)
        logical_steps = sum(steps for _, _, steps in engine.accountant.history)
        if logical_steps != config.steps or any(
            noise != config.noise or q != config.sample_rate for noise, q, _ in engine.accountant.history
        ):
            raise RuntimeError("Completed Opacus accounting differs from the reservation")
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError("Nonfinite parameters after smoke training")
        return {"logical_steps": logical_steps, "max_physical_batch": largest}
    finally:
        private.to_standard_module()


def check_disk(path: Path) -> None:
    if shutil.disk_usage(path).free < DISK_RESERVE + 16 * 1024**2:
        raise RuntimeError("Insufficient disk above the smoke-test reserve")


def _train_with_ledger(model, dataset, config, ledger_path, run_id, *, secure_mode):
    from opacus import PrivacyEngine
    from ppfcl_nids.differential_privacy.accounting import PrivacyLedger

    check_disk(ledger_path.parent)
    engine = PrivacyEngine(accountant="rdp", secure_mode=secure_mode)
    ledger = PrivacyLedger(ledger_path, delta=config.delta, epsilon_limit=config.epsilon_limit, create=False)
    return _train(model, dataset, config, ledger, run_id, engine)


def train_secure_smoke(model, dataset, config: SmokeConfig, ledger_path: Path, run_id: str) -> dict:
    """Require Opacus secure randomness for the secure execution path."""
    return _train_with_ledger(model, dataset, config, ledger_path, run_id, secure_mode=True)


def train_research_smoke(model, dataset, config: SmokeConfig, ledger_path: Path, run_id: str) -> dict:
    """Use nonsecure randomness with the same clipped mechanism and durable accounting."""
    metrics = _train_with_ledger(model, dataset, config, ledger_path, run_id, secure_mode=False)
    return {**metrics, **research_metadata()}


def synthetic_fixture(directory: Path, role: str, config: SmokeConfig, *,
                      contract_id: str = FEATURE_CONTRACT_ID):
    source = directory / "fixture.csv"
    fields = [name for name, _ in get_contract(contract_id).raw_fields]
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*fields, "Label"])
        for ordinal in range(config.sample_count + 4):
            label = (ordinal + (role == "B")) % 2
            values = dict.fromkeys(fields, 0)
            values.update(IN_BYTES=100 + ordinal + label * 500, IN_PKTS=1 + ordinal % 16, PROTOCOL=6)
            writer.writerow([*values.values(), label])
    # Put two nontraining rows before and two after the selected universe.
    index = directory / "train.npy"
    np.save(index, np.arange(2, config.sample_count + 2, dtype="<u4"), allow_pickle=False)
    fingerprint = _digest(source)
    manifest = directory / "fixture.json"
    manifest.write_text(json.dumps({
        "dataset_id": role, "schema": "initial-chronological-group-v1", "source_sha256": fingerprint,
        "source_rows": config.sample_count + 4,
        "partitions": {"train": {"file": index.name, "sha256": _digest(index), "rows": config.sample_count}},
    }))
    return source, manifest, {"dataset_id": role, "manifest_sha256": _digest(manifest),
                              "source_sha256": fingerprint, "source_bytes": source.stat().st_size}


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-nonsecure-synthetic", action="store_true")
    parser.add_argument("--contract-id", choices=(FEATURE_CONTRACT_ID, BIFLOW_CONTRACT_ID),
                        default=FEATURE_CONTRACT_ID)
    args = parser.parse_args()
    if args.contract_id == BIFLOW_CONTRACT_ID and not args.research_nonsecure_synthetic:
        parser.error("Biflow-11 uses the explicitly labeled synthetic research runner")
    config = SmokeConfig()
    job = json.loads(read_frame(sys.stdin.buffer))
    role = job["role"]
    spec = FlowModelSpec(args.contract_id)
    expected_metadata = spec.metadata() if args.contract_id == BIFLOW_CONTRACT_ID else None
    if (role not in {"A", "B"} or job["feature_contract"] != args.contract_id
            or job.get("model_metadata") != expected_metadata):
        raise ValueError("Invalid synthetic smoke job")
    parameters = tensor_state(job["parameters"])
    if _state_id(parameters) != job["base_id"]:
        raise ValueError("Bootstrap parameter digest mismatch")
    model = FlowMLP(spec)
    model.load_state_dict(parameters, strict=True)
    with tempfile.TemporaryDirectory(prefix="ppfcl-synthetic-smoke-") as temp:
        directory = Path(temp)
        source, manifest, metadata = synthetic_fixture(directory, role, config, contract_id=args.contract_id)
        started = time.perf_counter()
        dataset = load_training_subset(source, manifest, config, contract_id=args.contract_id, **metadata)
        load_seconds = time.perf_counter() - started
        path = directory / "synthetic.sqlite"
        from opacus import PrivacyEngine
        from ppfcl_nids.differential_privacy.accounting import PrivacyLedger

        ledger = PrivacyLedger(path, delta=config.delta, epsilon_limit=config.epsilon_limit)
        started = time.perf_counter()
        if args.research_nonsecure_synthetic:
            metrics = train_research_smoke(model, dataset, config, path, job["round_id"])
        else:
            metrics = _train(model, dataset, config, ledger, job["round_id"], PrivacyEngine(accountant="rdp"))
        metrics.update(load_seconds=load_seconds, training_seconds=time.perf_counter() - started,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    update = {"round_id": job["round_id"], "role": role, "base_id": job["base_id"],
              "update_id": f"{job['round_id']}:{role}", "sample_count": config.sample_count,
              "parameters": wire_state(model.state_dict())}
    if args.research_nonsecure_synthetic:
        update["artifact_metadata"] = research_metadata()
    if expected_metadata is not None:
        update["feature_contract"] = args.contract_id
        update["model_metadata"] = expected_metadata
        metrics["model_metadata"] = expected_metadata
    exchange_update(frame(update), metrics)


if __name__ == "__main__":
    main()
