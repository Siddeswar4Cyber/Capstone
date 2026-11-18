"""Guest-local read-only comparison of the two explicitly pinned research models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS, load_slots, manifest
from ppfcl_nids.continual_learning.training import evaluate_local
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC, model_template
from ppfcl_nids.federated_learning.asynchronous_aggregation import _checked_state, _load
from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0
from ppfcl_nids.models.neural_network import FlowMLP
from ppfcl_nids.organization.inference import read_private, write_once

V0 = "415907f3a95c5ee63d930b080f36799f8b8e09166a41ccf0ff5414031e31c29c"
V2 = "ccedc3c7c21bacee029c6b657781ab914b0cbba4e6070a53ce4a108686b1fff6"
JOBS = {"A": "3bb010e10c0844369ac0b2428abb12c4", "B": "53a366eb1e154beb937d91c2cd11b393"}
EVENTS = ("EVALUATION_STARTED", "V0_VALIDATED", "V2_VALIDATED", "COMPARISON_COMPLETED")


def sha(body):
    return hashlib.sha256(body).hexdigest()


def model_states(v0, v2):
    if sha(v0) != V0 or sha(v2) != V2:
        raise ValueError("Evaluation checkpoint hash mismatch")
    first = decode_verified_v0(v0, V0, model_spec=EWC_SPEC)["state_dict"]
    second = _checked_state(_load(v2), {"schema": "continual-local-research-v1", "version": 2,
                                        "model_metadata": EWC_SPEC.metadata()}, model_template())
    return first, second


def emit(event, **fields):
    print(json.dumps({"event": event, "at": datetime.now(timezone.utc).isoformat(), **fields},
                     allow_nan=False), flush=True)


def protected(path, mode):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
        raise ValueError("Unexpected private input permissions")


def main(client, staged):
    os.umask(0o077)
    root = Path.home() / ".local/share/ppfcl-nids"
    dataset, name, _, _, _ = ASSIGNMENTS[client]
    job = root / "continual-private" / JOBS[client]
    v0 = job.parent / (V0 + ".pt")
    v2 = staged / (V2 + ".pt")
    source = root / "datasets" / name
    frozen = root / "manifests" / f"chunks-{dataset}.json"
    ledger = root / "privacy/client.sqlite"
    update = job / "update.pt"
    for path, mode in ((source, 0o400), (ledger, 0o600), (update, 0o400), (job / "evaluation.json", 0o400)):
        protected(path, mode)
    def integrity():
        return {key: sha(read_private(path, mode, 1024**2)) for key, path, mode in (
            ("v0", v0, 0o400), ("v2", v2, 0o400), ("update", update, 0o400),
            ("ledger", ledger, 0o600), ("prior_evaluation", job / "evaluation.json", 0o400))}
    baseline = integrity()
    states = model_states(read_private(v0, 0o400, 1024**2), read_private(v2, 0o400, 1024**2))
    prior = json.loads(read_private(job / "evaluation.json", 0o400, 1024**2))
    if (prior["partition"], prior["threshold"], prior["update_sha256"]) != (
            "first-2048-frozen-validation-ranks", 0.5, baseline["update"]):
        raise ValueError("Prior validation scope mismatch")
    data = manifest(frozen, client)
    first = data["chunks"][0]
    validation = data["validation"]
    if first["start"] + 2048 > validation["start"] or validation["start"] + 2048 > validation["stop_exclusive"]:
        raise ValueError("Training and validation slot bounds failed")
    if data["rank_index"] is not None:
        from ppfcl_nids.datasets.partitions import load_ordinal_index
        rank = load_ordinal_index(frozen, data["rank_index"], "filename", "<u8")
        train = rank[first["start"]:first["start"] + 2048]
        valid = rank[validation["start"]:validation["start"] + 2048]
        if len(np.intersect1d(train, valid)) or len(np.unique(valid)) != 2048:
            raise ValueError("Validation membership overlaps or contains duplicates")
    ram = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    disk = shutil.disk_usage(root).free
    if not os.cpu_count() or ram < 1024**3 or disk < 3 * 1024**3 or torch.version.cuda is not None:
        raise ValueError("Client resource or CPU policy failed")
    x, y = load_slots(source, frozen, client=client, chunk="validation", count=2048)
    receipt = job / (V2 + ".comparison.json")
    attempt = job / (V2 + ".comparison-attempt.json")
    if receipt.exists() or attempt.exists():
        raise ValueError("Comparison already attempted")
    emit("PREFLIGHT_READY", client=client, ram_bytes=ram, disk_bytes=disk, cpu_count=os.cpu_count())
    if sys.stdin.readline(32) != "evaluate\n":
        raise ValueError("Coordinator did not release comparison")
    torch.set_num_threads(2)
    write_once(attempt, {"client": client, "training_job": JOBS[client], "v0_sha256": V0, "v2_sha256": V2})
    emit(EVENTS[0])
    metrics = {}
    for name, state, event in zip(("v0", "v2"), states, EVENTS[1:3], strict=True):
        model = FlowMLP(EWC_SPEC).cpu()
        model.load_state_dict(state, strict=True)
        metrics[name] = evaluate_local(model, [(torch.from_numpy(x), torch.from_numpy(y))])
        emit(event)
    if integrity() != baseline:
        raise ValueError("Existing client artifacts changed")
    pin = write_once(receipt, {"client": client, "training_job": JOBS[client], "models": {"v0": V0, "v2": V2},
                               "contract": EWC_SPEC.metadata(), "partition": prior["partition"], "threshold": 0.5,
                               "manifest_sha256": ASSIGNMENTS[client][4],
                               "validation_sha256": sha(x.astype("<f4").tobytes() + y.astype("<f4").tobytes()),
                               "metrics": metrics})
    if sha(read_private(receipt, 0o400, 1024**2)) != pin:
        raise ValueError("Private comparison receipt integrity failed")
    emit(EVENTS[3], receipt_sha256=pin)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
