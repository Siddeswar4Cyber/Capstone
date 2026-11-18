"""One-use paired CPU evaluation on the approved old-task A/B selections."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import torch

from ppfcl_nids.evaluation.continual import JOBS, V0, V2, emit, model_states, protected, sha
from ppfcl_nids.continual_learning.training import evaluate_local
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC
from ppfcl_nids.datasets.batches import iter_contract_batches
from ppfcl_nids.datasets.partitions import ordinal_ranges, load_ordinal_index
from ppfcl_nids.models.neural_network import FlowMLP
from ppfcl_nids.organization.inference import private_directory, read_private, write_once

SELECTIONS = {
    "A": "9e0361bff0043eda095bc90b4faaedba120c17b186dda6b6cc4d8600e7715023",
    "B": "fdc3a810cee121a2abc6b2f32f826bf24191d243befd26ba3247226bb6438494",
}
SAMPLE_COUNT = 8192
THRESHOLD = 0.5
SCOPE = "old-task-ab-v0-v2-v1"
PINS = {
    "A": ("e6df420c1a5458b2d83abb7b44aa5c36be2b235743ee64b7004ff64e11dd0099",
          "e3b89d8c523933c4b2ba1ac5c859254046aa800a413fc83f921f14611fec5fc8"),
    "B": ("24f064e6151c141d25de29935dfa8fcf4809226cf6ac08755028534f77f04af2",
          "03f916fe9067d9593bab5e51ddfab04f1fe2c5b2349a7b4e9ec13a9924a91f37"),
}


def selected_ordinals(path, manifest, role):
    indices = {name: load_ordinal_index(path, manifest["partitions"][name], "file", "<u4")
               for name in ("train", "validation", "test")}
    for index in indices.values():
        if not len(index) or index[-1] >= manifest["source_rows"] or np.any(index[1:] <= index[:-1]):
            raise ValueError("Invalid frozen index")
    validation = indices["validation"]
    ranks = ((2 * np.arange(SAMPLE_COUNT, dtype=np.int64) + 1) * len(validation)) // (2 * SAMPLE_COUNT)
    selected = validation[ranks]
    bounded = selected[selected < 3 * manifest["source_rows"] // 4]
    selected = bounded if len(bounded) else selected
    if (len(selected) != SAMPLE_COUNT or np.any(selected[1:] <= selected[:-1])
            or sha(selected.astype("<u4").tobytes()) != SELECTIONS[role]):
        raise ValueError("Selection pin mismatch")
    for name in ("train", "test"):
        index = indices[name]
        offsets = index.searchsorted(selected)
        in_bounds = offsets < len(index)
        if np.any(index[offsets[in_bounds]] == selected[in_bounds]):
            raise ValueError("Selection overlaps another partition")
    return selected


def compare(states, features, labels):
    if len(features) != SAMPLE_COUNT or len(labels) != SAMPLE_COUNT:
        raise ValueError("Unexpected evaluation sample count")
    batches = list(zip(torch.from_numpy(features).split(256), torch.from_numpy(labels).split(256), strict=True))
    metrics = {}
    for name, state in zip(("v0", "v2"), states, strict=True):
        model = FlowMLP(EWC_SPEC).cpu()
        model.load_state_dict(state, strict=True)
        metrics[name] = evaluate_local(model, batches)
    drops = {key: (metrics["v0"][key] - metrics["v2"][key]
                   if metrics["v0"][key] is not None and metrics["v2"][key] is not None else None)
             for key in ("accuracy", "precision", "recall", "f1")}
    return metrics, drops


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def run(role, staged, spec):
    os.umask(0o077)
    root = Path.home() / ".local/share/ppfcl-nids"
    home = root / "continual-private"
    private_directory(home)
    directory = home / SCOPE
    private_directory(directory, create=True)
    attempt, receipt = (directory / f"{role}.{suffix}.json" for suffix in ("attempt", "comparison"))
    if attempt.exists() or receipt.exists():
        raise ValueError("Existing attempt")

    manifest_path = root / "manifests" / spec["manifest_filename"]
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (sha(manifest_bytes) != spec["manifest_sha256"] or manifest["dataset_id"] != role
            or manifest["schema"] != "initial-chronological-group-v1"
            or manifest["source_sha256"] != spec["source_sha256"]
            or manifest["partitions"]["train"]["sha256"] != spec["train_index_sha256"]):
        raise ValueError("Manifest pin mismatch")
    source = root / "datasets" / spec["source_filename"]
    protected(source, 0o400)
    before = identity(source)
    if (before[1:4] != (spec["source_inode"], spec["source_bytes"], spec["source_mtime_ns"])
            or file_sha(source) != spec["source_sha256"] or identity(source) != before):
        raise ValueError("Source pin mismatch")
    selection = selected_ordinals(manifest_path, manifest, role)

    files = ((home / (V0 + ".pt"), 0o400, V0), (staged / (V2 + ".pt"), 0o400, V2),
             (home / JOBS[role] / "update.pt", 0o400, PINS[role][0]),
             (root / "privacy/client.sqlite", 0o600, PINS[role][1]))
    def verified_bytes():
        bodies = [read_private(path, mode, 1024**2) for path, mode, _ in files]
        if [sha(body) for body in bodies] != [pin for _, _, pin in files]:
            raise ValueError("Model, update, or ledger pin mismatch")
        return bodies

    originals = verified_bytes()
    with closing(sqlite3.connect(files[-1][0].as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Ledger integrity failed")
    ram = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    if (torch.version.cuda is not None or torch.version.hip is not None or not os.cpu_count()
            or ram < 1024**3 or shutil.disk_usage(root).free < 3 * 1024**3):
        raise ValueError("CPU resource check failed")
    torch.set_num_threads(2)
    states = model_states(*originals[:2])
    emit("PREFLIGHT_READY")
    if sys.stdin.readline(32) != "evaluate\n":
        raise ValueError("Missing coordinator release")
    scope = {"scope": SCOPE, "client": role, "dataset": role, "rows": SAMPLE_COUNT,
             "selection_sha256": SELECTIONS[role], "manifest_sha256": spec["manifest_sha256"],
             "models": {"v0": V0, "v2": V2}, "contract": EWC_SPEC.metadata(), "threshold": THRESHOLD}
    write_once(attempt, scope)
    emit("EVALUATION_STARTED")
    batches = list(iter_contract_batches(source, ordinal_ranges(selection, manifest["source_rows"]),
                                         contract_id="netflow-biflow-11-v1", batch_size=256))
    metrics, drops = compare(states, np.concatenate([batch.features for batch in batches]),
                             np.concatenate([batch.targets for batch in batches]))
    if (identity(source) != before or file_sha(manifest_path) != spec["manifest_sha256"]
            or verified_bytes() != originals):
        raise ValueError("Post-comparison integrity failed")
    pin = write_once(receipt, {**scope, "metrics": metrics, "v0_minus_v2": drops,
                               "development_exposed": True, "independence_verified": False,
                               "cross_dataset_overlap_verified": False, "continual_ewc_enabled": False})
    if sha(read_private(receipt, 0o400, 1024**2)) != pin:
        raise ValueError("Receipt integrity failed")
    write_once(directory / f"{role}.verified.json", {"receipt_sha256": pin})
    emit("COMPARISON_COMPLETED", receipt_sha256=pin)


if __name__ == "__main__":
    try:
        run(sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3]))
    except Exception:
        # Validation exceptions may contain private input details.
        emit("EVALUATION_FAILED")
        raise SystemExit(1) from None
