"""Pinned C/D readiness metadata and client-local bounded partition loading."""
import hashlib
import json
from pathlib import Path

from ppfcl_nids.datasets import DATASET_FILENAMES
from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID

ASSIGNMENTS = {
    "A": ("C", DATASET_FILENAMES["C"], 3817527532,
          "8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea",
          "f999c6dd5d89341832855f2bba68796b49b04d14bd8f1a0dae1ae2ac21d440ce"),
    "B": ("D", DATASET_FILENAMES["D"], 5302886266,
          "53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78",
          "ebef2d4282c0b7d08a5a3c467c08e6f77e2588e9fa88ecdd516f0dd9cd922ca1"),
}


def owned_chunk(client: str, chunk: str) -> bool:
    return isinstance(chunk, str) and client in ASSIGNMENTS and chunk in {
        f"{ASSIGNMENTS[client][0]}{n}" for n in range(1, 21)}


def manifest(path: Path, client: str) -> dict:
    dataset, filename, _, source_hash, digest = ASSIGNMENTS[client]
    with path.open("rb") as stream:
        body = stream.read(65537)
    if len(body) > 65536 or hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("Frozen C/D manifest digest mismatch")
    data = json.loads(body)
    if (data["manifest_version"], data["dataset_id"], data["source_filename"], data["source_sha256"]) != (
            1, dataset, filename, source_hash):
        raise ValueError("Frozen C/D assignment mismatch")

    def check_range(item, cursor):
        stop = item["stop_exclusive"]
        if item["start"] != cursor or not cursor < stop <= data["source_rows"] or item["rows"] != stop - cursor:
            raise ValueError("C/D partition gap, overlap or invalid length")
        return stop

    training = check_range(data["training"], 0)
    validation = check_range(data["validation"], training)
    if check_range(data["test"], validation) != data["source_rows"]:
        raise ValueError("C/D partitions do not cover source")
    cursor = 0
    if len(data["chunks"]) != 20:
        raise ValueError("Unexpected frozen chunk count")
    for index, chunk in enumerate(data["chunks"], 1):
        if chunk["id"] != f"{dataset}{index}":
            raise ValueError("Unexpected C/D chunk order")
        cursor = check_range(chunk, cursor)
    if cursor != training:
        raise ValueError("Chunks do not cover training partition")
    return data


def readiness(project: Path) -> dict:
    results = {}
    for client, (dataset, name, _, _, pin) in ASSIGNMENTS.items():
        path = project / "state/audit" / f"chunks-{dataset}.json"
        try:
            chunks = [item["id"] for item in manifest(path, client)["chunks"]]
        except (OSError, KeyError, ValueError, TypeError):
            chunks = None
        results[client] = {"dataset": dataset, "filename": name, "chunks": chunks,
                           "manifest_sha256": pin if chunks else None}
    return results


def load_slots(source: Path, manifest_path: Path, *, client: str, chunk: str, count: int):
    """Resolve a bounded prefix of frozen chunk membership in the owning client."""
    import numpy as np

    from ppfcl_nids.datasets.batches import iter_contract_batches
    from ppfcl_nids.datasets.partitions import continual_ranges

    _, filename, source_bytes, source_hash, _ = ASSIGNMENTS[client]
    data = manifest(manifest_path, client)
    selected = next((item for item in data["chunks"] if item["id"] == chunk), None)
    if chunk == "validation":
        selected = data["validation"]
    if selected is None or type(count) is not int or not 1 <= count <= min(2048, selected["rows"]):
        raise ValueError("Invalid bounded training chunk selection")
    if source.name != filename or source.stat().st_size != source_bytes:
        raise ValueError("Assigned source metadata mismatch")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source_hash:
            raise ValueError("Assigned source digest mismatch")
    start = selected["start"]
    ranges = continual_ranges(manifest_path, data, start, start + count)
    batches = list(iter_contract_batches(source, ranges, contract_id=BIFLOW_CONTRACT_ID, batch_size=count))
    features = np.concatenate([batch.features for batch in batches])
    labels = np.concatenate([batch.targets for batch in batches])
    if features.shape != (count, 11) or labels.shape != (count,) or not np.isfinite(features).all():
        raise ValueError("Incomplete or invalid fixed slots")
    return features, labels
