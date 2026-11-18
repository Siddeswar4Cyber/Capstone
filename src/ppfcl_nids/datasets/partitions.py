"""Resolve frozen initial partitions and continual chunks to source row ranges."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ppfcl_nids.datasets.batches import FlowBatch, iter_contract_batches


def load_ordinal_index(path: Path, spec: dict, filename_key: str, dtype: str) -> np.ndarray:
    """Memory-map an index only after its name, content hash, dtype, and shape pass."""
    name = spec[filename_key]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Invalid ordinal index filename")
    index_path = path.parent / name
    with index_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError("Ordinal index fingerprint mismatch")
    index = np.load(index_path, mmap_mode="r", allow_pickle=False)
    if index.dtype != np.dtype(dtype) or index.shape != (spec["rows"],):
        raise ValueError("Invalid ordinal index shape or dtype")
    return index


def ordinal_ranges(ordinals: Iterable[int], rows: int) -> Iterator[tuple[int, int]]:
    """Compress sorted, unique source ordinals into half-open CSV ranges."""
    start = stop = 0
    for value in ordinals:
        row = int(value)
        if not stop <= row < rows:
            raise ValueError("Ordinal index must be sorted, unique and in bounds")
        if row != stop:
            if start < stop:
                yield start, stop
            start = row
        stop = row + 1
    if start < stop:
        yield start, stop


def continual_ranges(path: Path, manifest: dict, start: int, stop: int) -> Iterator[tuple[int, int]]:
    """Resolve rank membership, then read selected records in source order."""
    rows = manifest["source_rows"]
    if not 0 <= start < stop <= rows:
        raise ValueError("Invalid continual partition range")
    rank = manifest["rank_index"]
    if rank is None:
        return iter(((start, stop),))
    index = load_ordinal_index(path, rank, "filename", "<u8")
    if len(index) != rows:
        raise ValueError("Rank index does not cover the source")
    return ordinal_ranges(np.sort(index[start:stop]), rows)


# Historical integrations imported these private names before the public API existed.
_verified_index = load_ordinal_index
_ordinal_ranges = ordinal_ranges


def iter_manifest_batches(
    source: Path, manifest_path: Path, *, contract_id: str, dataset_id: str, partition: str,
    manifest_sha256: str, source_sha256: str, source_bytes: int, batch_size: int = 4096,
) -> Iterator[FlowBatch]:
    """Read a pinned partition. Source SHA-256 comes from prior deployment verification."""
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest_sha256:
        raise ValueError("Manifest fingerprint mismatch")
    manifest = json.loads(content)
    if (
        manifest["dataset_id"] != dataset_id or manifest["source_sha256"] != source_sha256
        or source.stat().st_size != source_bytes
        or manifest.get("source_filename", source.name) != source.name
    ):
        raise ValueError("Source differs from the pinned partition")

    rows = manifest["source_rows"]
    if dataset_id in {"A", "B"} and manifest.get("schema") == "initial-chronological-group-v1":
        if partition not in {"train", "validation", "test"}:
            raise ValueError("Unknown initial split")
        index = load_ordinal_index(manifest_path, manifest["partitions"][partition], "file", "<u4")
        ranges = ordinal_ranges(index, rows)
    elif dataset_id in {"C", "D"} and manifest.get("manifest_version") == 1:
        if partition in {"validation", "test"}:
            selected = manifest[partition]
        else:
            matching = [item for item in manifest["chunks"] if item["id"] == partition]
            if len(matching) != 1:
                raise ValueError("Unknown continual chunk")
            selected = matching[0]
            training = manifest["training"]
            if not training["start"] <= selected["start"] < selected["stop_exclusive"] <= training["stop_exclusive"]:
                raise ValueError("Chunk is outside the continual training range")
        start, stop = selected["start"], selected["stop_exclusive"]
        if not 0 <= start < stop <= rows or selected["rows"] != stop - start:
            raise ValueError("Invalid continual partition range")
        ranges = continual_ranges(manifest_path, manifest, start, stop)
    else:
        raise ValueError("Unsupported partition manifest")
    yield from iter_contract_batches(source, ranges, contract_id=contract_id, batch_size=batch_size)
