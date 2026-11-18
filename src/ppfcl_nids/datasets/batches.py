"""Bounded-memory, sequential loading of explicit local CSV record ranges."""

import csv
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, islice
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ppfcl_nids.datasets.feature_contracts import FEATURE_CONTRACT_ID, get_contract, transform_contract_flows


class FlowBatch(NamedTuple):
    features: NDArray[np.float32]
    targets: NDArray[np.float32]


def _selected_columns(header: list[str], contract_id: str) -> list[tuple[str, int, int]]:
    raw_fields = (*get_contract(contract_id).raw_fields, ("Label", 1))
    if len(set(header)) != len(header) or any(name not in header for name, _ in raw_fields):
        raise ValueError("CSV header has duplicate columns or missing required fields")
    return [(name, header.index(name), (1 << bits) - 1) for name, bits in raw_fields]


def _parse_selected_row(row, header_size: int, columns, ordinal: int) -> list[int]:
    if row is None or len(row) != header_size:
        raise ValueError(f"Missing or malformed CSV record at source row {ordinal}")
    values = []
    for field, source_index, limit in columns:
        token = row[source_index].strip()
        if not token.isascii() or not token.isdecimal() or (value := int(token)) > limit:
            raise ValueError(f"Invalid {field} at source row {ordinal}")
        values.append(value)
    return values


def iter_flow_batches(
    path: Path, ranges: Sequence[tuple[int, int]], *, batch_size: int = 4096
) -> Iterator[FlowBatch]:
    """Compatibility entry point for the 41-input experiment."""
    return iter_contract_batches(path, ranges, contract_id=FEATURE_CONTRACT_ID, batch_size=batch_size)


def iter_contract_batches(
    path: Path, ranges: Iterable[tuple[int, int]], *, contract_id: str, batch_size: int = 4096
) -> Iterator[FlowBatch]:
    """Stream sorted, disjoint source ranges without retaining them in memory."""
    raw_fields = get_contract(contract_id).raw_fields
    remaining = iter(ranges)
    first = next(remaining, None)
    if batch_size <= 0 or first is None:
        raise ValueError("A positive batch size and at least one row range are required")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.reader(stream, strict=True)
        header = next(records, [])
        selected_columns = _selected_columns(header, contract_id)
        position = 0
        for start, stop in chain((first,), remaining):
            if start < position or stop <= start:
                raise ValueError("Row ranges must be nonempty, sorted and disjoint")
            selected = islice(records, start - position, stop - position)
            position = start
            while position < stop:
                count = min(batch_size, stop - position)
                values = np.empty((count, len(selected_columns)), dtype=np.uint32)
                for index in range(count):
                    values[index] = _parse_selected_row(next(selected, None), len(header),
                                                        selected_columns, position + index)
                features = transform_contract_flows({
                    field: values[:, column] for column, (field, _) in enumerate(raw_fields)
                }, contract_id=contract_id)
                yield FlowBatch(features, values[:, -1].astype(np.float32))
                position += count
