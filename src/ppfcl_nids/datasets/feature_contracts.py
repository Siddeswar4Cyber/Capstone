"""Fixed, label-independent NetFlow contracts and transforms."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FEATURE_CONTRACT_ID = "netflow-extended-v1"
BIFLOW_CONTRACT_ID = "netflow-biflow-11-v1"
BIFLOW_CONTRACT_FINGERPRINT = "9fd91f89d0a1ecebf6fea45afff4c3adfc550794dc9dce66a073e744b991bccd"
PROTOCOLS = (1, 6, 17, 58, 132)
FLAG_BITS = ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR")

# Field widths follow the nProbe reference template. Live parity is pending.
RAW_FIELDS = {
    "IN_BYTES": 32,
    "IN_PKTS": 32,
    "OUT_BYTES": 32,
    "OUT_PKTS": 32,
    "FLOW_DURATION_MILLISECONDS": 32,
    "PROTOCOL": 8,
    "TCP_FLAGS": 8,
    "CLIENT_TCP_FLAGS": 8,
    "SERVER_TCP_FLAGS": 8,
    "MIN_IP_PKT_LEN": 16,
    "MAX_IP_PKT_LEN": 16,
    "RETRANSMITTED_IN_PKTS": 32,
    "RETRANSMITTED_OUT_PKTS": 32,
    "TCP_WIN_MAX_IN": 16,
    "TCP_WIN_MAX_OUT": 16,
}


def _output_names(field: str) -> tuple[str, ...]:
    if field == "PROTOCOL":
        return tuple(f"{field}_{code}" for code in (*PROTOCOLS, "other"))
    if field.endswith("_FLAGS"):
        return tuple(f"{field}_{bit}" for bit in FLAG_BITS)
    return (field,)


FEATURE_NAMES = tuple(name for field in RAW_FIELDS for name in _output_names(field))
INPUT_WIDTH = len(FEATURE_NAMES)

_BIFLOW_FIELDS = (
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "PROTOCOL",
)
_BIFLOW_UNITS = ("ip_octets", "packets", "ip_octets", "packets", "milliseconds", "ip_protocol_number")
_BIFLOW_DIRECTIONS = ("forward", "forward", "reverse", "reverse", "whole_biflow", "biflow_key")


@dataclass(frozen=True)
class FlowContract:
    id: str
    raw_fields: tuple[tuple[str, int], ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name for field, _ in self.raw_fields for name in _output_names(field))

    @property
    def width(self) -> int:
        return len(self.feature_names)

    @property
    def preprocessing_version(self) -> str:
        return "fixed-flow-transform-v1"

    def descriptor(self) -> dict:
        return {
            "contract_id": self.id,
            "preprocessing_version": self.preprocessing_version,
            "raw_fields": [
                {
                    "name": field, "unsigned_bits": bits,
                    "unit": _BIFLOW_UNITS[index] if index < 6 else "extended_exporter_field",
                    "direction": _BIFLOW_DIRECTIONS[index] if index < 6 else "extended_exporter_field",
                    "encoding": ("protocol_one_hot" if field == "PROTOCOL" else
                                 "flag_bits" if field.endswith("_FLAGS") else "float64_log1p"),
                }
                for index, (field, bits) in enumerate(self.raw_fields)
            ],
            "feature_order": list(self.feature_names),
            "protocol_indicators": [*PROTOCOLS, "other"],
            "flag_bit_order": list(FLAG_BITS) if self.id == FEATURE_CONTRACT_ID else [],
            "direction_rule": "exported_biflow_source_to_destination; NFStream_first_observed_packet",
            "duration_rule": "last_observed_timestamp_ms_minus_first_observed_timestamp_ms",
            "output_dtype": "float32",
            "scaling": "none",
            "clipping": "none",
            "invalid_policy": "reject_missing_noninteger_negative_out_of_range",
        }

    @property
    def fingerprint(self) -> str:
        body = json.dumps(self.descriptor(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()


_CONTRACTS = {
    FEATURE_CONTRACT_ID: FlowContract(FEATURE_CONTRACT_ID, tuple(RAW_FIELDS.items())),
    BIFLOW_CONTRACT_ID: FlowContract(BIFLOW_CONTRACT_ID, tuple((field, RAW_FIELDS[field]) for field in _BIFLOW_FIELDS)),
}
if _CONTRACTS[BIFLOW_CONTRACT_ID].fingerprint != BIFLOW_CONTRACT_FINGERPRINT:
    raise RuntimeError("The frozen biflow-11 descriptor changed. Assign a new contract ID.")


def get_contract(contract_id: str) -> FlowContract:
    try:
        return _CONTRACTS[contract_id]
    except KeyError:
        raise ValueError(f"Unsupported NetFlow contract {contract_id!r}") from None


@dataclass(frozen=True)
class FlowModelSpec:
    contract_id: str
    hidden_sizes: tuple[int, ...] = (128, 64)

    def __post_init__(self):
        get_contract(self.contract_id)
        if not self.hidden_sizes or any(type(size) is not int or size <= 0 for size in self.hidden_sizes):
            raise ValueError("Hidden layer sizes must be positive integers")
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))

    def metadata(self) -> dict:
        contract = get_contract(self.contract_id)
        return {"contract_id": contract.id, "contract_fingerprint": contract.fingerprint,
                "preprocessing_version": contract.preprocessing_version,
                "architecture": [contract.width, *self.hidden_sizes, 1],
                "activation": "relu", "output": "binary_logit"}


def transform_contract_flows(
    columns: Mapping[str, NDArray[np.integer]], *, contract_id: str
) -> NDArray[np.float32]:
    """Transform exact integer columns for an explicitly selected contract."""
    contract = get_contract(contract_id)
    missing = {field for field, _ in contract.raw_fields} - columns.keys()
    if missing:
        raise ValueError(f"Missing NetFlow fields: {', '.join(sorted(missing))}")

    row_count = len(np.atleast_1d(columns["IN_BYTES"]))
    features = np.empty((row_count, contract.width), dtype=np.float32)
    offset = 0
    for field, width in contract.raw_fields:
        values = np.asarray(columns[field])
        if values.shape != (row_count,) or values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{field} must be a one-dimensional integer column")
        if np.any(values < 0) or np.any(values > (1 << width) - 1):
            raise ValueError(f"{field} is outside its unsigned {width}-bit range")

        count = len(_output_names(field))
        if field == "PROTOCOL":
            known = values[:, None] == PROTOCOLS
            features[:, offset:offset + count - 1] = known
            features[:, offset + count - 1] = ~known.any(axis=1)
        elif field.endswith("_FLAGS"):
            mask = values.astype(np.uint8, copy=False)[:, None]
            features[:, offset:offset + count] = (mask >> np.arange(count, dtype=np.uint8)) & 1
        else:
            features[:, offset] = np.log1p(values.astype(np.float64))
        offset += count
    return features


def transform_flows(columns: Mapping[str, NDArray[np.integer]]) -> NDArray[np.float32]:
    """Compatibility entry point for the existing 41-input experiment."""
    return transform_contract_flows(columns, contract_id=FEATURE_CONTRACT_ID)
