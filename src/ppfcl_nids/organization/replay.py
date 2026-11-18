"""Feature-only NFStream replay of an explicitly pinned local Ethernet PCAP."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
from tempfile import TemporaryDirectory

import numpy as np

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec
from ppfcl_nids.organization.feature_adapter import nfstream_request

SPEC = FlowModelSpec(BIFLOW_CONTRACT_ID)
MAX_BYTES = 8 * 1024**2
MAX_PACKETS = 50_000
MAX_FLOWS = 4_096
# One meter, IP-layer accounting, short fixture-specific expiry. No tunnel decoding or DPI.
SETTINGS = {
    "accounting_mode": 1, "idle_timeout": 1, "active_timeout": 2, "n_dissections": 0,
    "n_meters": 1, "decode_tunnels": False, "promiscuous_mode": False, "snapshot_length": 65535,
}
MAGIC = {b"\xd4\xc3\xb2\xa1": ("<", 10**6), b"\xa1\xb2\xc3\xd4": (">", 10**6),
         b"\x4d\x3c\xb2\xa1": ("<", 10**9), b"\xa1\xb2\x3c\x4d": (">", 10**9)}


@dataclass(frozen=True)
class ReplayFeatures:
    pcap_sha256: str
    packet_count: int
    features: np.ndarray


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("An exact fixture SHA-256 is required")
    if path.suffix.lower() != ".pcap" or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Expected a regular local .pcap file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 24 <= info.st_size <= MAX_BYTES:
            raise ValueError("Invalid PCAP type or size")
        body = stream.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES or hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("PCAP SHA-256 or size mismatch")
    return body


def _packet_count(body: bytes) -> int:
    encoding = MAGIC.get(body[:4])
    if not encoding or len(body) < 24:
        raise ValueError("Expected a classic PCAP header")
    endian, resolution = encoding
    major, minor, _, _, snaplen, link = struct.unpack_from(f"{endian}HHIIII", body, 4)
    if (major, minor, link) != (2, 4, 1) or not 14 <= snaplen <= 65535:
        raise ValueError("Unsupported PCAP version, snapshot length or link type")
    offset = 24
    count = 0
    while offset < len(body):
        if len(body) - offset < 16:
            raise ValueError("Truncated PCAP packet header")
        _, fraction, captured, original = struct.unpack_from(f"{endian}IIII", body, offset)
        if not (fraction < resolution and 14 <= captured == original <= snaplen):
            raise ValueError("Malformed or truncated PCAP packet record")
        offset += 16 + captured
        count += 1
        if offset > len(body) or count > MAX_PACKETS:
            raise ValueError("PCAP packet data truncated or packet limit exceeded")
    return count


def validated_flow(flow) -> np.ndarray:
    """Use the pinned adapter, then reject impossible packet/byte counts."""
    request = nfstream_request(flow, model_spec=SPEC, accounting_mode=1)
    row = request.features[0]
    if row[1] <= 0 or row[0] <= 0 or (row[2] == 0) != (row[3] == 0):
        raise ValueError("Empty flow or inconsistent directional counters")
    return row


def extract_pcap(path: Path, *, expected_sha256: str) -> ReplayFeatures:
    """Return validated features only. Never accept a network interface as the source."""
    body = _read_pinned(path, expected_sha256)
    packets = _packet_count(body)
    if not packets:
        return ReplayFeatures(expected_sha256, 0, np.empty((0, 11), dtype=np.float32))
    from nfstream import NFStreamer

    rows = []
    failure = None
    with TemporaryDirectory(prefix="ppfcl-replay-") as work:
        # NFStream opens only our private, verified absolute path. The original is never mutated.
        source = Path(work).resolve() / "verified.pcap"
        source.write_bytes(body)
        for flow in NFStreamer(source=str(source), max_nflows=MAX_FLOWS + 1, **SETTINGS):
            # Let NFStream reach its own bounded end so its meter process is joined.
            if failure is not None:
                continue
            try:
                if len(rows) == MAX_FLOWS:
                    raise ValueError("PCAP flow limit exceeded")
                rows.append(validated_flow(flow))
            except ValueError as exc:
                failure = exc
    if failure is not None:
        raise failure
    return ReplayFeatures(expected_sha256, packets, np.stack(rows) if rows else np.empty((0, 11), dtype=np.float32))
