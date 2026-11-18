"""Offline TCP refusal correlation for bounded classic Ethernet PCAPs."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import struct

MIN_PORTS = 4
WINDOW_NS = 10_000_000_000
MAX_BYTES = 128 * 1024
MAGIC = {b"\xd4\xc3\xb2\xa1": ("<", 1000), b"\xa1\xb2\xc3\xd4": (">", 1000),
         b"\x4d\x3c\xb2\xa1": ("<", 1), b"\xa1\xb2\x3c\x4d": (">", 1)}


@dataclass(frozen=True)
class Alert:
    source: bytes
    target: bytes
    distinct_ports: int
    first_syn_ns: int
    last_syn_ns: int
    kind: str = "POSSIBLE_PORT_SCAN"


def packets(body: bytes):
    if not 24 <= len(body) <= MAX_BYTES or body[:4] not in MAGIC:
        raise ValueError("expected a bounded classic PCAP")
    endian, scale = MAGIC[body[:4]]
    major, minor, _, _, snaplen, link = struct.unpack_from(endian + "HHIIII", body, 4)
    if (major, minor, link) != (2, 4, 1) or not 14 <= snaplen <= 262144:
        raise ValueError("unsupported PCAP header")
    offset = 24
    last = -1
    while offset < len(body):
        if len(body) - offset < 16:
            raise ValueError("truncated packet header")
        sec, fraction, size, original = struct.unpack_from(endian + "IIII", body, offset)
        offset += 16
        stamp = sec * 1_000_000_000 + fraction * scale
        if (not 14 <= size == original <= snaplen or offset + size > len(body)
                or fraction * scale >= 1_000_000_000 or stamp < last):
            raise ValueError("invalid or incomplete packet record")
        yield stamp, body[offset:offset + size]
        offset += size
        last = stamp


def tcp_metadata(frame: bytes):
    if frame[12:14] != b"\x08\x00":
        return None
    ip = frame[14:]
    if len(ip) < 20 or ip[0] >> 4 != 4:
        raise ValueError("malformed IPv4 packet")
    header = (ip[0] & 15) * 4
    total, _, fragment = struct.unpack_from("!HHH", ip, 2)
    if not 20 <= header <= total <= len(ip):
        raise ValueError("malformed IPv4 length")
    if ip[9] != 6 or fragment & 0x3fff:
        return None
    tcp = ip[header:total]
    if len(tcp) < 20 or not 20 <= (tcp[12] >> 4) * 4 <= len(tcp):
        raise ValueError("malformed TCP header")
    source_port, target_port, seq, ack = struct.unpack_from("!HHII", tcp)
    return ip[12:16], ip[16:20], source_port, target_port, seq, ack, tcp[13]


def detect(body: bytes) -> list[Alert]:
    """Return the first qualifying window per observed source–target pair."""
    pending = {}
    refusals = defaultdict(list)
    for stamp, frame in packets(body):
        metadata = tcp_metadata(frame)
        if metadata is None:
            continue
        source, target, sport, dport, seq, ack, flags = metadata
        if flags & 0x3f == 0x02:
            pending.setdefault((source, target, sport, dport, seq), stamp)
        elif flags & 0x3f == 0x14:
            started = pending.pop((target, source, dport, sport, (ack - 1) % (2**32)), None)
            if started is not None and stamp - started <= WINDOW_NS:
                refusals[target, source].append((started, sport))

    alerts = []
    for (source, target), attempts in refusals.items():
        window = deque()
        distinct = Counter()
        for stamp, port in sorted(attempts):
            while window and stamp - window[0][0] > WINDOW_NS:
                _, expired = window.popleft()
                distinct[expired] -= 1
                if not distinct[expired]:
                    del distinct[expired]
            window.append((stamp, port))
            distinct[port] += 1
            if len(distinct) >= MIN_PORTS:
                alerts.append(Alert(source, target, len(distinct), window[0][0], stamp))
                break
    return alerts
