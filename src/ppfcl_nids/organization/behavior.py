"""Bounded metadata-only behavioral correlation on Organization."""
from collections import OrderedDict, deque
import struct

from ppfcl_nids.organization.port_scan_detection import MIN_PORTS, WINDOW_NS, tcp_metadata

WINDOW = WINDOW_NS // 1_000_000
RULES = {
    "POSSIBLE_PORT_SCAN": (MIN_PORTS, WINDOW, "distinct_refused_ports", "medium"),
    "REPEATED_SYN_ATTEMPTS": (8, WINDOW, "unique_syn_attempts", "low"),
    "POSSIBLE_SERVICE_DISCOVERY": (6, WINDOW, "distinct_service_ports", "low"),
    "CONNECTION_BURST": (30, 5000, "unique_syn_attempts", "medium"),
}
CAP = 4096


class Behavior:
    def __init__(self, burst=30):
        self.attempts = OrderedDict()
        self.windows = OrderedDict()
        self.last = 0
        self.burst = burst
        self.cooldown = OrderedDict()

    def observe(self, stamp, metadata):
        """Consume timestamped packet metadata. Emit aggregate evidence only."""
        if metadata is None:
            return []
        src, dst, sport, port, seq, ack, flags = metadata
        self.last = max(self.last, stamp)
        if stamp < self.last - WINDOW:
            return []
        while self.attempts and next(iter(self.attempts.values()))[0] < self.last - WINDOW:
            self.attempts.popitem(last=False)
        while self.cooldown and next(iter(self.cooldown.values())) < self.last - WINDOW:
            self.cooldown.popitem(last=False)
        metrics = []
        if flags & 0x3f == 0x02:
            key = (src, dst, sport, port, seq)
            if key in self.attempts:
                return []
            self.attempts[key] = (stamp, False)
            if len(self.attempts) > CAP:
                self.attempts.popitem(last=False)
            metrics = ["REPEATED_SYN_ATTEMPTS", "POSSIBLE_SERVICE_DISCOVERY", "CONNECTION_BURST"]
        elif flags & 0x3f == 0x14:
            key = (dst, src, port, sport, (ack - 1) % 2**32)
            attempt = self.attempts.get(key)
            if attempt is None or attempt[1] or not 0 <= stamp - attempt[0] <= WINDOW:
                return []
            self.attempts[key] = (attempt[0], True)
            src, dst, port, stamp = dst, src, sport, attempt[0]
            metrics = ["POSSIBLE_PORT_SCAN"]
        alerts = []
        for rule in metrics:
            threshold, window, metric, severity = RULES[rule]
            if rule == "CONNECTION_BURST":
                threshold = self.burst
            key = (src, dst, rule)
            values = self.windows.setdefault(key, deque(maxlen=256))
            self.windows.move_to_end(key)
            if len(self.windows) > 512:
                self.windows.popitem(last=False)
            values.append((stamp, port))
            latest = max(item[0] for item in values)
            eligible = [(t, p) for t, p in values if latest - t <= window]
            if len(eligible) != len(values):
                values.clear()
                values.extend(eligible)
            count = len({p for _, p in values}) if metric.startswith("distinct") else len(values)
            if count >= threshold and key not in self.cooldown:
                self.cooldown[key] = self.last
                if len(self.cooldown) > 512:
                    self.cooldown.popitem(last=False)
                alerts.append({"rule": rule, "severity": severity, "metric": metric, "count": count,
                               "threshold": threshold, "window_ms": window,
                               "span_ms": latest - min(t for t, _ in values)})
        return alerts


def packet_metadata(packet):
    if packet.ip_version != 4 or packet.protocol != 6:
        return None
    try:
        return tcp_metadata(bytes(12) + b"\x08\x00" + packet.ip_packet)
    except (ValueError, struct.error):
        return None


def plugin_type():
    from nfstream import NFPlugin

    class MetadataPlugin(NFPlugin):
        def __init__(self, *, writer, burst):
            super().__init__()
            self.writer = writer
            self.behavior = Behavior(burst)

        def on_init(self, packet, flow):
            self.on_update(packet, flow)

        def on_update(self, packet, flow):
            for alert in self.behavior.observe(packet.time, packet_metadata(packet)):
                self.writer.send(("alert", packet.time, alert))

    return MetadataPlugin
