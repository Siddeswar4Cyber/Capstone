"""Bounded read-only infrastructure sampling, independent of worker commands."""

import ctypes
from ipaddress import ip_address
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread, Timer
from time import monotonic, sleep

from ppfcl_nids.configuration.settings import ConfigurationError, Role
from ppfcl_nids.ssh.connections import SSHConnectionError, ssh_session

WINDOWS_IP = "192.168.252.1"
HOSTS = {
    "Windows": WINDOWS_IP,
    Role.CLIENT_A: "192.168.252.10",
    Role.CLIENT_B: "192.168.252.11",
    Role.ORGANIZATION: "192.168.252.12",
}
METRICS = ("cpu_percent", "ram_total_bytes", "ram_used_bytes", "ram_available_bytes",
           "disk_available_bytes", "disk_used_percent")

# No project import, worker action, dataset access, or user-provided command arguments.
LINUX_PROBE = """
import ipaddress, json, os, platform, shutil, socket, subprocess, time
from pathlib import Path

def cpu():
    with open('/proc/stat') as stream:
        values = [int(i) for i in stream.readline().split()[1:9]]
    return sum(values), sum(values[3:5])

route = os.environ.get('SSH_CONNECTION', '').split()
result = {'route': [route[0], route[2]] if len(route) == 4 else []}
result['hostname'] = socket.gethostname()[:120]
result['cpu_cores'] = os.cpu_count()
try:
    result['os'] = platform.freedesktop_os_release()['PRETTY_NAME'][:120]
except (OSError, KeyError, ValueError):
    result['os'] = platform.system() + ' ' + platform.release()
result['addresses'] = []
try:
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address', 'show', 'up'],
                            timeout=1, stderr=subprocess.DEVNULL))
    for interface in interfaces:
        if 'LOOPBACK' in interface['flags']:
            continue
        for address in interface['addr_info']:
            if address.get('scope') == 'global':
                result['addresses'].append(str(ipaddress.ip_address(address['local'])))
except (OSError, ValueError, KeyError, subprocess.SubprocessError):
    pass
result['addresses'] = result['addresses'][:12]
try:
    with open('/proc/uptime') as stream:
        result['uptime_seconds'] = int(float(stream.read(128).split()[0]))
except (OSError, ValueError, IndexError):
    pass
runtime = Path.home() / RUNTIME
result['runtime_available'] = runtime.is_file() and os.access(runtime, os.X_OK)
try:
    before, idle_before = cpu()
    time.sleep(0.2)
    after, idle_after = cpu()
    if after > before:
        result['cpu_percent'] = 100 * (1 - (idle_after - idle_before) / (after - before))
except (OSError, ValueError):
    pass
try:
    with open('/proc/meminfo') as stream:
        mem = {line.split(':', 1)[0]: int(line.split()[1]) * 1024 for line in stream
               if line.startswith(('MemTotal:', 'MemAvailable:'))}
    result.update(ram_total_bytes=mem['MemTotal'], ram_available_bytes=mem['MemAvailable'],
                  ram_used_bytes=mem['MemTotal'] - mem['MemAvailable'])
except (OSError, ValueError, KeyError):
    pass
try:
    disk = shutil.disk_usage(Path.home())
    result.update(disk_total_bytes=disk.total, disk_used_bytes=disk.used,
                  disk_available_bytes=disk.free, disk_used_percent=100 * disk.used / disk.total)
except (OSError, ZeroDivisionError):
    pass
print(json.dumps(result, allow_nan=False))
"""


@dataclass(frozen=True)
class Thresholds:
    refresh_seconds: int = 60
    stale_seconds: int = 300
    low_disk_gib: float = 3
    high_memory_percent: float = 85
    prolonged_seconds: int = 120

    def __post_init__(self):
        if not (30 <= self.refresh_seconds <= self.stale_seconds
                and 0 < self.low_disk_gib < math.inf and 0 < self.high_memory_percent <= 100
                and self.prolonged_seconds > 0):
            raise ValueError("Invalid monitoring thresholds")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _vmnet_addresses() -> list[str]:
    """Find IPv4 addresses owned by the VMnet1 adapter, including a changed address."""
    try:
        executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/ipconfig.exe"
        output = subprocess.check_output([str(executable)], timeout=2, stderr=subprocess.DEVNULL,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return []
    section = next((block for block in re.split(rb"\r?\n\s*\r?\n(?=\S)", output)
                    if b"VMnet1:" in block.strip().split(b"\n", 1)[0]), b"")
    addresses = []
    for value in re.findall(rb"(?<![\d.:])(?:\d{1,3}\.){3}\d{1,3}(?![\d.:])", section):
        address = value.decode("ascii")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as check:
                check.bind((address, 0))
        except OSError:
            continue
        if address not in addresses:
            addresses.append(address)
    return addresses


def _windows_metrics(project: Path) -> dict:
    if os.name != "nt":
        raise OSError("Windows system counters unavailable")
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                   "virtual_total", "virtual_available", "extended_available")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    kernel.GetTickCount64.restype = ctypes.c_ulonglong
    release, version, _, _ = platform.win32_ver()
    addresses = _vmnet_addresses()
    result = {"hostname": socket.gethostname(), "os": f"Windows {release} ({version})",
              "addresses": addresses, "uptime_seconds": kernel.GetTickCount64() // 1000,
              "active_ip": WINDOWS_IP if WINDOWS_IP in addresses else next(iter(addresses), None)}
    if kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
        result.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available,
                      ram_used_bytes=memory.total - memory.available)

    def times():
        counts = [wintypes.FILETIME() for _ in range(3)]
        if not kernel.GetSystemTimes(*(ctypes.byref(count) for count in counts)):
            return None
        return tuple((count.dwHighDateTime << 32) | count.dwLowDateTime for count in counts)

    first = times()
    sleep(0.2)
    last = times()
    if first is not None and last is not None:
        idle, kernel_ticks, user_ticks = (b - a for a, b in zip(first, last))
        if kernel_ticks + user_ticks > 0:
            result["cpu_percent"] = 100 * (1 - idle / (kernel_ticks + user_ticks))
    try:
        disk = shutil.disk_usage(project)
        result.update(disk_total_bytes=disk.total, disk_used_bytes=disk.used,
                      disk_available_bytes=disk.free, disk_used_percent=100 * disk.used / disk.total)
    except (OSError, ZeroDivisionError):
        pass
    return result


def read_json_command(transport, command: str, timeout: float) -> dict:
    """Read a small JSON result from a caller-supplied fixed command, with one deadline."""
    deadline = monotonic() + timeout
    with transport.open_session(timeout=timeout) as channel:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError
        watchdog = Timer(remaining, channel.close)
        watchdog.daemon = True
        watchdog.start()
        try:
            channel.settimeout(remaining)
            channel.exec_command(command)
            output = bytearray()
            while monotonic() < deadline:
                if channel.recv_ready():
                    output.extend(channel.recv(2048))
                    if len(output) > 2048:
                        raise ValueError("Oversized infrastructure probe")
                elif channel.exit_status_ready() and (channel.eof_received or channel.closed):
                    if channel.recv_exit_status() == 0:
                        return json.loads(output)
                    break
                else:
                    sleep(0.02)
            if monotonic() >= deadline:
                raise TimeoutError
            raise ValueError("Infrastructure command failed")
        finally:
            watchdog.cancel()


def _fixed_command(transport, timeout: float = 5) -> dict:
    from ppfcl_nids.federated_learning.remote_chunks import RUNTIME

    probe = f"RUNTIME = {RUNTIME!r}\n" + LINUX_PROBE
    return read_json_command(transport, shlex.join(["python3", "-I", "-B", "-c", probe]), timeout)


def _linux_metrics(role: Role, project: Path) -> dict:
    expected = HOSTS[role]
    with ssh_session(role, env_file=project / ".env", expected_host=expected, timeout=4) as peer:
        transport = peer.get_transport()
        if transport is None or transport.getpeername()[0] != expected:
            raise ValueError("Unverified SSH peer address")
        result = _fixed_command(transport)
        route = result.get("route") if isinstance(result, dict) else None
        if (not isinstance(route, list) or len(route) != 2 or route[1] != expected
                or route[0] not in {WINDOWS_IP, *_vmnet_addresses()}):
            raise ValueError("Unverified SSH route")
        result["active_ip"] = route[1]
        return result


def _public_metrics(raw: dict) -> dict:
    result = {}
    for field in (*METRICS, "disk_total_bytes", "disk_used_bytes", "uptime_seconds", "cpu_cores"):
        value = raw.get(field)
        if field.endswith("_bytes") or field in ("uptime_seconds", "cpu_cores"):
            result[field] = value if type(value) is int and 0 <= value < 2**63 else None
        else:
            result[field] = value if type(value) in (int, float) and 0 <= value <= 100 else None
    return result


def _identity(raw: dict) -> dict:
    def address(value):
        try:
            return str(ip_address(value)) if isinstance(value, str) else None
        except ValueError:
            return None

    addresses = raw.get("addresses")
    active = address(raw.get("active_ip"))
    values = [address(value) for value in addresses[:12]] if isinstance(addresses, list) else []
    result = {"active_ip": active, "addresses": list(dict.fromkeys(filter(None, [active, *values])))}
    for field in ("hostname", "os"):
        value = raw.get(field)
        result[field] = value if isinstance(value, str) and 0 < len(value) <= 120 and value.isprintable() else None
    runtime = raw.get("runtime_available")
    result["runtime_available"] = runtime if type(runtime) is bool else None
    return result


class InfrastructureMonitor:
    def __init__(self, project: Path, *, ssh_enabled: bool = False, thresholds: Thresholds = Thresholds()):
        self.project = project
        self.ssh_enabled = ssh_enabled
        self.thresholds = thresholds
        self._state = Lock()
        self._refresh = Lock()
        self._next_refresh = 0.0
        self._next_manual = 0.0
        self._attempts: dict = {}
        self._success: dict = {}

    def queue_client_refresh(self, role: Role) -> None:
        """Share cached samples and the existing probe lock with other dashboard views."""
        if not self.ssh_enabled or role not in (Role.CLIENT_A, Role.CLIENT_B):
            return
        with self._state:
            last = self._attempts.get(role)
            if last and (utc_now() - datetime.fromisoformat(last["checked_at"])).total_seconds() < 8:
                return
            if not self._refresh.acquire(blocking=False):
                return
        try:
            Thread(target=self._refresh_client, args=(role,), name="client-health-probe", daemon=True).start()
        except RuntimeError:
            self._refresh.release()
            raise

    def _refresh_client(self, role: Role) -> None:
        try:
            _, row = self._probe(role)
            self._record(role, row)
        finally:
            self._refresh.release()

    def _record(self, role, row) -> None:
        with self._state:
            self._attempts[role] = row
            if "metrics" in row:
                self._success[role] = row

    def _probe(self, role) -> tuple:
        start = monotonic()
        source = "Windows native counters and project volume" if role == "Windows" else "Strict SSH · Linux /proc and home filesystem"
        sample = {"source": source, "address": HOSTS[role]}
        try:
            raw = _windows_metrics(self.project) if role == "Windows" else _linux_metrics(role, self.project)
            metrics = _public_metrics(raw)
            partial = any(metrics[field] is None for field in METRICS)
            sample.update(state="PARTIAL" if partial else "ONLINE", metrics=metrics,
                          identity=_identity(raw), error="MEASUREMENTS_INCOMPLETE" if partial else None)
        except SSHConnectionError as exc:
            sample.update(state="UNAVAILABLE", error=exc.reason)
        except TimeoutError:
            sample.update(state="UNAVAILABLE", error="PROBE_TIMEOUT")
        except ConfigurationError:
            sample.update(state="UNAVAILABLE", error="PROBE_CONFIGURATION_UNAVAILABLE")
        except Exception:
            # The probe boundary can receive OS, Paramiko or parse errors. Never return their text.
            sample.update(state="UNAVAILABLE", error="PROBE_OR_ROUTE_FAILED")
        sample.update(checked_at=utc_now().isoformat(), duration_seconds=monotonic() - start)
        return role, sample

    def _refresh_samples(self) -> None:
        try:
            roles = tuple(HOSTS) if self.ssh_enabled else ("Windows",)
            with ThreadPoolExecutor(max_workers=len(roles)) as pool:
                for future in as_completed(pool.submit(self._probe, role) for role in roles):
                    role, row = future.result()
                    self._record(role, row)
        finally:
            with self._state:
                finished = monotonic()
                self._next_refresh = finished + self.thresholds.refresh_seconds
                self._next_manual = finished + 5
            self._refresh.release()

    def _reserve_refresh(self, *, manual: bool = False) -> bool:
        with self._state:
            ready_at = self._next_manual if manual else self._next_refresh
            return monotonic() >= ready_at and self._refresh.acquire(blocking=False)

    def queue_refresh(self, *, manual: bool = False) -> bool:
        if self._reserve_refresh(manual=manual):
            try:
                Thread(target=self._refresh_samples, name="infrastructure-probes", daemon=True).start()
            except RuntimeError:
                self._refresh.release()
                raise
            return True
        return False

    def refresh(self) -> dict:
        if self._reserve_refresh():
            self._refresh_samples()
        return self.snapshot()

    def snapshot(self) -> dict:
        nodes, alerts = {}, []
        now = utc_now()
        with self._state:
            for role, address in HOSTS.items():
                last = self._success.get(role)
                row = dict(self._attempts.get(role, {"state": "UNKNOWN", "address": address}))
                if last:
                    age = (now - datetime.fromisoformat(last["checked_at"])).total_seconds()
                    row["last_known"] = last
                    if age >= self.thresholds.stale_seconds:
                        if row["state"] != "UNAVAILABLE":
                            row.update(state="STALE", metrics=None)
                        alerts.append({"code": "STALE_MONITORING", "node": role, "at": last["checked_at"]})
                if row["state"] == "UNAVAILABLE":
                    alerts.append({"code": "NODE_UNREACHABLE", "node": role, "at": row["checked_at"]})
                metrics = row.get("metrics") or {}
                if metrics.get("disk_available_bytes") is not None and metrics["disk_available_bytes"] < self.thresholds.low_disk_gib * 2**30:
                    alerts.append({"code": "LOW_DISK", "node": role, "at": row["checked_at"],
                                   "value": metrics["disk_available_bytes"] / 2**30, "unit": "GiB free"})
                total, used = metrics.get("ram_total_bytes"), metrics.get("ram_used_bytes")
                if total and used is not None and used / total * 100 >= self.thresholds.high_memory_percent:
                    alerts.append({"code": "HIGH_MEMORY", "node": role, "at": row["checked_at"],
                                   "value": used / total * 100, "unit": "% used"})
                nodes[role] = row
            stamp = monotonic()
            cooldown = max(0, math.ceil(self._next_refresh - stamp))
            manual_cooldown = max(0, math.ceil(self._next_manual - stamp))
        return {"nodes": nodes, "alerts": alerts, "ssh_enabled": self.ssh_enabled,
                "refreshing": self._refresh.locked(), "cooldown_seconds": cooldown,
                "manual_cooldown_seconds": manual_cooldown, "stale_seconds": self.thresholds.stale_seconds}
