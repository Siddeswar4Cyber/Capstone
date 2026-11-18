"""Open an explicitly requested, host-verified VM administration session."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ppfcl_nids.configuration.settings import ConfigurationError, Role

if TYPE_CHECKING:
    from paramiko import SSHClient


_VM_PREFIXES = {
    Role.CLIENT_A: "CLIENT_A",
    Role.CLIENT_B: "CLIENT_B",
    Role.ORGANIZATION: "ORG",
}
from ppfcl_nids.project_paths import PROJECT_ROOT

_PROJECT_ENV = PROJECT_ROOT / ".env"


class SSHConnectionError(ConnectionError):
    def __init__(self, reason: str):
        super().__init__("SSH connection failed. Check VM reachability, host trust, and authentication.")
        self.reason = reason


def _read_target(env_file: Path, prefix: str) -> dict[str, str]:
    names = {f"{prefix}_{suffix}" for suffix in (
        "HOST", "SSH_PORT", "SSH_USER", "SSH_PASSWORD", "SSH_KEY_PATH",
    )}
    try:
        lines = env_file.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        raise ConfigurationError("Could not read the SSH .env file as UTF-8") from None

    values: dict[str, str] = {}
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or name not in names:
            continue
        if name in values:
            raise ConfigurationError(f"Duplicate setting {name} in .env")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values[name] = value
    return values


@contextmanager
def ssh_session(
    role: Role, *, env_file: Path = _PROJECT_ENV, known_hosts: Path | None = None,
    expected_host: str | None = None, timeout: float = 10,
) -> Iterator[SSHClient]:
    """Read the selected VM's settings and close its connection on exit."""
    if role not in _VM_PREFIXES:
        raise ConfigurationError("SSH requires the role client-a, client-b, or organization")

    prefix = _VM_PREFIXES[role]
    values = _read_target(env_file, prefix)
    host = values.get(f"{prefix}_HOST", "")
    if expected_host is not None and host != expected_host:
        raise ConfigurationError("SSH address is outside the approved route")
    user = values.get(f"{prefix}_SSH_USER", "")
    password = values.get(f"{prefix}_SSH_PASSWORD") or None
    key_value = values.get(f"{prefix}_SSH_KEY_PATH", "")
    if not host or not user or not (key_value or password):
        raise ConfigurationError(
            f"Set {prefix}_HOST and {prefix}_SSH_USER, plus {prefix}_SSH_KEY_PATH "
            f"or {prefix}_SSH_PASSWORD, in .env"
        )

    try:
        port = int(values.get(f"{prefix}_SSH_PORT", ""))
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise ConfigurationError(f"{prefix}_SSH_PORT must be an integer from 1 to 65535")

    key_path = None
    if key_value:
        try:
            key_path = env_file.parent / Path(key_value).expanduser()
            key_exists = key_path.is_file()
        except (OSError, ValueError, RuntimeError):
            key_exists = False
        if not key_exists:
            raise ConfigurationError(f"{prefix}_SSH_KEY_PATH must point to a local key file")

    try:
        import paramiko
        from paramiko import RejectPolicy, SSHClient, SSHException
    except ImportError:
        raise ConfigurationError("SSH requires the optional ssh dependency group") from None

    with SSHClient() as client:
        try:
            host_keys = known_hosts if known_hosts is not None else Path.home() / ".ssh" / "known_hosts"
            client.load_system_host_keys(str(host_keys))
        except (OSError, SSHException, ValueError):
            raise ConfigurationError("Could not load the SSH known-hosts file") from None
        client.set_missing_host_key_policy(RejectPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=user,
                key_filename=str(key_path) if key_path is not None else None,
                password=password,
                allow_agent=False,
                look_for_keys=False,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
                channel_timeout=timeout,
            )
        except (OSError, SSHException, ValueError) as exc:
            reason = "SSH_CONNECTION_OR_HOST_TRUST_FAILED"
            if isinstance(exc, getattr(paramiko, "AuthenticationException", ())):
                reason = "SSH_AUTHENTICATION_FAILED"
            elif isinstance(exc, getattr(paramiko, "BadHostKeyException", ())):
                reason = "SSH_HOST_KEY_MISMATCH"
            elif isinstance(exc, TimeoutError):
                reason = "SSH_CONNECTION_TIMEOUT"
            raise SSHConnectionError(reason) from None
        yield client
