"""Side-effect-free validation of shared machine settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """A required machine setting is absent or invalid."""


class Role(StrEnum):
    SERVER = "server"
    CLIENT_A = "client-a"
    CLIENT_B = "client-b"
    ORGANIZATION = "organization"


class Device(StrEnum):
    CPU = "cpu"
    AUTO = "auto"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class Settings:
    role: Role
    state_dir: Path
    server_url: str
    device: Device

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if env is None else env
        try:
            role = Role(values.get("PPFCL_ROLE", "").strip())
        except ValueError:
            raise ConfigurationError(
                "PPFCL_ROLE must be server, client-a, client-b, or organization"
            ) from None

        try:
            device = Device(values.get("PPFCL_DEVICE", "cpu").strip())
        except ValueError:
            raise ConfigurationError("PPFCL_DEVICE must be cpu, auto, or cuda") from None

        state_dir = values.get("PPFCL_STATE_DIR", "state").strip()
        if not state_dir or "\0" in state_dir:
            raise ConfigurationError("PPFCL_STATE_DIR must be a non-empty path")

        server_url = values.get("PPFCL_SERVER_URL", "").strip()
        if not server_url and "PPFCL_SERVER_URL" not in values and role is Role.SERVER:
            server_url = "http://127.0.0.1:8000"

        return cls(role, Path(state_dir), _server_origin(server_url), device)


def _server_origin(value: str) -> str:
    try:
        url = urlsplit(value)
        hostname = url.hostname
        port = url.port
    except ValueError:
        raise ConfigurationError("PPFCL_SERVER_URL has an invalid host or port") from None

    if (
        not hostname
        or url.scheme not in {"http", "https"}
        or url.path not in {"", "/"}
        or any(char.isspace() or ord(char) < 32 for char in value)
        or any(char in value for char in "\\?#")
        or url.username is not None
        or port == 0
        or url.netloc.endswith(":")
    ):
        raise ConfigurationError("PPFCL_SERVER_URL must be an HTTP(S) origin")
    if url.scheme == "http" and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ConfigurationError("PPFCL_SERVER_URL requires HTTPS for a remote host")
    return f"{url.scheme}://{url.netloc}"
