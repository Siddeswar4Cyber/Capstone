"""Transactional FedAsync for explicitly approved, in-process synthetic updates."""

import hashlib
import io
import json
import math
import os
import pickle
import re
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

import torch
from torch import Tensor

from ppfcl_nids.federated_learning.sessions import ActiveDemoError, DemoSessionManager, Record, _timestamp
from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec
from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0, load_verified_v0, validate_parameters
from ppfcl_nids.models.neural_network import FlowMLP
from ppfcl_nids.federated_learning.synthetic_round import FRAME_LIMIT

_SPEC = FlowModelSpec(BIFLOW_CONTRACT_ID)
_UPDATE_SCHEMA = "synthetic-fedasync-full-model-v1"
_CHECKPOINT_SCHEMA = "synthetic-fedasync-checkpoint-v1"
_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_OBJECT = re.compile(r"[0-9a-f]{64}\.pt")
_RECEIPT = """
    SELECT u.*, m.parent_version, m.staleness, m.aggregation_weight,
           m.checkpoint_sha256, p.checkpoint_sha256 AS parent_checkpoint_sha256,
           b.checkpoint_sha256 AS base_checkpoint_sha256, c.policy_id, c.alpha_base
    FROM federated_updates u
    JOIN session_model_versions m ON m.session_id=u.session_id AND m.version=u.accepted_version
    JOIN session_model_versions p ON p.session_id=m.session_id AND p.version=m.parent_version
    JOIN session_model_versions b ON b.session_id=u.session_id AND b.version=u.base_version
    JOIN fedasync_checkpoints c ON c.session_id=m.session_id AND c.version=m.version
    WHERE u.session_id=? AND u.client_id=? AND u.update_id=?
"""


@dataclass(frozen=True, slots=True)
class SyntheticStalenessPolicy:
    policy_id: ClassVar[str] = "synthetic-inverse-staleness-v1"
    alpha_base: float = 0.5

    def __post_init__(self) -> None:
        if (type(self.alpha_base) not in (int, float) or not math.isfinite(self.alpha_base)
                or not 0 < self.alpha_base <= 1):
            raise ValueError("Synthetic alpha_base must be finite and in (0, 1]")

    def weight(self, staleness: int) -> float:
        if type(staleness) is not int or staleness < 0:
            raise ValueError("Invalid FedAsync staleness")
        alpha = self.alpha_base / (1 + staleness)
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Invalid FedAsync mixing weight")
        return alpha


@dataclass(frozen=True, slots=True)
class SyntheticUpdatePermit:
    """Out-of-band local allowlist entry. A matching digest establishes integrity only."""

    session_id: str
    client_id: str
    update_id: str
    run_id: str
    chunk_id: str
    base_version: int
    base_checkpoint_sha256: str
    artifact_sha256: str

    def metadata(self) -> dict:
        return {
            "schema": _UPDATE_SCHEMA, "session_id": self.session_id, "client_id": self.client_id,
            "update_id": self.update_id, "run_id": self.run_id, "chunk_id": self.chunk_id,
            "base_version": self.base_version, "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "model_metadata": _SPEC.metadata(),
        }


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _serialize(payload: dict) -> bytes:
    with io.BytesIO() as stream:
        torch.save(payload, stream)
        return stream.getvalue()


def _load(body: bytes) -> dict:
    if not 0 < len(body) <= FRAME_LIMIT:
        raise ValueError("Missing or oversized FedAsync artifact")
    try:
        payload = torch.load(io.BytesIO(body), weights_only=True, map_location="cpu")
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError("Unreadable FedAsync tensor artifact") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected a FedAsync tensor envelope")
    return payload


def _checked_state(payload: dict, metadata: dict, template: Mapping[str, Tensor]) -> dict[str, Tensor]:
    try:
        actual = json.dumps({key: value for key, value in payload.items() if key != "state_dict"},
                            sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed FedAsync metadata") from exc
    if "state_dict" not in payload or actual != json.dumps(metadata, sort_keys=True):
        raise ValueError("FedAsync artifact metadata or model contract mismatch")
    state = payload["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Expected a full model state")
    validate_parameters(state, template)
    return state


def decode_synthetic_update(permit: SyntheticUpdatePermit, artifact: bytes,
                            template: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Validate the exact full-model artifact. Callers enforce their own trust boundary."""
    if type(artifact) is not bytes or not 0 < len(artifact) <= FRAME_LIMIT:
        raise ValueError("Missing or oversized FedAsync artifact")
    if _digest(artifact) != permit.artifact_sha256:
        raise ValueError("FedAsync update artifact fingerprint mismatch")
    return _checked_state(_load(artifact), permit.metadata(), template)


def encode_synthetic_update(permit: SyntheticUpdatePermit, state: Mapping[str, Tensor]) -> bytes:
    """Serialize a synthetic full model with the existing FedAsync envelope."""
    return _serialize({**permit.metadata(), "state_dict": dict(state)})


def decode_training_checkpoint(permit: SyntheticUpdatePermit, body: bytes,
                               template: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not 0 < len(body) <= FRAME_LIMIT or _digest(body) != permit.base_checkpoint_sha256:
        raise ValueError("Training checkpoint size or fingerprint mismatch")
    if permit.base_version == 0:
        return decode_verified_v0(body, permit.base_checkpoint_sha256, model_spec=_SPEC)["state_dict"]
    metadata = {"schema": _CHECKPOINT_SCHEMA, "session_id": permit.session_id,
                "version": permit.base_version, "model_metadata": _SPEC.metadata()}
    return _checked_state(_load(body), metadata, template)


def mix_full_states(head, update, alpha, template):
    validate_parameters(head, template)
    validate_parameters(update, template)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Invalid FedAsync weight")
    with torch.no_grad():
        mixed = {name: ((1 - alpha) * value.double() + alpha * update[name].double()).to(value.dtype)
                 for name, value in head.items()}
    validate_parameters(mixed, template)
    return mixed


def _publish(path: Path, body: bytes) -> None:
    """Publish a same-volume, fsynced object without replacing an existing file."""
    if not path.exists():
        stream = NamedTemporaryFile(dir=path.parent, prefix=".stage-", delete=False)
        staging = Path(stream.name)
        try:
            with stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "nt":
                staging.rename(path)
            else:
                os.link(staging, path)
        finally:
            staging.unlink(missing_ok=True)
    if path.is_symlink() or path.stat().st_size != len(body) or path.read_bytes() != body:
        raise ValueError("Immutable FedAsync checkpoint collision or corruption")


class SyntheticFedAsync:
    """Local synthetic acceptance. Network authentication and real run approval are out of scope."""

    def __init__(self, sessions: DemoSessionManager, *, trusted_updates: Iterable[SyntheticUpdatePermit],
                 policy: SyntheticStalenessPolicy = SyntheticStalenessPolicy()):
        self.sessions = sessions
        self._trusted_updates = frozenset(trusted_updates)
        self.policy = policy
        self.checkpoint_directory = sessions.database.with_suffix(sessions.database.suffix + ".fedasync")
        with torch.device("meta"):
            self._template = FlowMLP(_SPEC).state_dict()

    def _permit(self, permit: SyntheticUpdatePermit) -> None:
        if type(permit) is not SyntheticUpdatePermit or permit not in self._trusted_updates:
            raise PermissionError("Update lacks a trusted synthetic permit")
        if (permit.client_id not in ("A", "B")
                or not all(isinstance(value, str) and _ID.fullmatch(value) for value in (
                    permit.session_id, permit.run_id, permit.update_id, permit.chunk_id))
                or type(permit.base_version) is not int or permit.base_version < 0
                or not isinstance(permit.base_checkpoint_sha256, str)
                or not _SHA.fullmatch(permit.base_checkpoint_sha256)):
            raise ValueError("Malformed synthetic permit")

    def _ensure_directory(self) -> None:
        self.checkpoint_directory.mkdir(exist_ok=True)
        if self.checkpoint_directory.is_symlink():
            raise ValueError("FedAsync checkpoint directory cannot be a symbolic link")

    def _version(self, db: sqlite3.Connection, session_id: str, version: int) -> sqlite3.Row:
        row = db.execute("SELECT * FROM session_model_versions WHERE session_id=? AND version=?",
                         (session_id, version)).fetchone()
        if row is None:
            raise ValueError("Base or model version does not belong to the session")
        return row

    def _body(self, db: sqlite3.Connection, row: sqlite3.Row) -> bytes:
        digest = row["checkpoint_sha256"]
        path = self.checkpoint_directory / f"{digest}.pt"
        if not _SHA.fullmatch(digest) or path != Path(row["checkpoint_path"]):
            raise ValueError("Unexpected FedAsync checkpoint location")
        stored = db.execute("SELECT body FROM fedasync_checkpoints WHERE session_id=? AND version=?",
                            (row["session_id"], row["version"])).fetchone()
        if stored is None or _digest(stored["body"]) != digest:
            raise ValueError("Missing or corrupted committed FedAsync recovery bytes")
        self._ensure_directory()
        _publish(path, stored["body"])
        return stored["body"]

    def _state(self, db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Tensor]:
        if row["version"] == 0:
            if row["checkpoint_sha256"] != self.sessions.v0.checkpoint_sha256:
                raise ValueError("Session root checkpoint does not match pinned v0")
            return load_verified_v0(Path(row["checkpoint_path"]), row["checkpoint_sha256"],
                                    model_spec=_SPEC)["state_dict"]
        body = self._body(db, row)
        metadata = {"schema": _CHECKPOINT_SCHEMA, "session_id": row["session_id"],
                    "version": row["version"], "model_metadata": _SPEC.metadata()}
        return _checked_state(_load(body), metadata, self._template)

    def load_training_base(self, session_id: str, version: int, checkpoint_sha256: str) -> dict[str, Tensor]:
        """Verify a pinned session base, including recovery of missing committed successor files."""
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = self._version(db, session_id, version)
            if row["checkpoint_sha256"] != checkpoint_sha256:
                raise ValueError("FedAsync training base checkpoint fingerprint mismatch")
            return self._state(db, row)

    def export_training_checkpoint(self, permit: SyntheticUpdatePermit) -> bytes:
        """Return the exact session checkpoint bytes for the approved remote synthetic run."""
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = self._version(db, permit.session_id, permit.base_version)
            if row["checkpoint_sha256"] != permit.base_checkpoint_sha256:
                raise ValueError("Training base differs from the captured session checkpoint")
            if permit.base_version == 0:
                body = self.sessions.v0.verify().read_bytes()
            else:
                body = self._body(db, row)
            decode_training_checkpoint(permit, body, self._template)
            return body

    def accept_async_update(self, permit: SyntheticUpdatePermit, artifact: bytes) -> Record:
        """Accept once, or return the same committed receipt for an exact permitted replay."""
        self._permit(permit)
        client_state = decode_synthetic_update(permit, artifact, self._template)
        key = (permit.session_id, permit.client_id, permit.update_id)
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(_RECEIPT, key).fetchone()
            if prior is not None:
                if prior["payload_sha256"] != permit.artifact_sha256:
                    raise ValueError("Conflicting replay for this FedAsync update ID")
                self._body(db, self._version(db, permit.session_id, prior["accepted_version"]))
                return dict(prior)

            session = self.sessions._session(db, permit.session_id)
            if session["status"] != "ACTIVE":
                raise ActiveDemoError("FedAsync requires an active demo")
            if (session["contract_id"], session["contract_fingerprint"], session["v0_reference_sha256"]) != (
                _SPEC.contract_id, _SPEC.metadata()["contract_fingerprint"], self.sessions.v0.reference_sha256,
            ):
                raise ValueError("Unexpected FedAsync session contract or v0 reference")
            chunk_key = (permit.session_id, permit.client_id, permit.chunk_id)
            managed = db.execute("""SELECT execution_id,session_id,client_id,chunk_id,state,base_version,
                                           base_checkpoint_sha256,update_sha256 FROM manual_chunk_runs
                                    WHERE execution_id IN (?,?) OR
                                    (session_id=? AND client_id=? AND chunk_id=? AND state!='FAILED')""",
                                 (permit.run_id, permit.update_id, *chunk_key)).fetchall()
            if managed and (permit.run_id != permit.update_id or [tuple(row) for row in managed] != [
                (permit.run_id, *chunk_key, "SUBMITTED", permit.base_version,
                 permit.base_checkpoint_sha256, permit.artifact_sha256),
            ]):
                raise PermissionError("FedAsync requires the exact submitted manual execution")
            if not db.execute("""SELECT 1 FROM client_chunk_progress WHERE session_id=? AND client_id=?
                                 AND chunk_id=?""", chunk_key).fetchone():
                raise PermissionError("Client does not own this session chunk")
            if db.execute("""SELECT 1 FROM federated_updates WHERE session_id=? AND client_id=?
                             AND chunk_id=?""", chunk_key).fetchone():
                raise ValueError("Session chunk already has an accepted FedAsync update")

            head_version = session["head_version"]
            if permit.base_version > head_version:
                raise ValueError("FedAsync update references a future base version")
            base = self._version(db, permit.session_id, permit.base_version)
            if base["checkpoint_sha256"] != permit.base_checkpoint_sha256:
                raise ValueError("FedAsync training base checkpoint fingerprint mismatch")
            head = self._version(db, permit.session_id, head_version)
            head_state = self._state(db, head)
            if base["version"] != head_version:
                self._state(db, base)
            staleness = head_version - permit.base_version
            alpha = self.policy.weight(staleness)
            mixed = mix_full_states(head_state, client_state, alpha, self._template)
            return self._commit(db, permit, head_version + 1, staleness, alpha, mixed)

    def _commit(self, db: sqlite3.Connection, permit: SyntheticUpdatePermit, version: int,
                staleness: int, alpha: float, state: dict[str, Tensor]) -> Record:
        body = _serialize({"schema": _CHECKPOINT_SCHEMA, "session_id": permit.session_id, "version": version,
                           "model_metadata": _SPEC.metadata(), "state_dict": state})
        digest = _digest(body)
        path = self.checkpoint_directory / f"{digest}.pt"
        self._ensure_directory()
        _publish(path, body)
        now = _timestamp()
        db.execute("""INSERT INTO session_model_versions (
                      session_id, version, checkpoint_path, checkpoint_sha256, created_at, parent_version,
                      client_id, chunk_id, base_version, staleness, aggregation_weight)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (permit.session_id, version, str(path), digest, now, version - 1, permit.client_id,
                    permit.chunk_id, permit.base_version, staleness, alpha))
        db.execute("""INSERT INTO fedasync_checkpoints (session_id, version, policy_id, alpha_base, body)
                      VALUES (?, ?, ?, ?, ?)""",
                   (permit.session_id, version, self.policy.policy_id, self.policy.alpha_base, body))
        db.execute("""INSERT INTO federated_updates (
                      session_id, client_id, update_id, run_id, chunk_id, base_version, accepted_version,
                      payload_sha256, accepted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (permit.session_id, permit.client_id, permit.update_id, permit.run_id, permit.chunk_id,
                    permit.base_version, version, permit.artifact_sha256, now))
        db.execute("INSERT INTO session_events (session_id, event_type, occurred_at) VALUES (?, ?, ?)",
                   (permit.session_id, "FEDASYNC_UPDATE_ACCEPTED", now))
        return dict(db.execute(_RECEIPT, (permit.session_id, permit.client_id, permit.update_id)).fetchone())

    def recover_checkpoints(self) -> None:
        """Restore missing committed objects and discard unpublished artifacts under the writer lock."""
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM session_model_versions").fetchall()
            referenced = {Path(row["checkpoint_path"]) for row in rows}
            for row in rows:
                if row["version"] > 0:
                    self._state(db, row)
            self._ensure_directory()
            for path in self.checkpoint_directory.iterdir():
                if (path not in referenced and path.is_file() and not path.is_symlink()
                        and (path.name.startswith(".stage-") or _OBJECT.fullmatch(path.name))):
                    path.unlink()
