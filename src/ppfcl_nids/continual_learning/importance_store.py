"""Client-local, versioned synthetic EWC artifacts and acknowledgment-gated promotion."""

import hashlib
import io
import json
import pickle
import re
import sqlite3
from contextlib import closing, contextmanager
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch import Tensor

from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_POLICY, EWC_SPEC, EWCConfig, EWCState, capture_state, fisher_population, model_template, validate_state
from ppfcl_nids.federated_learning.asynchronous_aggregation import SyntheticUpdatePermit, decode_synthetic_update
from ppfcl_nids.models.neural_network import FlowMLP

_APPLICATION_ID = 0x50504557
_ARTIFACT_LIMIT = 512 * 1024
_SCHEMA = """CREATE TABLE ewc_artifacts (
    session_id TEXT NOT NULL, run_id TEXT NOT NULL, task_id TEXT NOT NULL, update_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0), parent_sha256 TEXT, sha256 TEXT NOT NULL,
    request TEXT NOT NULL, created_at TEXT NOT NULL, artifact BLOB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','COMMITTED','DISCARDED')), acknowledgment TEXT,
    PRIMARY KEY (session_id, run_id), UNIQUE (session_id, update_id), UNIQUE (session_id, sha256),
    CHECK ((status = 'COMMITTED') = (acknowledgment IS NOT NULL))
) STRICT"""


@dataclass(frozen=True, slots=True)
class SyntheticEWCAck:
    """Receipt validated at the coordinator boundary, outside synthetic FedAsync acceptance."""

    permit: SyntheticUpdatePermit
    accepted_version: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class EWCReference:
    permit: SyntheticUpdatePermit
    revision: int
    parent_sha256: str | None
    sha256: str

    @property
    def opaque_ref(self) -> str:
        return f"ewc:{self.revision}:{self.sha256}"


@dataclass(frozen=True, slots=True)
class CommittedEWCReference:
    reference: EWCReference
    acknowledgment: SyntheticEWCAck


def _json(data: dict) -> str:
    return json.dumps(data, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalEWCStore:
    """One isolated client database. SQLite atomically stores artifact bytes and promotion state."""

    def __init__(self, database: Path, *, client_id: str):
        if client_id not in ("A", "B"):
            raise ValueError("Expected client A or B")
        self.database = database.resolve()
        self.client_id = client_id
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._template = model_template()
        with closing(self._connect("rwc")) as db, db:
            db.execute("BEGIN IMMEDIATE")
            app = db.execute("PRAGMA application_id").fetchone()[0]
            if app == _APPLICATION_ID:
                if db.execute("SELECT client_id FROM ewc_owner").fetchall() != [(client_id,)]:
                    raise PermissionError("EWC store belongs to another client")
                self.worker_id = self._create_manual_outbox(db)
                return
            if app or db.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone():
                raise ValueError("EWC store requires a separate client database")
            db.execute("CREATE TABLE ewc_owner (client_id TEXT PRIMARY KEY) STRICT")
            db.execute("INSERT INTO ewc_owner VALUES (?)", (client_id,))
            db.execute(_SCHEMA)
            db.execute("""CREATE UNIQUE INDEX one_pending_ewc ON ewc_artifacts(session_id) WHERE status='PENDING'""")
            db.execute("""CREATE UNIQUE INDEX ewc_revisions ON ewc_artifacts(session_id, revision)
                           WHERE status='COMMITTED'""")
            db.execute("""CREATE UNIQUE INDEX ewc_tasks ON ewc_artifacts(session_id, task_id)
                           WHERE status='COMMITTED'""")
            db.execute("""CREATE TRIGGER retain_ewc_artifact BEFORE DELETE ON ewc_artifacts
                           BEGIN SELECT RAISE(ABORT,'EWC artifacts are retained'); END""")
            db.execute("""CREATE TRIGGER immutable_ewc_artifact BEFORE UPDATE ON ewc_artifacts
                WHEN OLD.status != 'PENDING' OR NEW.status NOT IN ('COMMITTED','DISCARDED')
                  OR NEW.session_id != OLD.session_id OR NEW.run_id != OLD.run_id OR NEW.task_id != OLD.task_id
                  OR NEW.update_id != OLD.update_id OR NEW.revision != OLD.revision
                  OR NEW.parent_sha256 IS NOT OLD.parent_sha256 OR NEW.sha256 != OLD.sha256
                  OR NEW.request != OLD.request OR NEW.created_at != OLD.created_at
                  OR NEW.artifact != OLD.artifact
                BEGIN SELECT RAISE(ABORT,'EWC artifact bytes and finalized states are immutable'); END""")
            db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            self.worker_id = self._create_manual_outbox(db)

    @staticmethod
    def _create_manual_outbox(db: sqlite3.Connection) -> str:
        db.execute("""CREATE TABLE IF NOT EXISTS manual_worker_identity (
            id INTEGER PRIMARY KEY CHECK (id=1), worker_id TEXT NOT NULL
        ) STRICT""")
        db.execute("INSERT OR IGNORE INTO manual_worker_identity VALUES (1, ?)", (uuid4().hex,))
        db.execute("""CREATE TABLE IF NOT EXISTS manual_update_outbox (
            session_id TEXT NOT NULL, run_id TEXT NOT NULL, body BLOB NOT NULL,
            PRIMARY KEY (session_id, run_id),
            FOREIGN KEY (session_id, run_id) REFERENCES ewc_artifacts(session_id, run_id)
        ) STRICT""")
        for operation in ("UPDATE", "DELETE"):
            db.execute(f"""CREATE TRIGGER IF NOT EXISTS manual_outbox_no_{operation.lower()}
                BEFORE {operation} ON manual_update_outbox
                BEGIN SELECT RAISE(ABORT, 'Manual update outbox is immutable'); END""")
        return db.execute("SELECT worker_id FROM manual_worker_identity WHERE id=1").fetchone()[0]

    def _connect(self, mode: str = "rw") -> sqlite3.Connection:
        db = sqlite3.connect(f"{self.database.as_uri()}?mode={mode}", uri=True)
        db.execute("PRAGMA foreign_keys=ON")
        if mode != "ro":
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA recursive_triggers=ON")
        return db

    @contextmanager
    def execution_transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize workers and recovery for this client using its own SQLite writer lock."""
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def _find(db: sqlite3.Connection, where: str, params: tuple) -> sqlite3.Row | None:
        cursor = db.cursor()
        cursor.row_factory = sqlite3.Row
        return cursor.execute(f"SELECT * FROM ewc_artifacts WHERE {where}", params).fetchone()

    def _permit(self, permit: SyntheticUpdatePermit) -> None:
        if (type(permit) is not SyntheticUpdatePermit or permit.client_id != self.client_id
                or permit.chunk_id not in {f"{'C' if self.client_id == 'A' else 'D'}{i}" for i in range(1, 21)}):
            raise PermissionError("EWC task does not belong to this client")

    def _decode(self, row: sqlite3.Row) -> tuple[EWCReference, EWCState]:
        raw = row["artifact"]
        if not 0 < len(raw) <= _ARTIFACT_LIMIT or _digest(raw) != row["sha256"]:
            raise ValueError("EWC artifact fingerprint mismatch")
        request = json.loads(row["request"])
        permit = SyntheticUpdatePermit(**request["permit"])
        self._permit(permit)
        if (permit.session_id, permit.run_id, permit.chunk_id, permit.update_id) != (
            row["session_id"], row["run_id"], row["task_id"], row["update_id"],
        ):
            raise ValueError("EWC artifact scope mismatch")
        try:
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            raise ValueError("Unreadable EWC artifact") from exc
        expected = self._metadata(request, row["revision"], row["parent_sha256"], row["created_at"])
        if (not isinstance(payload, dict) or set(payload) != {"metadata", "theta_star", "fisher"}
                or _json(payload["metadata"]) != _json(expected)):
            raise ValueError("EWC artifact metadata or model contract mismatch")
        state = EWCState(payload["theta_star"], payload["fisher"], EWCConfig(**request["config"]))
        validate_state(state, self._template)
        return EWCReference(permit, row["revision"], row["parent_sha256"], row["sha256"]), state

    @staticmethod
    def _metadata(request: dict, revision: int, parent: str | None, created_at: str) -> dict:
        return {
            "schema": "synthetic-ewc-state-v1", "policy": EWC_POLICY, "revision": revision,
            "model_metadata": EWC_SPEC.metadata(), "parent_sha256": parent, "created_at": created_at,
            "calibration_source": "synthetic-only", **request,
        }

    def prepare(self, permit: SyntheticUpdatePermit, artifact: bytes, features: Tensor, targets: Tensor,
                *, config: EWCConfig = EWCConfig()) -> EWCReference:
        """Stage an anchor from the exact candidate update and a bounded synthetic population."""
        with self.execution_transaction() as db:
            return self._prepare(db, permit, artifact, features, targets, config=config)

    def _prepare(self, db: sqlite3.Connection, permit: SyntheticUpdatePermit, artifact: bytes, features: Tensor,
                 targets: Tensor, *, config: EWCConfig, retain_update: bool = False) -> EWCReference:
        self._permit(permit)
        weights = decode_synthetic_update(permit, artifact, self._template)
        x, y = fisher_population(features, targets, config)
        request = {"permit": asdict(permit), "config": asdict(config), "fisher_count": len(x),
                   "population_sha256": _digest(x.numpy().tobytes() + y.numpy().tobytes())}
        encoded = _json(request)
        key = (permit.session_id, permit.run_id)
        prior = self._find(db, "session_id=? AND run_id=?", key)
        if prior is not None:
            if prior["request"] != encoded or prior["status"] == "DISCARDED":
                raise ValueError("Conflicting or discarded EWC run")
            reference = self._decode(prior)[0]
        else:
            if self._find(db, "session_id=? AND status='PENDING'", (permit.session_id,)):
                raise ValueError("Resolve the pending EWC state before preparing another")
            if self._find(db, "session_id=? AND task_id=? AND status='COMMITTED'",
                          (permit.session_id, permit.chunk_id)):
                raise ValueError("EWC task is already committed")
            last = self._find(db, "session_id=? AND status='COMMITTED' ORDER BY revision DESC LIMIT 1",
                              (permit.session_id,))
            previous = self._decode(last)[1] if last is not None else None
            revision, parent = (last["revision"] + 1, last["sha256"]) if last is not None else (1, None)
            with torch.device("meta"):
                model = FlowMLP(EWC_SPEC)
            model.load_state_dict(weights, assign=True)
            state = capture_state(model, x, y, config=config, previous=previous)
            now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            with io.BytesIO() as stream:
                torch.save({"metadata": self._metadata(request, revision, parent, now),
                            "theta_star": state.theta_star, "fisher": state.fisher}, stream)
                body = stream.getvalue()
            if not 0 < len(body) <= _ARTIFACT_LIMIT:
                raise ValueError("Oversized EWC artifact")
            digest = _digest(body)
            db.execute("""INSERT INTO ewc_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', NULL)""",
                       (*key, permit.chunk_id, permit.update_id, revision, parent, digest, encoded, now, body))
            reference = EWCReference(permit, revision, parent, digest)
        if retain_update:
            db.execute("INSERT OR IGNORE INTO manual_update_outbox VALUES (?, ?, ?)", (*key, artifact))
            if db.execute("SELECT body FROM manual_update_outbox WHERE session_id=? AND run_id=?",
                          key).fetchone()[0] != artifact:
                raise ValueError("Conflicting manual outbox artifact")
        return reference

    def _prepared_update(self, db: sqlite3.Connection, session_id: str, run_id: str
                         ) -> tuple[EWCReference, bytes] | None:
        key = (session_id, run_id)
        row = self._find(db, "session_id=? AND run_id=?", key)
        if row is None or row["status"] == "DISCARDED":
            return None
        artifact = db.execute("SELECT body FROM manual_update_outbox WHERE session_id=? AND run_id=?", key).fetchone()
        if artifact is None:
            raise ValueError("Pending manual EWC revision has no durable update")
        reference, _ = self._decode(row)
        return reference, artifact[0]

    def _discard_if_pending(self, db: sqlite3.Connection, session_id: str, run_id: str) -> None:
        row = self._find(db, "session_id=? AND run_id=?", (session_id, run_id))
        if row is not None and row["status"] == "COMMITTED":
            raise ValueError("Committed EWC state cannot be discarded")
        db.execute("""UPDATE ewc_artifacts SET status='DISCARDED'
                      WHERE session_id=? AND run_id=? AND status='PENDING'""", (session_id, run_id))

    def load(self, reference: EWCReference) -> EWCState:
        """Read a specific verified artifact for inspection, including pending artifacts."""
        self._permit(reference.permit)
        with closing(self._connect("ro")) as db:
            row = self._find(db, "session_id=? AND run_id=?", (reference.permit.session_id, reference.permit.run_id))
            if row is None or row["status"] == "DISCARDED":
                raise LookupError("EWC artifact is missing or discarded")
            actual, state = self._decode(row)
            if actual != reference:
                raise ValueError("EWC reference does not match the stored artifact")
            return state

    def load_for_task(self, permit: SyntheticUpdatePermit) -> EWCState | None:
        """Select only the latest committed EWC state in this client's task session."""
        self._permit(permit)
        with closing(self._connect("ro")) as db:
            row = self._find(db, "session_id=? AND status='COMMITTED' ORDER BY revision DESC LIMIT 1",
                             (permit.session_id,))
            return self._decode(row)[1] if row is not None else None

    def restore(self, session_id: str) -> tuple[EWCReference | None, EWCReference | None]:
        """Return the committed and pending references. A pending artifact is never activated here."""
        with closing(self._connect("ro")) as db, db:
            db.execute("BEGIN")
            current = self._find(db, "session_id=? AND status='COMMITTED' ORDER BY revision DESC LIMIT 1",
                                 (session_id,))
            pending = self._find(db, "session_id=? AND status='PENDING'", (session_id,))
            return (self._decode(current)[0] if current is not None else None,
                    self._decode(pending)[0] if pending is not None else None)

    def commit(self, acknowledgment: SyntheticEWCAck) -> CommittedEWCReference:
        """Promote after a matching durable coordinator receipt, or replay the same promotion."""
        if (type(acknowledgment) is not SyntheticEWCAck
                or type(acknowledgment.accepted_version) is not int
                or acknowledgment.accepted_version <= acknowledgment.permit.base_version
                or not re.fullmatch(r"[0-9a-f]{64}", acknowledgment.checkpoint_sha256)):
            raise ValueError("A committed synthetic FedAsync receipt is required")
        permit = acknowledgment.permit
        self._permit(permit)
        encoded = _json(asdict(acknowledgment))
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = self._find(db, "session_id=? AND run_id=?", (permit.session_id, permit.run_id))
            if row is None or row["status"] == "DISCARDED":
                raise LookupError("No promotable EWC artifact for this receipt")
            reference, _ = self._decode(row)
            if reference.permit != permit:
                raise ValueError("EWC artifact and committed receipt differ")
            if row["status"] == "COMMITTED":
                if row["acknowledgment"] != encoded:
                    raise ValueError("Conflicting EWC promotion receipt")
            else:
                db.execute("UPDATE ewc_artifacts SET status='COMMITTED', acknowledgment=? WHERE session_id=? AND run_id=?",
                           (encoded, permit.session_id, permit.run_id))
            return CommittedEWCReference(reference, acknowledgment)

    def discard(self, session_id: str, run_id: str) -> None:
        """Discard after a definitive rejection or an abandoned, unsubmitted run. Keep its bytes."""
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = self._find(db, "session_id=? AND run_id=?", (session_id, run_id))
            if row is None or row["status"] == "COMMITTED":
                raise ValueError("Only a pending EWC run may be discarded")
            if row["status"] == "PENDING":
                db.execute("UPDATE ewc_artifacts SET status='DISCARDED' WHERE session_id=? AND run_id=?",
                           (session_id, run_id))
