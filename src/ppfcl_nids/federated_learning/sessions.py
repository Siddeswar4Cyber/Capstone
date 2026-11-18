"""Coordinator-side demo metadata. Client datasets and privacy ledgers stay separate."""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec
from ppfcl_nids.federated_learning.initial_aggregation import load_verified_v0

DEFAULT_DATABASE = Path("state/demo-sessions.sqlite")
_BASELINE_PLAN = "93c2ee0453bb2af6d5062281388eac3d02b2608892e91bc23ea7e5d693073825"
_REGISTRY = Path("state/task8b/research_nonsecure-real/netflow-biflow-11-baseline-v1") / _BASELINE_PLAN
_SPEC = FlowModelSpec(BIFLOW_CONTRACT_ID)
_APPLICATION_ID = 0x50464453
_SCHEMA_VERSION = 5
LOCAL_SYNTHETIC = "LOCAL_SYNTHETIC"
REMOTE_SYNTHETIC = "REMOTE_SYNTHETIC"
_CLIENTS = (("A", "C"), ("B", "D"))
_CHUNKS_PER_CLIENT = 20

Record = dict[str, str | int | float | None]


class ActiveDemoError(RuntimeError):
    """A different demo already occupies the active session slot."""


class UnknownDemoError(LookupError):
    """The requested session does not exist."""


@dataclass(frozen=True, slots=True)
class ResearchV0:
    reference_path: Path
    reference_sha256: str
    checkpoint_sha256: str

    def verify(self) -> Path:
        """Validate the pinned reference and checkpoint without writing either artifact."""
        if self.reference_path.is_symlink():
            raise ValueError("Research v0 reference must be the original registry file")
        content = self.reference_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.reference_sha256:
            raise ValueError("Research v0 registry reference fingerprint mismatch")
        reference = json.loads(content)
        if (not isinstance(reference, dict)
                or reference.get("schema") != "research-global-v0-reference-v1"
                or reference.get("model_version") != 0
                or reference.get("designation") != "local_research_demo_root"
                or reference.get("checkpoint_sha256") != self.checkpoint_sha256
                or reference.get("feature_contract") != _SPEC.metadata()
                or reference.get("training_mode") != "research_nonsecure"):
            raise ValueError("Research v0 registry reference or model contract mismatch")
        if reference.get("release") != {
            "coordinator_local_reference": True, "public_release": False, "production_release": False,
            "organization_inference": False, "new_training_authorization": False,
        }:
            raise ValueError("Research v0 registry release scope mismatch")
        name = reference.get("checkpoint_filename")
        if (not isinstance(name, str) or not name.endswith(".pt") or Path(name).name != name
                or "/" in name or "\\" in name or ":" in name):
            raise ValueError("Invalid research v0 checkpoint filename")
        checkpoint = self.reference_path.resolve().with_name(name)
        if checkpoint.is_symlink():
            raise ValueError("Research v0 checkpoint must be the original registry file")
        artifact = load_verified_v0(checkpoint, self.checkpoint_sha256, model_spec=_SPEC)
        tags = artifact.get("artifact_metadata", {})
        expected = {
            "training_mode": "research_nonsecure", "secure_rng": False, "arm": "baseline",
            "approval_mode": "offline_research_training_only",
            "public_model_release": "DISABLED", "production_release": "DISABLED",
            "organization_inference": "DISABLED",
        }
        if not isinstance(tags, dict) or any(tags.get(key) != value for key, value in expected.items()):
            raise ValueError("Research v0 checkpoint release scope mismatch")
        return checkpoint


PINNED_RESEARCH_V0 = ResearchV0(
    _REGISTRY / "research-global-v0.json",
    "353b4c9f885f8122505a6f67d8ff22e7acc8c8d6bb36fa5136babbaf0f3e685e",
    "415907f3a95c5ee63d930b080f36799f8b8e09166a41ccf0ff5414031e31c29c",
)

_TABLES = (
    """CREATE TABLE demo_sessions (
        session_id TEXT PRIMARY KEY,
        start_request_id TEXT UNIQUE,
        status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'CLOSED')),
        created_at TEXT NOT NULL,
        closed_at TEXT,
        v0_reference_sha256 TEXT NOT NULL,
        contract_id TEXT NOT NULL,
        contract_fingerprint TEXT NOT NULL,
        CHECK ((status = 'CLOSED') = (closed_at IS NOT NULL))
    ) STRICT""",
    """CREATE UNIQUE INDEX one_active_demo ON demo_sessions(status) WHERE status = 'ACTIVE'""",
    """CREATE TABLE client_chunk_progress (
        session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
        client_id TEXT NOT NULL CHECK (client_id IN ('A', 'B')),
        chunk_number INTEGER NOT NULL CHECK (chunk_number BETWEEN 1 AND 20),
        chunk_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
        PRIMARY KEY (session_id, client_id, chunk_id),
        CHECK (chunk_id = (CASE client_id WHEN 'A' THEN 'C' ELSE 'D' END) || chunk_number)
    ) STRICT""",
    """CREATE TABLE session_model_versions (
        session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
        version INTEGER NOT NULL CHECK (version >= 0),
        checkpoint_path TEXT NOT NULL,
        checkpoint_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        parent_version INTEGER,
        client_id TEXT,
        chunk_id TEXT,
        base_version INTEGER,
        staleness INTEGER,
        aggregation_weight REAL,
        PRIMARY KEY (session_id, version),
        FOREIGN KEY (session_id, parent_version) REFERENCES session_model_versions(session_id, version),
        FOREIGN KEY (session_id, base_version) REFERENCES session_model_versions(session_id, version),
        FOREIGN KEY (session_id, client_id, chunk_id)
            REFERENCES client_chunk_progress(session_id, client_id, chunk_id),
        CHECK (
            (version = 0 AND parent_version IS NULL AND client_id IS NULL AND chunk_id IS NULL
                AND base_version IS NULL AND staleness IS NULL AND aggregation_weight IS NULL)
            OR (version > 0 AND parent_version IS NOT NULL AND parent_version = version - 1
                AND client_id IS NOT NULL AND chunk_id IS NOT NULL AND base_version IS NOT NULL
                AND base_version <= parent_version AND staleness IS NOT NULL
                AND staleness = parent_version - base_version AND aggregation_weight IS NOT NULL
                AND aggregation_weight > 0 AND aggregation_weight <= 1)
        )
    ) STRICT""",
    """CREATE TABLE federated_updates (
        session_id TEXT NOT NULL,
        client_id TEXT NOT NULL,
        update_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        base_version INTEGER NOT NULL,
        accepted_version INTEGER NOT NULL CHECK (accepted_version > base_version),
        payload_sha256 TEXT NOT NULL,
        accepted_at TEXT NOT NULL,
        PRIMARY KEY (session_id, client_id, update_id),
        UNIQUE (session_id, client_id, chunk_id),
        UNIQUE (session_id, accepted_version),
        FOREIGN KEY (session_id, client_id, chunk_id)
            REFERENCES client_chunk_progress(session_id, client_id, chunk_id),
        FOREIGN KEY (session_id, base_version) REFERENCES session_model_versions(session_id, version),
        FOREIGN KEY (session_id, accepted_version) REFERENCES session_model_versions(session_id, version)
    ) STRICT""",
    """CREATE TABLE client_ewc_state (
        session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
        client_id TEXT NOT NULL CHECK (client_id IN ('A', 'B')),
        status TEXT NOT NULL CHECK (status IN ('UNINITIALIZED', 'READY')),
        state_ref TEXT,
        state_sha256 TEXT,
        anchor_checkpoint_sha256 TEXT,
        PRIMARY KEY (session_id, client_id),
        CHECK (
            (status = 'UNINITIALIZED' AND state_ref IS NULL AND state_sha256 IS NULL
                AND anchor_checkpoint_sha256 IS NULL)
            OR (status = 'READY' AND state_ref IS NOT NULL AND state_sha256 IS NOT NULL
                AND anchor_checkpoint_sha256 IS NOT NULL)
        )
    ) STRICT""",
    """CREATE TABLE session_events (
        event_id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
        event_type TEXT NOT NULL,
        occurred_at TEXT NOT NULL
    ) STRICT""",
    """CREATE INDEX events_by_session ON session_events(session_id, event_id)""",
    """CREATE TRIGGER demo_sessions_one_root BEFORE INSERT ON demo_sessions
        WHEN EXISTS (
            SELECT 1 FROM demo_sessions WHERE v0_reference_sha256 != NEW.v0_reference_sha256
        )
        BEGIN SELECT RAISE(ABORT, 'Demo database is bound to a different research v0'); END""",
    """CREATE TRIGGER demo_sessions_close_only BEFORE UPDATE ON demo_sessions
        WHEN OLD.status != 'ACTIVE' OR NEW.status != 'CLOSED'
            OR NEW.session_id != OLD.session_id OR NEW.start_request_id IS NOT OLD.start_request_id
            OR NEW.created_at != OLD.created_at OR NEW.v0_reference_sha256 != OLD.v0_reference_sha256
            OR NEW.contract_id != OLD.contract_id OR NEW.contract_fingerprint != OLD.contract_fingerprint
        BEGIN SELECT RAISE(ABORT, 'Only an active demo can be closed'); END""",
)

_APPEND_ONLY = ("session_model_versions", "federated_updates", "session_events")
_SESSION_LOCAL = ("client_chunk_progress", "client_ewc_state")
_SESSION_SELECT = """
    SELECT s.*, (SELECT MAX(m.version) FROM session_model_versions m
                 WHERE m.session_id = s.session_id) AS head_version
    FROM demo_sessions s
"""


def _create_schema(db: sqlite3.Connection) -> None:
    for statement in _TABLES:
        db.execute(statement)
    for table in (*_APPEND_ONLY, *_SESSION_LOCAL):
        db.execute(f"""CREATE TRIGGER {table}_active_insert BEFORE INSERT ON {table}
            WHEN (SELECT status FROM demo_sessions WHERE session_id = NEW.session_id) IS NOT 'ACTIVE'
            BEGIN SELECT RAISE(ABORT, 'Demo session is not active'); END""")
    for table in _SESSION_LOCAL:
        db.execute(f"""CREATE TRIGGER {table}_active_update BEFORE UPDATE ON {table}
            WHEN NEW.session_id != OLD.session_id
                OR (SELECT status FROM demo_sessions WHERE session_id = OLD.session_id) IS NOT 'ACTIVE'
            BEGIN SELECT RAISE(ABORT, 'Demo session is closed or changed'); END""")
    for table in _APPEND_ONLY:
        db.execute(f"""CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'Demo history is append-only'); END""")
    for table in ("demo_sessions", *_APPEND_ONLY, *_SESSION_LOCAL):
        db.execute(f"""CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT, 'Demo records are retained'); END""")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _create_fedasync_schema(db: sqlite3.Connection) -> None:
    # Keep the exact serialized successor in SQLite so a committed file can be restored.
    db.execute("""CREATE TABLE fedasync_checkpoints (
        session_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        policy_id TEXT NOT NULL,
        alpha_base REAL NOT NULL CHECK (alpha_base > 0 AND alpha_base <= 1),
        body BLOB NOT NULL CHECK (length(body) > 0),
        PRIMARY KEY (session_id, version),
        FOREIGN KEY (session_id, version) REFERENCES session_model_versions(session_id, version)
    ) STRICT""")
    db.execute("""CREATE TRIGGER fedasync_checkpoints_active_insert BEFORE INSERT ON fedasync_checkpoints
        WHEN (SELECT status FROM demo_sessions WHERE session_id = NEW.session_id) IS NOT 'ACTIVE'
        BEGIN SELECT RAISE(ABORT, 'Demo session is not active'); END""")
    for operation in ("UPDATE", "DELETE"):
        db.execute(f"""CREATE TRIGGER fedasync_checkpoints_no_{operation.lower()}
            BEFORE {operation} ON fedasync_checkpoints
            BEGIN SELECT RAISE(ABORT, 'Checkpoint recovery bytes are immutable'); END""")


def _create_ewc_history_schema(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE ewc_reference_history (
        session_id TEXT NOT NULL,
        client_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        update_id TEXT NOT NULL,
        state_ref TEXT NOT NULL,
        state_sha256 TEXT NOT NULL,
        parent_sha256 TEXT,
        anchor_checkpoint_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, client_id, revision),
        UNIQUE (session_id, client_id, update_id),
        FOREIGN KEY (session_id, client_id) REFERENCES client_ewc_state(session_id, client_id),
        FOREIGN KEY (session_id, client_id, update_id) REFERENCES federated_updates(session_id, client_id, update_id)
    ) STRICT""")
    db.execute("""CREATE TRIGGER ewc_reference_history_active_insert BEFORE INSERT ON ewc_reference_history
        WHEN (SELECT status FROM demo_sessions WHERE session_id = NEW.session_id) IS NOT 'ACTIVE'
        BEGIN SELECT RAISE(ABORT, 'Demo session is not active'); END""")
    for operation in ("UPDATE", "DELETE"):
        db.execute(f"""CREATE TRIGGER ewc_reference_history_no_{operation.lower()}
            BEFORE {operation} ON ewc_reference_history
            BEGIN SELECT RAISE(ABORT, 'EWC reference history is immutable'); END""")


def _create_manual_run_schema(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE manual_chunk_runs (
        execution_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL, client_id TEXT NOT NULL, chunk_id TEXT NOT NULL, worker_store_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN (
            'SELECTED','APPROVED','RUNNING','STAGED','SUBMITTED','ACCEPTED_AWAITING_EWC',
            'ABANDONING','COMPLETED','FAILED')),
        approved_at TEXT, base_version INTEGER, base_checkpoint_sha256 TEXT,
        update_sha256 TEXT, used_ewc INTEGER CHECK (used_ewc IN (0,1)),
        accepted_version INTEGER, ewc_sha256 TEXT, failure_reason TEXT,
        FOREIGN KEY (session_id, client_id, chunk_id)
            REFERENCES client_chunk_progress(session_id, client_id, chunk_id),
        FOREIGN KEY (session_id, base_version) REFERENCES session_model_versions(session_id, version),
        FOREIGN KEY (session_id, accepted_version) REFERENCES session_model_versions(session_id, version)
    ) STRICT""")
    db.execute("""CREATE UNIQUE INDEX one_unfinished_run_per_client ON manual_chunk_runs(session_id, client_id)
                  WHERE state NOT IN ('COMPLETED','FAILED')""")
    db.execute("""CREATE TABLE manual_run_events (
        event_id INTEGER PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES manual_chunk_runs(execution_id),
        state TEXT NOT NULL, occurred_at TEXT NOT NULL
    ) STRICT""")
    db.execute("CREATE INDEX events_by_run ON manual_run_events(execution_id, event_id)")
    db.execute("""CREATE TRIGGER manual_run_active_insert BEFORE INSERT ON manual_chunk_runs
        WHEN NEW.state != 'SELECTED'
          OR (SELECT status FROM demo_sessions WHERE session_id=NEW.session_id) IS NOT 'ACTIVE'
        BEGIN SELECT RAISE(ABORT, 'Manual selection requires an active demo'); END""")
    db.execute("""CREATE TRIGGER manual_run_transition BEFORE UPDATE ON manual_chunk_runs
        WHEN (SELECT status FROM demo_sessions WHERE session_id=OLD.session_id) IS NOT 'ACTIVE'
          OR NEW.execution_id != OLD.execution_id OR NEW.session_id != OLD.session_id
          OR NEW.client_id != OLD.client_id OR NEW.chunk_id != OLD.chunk_id
          OR NEW.worker_store_id != OLD.worker_store_id
          OR NOT (
               (OLD.state='SELECTED' AND NEW.state IN ('APPROVED','ABANDONING'))
            OR (OLD.state='APPROVED' AND NEW.state IN ('RUNNING','ABANDONING'))
            OR (OLD.state='RUNNING' AND NEW.state IN ('STAGED','ABANDONING'))
            OR (OLD.state='STAGED' AND NEW.state IN ('SUBMITTED','ABANDONING'))
            OR (OLD.state='SUBMITTED' AND NEW.state IN ('ACCEPTED_AWAITING_EWC','ABANDONING'))
            OR (OLD.state='ACCEPTED_AWAITING_EWC' AND NEW.state='COMPLETED')
            OR (OLD.state='ABANDONING' AND NEW.state='FAILED'))
        BEGIN SELECT RAISE(ABORT, 'Invalid manual run transition'); END""")
    db.execute("""CREATE TRIGGER manual_run_no_delete BEFORE DELETE ON manual_chunk_runs
        BEGIN SELECT RAISE(ABORT, 'Manual runs are retained'); END""")
    for operation in ("UPDATE", "DELETE"):
        db.execute(f"""CREATE TRIGGER manual_run_events_no_{operation.lower()}
            BEFORE {operation} ON manual_run_events
            BEGIN SELECT RAISE(ABORT, 'Manual run events are append-only'); END""")


def _create_execution_mode_schema(db: sqlite3.Connection) -> None:
    db.execute("""ALTER TABLE demo_sessions ADD COLUMN execution_mode TEXT NOT NULL
                  DEFAULT 'LOCAL_SYNTHETIC' CHECK (execution_mode IN ('LOCAL_SYNTHETIC','REMOTE_SYNTHETIC'))""")
    db.execute("""ALTER TABLE demo_sessions ADD COLUMN worker_namespace TEXT
                  CHECK ((execution_mode='LOCAL_SYNTHETIC' AND worker_namespace IS NULL) OR
                  (execution_mode='REMOTE_SYNTHETIC' AND length(worker_namespace)=32
                   AND worker_namespace NOT GLOB '*[^0-9a-f]*' AND worker_namespace IS NOT NULL))""")
    db.execute("""CREATE TRIGGER demo_execution_mode_immutable BEFORE UPDATE ON demo_sessions
                  WHEN OLD.execution_mode != NEW.execution_mode OR OLD.worker_namespace IS NOT NEW.worker_namespace
                  BEGIN SELECT RAISE(ABORT, 'Demo execution binding is immutable'); END""")
    db.execute("ALTER TABLE manual_chunk_runs ADD COLUMN worker_training_seconds REAL")
    db.execute("ALTER TABLE manual_chunk_runs ADD COLUMN worker_steps INTEGER")
    db.execute("""CREATE TABLE dashboard_worker_health (
                  namespace TEXT NOT NULL, client_id TEXT NOT NULL CHECK (client_id IN ('A','B')),
                  last_verified_at TEXT NOT NULL, PRIMARY KEY (namespace,client_id)
                  ) STRICT""")


class DemoSessionManager:
    """Session lifecycle only. No client data, learning, checkpoint copies or privacy ledger access."""

    def __init__(self, database: Path = DEFAULT_DATABASE, *, v0: ResearchV0 = PINNED_RESEARCH_V0):
        self.database = database.resolve()
        self.v0 = ResearchV0(v0.reference_path.absolute(), v0.reference_sha256, v0.checkpoint_sha256)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect("rwc")) as db, db:
            db.execute("BEGIN IMMEDIATE")
            application_id = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if (application_id, version) == (_APPLICATION_ID, _SCHEMA_VERSION):
                return
            if application_id != _APPLICATION_ID or version not in (1, 2, 3, 4):
                if application_id or version or db.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone():
                    raise ValueError("Database is not an empty or supported demo session database")
                _create_schema(db)
                db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            if version < 2:
                _create_fedasync_schema(db)
            if version < 3:
                _create_ewc_history_schema(db)
            if version < 4:
                _create_manual_run_schema(db)
            _create_execution_mode_schema(db)
            db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _connect(self, mode: str = "rw") -> sqlite3.Connection:
        db = sqlite3.connect(f"{self.database.as_uri()}?mode={mode}", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA recursive_triggers = ON")
        if mode != "ro":
            db.execute("PRAGMA synchronous = FULL")
        return db

    @staticmethod
    def _session(db: sqlite3.Connection, session_id: str) -> Record:
        row = db.execute(_SESSION_SELECT + " WHERE s.session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise UnknownDemoError(session_id)
        return dict(row)

    def start_demo(self, *, request_id: str | None = None, execution_mode: str = LOCAL_SYNTHETIC,
                   worker_namespace: str | None = None) -> Record:
        """Start one demo. An optional stable request ID replays the original result."""
        if request_id is not None and (not isinstance(request_id, str)
                                       or not request_id or len(request_id) > 128
                                       or not request_id.isascii() or not request_id.isprintable()
                                       or request_id != request_id.strip()):
            raise ValueError("Expected a nonempty printable start request ID up to 128 characters")
        if not ((execution_mode == LOCAL_SYNTHETIC and worker_namespace is None)
                or (execution_mode == REMOTE_SYNTHETIC and isinstance(worker_namespace, str)
                    and re.fullmatch(r"[0-9a-f]{32}", worker_namespace))):
            raise ValueError("Invalid synthetic execution binding")
        checkpoint = self.v0.verify()
        contract = _SPEC.metadata()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("""SELECT session_id, v0_reference_sha256, execution_mode, worker_namespace FROM demo_sessions
                                     WHERE start_request_id = ?""", (request_id,)).fetchone()
            if existing is not None:
                if (existing["v0_reference_sha256"], existing["execution_mode"], existing["worker_namespace"]) != (
                    self.v0.reference_sha256, execution_mode, worker_namespace,
                ):
                    raise ValueError("Start request belongs to a different demo binding")
                return self._session(db, existing["session_id"])
            if db.execute("SELECT 1 FROM demo_sessions WHERE status = 'ACTIVE'").fetchone():
                raise ActiveDemoError("Close the active demo before starting another")
            session_id, now = uuid4().hex, _timestamp()
            db.execute(
                """INSERT INTO demo_sessions (session_id, start_request_id, status, created_at,
                   v0_reference_sha256, contract_id, contract_fingerprint, execution_mode, worker_namespace)
                   VALUES (?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?)""",
                (session_id, request_id, now, self.v0.reference_sha256, _SPEC.contract_id,
                 contract["contract_fingerprint"], execution_mode, worker_namespace),
            )
            db.execute(
                """INSERT INTO session_model_versions
                   (session_id, version, checkpoint_path, checkpoint_sha256, created_at)
                   VALUES (?, 0, ?, ?, ?)""",
                (session_id, str(checkpoint), self.v0.checkpoint_sha256, now),
            )
            db.executemany(
                """INSERT INTO client_chunk_progress
                   (session_id, client_id, chunk_number, chunk_id, status) VALUES (?, ?, ?, ?, 'PENDING')""",
                ((session_id, client, number, f"{prefix}{number}")
                 for client, prefix in _CLIENTS for number in range(1, _CHUNKS_PER_CLIENT + 1)),
            )
            db.executemany(
                """INSERT INTO client_ewc_state (session_id, client_id, status)
                   VALUES (?, ?, 'UNINITIALIZED')""",
                ((session_id, client) for client, _ in _CLIENTS),
            )
            db.execute("INSERT INTO session_events (session_id, event_type, occurred_at) VALUES (?, ?, ?)",
                       (session_id, "DEMO_STARTED", now))
            return self._session(db, session_id)

    def get_session(self, session_id: str) -> Record:
        with closing(self._connect("ro")) as db:
            return self._session(db, session_id)

    def list_sessions(self) -> list[Record]:
        with closing(self._connect("ro")) as db:
            return [dict(row) for row in db.execute(_SESSION_SELECT + " ORDER BY s.created_at, s.session_id")]

    def get_chunk_progress(self, session_id: str) -> list[Record]:
        with closing(self._connect("ro")) as db:
            self._session(db, session_id)
            rows = db.execute("""SELECT * FROM client_chunk_progress WHERE session_id = ?
                                 ORDER BY client_id, chunk_number""", (session_id,))
            return [dict(row) for row in rows]

    def get_model_history(self, session_id: str) -> list[Record]:
        with closing(self._connect("ro")) as db:
            self._session(db, session_id)
            rows = db.execute("SELECT * FROM session_model_versions WHERE session_id = ? ORDER BY version",
                              (session_id,))
            return [dict(row) for row in rows]

    def close_demo(self, session_id: str) -> Record:
        """Close the named session once, preserving its records and creation event."""
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            session = self._session(db, session_id)
            if session["status"] == "CLOSED":
                return session
            if db.execute("""SELECT 1 FROM manual_chunk_runs WHERE session_id=?
                             AND state NOT IN ('COMPLETED','FAILED')""", (session_id,)).fetchone():
                raise ActiveDemoError("Resolve unfinished manual chunk runs before closing the demo")
            now = _timestamp()
            db.execute("INSERT INTO session_events (session_id, event_type, occurred_at) VALUES (?, ?, ?)",
                       (session_id, "DEMO_CLOSED", now))
            db.execute("UPDATE demo_sessions SET status = 'CLOSED', closed_at = ? WHERE session_id = ?",
                       (now, session_id))
            return self._session(db, session_id)
