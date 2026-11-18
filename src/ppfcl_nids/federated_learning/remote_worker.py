"""One bounded synthetic worker RPC on a trusted, authenticated SSH channel."""

import base64
import binascii
import json
import os
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, time

import torch

if sys.platform != "win32":
    from resource import RUSAGE_SELF, getrusage

from ppfcl_nids.federated_learning.sessions import PINNED_RESEARCH_V0
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC, EWCConfig
from ppfcl_nids.continual_learning.importance_store import LocalEWCStore, SyntheticEWCAck
from ppfcl_nids.federated_learning.asynchronous_aggregation import SyntheticUpdatePermit, decode_synthetic_update, decode_training_checkpoint
from ppfcl_nids.continual_learning.chunk_workflow import InvalidRunState, SYNTHETIC_TRAINING_STEPS, train_synthetic_and_stage
from ppfcl_nids.federated_learning.synthetic_round import frame, read_frame
from ppfcl_nids.virtual_machines.worker_health import ADDRESSES, resources as _resources, worker_database as _worker_database

PROTOCOL = "ppfcl-synthetic-worker-v1"
APPROVAL_SECONDS = 900
ARTIFACT_LIMIT = 256 * 1024
OPS = {
    "prepare": {"base", "expires_at", "parent_sha256"}, "execute": set(), "status": set(),
    "update": set(), "pending": set(), "reconcile": {"ack"}, "recover": set(), "discard": set(),
}
_HEX = re.compile(r"[0-9a-f]{32}")
_SHA = re.compile(r"[0-9a-f]{64}")
_RUN = re.compile(r"[A-Za-z0-9_-]{1,128}")


def encode_artifact(body: bytes) -> str:
    if not 0 < len(body) <= ARTIFACT_LIMIT:
        raise ValueError("Worker artifact exceeds its size limit")
    return base64.b64encode(body).decode("ascii")


def decode_artifact(value: str) -> bytes:
    if not isinstance(value, str) or not 0 < len(value) <= 4 * ((ARTIFACT_LIMIT + 2) // 3):
        raise ValueError("Invalid worker artifact length")
    try:
        body = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid worker artifact encoding") from None
    if not 0 < len(body) <= ARTIFACT_LIMIT:
        raise ValueError("Invalid worker artifact size")
    return body


def parse_permit(value: dict, client_id: str, *, staged: bool = False) -> SyntheticUpdatePermit:
    if not isinstance(value, dict) or value.keys() != SyntheticUpdatePermit.__dataclass_fields__.keys():
        raise ValueError("Invalid worker run scope")
    permit = SyntheticUpdatePermit(**value)
    prefix = {"A": "C", "B": "D"}.get(client_id)
    if (prefix is None or permit.client_id != client_id
            or permit.chunk_id not in {f"{prefix}{number}" for number in range(1, 21)}
            or not isinstance(permit.session_id, str) or not _HEX.fullmatch(permit.session_id)
            or not isinstance(permit.run_id, str) or not _RUN.fullmatch(permit.run_id)
            or permit.update_id != permit.run_id or type(permit.base_version) is not int
            or not 0 <= permit.base_version <= 39
            or not isinstance(permit.base_checkpoint_sha256, str) or not _SHA.fullmatch(permit.base_checkpoint_sha256)
            or (permit.base_version == 0 and permit.base_checkpoint_sha256 != PINNED_RESEARCH_V0.checkpoint_sha256)
            or (staged and (not isinstance(permit.artifact_sha256, str) or not _SHA.fullmatch(permit.artifact_sha256)))
            or (not staged and permit.artifact_sha256 != "")):
        raise ValueError("Worker run client, chunk, session, base or artifact scope is invalid")
    return permit


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


class SyntheticWorker:
    """Client-local command dispatcher. Construct only for an SSH-bound client identity."""

    def __init__(self, store: LocalEWCStore):
        self.store = store
        with store.execution_transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS remote_worker_runs (
                run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, scope TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('PREPARED','RUNNING','STAGED','FAILED')),
                expires_at INTEGER, parent_sha256 TEXT, base BLOB, measures TEXT
            ) STRICT""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_unstaged_remote_run
                ON remote_worker_runs(session_id) WHERE state IN ('PREPARED','RUNNING')""")

    def _row(self, db, permit):
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM remote_worker_runs WHERE run_id=?", (permit.run_id,)).fetchone()
        if row is not None and row["scope"] != _canonical(asdict(permit)):
            raise PermissionError("Conflicting worker run scope")
        return row

    def _lineage(self, permit, parent):
        current, pending = self.store.restore(permit.session_id)
        if pending is not None or (current.sha256 if current else None) != parent:
            raise InvalidRunState("Client EWC lineage differs from the approved coordinator reference")

    def _prepare(self, permit, params):
        expiry = params["expires_at"]
        parent = params["parent_sha256"]
        if (type(expiry) is not int or not time() < expiry <= time() + APPROVAL_SECONDS + 5
                or (parent is not None and (not isinstance(parent, str) or not _SHA.fullmatch(parent)))):
            raise PermissionError("Invalid or expired synthetic approval")
        base = decode_artifact(params["base"])
        decode_training_checkpoint(permit, base, self.store._template)
        with self.store.execution_transaction() as db:
            prior = self._row(db, permit)
            if prior is not None:
                if prior["state"] == "FAILED" or (prior["expires_at"], prior["parent_sha256"], prior["base"]) != (
                    expiry, parent, base,
                ):
                    raise PermissionError("Conflicting or failed worker approval")
                return {"state": prior["state"]}
            self._lineage(permit, parent)
            db.execute("""INSERT INTO remote_worker_runs
                       (run_id,session_id,scope,state,expires_at,parent_sha256,base)
                       VALUES (?,?,?,'PREPARED',?,?,?)""",
                       (permit.run_id, permit.session_id, _canonical(asdict(permit)), expiry, parent, base))
        return {"state": "PREPARED"}

    def _execute(self, permit):
        with self.store.execution_transaction() as db:
            row = self._row(db, permit)
            if row is None:
                raise PermissionError("No prepared worker approval")
            if row["state"] != "PREPARED":
                return {"state": row["state"]}
            if time() >= row["expires_at"]:
                raise PermissionError("Worker approval expired before execution")
            db.execute("UPDATE remote_worker_runs SET state='RUNNING' WHERE run_id=?", (permit.run_id,))
        with self.store.execution_transaction() as db:
            row = self._row(db, permit)
            if row["state"] != "RUNNING":
                return {"state": row["state"]}
            if time() >= row["expires_at"]:
                raise PermissionError("Worker approval expired before training")
            self._lineage(permit, row["parent_sha256"])
            weights = decode_training_checkpoint(permit, row["base"], self.store._template)
            start = perf_counter()
            train_synthetic_and_stage(self.store, db, permit, weights, EWCConfig())
            measures = {"training_and_fisher_seconds": perf_counter() - start,
                        "training_steps": SYNTHETIC_TRAINING_STEPS,
                        "peak_rss_kib": None if sys.platform == "win32" else getrusage(RUSAGE_SELF).ru_maxrss}
            db.execute("UPDATE remote_worker_runs SET state='STAGED', measures=? WHERE run_id=?",
                       (_canonical(measures), permit.run_id))
        return {"state": "STAGED", "measures": measures}

    def _recover(self, permit, *, discard: bool):
        with self.store.execution_transaction() as db:
            row = self._row(db, permit)
            if row is None:
                # A tombstone also fences a delayed Prepare or Run after a network failure.
                db.execute("""INSERT INTO remote_worker_runs (run_id,session_id,scope,state)
                              VALUES (?,?,?,'FAILED')""",
                           (permit.run_id, permit.session_id, _canonical(asdict(permit))))
            elif row["state"] == "STAGED" and not discard:
                return {"state": "STAGED"}
            else:
                self.store._discard_if_pending(db, permit.session_id, permit.run_id)
                db.execute("UPDATE remote_worker_runs SET state='FAILED' WHERE run_id=?", (permit.run_id,))
        return {"state": "FAILED"}

    def call(self, operation: str, request: dict) -> dict:
        fields = OPS.get(operation)
        if fields is None or request.keys() != fields | {"permit"}:
            raise ValueError("Invalid or unsupported worker operation")
        permit = parse_permit(request["permit"], self.store.client_id)
        if operation == "prepare":
            return self._prepare(permit, request)
        if operation == "execute":
            return self._execute(permit)
        if operation in ("recover", "discard"):
            return self._recover(permit, discard=operation == "discard")
        with closing(self.store._connect("ro")) as db, db:
            db.execute("BEGIN")
            row = self._row(db, permit)
            state = row["state"] if row is not None else "MISSING"
            if operation == "status":
                return {"state": state, "measures": json.loads(row["measures"]) if row and row["measures"] else None}
            if state != "STAGED":
                raise InvalidRunState("No staged worker result")
            staged = self.store._prepared_update(db, permit.session_id, permit.run_id)
            if staged is None:
                raise InvalidRunState("Durable worker result is missing")
            reference, artifact = staged
            actual = self.store._find(db, "session_id=? AND run_id=?", (permit.session_id, permit.run_id))
            decode_synthetic_update(reference.permit, artifact, self.store._template)
        if operation == "pending":
            return {"reference": asdict(reference), "promotion_state": actual["status"]}
        if operation == "update":
            return {"reference": asdict(reference), "artifact": encode_artifact(artifact)}
        ack = request["ack"]
        if not isinstance(ack, dict) or ack.keys() != {"permit", "accepted_version", "checkpoint_sha256"}:
            raise ValueError("Invalid coordinator acknowledgment")
        accepted = parse_permit(ack["permit"], self.store.client_id, staged=True)
        if accepted != reference.permit:
            raise PermissionError("Coordinator acknowledgment differs from worker outbox")
        started = perf_counter()
        committed = self.store.commit(SyntheticEWCAck(accepted, ack["accepted_version"], ack["checkpoint_sha256"]))
        return {"reference": asdict(committed.reference), "promotion_state": "COMMITTED",
                "promotion_seconds": perf_counter() - started}


def _check_resources(result: dict) -> None:
    if (not result["cpu_only"] or result["free_disk_bytes"] < 3 * 1024**3
            or result["available_ram_bytes"] < 1024**3):
        raise RuntimeError("CPU runtime and available disk and RAM reserves required")


def main():
    os.umask(0o077)
    client_id, namespace = sys.argv[1:]
    route = os.environ.get("SSH_CONNECTION", "").split()
    if (client_id not in ADDRESSES or not _HEX.fullmatch(namespace) or len(route) != 4
            or route[0] != "192.168.252.1" or route[2] != ADDRESSES[client_id]):
        raise SystemExit(2)
    torch.set_num_threads(1)
    request = json.loads(read_frame(sys.stdin.buffer))
    if (not isinstance(request, dict)
            or request.keys() != {"protocol", "operation", "worker_id", "contract", "params"}
            or request["protocol"] != PROTOCOL or request["contract"] != EWC_SPEC.metadata()
            or not isinstance(request["operation"], str) or not isinstance(request["params"], dict)):
        raise SystemExit(2)
    operation = request["operation"]
    database = _worker_database(namespace, client_id)
    worker_id = None
    try:
        if operation in ("health", "initialize"):
            if request["params"] or request["worker_id"] is not None:
                raise ValueError("Unexpected health or initialization parameters")
            result = {**_resources(database.parent if database.parent.is_dir() else Path.home()),
                      "address": route[2]}
            if operation == "initialize":
                _check_resources(result)
                worker_id = LocalEWCStore(database, client_id=client_id).worker_id
            elif database.is_file():
                with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
                    worker_id = db.execute("SELECT worker_id FROM manual_worker_identity WHERE id=1").fetchone()[0]
        elif operation in OPS and database.is_file():
            with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
                worker_id = db.execute("SELECT worker_id FROM manual_worker_identity WHERE id=1").fetchone()[0]
            if request["worker_id"] != worker_id:
                raise PermissionError("Worker store identity mismatch")
            if operation in ("prepare", "execute"):
                _check_resources(_resources(database.parent))
            result = SyntheticWorker(LocalEWCStore(database, client_id=client_id)).call(operation, request["params"])
        else:
            raise ValueError("Unknown operation or unprovisioned worker")
        response = {"result": result}
    except (ValueError, RuntimeError, OSError, TypeError, sqlite3.Error, LookupError) as exc:
        response = {"error": type(exc).__name__}
    sys.stdout.buffer.write(frame({"protocol": PROTOCOL, "operation": operation, "client_id": client_id,
                                  "namespace": namespace, "worker_id": worker_id,
                                  "contract": EWC_SPEC.metadata(), **response}))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
