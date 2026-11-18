"""Explicit, recoverable manual runs for the local synthetic demonstration."""

import hashlib
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import replace

import torch

from ppfcl_nids.federated_learning.sessions import ActiveDemoError, DemoSessionManager, Record, _timestamp
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC, EWCConfig, synthetic_task_loss
from ppfcl_nids.continual_learning.receipt_handoff import SyntheticEWCHandoff
from ppfcl_nids.continual_learning.importance_store import EWCReference, LocalEWCStore
from ppfcl_nids.federated_learning.asynchronous_aggregation import SyntheticFedAsync, SyntheticUpdatePermit, _RECEIPT, encode_synthetic_update
from ppfcl_nids.models.neural_network import FlowMLP

_TERMINAL = ("COMPLETED", "FAILED")
_EXECUTION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
SYNTHETIC_TRAINING_STEPS = 2


class InvalidRunState(ValueError):
    """The requested action is incompatible with the durable run state."""


def train_synthetic_and_stage(store: LocalEWCStore, local: sqlite3.Connection,
                              permit: SyntheticUpdatePermit, weights: Mapping[str, torch.Tensor],
                              config: EWCConfig, progress=None) -> EWCReference:
    """Shared bounded CPU trainer. The caller owns the client staging transaction."""
    previous = store.load_for_task(permit)
    with torch.device("meta"):
        model = FlowMLP(EWC_SPEC)
    model.load_state_dict(weights, assign=True)
    seed = int.from_bytes(hashlib.sha256(permit.chunk_id.encode()).digest()[:4], "little")
    features = torch.randn(16, 11, generator=torch.Generator().manual_seed(seed))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).float()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(1, SYNTHETIC_TRAINING_STEPS + 1):
        optimizer.zero_grad()
        loss = synthetic_task_loss(model, features, labels, previous)
        loss.backward()
        optimizer.step()
        if progress is not None:
            progress(step, SYNTHETIC_TRAINING_STEPS, step * len(labels),
                     SYNTHETIC_TRAINING_STEPS * len(labels), float(loss.detach()))
    artifact = encode_synthetic_update(permit, model.state_dict())
    permit = replace(permit, artifact_sha256=hashlib.sha256(artifact).hexdigest())
    return store._prepare(local, permit, artifact, features, labels, config=config, retain_update=True)


class SyntheticManualChunks:
    """In-process service bound to a trusted synthetic client store, with no network endpoint."""

    def __init__(self, sessions: DemoSessionManager, store: LocalEWCStore, *, ewc: EWCConfig = EWCConfig()):
        self.sessions = sessions
        self.store = store
        self.ewc = ewc
        self.handoff = SyntheticEWCHandoff(sessions)
        self.registry = SyntheticFedAsync(sessions, trusted_updates=())

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def _session(self, db: sqlite3.Connection, session_id: str, *, active: bool = False) -> Record:
        session = self.sessions._session(db, session_id)
        if active and session["status"] != "ACTIVE":
            raise ActiveDemoError("Manual changes require an active demo")
        if (session["contract_id"], session["contract_fingerprint"], session["v0_reference_sha256"]) != (
            EWC_SPEC.contract_id, EWC_SPEC.metadata()["contract_fingerprint"], self.sessions.v0.reference_sha256,
        ):
            raise ValueError("Manual run session contract or v0 reference mismatch")
        return session

    def _run(self, db: sqlite3.Connection, session_id: str, execution_id: str) -> sqlite3.Row:
        self._session(db, session_id)
        row = db.execute("""SELECT * FROM manual_chunk_runs
                            WHERE execution_id=? AND session_id=? AND client_id=?""",
                         (execution_id, session_id, self.store.client_id)).fetchone()
        if row is None:
            raise LookupError("No manual run in this client and session scope")
        if row["worker_store_id"] != self.store.worker_id:
            raise PermissionError("Manual run belongs to a different client store")
        return row

    @staticmethod
    def _event(db: sqlite3.Connection, execution_id: str, state: str) -> None:
        db.execute("INSERT INTO manual_run_events (execution_id,state,occurred_at) VALUES (?,?,?)",
                   (execution_id, state, _timestamp()))

    def _transition(self, db: sqlite3.Connection, row: sqlite3.Row, state: str, **fields) -> Record:
        updates = {"state": state, **fields}
        assignments = ", ".join(f"{field}=?" for field in updates)
        db.execute(f"UPDATE manual_chunk_runs SET {assignments} WHERE execution_id=?",
                   (*updates.values(), row["execution_id"]))
        self._event(db, row["execution_id"], state)
        return {**dict(row), **updates}

    def list_chunks(self, session_id: str) -> list[Record]:
        """List owned chunks with their latest manual state, in display order only."""
        with closing(self.sessions._connect("ro")) as db:
            self._session(db, session_id)
            return [dict(row) for row in db.execute("""
                SELECT c.chunk_id, c.chunk_number, COALESCE(r.state,c.status) AS state, r.execution_id
                FROM client_chunk_progress c
                LEFT JOIN manual_chunk_runs r ON r.rowid=(
                    SELECT MAX(attempt.rowid) FROM manual_chunk_runs attempt WHERE
                    attempt.session_id=c.session_id AND attempt.client_id=c.client_id AND attempt.chunk_id=c.chunk_id
                )
                WHERE c.session_id=? AND c.client_id=? ORDER BY c.chunk_number
            """, (session_id, self.store.client_id))]

    def select_chunk(self, session_id: str, chunk_id: str, execution_id: str) -> Record:
        """Select an owned chunk manually. Failed chunks require a new execution ID and approval."""
        if not isinstance(execution_id, str) or not _EXECUTION_ID.fullmatch(execution_id):
            raise ValueError("Invalid manual execution ID")
        with self._write() as db:
            self._session(db, session_id)
            prior = db.execute("SELECT * FROM manual_chunk_runs WHERE execution_id=?", (execution_id,)).fetchone()
            if prior is not None:
                if (prior["session_id"], prior["client_id"], prior["chunk_id"], prior["worker_store_id"]) != (
                    session_id, self.store.client_id, chunk_id, self.store.worker_id,
                ):
                    raise ValueError("Execution ID already has a different scope")
                return dict(prior)
            self._session(db, session_id, active=True)
            chunk = db.execute("""SELECT status FROM client_chunk_progress
                                  WHERE session_id=? AND client_id=? AND chunk_id=?""",
                               (session_id, self.store.client_id, chunk_id)).fetchone()
            if chunk is None:
                raise PermissionError("Chunk is outside this client's session scope")
            if chunk["status"] not in ("PENDING", "FAILED"):
                raise InvalidRunState("Chunk is not available for manual selection")
            if db.execute("""SELECT 1 FROM manual_chunk_runs WHERE session_id=? AND client_id=?
                             AND state NOT IN ('COMPLETED','FAILED')""", (session_id, self.store.client_id)).fetchone():
                raise InvalidRunState("Resolve the client's unfinished run before selecting another chunk")
            db.execute("""INSERT INTO manual_chunk_runs
                          (execution_id,session_id,client_id,chunk_id,worker_store_id,state)
                          VALUES (?,?,?,?,?,'SELECTED')""",
                       (execution_id, session_id, self.store.client_id, chunk_id, self.store.worker_id))
            self._event(db, execution_id, "SELECTED")
            return dict(self._run(db, session_id, execution_id))

    def approve_run(self, session_id: str, execution_id: str) -> Record:
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["approved_at"] is not None:
                return dict(row)
            if row["state"] != "SELECTED":
                raise InvalidRunState("Approval requires a selected run")
            return self._transition(db, row, "APPROVED", approved_at=_timestamp())

    def get_run(self, session_id: str, execution_id: str) -> Record:
        with closing(self.sessions._connect("ro")) as db:
            return dict(self._run(db, session_id, execution_id))

    def get_run_events(self, session_id: str, execution_id: str) -> list[Record]:
        with closing(self.sessions._connect("ro")) as db:
            self._run(db, session_id, execution_id)
            return [dict(row) for row in db.execute(
                "SELECT * FROM manual_run_events WHERE execution_id=? AND state NOT LIKE 'SYNTHETIC_STEP:%' "
                "ORDER BY event_id", (execution_id,))]

    @staticmethod
    def _permit(row: Record | sqlite3.Row, digest: str = "") -> SyntheticUpdatePermit:
        return SyntheticUpdatePermit(
            row["session_id"], row["client_id"], row["execution_id"], row["execution_id"], row["chunk_id"],
            row["base_version"], row["base_checkpoint_sha256"], digest,
        )

    def _receipt(self, db: sqlite3.Connection, row: Record | sqlite3.Row) -> Record | None:
        if row["update_sha256"] is None:
            return None
        permit = self._permit(row, row["update_sha256"])
        if self.handoff._ack(db, permit) is None:
            return None
        return dict(db.execute(_RECEIPT, (permit.session_id, permit.client_id, permit.update_id)).fetchone())

    def get_receipt(self, session_id: str, execution_id: str) -> Record | None:
        with closing(self.sessions._connect("ro")) as db:
            return self._receipt(db, self._run(db, session_id, execution_id))

    def _train_and_stage(self, local: sqlite3.Connection, row: Record) -> EWCReference:
        permit = self._permit(row)
        weights = self.registry.load_training_base(permit.session_id, permit.base_version,
                                                   permit.base_checkpoint_sha256)
        def progress(step, total, samples, workload, loss):
            with self._write() as db:
                self._event(db, permit.run_id, f"SYNTHETIC_STEP:{step}:{total}:{samples}:{workload}:{loss}")
        return train_synthetic_and_stage(self.store, local, permit, weights, self.ewc, progress)

    def _record_staged(self, reference: EWCReference, *, training_seconds: float | None = None,
                       training_steps: int | None = None) -> Record:
        permit = reference.permit
        with self._write() as db:
            row = self._run(db, permit.session_id, permit.run_id)
            if row["state"] != "RUNNING":
                return dict(row)
            if permit != self._permit(row, permit.artifact_sha256):
                raise ValueError("Staged update differs from the captured run base")
            parent = db.execute("""SELECT state_sha256 FROM client_ewc_state
                                   WHERE session_id=? AND client_id=?""",
                                (permit.session_id, permit.client_id)).fetchone()
            if parent is None or parent["state_sha256"] != reference.parent_sha256:
                raise ValueError("Staged EWC revision does not extend the session's client reference")
            return self._transition(db, row, "STAGED", update_sha256=permit.artifact_sha256,
                                    used_ewc=int(reference.parent_sha256 is not None),
                                    ewc_sha256=reference.sha256, worker_training_seconds=training_seconds,
                                    worker_steps=training_steps)

    def _claim_run(self, db, row, *, check_local: bool) -> tuple[Record, str | None]:
        session_id = row["session_id"]
        session = self._session(db, session_id, active=True)
        pointer = db.execute("""SELECT state_sha256 FROM client_ewc_state
                                WHERE session_id=? AND client_id=?""",
                             (session_id, self.store.client_id)).fetchone()
        if pointer is None:
            raise InvalidRunState("Missing coordinator EWC reference")
        parent = pointer["state_sha256"]
        if check_local:
            current, pending = self.store.restore(session_id)
            if pending is not None or parent != (current.sha256 if current else None):
                raise InvalidRunState("Client EWC state must match the coordinator reference before Run")
        head = self.registry._version(db, session_id, session["head_version"])
        running = self._transition(db, row, "RUNNING", base_version=head["version"],
                                   base_checkpoint_sha256=head["checkpoint_sha256"])
        db.execute("""UPDATE client_chunk_progress SET status='RUNNING'
                      WHERE session_id=? AND client_id=? AND chunk_id=?""",
                   (session_id, self.store.client_id, row["chunk_id"]))
        return running, parent

    def start_run(self, session_id: str, execution_id: str) -> Record:
        """Explicit Run action. Claim once, train on synthetic tensors, and stage durably."""
        with self.store.execution_transaction() as local:
            with self._write() as db:
                row = self._run(db, session_id, execution_id)
                if row["state"] == "SELECTED":
                    raise InvalidRunState("Explicit approval is required before Run")
                running = None
                if row["state"] == "APPROVED":
                    running, _ = self._claim_run(db, row, check_local=True)
            if running is None:
                if row["state"] == "RUNNING":
                    staged = self.store._prepared_update(local, session_id, execution_id)
                    if staged is not None:
                        return self._record_staged(staged[0])
                return dict(row)
            reference = self._train_and_stage(local, running)
        return self._record_staged(reference)

    def _record_acceptance(self, session_id: str, execution_id: str) -> Record:
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            receipt = self._receipt(db, row)
            if receipt is None:
                raise InvalidRunState("No matching durable FedAsync receipt")
            if row["state"] == "SUBMITTED":
                self._transition(db, row, "ACCEPTED_AWAITING_EWC", accepted_version=receipt["accepted_version"])
            elif row["state"] not in ("ACCEPTED_AWAITING_EWC", "COMPLETED"):
                raise InvalidRunState("Run cannot reconcile this receipt")
            return receipt

    def submit_run(self, session_id: str, execution_id: str) -> Record:
        """Submit or replay the exact local outbox bytes, with no training retry."""
        with self.store.execution_transaction() as local:
            with self._write() as db:
                row = self._run(db, session_id, execution_id)
                if row["state"] not in ("STAGED", "SUBMITTED", "ACCEPTED_AWAITING_EWC", "COMPLETED"):
                    raise InvalidRunState("Submit requires a durably staged run")
                if row["state"] == "STAGED":
                    self._transition(db, row, "SUBMITTED")
                existing = self._receipt(db, row)
            if existing is None:
                staged = self.store._prepared_update(local, session_id, execution_id)
                if staged is None:
                    raise InvalidRunState("Manual update is missing or discarded")
                reference, artifact = staged
                permit = self._permit(row, row["update_sha256"])
                if reference.permit != permit:
                    raise ValueError("Outbox scope differs from the approved execution")
                SyntheticFedAsync(self.sessions, trusted_updates=[permit]).accept_async_update(permit, artifact)
            return self._record_acceptance(session_id, execution_id)

    def complete_run(self, session_id: str, execution_id: str) -> Record:
        """Reconcile a durable receipt, local promotion, and coordinator reference."""
        self._record_acceptance(session_id, execution_id)
        row = self.get_run(session_id, execution_id)
        if row["state"] == "COMPLETED":
            return row
        permit = self._permit(row, row["update_sha256"])
        ack = self.handoff.get_acknowledgment(permit)
        if ack is None:
            raise InvalidRunState("Missing durable EWC acknowledgment")
        committed = self.store.commit(ack)
        return self._finish_completion(session_id, execution_id, committed)

    def _finish_completion(self, session_id, execution_id, committed) -> Record:
        ack = committed.acknowledgment
        row = self.get_run(session_id, execution_id)
        if row["ewc_sha256"] not in (None, committed.reference.sha256):
            raise InvalidRunState("Promotion differs from the staged EWC digest")
        published = self.handoff.publish_reference(committed)
        with self._write() as db:
            current = self._run(db, session_id, execution_id)
            if current["state"] == "COMPLETED":
                return dict(current)
            pointer = db.execute("""SELECT state_sha256 FROM client_ewc_state
                                    WHERE session_id=? AND client_id=?""",
                                 (session_id, self.store.client_id)).fetchone()
            if (pointer["state_sha256"], published["state_sha256"], current["accepted_version"]) != (
                committed.reference.sha256, committed.reference.sha256, ack.accepted_version,
            ):
                raise InvalidRunState("Current EWC reference or accepted version differs")
            db.execute("""UPDATE client_chunk_progress SET status='COMPLETED'
                          WHERE session_id=? AND client_id=? AND chunk_id=?""",
                       (session_id, self.store.client_id, current["chunk_id"]))
            return self._transition(db, current, "COMPLETED", ewc_sha256=committed.reference.sha256)

    def _abandon(self, session_id: str, execution_id: str, *, interrupted_training: bool) -> Record:
        with self.store.execution_transaction() as local:
            with self._write() as db:
                row = self._run(db, session_id, execution_id)
                if row["state"] == "FAILED":
                    return dict(row)
                if row["state"] in ("COMPLETED", "ACCEPTED_AWAITING_EWC") or self._receipt(db, row) is not None:
                    raise InvalidRunState("A committed FedAsync update must be reconciled")
                if row["state"] != "ABANDONING":
                    reason = "interrupted_training" if interrupted_training else "explicit_abandonment"
                    self._transition(db, row, "ABANDONING", failure_reason=reason)
            self.store._discard_if_pending(local, session_id, execution_id)
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["state"] == "FAILED":
                return dict(row)
            db.execute("""UPDATE client_chunk_progress SET status='FAILED'
                          WHERE session_id=? AND client_id=? AND chunk_id=?""",
                       (session_id, self.store.client_id, row["chunk_id"]))
            return self._transition(db, row, "FAILED")

    def resolve_run(self, session_id: str, execution_id: str) -> Record:
        """Explicitly abandon a run only after excluding a durable receipt under the client lock."""
        return self._abandon(session_id, execution_id, interrupted_training=False)

    def recover_run(self, session_id: str, execution_id: str) -> Record:
        """Explicit restart action. Recover durable work without re-running training."""
        with self.store.execution_transaction() as local:
            row = self.get_run(session_id, execution_id)
            if row["state"] == "RUNNING":
                staged = self.store._prepared_update(local, session_id, execution_id)
                if staged is not None:
                    return self._record_staged(staged[0])
        if row["state"] in ("RUNNING", "ABANDONING"):
            return self._abandon(session_id, execution_id, interrupted_training=row["state"] == "RUNNING")
        if row["state"] == "SUBMITTED":
            self.registry.recover_checkpoints()
            self.submit_run(session_id, execution_id)
            return self.complete_run(session_id, execution_id)
        if row["state"] == "ACCEPTED_AWAITING_EWC":
            return self.complete_run(session_id, execution_id)
        return row
