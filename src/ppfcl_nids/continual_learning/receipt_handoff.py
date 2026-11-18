"""Explicit synthetic receipt and reference handoff. The coordinator never loads EWC tensors."""

import re
import sqlite3
from contextlib import closing

from ppfcl_nids.federated_learning.sessions import ActiveDemoError, DemoSessionManager, Record, _timestamp
from ppfcl_nids.continual_learning.importance_store import CommittedEWCReference, SyntheticEWCAck
from ppfcl_nids.federated_learning.asynchronous_aggregation import SyntheticUpdatePermit, _RECEIPT


class SyntheticEWCHandoff:
    def __init__(self, sessions: DemoSessionManager):
        self.sessions = sessions

    @staticmethod
    def _ack(db: sqlite3.Connection, permit: SyntheticUpdatePermit) -> SyntheticEWCAck | None:
        row = db.execute(_RECEIPT, (permit.session_id, permit.client_id, permit.update_id)).fetchone()
        if row is None:
            return None
        if (row["run_id"], row["chunk_id"], row["base_version"], row["base_checkpoint_sha256"],
            row["payload_sha256"]) != (
            permit.run_id, permit.chunk_id, permit.base_version, permit.base_checkpoint_sha256,
            permit.artifact_sha256,
        ):
            raise ValueError("Synthetic update scope differs from the committed receipt")
        return SyntheticEWCAck(permit, row["accepted_version"], row["checkpoint_sha256"])

    def get_acknowledgment(self, permit: SyntheticUpdatePermit) -> SyntheticEWCAck | None:
        """Read the durable receipt. A missing receipt alone does not prove rejection."""
        with closing(self.sessions._connect("ro")) as db:
            return self._ack(db, permit)

    def publish_reference(self, committed: CommittedEWCReference) -> Record:
        """Record an opaque reference after client-local promotion. Safe to replay after a lost ACK."""
        if type(committed) is not CommittedEWCReference:
            raise ValueError("Client-local EWC promotion is required before publication")
        ref = committed.reference
        permit = ref.permit
        if (not re.fullmatch(r"[0-9a-f]{64}", ref.sha256)
                or type(ref.revision) is not int or ref.revision < 1
                or committed.acknowledgment.permit != permit):
            raise ValueError("Malformed or mismatched committed EWC reference")
        key = (permit.session_id, permit.client_id)
        with closing(self.sessions._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if self._ack(db, permit) != committed.acknowledgment:
                raise ValueError("EWC publication lacks the matching durable update receipt")
            session = self.sessions._session(db, permit.session_id)
            prior = db.execute("""SELECT * FROM ewc_reference_history
                                  WHERE session_id=? AND client_id=? AND revision=?""",
                               (*key, ref.revision)).fetchone()
            scope = (permit.update_id, ref.opaque_ref, ref.sha256, ref.parent_sha256, permit.artifact_sha256)
            if prior is not None:
                if tuple(prior[k] for k in (
                    "update_id", "state_ref", "state_sha256", "parent_sha256", "anchor_checkpoint_sha256",
                )) != scope:
                    raise ValueError("Conflicting EWC reference replay")
                return dict(prior)
            if session["status"] != "ACTIVE":
                raise ActiveDemoError("New EWC references require an active demo")
            current = db.execute("SELECT state_sha256 FROM client_ewc_state WHERE session_id=? AND client_id=?",
                                 key).fetchone()
            latest = db.execute("""SELECT MAX(revision) FROM ewc_reference_history
                                   WHERE session_id=? AND client_id=?""", key).fetchone()[0]
            if current is None or current["state_sha256"] != ref.parent_sha256 or ref.revision != (latest or 0) + 1:
                raise ValueError("EWC reference does not extend the published client history")
            now = _timestamp()
            db.execute("INSERT INTO ewc_reference_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (*key, ref.revision, *scope, now))
            db.execute("""UPDATE client_ewc_state SET status='READY', state_ref=?, state_sha256=?,
                          anchor_checkpoint_sha256=? WHERE session_id=? AND client_id=?""",
                       (ref.opaque_ref, ref.sha256, permit.artifact_sha256, *key))
            db.execute("INSERT INTO session_events (session_id, event_type, occurred_at) VALUES (?, ?, ?)",
                       (permit.session_id, "SYNTHETIC_EWC_REFERENCE_PUBLISHED", now))
            return dict(db.execute("""SELECT * FROM ewc_reference_history
                                      WHERE session_id=? AND client_id=? AND revision=?""",
                                   (*key, ref.revision)).fetchone())

    def get_reference_history(self, session_id: str, client_id: str) -> list[Record]:
        with closing(self.sessions._connect("ro")) as db:
            self.sessions._session(db, session_id)
            return [dict(row) for row in db.execute("""SELECT * FROM ewc_reference_history
                    WHERE session_id=? AND client_id=? ORDER BY revision""", (session_id, client_id))]
