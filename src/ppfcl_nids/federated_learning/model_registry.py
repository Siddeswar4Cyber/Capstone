"""Isolated durable coordinator for explicitly released real continual updates."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS, owned_chunk
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC, model_template
from ppfcl_nids.federated_learning.asynchronous_aggregation import _load, _publish, _serialize, mix_full_states
from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0, validate_parameters

SHA = re.compile("[a-f0-9]{64}")
JOB = re.compile("[a-f0-9]{32}")


def old_task_status(directory: Path) -> dict:
    """Read the separate status store without opening private receipts."""
    path = directory.parent / "old-task-evaluation" / "status.sqlite"
    if not path.is_file():
        return {}
    from ppfcl_nids.evaluation.old_tasks import SAMPLE_COUNT, SELECTIONS, V0, V2
    allowed = {"PREFLIGHT_READY", "EVALUATION_STARTED", "COMPARISON_COMPLETED",
               "EVALUATION_FAILED", "EXECUTION_UNCONFIRMED"}
    snapshots = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for client, event, at, receipt in db.execute(
                "SELECT client,event,at,receipt_sha256 FROM status ORDER BY rowid LIMIT 12"):
            if client not in SELECTIONS or event not in allowed:
                continue
            verified = event == "COMPARISON_COMPLETED"
            if verified and (not isinstance(receipt, str) or not SHA.fullmatch(receipt)):
                continue
            snapshot = snapshots.setdefault(client, {"dataset": client, "selection_sha256": SELECTIONS[client],
                "rows": None, "v0": V0, "v2": V2, "receipt_verified": False, "events": []})
            if event == "PREFLIGHT_READY":
                snapshot["rows"] = SAMPLE_COUNT
            if verified:
                snapshot["receipt_verified"] = True
                snapshot["receipt_sha256"] = receipt
            snapshot["events"].append({"event": event, "at": at})
    return snapshots


def first_chunk(client):
    return ASSIGNMENTS[client][0] + "1"


class ContinualRegistry:
    def __init__(self, directory, v0):
        self.directory = directory
        self.v0 = v0
        self.db = directory / "registry.sqlite"
        with closing(self.connect()) as db, db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs(job TEXT PRIMARY KEY, client TEXT NOT NULL,
                chunk TEXT NOT NULL, authorization TEXT NOT NULL, UNIQUE(client,chunk));
            CREATE TABLE IF NOT EXISTS events(job TEXT NOT NULL REFERENCES jobs(job),
                seq INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(job,seq));
            CREATE TABLE IF NOT EXISTS updates(job TEXT PRIMARY KEY REFERENCES jobs(job),
                version INTEGER NOT NULL UNIQUE, sha TEXT NOT NULL, checkpoint BLOB NOT NULL,
                receipt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS export_events(job TEXT NOT NULL REFERENCES jobs(job),
                event TEXT NOT NULL, at TEXT NOT NULL, bytes INTEGER,
                PRIMARY KEY(job,event));
            CREATE TABLE IF NOT EXISTS received_updates(
                job TEXT PRIMARY KEY REFERENCES jobs(job), sha TEXT NOT NULL, body BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS model_returns(
                job TEXT PRIMARY KEY REFERENCES jobs(job), version INTEGER NOT NULL,
                sha TEXT NOT NULL, at TEXT NOT NULL, bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS local_confirmations(
                job TEXT PRIMARY KEY REFERENCES jobs(job), body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_returns(
                job TEXT NOT NULL REFERENCES jobs(job), version INTEGER NOT NULL, sha TEXT NOT NULL,
                at TEXT NOT NULL, bytes INTEGER NOT NULL, PRIMARY KEY(job,version));
            """)

    def connect(self):
        db = sqlite3.connect(self.db)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _export_event(db, job, event, size=None):
        db.execute("INSERT OR IGNORE INTO export_events VALUES (?,?,?,?)",
                   (job, event, datetime.now(timezone.utc).isoformat(), size))

    def export_event(self, job, event, size=None):
        with closing(self.connect()) as db, db:
            self._export_event(db, job, event, size)

    def check_export(self, client, job, pin):
        """Check the recorded stage and approved sequence before reading guest bytes."""
        with closing(self.connect()) as db, db:
            row = db.execute("SELECT client FROM jobs WHERE job=?", (job,)).fetchone()
            event = db.execute("SELECT body FROM events WHERE job=? ORDER BY seq DESC LIMIT 1", (job,)).fetchone()
            if row != (client,) or event is None or (staged := json.loads(event[0]))["state"] != "STAGED" or staged["sha256"] != pin:
                raise PermissionError("No matching staged update for authenticated client")
            prior = db.execute("SELECT sha,checkpoint,receipt FROM updates WHERE job=?", (job,)).fetchone()
            if prior is not None:
                if prior[0] != pin:
                    raise ValueError("Conflicting export replay")
                return self._recover_receipt(prior)
            if db.execute("SELECT 1 FROM received_updates WHERE job=?", (job,)).fetchone():
                raise ValueError("Update already received")

    def receive(self, client, job, artifact, pin):
        if hashlib.sha256(artifact).hexdigest() != pin or len(artifact) > 512 * 1024:
            raise ValueError("Received update digest mismatch")
        self._validate_update(artifact)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT client FROM jobs WHERE job=?", (job,)).fetchone()
            if owner != (client,) or db.execute("SELECT 1 FROM updates WHERE job=?", (job,)).fetchone():
                raise PermissionError("Update ownership or acceptance state changed")
            stage = db.execute("SELECT body FROM events WHERE job=? ORDER BY seq DESC LIMIT 1", (job,)).fetchone()
            if not stage or (record := json.loads(stage[0])).get("state") != "STAGED" or record.get("sha256") != pin:
                raise PermissionError("Exact staged update required")
            db.execute("INSERT INTO received_updates VALUES (?,?,?)", (job, pin, artifact))
            self._export_event(db, job, "UPDATE_EXPORTED", len(artifact))
            self._export_event(db, job, "UPDATE_RECEIVED", len(artifact))

    def aggregate_received(self, client, job, approval):
        with closing(self.connect()) as db:
            row = db.execute("SELECT sha,body FROM received_updates JOIN jobs USING(job) WHERE job=? AND client=?",
                             (job, client)).fetchone()
        if not row or row[0] != approval.get("sha256"):
            raise PermissionError("Exact received update required")
        return self.accept(client, job, row[1], approval, separately_authorized=True)

    def start_export(self, client, job):
        """Claim the export once before reading model bytes over SSH."""
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT client FROM jobs WHERE job=?", (job,)).fetchone() != (client,):
                raise PermissionError("Export client mismatch")
            result = db.execute("""INSERT OR IGNORE INTO export_events VALUES
                                (?, 'UPDATE_EXPORT_STARTED', ?, NULL)""",
                                (job, datetime.now(timezone.utc).isoformat()))
            if result.rowcount != 1:
                raise ValueError("Export already started; reconcile durable receipt before re-export")

    def record_model_return(self, client, job, version, digest, size):
        if type(version) is not int or version < 1 or version == 2 or not SHA.fullmatch(digest) or not 0 < size <= 512 * 1024:
            raise ValueError("Invalid returned model record")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT client FROM jobs JOIN updates USING(job) WHERE job=?", (job,)).fetchone() != (client,):
                raise PermissionError("Return requires an accepted job owned by the recipient")
            prior = db.execute("SELECT version,sha,bytes FROM model_returns WHERE job=?", (job,)).fetchone()
            if prior and prior != (version, digest, size):
                raise ValueError("Conflicting model return")
            model = db.execute("SELECT receipt,LENGTH(checkpoint) FROM updates WHERE version=?", (version,)).fetchone()
            if not model or (json.loads(model[0])["checkpoint_sha256"], model[1]) != (digest, size):
                raise ValueError("Returned model does not match a committed checkpoint")
            db.execute("INSERT OR IGNORE INTO model_returns VALUES (?,?,?,?,?)",
                       (job, version, digest, datetime.now(timezone.utc).isoformat(), size))

    def _recover_receipt(self, row):
        receipt = json.loads(row[2])
        if hashlib.sha256(row[1]).hexdigest() != receipt["checkpoint_sha256"]:
            raise ValueError("Corrupt committed checkpoint")
        _publish(self.directory / (receipt["checkpoint_sha256"] + ".pt"), row[1])
        return receipt

    @staticmethod
    def _check_order(db, client, extended=False):
        accepted = db.execute("SELECT jobs.client,jobs.chunk FROM updates JOIN jobs USING(job) ORDER BY version LIMIT 2").fetchall()
        if extended and accepted == [("A", "C1"), ("B", "D1")]:
            return
        if not extended and accepted == ([] if client == "A" else [("A", "C1")]):
            return
        raise PermissionError("Real FedAsync sequence requires C1 followed by D1")

    @staticmethod
    def dashboard_snapshot(directory: Path):
        path = directory / "registry.sqlite"
        if not path.is_file():
            return []
        retention = old_task_status(directory)
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            jobs = db.execute("""SELECT job,client,chunk FROM jobs
                              WHERE (client='A' AND chunk='C1') OR (client='B' AND chunk='D1')
                              ORDER BY rowid DESC""").fetchall()
            has_exports = db.execute("SELECT 1 FROM sqlite_master WHERE name='export_events'").fetchone()
            has_evaluation = db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluation_events'").fetchone()
            return [{"job": job, "client": client, "chunk": chunk,
                     "old_task_evaluation": retention.get(client),
                     "events": [json.loads(body) for body, in db.execute(
                         "SELECT body FROM events WHERE job=? ORDER BY seq LIMIT 34", (job,))],
                     "export_events": [{"event": event, "at": at, "bytes": size} for event, at, size in db.execute(
                         "SELECT event,at,bytes FROM export_events WHERE job=? ORDER BY rowid", (job,))] if has_exports else [],
                     "evaluation_events": [{"event": event, "at": at, "v0": v0, "v2": v2}
                                           for event, at, v0, v2 in db.execute(
                         "SELECT event,at,v0,v2 FROM evaluation_events WHERE job=? ORDER BY seq", (job,))] if has_evaluation else [],
                     "receipt": (json.loads(row[0]) if (row := db.execute(
                         "SELECT receipt FROM updates WHERE job=?", (job,)).fetchone()) else None)}
                    for job, client, chunk in jobs]

    def require_job(self, job, client, chunk, authorization):
        with closing(self.connect()) as db:
            if db.execute("SELECT client,chunk,authorization FROM jobs WHERE job=?", (job,)).fetchone() != (
                    client, chunk, authorization):
                raise PermissionError("Job does not match the registered authorization")

    def register(self, job, client, authorization, chunk=None, *, confirmation=None):
        if not JOB.fullmatch(job) or not SHA.fullmatch(authorization):
            raise ValueError("Invalid job or authorization")
        chunk = chunk or first_chunk(client)
        if not owned_chunk(client, chunk):
            raise ValueError("Invalid client chunk")
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)",
                       (job, client, chunk, authorization))
            if db.execute("SELECT client,chunk,authorization FROM jobs WHERE job=?", (job,)).fetchone() != (
                    client, chunk, authorization):
                raise ValueError("Conflicting job registration")
            if confirmation is not None:
                body = json.dumps(confirmation, sort_keys=True)
                if hashlib.sha256(body.encode()).hexdigest() != authorization:
                    raise ValueError("Confirmation digest mismatch")
                db.execute("INSERT INTO local_confirmations VALUES (?,?)", (job, body))

    def local_confirmation(self, client, job):
        with closing(self.connect()) as db:
            row = db.execute("""SELECT body,authorization FROM local_confirmations JOIN jobs USING(job)
                                WHERE job=? AND client=?""", (job, client)).fetchone()
        if row and hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("Stored local confirmation digest mismatch")
        return json.loads(row[0]) if row else None

    def record_research_return(self, client, job, version, digest, size):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT client FROM jobs WHERE job=?", (job,)).fetchone() != (client,):
                raise PermissionError("Research inventory recipient mismatch")
            if version == 0:
                model_pin = self.v0.checkpoint_sha256
                model_size = self.v0.verify().stat().st_size
            else:
                row = db.execute("SELECT receipt,LENGTH(checkpoint) FROM updates WHERE version=?", (version,)).fetchone()
                if not row:
                    raise ValueError("Research checkpoint is absent")
                model_pin, model_size = json.loads(row[0])["checkpoint_sha256"], row[1]
            if (digest, size) != (model_pin, model_size):
                raise ValueError("Research inventory receipt does not match registry")
            db.execute("INSERT OR IGNORE INTO research_returns VALUES (?,?,?,?,?)",
                       (job, version, digest, datetime.now(timezone.utc).isoformat(), size))

    def record(self, client, event):
        fields = {"sequence", "at", "job", "client", "chunk", "state"}
        state = event.get("state")
        chunk = event.get("chunk")
        extended = owned_chunk(client, chunk) and chunk != first_chunk(client)
        extra = {"RUNNING": set(), "STEP": {"completed", "total"}, "STAGED": {"sha256", "steps"},
                 "FAILED": set()}
        if extended:
            extra.update(DATASET_VERIFIED=set(), PREFLIGHT_VERIFIED=set(), RESERVED=set(),
                         STAGED={"sha256", "steps", "evaluation_sha256"}, FAILED={"reason"})
        if (not isinstance(state, str) or state not in extra or set(event) != fields | extra[state]
                or event["client"] != client or not owned_chunk(client, chunk)
                or not isinstance(event["job"], str) or not JOB.fullmatch(event["job"])
                or type(event["sequence"]) is not int or not 1 <= event["sequence"] <= (37 if extended else 34)
                or type(event["at"]) is not int or event["at"] <= 0):
            raise ValueError("Invalid event schema")
        if state == "STEP" and (type(event["completed"]) is not int or type(event["total"]) is not int
                                or not 1 <= event["completed"] <= 32 or event["total"] != 32):
            raise ValueError("Invalid step")
        if state == "STAGED" and (not isinstance(event["sha256"], str) or not SHA.fullmatch(event["sha256"])
                                  or type(event["steps"]) is not int or event["steps"] != 32
                                  or extended and not SHA.fullmatch(event["evaluation_sha256"])):
            raise ValueError("Invalid checkpoint event")
        if extended and state == "FAILED" and event["reason"] != "Worker verification or execution failed":
            raise ValueError("Invalid failure reason")
        body = json.dumps(event, sort_keys=True, allow_nan=False)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            job = event["job"]
            if db.execute("SELECT client,chunk FROM jobs WHERE job=?", (job,)).fetchone() != (client, chunk):
                raise PermissionError("Job does not belong to authenticated client")
            previous = db.execute("SELECT body FROM events WHERE job=? AND seq=?", (job, event["sequence"])).fetchone()
            if previous:
                if previous[0] != body:
                    raise ValueError("Conflicting event replay")
                return
            prior = db.execute("SELECT seq,body FROM events WHERE job=? ORDER BY seq DESC LIMIT 1", (job,)).fetchone()
            if extended:
                last = json.loads(prior[1]) if prior else {}
                previous_state = last.get("state")
                successors = {None: "DATASET_VERIFIED", "DATASET_VERIFIED": "PREFLIGHT_VERIFIED",
                              "PREFLIGHT_VERIFIED": "RUNNING", "RUNNING": "RESERVED", "RESERVED": "STEP",
                              "STEP": "STAGED" if last.get("completed") == 32 else "STEP"}
                valid = (event["sequence"] == (prior[0] + 1 if prior else 1)
                         and event["at"] >= last.get("at", 0) and previous_state in successors
                         and (state == "FAILED" or state == successors[previous_state])
                         and (state != "STEP" or event["completed"] == last.get("completed", 0) + 1))
            elif prior:
                last = json.loads(prior[1])
                valid = (event["sequence"] == prior[0] + 1 and event["at"] >= last["at"]
                         and last["state"] in ("RUNNING", "STEP")
                         and (state == "FAILED" or state == "STEP" and event["completed"] == prior[0]
                              or state == "STAGED" and prior[0] == 33))
            else:
                valid = event["sequence"] == 1 and state == "RUNNING"
            if not valid:
                raise ValueError("Out-of-order job event")
            db.execute("INSERT INTO events VALUES (?,?,?)", (job, event["sequence"], body))

    def accept(self, client, job, artifact, approval, *, separately_authorized=False):
        """Caller verifies the operator's release grant and pinned SSH source."""
        pin = hashlib.sha256(artifact).hexdigest()
        expected = {"operation": "submit_continual_update", "client": client, "job": job,
                    "sha256": pin, "base_sha256": self.v0.checkpoint_sha256,
                    "alpha_policy": "inverse-staleness-0.5-v1"}
        if separately_authorized:
            expected = {**expected, "operation": "aggregate_continual_update",
                        "head_version": approval.get("head_version"), "head_sha256": approval.get("head_sha256")}
        if approval != expected:
            raise PermissionError("Exact model update release approval required")
        update, template = self._validate_update(artifact)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute("SELECT client,chunk FROM jobs WHERE job=?", (job,)).fetchone()
            if not owner or owner[0] != client:
                raise PermissionError("Update source mismatch")
            prior = db.execute("SELECT sha,checkpoint,receipt FROM updates WHERE job=?", (job,)).fetchone()
            if prior:
                if prior[0] != pin:
                    raise ValueError("Conflicting update replay")
                return self._recover_receipt(prior)
            extended = owner[1] != first_chunk(client)
            if extended and not separately_authorized:
                raise PermissionError("Separate aggregation approval required")
            self._check_order(db, client, extended)
            event = db.execute("SELECT body FROM events WHERE job=? ORDER BY seq DESC LIMIT 1", (job,)).fetchone()
            if not event or (latest := json.loads(event[0]))["state"] != "STAGED" or latest["sha256"] != pin:
                raise ValueError("No matching durable staged update")
            latest_checkpoint = db.execute(
                "SELECT version,checkpoint,receipt FROM updates ORDER BY version DESC LIMIT 1").fetchone()
            if latest_checkpoint:
                staleness, body, metadata = latest_checkpoint
                parent_sha = json.loads(metadata)["checkpoint_sha256"]
                if hashlib.sha256(body).hexdigest() != parent_sha:
                    raise ValueError("Corrupt committed head")
                head = _load(body)["state_dict"]
            else:
                staleness = 0
                parent_sha = self.v0.checkpoint_sha256
                head = decode_verified_v0(self.v0.verify().read_bytes(), self.v0.checkpoint_sha256,
                                          model_spec=EWC_SPEC)["state_dict"]
            if separately_authorized and (staleness, parent_sha) != (approval["head_version"], approval["head_sha256"]):
                raise PermissionError("Approved coordinator head changed")
            alpha = 0.5 / (1 + staleness)
            version = staleness + 1
            body = _serialize({"schema": "continual-local-research-v1", "version": version,
                               "model_metadata": EWC_SPEC.metadata(),
                               "state_dict": mix_full_states(head, update, alpha, template)})
            digest = hashlib.sha256(body).hexdigest()
            receipt = {**expected, "operation": "submit_continual_update", "chunk": owner[1], "version": version, "base_version": 0,
                       "parent_version": version - 1, "parent_checkpoint_sha256": parent_sha,
                       "accepted_at": datetime.now(timezone.utc).isoformat(),
                       "staleness": staleness, "alpha": alpha, "checkpoint_sha256": digest,
                       "designation": "unvalidated_aggregation_checkpoint"}
            _publish(self.directory / (digest + ".pt"), body)
            db.execute("INSERT INTO updates VALUES (?,?,?,?,?)", (job, version, pin, body, json.dumps(receipt)))
            for event in ("UPDATE_VALIDATED", "UPDATE_ACCEPTED", "FEDASYNC_AGGREGATED", "RESEARCH_VERSION_CREATED"):
                self._export_event(db, job, event)
            return receipt

    @staticmethod
    def _validate_update(artifact):
        payload = _load(artifact)
        if set(payload) != {"metadata", "state_dict"} or payload["metadata"] != EWC_SPEC.metadata():
            raise ValueError("Invalid continual update contract")
        template = model_template()
        update = payload["state_dict"]
        validate_parameters(update, template)
        return update, template
