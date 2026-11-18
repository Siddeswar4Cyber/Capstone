"""Explicit approval and dispatch boundary for bounded continual work."""

from contextlib import closing
from datetime import timedelta
import hashlib
from math import isfinite
import json
from pathlib import Path
import sqlite3
from threading import Lock, Thread
from uuid import uuid4

from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS
from ppfcl_nids.federated_learning.model_registry import ContinualRegistry, JOB, SHA
from ppfcl_nids.dashboard.client_views import ACTIONS, _readonly, _schedule, chunk_snapshot, directory, dispatch_events, jobs_snapshot, project_job, utc
from ppfcl_nids.organization.disclosure import local_json, owner_storage
from ppfcl_nids.ssh.connections import SSHConnectionError

RESEARCH_V2 = "ccedc3c7c21bacee029c6b657781ab914b0cbba4e6070a53ce4a108686b1fff6"
PREFLIGHT_BLOCKERS = {
    "CPU_RUNTIME_REQUIRED": "The guest requires a CPU-only PyTorch runtime.",
    "INSUFFICIENT_RAM": "Available guest RAM is below the worker minimum.",
    "INSUFFICIENT_DISK": "Available guest disk is below the worker reserve.",
    "CLIENT_INPUT_PERMISSIONS": "The dataset, manifest or ledger is not an owner-only regular file.",
    "CLIENT_INPUT_UNAVAILABLE": "An assigned guest input or resource counter is unavailable.",
    "FROZEN_SELECTION_VERIFICATION_FAILED": "The guest source, manifest or exact frozen slot selection failed verification.",
    "PRIVACY_LEDGER_VERIFICATION_FAILED": "The durable guest ledger could not verify its schema, cap or delta.",
    "MODEL_VERIFICATION_FAILED": "Guest Global v0 failed checkpoint or feature-contract verification.",
}


class ClientActionError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ClientRuns:
    def __init__(self, project: Path, state_dir: Path, v0, monitor):
        self.project = project
        # Real attempts share one fence across dashboard sessions and ports.
        self.database = project / "state/continual-real/client-dispatch.sqlite"
        self.legacy_database = state_dir / "client-dispatch.sqlite"
        self.v0 = v0
        self.monitor = monitor
        self.lock = Lock()
        self.running: dict[str, str] = {}
        self.preparations: dict[str, dict] = {}

    def snapshot(self, client: str, *, local=False) -> dict:
        with self.lock:
            active = self.running.get(client)
            prep = self.preparations.get(client)
            if prep:
                prep = {key: value for key, value in prep.items() if key != "deadline"}
                if prep["expires_at"] and prep["expires_at"] < utc().isoformat() and prep["status"] == "APPROVAL REQUIRED":
                    prep.update(status="Expired", message="Run preflight again for a fresh proposal.")
        jobs, available = jobs_snapshot(self.project, client, self.v0, self.database, active)
        claimed = None
        if self.database.is_file():
            with closing(_readonly(self.database)) as db:
                claimed = db.execute("SELECT job,chunk FROM attempts WHERE client=?", (client,)).fetchone()
            if claimed and not any(job["id"] == claimed[0] for job in jobs):
                job, chunk = claimed
                events = dispatch_events(self.database, client, job)
                jobs.insert(0, project_job(client, job, chunk, None, [], [], events, None, job == active))
        result = chunk_snapshot(self.project, client, jobs, available)
        if active or claimed:
            for row in result["chunks"]:
                if row["eligible"]:
                    row.update(eligible=False, reason="A client operation is in progress or needs reconciliation.")
        head = None
        registry_path = directory(self.project) / "registry.sqlite"
        if registry_path.is_file():
            with closing(_readonly(registry_path)) as db:
                current = db.execute("SELECT version,receipt FROM updates ORDER BY version DESC LIMIT 1").fetchone()
            head = {"version": current[0], "sha256": json.loads(current[1])["checkpoint_sha256"]} if current else None
        logs = [event for job in reversed(jobs) for event in job["events"]][-120:]
        models = []
        if local:
            from ppfcl_nids.dashboard.server_views import registry_snapshot
            # Only the small confirmed model catalogue is shared between clients.
            models = [{k: model[k] for k in ("version", "sha256", "contract")}
                      for model in registry_snapshot(self.project, self.v0)["models"]
                      if model["integrity"] == "Verified" and
                      (model["version"] != 2 or model["sha256"] == RESEARCH_V2)]
        return {"client": client, **result, "jobs": jobs, "logs": logs, "active_job": active,
                "preflight": prep, "head": head, "research_models": models,
                "system_logs": "System logs unavailable · no authorized client service log source"}

    def _assert_eligible(self, client: str, chunk: str) -> None:
        if self.legacy_database.is_file():
            with closing(_readonly(self.legacy_database)) as db:
                if db.execute("SELECT 1 FROM attempts WHERE client=?", (client,)).fetchone():
                    raise ClientActionError(409, "An earlier dashboard dispatch is unresolved.")
        if self.database.is_file():
            with closing(_readonly(self.database)) as db:
                if db.execute("SELECT 1 FROM attempts WHERE client=?", (client,)).fetchone():
                    raise ClientActionError(409, "A dispatch was already claimed. Reconcile its outcome before any action.")
        registered = directory(self.project) / "registry.sqlite"
        if registered.is_file():
            with closing(_readonly(registered)) as db:
                if db.execute("SELECT 1 FROM jobs WHERE client=? AND chunk=?", (client, chunk)).fetchone():
                    raise ClientActionError(409, "A durable attempt already exists for this client. Retraining is blocked.")
        row = next((row for row in chunk_snapshot(self.project, client, [], True)["chunks"] if row["id"] == chunk), None)
        if not row or not row["eligible"]:
            raise ClientActionError(409, row["reason"] if row else "Chunk does not belong to this client.")

    def preflight(self, client: str, chunk: str, *, local=False) -> dict:
        self._assert_eligible(client, chunk)
        if not self.monitor.ssh_enabled:
            raise ClientActionError(409, "Pinned client monitoring is disabled. Start with --monitor-ssh.")
        with self.lock:
            if client in self.running or self.preparations.get(client, {}).get("status") == "Checking":
                raise ClientActionError(409, "A client operation is in progress.")
            pending = {"job": uuid4().hex, "chunk": chunk, "status": "Checking",
                       "message": "Verifying source, checkpoint, resources and privacy projection over pinned SSH.",
                       "expires_at": None}
            self.preparations[client] = pending
            try:
                Thread(target=self._prepare, args=(client, pending["job"], chunk), kwargs={"local": local}, daemon=True).start()
            except RuntimeError:
                self.preparations.pop(client, None)
                raise
        return {"status": "Checking", "job": pending["job"]}

    def _prepare(self, client, job, chunk, *, local=False) -> None:
        try:
            from ppfcl_nids.federated_learning.communication import probe
            from ppfcl_nids.continual_learning.worker import DISK_RESERVE, MAX_SECONDS, RAM_RESERVE, scope

            self.v0.verify()
            planned = scope(client, chunk)
            if planned["base_sha256"] != self.v0.checkpoint_sha256:
                raise ValueError("Unapproved base")
            data = probe(self.project, client, dashboard=True, chunk=chunk)
            if "blocker" in data:
                raise ClientActionError(409, PREFLIGHT_BLOCKERS.get(data["blocker"], "Worker preflight failed."))
            _, _, _, source, frozen = ASSIGNMENTS[client]
            if ((data["source_sha256"], data["manifest_sha256"]) != (source, frozen)
                    or not SHA.fullmatch(data["slot_sha256"])):
                raise ValueError("Contract or selection check failed")
            budget = data["budget"]
            schedule = _schedule(planned)
            if not all(type(budget[key]) in (int, float) and isfinite(budget[key]) and budget[key] >= 0
                       for key in ("current_epsilon", "projected_epsilon")):
                raise ValueError("Untrustworthy privacy projection")
            manifest_view = chunk_snapshot(self.project, client, [], True)
            chunk_row = next(row for row in manifest_view["chunks"] if row["id"] == chunk)
            start, stop = data["training_range"]
            val_start, val_stop = data["validation_range"]
            if (data["chunk_rows"] != chunk_row["rows"] or start != chunk_row["start"]
                    or stop != start + schedule["sample_count"] or stop > chunk_row["stop"]
                    or val_start < manifest_view["chunks"][-1]["stop"] or val_stop != val_start + 2048):
                raise ValueError("Frozen rank selection mismatch")
            blockers = []
            if budget["within_cap"] is not True:
                blockers.append("Projected cumulative epsilon exceeds the client cap.")
            if planned.get("allow_base_transfer") is False and data["local_model_verified"] is not True:
                blockers.append("The client needs a separately authorized, verified Global v0 copy.")
            deadline = utc() + timedelta(minutes=15)
            result = {"status": "Blocked" if blockers else "APPROVAL REQUIRED", "scope": planned, "schedule": schedule,
                      "base_sha256": self.v0.checkpoint_sha256, "contract": planned["contract"]["contract_id"],
                      "selection": planned["selection"], "slot_sha256": data["slot_sha256"],
                      "local_model_verified": data["local_model_verified"] is True,
                      "attempt_limit": 1, "privacy": {"current": budget["current_epsilon"],
                          "projected": budget["projected_epsilon"], "delta": schedule["delta"],
                          "headroom_after_run": schedule["epsilon_limit"] - budget["projected_epsilon"]},
                      "full_rows": data["chunk_rows"], "training_range": data["training_range"],
                      "validation_range": data["validation_range"], "ram_min": RAM_RESERVE,
                      "disk_min": DISK_RESERVE, "ram_available": data["available_ram"],
                      "disk_available": data["free_disk"], "maximum_seconds": MAX_SECONDS,
                      "deadline": int(deadline.timestamp()),
                      "expires_at": deadline.isoformat(),
                      "message": " ".join(blockers) if blockers else
                      "Review these parameters before starting the training attempt." if local else
                      "Owner grant and explicit dispatch confirmation required."}
        except ClientActionError as exc:
            result = {"status": "Blocked", "expires_at": None, "message": str(exc)}
        except SSHConnectionError as exc:
            result = {"status": "Blocked", "expires_at": None, "message": f"Pinned SSH connection failed · {exc.reason}"}
        except TimeoutError:
            result = {"status": "Blocked", "expires_at": None, "message": "Pinned worker preflight timed out."}
        except Exception:
            result = {"status": "Blocked", "expires_at": None,
                      "message": "Pinned SSH, source, model, resource or privacy preflight failed."}
        with self.lock:
            self.preparations[client] = {"job": job, "chunk": chunk, **result}

    def _ensure_dispatch_tables(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (client TEXT PRIMARY KEY, job TEXT UNIQUE, chunk TEXT NOT NULL, at TEXT);
                CREATE TABLE IF NOT EXISTS stages (
                    seq INTEGER PRIMARY KEY, client TEXT NOT NULL, job TEXT NOT NULL,
                    at TEXT NOT NULL, state TEXT NOT NULL, bytes INTEGER);
            """)
    def _claim(self, client, job, chunk) -> None:
        self._ensure_dispatch_tables()
        with closing(sqlite3.connect(self.database)) as db, db:
            try:
                db.execute("INSERT INTO attempts VALUES (?,?,?,?)", (client, job, chunk, utc().isoformat()))
            except sqlite3.IntegrityError:
                raise ClientActionError(409, "Duplicate or conflicting dispatch.") from None

    def _stage(self, client, job, state, size=None) -> None:
        if state not in ACTIONS:
            raise ValueError("Unexpected dispatch stage")
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("INSERT INTO stages(client,job,at,state,bytes) VALUES (?,?,?,?,?)",
                       (client, job, utc().isoformat(), state, size))

    def approve(self, client: str, chunk: str, job: str, *, local=False) -> dict:
        if not JOB.fullmatch(job):
            raise ClientActionError(400, "Invalid attempt identifier.")
        self._assert_eligible(client, chunk)
        with self.lock:
            prepared = self.preparations.get(client)
            if (not prepared or client in self.running or (prepared["job"], prepared["chunk"], prepared["status"])
                    != (job, chunk, "APPROVAL REQUIRED") or utc().timestamp() >= prepared["deadline"]):
                raise ClientActionError(409, "Obtain a current matching preflight before confirming training.")
            registry = directory(self.project)
            approvals = registry.parent / "approvals"
            expected = {"operation": "local_continual_training", "job": job, "scope": prepared["scope"],
                        "slot_sha256": prepared["slot_sha256"]}
            if prepared["scope"].get("protocol"):
                expected["privacy_projection"] = prepared["privacy"]
            if local:
                grant = {**expected, "expires": prepared["deadline"]}
            else:
                try:
                    owner_storage(approvals)
                    owner_storage(registry)
                    grant = local_json(approvals / f"{job}.train.json")
                except (OSError, ValueError):
                    raise ClientActionError(403, "APPROVAL REQUIRED · owner-only grant missing or unavailable.") from None
            expires = grant.get("expires") if isinstance(grant, dict) else None
            if (type(expires) is not int or not utc().timestamp() < expires <= prepared["deadline"]
                    or grant != {**expected, "expires": expires}):
                raise ClientActionError(403, "APPROVAL REQUIRED · owner grant differs from the exact proposal.")
            self._claim(client, job, chunk)
            try:
                bound = hashlib.sha256(json.dumps(grant, sort_keys=True).encode()).hexdigest()
                ContinualRegistry(registry, self.v0).register(job, client, bound, chunk, confirmation=grant if local else None)
            except (OSError, sqlite3.Error, ValueError):
                self._stage(client, job, "INTERRUPTED")
                raise ClientActionError(409, "Attempt claimed but registration could not be verified. Reconcile before continuing.") from None
            self.running[client] = job
            self.preparations.pop(client, None)
            try:
                Thread(target=self._run, args=(client, job, grant), daemon=True).start()
            except RuntimeError:
                self.running.pop(client, None)
                self._stage(client, job, "INTERRUPTED")
                raise ClientActionError(503, "Dispatch interrupted. Reconcile the claimed attempt.") from None
        return {"status": "Dispatch claimed", "job": job}

    def _run(self, client, job, grant) -> None:
        from ppfcl_nids.federated_learning.communication import worker_action

        try:
            registry = ContinualRegistry(directory(self.project), self.v0)
            if not self.monitor.ssh_enabled:
                raise PermissionError("Pinned SSH dispatch unavailable")
            worker_action(self.project, registry, client, job, "run", expected_grant=grant,
                          on_stage=lambda state, size: self._stage(client, job, state, size))
            self._stage(client, job, "DISPATCH_FINISHED")
            if any(row["id"] == job and row["status"] in ("Completed", "Failed")
                   for row in self.snapshot_without_claim(client)):
                self._release_claim(client, job)
            else:
                self._stage(client, job, "INTERRUPTED")
        except Exception:
            self._stage(client, job, "INTERRUPTED")
        finally:
            with self.lock:
                self.running.pop(client, None)

    def _release_claim(self, client, job) -> None:
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("DELETE FROM attempts WHERE client=? AND job=?", (client, job))

    def reconcile(self, client, chunk, job):
        """Read durable guest evidence. Never start or resume execution."""
        from ppfcl_nids.federated_learning.communication import worker_action

        with self.lock:
            if client in self.running or not self.database.is_file():
                raise ClientActionError(409, "No interrupted dispatch available for this client.")
            with closing(_readonly(self.database)) as db:
                claim = db.execute("SELECT job,chunk FROM attempts WHERE client=?", (client,)).fetchone()
            if claim != (job, chunk):
                raise ClientActionError(409, "No matching interrupted dispatch.")
            self.running[client] = job

        def recover():
            try:
                registry = ContinualRegistry(directory(self.project), self.v0)
                with closing(registry.connect()) as db:
                    registered = db.execute("SELECT 1 FROM jobs WHERE job=? AND client=? AND chunk=?",
                                            (job, client, chunk)).fetchone()
                if not registered:
                    raise ValueError("Dispatch registration is unconfirmed")
                worker_action(self.project, registry, client, job, "sync")
                rows = self.snapshot_without_claim(client)
                if any(row["id"] == job and row["status"] in ("Completed", "Failed") for row in rows):
                    self._release_claim(client, job)
            except Exception:
                self._stage(client, job, "INTERRUPTED")
            finally:
                with self.lock:
                    self.running.pop(client, None)

        try:
            Thread(target=recover, daemon=True).start()
        except RuntimeError:
            with self.lock:
                self.running.pop(client, None)
            raise
        return {"status": "Reconciling durable worker evidence", "job": job}

    def snapshot_without_claim(self, client):
        return jobs_snapshot(self.project, client, self.v0, self.database)[0]

    def _return_research(self, client, chunk, job, version, digest):
        from ppfcl_nids.federated_learning.communication import return_checkpoint
        from ppfcl_nids.dashboard.server_views import registry_snapshot

        row = next((row for row in self.snapshot_without_claim(client) if (row["id"], row["chunk"]) == (job, chunk)), None)
        if not row or not self.monitor.ssh_enabled:
            raise ClientActionError(409, "An existing job for this client and pinned SSH are required.")
        if type(version) is not int or version < 0 or not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise ClientActionError(400, "Select an exact verified model version and SHA-256.")
        if any(item["version"] == version for item in row["research_returns"]):
            raise ClientActionError(409, "This research checkpoint already has a verified receipt.")
        if version == 2 and digest != RESEARCH_V2:
            raise ClientActionError(409, "Research v2 does not match the preserved pin.")
        if not any((model["version"], model["sha256"], model["integrity"]) == (version, digest, "Verified")
                   for model in registry_snapshot(self.project, self.v0)["models"]):
            raise ClientActionError(409, "The selected checkpoint failed registry verification.")
        registry_dir = directory(self.project)
        checkpoint = self.v0.verify() if version == 0 else registry_dir / f"{digest}.pt"
        with self.lock:
            if client in self.running:
                raise ClientActionError(409, "A client operation is already active.")
            self.running[client] = job

        def deliver():
            try:
                self._ensure_dispatch_tables()
                self._stage(client, job, "RETURN_ACTIVE")
                body = checkpoint.read_bytes()
                return_checkpoint(self.project, client, digest, body, research_version=version)
                ContinualRegistry(registry_dir, self.v0).record_research_return(client, job, version, digest, len(body))
            except Exception:
                self._stage(client, job, "RETURN_FAILED")
            finally:
                with self.lock:
                    self.running.pop(client, None)

        try:
            Thread(target=deliver, daemon=True).start()
        except RuntimeError:
            with self.lock:
                self.running.pop(client, None)
            raise
        return {"status": "Research-only transfer started", "job": job}

    def return_model(self, client, chunk, job, *, local=False, version=None, digest=None):
        if local:
            return self._return_research(client, chunk, job, version, digest)
        from ppfcl_nids.federated_learning.communication import return_checkpoint
        from ppfcl_nids.dashboard.server_views import registry_snapshot

        row = next((row for row in self.snapshot_without_claim(client) if (row["id"], row["chunk"]) == (job, chunk)), None)
        if not row or not row["receipt"]:
            raise ClientActionError(409, "A verified client acceptance receipt is required.")
        if row["returned"]:
            raise ClientActionError(409, "A verified return receipt already exists.")
        registry_dir = directory(self.project)
        approvals = registry_dir.parent / "approvals"
        try:
            owner_storage(approvals)
            owner_storage(registry_dir)
            grant = local_json(approvals / f"{job}.return.json")
        except (OSError, ValueError):
            raise ClientActionError(409, "AWAITING APPROVED GLOBAL MODEL · no exact owner client-release approval.") from None
        if not isinstance(grant, dict):
            raise ClientActionError(403, "Invalid model release grant.")
        contract = row["approval"]["contract"] if row["approval"] else None
        expected = {"operation": "release_verified_model_to_client", "client": client, "job": job,
                    "version": grant.get("version"), "sha256": grant.get("sha256"), "contract": contract,
                    "purpose": "client_inventory_only", "expires": grant.get("expires")}
        if (grant != expected or not contract or type(grant["version"]) is not int
                or grant["version"] <= row["receipt"]["base_version"] or grant["version"] == 2
                or not isinstance(grant["sha256"], str) or not SHA.fullmatch(grant["sha256"])
                or type(grant["expires"]) is not int or grant["expires"] <= utc().timestamp()):
            raise ClientActionError(403, "Exact approved client inventory release required. Research v2 is restricted.")
        model = next((model for model in registry_snapshot(self.project, self.v0)["models"]
                      if (model["version"], model["sha256"], model["contract"], model["integrity"])
                      == (grant["version"], grant["sha256"], contract, "Verified")), None)
        if not model or not self.monitor.ssh_enabled:
            raise ClientActionError(409, "Verified release checkpoint or pinned SSH unavailable.")
        with self.lock:
            if client in self.running:
                raise ClientActionError(409, "A client operation is already active.")
            self.running[client] = job

        def deliver():
            try:
                self._ensure_dispatch_tables()
                self._stage(client, job, "RETURN_ACTIVE")
                if grant["expires"] <= utc().timestamp() or local_json(approvals / f"{job}.return.json") != grant:
                    raise PermissionError("Return approval expired or changed")
                checkpoint = (registry_dir / f'{grant["sha256"]}.pt').read_bytes()
                return_checkpoint(self.project, client, grant["sha256"], checkpoint)
                registry = ContinualRegistry(registry_dir, self.v0)
                registry.record_model_return(client, job, grant["version"], grant["sha256"], len(checkpoint))
            except Exception:
                self._stage(client, job, "RETURN_FAILED")
            finally:
                with self.lock:
                    self.running.pop(client, None)

        try:
            Thread(target=deliver, daemon=True).start()
        except RuntimeError:
            with self.lock:
                self.running.pop(client, None)
            raise
        return {"status": "Approved model inventory transfer started", "job": job}

    def federation(self, client, chunk, job, action, *, local=False, digest=None, version=None, update_digest=None):
        from ppfcl_nids.federated_learning.communication import worker_action

        row = next((row for row in self.snapshot_without_claim(client) if (row["id"], row["chunk"]) == (job, chunk)), None)
        if not row or not row["checkpoint"] or row["receipt"]:
            raise ClientActionError(409, "A staged, unaccepted update from this client is required.")
        exporting = action == "export"
        if exporting and not self.monitor.ssh_enabled:
            raise ClientActionError(409, "Pinned SSH export is unavailable.")
        received = row["submission"].startswith("Received")
        if exporting == received:
            raise ClientActionError(409, "Update already received." if exporting else "Export approval and verified receipt required.")
        registry_dir = directory(self.project)
        expected = {"operation": "submit_continual_update" if exporting else "aggregate_continual_update",
                    "job": job, "client": client, "sha256": row["checkpoint"], "base_sha256": self.v0.checkpoint_sha256,
                    "alpha_policy": "inverse-staleness-0.5-v1"}
        approvals = registry_dir.parent / "approvals"
        suffix = "submit" if exporting else "aggregate"
        if not exporting:
            with closing(_readonly(registry_dir / "registry.sqlite")) as db:
                head = db.execute("SELECT version,receipt FROM updates ORDER BY version DESC LIMIT 1").fetchone()
            if not head:
                raise ClientActionError(409, "No server head is available for the approved lineage.")
            expected.update(head_version=head[0], head_sha256=json.loads(head[1])["checkpoint_sha256"])
        if local:
            if (exporting and digest != expected["sha256"]
                    or not exporting and (version, digest, update_digest) !=
                    (expected["head_version"], expected["head_sha256"], expected["sha256"])):
                raise ClientActionError(409, "Confirmed update or server head changed. Review the latest proposal.")
            grant = expected
        else:
            try:
                owner_storage(approvals)
                owner_storage(registry_dir)
                grant = local_json(approvals / f"{job}.{suffix}.json")
            except (OSError, ValueError):
                raise ClientActionError(403, f"Exact owner {action} grant is required.") from None
        if grant != expected:
            raise ClientActionError(403, f"Owner {action} grant would authorize different bytes or a different server head.")
        with self.lock:
            if client in self.running:
                raise ClientActionError(409, "A client operation is already active.")
            self.running[client] = job

        def execute():
            try:
                registry = ContinualRegistry(registry_dir, self.v0)
                self._ensure_dispatch_tables()
                self._stage(client, job, "EXPORT_ACTIVE" if exporting else "AGGREGATION_ACTIVE")
                if exporting:
                    worker_action(self.project, registry, client, job, "export", expected_grant=grant, local_export=local)
                else:
                    if not local and local_json(approvals / f"{job}.{suffix}.json") != grant:
                        raise PermissionError("Aggregation approval changed")
                    registry.aggregate_received(client, job, grant)
            except Exception:
                self._stage(client, job, "EXPORT_FAILED" if exporting else "AGGREGATION_FAILED")
            finally:
                with self.lock:
                    self.running.pop(client, None)

        try:
            Thread(target=execute, daemon=True).start()
        except RuntimeError:
            with self.lock:
                self.running.pop(client, None)
            raise
        return {"status": f"{action.capitalize()} started", "job": job}
