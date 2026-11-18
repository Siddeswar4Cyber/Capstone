"""Coordinator-side adapter for approved synthetic chunk runs over pinned SSH."""

import json
import math
import re
import shlex
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter, sleep, time

from ppfcl_nids.configuration.settings import Role
from ppfcl_nids.dashboard.infrastructure import read_json_command
from ppfcl_nids.federated_learning.sessions import DemoSessionManager, Record
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC
from ppfcl_nids.continual_learning.importance_store import CommittedEWCReference, EWCReference
from ppfcl_nids.federated_learning.asynchronous_aggregation import SyntheticFedAsync, decode_synthetic_update
from ppfcl_nids.continual_learning.chunk_workflow import InvalidRunState, SYNTHETIC_TRAINING_STEPS, SyntheticManualChunks
from ppfcl_nids.federated_learning.remote_worker import ADDRESSES, APPROVAL_SECONDS, OPS, PROTOCOL, decode_artifact, encode_artifact, parse_permit
from ppfcl_nids.federated_learning.synthetic_round import frame, read_frame
from ppfcl_nids.virtual_machines.round_execution import GUESTS, client_package, make_package
from ppfcl_nids.ssh.connections import ssh_session

RUNTIME = ".local/share/ppfcl-nids/runtime/bin/python"
MODULES = ("demo_sessions", "fedasync", "ewc", "ewc_store", "ewc_handoff", "manual_chunks",
           "remote_worker", "worker_health")


class SSHWorker:
    """One allowlisted client route and isolated namespace. No caller-supplied paths or commands."""

    def __init__(self, client_id: str, namespace: str, *, project: Path):
        if client_id not in ADDRESSES or not re.fullmatch(r"[0-9a-f]{32}", namespace):
            raise ValueError("Invalid synthetic worker client or namespace")
        self.client_id = client_id
        self.namespace = namespace
        self.project = project
        self.worker_id: str | None = None
        self.package = None
        self.measurements: list[dict] = []

    def _exchange(self, peer, environment: str, request: dict) -> tuple[dict, dict]:
        command = environment + shlex.join([
            RUNTIME, "-B", "-m", "ppfcl_nids.federated_learning.remote_worker", self.client_id, self.namespace,
        ])
        started = perf_counter()
        stdin, stdout, _ = peer.exec_command(command, timeout=45)
        metrics = {"command_open_seconds": perf_counter() - started}
        try:
            payload = frame(request)
            started = perf_counter()
            stdin.write(payload)
            stdin.flush()
            stdin.channel.shutdown_write()
            metrics["request_send_seconds"] = perf_counter() - started
            started = perf_counter()
            response = read_frame(stdout)
            metrics["response_wait_seconds"] = perf_counter() - started
            metrics["request_frame_bytes"] = len(payload)
            metrics["response_frame_bytes"] = len(response)
            deadline = monotonic() + 5
            while not stdout.channel.exit_status_ready():
                if monotonic() >= deadline:
                    raise ConnectionError("Worker channel did not exit")
                sleep(0.02)
            if stdout.channel.recv_exit_status():
                raise ConnectionError("Worker command failed")
            return json.loads(response), metrics
        finally:
            stdout.channel.close()

    def call(self, operation: str, **params) -> dict:
        if operation not in {*OPS, "health", "initialize"}:
            raise ValueError("Worker command is not allowlisted")
        is_admin = operation in ("health", "initialize")
        if not is_admin and self.worker_id is None:
            raise PermissionError("Verify the provisioned worker identity before running work")
        role = {"A": Role.CLIENT_A, "B": Role.CLIENT_B}[self.client_id]
        _, address = GUESTS[role]
        env = self.project / ".env"
        if operation == "health":
            if params:
                raise ValueError("Health takes no parameters")
            return self._read_health(role, env, address)
        request = {"protocol": PROTOCOL, "operation": operation, "worker_id": None if is_admin else self.worker_id,
                   "contract": EWC_SPEC.metadata(), "params": params}
        started = perf_counter()
        with ssh_session(role, env_file=env, expected_host=address) as peer:
            from paramiko import SSHException

            if peer.get_transport().getpeername()[0] != address:
                raise PermissionError("Authenticated SSH peer is outside the approved route")
            connected = perf_counter() - started
            try:
                if self.package is None:
                    self.package = make_package(MODULES)
                with client_package(peer, self.package) as environment:
                    response, metrics = self._exchange(peer, environment, request)
            except (OSError, EOFError, ValueError, SSHException) as exc:
                raise ConnectionError("Worker RPC did not return a valid response") from exc
        self.measurements.append({"operation": operation, "ssh_connect_seconds": connected,
                                  "rpc_seconds": perf_counter() - started, **metrics})
        if not isinstance(response, dict):
            raise ConnectionError("Malformed worker response")
        expected = {"protocol": PROTOCOL, "operation": operation, "client_id": self.client_id,
                    "namespace": self.namespace, "contract": EWC_SPEC.metadata()}
        if any(response.get(key) != value for key, value in expected.items()):
            raise PermissionError("Worker response identity, route or contract mismatch")
        identity = response.get("worker_id")
        if self.worker_id is not None and identity != self.worker_id:
            raise PermissionError("Worker store identity changed")
        if "error" in response:
            raise InvalidRunState(f"Worker rejected {operation}")
        if not isinstance(response.get("result"), dict):
            raise ConnectionError("Worker result is missing")
        if identity is not None and (not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity)):
            raise PermissionError("Malformed worker store identity")
        self.worker_id = identity
        return response["result"]

    def _read_health(self, role, env: Path, address: str) -> dict:
        from ppfcl_nids.deployment import module_path

        script = module_path("worker_health").read_text(encoding="utf-8")
        command = shlex.join([RUNTIME, "-I", "-B", "-c", script, self.client_id, self.namespace])
        with ssh_session(role, env_file=env, expected_host=address, timeout=4) as peer:
            from paramiko import SSHException

            transport = peer.get_transport()
            if transport is None or transport.getpeername()[0] != address:
                raise PermissionError("Authenticated SSH peer is outside the approved route")
            try:
                response = read_json_command(transport, command, timeout=8)
            except (OSError, ValueError, SSHException) as exc:
                raise ConnectionError("Read-only worker health probe failed") from exc
        if (not isinstance(response, dict) or response.get("address") != address
                or response.get("namespace") != self.namespace or response.get("client_id") != self.client_id):
            raise PermissionError("Worker status scope mismatch")
        identity = response.get("worker_id")
        if identity is not None and (not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity)):
            raise PermissionError("Malformed worker store identity")
        if self.worker_id is not None and identity != self.worker_id:
            raise PermissionError("Worker store identity changed")
        if (type(response.get("cpu_only")) is not bool or not isinstance(response.get("torch"), str)
                or not re.fullmatch(r"[0-9A-Za-z.+_-]{1,80}", response["torch"])
                or any(type(response.get(key)) is not int or not 0 <= response[key] < 2**63
                       for key in ("free_disk_bytes", "available_ram_bytes"))):
            raise ValueError("Malformed worker health measurements")
        self.worker_id = identity
        return {key: response[key] for key in ("address", "cpu_only", "torch", "free_disk_bytes", "available_ram_bytes")}


class RemoteSyntheticChunks(SyntheticManualChunks):
    """Use the existing session state machine with authenticated client-local staging."""

    def __init__(self, sessions: DemoSessionManager, worker: SSHWorker):
        if worker.worker_id is None:
            raise PermissionError("A provisioned and verified worker is required")
        super().__init__(sessions, worker)
        self.worker = worker

    def _rpc(self, operation, row, **params):
        return self.worker.call(operation, permit=asdict(self._permit(row)), **params)

    def _reference(self, row, result) -> EWCReference:
        value = result.get("reference")
        if not isinstance(value, dict) or value.keys() != {"permit", "revision", "parent_sha256", "sha256"}:
            raise ValueError("Missing worker EWC reference")
        permit = parse_permit(value["permit"], self.worker.client_id, staged=True)
        hashes = [value["sha256"]]
        if value["parent_sha256"] is not None:
            hashes.append(value["parent_sha256"])
        if (permit != self._permit(row, permit.artifact_sha256)
                or (row["update_sha256"] is not None and row["update_sha256"] != permit.artifact_sha256)
                or (row["ewc_sha256"] is not None and row["ewc_sha256"] != value["sha256"])
                or type(value["revision"]) is not int or not 1 <= value["revision"] <= 20
                or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes)):
            raise ValueError("Worker reference differs from the approved execution or contract")
        return EWCReference(permit, value["revision"], value["parent_sha256"], value["sha256"])

    def retrieve_update(self, session_id: str, execution_id: str) -> tuple[EWCReference, bytes]:
        row = self.get_run(session_id, execution_id)
        if row["state"] not in ("RUNNING", "STAGED", "SUBMITTED", "ACCEPTED_AWAITING_EWC", "COMPLETED"):
            raise InvalidRunState("Run cannot retrieve a staged update")
        result = self._rpc("update", row)
        ref = self._reference(row, result)
        artifact = decode_artifact(result.get("artifact"))
        decode_synthetic_update(ref.permit, artifact, self.registry._template)
        return ref, artifact

    def worker_run_status(self, session_id: str, execution_id: str) -> dict:
        row = self.get_run(session_id, execution_id)
        if row["base_version"] is None:
            raise InvalidRunState("Run has not captured a training base")
        return self._rpc("status", row)

    def pending_reference(self, session_id: str, execution_id: str) -> tuple[EWCReference, str]:
        row = self.get_run(session_id, execution_id)
        result = self._rpc("pending", row)
        return self._reference(row, result), result["promotion_state"]

    def _record_worker_staging(self, session_id, execution_id, result):
        reference, _ = self.retrieve_update(session_id, execution_id)
        measures = result.get("measures") or {}
        duration, steps = measures.get("training_and_fisher_seconds"), measures.get("training_steps")
        if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError("Invalid worker duration")
        if steps is not None and (type(steps) is not int or steps != SYNTHETIC_TRAINING_STEPS):
            raise ValueError("Invalid worker step count")
        return self._record_staged(reference, training_seconds=duration, training_steps=steps)

    def start_run(self, session_id: str, execution_id: str) -> Record:
        """Only the caller that durably claims explicit Run may send Execute."""
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["state"] == "SELECTED":
                raise InvalidRunState("Explicit approval is required before remote Run")
            if row["state"] != "APPROVED":
                return dict(row)
            expiry = int(datetime.fromisoformat(row["approved_at"]).timestamp()) + APPROVAL_SECONDS
            if time() >= expiry:
                raise InvalidRunState("Remote run approval has expired")
            running, parent = self._claim_run(db, row, check_local=False)
        base = self.registry.export_training_checkpoint(self._permit(running))
        self._rpc("prepare", running, expires_at=expiry, parent_sha256=parent, base=encode_artifact(base))
        result = self._rpc("execute", running)
        if result.get("state") != "STAGED":
            raise InvalidRunState("Worker did not confirm durable staging. Inspect or recover the run.")
        return self._record_worker_staging(session_id, execution_id, result)

    def submit_run(self, session_id: str, execution_id: str) -> Record:
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["state"] not in ("STAGED", "SUBMITTED", "ACCEPTED_AWAITING_EWC", "COMPLETED"):
                raise InvalidRunState("Submit requires a durably staged remote run")
            if row["state"] == "STAGED":
                self._transition(db, row, "SUBMITTED")
            existing = self._receipt(db, row)
        if existing is None:
            reference, artifact = self.retrieve_update(session_id, execution_id)
            SyntheticFedAsync(self.sessions, trusted_updates=[reference.permit]).accept_async_update(
                reference.permit, artifact)
        return self._record_acceptance(session_id, execution_id)

    def complete_run(self, session_id: str, execution_id: str) -> Record:
        self._record_acceptance(session_id, execution_id)
        row = self.get_run(session_id, execution_id)
        if row["state"] == "COMPLETED":
            return row
        acknowledgment = self.handoff.get_acknowledgment(self._permit(row, row["update_sha256"]))
        if acknowledgment is None:
            raise InvalidRunState("Missing durable acceptance receipt")
        result = self._rpc("reconcile", row, ack=asdict(acknowledgment))
        if result.get("promotion_state") != "COMMITTED":
            raise InvalidRunState("Worker did not confirm EWC promotion")
        committed = CommittedEWCReference(self._reference(row, result), acknowledgment)
        return self._finish_completion(session_id, execution_id, committed)

    def _fail_run(self, session_id, execution_id, *, discard: bool):
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["state"] == "FAILED":
                return dict(row)
            if row["state"] in ("COMPLETED", "ACCEPTED_AWAITING_EWC") or self._receipt(db, row) is not None:
                raise InvalidRunState("A committed update must be reconciled")
            if row["state"] != "ABANDONING":
                row = self._transition(db, row, "ABANDONING",
                                       failure_reason="explicit_abandonment" if discard else "interrupted_training")
        if row["base_version"] is not None and self._rpc("discard", row).get("state") != "FAILED":
            raise InvalidRunState("Worker did not confirm abandonment")
        with self._write() as db:
            row = self._run(db, session_id, execution_id)
            if row["state"] == "FAILED":
                return dict(row)
            db.execute("""UPDATE client_chunk_progress SET status='FAILED'
                          WHERE session_id=? AND client_id=? AND chunk_id=?""",
                       (session_id, self.worker.client_id, row["chunk_id"]))
            return self._transition(db, row, "FAILED")

    def resolve_run(self, session_id: str, execution_id: str) -> Record:
        return self._fail_run(session_id, execution_id, discard=True)

    def recover_run(self, session_id: str, execution_id: str) -> Record:
        row = self.get_run(session_id, execution_id)
        if row["state"] == "RUNNING":
            result = self._rpc("recover", row)
            if result.get("state") == "STAGED":
                return self._record_worker_staging(session_id, execution_id,
                                                   self.worker_run_status(session_id, execution_id))
            if result.get("state") != "FAILED":
                raise InvalidRunState("Worker recovery was inconclusive")
            return self._fail_run(session_id, execution_id, discard=False)
        if row["state"] == "ABANDONING":
            return self._fail_run(session_id, execution_id, discard=True)
        if row["state"] in ("SUBMITTED", "ACCEPTED_AWAITING_EWC"):
            self.registry.recover_checkpoints()
            self.submit_run(session_id, execution_id)
            return self.complete_run(session_id, execution_id)
        return row
