"""Loopback operator dashboard for the local synthetic workflow."""

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import parse_qs

from ppfcl_nids.dashboard.analytics import HISTORICAL_PHASES, historical_performance, session_analytics
from ppfcl_nids.dashboard.client_actions import ClientActionError, ClientRuns
from ppfcl_nids.dashboard.client_views import CLIENT_ROUTES, client_health
from ppfcl_nids.datasets.continual_partitions import readiness
from ppfcl_nids.dashboard.hybrid_view import snapshot as hybrid_snapshot
from ppfcl_nids.dashboard.infrastructure import InfrastructureMonitor, Thresholds
from ppfcl_nids.dashboard.organization_view import organization_snapshot
from ppfcl_nids.dashboard.server_views import CLIENT_ROLES, machine_snapshot, registry_snapshot
from ppfcl_nids.federated_learning.sessions import ActiveDemoError, DemoSessionManager, LOCAL_SYNTHETIC, PINNED_RESEARCH_V0, REMOTE_SYNTHETIC
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC
from ppfcl_nids.continual_learning.importance_store import LocalEWCStore
from ppfcl_nids.continual_learning.chunk_workflow import InvalidRunState, SyntheticManualChunks
from ppfcl_nids.organization.disclosure import display_summary, owner_storage
from ppfcl_nids.organization.telemetry import FlowReceiver, INGEST, VIEW

from ppfcl_nids.project_paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
STATIC = Path(__file__).with_name("frontend")
PAGES = ("overview", "server", "client-a", "client-b", "models", "privacy", "health", "audit",
         "server-tools", "client-a-tools", "client-b-tools")
DASHBOARD_V0 = replace(PINNED_RESEARCH_V0, reference_path=ROOT / PINNED_RESEARCH_V0.reference_path)
ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
RUN_ACTIONS = {
    "approve": "approve_run", "run": "start_run", "submit": "submit_run",
    "complete": "complete_run", "recover": "recover_run", "abandon": "resolve_run",
}
ACTION_ROUTES = {f"/api/actions/{action}": action for action in (*RUN_ACTIONS, "start", "close", "select")}
REQUEST_TIMEOUT = 5
MAX_REQUEST_BODY = 4096
REJECT_DRAIN_TIMEOUT = 0.25
ORGANIZATION_ROUTE = "/api/organization"
HYBRID_ROUTE = "/api/organization/hybrid-lab"
FL_REGISTRY = "/api/fl-server"
FL_HEALTH = "/api/fl-server/health"
CLIENT_HEALTH = {f"{path}/health": role for path, role in CLIENT_ROUTES.items()}
CLIENT_ACTIONS = {f"{path}/{action}": (role, action) for path, role in CLIENT_ROUTES.items()
                  for action in ("preflight", "approve", "export", "aggregate", "reconcile", "return-model")}
_ACCESS_LOG = logging.getLogger("ppfcl_nids.dashboard.server.access")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("Non-JSON numeric constant")


def _matches_secret(candidate, expected: str) -> bool:
    return isinstance(candidate, str) and candidate.isascii() and hmac.compare_digest(candidate, expected)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def matches(path: Path, expected: str) -> bool:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected
    except OSError:
        return False


class DashboardData:
    def __init__(self, state_dir: Path, *, project: Path = ROOT, v0=DASHBOARD_V0,
                 worker_status_namespace: str | None = None, remote_worker_namespace: str | None = None,
                 monitor_ssh: bool = False, thresholds: Thresholds = Thresholds()):
        self.state_dir = state_dir
        self.project = project
        self.v0 = v0
        for namespace in (worker_status_namespace, remote_worker_namespace):
            if namespace is not None and not re.fullmatch(r"[0-9a-f]{32}", namespace):
                raise ValueError("Invalid worker namespace")
        if worker_status_namespace and remote_worker_namespace and worker_status_namespace != remote_worker_namespace:
            raise ValueError("Worker status and execution namespaces must match")
        self.remote_worker_namespace = remote_worker_namespace
        self.worker_status_namespace = remote_worker_namespace or worker_status_namespace
        self.database = state_dir / "sessions.sqlite"
        self._audit_lock = Lock()
        self._initialization_lock = Lock()
        self._manager = None
        self._clients = {}
        self._dispatch_lock = Lock()
        self._dispatching: dict[tuple[str, str], str] = {}
        self.monitor = InfrastructureMonitor(project, ssh_enabled=monitor_ssh, thresholds=thresholds)
        self.client_runs = ClientRuns(project, state_dir, v0, self.monitor)
        try:
            self.v0.verify()
            self.baseline_loaded = True
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            self.baseline_loaded = False

    def manager(self, *, create=False):
        with self._initialization_lock:
            if self._manager is None and (create or self.database.is_file()):
                self._manager = DemoSessionManager(self.database, v0=self.v0)
            return self._manager

    def client(self, role: str) -> SyntheticManualChunks:
        if role not in ("A", "B"):
            raise PermissionError("Unknown client scope")
        manager = self.manager()
        if manager is None:
            raise LookupError("Start a synthetic demo first")
        with self._initialization_lock:
            if role not in self._clients:
                store = LocalEWCStore(self.state_dir / f"client-{role}.sqlite", client_id=role)
                self._clients[role] = SyntheticManualChunks(manager, store)
            return self._clients[role]

    def _worker_probe(self, role: str, namespace: str, *, provision: bool = False):
        from ppfcl_nids.federated_learning.remote_chunks import SSHWorker

        worker = SSHWorker(role, namespace, project=self.project)
        checked = now()
        try:
            health = worker.call("health")
            if provision and worker.worker_id is None:
                health = worker.call("initialize")
        except (OSError, ValueError, RuntimeError):
            status = {"state": "UNAVAILABLE"}
        else:
            status = {"state": "VERIFIED" if worker.worker_id else "UNPROVISIONED",
                      **{key: health[key] for key in
                         ("cpu_only", "torch", "free_disk_bytes", "available_ram_bytes", "address")}}
            status["ready"] = (status["state"] == "VERIFIED" and status["cpu_only"]
                               and status["free_disk_bytes"] >= 3 * 2**30
                               and status["available_ram_bytes"] >= 2**30)
        status["checked_at"] = checked
        manager = self.manager()
        if status["state"] == "VERIFIED":
            status["last_verified_at"] = checked
            if manager:
                with closing(manager._connect()) as db, db:
                    db.execute("""INSERT INTO dashboard_worker_health VALUES (?,?,?)
                        ON CONFLICT(namespace,client_id) DO UPDATE SET last_verified_at =
                        MAX(last_verified_at,excluded.last_verified_at)""", (namespace, role, checked))
        elif manager:
            with closing(manager._connect("ro")) as db:
                previous = db.execute("""SELECT last_verified_at FROM dashboard_worker_health
                    WHERE namespace=? AND client_id=?""", (namespace, role)).fetchone()
            status["last_verified_at"] = previous[0] if previous else None
        return worker, status

    def worker_status(self, role: str) -> dict:
        if self.worker_status_namespace is None:
            return {"state": "DISABLED"}
        return self._worker_probe(role, self.worker_status_namespace)[1]

    def artifact(self, name: str, *, stage: str = "task9e") -> tuple[dict, str | None]:
        path = self.project / "state" / stage / name
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            return (value, stamp) if isinstance(value, dict) else ({}, None)
        except (OSError, ValueError):
            return {}, None

    def historical(self) -> dict:
        local, stamp = self.artifact("local-integration.json")
        probes, probe_stamp = self.artifact("vm-preflight.json")
        prior, prior_stamp = self.artifact("synthetic-demo.json", stage="task9d")
        remote, remote_stamp = self.artifact("integration.json", stage="task10b/7944aad520064a35bb560bc470765989")
        runs = []
        if local.get("v0_sha256") == self.v0.checkpoint_sha256 and local.get("exact_six_tensor_checks") is True:
            raw = local.get("runs")
            if isinstance(raw, list):
                for row in raw:
                    if isinstance(row, dict):
                        runs.append({key: row.get(key) for key in (
                            "chunk_id", "base_version", "accepted_version", "staleness",
                            "aggregation_weight", "checkpoint_sha256",
                        )})
        vm = {}
        for role in ("A", "B", "organization"):
            row = probes.get(role, {})
            if isinstance(row, dict) and row.get("strict_ssh") is True:
                vm[role] = {key: row.get(key) for key in (
                    "strict_ssh", "available_ram_bytes", "scratch_free_bytes", "peak_process_rss_kib",
                    "isolated_optimizer_seconds", "isolated_fisher_seconds",
                )}
        timings = local.get("phase_timings_ms", {}) if runs else {}
        revisions = local.get("ewc_revision_counts", {}) if runs else {}
        prior_count = None
        if (prior.get("source_v0_sha256") == self.v0.checkpoint_sha256
                and prior.get("exact_mixing_and_all_checkpoint_hashes_verified") is True
                and isinstance(prior.get("runs"), list)):
            prior_count = len(prior["runs"])
        remote_runs = []
        if remote.get("versions", {}).get("0") == self.v0.checkpoint_sha256:
            for chunk, entry in remote.get("runs", {}).items():
                receipt = entry.get("receipt", {})
                role = receipt.get("client_id")
                version = receipt.get("accepted_version")
                prefix = {"A": "C", "B": "D"}.get(role)
                if (entry.get("tensors_verified") == 6 and prefix and receipt.get("chunk_id") == chunk
                        and re.fullmatch(fr"{prefix}(?:[1-9]|1[0-9]|20)", chunk)
                        and receipt.get("checkpoint_sha256") == remote["versions"].get(str(version))):
                    remote_runs.append({"chunk_id": chunk, "client_id": role, "version": version,
                                        "staleness": receipt["staleness"], "alpha": receipt["aggregation_weight"],
                                        "at": receipt["accepted_at"], "timings": {
                                            phase: (entry.get("worker", {}).get(phase)
                                                    if phase == "training_and_fisher_seconds" else entry.get(phase))
                                            for phase in HISTORICAL_PHASES}})
        return {"runs": runs, "artifact_modified_at": stamp, "timings": timings, "ewc_revisions": revisions,
                "task10b_runs": remote_runs, "task10b_artifact_modified_at": remote_stamp,
                "task10b_performance": historical_performance(remote_runs), "task10b_phases": HISTORICAL_PHASES,
                "vm_checks": vm, "vm_artifact_modified_at": probe_stamp,
                "task9d_verified_updates": prior_count, "task9d_artifact_modified_at": prior_stamp}

    def audit(self, action: str, *, outcome: str, session_id=None, client=None, execution=None) -> bool:
        record = {"event": action, "outcome": outcome, "session_id": session_id,
                  "client_id": client, "execution_id": execution, "time": now()}
        try:
            with self._audit_lock:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                with (self.state_dir / "actions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        except OSError:
            return False
        return True

    def _action_history(self) -> list[dict]:
        from collections import deque
        with self._audit_lock:
            try:
                with (self.state_dir / "actions.jsonl").open(encoding="utf-8") as stream:
                    lines = deque(stream, maxlen=300)
            except OSError:
                return []
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except ValueError:
                continue
        return result

    def snapshot(self, selected: str | None) -> dict:
        history = self.historical()
        sessions, models, chunks, updates, runs, revisions, audit = [], [], [], [], [], [], []
        current = None
        manager = self.manager()
        if manager is not None:
            with closing(manager._connect("ro")) as db, db:
                db.execute("BEGIN")

                def rows(sql, params=()):
                    return [dict(row) for row in db.execute(sql, params)]

                sessions = rows("""SELECT s.session_id,s.status,s.created_at,s.closed_at,s.execution_mode,
                    (SELECT MAX(m.version) FROM session_model_versions m WHERE m.session_id=s.session_id)
                    AS head_version FROM demo_sessions s ORDER BY created_at DESC""")
                if selected:
                    current = next((s for s in sessions if s["session_id"] == selected), None)
                    if current is None:
                        raise LookupError("Unknown demo session")
                elif sessions:
                    current = next((s for s in sessions if s["status"] == "ACTIVE"), sessions[0])
                if current:
                    key = (current["session_id"],)
                    models = rows("""SELECT version,parent_version,base_version,client_id,chunk_id,checkpoint_path,
                        checkpoint_sha256,staleness,aggregation_weight,created_at FROM session_model_versions
                        WHERE session_id=? ORDER BY version""", key)
                    chunks = rows("""SELECT c.client_id,c.chunk_id,c.chunk_number,
                        COALESCE(r.state,c.status) AS state,r.execution_id
                        FROM client_chunk_progress c LEFT JOIN manual_chunk_runs r ON r.rowid=(
                          SELECT MAX(p.rowid) FROM manual_chunk_runs p WHERE p.session_id=c.session_id
                          AND p.client_id=c.client_id AND p.chunk_id=c.chunk_id)
                        WHERE c.session_id=? ORDER BY c.client_id,c.chunk_number""", key)
                    updates = rows("""SELECT u.client_id,u.chunk_id,u.update_id,u.base_version,u.accepted_version,
                        u.accepted_at,u.payload_sha256,m.staleness,m.aggregation_weight,m.checkpoint_sha256
                        FROM federated_updates u JOIN session_model_versions m ON m.session_id=u.session_id
                        AND m.version=u.accepted_version WHERE u.session_id=? ORDER BY u.accepted_version""", key)
                    runs = rows("""SELECT r.execution_id,r.client_id,r.chunk_id,r.state,r.base_version,
                        r.accepted_version,r.used_ewc,r.failure_reason,r.approved_at,r.base_checkpoint_sha256,
                        r.update_sha256,r.ewc_sha256,r.worker_training_seconds,r.worker_steps,
                        (SELECT occurred_at FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         AND e.state='RUNNING') AS started_at,
                        (SELECT occurred_at FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         AND e.state='STAGED') AS staged_at,
                        (SELECT occurred_at FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         AND e.state='SUBMITTED') AS submitted_at,
                        (SELECT event_id FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         AND e.state='RUNNING') AS training_order,
                        (SELECT occurred_at FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         ORDER BY e.event_id DESC LIMIT 1) AS state_at,
                        (SELECT occurred_at FROM manual_run_events e WHERE e.execution_id=r.execution_id
                         AND e.state='COMPLETED') AS completed_at
                        FROM manual_chunk_runs r WHERE r.session_id=? ORDER BY r.rowid DESC""", key)
                    revisions = rows("""SELECT client_id,revision,update_id,state_ref,state_sha256,
                        anchor_checkpoint_sha256,created_at FROM ewc_reference_history
                        WHERE session_id=? ORDER BY created_at""", key)
                audit = rows("""SELECT event_type AS event,session_id,occurred_at AS time,'Recorded' AS outcome
                    FROM session_events ORDER BY event_id DESC LIMIT 300""")
                audit += rows("""SELECT CASE WHEN s.execution_mode='REMOTE_SYNTHETIC' THEN 'REMOTE ' || e.state
                         ELSE e.state END AS event,r.session_id,r.client_id,e.execution_id,
                    e.occurred_at AS time,'Recorded' AS outcome
                    FROM manual_run_events e JOIN manual_chunk_runs r USING(execution_id)
                    JOIN demo_sessions s ON s.session_id=r.session_id
                    ORDER BY e.event_id DESC LIMIT 300""")
        elif selected:
            raise LookupError("Unknown demo session")
        for model in models:
            path = Path(model.pop("checkpoint_path"))
            model["integrity"] = "Verified" if matches(path, model["checkpoint_sha256"]) else "Unavailable or mismatch"
        with self._dispatch_lock:
            in_flight = dict(self._dispatching)
        for run in runs:
            run["operation_in_progress"] = in_flight.get((current["session_id"], run["client_id"])) == run["execution_id"]
            if run["started_at"] and run["staged_at"]:
                run["elapsed_ms"] = (datetime.fromisoformat(run["staged_at"])
                                     - datetime.fromisoformat(run["started_at"])).total_seconds() * 1000
        baseline = False
        if self.baseline_loaded:
            try:
                reference = self.v0.reference_path.read_bytes()
                baseline = (hashlib.sha256(reference).hexdigest() == self.v0.reference_sha256
                            and matches(self.v0.reference_path.with_name(json.loads(reference)["checkpoint_filename"]),
                                        self.v0.checkpoint_sha256))
            except OSError:
                pass
        audit = sorted(audit + self._action_history(), key=lambda r: r["time"], reverse=True)
        analytics = session_analytics(current, models, chunks, updates, runs, revisions, audit, now=now(),
                                      prolonged_seconds=self.monitor.thresholds.prolonged_seconds)
        if not baseline:
            analytics["alerts"].append({"code": "BASELINE_VERIFICATION_FAILED"})
        from ppfcl_nids.federated_learning.model_registry import ContinualRegistry
        try:
            real_jobs = ContinualRegistry.dashboard_snapshot(self.project / "state/continual-real/registry")
        except (OSError, ValueError, sqlite3.Error):
            real_jobs = None
        return {
            "baseline": {"sha256": self.v0.checkpoint_sha256, "verified": baseline, **EWC_SPEC.metadata()},
            "sessions": sessions, "session": current, "models": models, "chunks": chunks,
            "updates": updates, "runs": runs, "revisions": revisions, "historical": history,
            "remote_enabled": self.remote_worker_namespace is not None,
            "audit": audit, "analytics": analytics,
            "continual_readiness": readiness(self.project),
            "continual_jobs": real_jobs,
            "health": {"checked_at": now(), "windows_disk_free_bytes": shutil.disk_usage(self.project).free},
        }

    def action(self, name: str, body: dict) -> dict:
        if name == "start":
            _require_fields(body, "request_id", *(("execution_mode",) if "execution_mode" in body else ()))
            mode = body.get("execution_mode", LOCAL_SYNTHETIC)
            if mode not in (LOCAL_SYNTHETIC, REMOTE_SYNTHETIC):
                raise ValueError("Invalid synthetic execution mode")
            if mode == REMOTE_SYNTHETIC and self.remote_worker_namespace is None:
                raise InvalidRunState("Remote synthetic execution is disabled on this dashboard")
            return self.manager(create=True).start_demo(request_id=body["request_id"], execution_mode=mode,
                worker_namespace=self.remote_worker_namespace if mode == REMOTE_SYNTHETIC else None)
        if name == "close":
            _require_fields(body, "session_id")
            manager = self.manager()
            if manager is None:
                raise LookupError("No demo exists")
            return manager.close_demo(body["session_id"])
        if name not in {*RUN_ACTIONS, "select"}:
            raise LookupError("Unknown synthetic action")
        fields = ("session_id", "client_id", "execution_id")
        fields += ("chunk_id",) if name == "select" else ()
        confirmation = "confirm_remote" in body and name == "run"
        _require_fields({k: v for k, v in body.items() if not (confirmation and k == "confirm_remote")}, *fields)
        manager = self.manager()
        if manager is None:
            raise LookupError("No demo exists")
        session = manager.get_session(body["session_id"])
        if session["status"] != "ACTIVE":
            raise InvalidRunState("The selected session is closed")
        if session["execution_mode"] == REMOTE_SYNTHETIC:
            return self._remote_action(manager, session, name, body)
        if confirmation:
            raise ValueError("Remote confirmation does not match the session mode")
        client = self.client(body["client_id"])
        if name == "select":
            return client.select_chunk(body["session_id"], body["chunk_id"], body["execution_id"])
        return getattr(client, RUN_ACTIONS[name])(body["session_id"], body["execution_id"])

    def _remote_action(self, manager, session, name, body):
        from ppfcl_nids.federated_learning.remote_chunks import RemoteSyntheticChunks

        sid, role, execution = (body[key] for key in ("session_id", "client_id", "execution_id"))
        namespace = session["worker_namespace"]
        if namespace != self.remote_worker_namespace:
            raise InvalidRunState("This session requires its original remote worker namespace")
        if role not in ("A", "B"):
            raise PermissionError("Unknown client scope")
        if name == "run" and body.get("confirm_remote") is not True:
            raise InvalidRunState("Confirm remote synthetic execution before Run")
        if name == "select":
            prefix = {"A": "C", "B": "D"}[role]
            if not re.fullmatch(fr"{prefix}(?:[1-9]|1[0-9]|20)", body["chunk_id"]):
                raise PermissionError("Chunk belongs to a different client")
        else:
            with closing(manager._connect("ro")) as db:
                row = db.execute("""SELECT state FROM manual_chunk_runs
                    WHERE session_id=? AND client_id=? AND execution_id=?""", (sid, role, execution)).fetchone()
            if row is None:
                raise LookupError("No matching remote execution")
            if name == "run" and row["state"] == "SELECTED":
                raise InvalidRunState("Approve this remote chunk before Run")
        key = (sid, role)
        with self._dispatch_lock:
            if key in self._dispatching:
                raise InvalidRunState("A client operation is in progress. Refresh its recorded state.")
            self._dispatching[key] = execution
        try:
            worker, status = self._worker_probe(role, namespace, provision=name == "select")
            if status["state"] != "VERIFIED":
                raise ConnectionError(f"Client {role} worker unavailable")
            if name == "run" and not status["ready"]:
                raise InvalidRunState("Worker does not meet the CPU, disk or memory requirements")
            client = RemoteSyntheticChunks(manager, worker)
            if name == "select":
                return client.select_chunk(sid, body["chunk_id"], execution)
            return getattr(client, RUN_ACTIONS[name])(sid, execution)
        finally:
            with self._dispatch_lock:
                del self._dispatching[key]


def _require_fields(body: dict, *fields: str) -> None:
    if set(body) != set(fields) or any(not isinstance(body[key], str) or not ID.fullmatch(body[key]) for key in fields):
        raise ValueError("Invalid action fields")


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, data: DashboardData, port: int, access_key: str | None, telemetry=None):
        self.data = data
        self.access_key = access_key
        self.local_only = access_key is None
        self.local_csrf = secrets.token_urlsafe(32) if self.local_only else None
        self.telemetry = telemetry
        self.tokens: dict[str, tuple[str, float]] = {}
        self.token_lock = Lock()
        super().__init__(("127.0.0.1", port), DashboardHandler)
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host}"

    def login(self) -> tuple[str, str]:
        with self.token_lock:
            stamp = monotonic()
            self.tokens = {key: value for key, value in self.tokens.items() if value[1] > stamp}
            if len(self.tokens) >= 16:
                self.tokens.pop(next(iter(self.tokens)))
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self.tokens[token] = (csrf, stamp + 8 * 3600)
            return token, csrf


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer
    timeout = REQUEST_TIMEOUT
    server_version = "FederatedIDS"
    sys_version = ""

    def log_message(self, *_args):
        pass

    def respond(self, status, body, content_type="application/json", *, cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        try:
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # The operator may reload while a durable action is running.
            self.close_connection = True
            return
        if self.command == "POST" and status >= 400 and not getattr(self, "_body_received", False):
            self._finish_rejected_post()

    def _finish_rejected_post(self):
        """Send EOF before closing a socket with a potentially unread request body."""
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_WR)
            # Discard pending bytes without parsing or waiting on an unbounded peer.
            deadline = monotonic() + REJECT_DRAIN_TIMEOUT
            budget = MAX_REQUEST_BODY
            while budget:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self.connection.settimeout(remaining)
                part = self.rfile.read1(budget)
                if not part:
                    break
                budget -= len(part)
        except OSError:
            pass

    def json_response(self, status, payload, **headers):
        self.respond(status, json.dumps(payload, allow_nan=False).encode(), **headers)

    def _token(self):
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            return cookie["ids_operator"].value
        except (CookieError, KeyError, ValueError):
            return ""

    def _csrf(self):
        if self.server.local_only:
            return self.server.local_csrf
        with self.server.token_lock:
            token = self.server.tokens.get(self._token())
            return token[0] if token and token[1] > monotonic() else None

    def _target(self):
        target = self.requestline.split()[1]
        if not target.startswith("/") or target.startswith("//") or "#" in target or "\\" in target:
            raise ValueError("Invalid request target")
        return target.partition("?")

    def _host_is_valid(self):
        return self.client_address[0] == "127.0.0.1" and self.headers.get_all("Host", []) == [self.server.host]

    def _body(self) -> dict:
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,4}", lengths[0])
                or len(self.headers.get_all("Content-Type", [])) != 1
                or self.headers.get_content_type() != "application/json"
                or "Transfer-Encoding" in self.headers or "Expect" in self.headers):
            raise ValueError("Invalid request framing")
        size = int(lengths[0])
        if not 0 < size <= MAX_REQUEST_BODY:
            raise ValueError("Invalid request size")
        deadline = monotonic() + REQUEST_TIMEOUT
        raw = bytearray()
        try:
            while len(raw) < size:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.connection.settimeout(remaining)
                part = self.rfile.read1(size - len(raw))
                if not part:
                    raise ValueError("Incomplete request")
                raw.extend(part)
        finally:
            self.connection.settimeout(REQUEST_TIMEOUT)
        self._body_received = True
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        return body

    def do_GET(self):
        if (self.server.local_only and self.client_address[0] == "127.0.0.1"
                and self.headers.get_all("Host", []) == [f"localhost:{self.server.server_port}"]
                and self.path in ("/", *(f"/{page}" for page in PAGES))):
            self.send_response(307)
            self.send_header("Location", self.server.origin + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self._host_is_valid():
            return self.json_response(403, {"error": "Unexpected host"})
        try:
            path, separator, query = self._target()
        except ValueError:
            return self.json_response(400, {"error": "Invalid request target"})
        assets = {"/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/app.css": ("app.css", "text/css; charset=utf-8"),
                  "/fl-server.js": ("fl-server.js", "text/javascript; charset=utf-8"),
                  "/fl-server.css": ("fl-server.css", "text/css; charset=utf-8"),
                  "/client-dashboard.js": ("client-dashboard.js", "text/javascript; charset=utf-8"),
                  "/client-dashboard.css": ("client-dashboard.css", "text/css; charset=utf-8")}
        if path == "/" or path[1:] in PAGES:
            return self.respond(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in assets:
            name, mime = assets[path]
            return self.respond(200, (STATIC / name).read_bytes(), mime)
        worker_routes = {"/api/workers/A": "A", "/api/workers/B": "B"}
        if path not in ("/api/state", "/api/monitor", ORGANIZATION_ROUTE, HYBRID_ROUTE, VIEW,
                        FL_REGISTRY, FL_HEALTH) and path not in worker_routes and path not in CLIENT_ROUTES and path not in CLIENT_HEALTH:
            return self.json_response(404, {"error": "Unknown route"})
        origins = self.headers.get_all("Origin", [])
        if self.server.local_only and (origins not in ([], [self.server.origin])
                or self.headers.get_all("Sec-Fetch-Site", []) not in ([], ["same-origin"], ["none"])):
            return self.json_response(403, {"error": "Same-origin request required"})
        csrf = self._csrf()
        if csrf is None:
            return self.json_response(401, {"error": "Operator login required"})
        try:
            if path in CLIENT_ROUTES or path in CLIENT_HEALTH:
                if separator:
                    return self.json_response(400, {"error": "Client monitoring accepts no target parameters"})
                service = self.server.data
                if path in CLIENT_ROUTES:
                    return self.json_response(200, {**service.client_runs.snapshot(CLIENT_ROUTES[path], local=self.server.local_only),
                                                    "csrf": csrf, "local_only": self.server.local_only})
                client = CLIENT_HEALTH[path]
                service.monitor.queue_client_refresh(CLIENT_ROLES[client])
                return self.json_response(200, client_health(client, service.monitor.snapshot()))
            if path in (FL_REGISTRY, FL_HEALTH):
                if separator:
                    return self.json_response(400, {"error": "FL Server accepts no target parameters"})
                data = self.server.data
                if path == FL_REGISTRY:
                    return self.json_response(200, registry_snapshot(data.project, data.v0))
                data.monitor.queue_refresh()
                return self.json_response(200, machine_snapshot(data.project, data.monitor.snapshot(), self.server.telemetry))
            if path in (HYBRID_ROUTE, VIEW):
                tokens = self.headers.get_all("X-CSRF-Token", [])
                if len(tokens) != 1 or not _matches_secret(tokens[0], csrf):
                    return self.json_response(403, {"error": "Owner session verification required"})
                if separator:
                    return self.json_response(400, {"error": "Hybrid evaluation accepts no parameters"})
                if path == VIEW:
                    return self.json_response(200, self.server.telemetry.snapshot() if self.server.telemetry else None)
                return self.json_response(200, hybrid_snapshot(self.server.data.project))
            if path == ORGANIZATION_ROUTE:
                if separator:
                    return self.json_response(400, {"error": "Organization metadata accepts no parameters"})
                payload = organization_snapshot(self.server.data.project, self.server.data.monitor.snapshot())
                release = payload["release"] or {}
                if payload["evidence_state"] == "RECORDED_VERIFIED":
                    summary = display_summary(self.server.data.project, source_revoked=release.get("window") == "REVOKED")
                    if summary is not None:
                        payload["approved_summary"] = summary
                _ACCESS_LOG.info("organization_metadata_read outcome=%s", payload["evidence_state"])
                return self.json_response(200, payload)
            if path == "/api/monitor" or path in worker_routes:
                if query:
                    return self.json_response(400, {"error": "Monitoring accepts no target parameters"})
                payload = (self.server.data.monitor.snapshot() if path == "/api/monitor" else
                           self.server.data.worker_status(worker_routes[path]))
                return self.json_response(200, payload)
            try:
                options = parse_qs(query, strict_parsing=True, keep_blank_values=True, max_num_fields=1)
                selected = options.get("session", [None])[0]
                if options and (selected is None or not ID.fullmatch(selected)):
                    raise ValueError
            except ValueError:
                return self.json_response(400, {"error": "Invalid session query"})
            return self.json_response(200, {**self.server.data.snapshot(selected), "csrf": csrf,
                                            "local_only": self.server.local_only})
        except LookupError:
            return self.json_response(404, {"error": "Unknown demo session"})
        except (OSError, sqlite3.Error, ValueError):
            return self.json_response(503, {"error": "Dashboard data unavailable"})

    def do_POST(self):
        if not self._host_is_valid():
            return self.json_response(403, {"error": "Same-origin request required"})
        try:
            path, separator, _ = self._target()
        except ValueError:
            return self.json_response(400, {"error": "Invalid request target"})
        if path == INGEST:
            return self.ingest(separator)
        if self.headers.get_all("Origin", []) != [self.server.origin]:
            return self.json_response(403, {"error": "Same-origin request required"})
        if self.server.local_only and path in ("/api/login", "/api/logout"):
            return self.json_response(404, {"error": "Local dashboard has no sign-in or sign-out"})
        if path not in ("/api/login", "/api/logout", "/api/monitor", FL_HEALTH) and path not in ACTION_ROUTES and path not in CLIENT_ACTIONS:
            return self.json_response(404, {"error": "Unknown route"})
        if path != "/api/login":
            csrf = self._csrf()
            if csrf is None:
                return self.json_response(401, {"error": "Operator login required"})
            tokens = self.headers.get_all("X-CSRF-Token", [])
            if len(tokens) != 1 or not _matches_secret(tokens[0], csrf):
                return self.json_response(403, {"error": "Invalid CSRF token"})
        try:
            if separator:
                raise ValueError("Unexpected query")
            body = self._body()
        except TimeoutError:
            return self.json_response(408, {"error": "Request body timed out"})
        except ConnectionError:
            self.close_connection = True
            return
        except (ValueError, UnicodeError, RecursionError):
            return self.json_response(400, {"error": "Expected a bounded JSON object"})
        if path in CLIENT_ACTIONS:
            client, action = CLIENT_ACTIONS[path]
            fields = {"chunk"} if action == "preflight" else {"chunk", "job", "confirm"}
            if self.server.local_only and action in ("return-model", "aggregate", "export"):
                fields |= {"sha256"}
                if action in ("return-model", "aggregate"):
                    fields |= {"version"}
                if action == "aggregate":
                    fields |= {"update_sha256"}
            if (set(body) != fields or not isinstance(body.get("chunk"), str)
                    or action != "preflight" and (type(body.get("job")) is not str or body.get("confirm") is not True)
                    or "sha256" in fields and (not isinstance(body.get("sha256"), str)
                                              or not re.fullmatch(r"[a-f0-9]{64}", body["sha256"]))
                    or "version" in fields and (type(body.get("version")) is not int or body["version"] < 0)
                    or "update_sha256" in fields and (not isinstance(body.get("update_sha256"), str)
                        or not re.fullmatch(r"[a-f0-9]{64}", body["update_sha256"]))):
                return self.json_response(400, {"error": "Invalid client action"})
            try:
                service = self.server.data.client_runs
                if action == "preflight":
                    result = service.preflight(client, body["chunk"], local=self.server.local_only)
                elif action == "approve":
                    result = service.approve(client, body["chunk"], body["job"], local=self.server.local_only)
                elif action == "reconcile":
                    result = service.reconcile(client, body["chunk"], body["job"])
                elif action == "return-model":
                    result = service.return_model(client, body["chunk"], body["job"], local=self.server.local_only,
                                                  version=body.get("version"), digest=body.get("sha256"))
                else:
                    result = service.federation(client, body["chunk"], body["job"], action, local=self.server.local_only,
                                                digest=body.get("sha256"), version=body.get("version"),
                                                update_digest=body.get("update_sha256"))
                return self.json_response(202, result)
            except ClientActionError as exc:
                return self.json_response(exc.status, {"error": str(exc)})
            except (OSError, sqlite3.Error, ValueError, RuntimeError):
                return self.json_response(503, {"error": "Client operation unavailable"})
        if path == "/api/login":
            if set(body) != {"access_key"}:
                return self.json_response(400, {"error": "Expected an access key"})
            if not _matches_secret(body["access_key"], self.server.access_key):
                return self.json_response(401, {"error": "Invalid operator access key"})
            token, csrf = self.server.login()
            return self.json_response(200, {"csrf": csrf},
                                      cookie=f"ids_operator={token}; Path=/; HttpOnly; SameSite=Strict")
        if path == "/api/monitor":
            if body:
                return self.json_response(400, {"error": "Monitoring accepts no target parameters"})
            return self.json_response(200, self.server.data.monitor.refresh())
        if path == FL_HEALTH:
            if body:
                return self.json_response(400, {"error": "Monitoring accepts no target parameters"})
            monitor = self.server.data.monitor
            queued = monitor.queue_refresh(manual=True)
            state = monitor.snapshot()
            return self.json_response(202 if queued else 200, {
                "queued": queued, "refreshing": state["refreshing"],
                "cooldown_seconds": state["manual_cooldown_seconds"],
            })
        if path == "/api/logout":
            if body:
                return self.json_response(400, {"error": "Logout expects an empty object"})
            with self.server.token_lock:
                self.server.tokens.pop(self._token(), None)
            return self.json_response(200, {"signed_out": True},
                                      cookie="ids_operator=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
        action = ACTION_ROUTES[path]
        data = self.server.data
        scope = {k: body.get(k) if isinstance(body.get(k), str) and ID.fullmatch(body[k]) else None
                 for k in ("session_id", "client_id", "execution_id")}
        try:
            result = data.action(action, body)
        except (PermissionError, LookupError, ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            if isinstance(exc, PermissionError):
                status, message = HTTPStatus.FORBIDDEN, "Action is outside the client scope"
            elif isinstance(exc, LookupError):
                status, message = HTTPStatus.NOT_FOUND, "No matching session or execution"
            elif isinstance(exc, (InvalidRunState, ActiveDemoError)):
                status, message = HTTPStatus.CONFLICT, str(exc)
            elif isinstance(exc, (ValueError, RuntimeError)):
                status, message = HTTPStatus.CONFLICT, "Action rejected by the session or workflow state"
            else:
                status, message = HTTPStatus.SERVICE_UNAVAILABLE, "Action interrupted. Inspect status before recovery"
            data.audit(action, outcome=f"Rejected · {type(exc).__name__}", session_id=scope["session_id"],
                       client=scope["client_id"], execution=scope["execution_id"])
            return self.json_response(status, {"error": message})
        recorded = data.audit(action, outcome="Succeeded", session_id=result.get("session_id", scope["session_id"]),
                              client=scope["client_id"], execution=scope["execution_id"])
        payload = {"result": {k: v for k, v in result.items() if k not in ("worker_store_id", "worker_namespace")}}
        if not recorded:
            payload["audit_recorded"] = False
        return self.json_response(200, payload)

    def ingest(self, separator):
        receiver = self.server.telemetry
        if receiver is None:
            return self.json_response(404, {"error": "Ingestion disabled"})
        if self.headers.get_all("Origin", []) or not receiver.authenticate(self.headers.get_all("Authorization", [])):
            receiver.reject()
            return self.json_response(401, {"error": "Agent authentication required"})
        if not receiver.admit():
            receiver.reject()
            return self.json_response(429, {"error": "Ingestion rate exceeded"})
        try:
            if separator:
                raise ValueError("Unexpected query")
            accepted = receiver.insert(self._body())
        except (ValueError, UnicodeError, RecursionError):
            receiver.reject()
            return self.json_response(400, {"error": "Invalid telemetry event"})
        except (OSError, sqlite3.Error, TimeoutError):
            receiver.reject()
            return self.json_response(503, {"error": "Receiver unavailable"})
        except ConnectionError:
            self.close_connection = True
            return
        return self.json_response(200, {"accepted": accepted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--require-operator-key", action="store_true", help="Use the legacy keyed loopback mode")
    parser.add_argument("--organization-telemetry", action="store_true", help="Enable owner-only Organization receiver")
    parser.add_argument("--worker-status-namespace", help="Enable authenticated read-only A/B worker probes")
    parser.add_argument("--remote-worker-namespace", help="Enable remote synthetic sessions in a dedicated namespace")
    parser.add_argument("--monitor-ssh", action="store_true", help="Enable read-only SSH infrastructure probes")
    parser.add_argument("--monitor-interval", type=int, default=60, help="Minimum seconds between health probes")
    parser.add_argument("--monitor-stale", type=int, default=300, help="Seconds before samples become stale")
    parser.add_argument("--monitor-low-disk-gib", type=float, default=3)
    parser.add_argument("--monitor-high-memory", type=float, default=85)
    parser.add_argument("--monitor-prolonged", type=int, default=120, help="Seconds before warning about a running worker")
    options = parser.parse_args()
    key = secrets.token_urlsafe(24) if options.require_operator_key else None
    namespace = options.remote_worker_namespace
    if namespace is not None and not re.fullmatch(r"[0-9a-f]{32}", namespace):
        parser.error("Remote worker namespace must be 32 lowercase hexadecimal characters")
    directory = ROOT / "state" / ("task10c-dashboard" if namespace else "task10a-dashboard")
    try:
        thresholds = Thresholds(options.monitor_interval, options.monitor_stale, options.monitor_low_disk_gib,
                                options.monitor_high_memory, options.monitor_prolonged)
    except ValueError as exc:
        parser.error(str(exc))
    data = DashboardData(directory / namespace if namespace else directory,
                         worker_status_namespace=options.worker_status_namespace, remote_worker_namespace=namespace,
                         monitor_ssh=options.monitor_ssh, thresholds=thresholds)
    receiver = None
    if options.organization_telemetry:
        telemetry = ROOT / "state/organization-telemetry"
        try:
            owner_storage(telemetry)
            with (telemetry / "ingest.key").open(encoding="ascii") as stream:
                token = stream.read(130).strip()
            receiver = FlowReceiver(telemetry, token, telemetry / "live-grant.json")
        except (OSError, ValueError, sqlite3.Error):
            parser.error("Initialize the Organization receiver's owner-only storage first")
    with DashboardServer(data, options.port, key, receiver) as server:
        print(f"Dashboard URL  {server.origin}", flush=True)
        if key is not None:
            try:
                with open("CONOUT$" if os.name == "nt" else "/dev/tty", "w", encoding="utf-8") as terminal:
                    print(f"Operator access key  {key}", file=terminal, flush=True)
            except OSError:
                if not sys.stdout.isatty():
                    parser.error("Start the dashboard from an interactive console to receive the operator key")
                print(f"Operator access key  {key}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
