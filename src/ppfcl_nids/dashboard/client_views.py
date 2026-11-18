"""Client-scoped dashboard projections from the existing research registry."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS, manifest, owned_chunk
from ppfcl_nids.federated_learning.model_registry import JOB, SHA
from ppfcl_nids.dashboard.infrastructure import HOSTS
from ppfcl_nids.dashboard.server_views import CLIENT_ROLES, _baseline, _research_models
from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID
from ppfcl_nids.organization.disclosure import local_json

CLIENT_ROUTES = {f"/api/{role}": client for client, role in CLIENT_ROLES.items()}
EXPORT = {
    "UPDATE_EXPORT_STARTED": "Update export started",
    "UPDATE_EXPORTED": "Update exported over pinned SSH",
    "UPDATE_RECEIVED": "Coordinator received update",
    "UPDATE_VALIDATED": "Coordinator validated update",
    "UPDATE_ACCEPTED": "Coordinator accepted update",
    "FEDASYNC_AGGREGATED": "FedAsync aggregation committed",
    "RESEARCH_VERSION_CREATED": "Research version recorded",
}
ACTIONS = {
    "SSH_CONNECTING": "Establishing authenticated SSH connection",
    "SSH_CONNECTED": "Pinned SSH connection authenticated",
    "MODEL_CHECKING": "Verifying the coordinator input checkpoint and guest copy",
    "MODEL_LOCAL": "Verified local model copy reused",
    "MODEL_RECEIVED": "Model transferred and destination digest verified",
    "WORKER_DISPATCHED": "Worker dispatched; validating the frozen selection and privacy projection",
    "INTERRUPTED": "Execution interrupted or outcome unconfirmed; reconcile durable state before any retry",
    "DISPATCH_FINISHED": "Worker dispatch finished",
    "EXPORT_ACTIVE": "Exporting the exact approved checkpoint to Windows",
    "EXPORT_FAILED": "Update export failed or its outcome is unconfirmed",
    "AGGREGATION_ACTIVE": "Validating the received update against the approved server head",
    "AGGREGATION_FAILED": "Aggregation rejected or interrupted; inspect durable receipts",
    "RETURN_ACTIVE": "Transferring separately approved checkpoint into client inventory",
    "RETURN_FAILED": "Model return failed or its receipt is unconfirmed",
}
SAFE_DT = ("sample_count", "batch_size", "microbatch_size", "epochs", "learning_rate",
           "clip", "noise", "delta", "epsilon_limit")


def utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value) -> str:
    stamp = (datetime.fromtimestamp(value, timezone.utc) if type(value) is int else datetime.fromisoformat(value))
    if stamp.utcoffset() is None:
        raise ValueError("Expected a timezone-aware timestamp")
    return stamp.isoformat()


def directory(project: Path) -> Path:
    return project / "state/continual-real/registry"


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)


def _schedule(scope: dict) -> dict:
    schedule = {key: scope["schedule"][key] for key in SAFE_DT}
    schedule["steps"] = schedule["sample_count"] // schedule["batch_size"]
    return schedule


def _historical_grant(registry: Path, client: str, chunk: str, job: str, authorization: str, v0, confirmation=None) -> dict | None:
    try:
        grant = json.loads(confirmation) if confirmation else local_json(registry.parent / "approvals" / f"{job}.train.json")
        scope = grant["scope"]
        if (hashlib.sha256(json.dumps(grant, sort_keys=True).encode()).hexdigest() != authorization
                or grant["job"] != job or grant["operation"] != "local_continual_training"
                or (scope["client"], scope["chunk"], scope["base_version"], scope["base_sha256"], scope["contract"]["contract_id"])
                != (client, chunk, 0, v0.checkpoint_sha256, BIFLOW_CONTRACT_ID)
                or not SHA.fullmatch(grant["slot_sha256"])):
            return None
        return {"schedule": _schedule(scope), "slot_sha256": grant["slot_sha256"],
                "base": 0, "base_sha256": v0.checkpoint_sha256, "contract": BIFLOW_CONTRACT_ID,
                "ewc": scope["ewc"] is True, "selection": scope["selection"], "attempt_limit": 1}
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError):
        return None


def _event(job, client, key, at, category, state, message, **details) -> dict:
    return {"id": f"{job}:{key}", "job": job, "client": client, "at": _iso(at),
            "category": category, "state": state, "message": message,
            "severity": "ERROR" if state.endswith("FAILED") or state == "INTERRUPTED" else "INFO", **details}


def _worker_events(db, job, client, chunk) -> list[dict]:
    events = []
    for sequence, body in db.execute("SELECT seq,body FROM events WHERE job=? ORDER BY seq LIMIT 37", (job,)):
        row = json.loads(body)
        if (row.get("job"), row.get("client"), row.get("chunk"), row.get("sequence")) != (job, client, chunk, sequence):
            raise ValueError("Worker event scope mismatch")
        state = row["state"]
        key = f"worker:{sequence}"
        lifecycle = {"DATASET_VERIFIED": "Frozen source, chunk ranks and separate validation selection verified",
                     "PREFLIGHT_VERIFIED": "Read-only privacy projection and selection verified",
                     "RESERVED": "Privacy reservation committed to the durable client ledger"}
        if state in lifecycle:
            events.append(_event(job, client, key, row["at"], "Worker", state, lifecycle[state]))
        elif state == "RUNNING":
            events.append(_event(job, client, key, row["at"], "Worker", state, "Worker passed dataset preflight and entered training"))
        elif state == "STEP":
            done, total = row["completed"], row["total"]
            if type(done) is not int or type(total) is not int or not 1 <= done <= total <= 32:
                raise ValueError("Invalid step evidence")
            events.append(_event(job, client, key, row["at"], "DP-SGD", state,
                                 f"Completed logical step {done} of {total}", steps=done, total=total))
        elif state == "STAGED" and SHA.fullmatch(row["sha256"]):
            events.append(_event(job, client, key, row["at"], "Checkpoint", state,
                                 "Local checkpoint and evaluation receipt recorded", sha256=row["sha256"], steps=row["steps"],
                                 evaluation_sha256=row.get("evaluation_sha256")))
        elif state == "FAILED":
            events.append(_event(job, client, key, row["at"], "Worker", state, "Worker recorded a failure"))
    return events


def _related_events(db, tables, job, client) -> list[dict]:
    events = []
    if "export_events" in tables:
        for state, stamp, size in db.execute("SELECT event,at,bytes FROM export_events WHERE job=? ORDER BY rowid LIMIT 20", (job,)):
            if state in EXPORT:
                details = {"bytes": size} if type(size) is int and 0 <= size <= 512 * 1024 else {}
                events.append(_event(job, client, f"export:{state}", stamp, "Federation", state, EXPORT[state], **details))
    if "evaluation_events" in tables:
        for seq, state, stamp in db.execute("SELECT seq,event,at FROM evaluation_events WHERE job=? ORDER BY seq LIMIT 4", (job,)):
            if state in {"EVALUATION_STARTED", "V0_VALIDATED", "V2_VALIDATED", "COMPARISON_COMPLETED"}:
                events.append(_event(job, client, f"comparison:{seq}", stamp, "Research comparison", state,
                                     state.replace("_", " ").capitalize()))
    return events


def dispatch_events(db_path, client, job) -> list[dict]:
    if not db_path.is_file():
        return []
    with closing(_readonly(db_path)) as db:
        return [_event(job, client, f"dispatch:{seq}", at, "Dispatch", state, ACTIONS[state],
                       **({"bytes": count} if type(count) is int else {}))
                for seq, at, state, count in db.execute(
                    "SELECT seq,at,state,bytes FROM stages WHERE job=? AND client=? ORDER BY seq LIMIT 24", (job, client))
                if state in ACTIONS]


def jobs_snapshot(project, client, v0, dispatch_db, active_job=None) -> tuple[list, bool]:
    registry = directory(project)
    path = registry / "registry.sqlite"
    if not path.is_file():
        return [], True
    with closing(_readonly(path)) as db, db:
        db.execute("BEGIN")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        jobs = db.execute("SELECT job,chunk,authorization FROM jobs WHERE client=? ORDER BY rowid DESC LIMIT 20", (client,)).fetchall()
        accepted = {}
        if db.execute("SELECT 1 FROM updates JOIN jobs USING(job) WHERE client=? LIMIT 1", (client,)).fetchone():
            try:
                _, verified, _ = _research_models(registry, _baseline(v0), v0)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error):
                verified = []
            accepted = {row["update_id"]: {key: row[key] for key in
                        ("version", "base_version", "parent_version", "alpha", "staleness", "sha256", "created_at")}
                        for row in verified if row["client"] == client}
        result = []
        for job, chunk, authorization in jobs:
            if not JOB.fullmatch(job) or not owned_chunk(client, chunk):
                continue
            local = db.execute("SELECT body FROM local_confirmations WHERE job=?", (job,)).fetchone() if "local_confirmations" in tables else None
            grant = _historical_grant(registry, client, chunk, job, authorization, v0, local[0] if local else None)
            worker = _worker_events(db, job, client, chunk)
            related = _related_events(db, tables, job, client)
            dispatch = dispatch_events(dispatch_db, client, job)
            returned = None
            research_returns = []
            if "research_returns" in tables:
                for ver, digest, at, size in db.execute("""SELECT version,sha,at,bytes FROM research_returns
                                                            WHERE job=? ORDER BY at,version""", (job,)):
                    research_returns.append({"version": ver, "sha256": digest, "at": at, "bytes": size})
                    related.append(_event(job, client, f"research-return:{ver}", at, "Federation", "RESEARCH_RETURN_VERIFIED",
                                          f"Research-only checkpoint v{ver} verified in client inventory"))
            if "model_returns" in tables:
                row = db.execute("SELECT version,sha,at,bytes FROM model_returns WHERE job=?", (job,)).fetchone()
                if row:
                    returned = {"version": row[0], "sha256": row[1], "at": row[2], "bytes": row[3]}
                    related.append(_event(job, client, "model-return", row[2], "Federation", "RETURN_VERIFIED",
                                          f"Research v{row[0]} verified in client inventory"))
            result.append(project_job(client, job, chunk, grant, worker, related, dispatch, accepted.get(job),
                                      job == active_job, returned=returned, research_returns=research_returns))
        return result, True


def project_job(client, job, chunk, grant, worker, related, dispatch, receipt, active, *, returned=None, research_returns=None) -> dict:
    latest = worker[-1] if worker else None
    staged = latest if latest and latest["state"] == "STAGED" else None
    failed = latest if latest and latest["state"] == "FAILED" else None
    trained = next((e for e in reversed(worker) if e["state"] == "STEP"), None)
    started = next((e for e in worker if e["state"] == "RUNNING"), None)
    marks = {e["state"]: e for e in dispatch + worker + related}
    now = utc()
    if active and latest and latest["state"] in {"RUNNING", "STEP"} and now - datetime.fromisoformat(latest["at"]) > timedelta(seconds=45):
        active = False
    # Worker stamps have second precision. Preserve lifecycle order when dispatch
    # and the first worker event occur in the same second.
    events = sorted(dispatch + worker + related, key=lambda e: (
        datetime.fromisoformat(e["at"]).replace(microsecond=0),
        2 if e["state"] in {"DISPATCH_FINISHED", "INTERRUPTED"} else 1 if ":worker:" in e["id"] else 0,
    ))
    steps = trained["steps"] if trained else 0
    total = (trained or {}).get("total") or (grant or {}).get("schedule", {}).get("steps")
    stage = []

    def add(label, state="PENDING", mark=None, note=""):
        stage.append({"label": label, "state": state, "at": mark["at"] if mark else None, "note": note})

    ssh = marks.get("SSH_CONNECTED")
    model = marks.get("MODEL_LOCAL") or marks.get("MODEL_RECEIVED")
    dispatched = marks.get("WORKER_DISPATCHED")
    interrupted = marks.get("INTERRUPTED")
    connecting = marks.get("SSH_CONNECTING")
    checking_model = marks.get("MODEL_CHECKING")
    add("SSH connection", "COMPLETED" if ssh else "RUNNING" if active and connecting else
        "FAILED" if connecting and interrupted else "INTERRUPTED" if connecting else "UNRECORDED",
        ssh or (interrupted if connecting else None), "Pinned SSH authenticated" if ssh else
        "Authenticated SSH could not be established" if connecting and interrupted else "")
    add("Receive global model", ("VERIFIED LOCAL COPY" if model["state"] == "MODEL_LOCAL" else "COMPLETED") if model
        else "RUNNING" if active and checking_model else "FAILED" if checking_model and interrupted else
        "INTERRUPTED" if checking_model else
        "VERIFIED LOCAL COPY" if started and grant else "PENDING", model,
        (model["message"] + (f' · {model["bytes"]:,} bytes' if "bytes" in model else "")) if model
        else "Global v0 verified by the worker; transfer timing unavailable" if started and grant else "")
    dataset = marks.get("DATASET_VERIFIED") or started
    reservation = marks.get("RESERVED") or trained
    add("Verify dataset chunk", "COMPLETED" if dataset else "FAILED" if failed else "RUNNING" if active and dispatched else
        "INTERRUPTED" if dispatched else "PENDING", dataset or failed, "Worker passed frozen selection preflight" if dataset else "")
    add("Privacy preflight and reservation", "COMPLETED" if reservation else "FAILED" if failed and dataset else
        "RUNNING" if active and dataset else "INTERRUPTED" if dataset and started else "PENDING",
        reservation or (failed if dataset else None),
        "Durable reservation recorded" if marks.get("RESERVED") else
        "Reservation precedes the first logical step" if trained else "Awaiting worker reservation evidence")
    add("Local training", "COMPLETED" if staged or total and steps == total else "FAILED" if failed else
        "RUNNING" if active and started and reservation else "INTERRUPTED" if started and reservation else "PENDING",
        trained if total and steps == total else failed, f"{steps} / {total} steps" if total else "")
    add("Checkpoint and evaluation", "COMPLETED" if staged else "FAILED" if failed and total and steps == total else
        "RUNNING" if active and total and steps == total else "INTERRUPTED" if total and steps == total else "PENDING",
        staged or (failed if total and steps == total else None),
        "Checkpoint staged; evaluation stays client-local" if staged else "")
    exported = marks.get("UPDATE_EXPORTED")
    operation = dispatch[-1]["state"] if dispatch else None
    exporting = active and operation == "EXPORT_ACTIVE"
    aggregating = active and operation == "AGGREGATION_ACTIVE"
    add("Send model update", "COMPLETED" if exported else "RUNNING" if exporting else
        "FAILED" if operation == "EXPORT_FAILED" else "INTERRUPTED" if operation == "EXPORT_ACTIVE" or marks.get("UPDATE_EXPORT_STARTED") else
        "AWAITING APPROVAL" if staged else "PENDING", exported,
        f'{exported["bytes"]:,} bytes · digest verified' if exported and "bytes" in exported else "")
    add("FL validation and FedAsync", "COMPLETED" if receipt else "RUNNING" if aggregating
        else "FAILED" if operation == "AGGREGATION_FAILED" else "INTERRUPTED" if operation == "AGGREGATION_ACTIVE"
        else "UNVERIFIED" if marks.get("UPDATE_ACCEPTED")
        else "AWAITING APPROVAL" if marks.get("UPDATE_RECEIVED") else "PENDING",
        (marks.get("FEDASYNC_AGGREGATED") or {"at": receipt["created_at"]}) if receipt else None,
        f'v{receipt["parent_version"]} → Research v{receipt["version"]} · α {receipt["alpha"]:.2f} · staleness {receipt["staleness"]}'
        if receipt else ACTIONS.get(operation, "") if aggregating or operation == "AGGREGATION_FAILED" else "")
    returning = active and operation == "RETURN_ACTIVE"
    research_returns = research_returns or []
    research = research_returns[-1] if research_returns else None
    return_complete = marks.get("RESEARCH_RETURN_VERIFIED") or marks.get("RETURN_VERIFIED")
    return_interrupted = operation == "RETURN_ACTIVE" and (not return_complete or return_complete["at"] < dispatch[-1]["at"])
    add("Receive updated global model", "RUNNING" if returning else "FAILED" if operation == "RETURN_FAILED" else
        "INTERRUPTED" if return_interrupted else "COMPLETED" if returned or research else
        "AWAITING APPROVAL" if receipt else "PENDING", returned or research,
        note=f'Research v{returned["version"]} · {returned["bytes"]:,} bytes verified in inventory' if returned else
        f'Research checkpoint received · v{research["version"]} · {research["bytes"]:,} bytes verified' if research else
        "Awaiting approved global model" if receipt else "Not executed")
    current = ("Completed" if staged else "Failed" if failed else "Training" if active and started else
               "Preparing" if active else "Interrupted / outcome unconfirmed")
    return {"id": job, "client": client, "chunk": chunk, "status": current,
            "active": active and (not (staged or failed) or exporting or aggregating or returning),
            "approval": grant, "steps": steps if trained else None, "total": total,
            "started_at": started["at"] if started else None, "ended_at": (staged or failed or {}).get("at"),
            "last_event": events[-1]["at"] if events else None, "checkpoint": staged["sha256"] if staged else None,
            "evaluation_sha256": staged.get("evaluation_sha256") if staged else None,
            "evaluation": "Client-local receipt hash verified" if staged and staged.get("evaluation_sha256")
            else "Client-local receipt recorded" if staged else "Unavailable",
            "privacy": "Durable reservation recorded" if marks.get("RESERVED") else "Reservation evidenced by logical step" if trained else "Unconfirmed",
            "submission": "Received · digest verified" if marks.get("UPDATE_RECEIVED") else
            "Export started" if marks.get("UPDATE_EXPORT_STARTED") else "Ready for submission" if staged else "Not recorded",
            "receipt": receipt, "returned": returned, "research_returns": research_returns, "stages": stage, "events": events,
            "action": ACTIONS[operation] if exporting or aggregating or returning or operation in
            ("EXPORT_FAILED", "AGGREGATION_FAILED", "RETURN_FAILED") else
            "Research checkpoint received" if research else
            "Approved checkpoint verified in inventory" if returned else "Awaiting approved global model" if receipt else
            interrupted["message"] if interrupted and not (failed or staged) else
            (latest or (dispatch[-1] if dispatch else {})).get("message", "Awaiting worker evidence")}


def chunk_snapshot(project: Path, client: str, jobs: list, registry_ok: bool) -> dict:
    dataset, filename, _, _, digest = ASSIGNMENTS[client]
    try:
        verified = manifest(project / "state/audit" / f"chunks-{dataset}.json", client)
    except (OSError, KeyError, ValueError, TypeError):
        verified = None
    job_by_chunk = {job["chunk"]: job for job in reversed(jobs)}
    rows = []
    for index in range(1, 21):
        chunk = f"{dataset}{index}"
        job = job_by_chunk.get(chunk)
        item = verified["chunks"][index - 1] if verified else None
        eligible = bool(verified and registry_ok and not job)
        reason = ("A durable attempt already exists. Automatic retraining is blocked." if job else
                  "Registry or frozen manifest unavailable." if not eligible else "Read-only preflight required.")
        rows.append({"id": chunk, "rows": item["rows"] if item else None, "start": item["start"] if item else None,
                     "stop": item["stop_exclusive"] if item else None,
                     "selection_size": job["approval"]["schedule"]["sample_count"] if job and job["approval"] else None,
                     "verification": "Manifest verified" if item else "Unavailable",
                     "status": job["status"] if job else "Not trained" if registry_ok and item else "Blocked",
                     "job": job["id"] if job else None, "eligible": eligible, "reason": reason})
    return {"dataset": dataset, "filename": filename, "manifest_sha256": digest if verified else None, "chunks": rows}


def client_health(client: str, monitor: dict) -> dict:
    role = CLIENT_ROLES[client]
    row = monitor["nodes"][role]
    last = row.get("last_known") or {}
    state = row["state"]
    if state == "UNAVAILABLE":
        state = "UNKNOWN" if row.get("error") == "PROBE_CONFIGURATION_UNAVAILABLE" else "OFFLINE"
    if state == "PARTIAL":
        state = "ONLINE"
    return {"client": client, "status": state, "identity": row.get("identity") or last.get("identity") or {},
            "metrics": row.get("metrics"), "checked_at": row.get("checked_at"), "last_success": last.get("checked_at"),
            "error": row.get("error"), "ssh_enabled": monitor["ssh_enabled"], "configured_ip": HOSTS[role],
            "server_ip": HOSTS["Windows"]}
