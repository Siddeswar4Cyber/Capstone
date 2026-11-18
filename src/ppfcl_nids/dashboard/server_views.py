"""Operator-safe projections of the preserved research registry and health cache."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from time import time

from ppfcl_nids.configuration.settings import Role
from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS
from ppfcl_nids.federated_learning.model_registry import JOB, SHA, first_chunk
from ppfcl_nids.datasets.continual_partitions import owned_chunk
from ppfcl_nids.dashboard.organization_view import organization_snapshot
from ppfcl_nids.federated_learning.sessions import ResearchV0
from ppfcl_nids.organization.telemetry import FRESH, scope_digest

CLIENT_ROLES = {"A": Role.CLIENT_A, "B": Role.CLIENT_B}


def _timestamp(value: str) -> str:
    instant = datetime.fromisoformat(value)
    if instant.utcoffset() is None:
        raise ValueError("Registry time has no offset")
    return instant.isoformat()


def _verified_file(path: Path, digest: str) -> bool:
    try:
        if not path.is_symlink():
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == digest
    except OSError:
        pass
    return False


def _baseline(v0: ResearchV0) -> dict | None:
    reference_bytes = v0.reference_path.read_bytes()
    checkpoint = v0.verify()
    if hashlib.sha256(reference_bytes).hexdigest() != v0.reference_sha256:
        return None
    reference = json.loads(reference_bytes)
    stamp = reference.get("created_at") or reference.get("designated_utc")
    contract = reference["feature_contract"]
    return {
        "version": 0, "name": "Global v0", "type": "Global (initial)", "created_at": _timestamp(stamp) if stamp else None,
        "time_basis": "Created" if reference.get("created_at") else "Registry designation",
        "source": reference.get("aggregation", {}).get("algorithm", "Initial baseline"),
        "contributors": sorted(reference.get("aggregation", {}).get("weights", {})),
        "sha256": v0.checkpoint_sha256, "architecture": contract["architecture"], "contract": contract["contract_id"],
        "integrity": "Verified" if _verified_file(checkpoint, v0.checkpoint_sha256) else "Unavailable or mismatch",
        "status": "Approved research baseline",
        "evaluation": "Client-local checks recorded" if reference.get("evaluation", {}).get("client_local_validation_complete")
        else "Evaluation evidence unavailable",
        "limitations": reference.get("limitations", []) + ["Approval is scoped to the local coordinator reference."],
    }


def _training_evidence(directory: Path, db: sqlite3.Connection, v0: ResearchV0) -> dict:
    results = {}
    rows = db.execute("""SELECT j.job,j.client,j.chunk,j.authorization,e.body
                         FROM jobs j JOIN events e ON j.job=e.job
                         WHERE e.seq=34 ORDER BY j.rowid""")
    for job, client, chunk, authorization, body in rows:
        try:
            if client not in ASSIGNMENTS or chunk != first_chunk(client) or not JOB.fullmatch(job):
                continue
            staged = json.loads(body)
            if (staged["state"], staged["client"], staged["job"], staged["chunk"], staged["steps"]) != (
                    "STAGED", client, job, chunk, 32) or not SHA.fullmatch(staged["sha256"]):
                continue
            events = [json.loads(event) for event, in db.execute(
                "SELECT body FROM events WHERE job=? AND seq BETWEEN 1 AND 33 ORDER BY seq", (job,))]
            expected = [("RUNNING", None)] + [("STEP", index) for index in range(1, 33)]
            grant = json.loads((directory.parent / "approvals" / f"{job}.train.json").read_bytes())
            scope = grant["scope"]
            schedule = scope["schedule"]
            bound = (hashlib.sha256(json.dumps(grant, sort_keys=True).encode()).hexdigest() == authorization
                     and grant["job"] == job and grant["operation"] == "local_continual_training"
                     and (scope["client"], scope["chunk"], scope["base_version"], scope["base_sha256"])
                     == (client, chunk, 0, v0.checkpoint_sha256)
                     and schedule["clip"] > 0 and schedule["noise"] > 0)
            results[client] = {
                "job": job, "chunk": chunk, "completed_at": datetime.fromtimestamp(staged["at"], timezone.utc).isoformat(),
                "checkpoint": "Staged · digest recorded", "digest": staged["sha256"],
                "bounded_training": bound and [(e["state"], e.get("completed")) for e in events] == expected,
                "ewc_disabled": bound and scope["ewc"] is False,
            }
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            continue
    return results


def _research_models(directory: Path, baseline: dict | None, v0: ResearchV0) -> tuple[list, list, dict]:
    models, accepted = [], []
    parent = baseline
    contributors = []
    with closing(sqlite3.connect((directory / "registry.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db, db:
        db.execute("BEGIN")
        training = _training_evidence(directory, db, v0)
        evaluation_available = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluation_events'").fetchone()
        rows = db.execute("""SELECT j.job,j.client,j.chunk,u.version,u.sha,u.checkpoint,u.receipt
                             FROM updates u JOIN jobs j USING(job) ORDER BY u.version""")
        for job, client, chunk, version, update_sha, checkpoint, raw_receipt in rows:
            try:
                receipt = json.loads(raw_receipt)
                digest = receipt["checkpoint_sha256"]
                stamp = _timestamp(receipt["accepted_at"])
                if (not owned_chunk(client, chunk) or not JOB.fullmatch(job)
                        or not SHA.fullmatch(digest) or not SHA.fullmatch(update_sha)
                        or (receipt["job"], receipt["client"], receipt["chunk"], receipt["version"], receipt["sha256"])
                        != (job, client, chunk, version, update_sha)
                        or receipt["designation"] != "unvalidated_aggregation_checkpoint"
                        or receipt["operation"] != "submit_continual_update"
                        or receipt["alpha_policy"] != "inverse-staleness-0.5-v1"
                        or type(version) is not int or version < 1
                        or type(receipt["base_version"]) is not int or receipt["base_version"] != 0
                        or receipt["base_sha256"] != v0.checkpoint_sha256
                        or type(receipt["staleness"]) is not int or receipt["staleness"] != version - 1
                        or type(receipt["alpha"]) not in (float, int) or receipt["alpha"] != 0.5 / version):
                    continue
                linked = bool(parent and parent["integrity"] == "Verified"
                              and parent["version"] == version - 1 == receipt["parent_version"]
                              and parent["sha256"] == receipt["parent_checkpoint_sha256"]
                              and (not accepted or datetime.fromisoformat(stamp) >= datetime.fromisoformat(accepted[-1]["created_at"]))
                              )
                integrity = (linked and hashlib.sha256(checkpoint).hexdigest() == digest
                             and _verified_file(directory / f"{digest}.pt", digest))
                if linked and client not in contributors:
                    contributors.append(client)
                evaluated = evaluation_available and db.execute(
                    """SELECT 1 FROM evaluation_events WHERE event='COMPARISON_COMPLETED'
                       AND v0=? AND v2=? LIMIT 1""", (v0.checkpoint_sha256, digest)).fetchone()
                model = {
                    "version": version, "name": f"Research v{version}", "type": "Research", "created_at": stamp,
                    "source": "FedAsync", "client": client, "chunk": chunk, "contributors": list(contributors) if linked else [client],
                    "status": "EXPERIMENTAL / UNVALIDATED", "integrity": "Verified" if integrity else "Unavailable or mismatch",
                    "sha256": digest, "update_id": job, "parent_version": receipt["parent_version"],
                    "base_version": receipt["base_version"], "alpha": receipt["alpha"], "staleness": receipt["staleness"],
                    "evaluation": ("Client-local comparison recorded · independent validation pending"
                                   if evaluated else "Independent validation pending"),
                    "limitations": list(baseline["limitations"][:-1]) if baseline else [],
                    "architecture": baseline["architecture"] if baseline else None, "contract": baseline["contract"] if baseline else None,
                }
                models.append(model)
                parent = model
                if integrity:
                    accepted.append(model)
                    if client in training and training[client]["digest"] == update_sha:
                        training[client]["checkpoint"] = "Staged · accepted update and resulting model verified"
            except (ValueError, TypeError, KeyError):
                continue
    return models, accepted, training


def registry_snapshot(project: Path, v0: ResearchV0) -> dict:
    baseline = None
    try:
        baseline = _baseline(v0)
    except (OSError, KeyError, ValueError, RuntimeError, TypeError):
        pass
    models, accepted, training, readable = [], [], {}, False
    try:
        models, accepted, training = _research_models(project / "state/continual-real/registry", baseline, v0)
        readable = True
    except (OSError, sqlite3.Error, ValueError):
        pass
    try:
        evidence = json.loads((project / "state/task9e/local-integration.json").read_bytes())
        ewc = (evidence["v0_sha256"] == v0.checkpoint_sha256 and evidence["exact_six_tensor_checks"] is True
               and evidence["c2_used_prior_ewc"] is True)
    except (OSError, ValueError, KeyError, TypeError):
        ewc = False
    return {
        "baseline": baseline, "models": ([baseline] if baseline else []) + models, "aggregations": accepted,
        "registry_readable": readable, "training": {
            client: {k: row[k] for k in ("chunk", "completed_at", "checkpoint")} for client, row in training.items()},
        "dp_sgd_runs": [row["chunk"] for row in training.values() if row["bounded_training"]],
        "ewc_disabled_runs": [row["chunk"] for row in training.values() if row["ewc_disabled"]],
        "ewc_synthetic_verified": ewc,
    }


def _organization_status(project: Path, monitor: dict, receiver) -> dict:
    historical = organization_snapshot(project, monitor)
    model = historical.get("model")
    result = {"agent": "Live telemetry unavailable", "heartbeat_at": None,
              "inference_model": {"version": model["version"], "sha256": model["checkpoint_sha256"],
                                  "scope": "Authorized historical offline inference"} if model else None}
    try:
        grant = receiver.grant() if receiver else None
        if not grant:
            return result
        result["agent"] = "Awaiting live heartbeat"
        if grant["model_sha256"]:
            result["inference_model"] = {"version": 0, "sha256": grant["model_sha256"], "scope": "Current live authorization"}
        with closing(sqlite3.connect(receiver.db.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("""SELECT received,json_extract(body,'$.at'),json_extract(body,'$.state')
                                FROM telemetry WHERE mode='live' AND scope=? AND kind='heartbeat'
                                ORDER BY received DESC,rowid DESC LIMIT 1""", (scope_digest(grant),)).fetchone()
        if row:
            received, sent, state = row
            if state in ("running", "stopped", "error") and type(sent) is int:
                result["heartbeat_at"] = datetime.fromtimestamp(sent, timezone.utc).isoformat()
                result["agent"] = "STALE" if time() - min(received, sent) > FRESH else state.upper()
    except (OSError, ValueError, TypeError, sqlite3.Error, OverflowError):
        result["agent"] = "Live telemetry unavailable"
    return result


def machine_snapshot(project: Path, monitor: dict, receiver=None) -> dict:
    roles = {"Windows": ("FL Server", "Windows FL coordinator", None)}
    roles.update({CLIENT_ROLES[client]: (f"Client {client}", f"Training (Dataset {values[0]})", values[0])
                  for client, values in ASSIGNMENTS.items()})
    roles[Role.ORGANIZATION] = ("Organization", "Network IDS monitoring", None)
    machines = []
    for role, (name, label, dataset) in roles.items():
        sample = monitor["nodes"][role]
        last = sample.get("last_known") or {}
        identity = sample.get("identity") or last.get("identity") or {}
        state = sample["state"]
        error = sample.get("error")
        if state == "UNAVAILABLE":
            state = "UNKNOWN" if role == "Windows" or error == "PROBE_CONFIGURATION_UNAVAILABLE" else "OFFLINE"
        elif state == "PARTIAL":
            state = "ONLINE"
        machines.append({
            "id": str(role), "name": name, "role": label, "dataset": dataset, "status": state,
            "configured_ip": sample["address"], "identity": identity,
            "metrics": sample.get("metrics"), "last_checked": sample.get("checked_at"),
            "last_success": last.get("checked_at"), "error": error,
            "identity_historical": bool(identity and state != "ONLINE"),
            "ssh": ("Not applicable" if role == "Windows" else "Authenticated" if state == "ONLINE"
                    else "Disabled" if not monitor["ssh_enabled"] else state),
        })
    clients = [row for row in machines if row["dataset"]]
    return {"machines": machines, "registered_clients": len(clients),
            "connected_clients": sum(row["status"] == "ONLINE" for row in clients), "refreshing": monitor["refreshing"],
            "manual_cooldown_seconds": monitor["manual_cooldown_seconds"],
            "organization": _organization_status(project, monitor, receiver)}
