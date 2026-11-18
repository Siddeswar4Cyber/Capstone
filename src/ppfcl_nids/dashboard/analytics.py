"""Read-only projection of one dashboard session's persisted evidence."""

from collections import Counter
from datetime import datetime
from math import isfinite


def elapsed_seconds(start, end) -> float | None:
    if not start or not end:
        return None
    try:
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def measured(values) -> dict:
    valid = [value for value in values if type(value) in (int, float) and isfinite(value) and value >= 0]
    if not valid:
        return {"count": 0, "minimum": None, "maximum": None, "average": None}
    return {"count": len(valid), "minimum": min(valid), "maximum": max(valid),
            "average": sum(valid) / len(valid)}


PHASES = {
    "worker_training_seconds": "Worker training, Fisher and staging",
    "run_to_staged_seconds": "Coordinator Run → staged",
    "submission_seconds": "Submitted → accepted",
    "reconciliation_seconds": "Accepted → completed",
    "completion_seconds": "Run → completed",
}

HISTORICAL_PHASES = {
    "training_and_fisher_seconds": "Worker training, Fisher and staging",
    "submit_seconds": "Submit action incl. transfer, validation and commit",
    "fedasync_seconds": "FedAsync validation and commit",
    "promotion_seconds": "Worker EWC promotion",
    "reconcile_seconds": "Reconcile action incl. recovery and acknowledgment",
    "total_seconds": "Select → verified completion",
}


def historical_performance(runs: list[dict]) -> dict:
    return {phase: measured(row.get("timings", {}).get(phase) for row in runs)
            for phase in HISTORICAL_PHASES}


def session_analytics(session, models, chunks, updates, runs, revisions, audit, *,
                      now: str, prolonged_seconds: int) -> dict:
    if session is None:
        return {"source": "No session selected", "updated_at": now, "updates": [], "clients": {},
                "staleness": {}, "runs": [], "revisions": [], "alerts": [], "outcomes": {}, "recovery": [],
                "unverified_receipts": 0, "phases": PHASES, "performance": {}, "restricted": "RESTRICTED"}

    sid = session["session_id"]
    owned = {(row["client_id"], row["chunk_id"]): row for row in chunks}
    by_version = {model["version"]: model for model in models}
    receipts, seen = {}, set()
    for row in updates:
        role, chunk, identity = row["client_id"], row["chunk_id"], row["update_id"]
        version = row["accepted_version"]
        model = by_version.get(version)
        if ((role, chunk) not in owned or (role, identity) in receipts or version in seen
                or model is None or model["integrity"] != "Verified"
                or any(row[key] != model[key] for key in
                       ("client_id", "chunk_id", "base_version", "checkpoint_sha256", "staleness", "aggregation_weight"))):
            continue
        seen.add(version)
        receipts[role, identity] = {**row, "published_at": model["created_at"]}

    scoped_runs, alerts = [], []
    for run in runs:
        role, chunk = run["client_id"], run["chunk_id"]
        if (role, chunk) not in owned:
            continue
        receipt = receipts.get((role, run["execution_id"]))
        if receipt and (receipt["chunk_id"] != chunk or receipt["payload_sha256"] != run["update_sha256"]):
            receipt = None
        accepted = receipt["accepted_at"] if receipt else None
        scoped_runs.append({**run,
            "verified_accepted_version": receipt["accepted_version"] if receipt else None,
            "run_to_staged_seconds": elapsed_seconds(run["started_at"], run["staged_at"]),
            "submission_seconds": elapsed_seconds(run["submitted_at"], accepted),
            "reconciliation_seconds": elapsed_seconds(accepted, run["completed_at"]),
            "completion_seconds": elapsed_seconds(run["started_at"], run["completed_at"]),
        })
        if owned[role, chunk]["execution_id"] != run["execution_id"]:
            continue
        state = run["state"]
        item = {"client": role, "chunk": chunk, "at": run["state_at"]}
        if state in ("FAILED", "ABANDONING", "SUBMITTED"):
            alerts.append({"code": "CHUNK_FAILED" if state == "FAILED" else "CHUNK_INCOMPLETE", **item})
        if state == "ACCEPTED_AWAITING_EWC":
            alerts.append({"code": "EWC_PENDING", **item})
        age = elapsed_seconds(run["started_at"], now)
        if state == "RUNNING" and age is not None and age >= prolonged_seconds:
            alerts.append({"code": "PROLONGED_WORKER", "value": age, "unit": "s", **item})
    for model in models:
        if model["integrity"] != "Verified":
            alerts.append({"code": "CHECKPOINT_VERIFICATION_FAILED", "version": model["version"],
                           "at": model["created_at"]})

    scoped_revisions = []
    for row in revisions:
        receipt = receipts.get((row["client_id"], row["update_id"]))
        if receipt and row["anchor_checkpoint_sha256"] == receipt["payload_sha256"]:
            scoped_revisions.append(row)
    recovery = [row for row in audit if row["session_id"] == sid and
                row["event"].removeprefix("REMOTE ").lower() in ("recover", "abandon", "abandoning", "failed")]
    verified = list(receipts.values())
    outcomes = Counter(run["state"] for run in scoped_runs)
    rejected = {row["execution_id"] for row in audit if row["session_id"] == sid
                and row["event"] == "submit" and row["outcome"].startswith("Rejected")
                and any(run["execution_id"] == row["execution_id"] and run["client_id"] == row.get("client_id")
                        for run in scoped_runs)}
    outcomes["SUBMISSION_REJECTED_RUNS"] = len(rejected)
    outcomes["VERIFIED_ACCEPTANCES"] = len(verified)
    return {
        "source": "Selected session · SQLite transitions, verified checkpoint files and scoped receipts",
        "session_id": sid, "execution_mode": session["execution_mode"], "updated_at": now,
        "updates": verified, "staleness": dict(Counter(row["staleness"] for row in verified)),
        "clients": {role: {"chunk_states": dict(Counter(row["state"] for row in chunks if row["client_id"] == role)),
                           "contributions": sum(row["client_id"] == role for row in verified),
                           "performance": {key: measured(run.get(key) for run in scoped_runs if run["client_id"] == role)
                                           for key in PHASES}}
                    for role in ("A", "B")},
        "runs": sorted(scoped_runs, key=lambda run: run.get("training_order") or float("inf")),
        "revisions": scoped_revisions, "recovery": recovery, "alerts": alerts, "outcomes": dict(outcomes),
        "unverified_receipts": len({(row["client_id"], row["update_id"]) for row in updates} - receipts.keys()),
        "phases": PHASES, "performance": {key: measured(run.get(key) for run in scoped_runs) for key in PHASES},
        "restricted": "RESTRICTED",
    }
