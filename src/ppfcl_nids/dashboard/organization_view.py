"""Read-only projection of the reviewed Organization release and execution audit."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import stat

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FlowModelSpec

RELEASE = "org-v0-offline-research-11b1-v1"
CHECKPOINT = "415907f3a95c5ee63d930b080f36799f8b8e09166a41ccf0ff5414031e31c29c"
IDENTITY = {"address": "192.168.252.12", "hostname": "ppfcl-organization", "account": "ppfcl"}
FIXTURES = ("controlled.pcap", "additional.pcap")
AUDIT = Path("state/task11b-2") / RELEASE
SOURCES = {
    "manifest": (Path("configuration/organization/research_release_draft.json"), "82bc63b1064eda979be0a6db496c58b087c94dd5600a497808ea6aef89691ff0"),
    "approval": (AUDIT / "approval.json", "b3f5bde47815ae67583c5e700396e5ec96ac7f24fd701b8f582b4903c166fffd"),
    "transfer": (AUDIT / "transfer-receipt.json", "168e9d3e231521d747ea6327dbc32df7f3de3f1bd235801672955b66a643dd8e"),
    "execution": (AUDIT / "inference-receipt.json", "6cca4f434a9684da43eea55cddde988ed67fc5eb774d928e4a01d94444f78f64"),
}
RESTRICTED = "RESTRICTED — SEPARATE DISCLOSURE APPROVAL REQUIRED"
HEALTH_ERRORS = {"SSH_AUTHENTICATION_FAILED", "SSH_CONNECTION_TIMEOUT", "SSH_CONNECTION_OR_HOST_TRUST_FAILED",
                 "SSH_HOST_KEY_MISMATCH", "PROBE_TIMEOUT", "PROBE_OR_ROUTE_FAILED", "MEASUREMENTS_INCOMPLETE"}


def _instant(value: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("Invalid audit timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Timezone required")
    return parsed.astimezone(timezone.utc)


def _hex(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid audit digest")
    return value


def _read(project: Path, path: Path, digest: str) -> dict:
    target = project / path
    # The endpoint accepts no path inputs and never follows paths inside the documents.
    if not stat.S_ISREG(target.lstat().st_mode) or target.resolve() != project.resolve() / path:
        raise ValueError("Unexpected audit file location")
    with target.open("rb") as stream:
        content = stream.read(16_385)
    if len(content) > 16_384 or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Audit digest mismatch")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Expected an audit object")
    return result


def _verified_projection(project: Path, now: datetime) -> dict:
    docs = {name: _read(project, path, digest) for name, (path, digest) in SOURCES.items()}
    manifest, approval, transfer, execution = (docs[name] for name in ("manifest", "approval", "transfer", "execution"))
    hashes = {name: digest for name, (_, digest) in SOURCES.items()}
    dest, artifact = manifest["destination"], manifest["artifact"]
    received = transfer["received"]
    contract = FlowModelSpec(BIFLOW_CONTRACT_ID).metadata()
    if (any(d["release_id"] != RELEASE for d in docs.values()) or manifest["status"] != "PENDING"
            or any(manifest["current_permissions"].values()) or approval["recipient"] != dest
            or (dest["principal"], dest["hostname"], dest["ssh_destination_ipv4"]) !=
            (IDENTITY["account"], IDENTITY["hostname"], IDENTITY["address"])
            or any(d["manifest_sha256"] != hashes["manifest"] for d in (approval, transfer, execution))
            or any(d["approval_sha256"] != hashes["approval"] for d in (transfer, execution))
            or execution["transfer_receipt_sha256"] != hashes["transfer"]
            or {artifact["checkpoint_sha256"], approval["checkpoint_sha256"], received["sha256"],
                execution["checkpoint_sha256"]} != {CHECKPOINT}
            or artifact["version"] != 0 or artifact["training_mode"] != "research_nonsecure"
            or artifact["architecture"] != contract["architecture"] or execution["contract"] != contract
            or artifact["contract_id"] != contract["contract_id"]
            or artifact["contract_sha256"] != contract["contract_fingerprint"]
            or received["contract_sha256"] != contract["contract_fingerprint"]
            or received["mode"] != "0o400" or dest["artifact_mode"] != "0400" or received["verified_tensors"] != 6
            or approval["transfer_limit"] != 1 or transfer["transfer_count"] != 1
            or transfer["remaining_transfers"] != 0 or execution["remaining_transfers"] != 0
            or execution["additional_model_transfers"] != 0 or execution["passes_remaining_per_fixture"] != 0
            or execution["detailed_predictions_exported"] is not False):
        raise ValueError("Inconsistent release evidence")
    issued, expires, transferred, verified = (_instant(value) for value in
        (approval["recorded_utc"], approval["expires_utc"], transfer["verified_utc"], execution["verified_utc"]))
    if (expires - issued != timedelta(days=7) or transfer["expires_utc"] != approval["expires_utc"]
            or not issued <= transferred <= verified < expires):
        raise ValueError("Inconsistent approval window")
    candidates = manifest["proposed_future_scope"]["candidate_pcaps"]
    if len(candidates) != len(FIXTURES) or {p["filename"] for p in candidates} != set(FIXTURES):
        raise ValueError("Wrong fixture allowlist")
    fixture_hashes = {p["filename"]: _hex(p["sha256"]) for p in candidates}
    allowed = [{"filename": name, "sha256": fixture_hashes[name], "inference_limit": 1} for name in FIXTURES]
    if (approval["approved_fixtures"] != allowed or transfer["inference"] != [
        {**row, "passes_consumed": 0, "passes_remaining": 1} for row in allowed
    ] or set(execution["results"]) != set(FIXTURES)):
        raise ValueError("Inconsistent fixture authorization")
    history = []
    for name in FIXTURES:
        row = execution["results"][name]
        completed = _instant(row["completed_utc"])
        _hex(row["result_sha256"])
        if (row["state"] != "VERIFIED" or row["passes_consumed"] != 1 or row["pcap_sha256"] != fixture_hashes[name]
                or not transferred <= completed < expires):
            raise ValueError("Inconsistent fixture receipt")
        history.append({"pcap": name, "pcap_sha256": fixture_hashes[name], "execution_state": "COMPLETED",
                        "completed_at": completed.isoformat(), "result_integrity": "RECORDED_VERIFIED",
                        "checkpoint_sha256": CHECKPOINT, "receipt_sha256": hashes["execution"], "passes_remaining": 0})
    # A revocation marker is a fail-closed override; its contents never enter the API.
    revoked = (project / AUDIT / "revocation.json").exists() or (project / AUDIT / "revocation.json").is_symlink()
    window = "REVOKED" if revoked else "EXPIRED" if now >= expires else "RECORDED"
    return {"evidence_state": "RECORDED_VERIFIED", "model": {
                "version": 0, "checkpoint_sha256": CHECKPOINT, "contract_id": BIFLOW_CONTRACT_ID,
                "contract_fingerprint": contract["contract_fingerprint"], "architecture": contract["architecture"],
                "training_mode": "research_nonsecure", "integrity": "RECORDED_VERIFIED", "checkpoint_mode": "0400",
                "integrity_checked_at": verified.isoformat()},
            "release": {"id": RELEASE, "authorization": "EXHAUSTED", "window": window, "transfers_remaining": 0,
                        "passes_remaining": 0, "approved_at": issued.isoformat(), "expires_at": expires.isoformat(),
                        "transfer_verified_at": transferred.isoformat(), "receipt_verified_at": verified.isoformat()},
            "history": history}


def _health(monitor: dict) -> dict:
    node = monitor.get("nodes", {}).get("organization", {})
    state = node.get("state", "UNKNOWN")
    if node.get("address") != IDENTITY["address"] or state not in {"UNKNOWN", "ONLINE", "PARTIAL", "UNAVAILABLE", "STALE"}:
        return {"state": "UNKNOWN", "checked_at": None, "last_verified_at": None, "reason": None}
    last = node.get("last_known") or {}
    checked = node.get("checked_at")
    try:
        checked = _instant(checked).isoformat() if checked else None
        verified = _instant(last["checked_at"]).isoformat() if last else None
        if state in {"ONLINE", "PARTIAL"} and (not checked or checked != verified):
            state = "UNKNOWN"
    except (TypeError, KeyError, ValueError, OverflowError):
        checked, verified, state = None, None, "UNKNOWN"
    reason = node.get("error")
    return {"state": state, "checked_at": checked, "last_verified_at": verified,
            "reason": reason if isinstance(reason, str) and reason in HEALTH_ERRORS else None}


def organization_snapshot(project: Path, monitor: dict, *, now: datetime | None = None) -> dict:
    """Schema-checked scalars, explicit field allowlist, no remote or artifact reads."""
    try:
        view = _verified_projection(project, now or datetime.now(timezone.utc))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        view = {"evidence_state": "UNAVAILABLE", "model": None, "release": None, "history": []}
    return {"schema": "organization-metadata-v1", "identity": dict(IDENTITY), "health": _health(monitor),
            "research_only": True, "restricted": RESTRICTED, **view}
