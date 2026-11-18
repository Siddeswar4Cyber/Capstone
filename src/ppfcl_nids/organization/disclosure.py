"""Allowlisted pooled disclosure. No model, packet, or training dependencies."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

DISCLOSURE_ID = "org-v0-offline-stats-11f1-v1"
PROPOSAL_SHA = "2a20fe5498514a0f7a7619bd7a5b3e97946fc9db1dc9415c04fe69fc6fa5bb58"
PROPOSAL = Path("configuration/organization/prediction_disclosure_draft.json")
LOCAL = Path("state/task11f-2") / DISCLOSURE_ID
ACCOUNT = "SIDDESWAR_PC\\nimma"
RELEASE = "org-v0-offline-research-11b1-v1"
RECEIPT = "6cca4f434a9684da43eea55cddde988ed67fc5eb774d928e4a01d94444f78f64"
FIELDS = {"schema", "disclosure_id", "release_id", "execution_receipt_sha256", "processed_flow_total"}
THRESHOLD = "Exploratory sigmoid threshold 0.5; not calibrated or validated."


def canonical(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def utc() -> datetime:
    return datetime.now(timezone.utc)


def window(approval: dict) -> tuple[datetime, datetime]:
    start, end = (datetime.fromisoformat(approval[k]) for k in ("approved_utc", "expires_utc"))
    if (start.utcoffset() != timedelta(0) or end - start != timedelta(days=7)
            or approval.get("proposal_sha256") != PROPOSAL_SHA or approval.get("disclosure_id") != DISCLOSURE_ID
            or approval.get("recipient") != ACCOUNT or approval.get("residual_risk_accepted") is not True
            or approval.get("limits") != {"projection_attempts": 1, "reads_per_result": 1, "deliveries": 1}):
        raise ValueError("Invalid disclosure approval")
    return start, end


def valid_payload(data: dict) -> None:
    if (set(data) != FIELDS or data["schema"] != "organization-pooled-flow-count-v1"
            or data["disclosure_id"] != DISCLOSURE_ID or data["release_id"] != RELEASE
            or data["execution_receipt_sha256"] != RECEIPT or type(data["processed_flow_total"]) is not int
            or not 2 <= data["processed_flow_total"] <= 8192):
        raise ValueError("Invalid disclosure payload")


def payload(total: int) -> dict:
    value = {"schema": "organization-pooled-flow-count-v1", "disclosure_id": DISCLOSURE_ID, "release_id": RELEASE,
             "execution_receipt_sha256": RECEIPT, "processed_flow_total": total}
    valid_payload(value)
    return value


def count_from_verified_result(body: bytes, item: dict, contract: dict) -> int:
    if sha(body) != item["sha256"]:
        raise ValueError("Result digest mismatch")
    row = json.loads(body)
    if (row["release_id"] != RELEASE or row["attempt_id"] != item["attempt_id"] or row["pcap"] != item["pcap"]
            or row["pcap_sha256"] != item["pcap_sha256"] or row["approval_sha256"] != item["inference_approval_sha256"]
            or row["checkpoint_sha256"] != item["checkpoint_sha256"] or row["contract"] != contract
            or type(row["valid_flows"]) is not int or not 1 <= row["valid_flows"] <= 4096):
        raise ValueError("Result provenance mismatch")
    return row["valid_flows"]


def exclusive(path: Path, value: dict) -> None:
    """Exclusive durable write. An incomplete record remains a consumed-attempt fence."""
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def local_json(path: Path, limit: int = 16_384) -> dict:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400 or info.st_size > limit:
        raise ValueError("Invalid local metadata file")
    with path.open("rb") as stream:
        if not os.path.samestat(info, os.fstat(stream.fileno())):
            raise ValueError("Local metadata changed")
        body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Local metadata exceeds limit")
    return json.loads(body)


ACL = r"""
$ErrorActionPreference='Stop'
$p=$env:PPFCL_DISCLOSURE_ACL_PATH
$id=[Security.Principal.WindowsIdentity]::GetCurrent()
if ($id.Name -ine 'SIDDESWAR_PC\nimma') { throw 'Wrong Windows identity' }
$s=$id.User
$t=[Security.Principal.SecurityIdentifier]
$item=Get-Item -LiteralPath $p -Force
if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected reparse point' }
if ($env:PPFCL_DISCLOSURE_ACL_ACTION -eq 'create') {
  if (!$item.PSIsContainer -or @(Get-ChildItem -LiteralPath $p -Force).Count) { throw 'Expected empty directory' }
  $acl=[Security.AccessControl.DirectorySecurity]::new()
  $acl.SetOwner($s)
  $acl.SetAccessRuleProtection($true,$false)
  $rule=[Security.AccessControl.FileSystemAccessRule]::new($s,'FullControl','ContainerInherit,ObjectInherit','None','Allow')
  $acl.AddAccessRule($rule)
  [IO.Directory]::SetAccessControl($p,$acl)
}
$check=@($item)
if ($item.PSIsContainer) { $check+=@(Get-ChildItem -LiteralPath $p -Force) }
foreach ($i in $check) {
  if (($i.Attributes -band [IO.FileAttributes]::ReparsePoint) -or ($i.FullName -ne $p -and $i.PSIsContainer)) { throw 'Unsafe disclosure file' }
  if ($i.PSIsContainer) { $a=[IO.Directory]::GetAccessControl($i.FullName) }
  else { $a=[IO.File]::GetAccessControl($i.FullName) }
  if ($a.GetOwner($t).Value -ne $s.Value) { throw 'Wrong ACL owner' }
  $rules=@($a.GetAccessRules($true,$true,$t))
  if (!$rules.Count -or @($rules | Where-Object { $_.IdentityReference.Value -ne $s.Value -or $_.AccessControlType -ne 'Allow' }).Count) { throw 'ACL is not owner only' }
  if ($i.PSIsContainer -and (!$a.AreAccessRulesProtected -or !@($rules | Where-Object { ([int]$_.InheritanceFlags -band 3) -eq 3 }).Count)) { throw 'Missing protected inheritable ACL' }
}
"""


def owner_storage(path: Path, *, create: bool = False) -> None:
    if os.name != "nt" or path.resolve() != path.absolute():
        raise ValueError("Invalid Windows owner storage")
    try:
        subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ACL],
                       env={**os.environ, "PPFCL_DISCLOSURE_ACL_PATH": str(path),
                            "PPFCL_DISCLOSURE_ACL_ACTION": "create" if create else "verify"},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=4,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.SubprocessError:
        raise ValueError("Windows identity or owner-only ACL check failed") from None


def display_summary(project: Path, *, source_revoked: bool = False) -> dict | None:
    """Return only the approved cache projection; never contact the guest from HTTP."""
    directory = project / LOCAL
    cache = directory / "pooled.json"
    if not cache.is_file():
        return None
    try:
        owner_storage(directory)
        approval = local_json(directory / "approval.json")
        issued, expires = window(approval)
        if (source_revoked or utc() >= expires or (directory / "revocation.json").exists()
                or (directory / "revocation.json").is_symlink()):
            cache.unlink(missing_ok=True)  # Owner-only cache, not the audit or source files.
            return None
        proposal_bytes = (project / PROPOSAL).read_bytes()
        if utc() < issued or sha(proposal_bytes) != PROPOSAL_SHA:
            return None
        received = local_json(directory / "receipt.json")
        pooled = local_json(cache)
        valid_payload(pooled)
        approval_sha = sha(canonical(approval))
        payload_sha = sha(canonical(pooled))
        source_hashes = [p["sha256"] for p in json.loads(proposal_bytes)["inference_provenance"]["result_artifacts"]]
        attempt = local_json(directory / "attempt.json")
        delivery = local_json(directory / "delivery-started.json")
        aggregation = local_json(directory / "aggregation.json")
        if (received["outcome"] != "VERIFIED_DELIVERED" or received["approval_sha256"] != approval_sha
                or received["payload_sha256"] != payload_sha or received["deliveries_consumed"] != 1
                or received["recipient"] != ACCOUNT or received["disclosure_id"] != DISCLOSURE_ID
                or received["proposal_sha256"] != PROPOSAL_SHA or received["expires_utc"] != approval["expires_utc"]
                or received["fields"] != sorted(FIELDS) or received["source_hashes"] != source_hashes
                or received["artifacts_verified"] != 2 or received["reads_consumed_per_result"] != 1
                or attempt["recipient"] != ACCOUNT or attempt["approval_sha256"] != approval_sha
                or attempt["source_hashes"] != source_hashes or delivery["payload_sha256"] != payload_sha
                or aggregation["payload_sha256"] != payload_sha or aggregation["artifacts_verified"] != 2
                or aggregation["outcome"] != "VERIFIED_NOT_DELIVERED"
                or any(row["attempt_id"] != approval_sha[:32] for row in (attempt, delivery, aggregation, received))):
            return None
        from ppfcl_nids.dashboard.organization_view import SOURCES

        raw = (project / SOURCES["execution"][0]).read_bytes()
        if sha(raw) != RECEIPT:
            return None
        rows = json.loads(raw)["results"]
        durations = {}
        for name in ("controlled.pcap", "additional.pcap"):
            number = Decimal(str(rows[name]["inference_seconds"]))
            if not number.is_finite() or not 0 <= number <= 60:
                return None
            durations[name] = int((number * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        return {"processed_flow_total": pooled["processed_flow_total"], "inference_duration_ms": durations,
                "threshold_provenance": THRESHOLD, "expires_at": expires.isoformat()}
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return None
