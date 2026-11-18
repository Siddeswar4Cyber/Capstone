"""Read-only owner dashboard projection from the protected Task 12.11 cache."""
from pathlib import Path

from ppfcl_nids.organization.disclosure import canonical, local_json, owner_storage, sha

CACHE = Path("state/task12-11/hybrid-owner/summary.json")
SOURCE_PINS = {
    "benign_ml": "d7e08a2fca3fbb467dfda7f3b30fbf12b5d8a478e4cec17935f8f5c86ae7aff2",
    "sweep_ml": "522fde771d8b0898649f72efeb9af2f7b6d82fd57c5b06aeeb6518f719171377",
    "behavioral": "d0cdf524a04b1c399c2e9591c52469d1ae05d10c491514e87f615a243499b72c",
}
SUMMARY_PIN = "fa358cc329fa2713c59cb13d72c9bc5adec8ed28b9fd86f98c18fcce5eba6659"


def validate(summary: dict) -> None:
    ml_keys = {"benign_correct", "benign_flows", "scan_detected", "scan_missed", "scan_flows"}
    rule_keys = {"alert_type", "scan_alerts", "benign_alerts", "distinct_refused_ports", "threshold_ports",
                 "window_seconds", "observed_span_ns"}
    if (not isinstance(summary, dict) or set(summary) != {"schema", "sources", "ml", "behavioral"}
            or summary["schema"] != "hybrid-lab-v1" or summary["sources"] != SOURCE_PINS
            or not isinstance(summary["ml"], dict) or set(summary["ml"]) != ml_keys
            or not isinstance(summary["behavioral"], dict) or set(summary["behavioral"]) != rule_keys):
        raise ValueError("Unexpected hybrid fields or provenance")
    ml, rule = summary["ml"], summary["behavioral"]
    if (any(type(v) is not int or v < 0 for v in ml.values())
            or any(type(rule[key]) is not int or rule[key] < 0 for key in rule_keys - {"alert_type"})
            or (ml["benign_flows"], ml["scan_flows"]) != (1, 4)
            or ml["benign_correct"] > ml["benign_flows"]
            or ml["scan_detected"] + ml["scan_missed"] != ml["scan_flows"]
            or rule["alert_type"] != "POSSIBLE_PORT_SCAN"
            or (rule["threshold_ports"], rule["window_seconds"]) != (4, 10)
            or rule["distinct_refused_ports"] < rule["threshold_ports"]
            or rule["observed_span_ns"] > rule["window_seconds"] * 1_000_000_000):
        raise ValueError("Invalid hybrid aggregate")


def snapshot(project: Path) -> dict:
    path = project / CACHE
    if not path.is_file():
        return {"state": "UNAVAILABLE"}
    try:
        owner_storage(path.parent)
        revocation = path.parent / "revocation.json"
        if revocation.exists() or revocation.is_symlink():
            raise ValueError("Owner disclosure revoked")
        data = local_json(path, 4096)
        validate(data)
        if sha(canonical(data)) != SUMMARY_PIN:
            raise ValueError("Hybrid summary digest mismatch")
    except (OSError, ValueError, TypeError):
        return {"state": "UNAVAILABLE"}
    return {"state": "VERIFIED", **data}
