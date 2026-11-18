"""Versioned, bounded Organization telemetry receiver and owner projection."""
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing
from threading import Lock
from time import monotonic, time

import numpy as np

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, get_contract, transform_contract_flows
from ppfcl_nids.organization.behavior import RULES

CONTRACT = get_contract(BIFLOW_CONTRACT_ID)
INGEST = "/api/v1/organization/ingest"
VIEW = "/api/organization/live"
LIMIT = 512
RETENTION = 3600
FRESH = 15
ID = re.compile(r"[0-9a-f]{32}")


def policy(path):
    """A new explicit owner grant is required on both hosts for live publication."""
    with Path(path).open("rb") as stream:
        grant = json.loads(stream.read(4097))
    fields = {"version", "id", "interface", "expires", "publish_features", "publish_ports", "model_sha256",
              "bpf", "publish_alerts", "publish_scores", "burst_threshold"}
    if (not isinstance(grant, dict) or set(grant) != fields or type(grant["version"]) is not int or grant["version"] != 1
            or not isinstance(grant["id"], str) or not ID.fullmatch(grant["id"])
            or grant["interface"] != "ens37" or grant["publish_features"] is not True
            or type(grant["publish_ports"]) is not bool or type(grant["expires"]) is not int
            or any(type(grant[k]) is not bool for k in ("publish_alerts", "publish_scores"))
            or not isinstance(grant["bpf"], str) or not 1 <= len(grant["bpf"].strip()) <= 512
            or type(grant["burst_threshold"]) is not int or not 10 <= grant["burst_threshold"] <= 256
            or grant["publish_scores"] and not grant["model_sha256"]
            or not time() < grant["expires"] <= time() + 86400
            or grant["model_sha256"] not in (None, "415907f3a95c5ee63d930b080f36799f8b8e09166a41ccf0ff5414031e31c29c")):
        raise ValueError("Invalid or expired Organization live grant")
    return grant


def verify_event(event, grant=None):
    common = {"version", "contract", "id", "run", "mode", "at", "kind", "scope"}
    if not isinstance(event, dict) or event.get("version") != 1 or type(event["version"]) is not int:
        raise ValueError("Invalid version")
    if event.get("contract") != CONTRACT.fingerprint or any(
            not isinstance(event.get(k), str) or not ID.fullmatch(event[k]) for k in ("id", "run")):
        raise ValueError("Invalid event identity")
    if type(event.get("at")) is not int or not time() - RETENTION <= event["at"] <= time() + 10:
        raise ValueError("Invalid event timestamp")
    mode = event.get("mode")
    live = mode == "live"
    if live:
        if grant is None or event.get("scope") != scope_digest(grant):
            raise ValueError("Live scope unavailable")
    elif mode != "fixture" or event.get("scope") is not None:
        raise ValueError("Invalid event mode")
    if event.get("kind") == "heartbeat":
        if (set(event) != common | {"state", "dropped", "errors", "interface"}
                or event["state"] not in ("running", "stopped", "error")
                or any(type(event[k]) is not int or not 0 <= event[k] < 2**32 for k in ("dropped", "errors"))
                or event["interface"] != (grant["interface"] if live else None)):
            raise ValueError("Invalid heartbeat")
        return
    if event.get("kind") == "alert":
        fields = {"rule", "severity", "metric", "count", "threshold", "window_ms", "span_ms"}
        if not live or not grant.get("publish_alerts") or set(event) != common | fields:
            raise ValueError("Alert disclosure unavailable")
        rule = event["rule"]
        if not isinstance(rule, str) or rule not in RULES:
            raise ValueError("Unknown behavioral rule")
        threshold, window, metric, severity = RULES[rule]
        if rule == "CONNECTION_BURST":
            threshold = grant["burst_threshold"]
        if ((event["threshold"], event["window_ms"], event["metric"], event["severity"]) !=
                (threshold, window, metric, severity)
                or any(type(event[k]) is not int for k in ("count", "span_ms", "threshold", "window_ms"))
                or not threshold <= event["count"] <= 256 or not 0 <= event["span_ms"] <= window):
            raise ValueError("Invalid behavioral evidence")
        return
    if event.get("kind") != "flow" or set(event) not in (
            common | {"raw", "features", "destination_port", "prediction"},
            common | {"raw", "features", "destination_port", "prediction", "score"}):
        raise ValueError("Invalid flow fields")
    raw, row = event["raw"], event["features"]
    if (not isinstance(raw, list) or len(raw) != 6 or any(
            type(value) is not int or not 0 <= value < 2**width
            for value, (_, width) in zip(raw, CONTRACT.raw_fields, strict=True))
            or raw[0] <= 0 or raw[1] <= 0 or (raw[2] == 0) != (raw[3] == 0)
            or not isinstance(row, list) or len(row) != CONTRACT.width
            or any(type(value) not in (int, float) or not 0 <= value <= 24 for value in row)):
        raise ValueError("Invalid flow values")
    expected = transform_contract_flows(
        {name: np.asarray([value]) for (name, _), value in zip(CONTRACT.raw_fields, raw, strict=True)},
        contract_id=CONTRACT.id)[0]
    if not np.array_equal(expected, np.asarray(row, dtype=np.float32)):
        raise ValueError("Feature values do not match contract")
    port = event["destination_port"]
    if port is not None and (type(port) is not int or not 0 <= port <= 65535
                             or live and not grant["publish_ports"]):
        raise ValueError("Destination port disclosure unavailable")
    prediction = event["prediction"]
    if prediction is not None and (not live or grant["model_sha256"] is None
                                   or prediction not in ("benign", "malicious")):
        raise ValueError("Prediction disclosure unavailable")
    score = event.get("score")
    if score is not None and (not live or not grant.get("publish_scores") or prediction is None
                              or type(score) not in (int, float) or not 0 <= score <= 1
                              or (score >= 0.5) != (prediction == "malicious")):
        raise ValueError("Invalid score disclosure")


def scope_digest(grant):
    return hashlib.sha256(json.dumps(grant, sort_keys=True).encode()).hexdigest()


class FlowReceiver:
    def __init__(self, directory: Path, token: str, grant_path: Path | None = None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{43,128}", token):
            raise ValueError("Expected a dedicated random ingestion credential")
        self.token = token
        self.grant_path = grant_path
        self.db = directory / "telemetry.sqlite"
        self.lock = Lock()
        self.requests = 0
        self.minute = -1
        self.rejected = 0
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id TEXT PRIMARY KEY, mode TEXT NOT NULL, scope TEXT,
                    received REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS telemetry_received ON telemetry(received);
            """)
            self.prune(db)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=3)
        db.execute("PRAGMA secure_delete=ON")
        return db

    def grant(self):
        if self.grant_path:
            try:
                return policy(self.grant_path)
            except (OSError, ValueError, TypeError):
                pass
        return None

    def authenticate(self, headers):
        return (len(headers) == 1 and headers[0].isascii()
                and hmac.compare_digest(headers[0], "Bearer " + self.token))

    def admit(self):
        with self.lock:
            minute = int(monotonic() // 60)
            if minute != self.minute:
                self.minute, self.requests = minute, 0
            self.requests += 1
            return self.requests <= 240

    def reject(self):
        with self.lock:
            self.rejected += 1

    @staticmethod
    def prune(db):
        db.execute("DELETE FROM telemetry WHERE received < ?", (time() - RETENTION,))
        db.execute("""DELETE FROM telemetry WHERE id IN (
            SELECT id FROM telemetry ORDER BY received DESC, rowid DESC LIMIT -1 OFFSET ?)""", (LIMIT,))

    def insert(self, event):
        verify_event(event, self.grant())
        body = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.lock, closing(self.connect()) as db, db:
            seen = db.execute("SELECT body FROM telemetry WHERE id=?", (event["id"],)).fetchone()
            if seen:
                if not hmac.compare_digest(hashlib.sha256(seen[0].encode()).digest(),
                                           hashlib.sha256(body.encode()).digest()):
                    raise ValueError("Event ID reused")
                return False
            db.execute("INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?)",
                       (event["id"], event["mode"], event["scope"], time(), event["kind"], body))
            self.prune(db)
            return True

    def snapshot(self):
        stamp = time()
        grant = self.grant()
        mode = "live" if grant else "fixture"
        scope = scope_digest(grant) if grant else None
        with self.lock, closing(self.connect()) as db, db:
            self.prune(db)
            stored = db.execute("""SELECT received, body FROM telemetry WHERE mode=? AND scope IS ?
                ORDER BY received DESC, rowid DESC""", (mode, scope)).fetchall()
            rejected = self.rejected
        rows = [(received, json.loads(body)) for received, body in stored]
        heartbeat = next(((received, event) for received, event in rows if event["kind"] == "heartbeat"), None)
        flows = [(received, event) for received, event in rows if event["kind"] == "flow"]
        recent = sum(received > stamp - 60 for received, _ in flows)
        state = "OFFLINE"
        if heartbeat:
            received, event = heartbeat
            state = event["state"].upper() if stamp - received <= FRESH and stamp - event["at"] <= FRESH else "STALE"
            if mode == "live" and state == "RUNNING":
                state = "LIVE" if any(event["kind"] in ("flow", "alert")
                                      and event["run"] == heartbeat[1]["run"]
                                      and stamp - event["at"] <= FRESH for _, event in rows) else "WAITING_FOR_TRAFFIC"
        behavior = [{**{k: event[k] for k in ("id", "at", "rule", "severity", "metric", "count",
                                             "threshold", "window_ms", "span_ms")},
                     "source": "BEHAVIORAL_RULE", "status": "OBSERVED"}
                    for _, event in rows if event["kind"] == "alert"]
        ml_alerts = [{"id": event["id"], "at": event["at"], "rule": "MALICIOUS_CLASSIFICATION",
                      "severity": "unrated", "source": "GLOBAL_MODEL", "prediction": "malicious",
                      "score": event.get("score"), "status": "EXPLORATORY"}
                     for _, event in flows if event["prediction"] == "malicious"]
        bins = {}
        for received, event in reversed(flows):
            bucket = int(received // 60 * 60)
            counts = bins.setdefault(bucket, {"at": bucket, "flows": 0, "benign": 0, "malicious": 0})
            counts["flows"] += 1
            if event["prediction"]:
                counts[event["prediction"]] += 1
        return {"mode": mode, "state": state, "heartbeat": heartbeat[1] if heartbeat else None,
                "last_received": rows[0][0] if rows else None, "flow_count": len(flows), "per_minute": recent,
                "rejected": rejected, "feature_names": CONTRACT.feature_names,
                "behavioral_count": len(behavior),
                "alerts": sorted(behavior + ml_alerts, key=lambda row: row["at"], reverse=True)[:30],
                "inference_authorized": bool(grant and grant["model_sha256"]),
                "buckets": list(bins.values()),
                "predictions": {label: sum(event["prediction"] == label for _, event in flows)
                                for label in ("benign", "malicious")},
                "last_prediction": next((event["at"] for _, event in flows if event["prediction"]), None),
                "flows": [{**{key: event[key] for key in
                             ("id", "at", "raw", "features", "prediction")},
                           "destination_port": event["destination_port"] if not grant or grant["publish_ports"] else None}
                          | ({"score": event["score"]} if event.get("score") is not None else {})
                           for _, event in flows[:30]]}
