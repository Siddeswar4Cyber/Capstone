"""Synthetic-only FedAvg round over an authenticated SSH stdin/stdout channel."""

import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from ppfcl_nids.datasets.feature_contracts import BIFLOW_CONTRACT_ID, FEATURE_CONTRACT_ID, RAW_FIELDS, transform_flows
from ppfcl_nids.federated_learning.initial_aggregation import InitialFedAvg, InitialUpdate, _state_id, load_verified_v0
from ppfcl_nids.models.neural_network import ExtendedFlowMLP

FRAME_LIMIT = 512 * 1024
COUNTS = {"A": 16, "B": 24}
RESEARCH_NONSECURE = "research_nonsecure"


def research_metadata() -> dict:
    return {"training_mode": RESEARCH_NONSECURE, "secure_rng": False, "production_privacy_claim": False}


def frame(message: dict) -> bytes:
    result = json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(result) > FRAME_LIMIT:
        raise ValueError("Oversized round frame")
    return result


def read_frame(stream) -> bytes:
    result = stream.readline(FRAME_LIMIT + 1)
    if isinstance(result, str):
        result = result.encode("utf-8")
    if not result.endswith(b"\n") or len(result) > FRAME_LIMIT:
        raise ValueError("Missing or oversized round frame")
    return result


def wire_state(parameters):
    return {name: value.detach().cpu().tolist() for name, value in parameters.items()}


def tensor_state(values):
    return {name: torch.tensor(value, dtype=torch.float32) for name, value in values.items()}


class InitialRoundRegistry:
    """Durable receipts for an isolated round and channel-authenticated roles."""

    def __init__(self, directory: Path, round_id: str, *, counts: dict[str, int], checkpoint_name: str,
                 research: bool = False, contract_id: str = FEATURE_CONTRACT_ID,
                 artifact_metadata: dict | None = None, seed: int = 2026):
        if set(counts) != {"A", "B"} or any(type(n) is not int or n < 1 for n in counts.values()):
            raise ValueError("Expected fixed positive sample counts for A and B")
        allowed = {"research_nonsecure-smoke.pt"} if research else {"dry-run-v0.pt", "smoke-test.pt"}
        if contract_id == BIFLOW_CONTRACT_ID:
            allowed = {"research_nonsecure-biflow-11-smoke.pt"} if research else set()
        if checkpoint_name not in allowed:
            raise ValueError("Use a separate test checkpoint name")
        self.counts = counts.copy()
        if artifact_metadata is not None and (
            not research or any(artifact_metadata.get(key) != value for key, value in research_metadata().items())
        ):
            raise ValueError("Artifact metadata must preserve research-only labels")
        self.metadata = (artifact_metadata or research_metadata()) if research else None
        self.round_id = round_id
        self.coordinator = InitialFedAvg(seed=seed, contract_id=contract_id)
        self.model_metadata = self.coordinator.spec.metadata() if contract_id != FEATURE_CONTRACT_ID else None
        directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint = directory / checkpoint_name
        self.database = directory / "receipts.sqlite"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS updates (role TEXT PRIMARY KEY, digest TEXT, body BLOB)")
            db.execute("CREATE TABLE IF NOT EXISTS round (id INTEGER PRIMARY KEY, name TEXT, checkpoint_sha TEXT)")
            db.execute("INSERT OR IGNORE INTO round VALUES (1, ?, NULL)", (round_id,))
            if db.execute("SELECT name FROM round").fetchone()[0] != round_id:
                raise ValueError("Synthetic round registry belongs to another round")
            db.execute("CREATE TABLE IF NOT EXISTS round_config (id INTEGER PRIMARY KEY, body TEXT)")
            fields = [self.counts, checkpoint_name, self.coordinator.base_id]
            if self.metadata is not None:
                fields.append(self.metadata)
            if self.model_metadata is not None:
                fields.append(self.model_metadata)
            configuration = json.dumps(fields, sort_keys=True)
            db.execute("INSERT OR IGNORE INTO round_config VALUES (1, ?)", (configuration,))
            if db.execute("SELECT body FROM round_config").fetchone()[0] != configuration:
                raise ValueError("Round configuration differs from stored receipts")

    def bootstrap(self, role: str) -> bytes:
        job = {"round_id": self.round_id, "role": role, "base_id": self.coordinator.base_id,
               "feature_contract": self.coordinator.spec.contract_id,
               "parameters": wire_state(self.coordinator.client_model().state_dict())}
        if self.model_metadata is not None:
            job["model_metadata"] = self.model_metadata
        return frame(job)

    def _parse(self, role: str, body: bytes) -> InitialUpdate:
        if len(body) > FRAME_LIMIT:
            raise ValueError("Oversized model update")
        data = json.loads(body)
        if (
            role not in self.counts or data["role"] != role or data["round_id"] != self.round_id
            or data["update_id"] != f"{self.round_id}:{role}" or data["sample_count"] != self.counts[role]
            or data.get("artifact_metadata") != self.metadata
            or data.get("feature_contract", self.coordinator.spec.contract_id) != self.coordinator.spec.contract_id
        ):
            raise ValueError("Update does not match authenticated role or synthetic round")
        self.coordinator.validate_model_metadata(data.get("model_metadata"))
        update = InitialUpdate(role, data["base_id"], data["sample_count"],
                               tensor_state(data["parameters"]), data.get("model_metadata"))
        self.coordinator.validate_update(update)
        return update

    def submit(self, role: str, body: bytes) -> dict:
        self._parse(role, body)
        digest = hashlib.sha256(body).hexdigest()
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT digest FROM updates WHERE role=?", (role,)).fetchone()
            if prior is not None and prior[0] != digest:
                raise ValueError("Conflicting replay for this client's update")
            if prior is None:
                db.execute("INSERT INTO updates VALUES (?, ?, ?)", (role, digest, body))
            rows = db.execute("SELECT role, body FROM updates ORDER BY role").fetchall()
            checkpoint_sha, = db.execute("SELECT checkpoint_sha FROM round").fetchone()
            if len(rows) == 2 and checkpoint_sha is None:
                updates = [self._parse(client, payload) for client, payload in rows]
                if self.checkpoint.exists():
                    checkpoint_sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
                    payload = load_verified_v0(self.checkpoint, checkpoint_sha, model_spec=self.coordinator.spec)
                    if (payload["parent_initialization_sha256"] != self.coordinator.base_id
                        or payload["client_sample_counts"] != self.counts
                        or payload.get("artifact_metadata") != self.metadata
                        or _state_id(payload["state_dict"]) != _state_id(self.coordinator.aggregate(updates))):
                        raise ValueError("Existing synthetic checkpoint differs from accepted updates")
                else:
                    if self.metadata is None:
                        checkpoint_sha = self.coordinator.publish_v0(self.checkpoint, updates)
                    else:
                        checkpoint_sha = self.coordinator.publish_v0(
                            self.checkpoint, updates, artifact_metadata=self.metadata
                        )
                db.execute("UPDATE round SET checkpoint_sha=?", (checkpoint_sha,))
            if checkpoint_sha is not None:
                load_verified_v0(self.checkpoint, checkpoint_sha, model_spec=self.coordinator.spec)
        receipt = {"status": "duplicate" if prior else "accepted", "update_sha256": digest,
                   "checkpoint_sha256": checkpoint_sha}
        if self.metadata is not None:
            receipt["artifact_metadata"] = self.metadata
        if self.model_metadata is not None:
            receipt["model_metadata"] = self.model_metadata
        return receipt


class SyntheticRound(InitialRoundRegistry):
    def __init__(self, directory: Path, round_id: str):
        super().__init__(directory, round_id, counts=COUNTS, checkpoint_name="dry-run-v0.pt")


def train_synthetic(bootstrap: bytes) -> bytes:
    job = json.loads(bootstrap)
    if job["feature_contract"] != FEATURE_CONTRACT_ID:
        raise ValueError("Synthetic feature contract mismatch")
    role = job["role"]
    count = COUNTS[role]
    torch.set_num_threads(1)
    model = ExtendedFlowMLP()
    parameters = tensor_state(job["parameters"])
    if _state_id(parameters) != job["base_id"]:
        raise ValueError("Bootstrap parameter digest mismatch")
    model.load_state_dict(parameters, strict=True)
    index = np.arange(count)
    labels = ((index + (role == "B")) % 3 == 0).astype(np.float32)
    columns = {name: np.zeros(count, dtype=np.uint32) for name in RAW_FIELDS}
    columns["PROTOCOL"][:] = 6
    columns["IN_BYTES"][:] = 100 + 17 * index + 200 * labels
    columns["IN_PKTS"][:] = 1 + index
    columns["TCP_FLAGS"][:] = 2 + 16 * (1 - labels)
    inputs, targets = torch.from_numpy(transform_flows(columns)), torch.from_numpy(labels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(3):
        optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets).backward()
        optimizer.step()
    return frame({"round_id": job["round_id"], "role": role, "base_id": job["base_id"],
                  "update_id": f"{job['round_id']}:{role}", "sample_count": count,
                  "parameters": wire_state(model.state_dict())})


def exchange_update(update: bytes, diagnostics: dict | None = None):
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    digest = hashlib.sha256(update).hexdigest()
    envelope = json.loads(update)
    for expected in ({"accepted", "duplicate"}, {"duplicate"}):
        sink.write(update)
        sink.flush()
        ack = json.loads(read_frame(source))
        if (ack["status"] not in expected or ack["update_sha256"] != digest
                or ack.get("artifact_metadata") != envelope.get("artifact_metadata")
                or ack.get("model_metadata") != envelope.get("model_metadata")):
            raise ValueError("Server did not acknowledge the update")
    sink.write(frame({"verified": True, "diagnostics": diagnostics}))
    sink.flush()


def main():
    exchange_update(train_synthetic(read_frame(sys.stdin.buffer)))

if __name__ == "__main__":
    main()
