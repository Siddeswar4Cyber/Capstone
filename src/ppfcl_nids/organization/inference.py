"""CPU-only inference and write-once research output, with no network inputs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from tempfile import NamedTemporaryFile
from time import perf_counter, time

import numpy as np
import torch

from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0
from ppfcl_nids.models.neural_network import FlowMLP
from ppfcl_nids.organization.feature_adapter import InferenceRequest, infer_logits
from ppfcl_nids.organization.replay import MAX_FLOWS, SPEC, extract_pcap

FIXTURE_NAMES = {"controlled.pcap", "additional.pcap"}
THRESHOLD = 0.5
THRESHOLD_SOURCE = "Exploratory default on sigmoid(logit); not calibrated or validated"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_private(path: Path, mode: int, max_bytes: int) -> bytes:
    """Read one bounded, owned, non-symlink guest file."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != mode or not 0 < info.st_size <= max_bytes):
            raise ValueError("Unexpected file ownership, mode, or size")
        body = stream.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError("File exceeds limit")
    return body


def private_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Expected a private directory owned by the guest account")


def write_once(path: Path, event: dict) -> str:
    """Flush and publish a private file without replacing any previous record."""
    body = (json.dumps(event, allow_nan=False, indent=2) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stage = Path(stream.name)
        try:
            stream.write(body)
            stream.flush()
            if os.name != "nt":
                os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
            stream.close()
            if os.name == "nt":
                stage.rename(path)  # Windows rename refuses to overwrite.
            else:
                os.link(stage, path)
        finally:
            stage.unlink(missing_ok=True)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return digest(body)


def load_cpu_checkpoint(body: bytes, sha256: str, *, contract: dict) -> FlowMLP:
    if contract != SPEC.metadata() or torch.version.cuda is not None or torch.version.hip is not None:
        raise ValueError("CPU build and exact 11-feature contract required")
    payload = decode_verified_v0(body, sha256, model_spec=SPEC)
    model = FlowMLP(SPEC).cpu().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def checked_features(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    if (actual.dtype != np.float32 or actual.ndim != 2 or actual.shape[1] != 11
            or not 0 < len(actual) <= MAX_FLOWS or not np.isfinite(actual).all() or (actual < 0).any()):
        raise ValueError("Empty or invalid flow features")
    if expected.shape != actual.shape or not np.allclose(
        sorted(map(tuple, actual)), sorted(map(tuple, expected)), rtol=1e-7, atol=0,
    ):
        raise ValueError("Extracted vectors differ from the pinned fixture expectations")
    return actual


def predict(model: FlowMLP, features: np.ndarray) -> list[dict]:
    if model.training or any(t.device.type != "cpu" for t in model.parameters()):
        raise ValueError("An evaluation-mode CPU model is required")
    if not 0 < len(features) <= MAX_FLOWS:
        raise ValueError("An empty or oversized flow batch is not eligible")
    logits = infer_logits(model, InferenceRequest(SPEC.metadata(), features))
    if logits.shape != (len(features),) or not torch.isfinite(logits).all():
        raise ValueError("Invalid model output")
    return [{"flow_ordinal": i, "logit": logit, "sigmoid_score": score,
             "exploratory_class": "malicious" if score >= THRESHOLD else "benign"}
            for i, (logit, score) in enumerate(zip(logits.tolist(), logits.sigmoid().tolist(), strict=True), 1)]


def reserve(path: Path, event: dict, expires: float) -> None:
    if time() >= expires:
        raise ValueError("Authorization expired")
    write_once(path, event)


def verify_result(body: bytes, sha256: str) -> None:
    if digest(body) != sha256:
        raise ValueError("Result integrity mismatch")


def execute(root: Path, plan: dict, inputs: Path, expected: dict) -> dict:
    """Validate both inputs before either trained forward call; reserve separately on disk."""
    os.umask(0o077)
    torch.set_num_threads(1)
    if (set(plan["fixtures"]) != FIXTURE_NAMES or time() >= plan["expires"]
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", plan["release_id"])
            or not re.fullmatch(r"[0-9a-f]{32}", plan["dispatch_id"])):
        raise ValueError("Fixture scope, dispatch ID or expiry mismatch")
    checkpoint = Path(plan["model_path"])
    private_directory(root)
    private_directory(checkpoint.parent)
    if checkpoint.parent.parent != root or checkpoint.name != plan["model_sha256"] + ".pt":
        raise ValueError("Unexpected checkpoint path")
    model = load_cpu_checkpoint(read_private(checkpoint, 0o400, plan["model_bytes"]), plan["model_sha256"],
                                contract=plan["contract"])
    output = root / "offline-inference"
    private_directory(output, create=True)
    output /= plan["release_id"]
    private_directory(output, create=True)
    if any(output.iterdir()):
        raise ValueError("Prior or ambiguous inference evidence; do not replay")
    batches = {}
    for name, fixture in plan["fixtures"].items():
        start = perf_counter()
        path = inputs / name
        read_private(path, 0o400, fixture["size_bytes"])
        capture = extract_pcap(path, expected_sha256=fixture["sha256"])
        vectors = np.asarray(expected["fixtures"][name]["transformed_float32"], dtype=np.float32)
        features = checked_features(capture.features, vectors)
        if len(features) != fixture["expected_flows"]:
            raise ValueError("Unexpected fixture flow count")
        batches[name] = features, perf_counter() - start
    receipts = []
    for name, (features, extraction_seconds) in batches.items():
        fixture = plan["fixtures"][name]
        pcap_sha256 = fixture["sha256"]
        if digest(read_private(inputs / name, 0o400, fixture["size_bytes"])) != pcap_sha256:
            raise ValueError("Approved PCAP changed after feature verification")
        attempt = {"release_id": plan["release_id"], "attempt_id": f"{plan['dispatch_id']}-{name}",
                   "approval_sha256": plan["approval_sha256"], "pcap": name, "pcap_sha256": pcap_sha256,
                   "checkpoint_sha256": plan["model_sha256"], "contract": plan["contract"],
                   "started_utc": datetime.now(timezone.utc).isoformat()}
        reserve(output / f"{name}.attempt.json", attempt, plan["expires"])
        start = perf_counter()
        predictions = predict(model, features)
        inference_seconds = perf_counter() - start
        counts = {label: sum(p["exploratory_class"] == label for p in predictions) for label in ("benign", "malicious")}
        result_path = output / f"{name}.predictions.json"
        result_sha256 = write_once(result_path, {**attempt, "threshold": THRESHOLD, "threshold_source": THRESHOLD_SOURCE,
            "valid_flows": len(predictions), "rejected_flows": 0, "extraction_errors": [],
            "prediction_counts": counts, "predictions": predictions, "extraction_seconds": extraction_seconds,
            "inference_seconds": inference_seconds})
        verify_result(read_private(result_path, 0o400, 2 * 1024**2), result_sha256)
        receipt = {"pcap": name, "attempt_id": attempt["attempt_id"], "pcap_sha256": pcap_sha256,
                   "result_sha256": result_sha256, "result_path": str(result_path), "passes_consumed": 1,
                   "extraction_seconds": round(extraction_seconds, 6), "inference_seconds": round(inference_seconds, 6),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_once(output / f"{name}.receipt.json", receipt)
        receipts.append(receipt)
    return {"release_id": plan["release_id"], "checkpoint_sha256": plan["model_sha256"], "passes": receipts}
