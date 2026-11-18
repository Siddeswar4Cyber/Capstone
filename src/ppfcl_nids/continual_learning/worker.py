"""First C1/D1 client-local worker. Only a separate, exact grant can authorize Run."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import struct
import time

import torch

from ppfcl_nids.datasets.continual_partitions import ASSIGNMENTS, load_slots, manifest, owned_chunk
from ppfcl_nids.continual_learning.training import evaluate_local, train_local
from ppfcl_nids.continual_learning.elastic_weight_consolidation import EWC_SPEC
from ppfcl_nids.organization.inference import read_private, write_once
from ppfcl_nids.differential_privacy.training import SmokeConfig

V0 = "415907f3a95c5ee63d930b080f36799f8b8e09166a41ccf0ff5414031e31c29c"
CONFIG = SmokeConfig()
ROOT = Path.home() / ".local/share/ppfcl-nids"
MAX_SECONDS = 900
RAM_RESERVE = 1024**3
DISK_RESERVE = 3 * 1024**3 + 16 * 1024**2


class PreflightError(ValueError):
    pass


def scope(client, chunk=None):
    legacy = chunk is None or chunk == ASSIGNMENTS[client][0] + "1"
    chunk = chunk or ASSIGNMENTS[client][0] + "1"
    if not owned_chunk(client, chunk):
        raise ValueError("Chunk outside client assignment")
    result = {"client": client, "chunk": chunk, "base_version": 0,
            "base_sha256": V0, "contract": EWC_SPEC.metadata(), "schedule": asdict(CONFIG),
            "selection": "first-2048-frozen-ranks", "ewc": False,
            "evaluation": "client-local-first-2048-frozen-validation-ranks"}
    if not legacy:
        result.update(protocol="bounded-continual-v2", attempt_limit=1, maximum_seconds=MAX_SECONDS,
                      source_sha256=ASSIGNMENTS[client][3], manifest_sha256=ASSIGNMENTS[client][4],
                      optimizer="SGD", sampling="Poisson", allow_base_transfer=False)
    return result


def fixture_worker():
    from tempfile import TemporaryDirectory
    from ppfcl_nids.federated_learning.initial_aggregation import InitialFedAvg, InitialUpdate
    from ppfcl_nids.differential_privacy.accounting import PrivacyLedger

    config = SmokeConfig(sample_count=64, batch_size=8, microbatch_size=2, epsilon_limit=100)
    fed = InitialFedAvg(seed=72, contract_id=EWC_SPEC.contract_id)
    x = torch.randn(64, 11, generator=torch.Generator().manual_seed(19))
    y = (x[:, 0] > 0).float()
    with TemporaryDirectory(prefix="ppfcl-continual-fixture-") as temp:
        base = Path(temp) / "v0.pt"
        pin = fed.publish_v0(base, [InitialUpdate(role, fed.base_id, 1, fed.client_model().state_dict(),
                                                 EWC_SPEC.metadata()) for role in ("A", "B")])
        ledger = PrivacyLedger(Path(temp) / "fixture.sqlite", delta=config.delta, epsilon_limit=config.epsilon_limit)
        steps = []
        model, result = train_local(base.read_bytes(), pin, x, y, config, ledger, "fixture",
                                    on_step=lambda done, total: steps.append(done))
        if steps != list(range(1, config.steps + 1)) or not ledger.has_reservation("fixture", **config.mechanism()):
            raise RuntimeError("Fixture accounting or progress mismatch")
        return {"synthetic": True, "steps": result["logical_steps"],
                "support": sum(evaluate_local(model, [(x[:8], y[:8])])["support"])}


def preflight(root, client, chunk=None, *, on_dataset=None):
    from ppfcl_nids.differential_privacy.accounting import PrivacyLedger
    dataset, name, _, source_sha, manifest_sha = ASSIGNMENTS[client]
    ram = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    free = shutil.disk_usage(root).free
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise PreflightError("CPU_RUNTIME_REQUIRED")
    if ram < RAM_RESERVE:
        raise PreflightError("INSUFFICIENT_RAM")
    if free < DISK_RESERVE:
        raise PreflightError("INSUFFICIENT_DISK")
    files = root / "datasets" / name, root / "manifests" / f"chunks-{dataset}.json", root / "privacy/client.sqlite"
    for path in files:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PreflightError("CLIENT_INPUT_PERMISSIONS")
    source, pin, ledger_path = files
    planned = scope(client, chunk)
    try:
        frozen = manifest(pin, client)
        item = next(row for row in frozen["chunks"] if row["id"] == planned["chunk"])
        x, y = load_slots(source, pin, client=client, chunk=planned["chunk"], count=CONFIG.sample_count)
        vx, vy = load_slots(source, pin, client=client, chunk="validation", count=2048)
    except (OSError, ValueError, KeyError):
        raise PreflightError("FROZEN_SELECTION_VERIFICATION_FAILED") from None
    if planned.get("protocol"):
        # Bind the frozen source, rank-index manifest and exact bounds without
        # exporting a fingerprint of transformed private features or labels.
        selection = [source_sha, manifest_sha, planned["chunk"], item["start"], CONFIG.sample_count,
                     frozen["validation"]["start"], 2048]
        slot_hash = hashlib.sha256(json.dumps(selection, separators=(",", ":")).encode()).hexdigest()
    else:
        slot_hash = hashlib.sha256(struct.pack("<II", *x.shape) + x.astype("<f4").tobytes() +
                                   y.astype("<f4").tobytes()).hexdigest()
    if on_dataset:
        on_dataset()
    import sqlite3
    try:
        ledger = PrivacyLedger(ledger_path, delta=CONFIG.delta, epsilon_limit=CONFIG.epsilon_limit, create=False)
        budget = ledger.projection("continual-readonly-preflight", **CONFIG.mechanism())
    except (sqlite3.Error, ValueError):
        raise PreflightError("PRIVACY_LEDGER_VERIFICATION_FAILED") from None
    result = {**planned, "source_sha256": source_sha, "manifest_sha256": manifest_sha,
              "chunk_rows": item["rows"], "slot_sha256": slot_hash, "sample_positive": int(y.sum()),
              "available_ram": ram, "free_disk": free, "vcpus": os.cpu_count(),
              "load_1m": os.getloadavg()[0],
              "training_range": [item["start"], item["start"] + CONFIG.sample_count],
              "validation_range": [frozen["validation"]["start"], frozen["validation"]["start"] + 2048],
              "budget": budget}
    return result, x, y, vx, vy, ledger


def run(root, client, grant, checkpoint, job):
    """Publish only lifecycle counters. The local update stays on the client."""
    expected = {"operation", "job", "scope", "slot_sha256", "expires"}
    planned = grant.get("scope", {})
    chunk = planned.get("chunk")
    extended = planned.get("protocol") == "bounded-continual-v2"
    if extended:
        expected.add("privacy_projection")
    if (set(grant) != expected or grant["operation"] != "local_continual_training" or grant["job"] != job.name
            or not owned_chunk(client, chunk) or planned != scope(client, chunk)
            or type(grant["expires"]) is not int or time.time() >= grant["expires"]
            or hashlib.sha256(checkpoint).hexdigest() != V0):
        raise PermissionError("Exact training grant and checkpoint required")
    job.mkdir(mode=0o700, exist_ok=True)
    if job.is_symlink() or job.stat().st_uid != os.getuid() or job.stat().st_mode & 0o077:
        raise PermissionError("Private job directory required")
    if time.time() >= grant["expires"]:
        raise PermissionError("Grant expired before execution")
    write_once(job / "attempt.json", {"job": job.name, "scope": grant["scope"]})
    started = time.monotonic()
    sequence = 0

    def event(state, **fields):
        nonlocal sequence
        sequence += 1
        write_once(job / f"{sequence:04}.json", {"sequence": sequence, "at": int(time.time()), "job": job.name,
                                                "client": client, "chunk": chunk, "state": state, **fields})

    def stopped():
        return time.monotonic() - started >= MAX_SECONDS or time.time() >= grant["expires"] or (job / "cancel").exists()

    try:
        info, x, y, vx, vy, ledger = preflight(root, client, chunk,
            on_dataset=(lambda: event("DATASET_VERIFIED")) if extended else None)
        if info["slot_sha256"] != grant["slot_sha256"] or not info["budget"]["within_cap"] or stopped():
            raise PermissionError("Slot integrity, deadline or privacy projection failed")
        if extended:
            budget = info["budget"]
            if grant["privacy_projection"] != {"current": budget["current_epsilon"], "projected": budget["projected_epsilon"],
                    "delta": CONFIG.delta, "headroom_after_run": CONFIG.epsilon_limit - budget["projected_epsilon"]}:
                raise PermissionError("Approved privacy projection is stale")
            event("PREFLIGHT_VERIFIED")
        event("RUNNING")
        model, report = train_local(checkpoint, V0, torch.from_numpy(x), torch.from_numpy(y), CONFIG, ledger,
                                    job.name, cancelled=stopped,
                                    on_reserved=(lambda: event("RESERVED")) if extended else None,
                                    expected_projection=grant.get("privacy_projection"),
                                    on_step=lambda n, total: event("STEP", completed=n, total=total))
        stage = job / "update.part"
        with stage.open("xb") as f:
            torch.save({"metadata": EWC_SPEC.metadata(), "state_dict": model.state_dict()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(stage, 0o400)
        final = job / "update.pt"
        os.link(stage, final)
        stage.unlink()
        receipt = job / "evaluation.json"
        write_once(receipt, {"partition": "first-2048-frozen-validation-ranks", "threshold": 0.5,
                                             "update_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                                             "metrics": evaluate_local(model, [(torch.from_numpy(vx), torch.from_numpy(vy))])})
        checkpoint_sha = hashlib.sha256(read_private(final, 0o400, 512 * 1024)).hexdigest()
        evaluation = read_private(receipt, 0o400, 8192)
        if json.loads(evaluation)["update_sha256"] != checkpoint_sha:
            raise ValueError("Checkpoint and evaluation receipt mismatch")
        evidence = {"evaluation_sha256": hashlib.sha256(evaluation).hexdigest()} if extended else {}
        event("STAGED", sha256=checkpoint_sha, steps=report["logical_steps"], **evidence)
    except Exception:
        if sequence or extended:
            event("FAILED", **({"reason": "Worker verification or execution failed"} if extended else {}))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("client", choices=("A", "B"))
    parser.add_argument("mode", choices=("probe", "dashboard-preflight", "run", "fixture"))
    parser.add_argument("--job")
    parser.add_argument("--chunk")
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(MAX_SECONDS)
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS * 2, MAX_SECONDS * 2 + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    if args.mode == "fixture":
        print(json.dumps(fixture_worker(), allow_nan=False))
        return
    if args.mode in ("probe", "dashboard-preflight"):
        try:
            info, *_ = preflight(ROOT, args.client, args.chunk)
        except (PreflightError, OSError, TimeoutError) as exc:
            if args.mode != "dashboard-preflight":
                raise
            print(json.dumps({"blocker": str(exc) if isinstance(exc, PreflightError) else "CLIENT_INPUT_UNAVAILABLE"}))
            return
        if args.mode == "dashboard-preflight":
            budget = info["budget"]
            checkpoint = ROOT / "continual-private" / (V0 + ".pt")
            if checkpoint.exists():
                from ppfcl_nids.federated_learning.initial_aggregation import decode_verified_v0
                try:
                    decode_verified_v0(read_private(checkpoint, 0o400, 1024**2), V0, model_spec=EWC_SPEC)
                except (OSError, ValueError, RuntimeError, TypeError, KeyError):
                    print(json.dumps({"blocker": "MODEL_VERIFICATION_FAILED"}))
                    return
            info = {key: info[key] for key in ("slot_sha256", "source_sha256", "manifest_sha256",
                    "chunk_rows", "training_range", "validation_range", "available_ram", "free_disk", "vcpus")}
            # Budget values depend only on the public mechanism and ledger spend.
            info.update(budget=budget, local_model_verified=checkpoint.exists())
        print(json.dumps(info, allow_nan=False))
        return
    import re
    if not args.job or not re.fullmatch("[0-9a-f]{32}", args.job):
        raise ValueError("Invalid approved job")
    directory = ROOT / "continual-private"
    if directory.is_symlink() or directory.stat().st_mode & 0o077:
        raise PermissionError("Private worker root required")
    grant = json.loads(read_private(directory / (args.job + ".grant.json"), 0o400, 8192))
    checkpoint = read_private(directory / (V0 + ".pt"), 0o400, 1024**2)
    run(ROOT, args.client, grant, checkpoint, directory / args.job)


if __name__ == "__main__":
    main()
