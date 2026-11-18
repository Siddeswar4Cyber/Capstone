"""Synthetic Opacus integration and a persistent, client-wide RDP ledger."""

import math
import sqlite3
from contextlib import closing
from pathlib import Path

import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ppfcl_nids.models.neural_network import ExtendedFlowMLP


class PrivacyLedger:
    """Use one durable file per client across all rounds and demo sessions."""

    def __init__(self, path: Path, *, delta: float, epsilon_limit: float, create: bool = True):
        if not 0 < delta < 1 or not math.isfinite(epsilon_limit) or epsilon_limit <= 0:
            raise ValueError("Invalid privacy policy")
        self.path = path
        if not create:
            with closing(self._connect()) as db:
                if db.execute("SELECT delta, epsilon_limit FROM policy").fetchone() != (delta, epsilon_limit):
                    raise ValueError("Existing privacy ledger policy differs")
                db.execute("SELECT run_id, sample_rate, noise, clip, steps FROM spend LIMIT 0")
            return
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), "
                       "delta REAL NOT NULL, epsilon_limit REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS spend (run_id TEXT PRIMARY KEY, "
                       "sample_rate REAL NOT NULL, noise REAL NOT NULL, clip REAL NOT NULL, steps INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO policy VALUES (1, ?, ?)", (delta, epsilon_limit))
            if db.execute("SELECT delta, epsilon_limit FROM policy").fetchone() != (delta, epsilon_limit):
                raise ValueError("Existing privacy ledger policy differs")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True)

    @staticmethod
    def _accountant(db: sqlite3.Connection) -> RDPAccountant:
        accountant = RDPAccountant()
        # Opacus stores consecutive steps with the same mechanism as history tuples.
        accountant.history = db.execute(
            "SELECT noise, sample_rate, steps FROM spend ORDER BY rowid"
        ).fetchall()
        return accountant

    def epsilon(self) -> float:
        with closing(self._connect()) as db:
            delta, = db.execute("SELECT delta FROM policy").fetchone()
            return float(self._accountant(db).get_epsilon(delta=delta))

    def projection(self, run_id: str, *, sample_rate: float, noise: float, clip: float, steps: int) -> dict:
        """Read one snapshot, including prior spend across contracts and sessions."""
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            delta, cap = db.execute("SELECT delta, epsilon_limit FROM policy").fetchone()
            accountant = self._accountant(db)
            current = float(accountant.get_epsilon(delta=delta))
            count = len(accountant.history)
            row = db.execute(
                "SELECT sample_rate, noise, clip, steps FROM spend WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                accountant.history.append((noise, sample_rate, steps))
            elif row != (sample_rate, noise, clip, steps):
                raise ValueError("Stored privacy reservation differs from the requested mechanism")
            projected = float(accountant.get_epsilon(delta=delta))
        return {"current_epsilon": current, "projected_epsilon": projected, "reservations": count,
                "already_reserved": row is not None, "within_cap": math.isfinite(projected) and projected <= cap}

    def has_reservation(self, run_id: str, *, sample_rate: float, noise: float, clip: float, steps: int) -> bool:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT sample_rate, noise, clip, steps FROM spend WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is not None and row != (sample_rate, noise, clip, steps):
            raise ValueError("Stored privacy reservation differs from the requested mechanism")
        return row is not None

    def reserve(self, run_id: str, *, sample_rate: float, noise: float, clip: float, steps: int,
                expected_projection: dict | None = None) -> float:
        if (
            not run_id or not 0 < sample_rate <= 1 or not math.isfinite(noise) or noise <= 0
            or not math.isfinite(clip) or clip <= 0 or type(steps) is not int or steps <= 0
        ):
            raise ValueError("Invalid DP mechanism parameters")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM spend WHERE run_id=?", (run_id,)).fetchone():
                raise ValueError("DP run already reserved")
            delta, epsilon_limit = db.execute("SELECT delta, epsilon_limit FROM policy").fetchone()
            accountant = self._accountant(db)
            current = float(accountant.get_epsilon(delta=delta)) if expected_projection is not None else None
            accountant.history.append((noise, sample_rate, steps))
            epsilon = float(accountant.get_epsilon(delta=delta))
            if expected_projection is not None and expected_projection != {
                    "current": current, "projected": epsilon, "delta": delta, "headroom_after_run": epsilon_limit - epsilon}:
                raise ValueError("Approved privacy projection changed before reservation")
            if not math.isfinite(epsilon) or epsilon > epsilon_limit:
                raise ValueError("Privacy budget exceeded")
            db.execute("INSERT INTO spend VALUES (?, ?, ?, ?, ?)", (run_id, sample_rate, noise, clip, steps))
        return epsilon


def synthetic_dp_step(
    model: ExtendedFlowMLP, features: Tensor, targets: Tensor, ledger: PrivacyLedger,
    *, run_id: str, noise: float, clip: float, learning_rate: float,
) -> None:
    """Exercise one full-population DP-SGD step at q=1 on synthetic tensors."""
    if not len(features) or features.ndim != 2 or targets.shape != (len(features),):
        raise ValueError("Expected a nonempty synthetic feature batch and binary targets")
    model.train()
    ModuleValidator.validate(model, strict=True)
    engine = PrivacyEngine(accountant="rdp")
    loader = DataLoader(TensorDataset(features, targets), batch_size=len(features))
    private, optimizer, batches = engine.make_private(
        module=model, optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
        data_loader=loader, noise_multiplier=noise, max_grad_norm=clip, loss_reduction="sum",
    )
    try:
        if batches.sample_rate != 1.0:
            raise RuntimeError("Synthetic harness requires full-population sampling")
        ledger.reserve(run_id, sample_rate=batches.sample_rate, noise=noise, clip=clip, steps=1)
        inputs, labels = next(iter(batches))
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(private(inputs), labels, reduction="sum")
        loss.backward()
        optimizer.step()
    finally:
        private.to_standard_module()
