"""Fixed read-only worker probe, executed from memory over authenticated SSH."""

import json
import os
import re
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ADDRESSES = {"A": "192.168.252.10", "B": "192.168.252.11"}


def worker_database(namespace: str, client_id: str) -> Path:
    if client_id not in ADDRESSES or not re.fullmatch(r"[0-9a-f]{32}", namespace):
        raise ValueError("Invalid worker scope")
    root = Path.home() / ".local/share/ppfcl-nids/task10b-synthetic"
    directory = root / namespace / client_id
    for entry in (root, directory.parent, directory, directory / "worker.sqlite"):
        if entry.is_symlink() or (entry.exists() and
                                  (entry.stat().st_uid != os.getuid() or entry.stat().st_mode & 0o077)):
            raise PermissionError("Unsafe synthetic worker storage permissions")
    return directory / "worker.sqlite"


def worker_identity(database: Path) -> str | None:
    if not database.is_file():
        return None
    # These stores use SQLite rollback journaling. immutable also prevents auxiliary files.
    # An in-progress store change can fail this advisory probe; dispatch still rechecks identity.
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
        return db.execute("SELECT worker_id FROM manual_worker_identity WHERE id=1").fetchone()[0]


def resources(path: Path) -> dict:
    import torch

    available = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return {"cpu_only": torch.version.cuda is None, "torch": str(torch.__version__),
            "free_disk_bytes": shutil.disk_usage(path).free, "available_ram_bytes": available}


def main():
    client_id, namespace = sys.argv[1:]
    route = os.environ.get("SSH_CONNECTION", "").split()
    if (client_id not in ADDRESSES or len(route) != 4
            or route[0] != "192.168.252.1" or route[2] != ADDRESSES[client_id]):
        raise SystemExit(2)
    database = worker_database(namespace, client_id)
    result = resources(database.parent if database.parent.is_dir() else Path.home())
    print(json.dumps({"address": route[2], "namespace": namespace, "client_id": client_id,
                      "worker_id": worker_identity(database), **result}, allow_nan=False))


if __name__ == "__main__":
    main()
