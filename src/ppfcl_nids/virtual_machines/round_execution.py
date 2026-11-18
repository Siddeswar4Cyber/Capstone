"""Verify the three configured VMs and optionally run a synthetic SSH FedAvg round."""

import argparse
import hashlib
import io
import json
import shlex
import sys
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from ppfcl_nids.deployment import add_modules

from ppfcl_nids.configuration.settings import Role
from ppfcl_nids.ssh.connections import _read_target, ssh_session

GUESTS = {
    Role.CLIENT_A: ("CLIENT_A", "192.168.252.10"),
    Role.CLIENT_B: ("CLIENT_B", "192.168.252.11"),
    Role.ORGANIZATION: ("ORG", "192.168.252.12"),
}
PACKAGE = ("__init__", "flow_features", "model", "initial_fedavg", "synthetic_round")
INVENTORY = """
import importlib, json, os, platform, shutil, socket, sys
from pathlib import Path
cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
            if s.startswith('model name')), platform.machine())
memory = Path('/proc/meminfo').read_text().splitlines()[0].split()[1]
versions = {}
if sys.argv[1] == 'client':
    for module in ('numpy', 'torch'):
        try:
            versions[module] = importlib.import_module(module).__version__
        except Exception as exc:
            versions[module] = type(exc).__name__
print(json.dumps({'hostname': socket.gethostname(), 'python': platform.python_version(),
                  'executable': sys.executable, 'cpu': cpu, 'vcpus': os.cpu_count(),
                  'memory_kib': int(memory), 'free_bytes': shutil.disk_usage('/tmp').free,
                  'dependencies': versions, 'nprobe_present': shutil.which('nprobe') is not None}))
"""


def command(ssh, script: str) -> str:
    _, stdout, _ = ssh.exec_command(script, timeout=30)
    output = stdout.read(16_384)
    if stdout.channel.recv_exit_status():
        raise RuntimeError("Remote inventory or deployment command failed")
    return output.decode().strip()


def make_package(extra_modules: tuple[str, ...] = ()) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        add_modules(archive, (*PACKAGE, *extra_modules))
    return memory.getvalue()


@contextmanager
def client_package(ssh, package: bytes):
    directory = command(ssh, "mktemp -d /tmp/ppfcl-task5.XXXXXXXX")
    if not directory.startswith("/tmp/ppfcl-task5.") or not directory.rsplit(".", 1)[-1].isalnum():
        raise ValueError("Unexpected remote temporary directory")
    path = directory + "/client.zip"
    with ssh.open_sftp() as sftp:
        try:
            with sftp.open(path, "wb") as remote:
                remote.write(package)
            yield f"PYTHONPATH={shlex.quote(path)} "
        finally:
            try:
                sftp.remove(path)
            finally:
                sftp.rmdir(directory)


def run_client(ssh, interpreter: str, role: str, registry, package: bytes, *,
               module: str = "ppfcl_nids.federated_learning.synthetic_round", module_args: tuple[str, ...] = ()) -> dict:
    from ppfcl_nids.federated_learning.synthetic_round import frame, read_frame

    with client_package(ssh, package) as environment:
        script = environment + shlex.join([interpreter, "-B", "-m", module, *module_args])
        stdin, stdout, _ = ssh.exec_command(script, timeout=60)
        try:
            stdin.write(registry.bootstrap(role))
            stdin.flush()
            acks = []
            for expected in ({"accepted", "duplicate"}, {"duplicate"}):
                ack = registry.submit(role, read_frame(stdout))
                if ack["status"] not in expected:
                    raise RuntimeError("Unexpected update receipt")
                stdin.write(frame(ack))
                stdin.flush()
                acks.append(ack)
            verified = json.loads(read_frame(stdout))
            if not verified.get("verified") or stdout.channel.recv_exit_status():
                raise RuntimeError("Remote client did not verify its receipts")
            return {"role": role, "receipts": acks, "diagnostics": verified.get("diagnostics")}
        finally:
            stdout.channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--python", default="python3", help="Guest interpreter or isolated interpreter wrapper")
    args = parser.parse_args()
    env = Path(".env")
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    inventory = {}
    for role, (prefix, expected) in GUESTS.items():
        try:
            if _read_target(env, prefix).get(f"{prefix}_HOST") != expected:
                raise ValueError("Configured address differs from Task 5")
            with ssh_session(role, env_file=env, known_hosts=known_hosts) as ssh:
                mode = "inventory" if role is Role.ORGANIZATION else "client"
                interpreter = "python3" if mode == "inventory" else args.python
                result = json.loads(command(ssh, f"{shlex.quote(interpreter)} -B -c {shlex.quote(INVENTORY)} {mode}"))
                inventory[role] = result
                print(role, "SSH verified", json.dumps(result), flush=True)
        except Exception as exc:
            # Connection and config helpers emit sanitized diagnostics.
            print(role, "verification failed", type(exc).__name__, flush=True)
    if len(inventory) != len(GUESTS):
        sys.exit(2)
    if not args.run:
        return

    from ppfcl_nids.federated_learning.synthetic_round import SyntheticRound

    for role in (Role.CLIENT_A, Role.CLIENT_B):
        result = inventory[role]
        dependencies = result["dependencies"]
        if (tuple(map(int, result["python"].split(".")[:2])) < (3, 12)
            or any(not version or not version[0].isdigit() for version in dependencies.values())
            or int(dependencies["torch"].split(".")[0]) != 2
            or tuple(map(int, dependencies["torch"].split(".")[:2])) < (2, 5)
            or not (1, 26) <= tuple(map(int, dependencies["numpy"].split(".")[:2])) < (3, 0)):
            parser.exit(3, f"{role} lacks compatible Python, NumPy or PyTorch\n")
    round_id = uuid.uuid4().hex
    directory = Path("state/task5") / round_id
    registry = SyntheticRound(directory, round_id)
    package = make_package()
    print("Synthetic package", len(package), "bytes; SHA-256", hashlib.sha256(package).hexdigest(), flush=True)
    receipts = []
    for role, client in ((Role.CLIENT_A, "A"), (Role.CLIENT_B, "B")):
        with ssh_session(role, env_file=env, known_hosts=known_hosts) as ssh:
            receipt = run_client(ssh, args.python, client, registry, package)
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    digest = receipts[-1]["receipts"][-1]["checkpoint_sha256"]
    if not digest:
        raise RuntimeError("Synthetic round did not produce a checkpoint")
    (directory / "result.json").write_text(json.dumps({"receipts": receipts, "sha256": digest}, indent=2))
    print("Verified synthetic checkpoint", registry.checkpoint, flush=True)


if __name__ == "__main__":
    main()
