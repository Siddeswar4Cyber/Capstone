"""Pinned SSH transport for read-only continual preflight and approved worker jobs."""
import argparse
import json
from pathlib import Path
import shlex

from ppfcl_nids.configuration.settings import Role
from ppfcl_nids.federated_learning.remote_chunks import RUNTIME
from ppfcl_nids.virtual_machines.round_execution import GUESTS, client_package, make_package
from ppfcl_nids.ssh.connections import ssh_session

MODULES = ("continual_worker", "continual_data", "continual_training", "training_smoke", "private_step",
           "flow_batches", "flow_partitions", "ewc", "organization_inference", "organization_replay",
           "nfstream_adapter")


def return_checkpoint(project, client, digest, body, *, research_version=None):
    """Deliver one approved checkpoint to the pinned client's immutable inventory."""
    import hashlib
    import stat
    from uuid import uuid4
    from ppfcl_nids.virtual_machines.round_execution import command

    if hashlib.sha256(body).hexdigest() != digest or not 0 < len(body) <= 512 * 1024:
        raise ValueError("Invalid return checkpoint")
    role = {"A": Role.CLIENT_A, "B": Role.CLIENT_B}[client]
    with ssh_session(role, env_file=project / ".env", expected_host=GUESTS[role][1], timeout=5) as peer:
        with peer.open_sftp() as sftp:
            sftp.get_channel().settimeout(10)
            directory = sftp.normalize(".") + "/.local/share/ppfcl-nids/continual-private/returned-models"
            command(peer, "umask 077; mkdir -m 700 -- " + shlex.quote(directory) +
                    " 2>/dev/null || test -d " + shlex.quote(directory))
            directory_stat = sftp.lstat(directory)
            if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_IMODE(directory_stat.st_mode) != 0o700:
                raise PermissionError("Private model inventory required")
            name = digest if research_version is None else f"research-only-v{research_version}-{digest}"
            target = directory + "/" + name + ".pt"

            def verify(path):
                attributes = sftp.lstat(path)
                if (not stat.S_ISREG(attributes.st_mode) or stat.S_IMODE(attributes.st_mode) != 0o400
                        or attributes.st_uid != directory_stat.st_uid or attributes.st_size != len(body)):
                    raise ValueError("Unexpected returned model metadata")
                with sftp.file(path, "rb") as remote:
                    if hashlib.sha256(remote.read(len(body) + 1)).hexdigest() != digest:
                        raise ValueError("Returned checkpoint hash mismatch")

            try:
                verify(target)
            except FileNotFoundError:
                staged = target + "." + uuid4().hex
                try:
                    with sftp.file(staged, "x+b") as remote:
                        remote.write(body)
                    sftp.chmod(staged, 0o400)
                    verify(staged)
                    try:
                        sftp.rename(staged, target)
                    except OSError:
                        # Another exact attempt may already have committed these bytes.
                        verify(target)
                    verify(target)
                finally:
                    try:
                        sftp.remove(staged)
                    except FileNotFoundError:
                        pass


def probe(project, client, *, fixture=False, dashboard=False, chunk=None):
    role = {"A": Role.CLIENT_A, "B": Role.CLIENT_B}[client]
    with ssh_session(role, env_file=project / ".env", expected_host=GUESTS[role][1], timeout=5) as peer:
        with client_package(peer, make_package(MODULES)) as env:
            mode = "fixture" if fixture else "dashboard-preflight" if dashboard else "probe"
            args = ["--chunk", chunk] if chunk else []
            command = env + shlex.join([RUNTIME, "-B", "-m", "ppfcl_nids.continual_learning.worker", client, mode, *args])
            if dashboard:
                from ppfcl_nids.dashboard.infrastructure import read_json_command

                return read_json_command(peer.get_transport(), command, 190)
            _, output, _ = peer.exec_command(command, timeout=180)
            data = output.read(16385)
            if output.channel.recv_exit_status() or len(data) > 16384:
                raise RuntimeError("Client preflight failed")
            return json.loads(data)



def worker_action(project, registry, client, job, operation, *, verify_receipt=None, on_stage=None,
                  expected_grant=None, local_export=False):
    """Trust the pinned client channel and exact owner grant."""
    import hashlib
    import re
    from ppfcl_nids.organization.disclosure import local_json, owner_storage
    from ppfcl_nids.continual_learning.worker import scope

    if not re.fullmatch("[0-9a-f]{32}", job) or operation not in ("run", "sync", "submit", "export"):
        raise ValueError("Invalid worker request")
    role = {"A": Role.CLIENT_A, "B": Role.CLIENT_B}[client]
    approvals = registry.directory.parent / "approvals"
    local_grant = registry.local_confirmation(client, job)
    if local_grant is None:
        owner_storage(approvals)
    grant = local_grant if local_grant is not None else local_json(approvals / (job + ".train.json"))
    if operation == "run" and expected_grant is not None and grant != expected_grant:
        raise PermissionError("Owner training grant changed after approval")
    planned = grant.get("scope", {})
    chunk = planned.get("chunk")
    if not chunk or planned != scope(client, chunk) or grant.get("job") != job:
        raise PermissionError("Exact owner training grant required")
    if operation == "run" and planned.get("protocol") and expected_grant is None:
        raise PermissionError("Later chunks require a dashboard dispatch claim")
    fields = {"operation", "job", "scope", "slot_sha256", "expires"}
    if planned.get("protocol"):
        fields.add("privacy_projection")
    if (set(grant) != fields
            or grant["operation"] != "local_continual_training"
            or not isinstance(grant["slot_sha256"], str) or not re.fullmatch("[a-f0-9]{64}", grant["slot_sha256"])):
        raise PermissionError("Malformed owner training grant")
    digest = hashlib.sha256(json.dumps(grant, sort_keys=True).encode()).hexdigest()
    if operation == "run":
        registry.register(job, client, digest, chunk)
    else:
        registry.require_job(job, client, chunk, digest)

    def emit(state, size=None):
        if on_stage is not None:
            on_stage(state, size)

    emit("SSH_CONNECTING")
    with ssh_session(role, env_file=project / ".env", expected_host=GUESTS[role][1], timeout=5) as peer:
        emit("SSH_CONNECTED")
        with peer.open_sftp() as sftp:
            sftp.get_channel().settimeout(10)
            home = sftp.normalize(".")
            root = home + "/.local/share/ppfcl-nids/continual-private"
            path = root + "/" + job

            def read(name, limit):
                with sftp.file(name, "rb") as stream:
                    body = stream.read(limit + 1)
                if len(body) > limit:
                    raise ValueError("Oversized worker artifact")
                return body

            next_sequence = 1

            def sync():
                nonlocal next_sequence
                for seq in range(next_sequence, 38):
                    try:
                        event = json.loads(read(f"{path}/{seq:04}.json", 4096))
                    except FileNotFoundError:
                        break
                    registry.record(client, event)
                    next_sequence = seq + 1

            if operation == "run":
                from ppfcl_nids.virtual_machines.round_execution import command
                import time
                from ppfcl_nids.continual_learning.worker import V0
                if type(grant.get("expires")) is not int or time.time() >= grant["expires"]:
                    raise PermissionError("Training grant expired")
                command(peer, "umask 077; mkdir -p -- " + shlex.quote(root))
                import stat
                meta = sftp.lstat(root)
                if not stat.S_ISDIR(meta.st_mode) or stat.S_IMODE(meta.st_mode) != 0o700:
                    raise PermissionError("Worker directory must be private")
                emit("MODEL_CHECKING")
                for name, body in ((root + "/" + V0 + ".pt", registry.v0.verify().read_bytes()),
                                   (root + "/" + job + ".grant.json", json.dumps(grant).encode())):
                    try:
                        existing = read(name, 1024**2)
                    except FileNotFoundError:
                        if name.endswith(".pt") and planned.get("allow_base_transfer") is False:
                            raise PermissionError("Approved attempt requires verified local Global v0")
                        from uuid import uuid4
                        stage = name + ".part-" + uuid4().hex
                        try:
                            with sftp.file(stage, "x+b") as stream:
                                stream.write(body)
                            sftp.chmod(stage, 0o400)
                            if read(stage, 1024**2) != body:
                                raise ValueError("Guest staging verification failed")
                            sftp.rename(stage, name)
                        finally:
                            try:
                                sftp.remove(stage)
                            except FileNotFoundError:
                                pass
                        if name.endswith(".pt"):
                            emit("MODEL_RECEIVED", len(body))
                    else:
                        if existing != body:
                            raise ValueError("Existing worker artifact differs")
                        if name.endswith(".pt"):
                            emit("MODEL_LOCAL")
                with client_package(peer, make_package(MODULES)) as env:
                    cmd = env + shlex.join([RUNTIME, "-B", "-m", "ppfcl_nids.continual_learning.worker",
                                             client, "run", "--job", job])
                    _, out, _ = peer.exec_command(cmd, timeout=940)
                    emit("WORKER_DISPATCHED")
                    from time import monotonic, sleep
                    deadline = monotonic() + 920
                    while not out.channel.exit_status_ready() and monotonic() < deadline:
                        sync()
                        sleep(2)
                    if not out.channel.exit_status_ready():
                        out.channel.close()
                        raise TimeoutError("Worker outcome unknown; sync status before retry")
                    status = out.channel.recv_exit_status()
                    sync()
                if status:
                    raise RuntimeError("Worker failed; sync durable status before any retry")
            elif operation == "sync":
                sync()
            else:
                if operation == "submit" and planned.get("protocol"):
                    raise PermissionError("Later chunks require distinct export and aggregation approvals")
                release = expected_grant if operation == "export" and local_export else local_json(
                    approvals / (job + ".submit.json"))
                if (release.get("operation") != "submit_continual_update" or release.get("client") != client
                        or release.get("job") != job or release.get("base_sha256") != registry.v0.checkpoint_sha256
                        or release.get("alpha_policy") != "inverse-staleness-0.5-v1"
                        or not isinstance(release.get("sha256"), str)
                        or not re.fullmatch("[0-9a-f]{64}", release["sha256"])
                        or len(release) != 6):
                    raise PermissionError("Explicit update release required before transfer")
                if operation == "export" and release != expected_grant:
                    raise PermissionError("Export confirmation changed")
                from ppfcl_nids.virtual_machines.round_execution import command
                existing = registry.check_export(client, job, release["sha256"])
                if existing is not None:
                    return existing
                if command(peer, "sha256sum -- " + shlex.quote(path + "/update.pt")).split()[0] != release["sha256"]:
                    raise ValueError("Staged guest update differs from release grant")
                if operation == "export":
                    registry.start_export(client, job)
                else:
                    registry.export_event(job, "UPDATE_EXPORT_STARTED")
                artifact = read(path + "/update.pt", 512 * 1024)
                if hashlib.sha256(artifact).hexdigest() != release["sha256"]:
                    raise ValueError("Transferred update digest mismatch")
                if operation == "export":
                    registry.receive(client, job, artifact, release["sha256"])
                    return {"job": job, "operation": operation}
                registry.export_event(job, "UPDATE_EXPORTED", len(artifact))
                registry.export_event(job, "UPDATE_RECEIVED", len(artifact))
                receipt = registry.accept(client, job, artifact, release)
                if verify_receipt is not None:
                    verify_receipt(artifact, receipt)
                return receipt
    return {"job": job, "operation": operation}


def main():
    from ppfcl_nids.federated_learning.model_registry import ContinualRegistry
    from ppfcl_nids.dashboard.server import DASHBOARD_V0
    from ppfcl_nids.organization.disclosure import owner_storage

    parser = argparse.ArgumentParser()
    parser.add_argument("client", choices=("A", "B"))
    parser.add_argument("--operation", choices=("probe", "fixture", "run", "sync", "submit"), default="probe")
    parser.add_argument("--job")
    args = parser.parse_args()
    project = Path.cwd()
    if args.operation in ("probe", "fixture"):
        print(json.dumps(probe(project, args.client, fixture=args.operation == "fixture"), indent=2))
        return
    directory = project / "state/continual-real/registry"
    if not directory.exists():
        directory.mkdir(parents=True)
        owner_storage(directory, create=True)
    owner_storage(directory)
    print(json.dumps(worker_action(project, ContinualRegistry(directory, DASHBOARD_V0),
                                   args.client, args.job or "", args.operation)))


if __name__ == "__main__":
    main()
