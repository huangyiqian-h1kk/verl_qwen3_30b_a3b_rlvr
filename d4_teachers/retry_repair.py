"""Replace the login-node retry owner without changing any training snapshot."""

import os
from pathlib import Path
import shutil
import signal
import socket
import time
import uuid

from . import controller
from . import guardian_handoff
from .common import (
    active,
    below,
    digest,
    file_sha,
    files,
    is_fenced,
    lock,
    need,
    now,
    read,
    write,
)
from .source import load_campaign
from d4_training_continuation import process_signal
from d4_training_continuation.common import acquire, atomic_text, safe_id
from d4_training_continuation.scheduler import PBS, retryable

REPAIR = "four-domain-retries-v1"
ORIGINAL_SOURCES = {
    "controller.py": "6ea0f1110b3cc73ad16ba1d5f30d3a00ffe6bc46b984900be59db2e637eff3ef",
    "cli.py": "4a41847623814d848ca621afdede506f130095017559806fcba2a59a1f12a329",
}
# Campaigns prepared after the first retry repair froze this published version.
# Accept it as well as the initial and current sources; never edit the template.
PREVIOUS_CONTROLLER_SHA = (
    "03463b5f3e9d01d9d64e30d0bd4210dad683b74f9135384a2235f6577c07076e"
)
PREVIOUS_OBSERVER_SHA = (
    "1280812857ae4f6ae3c26ffc04daba21f790e2b9fb566e9ce86538a29e0fc35c"
)
PREVIOUS_CLI_SHA = "d54d27ded81160c317fb656d6d82b95add93de989226235d3c3d3c7d1d7605da"
ORIGINAL_ENTRY = (
    "from pathlib import Path\nimport sys\n"
    "sys.path.insert(0,str(Path(__file__).resolve().parent/'template/runtime'))\n"
    "from d4_teachers.cli import main\nmain()\n"
)
# This launcher is outside the immutable training contract. Old start/resume
# commands also invoke it, so they cannot silently resurrect the old controller.
ENTRY = """from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
pointer = root / "controller_revision.json"
sys.path.insert(0, str(root / "template/runtime"))
if pointer.exists():
    revision = json.loads(pointer.read_text())
    payload = {k: v for k, v in revision.items() if k != "identity"}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != revision["identity"]:
        raise RuntimeError("Controller revision identity changed")
    campaign = json.loads((root / "campaign.json").read_text())
    if revision["campaign_identity"] != campaign["identity"]:
        raise RuntimeError("Controller belongs to another campaign")
    runtime = root / "controller_revisions" / revision["identity"] / "runtime"
    if runtime.resolve() != runtime or runtime.is_symlink():
        raise RuntimeError("Controller runtime path changed")
    for name, expected in revision["files"].items():
        path = runtime / name
        if path.resolve() != path or runtime not in path.parents:
            raise RuntimeError("Controller source path changed")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Controller source changed: " + str(path))
    sys.path.insert(0, str(runtime))
from d4_teachers.cli import main
main()
"""


def prepare_revision(c):
    root = Path(c["path"])
    entry = root / "control.py"
    need(
        not entry.is_symlink() and entry.read_text() in (ORIGINAL_ENTRY, ENTRY),
        "Unknown campaign launcher; no automatic replacement",
    )
    source = root / "template/runtime/d4_teachers"
    replacements = {
        name: Path(__file__).with_name(name)
        for name in (
            *ORIGINAL_SOURCES,
            "retry_repair.py",
            "source.py",
            "exporting.py",
            "math_test_repair.py",
            "math_and_compat.py",
            "guardian_handoff.py",
            "checkpoint_evaluation.py",
            "runtime.py",
        )
    }
    for name, original in ORIGINAL_SOURCES.items():
        accepted = {original, file_sha(replacements[name])}
        if name == "controller.py":
            accepted.add(
                "dda7bb3735ffa7d1cacce4faad8c4a54ed049ab16da0e193dd0580311d3ff2b5"
            )
            accepted.add(PREVIOUS_CONTROLLER_SHA)
            accepted.add(PREVIOUS_OBSERVER_SHA)
        if name == "cli.py":
            accepted.add(
                "b74043777f8de413a5fdf024905d8ebe11200e4c526c1a49755affede86dc683"
            )
            accepted.add(PREVIOUS_CLI_SHA)
        need(
            file_sha(source / name) in accepted,
            "Unsupported frozen controller source: " + name,
        )
    expected = {
        "d4_teachers/" + name: checksum for name, checksum in files(source).items()
    }
    expected.update(
        {"d4_teachers/" + name: file_sha(path) for name, path in replacements.items()}
    )
    payload = dict(repair=REPAIR, campaign_identity=c["identity"], files=expected)
    revision = dict(payload, identity=digest(payload))
    target = root / "controller_revisions" / revision["identity"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        staging = target.parent / (".staging_" + uuid.uuid4().hex)
        try:
            shutil.copytree(
                source,
                staging / "runtime/d4_teachers",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            for name, path in replacements.items():
                shutil.copy2(path, staging / "runtime/d4_teachers" / name)
            need(files(staging / "runtime") == expected, "Controller snapshot mismatch")
            write(staging / "manifest.json", revision)
            os.rename(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    need(read(target / "manifest.json") == revision, "Controller manifest changed")
    need(files(target / "runtime") == expected, "Controller snapshot changed")
    return revision


def runtime_status(c, revision):
    root = Path(c["path"])
    runtime_file = root / "guardian_runtime.json"
    runtime = read(runtime_file) if runtime_file.exists() else {}
    heartbeat = (
        read(root / "heartbeat.json") if (root / "heartbeat.json").exists() else {}
    )
    expected = (
        root
        / "controller_revisions"
        / revision["identity"]
        / "runtime/d4_teachers/controller.py"
    )
    checksum = revision["files"]["d4_teachers/controller.py"]
    verified = bool(
        active(root / "guardian.lock")
        and runtime.get("controller_file") == str(expected)
        and runtime.get("controller_sha256") == checksum
        and runtime.get("retry_policy") == controller.RETRY_POLICY
        and runtime.get("observation_policy") == controller.OBSERVATION_POLICY
    )
    completed = bool(
        verified
        and heartbeat.get("pid") == runtime.get("pid")
        and heartbeat.get("host") == runtime.get("host")
        and heartbeat.get("controller_sha256") == checksum
        and heartbeat.get("at_utc", "") >= runtime.get("at_utc", "")
    )
    return dict(
        verified=verified,
        first_tick_completed=completed,
        runtime=runtime,
        heartbeat_utc=heartbeat.get("at_utc"),
    )


def _process_identity(pid):
    proc = Path("/proc") / str(pid)
    return proc.stat().st_uid, [
        os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x
    ]


def signal_local_owner(root, target, before_signal=lambda: None):
    """Run on the owner's host, locally or through the legacy SSH helper."""
    need(guardian_handoff.owner(root) == target, "Guardian owner changed")
    record = target["value"]
    host = str(record.get("host", ""))
    need(
        host.split(".")[0] == socket.gethostname().split(".")[0],
        "Guardian signal helper reached the wrong host; no PID will be signalled",
    )
    pid = record.get("pid")
    need(type(pid) is int and pid > 1 and pid != os.getpid(), "Invalid guardian PID")
    process_signal.probe()
    try:
        descriptor = process_signal.open_process(pid)
    except ProcessLookupError:
        descriptor = None
    if descriptor is not None:
        try:
            try:
                uid, argv = _process_identity(pid)
            except FileNotFoundError:
                argv = None
            if argv is not None:
                need(uid == os.getuid(), "Guardian process owner mismatch")
                need(
                    argv[1:]
                    == [
                        "-I",
                        str(root / "control.py"),
                        "guardian",
                        "--campaign",
                        str(root),
                    ],
                    "Guardian command line mismatch; no signal sent",
                )
                before_signal()
                try:
                    process_signal.send_signal(descriptor, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        finally:
            os.close(descriptor)


def stop_and_acquire(root, timeout=45):
    """Hold launch/control locks; never signal a remote PID in the local namespace."""
    try:
        return acquire(root / "guardian.lock")
    except BlockingIOError:
        pass
    target = guardian_handoff.owner(root)
    request, request_lease = guardian_handoff.create(root, target, timeout)
    until = time.monotonic() + timeout
    method, error = "shared_filesystem", None
    try:
        if target["value"].get("handoff_policy") != guardian_handoff.POLICY:
            try:
                if (
                    str(target["value"]["host"]).split(".")[0]
                    == socket.gethostname().split(".")[0]
                ):
                    method = "local_legacy"
                    signal_local_owner(
                        root, target, lambda: guardian_handoff.check(root, request)
                    )
                else:
                    method = "ssh_legacy"
                    guardian_handoff.remote_signal(root, request, timeout)
            except Exception as caught:
                error = caught
        while True:
            try:
                lease = acquire(root / "guardian.lock")
            except BlockingIOError:
                if error:
                    raise RuntimeError(
                        "Guardian lock is still held; no replacement was started. "
                        + str(error)
                    ) from error
                need(
                    time.monotonic() < until,
                    "Guardian handoff timed out; lock remains owned",
                )
                time.sleep(0.1)
                continue
            try:
                guardian_handoff.receipt(
                    root, request, status="LOCK_ACQUIRED", method=method
                )
            except Exception:
                lease.close()
                raise
            return lease
    finally:
        try:
            path = root / guardian_handoff.REQUEST
            if path.exists() and read(path).get("nonce") == request["nonce"]:
                path.unlink()
        finally:
            request_lease.close()


def recover_blocks(c, pbs):
    """Rearm only legacy retry-cap blocks after a fresh, identity-checked PBS read."""
    root, results = Path(c["path"]), {}
    for domain in controller.tasks(c):
        path = controller.state_path(c, domain)
        state = read(path)
        if state["status"] != "BLOCKED_RETRIES":
            continue
        try:
            need(not is_fenced(root, domain), "Explicit stop fence is present")
            need(
                not state.get("job") and not state.get("pending"),
                "Job/intent still owned",
            )
            job = state["previous_job"]
            folder = below(job["folder"], root / "runs" / domain / "jobs")
            request = read(folder / "request.json")
            need(
                digest(request["contract"]) == request["identity"]
                and all(request.get(k) == v for k, v in request["contract"].items()),
                "Previous job contract changed",
            )
            need(
                request["campaign_identity"] == c["identity"]
                and request.get("task", request["domain"]) == domain
                and request["role"] == state["phase"]
                and request["job_dir"] == str(folder)
                and request["job_name"] == job["job_name"],
                "Previous job belongs to another run",
            )
            row = pbs.observe(
                job
            )  # PBS verifies owner, queue, name and exact submit path.
            need(
                retryable(row),
                "Previous job is not a confirmed 271/walltime interruption",
            )
            value = controller.progress(c, domain, state["phase"])
            receipt = (
                root
                / "repairs"
                / REPAIR
                / (domain + "_" + safe_id(job["job_id"]) + ".json")
            )
            if not receipt.exists():
                write(
                    receipt,
                    dict(
                        before=state,
                        observation=row,
                        at_utc=now(),
                        campaign_identity=c["identity"],
                        progress=value,
                    ),
                )
            need(
                not is_fenced(root, domain),
                "Explicit stop fence appeared during observation",
            )
            need(read(path) == state, "Teacher state changed during recovery")
            if state.get("error"):
                state["previous_error"] = dict(
                    message=state.pop("error"), observed_utc=state.get("observed_utc")
                )
            state.update(
                status="READY",
                pbs=row,
                last_progress=value,
                retry_repair=REPAIR,
                recovered_utc=now(),
            )
            write(path, state)
            results[domain] = dict(
                status="REARMED",
                previous_job=job["job_id"],
                Exit_status=row["Exit_status"],
            )
        except Exception as error:
            results[domain] = dict(status="LEFT_BLOCKED", reason=str(error))
    return results


def repair(path, pbs=None, wait_seconds=20, install_under_lease=None):
    from .cli import start

    root = Path(path).resolve()
    with lock(root / "retry_repair.lock"):
        c = load_campaign(root)
        need(
            not is_fenced(root),
            "Campaign stopped; no repair/resume across its stop fence",
        )
        proof = read(root / "preflight.json")
        need(
            proof["status"] == "PASS" and proof["campaign_identity"] == c["identity"],
            "Matching CPU preflight is required",
        )
        revision = prepare_revision(c)
        pointer = root / "controller_revision.json"
        with lock(root / "launch.lock", wait=45):
            with lock(root / "control.lock", wait=45):
                need(not is_fenced(root), "Campaign stopped during repair")
                binding = runtime_status(c, revision)
                current = pointer.exists() and read(pointer) == revision
                restart = install_under_lease is not None or not (
                    current
                    and binding["verified"]
                    and (root / "control.py").read_text() == ENTRY
                )
                lease = stop_and_acquire(root) if restart else None
                install_error = None
                installed = None
                try:
                    # Preserve campaign.json, data, template, checkpoints, jobs and journals.
                    backup = root / "repairs" / REPAIR / "control.before.py"
                    if not backup.exists():
                        atomic_text(backup, (root / "control.py").read_text())
                    atomic_text(root / "control.py", ENTRY)
                    write(pointer, revision)
                    recovered = recover_blocks(c, pbs or PBS())
                    if install_under_lease is not None:
                        try:
                            installed = install_under_lease(c)
                        except Exception as error:
                            # Restore monitoring of other teachers even if the
                            # guarded test repair lost fresh PBS evidence.
                            install_error = error
                    write(
                        root / "repairs" / REPAIR / "latest.json",
                        dict(
                            at_utc=now(),
                            revision=revision["identity"],
                            recovered=recovered,
                            previous_runtime=binding,
                            controller_restart_required=restart,
                            campaign_identity=c["identity"],
                            retry_policy=controller.RETRY_POLICY,
                            observation_policy=controller.OBSERVATION_POLICY,
                            installed=installed,
                            installation_error=str(install_error)
                            if install_error
                            else None,
                        ),
                    )
                finally:
                    if lease is not None:
                        lease.close()
        launch = start(root, True)
        if install_error is not None:
            raise RuntimeError(
                "Test repair failed; guardian restart attempted: " + str(install_error)
            ) from install_error
        until = time.monotonic() + wait_seconds
        while True:
            binding = runtime_status(c, revision)
            if binding["first_tick_completed"] or time.monotonic() >= until:
                break
            time.sleep(0.2)
        states = {d: read(controller.state_path(c, d)) for d in controller.tasks(c)}
        return dict(
            status="REPAIRED_GUARDIAN_RUNNING"
            if binding["first_tick_completed"]
            else "REPAIR_INSTALLED_CHECK_GUARDIAN",
            retry_policy=controller.RETRY_POLICY,
            observation_policy=controller.OBSERVATION_POLICY,
            campaign_identity=c["identity"],
            controller_revision=revision["identity"],
            controller_runtime=binding,
            recovered=recovered,
            launch=launch,
            installed=installed,
            states={
                d: dict(
                    status=s["status"],
                    phase=s["phase"],
                    job_id=(s.get("job") or {}).get("job_id"),
                )
                for d, s in states.items()
            },
        )
