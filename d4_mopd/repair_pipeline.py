"""Detached, one-shot environment repair followed by qualification registration."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid

from . import ARMS
from .assets import verify
from d4_teachers.common import active, age_seconds, files, need, now, read, verify_files, write
from d4_training_continuation.common import acquire


def lock_path(m):
    # Different manifests using this same venv must not replace it concurrently.
    return Path(m["python"]).absolute().parent.parent / "mopd_environment_repair.lock"


@contextmanager
def environment_lease(m, inherited_fd=None):
    if inherited_fd is None:
        handle = acquire(lock_path(m))
    else:
        expected, actual = lock_path(m).stat(), os.fstat(inherited_fd)
        need((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino),
             "Inherited repair lease is not for the manifest environment")
        handle = os.fdopen(os.dup(inherited_fd), "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle.fileno()
    finally:
        # Do not LOCK_UN: the detached worker/build shell may share this lease.
        handle.close()


def status(manifest_path):
    root = Path(manifest_path).parent
    path = root / "repair_pipeline.json"
    if not path.exists():
        return None
    pointer = read(path)
    folder = Path(pointer["folder"])
    need(folder.parent == root / "repair_pipelines", "Invalid repair pipeline directory")
    state = read(folder / "status.json") if (folder / "status.json").exists() else dict(status="STARTING")
    held = active(lock_path(read(manifest_path)))
    result = dict(pointer, **state, environment_lock_held=held, log=str(folder / "pipeline.log"))
    result["heartbeat_stale"] = not state.get("heartbeat_utc") or age_seconds(state["heartbeat_utc"]) > 60
    if state.get("status", "STARTING") in ("STARTING", "RUNNING", "WAITING_PBS") and not held:
        result["status"] = "INTERRUPTED"
    return result


def snapshot(folder):
    repo = Path(__file__).resolve().parents[1]
    code = folder / "code"
    for name in ("d4_mopd", "multidomain", "d4_teachers", "d4_training_continuation", "config/mopd"):
        shutil.copytree(repo / name, code / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "scripts").mkdir()
    for name in ("mopd.py", "mopd_bootstrap.sh"):
        shutil.copy2(repo / "scripts" / name, code / "scripts" / name)
    return files(code)


def worker_command(m, folder, fd):
    return [m["python"], "-u", "-I", str(folder / "code/scripts/mopd.py"),
            "repair-worker", "--manifest", m["manifest_path"],
            "--repair-folder", str(folder), "--lease-fd", str(fd)]


def wait_for_pbs_retry(root, seconds):
    """Back off read-only observations while remaining responsive to STOP."""
    deadline = time.monotonic() + seconds
    while True:
        need(not (root / "STOP").exists(), "Experiment STOP requested while waiting for PBS")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1, remaining))


def start(manifest_path, repair_kind="te"):
    need(repair_kind in ("te", "megatron-import", "ray-temp", "cuda-runtime", "batch-metadata", "pbs-sigterm"), "Unknown environment repair kind")
    m = verify(manifest_path)
    root = Path(manifest_path).parent
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(), "Use the manifest interpreter")
    previous = status(manifest_path)
    if previous and previous["status"] == "COMPLETE" and previous.get("repair_kind", "te") == repair_kind:
        need(previous["identity"] == m["identity"], "Repair pipeline identity changed")
        return dict(status="ALREADY_COMPLETED", repair_pipeline=previous)
    try:
        handle = acquire(lock_path(m))
    except BlockingIOError:
        return dict(status="REPAIR_LOCK_HELD", repair_pipeline=status(manifest_path),
                    note="No new process started; inspect the existing repair before retrying")
    with handle:
        need(not (root / "STOP").exists(), "Experiment is stopped")
        plan = read(root / "pipeline.json")
        need(plan == dict(schema="d4-mopd-pipeline-v1", identity=m["identity"], arms=list(ARMS)),
             "repair-and-retry requires the existing two-arm automatic pipeline")
        state = read(root / "states.json")
        need(set(state) == {"qualification:" + a for a in ARMS}
             and all(s.get("status") == "BLOCKED" and s.get("phase") == "qualification"
                     and s.get("job") and not s.get("pending") for s in state.values()),
             "Repair requires both BLOCKED qualifications and no pending submissions")
        need(not active(root / "guardian.lock"), "Guardian still active; inspect status")
        folder = root / "repair_pipelines" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        code_files = snapshot(folder)
        write(folder / "request.json", dict(manifest=str(manifest_path), identity=m["identity"], repair_kind=repair_kind,
                                            code_files=code_files, at_utc=now()))
        pointer = dict(folder=str(folder), identity=m["identity"], repair_kind=repair_kind)
        write(root / "repair_pipeline.json", pointer)
        try:
            with (folder / "pipeline.log").open("ab") as log:
                child = subprocess.Popen(worker_command(m, folder, handle.fileno()),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(handle.fileno(),), cwd=str(root))
            write(folder / "launch.json", dict(host=socket.gethostname(), pid=child.pid, at_utc=now()))
        except BaseException as exc:
            write(folder / "status.json", dict(status="FAIL", phase="launch", error=str(exc), at_utc=now()))
            raise
        # Bounded startup acknowledgement; never wait for the expensive build.
        deadline = time.monotonic() + 5
        while not (folder / "status.json").exists() and time.monotonic() < deadline:
            need(child.poll() is None, "Repair worker exited during startup; inspect " + str(folder / "pipeline.log"))
            time.sleep(0.05)
    return dict(status="LAUNCHED", host=socket.gethostname(), pid=child.pid,
                repair_pipeline=status(manifest_path),
                note="Detached repair -> CPU preflight -> both qualifications; GPU tests have not passed yet")


def worker(manifest_path, folder, inherited_fd):
    from . import control
    from .preflight import cpu

    folder, root = Path(folder), Path(manifest_path).parent
    request = read(folder / "request.json")
    repair_kind = request.get("repair_kind", "te")
    need(repair_kind in ("te", "megatron-import", "ray-temp", "cuda-runtime", "batch-metadata", "pbs-sigterm"), "Unknown environment repair kind")
    need(request["manifest"] == str(manifest_path), "Repair request manifest mismatch")
    need(folder.parent == root / "repair_pipelines", "Invalid repair worker directory")
    record = dict(status="RUNNING", phase="verify", identity=request["identity"], repair_kind=repair_kind,
                  host=socket.gethostname(), pid=os.getpid(), started_utc=now(),
                  pbs_observation_retries=0, pbs_wait=None)
    mutex, stop = threading.Lock(), threading.Event()

    def report(**changes):
        with mutex:
            record.update(changes, heartbeat_utc=now())
            write(folder / "status.json", record)

    def heartbeat():
        while not stop.wait(5):
            report()

    def on_pbs_unknown(job, exc):
        attempt = record["pbs_observation_retries"] + 1
        delay = 30 if attempt == 1 else 60
        observed = now()
        evidence = dict(job=job, error=str(exc), attempt=attempt, observed_utc=observed,
                        retry_after_seconds=delay,
                        next_retry_utc=(datetime.fromisoformat(observed) + timedelta(seconds=delay)).isoformat())
        report(status="WAITING_PBS", pbs_observation_retries=attempt, pbs_wait=evidence)
        print(f"[{observed}] WAITING_PBS {job['job_id']}: {exc}; recheck in {delay}s", flush=True)
        wait_for_pbs_retry(root, delay)
        report(status="RUNNING", pbs_wait=None)

    # The parent passed an already-locked open file description. Keep its
    # original descriptor open until the entire handoff is finished.
    with os.fdopen(inherited_fd, "a") as inherited:
        report()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            verify_files(folder / "code", request["code_files"])
            m = verify(manifest_path)
            need(m["identity"] == request["identity"], "Repair manifest changed")
            with environment_lease(m, inherited.fileno()) as fd:
                from .launcher_repair import repair as repair_launcher
                from .cuda_repair import repair as repair_cuda
                from .metadata_repair import repair as repair_metadata
                from .sigterm_repair import repair as repair_sigterm
                repair_name, repair_action = {
                    "te": ("repair_te", control.repair_te),
                    "megatron-import": ("repair_megatron_import", control.repair_megatron_import),
                    "ray-temp": ("repair_ray_temp", repair_launcher),
                    "cuda-runtime": ("repair_cuda_runtime", repair_cuda),
                    "batch-metadata": ("repair_batch_metadata", repair_metadata),
                    "pbs-sigterm": ("repair_pbs_sigterm", repair_sigterm),
                }[repair_kind]
                stages = (
                    (repair_name, lambda: repair_action(manifest_path, lease_fd=fd,
                                                        on_pbs_unknown=on_pbs_unknown)),
                    ("preflight", lambda: cpu(manifest_path)),
                    ("retry_qualification", lambda: control.retry_qualification(
                        manifest_path, ARMS, on_pbs_unknown=on_pbs_unknown)),
                )
                # CPU preflight must import the pinned verl checkout.
                sys.path.insert(0, m["verl_src"])
                for phase, action in stages:
                    need(not (root / "STOP").exists(), "Experiment STOP requested; no next stage started")
                    report(phase=phase, phase_started_utc=now())
                    print(f"[{now()}] {phase}: START", flush=True)
                    result = action()
                    need(result.get("status") == ("REGISTERED" if phase == "retry_qualification" else "PASS"),
                         "Stage returned without verified success: " + phase)
                    write(folder / (phase + ".json"), result)
                    print(f"[{now()}] {phase}: PASS", flush=True)
                report(status="COMPLETE", phase="qualification_registered", completed_utc=now(),
                       note="Repair completed; both qualifications registered. GPU results are pending.")
        except BaseException as exc:
            report(status="FAIL", error=str(exc), traceback=traceback.format_exc(), completed_utc=now())
            raise
        finally:
            stop.set()
            thread.join(timeout=6)
