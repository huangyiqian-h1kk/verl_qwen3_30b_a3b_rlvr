"""Real-process coverage of disconnects/leases; native builds are test doubles."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from d4_mopd import ARMS, control, preflight, repair_pipeline as repair
from d4_teachers.common import active, read, write


@pytest.fixture
def repair_case(manifest, tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "pipeline.json", dict(schema="d4-mopd-pipeline-v1",
                                         identity=manifest["identity"], arms=list(ARMS)))
    write(tmp_path / "states.json", {
        "qualification:" + a: dict(phase="qualification", arm=a, status="BLOCKED",
                                  job={"job_id": str(i)}, pending=None)
        for i, a in enumerate(ARMS)
    })
    monkeypatch.setattr(repair, "verify", lambda _: manifest)
    monkeypatch.setattr(repair, "lock_path", lambda _: tmp_path / "environment.lock")
    return manifest


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    pytest.fail("Timed out waiting for child-process evidence")


@pytest.mark.parametrize("repair_kind", ["te", "megatron-import"])
def test_background_repair_survives_client_hangup_and_reuses_active_operation(repair_case, tmp_path, repair_kind):
    m = repair_case
    if repair_kind == "megatron-import":
        old = tmp_path / "repair_pipelines" / "earlier_te"
        write(old / "status.json", dict(status="COMPLETE", identity=m["identity"], repair_kind="te"))
        write(tmp_path / "repair_pipeline.json", dict(folder=str(old), identity=m["identity"]))
    expected_pipelines = 2 if repair_kind == "megatron-import" else 1
    driver = tmp_path / "driver.py"
    repo = Path(__file__).resolve().parents[2]
    driver.write_text(f"import sys\nsys.path.insert(0, {str(repo)!r})\nrepair_kind={repair_kind!r}\n" + r'''
import json, os, time
from pathlib import Path
from d4_mopd import control, preflight, repair_pipeline as repair
from d4_teachers.common import read, write
root = Path(sys.argv[2])
m = read(root / "manifest.json")
repair.verify = lambda _: m
repair.lock_path = lambda _: root / "environment.lock"
if sys.argv[1] == "client":
    repair.worker_command = lambda m, folder, fd: [sys.executable, "-u", "-I", __file__,
                                                 "worker", str(root), str(folder), str(fd)]
    write(root / "launched.json", repair.start(root / "manifest.json", repair_kind=repair_kind))
    while True:
        time.sleep(1)
else:
    def build(path, *, lease_fd):
        assert sys.stdin.read() == ""
        assert os.fstat(lease_fd).st_ino == (root / "environment.lock").stat().st_ino
        write(root / "build_started.json", dict(pid=os.getpid(), sid=os.getsid(0)))
        print("test native build started", flush=True)
        while not (root / "finish_build").exists():
            time.sleep(0.05)
        return dict(status="PASS")
    def forbidden(*args, **kwargs):
        raise RuntimeError("Wrong repair action")
    control.repair_te = build if repair_kind == "te" else forbidden
    control.repair_megatron_import = build if repair_kind == "megatron-import" else forbidden
    preflight.cpu = lambda _: dict(status="PASS")
    def register(path, arms):
        write(root / "registered.json", list(arms))
        return dict(status="REGISTERED")
    control.retry_qualification = register
    repair.worker(root / "manifest.json", Path(sys.argv[3]), int(sys.argv[4]))
''')
    worker_pid = None
    with (tmp_path / "client.log").open("w") as log:
        client = subprocess.Popen([sys.executable, "-I", str(driver), "client", str(tmp_path)],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for(lambda: (tmp_path / "launched.json").exists())
            evidence = wait_for(lambda: read(tmp_path / "build_started.json")
                                if (tmp_path / "build_started.json").exists() else None)
            worker_pid = evidence["pid"]
            assert evidence["sid"] == worker_pid and worker_pid != client.pid
            assert repair.start(tmp_path / "manifest.json", repair_kind=repair_kind)["status"] == "REPAIR_LOCK_HELD"
            assert len(list((tmp_path / "repair_pipelines").iterdir())) == expected_pipelines
            os.killpg(client.pid, signal.SIGHUP)
            assert client.wait(timeout=5) == -signal.SIGHUP
            # Disconnect does not close the shared environment lease or the log.
            assert active(tmp_path / "environment.lock")
            first = repair.status(tmp_path / "manifest.json")["heartbeat_utc"]
            wait_for(lambda: repair.status(tmp_path / "manifest.json")["heartbeat_utc"] != first)
            (tmp_path / "finish_build").touch()
            completed = wait_for(lambda: (s if (s := repair.status(tmp_path / "manifest.json"))["status"]
                                         == "COMPLETE" and not s["environment_lock_held"] else None))
            assert read(tmp_path / "registered.json") == list(ARMS)
            assert "retry_qualification: PASS" in Path(completed["log"]).read_text()
            assert completed["repair_kind"] == repair_kind
            assert repair.start(tmp_path / "manifest.json", repair_kind=repair_kind)["status"] == "ALREADY_COMPLETED"
            assert len(list((tmp_path / "repair_pipelines").iterdir())) == expected_pipelines
        finally:
            if client.poll() is None:
                os.killpg(client.pid, signal.SIGKILL)
                client.wait(timeout=5)
            if worker_pid:
                try:
                    os.killpg(worker_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("failure", ["build", "preflight", "bad_preflight", "stop", "code_changed"])
def test_failed_or_stopped_repair_never_registers_gpu_jobs(repair_case, tmp_path, monkeypatch, failure):
    folder = tmp_path / "repair_pipelines" / "test"
    folder.mkdir(parents=True)
    code = repair.snapshot(folder)
    write(folder / "request.json", dict(manifest=str(tmp_path / "manifest.json"),
                                       identity=repair_case["identity"], code_files=code))
    write(tmp_path / "repair_pipeline.json", dict(folder=str(folder), identity=repair_case["identity"]))
    calls = []

    def build(path, *, lease_fd):
        calls.append("build")
        if failure == "build":
            raise RuntimeError("native build failed")
        if failure == "stop":
            (tmp_path / "STOP").touch()
        return dict(status="PASS")

    def cpu(path):
        calls.append("preflight")
        if failure == "preflight":
            raise RuntimeError("preflight failed")
        return dict(status="FAIL" if failure == "bad_preflight" else "PASS")

    monkeypatch.setattr(control, "repair_te", build)
    monkeypatch.setattr(preflight, "cpu", cpu)
    monkeypatch.setattr(control, "retry_qualification", lambda *a: pytest.fail("GPU registration forbidden"))
    if failure == "code_changed":
        (folder / "code/scripts/mopd_bootstrap.sh").write_text("changed")
    with repair.environment_lease(repair_case) as fd:
        with pytest.raises(RuntimeError):
            repair.worker(tmp_path / "manifest.json", folder, os.dup(fd))
    result = repair.status(tmp_path / "manifest.json")
    assert result["status"] == "FAIL" and not result["environment_lock_held"]
    assert not (folder / "retry_qualification.json").exists()
    assert calls == ([] if failure == "code_changed" else
                     ["build"] if failure in ("build", "stop") else ["build", "preflight"])


def test_build_shell_keeps_environment_lease_after_controller_exit(repair_case, tmp_path):
    # Exercise real FD inheritance, including a compiler shell whose Python
    # controller disappears before writing a result.
    repo = Path(__file__).resolve().parents[2]
    source = f"import sys\nsys.path.insert(0, {str(repo)!r})\n" + r'''
import os, subprocess
from pathlib import Path
from d4_mopd import repair_pipeline as repair
from d4_teachers.common import read, write
root = Path(sys.argv[1])
m = read(root / "manifest.json")
repair.lock_path = lambda _: root / "environment.lock"
with repair.environment_lease(m) as fd:
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                             pass_fds=(fd,), start_new_session=True)
    write(root / "build_child.json", dict(pid=child.pid))
    os._exit(0)
'''
    subprocess.run([sys.executable, "-I", "-c", source, str(tmp_path)], check=True, timeout=10)
    pid = read(tmp_path / "build_child.json")["pid"]
    try:
        assert active(tmp_path / "environment.lock")
        result = repair.start(tmp_path / "manifest.json")
        assert result["status"] == "REPAIR_LOCK_HELD"
        assert not (tmp_path / "repair_pipelines").exists()
    finally:
        os.killpg(pid, signal.SIGTERM)
    wait_for(lambda: not active(tmp_path / "environment.lock"))


@pytest.mark.parametrize("condition", ["stop", "pending", "running", "guardian"])
def test_background_start_refuses_nonquiescent_pipeline(repair_case, tmp_path, monkeypatch, condition):
    if condition == "stop":
        (tmp_path / "STOP").touch()
    elif condition == "guardian":
        original = repair.active
        monkeypatch.setattr(repair, "active", lambda p: True if Path(p).name == "guardian.lock" else original(p))
    else:
        states = read(tmp_path / "states.json")
        if condition == "pending":
            states["qualification:domain_mix"]["pending"] = {"folder": "ambiguous"}
        else:
            states["qualification:domain_mix"]["status"] = "R"
        write(tmp_path / "states.json", states)
    with pytest.raises(RuntimeError):
        repair.start(tmp_path / "manifest.json")
    assert not (tmp_path / "repair_pipelines").exists()


def test_interrupted_worker_status_does_not_report_running(repair_case, tmp_path):
    folder = tmp_path / "repair_pipelines" / "test"
    write(tmp_path / "repair_pipeline.json", dict(folder=str(folder), identity=repair_case["identity"]))
    write(folder / "status.json", dict(status="RUNNING", phase="repair_te"))
    assert repair.status(tmp_path / "manifest.json")["status"] == "INTERRUPTED"
