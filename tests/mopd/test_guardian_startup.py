"""Guardian startup must not compete with its own shared-filesystem lease."""
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from d4_mopd import control
from d4_teachers.common import active, read, write
from d4_training_continuation.common import locked


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Child process did not reach the test barrier"
        time.sleep(0.01)


@pytest.fixture
def startup(manifest, tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"],
                                               native_abi={"status": "PASS"}))
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    children = []
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        wait_for(lambda: (tmp_path / "before_lease").exists() or child.poll() is not None)
        return child

    # Run the real guardian and flock acquisition in a separate interpreter.
    # Only asset verification and PBS work are replaced with local test data.
    source = r'''
import sys
sys.path.insert(0, sys.argv[1])
from contextlib import contextmanager
from pathlib import Path
import time
from d4_mopd import control
from d4_teachers.common import read, write
root = Path(sys.argv[2])
control.verify = read
control.PBS = lambda: None

def observed(acquire):
    @contextmanager
    def wrapped(path, *args, **kwargs):
        if Path(path).name == "guardian.lock":
            (root / "before_lease").touch()
            deadline = time.monotonic() + 0.5
            while not (root / "attempt_lease").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            (root / "attempting_lease").touch()
        with acquire(path, *args, **kwargs):
            yield
    return wrapped

control.lock = observed(control.lock)
control.locked = observed(control.locked)

def tick(m, states, pbs):
    (root / "inside_tick").touch()
    while not (root / "finish_tick").exists():
        time.sleep(0.01)
    for state in states.values():
        state["status"] = "BLOCKED"
    write(root / "states.json", states)
    return states

control.tick = tick
control.guardian(root / "manifest.json")
'''
    monkeypatch.setattr(control, "entry_command", lambda _: [sys.executable, "-I", "-c", source,
                                                             str(Path(__file__).resolve().parents[2]), str(tmp_path)])
    monkeypatch.setattr(control.subprocess, "Popen", launch)
    yield SimpleNamespace(manifest=manifest, root=tmp_path, children=children, launch=launch)
    for child in children:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_start_does_not_probe_lease_while_child_acquires_it(startup, monkeypatch):
    root = startup.root
    probes = []

    def slow_probe(path):
        probes.append(path)
        if startup.children:
            # Deterministically reproduce a slow filesystem probe overlapping
            # the child's first flock. Old start() kills its own guardian here.
            with locked(path):
                (root / "attempt_lease").touch()
                wait_for(lambda: (root / "attempting_lease").exists())
                wait_for(lambda: startup.children[0].poll() is not None)
            return False
        return active(path)

    monkeypatch.setattr(control, "active", slow_probe)
    result = control.start(startup.manifest["manifest_path"], "qualification", ("domain_mix", "domain_only"))
    child = startup.children[0]
    assert result["status"] == "REGISTERED"
    assert probes == [root / "guardian.lock"]
    assert active(root / "guardian.lock") and child.poll() is None
    wait_for(lambda: (root / "inside_tick").exists())
    # Startup acknowledges the lease without waiting for a slow PBS operation.
    assert not (root / "heartbeat.json").exists()
    ack = read(root / "guardian_startup.json")
    assert ack["pid"] == child.pid and ack["host"] == socket.gethostname() and ack["token"]
    (root / "finish_tick").touch()
    assert child.wait(timeout=5) == 0
    assert not active(root / "guardian.lock")


def test_guardian_tolerates_an_independent_status_probe_at_startup(startup, monkeypatch):
    root = startup.root

    def launch_during_probe(*args, **kwargs):
        with locked(root / "guardian.lock"):
            child = startup.launch(*args, **kwargs)
            (root / "attempt_lease").touch()
            wait_for(lambda: (root / "attempting_lease").exists())
            time.sleep(0.2)
            assert child.poll() is None, "A transient status probe must not kill the guardian"
            assert not (root / "guardian_startup.json").exists()
        return child

    monkeypatch.setattr(control.subprocess, "Popen", launch_during_probe)
    result = control.start(startup.manifest["manifest_path"], "qualification", ("domain_mix",))
    assert result["status"] == "REGISTERED" and active(root / "guardian.lock")
    (root / "finish_tick").touch()
    assert startup.children[0].wait(timeout=5) == 0


@pytest.mark.parametrize("wrong", ["token", "pid", "host"])
def test_start_rejects_stale_or_other_process_ack(startup, monkeypatch, wrong):
    def launch(*args, **kwargs):
        ack = dict(token=kwargs["env"]["D4_MOPD_GUARDIAN_START_TOKEN"], pid=12345,
                   host=socket.gethostname())
        ack[wrong] = "old-launch" if wrong != "pid" else 54321
        write(startup.root / "guardian_startup.json", ack)
        polls = iter([None, 1])
        return SimpleNamespace(pid=12345, poll=lambda: next(polls))

    monkeypatch.setattr(control.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Guardian exited during startup"):
        control.start(startup.manifest["manifest_path"], "qualification", ("domain_mix",))
    # Registration remains durable for a subsequent operational recovery.
    state = read(startup.root / "states.json")["qualification:domain_mix"]
    assert state["status"] == "READY" and state["job"] is None and state["pending"] is None
