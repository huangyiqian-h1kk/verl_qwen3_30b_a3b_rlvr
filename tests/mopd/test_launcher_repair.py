"""Pinned Ray path validation, real shell processes, and the frozen launcher."""
import ast
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from d4_mopd import ARMS, control, launcher_repair as repair, preflight
from d4_mopd import megatron_repair
from d4_teachers.common import files, read, write
from d4_training_continuation.scheduler import Unknown

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/launcher_v3"


@pytest.fixture
def frozen(manifest, tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    shutil.copytree(ROOT / "d4_mopd", runtime / "d4_mopd",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("node.sh", "control.py"):
        shutil.copy2(FIXTURE / name, runtime / "d4_mopd" / name)
    # The deployed driver wrote FAIL before shutdown, but captured no exception
    # attributes. Keep that historical catch block when newer CLI code is copied.
    cli = runtime / "d4_mopd/cli.py"
    from d4_mopd.sigterm_repair import NEW_FAILURE, OLD_FAILURE
    cli.write_text(cli.read_text().replace("from .job_exit_v2 import failure_record\n", "")
                   .replace(NEW_FAILURE, OLD_FAILURE).replace("import time\n", "import time\nimport traceback\n"))
    for name in ("job_exit_v2.py", "sigterm_repair.py", "cpu_replay.sh", "cpu_replay.py"):
        (runtime / "d4_mopd" / name).unlink()
    manifest["runtime_files"] = files(runtime)
    write(tmp_path / "manifest.json", manifest)
    state = {}
    for i, arm in enumerate(ARMS):
        folder = tmp_path / "qualification" / arm / "attempts" / "failed"
        folder.mkdir(parents=True)
        (folder / "ray_start_0.log").write_text(
            "OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes: /local/job/long\n")
        state["qualification:" + arm] = dict(phase="qualification", arm=arm, status="BLOCKED", pending=None,
            job=dict(folder=str(folder), job_name="0390_mp_q" + arm, job_id=f"{10+i}.pbs1"))
    write(tmp_path / "states.json", state)
    write(tmp_path / "pipeline.json", dict(schema="d4-mopd-pipeline-v1", identity=manifest["identity"], arms=list(ARMS)))
    monkeypatch.setattr(repair, "verify", lambda _: manifest)
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    from d4_mopd import repair_pipeline
    monkeypatch.setattr(repair_pipeline, "lock_path", lambda _: tmp_path / "environment.lock")
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    return manifest


def test_repair_keeps_original_and_only_changes_two_launcher_files(frozen, tmp_path):
    original_manifest = (tmp_path / "manifest.json").read_bytes()
    original_state = read(tmp_path / "states.json")
    result = repair.repair(frozen["manifest_path"])
    runtime = repair.active_runtime(frozen)
    inventory = files(runtime)
    assert {p for p in inventory if inventory[p] != frozen["runtime_files"][p]} == {
        "d4_mopd/node.sh", "d4_mopd/control.py"}
    assert files(tmp_path / "runtime") == frozen["runtime_files"]
    assert (tmp_path / "manifest.json").read_bytes() == original_manifest
    assert read(tmp_path / "states.json") == original_state
    assert str(runtime) in control.entry_command(frozen)[-1]
    assert repair.repair(frozen["manifest_path"])["revision"] == result["revision"]
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == 1
    # The reviewed short-path block is shared by fresh and repaired campaigns.
    assert repair.NEW_TMP in (ROOT / "d4_mopd/node.sh").read_text()
    assert repair.OLD_TMP not in (runtime / "d4_mopd/node.sh").read_text()


@pytest.mark.parametrize("change", ["running", "pending", "checkpoint", "receipt", "learner", "wrong_error", "stop", "layout"])
def test_repair_refuses_progress_or_unrelated_failure(frozen, tmp_path, monkeypatch, change):
    state = read(tmp_path / "states.json")
    one = state["qualification:domain_mix"]
    run = tmp_path / "qualification/domain_mix"
    if change == "running":
        monkeypatch.setattr(control, "PBS", lambda: type("PBS", (), {"observe": lambda *a: {"job_state": "R"}})())
    elif change == "pending":
        one["pending"] = {"folder": "ambiguous"}
        write(tmp_path / "states.json", state)
    elif change == "checkpoint":
        (run / "checkpoints/global_step_4").mkdir(parents=True)
    elif change == "receipt":
        write(run / "receipt.json", {"status": "PASS"})
    elif change == "learner":
        write(Path(one["job"]["folder"]) / "result.json", {"status": "RUNNING"})
    elif change == "wrong_error":
        (Path(one["job"]["folder"]) / "ray_start_0.log").write_text("Different error")
    elif change == "stop":
        (tmp_path / "STOP").touch()
    elif change == "layout":
        p = tmp_path / "runtime/d4_mopd/node.sh"
        p.write_text(p.read_text().replace(repair.OLD_TMP, "export RAY_TMPDIR=/unknown\n"))
        frozen["runtime_files"] = files(tmp_path / "runtime")
    before = read(tmp_path / "states.json")
    with pytest.raises(RuntimeError):
        repair.repair(frozen["manifest_path"])
    assert not (tmp_path / "launcher_repair.json").exists()
    assert read(tmp_path / "states.json") == before


def test_pbs_unknown_waits_without_installing_snapshot(frozen, tmp_path, monkeypatch):
    calls, waits = [], []
    class PBS:
        def observe(self, job):
            calls.append(job["job_id"])
            if len(calls) == 2:
                raise Unknown("qstat timed out")
            return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    def wait(job, exc):
        assert not (tmp_path / "launcher_repair.json").exists()
        assert not (tmp_path / "launcher_revisions").exists()
        waits.append(str(exc))
    repair.repair(frozen["manifest_path"], on_pbs_unknown=wait)
    assert calls == ["10.pbs1", "11.pbs1", "10.pbs1", "11.pbs1"]
    assert waits == ["qstat timed out"]


@pytest.mark.parametrize("change", ["original", "repaired", "addition", "pointer", "receipt"])
def test_changed_snapshot_cannot_start_guardian(frozen, tmp_path, change):
    repair.repair(frozen["manifest_path"])
    runtime = repair.active_runtime(frozen)
    if change == "original":
        (tmp_path / "runtime/d4_mopd/node.sh").write_text("changed")
    elif change == "repaired":
        (runtime / "d4_mopd/node.sh").write_text("changed")
    elif change == "addition":
        (runtime / "unreviewed.py").write_text("changed")
    elif change == "pointer":
        pointer = read(tmp_path / "launcher_repair.json")
        pointer["folder"] = "/other/path"
        write(tmp_path / "launcher_repair.json", pointer)
    elif change == "receipt":
        write(runtime.parent / "receipt.json", {"status": "FAIL"})
    with pytest.raises(RuntimeError):
        control.entry_command(frozen)


def test_repaired_frozen_controller_uses_fixed_shell_for_every_stage(frozen, tmp_path):
    repair.repair(frozen["manifest_path"])
    runtime = repair.active_runtime(frozen)
    spec = importlib.util.spec_from_file_location("d4_mopd._repaired_test", runtime / "d4_mopd/control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert str(runtime) in module.entry_command(frozen)[-1]
    for arm in ARMS:
        write(tmp_path / "runs" / arm / "receipt.json", dict(status="PASS", identity=frozen["identity"],
                                                            selection={"step": 10}))
        for phase in ("qualification", "train", "test"):
            job = module.new_attempt(frozen, phase, arm)
            folder = Path(job["folder"])
            assert str(runtime / "d4_mopd/node.sh") in (folder / "stage.pbs").read_text()
            entry = (folder / "entry.py").read_text()
            assert str(tmp_path / "runtime") in entry
            assert str(runtime) not in entry  # All GPU Python uses the unchanged learner.
    state = read(tmp_path / "states.json")
    module.advance_pipeline(frozen, state)
    assert not any(s["phase"] == "train" for s in state.values())
    for arm in ARMS:
        state["qualification:" + arm]["status"] = "DONE"
        write(tmp_path / "qualification" / arm / "receipt.json", dict(status="PASS", identity=frozen["identity"],
              arm=arm, scope="qualification", step=8, resumed_from=4))
        module.advance_pipeline(frozen, state)
        assert any(s["phase"] == "train" for s in state.values()) == (arm == ARMS[-1])
    assert set(state) == {phase + ":" + arm for phase in ("qualification", "train") for arm in ARMS}


def test_retry_requires_preflight_for_selected_launcher(frozen, tmp_path, monkeypatch):
    repair.repair(frozen["manifest_path"])
    runtime = repair.active_runtime(frozen)
    abi = dict(status="PASS", files=[], incompatible=[])
    guard = dict(status="PASS")
    proof = dict(status="PASS", identity=frozen["identity"], versions={}, native_abi=abi, megatron_import_guard=guard)
    write(tmp_path / "cpu_preflight.json", proof)
    monkeypatch.setattr(preflight, "environment", lambda **k: {})
    monkeypatch.setattr(preflight, "native_abi", lambda: abi)
    monkeypatch.setattr(megatron_repair, "inspect", lambda: guard)
    launches = []
    monkeypatch.setattr(control, "start", lambda *a: launches.append(a) or dict(status="REGISTERED"))
    with pytest.raises(RuntimeError, match="preflight after the launcher"):
        control.retry_qualification(frozen["manifest_path"], ARMS)
    assert not launches
    proof["launcher_runtime"] = str(runtime)
    write(tmp_path / "cpu_preflight.json", proof)
    control.retry_qualification(frozen["manifest_path"], ARMS)
    assert launches == [(frozen["manifest_path"], "pipeline", list(ARMS))]
    assert {s["status"] for s in read(tmp_path / "states.json").values()} == {"READY"}


def ray_socket_validator():
    # Run the installed pinned Ray validator without starting any Ray services.
    assert importlib.metadata.version("ray") == "2.55.1"
    source = Path(importlib.metadata.distribution("ray").locate_file("ray/_private/utils.py"))
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "validate_socket_filepath")
    namespace = {"sys": sys}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["validate_socket_filepath"]


def test_old_abci_path_fails_pinned_ray_validator_but_short_path_passes():
    suffix = "/session_2026-11-12_00-54-36_403236_2361321/sockets/plasma_store"
    old = "/local/2917679.pbs1/mopd_ed28c77da7ba488fb42f3e135cb63a0e" + suffix
    new = "/local/2917679.pbs1/r" + suffix
    assert len(os.fsencode(old)) == 121 and len(os.fsencode(new)) == 85
    validate = ray_socket_validator()
    with pytest.raises(OSError, match="AF_UNIX path length cannot exceed 107 bytes"):
        validate(old)
    validate(new)
    validate(new.replace("2361321", "2147483647"))


@pytest.mark.parametrize("rank", [0, 1])
def test_repaired_shell_runs_with_job_local_short_path(frozen, tmp_path, rank):
    repair.repair(frozen["manifest_path"])
    runtime = repair.active_runtime(frozen)
    attempt = tmp_path / ("shell_" + str(rank))
    attempt.mkdir()
    stub = attempt / "python"
    stub.write_text("#!" + sys.executable + "\n" + r'''
import ast, importlib.metadata as md, os, pathlib, subprocess, sys
a = sys.argv[1:]
root = pathlib.Path(os.environ["TEST_ATTEMPT"])
if a[:2] == ["-I", "-"] or a[0] == "-c":
    raise SystemExit(subprocess.call([sys.executable] + a))
if a[:2] == ["-m", "ray.scripts.scripts"]:
    if a[2] == "start":
        assert not any(k.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_")) for k in os.environ)
        temp = os.environ["RAY_TMPDIR"]
        assert temp == os.environ["PBS_LOCALDIR"] + "/r"
        if "--head" in a:
            assert "--temp-dir=" + temp in a
        path = pathlib.Path(temp) / "session_9999-12-31_23-59-59_999999_2147483647/sockets/plasma_store"
        path.parent.mkdir(parents=True)
        assert md.version("ray") == "2.55.1"
        source = pathlib.Path(md.distribution("ray").locate_file("ray/_private/utils.py"))
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "validate_socket_filepath")
        ns = {"sys": sys}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
        ns["validate_socket_filepath"](str(path))
        (root / "socket_validated").write_text(str(path))
    raise SystemExit(0)
if "paths" in a:
    print("/test/pinned/runtime")
elif "execute" in a:
    (root / "executed").touch()
elif "gpu-node" not in a:
    raise RuntimeError(a)
''')
    stub.chmod(0o755)
    (attempt / "python.txt").write_text(str(stub))
    if rank:
        (attempt / "head_address").write_text("127.0.0.1:6379")
        (attempt / "finished").write_text("0")
    with tempfile.TemporaryDirectory(prefix="mp", dir="/tmp") as local:
        p = subprocess.run(["bash", str(runtime / "d4_mopd/node.sh"), str(attempt / "request.json")],
            env=dict(os.environ, PBS_LOCALDIR=local, OMPI_COMM_WORLD_RANK=str(rank), TEST_ATTEMPT=str(attempt)),
            capture_output=True, text=True, timeout=15)
        assert p.returncode == 0, p.stdout + p.stderr
        assert (attempt / "socket_validated").exists()
        assert (attempt / "executed").exists() == (rank == 0)
        assert (attempt / "finished").read_text().strip() == "0"
        assert "socket path budget:" in p.stdout
