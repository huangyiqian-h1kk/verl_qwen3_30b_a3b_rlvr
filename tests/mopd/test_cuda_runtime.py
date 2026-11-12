"""Frozen revision migration, tool inheritance, and conservative signal recovery."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from d4_mopd import ARMS, control, cuda_repair, cuda_runtime, job_exit, launcher_repair
from d4_teachers.common import files, read, write
from d4_training_continuation.scheduler import Unknown
from test_launcher_repair import frozen  # A manifest-bound original launcher fixture.


ROOT = Path(__file__).resolve().parents[2]
TRACE = '''Traceback (most recent call last):
  File "driver.py", line 1, in execute
    ray.get(task)
  File "python/ray/_raylet.pyx", line 3012, in ray._raylet.CoreWorker.get_objects
  File "python/ray/includes/common.pxi", line 124, in ray._raylet.check_status
  File "python/ray/_raylet.pyx", line 747, in ray._raylet.raise_sys_exit_with_custom_error_message
SystemExit: 1
'''


@pytest.fixture
def cuda_case(frozen, tmp_path, monkeypatch):
    # Model the historical snapshot: these operational helpers did not exist.
    for name in ("cuda_repair.py", "cuda_runtime.py", "cuda_runtime.sh", "job_exit.py"):
        (tmp_path / "runtime/d4_mopd" / name).unlink()
    frozen["runtime_files"] = files(tmp_path / "runtime")
    write(tmp_path / "manifest.json", frozen)
    launcher_repair.repair(frozen["manifest_path"])
    states = read(tmp_path / "states.json")
    for key, s in states.items():
        folder = Path(s["job"]["folder"])
        write(folder / "request.json", dict(identity=frozen["identity"], phase="qualification", arm=s["arm"], folder=str(folder)))
        if s["arm"] == "domain_mix":
            write(folder / "result.json", dict(status="FAIL", identity=frozen["identity"], error="RayTaskError",
                traceback="RuntimeError: Engine core initialization failed"))
            (folder / "pbs.log").write_text("RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist\n")
        else:
            write(folder / "result.json", dict(status="FAIL", identity=frozen["identity"], error="1", traceback=TRACE))
            (folder / "pbs.log").write_text("Loading model weights\n")
    class PBS:
        def observe(self, job):
            if "domain_only" in job["folder"]:
                return dict(job_state="F", Exit_status=271, comment="Job run at example and terminated by root@fixture")
            return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    monkeypatch.setattr(cuda_repair, "verify", lambda _: frozen)
    return frozen


def test_cuda_revision_migrates_v1_preserving_learner_and_blocked_evidence(cuda_case, tmp_path):
    m = cuda_case
    before_manifest = (tmp_path / "manifest.json").read_bytes()
    before_state = read(tmp_path / "states.json")
    previous = read(tmp_path / "launcher_repair.json")
    old_inventory = files(Path(previous["folder"]) / "runtime")
    result = cuda_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    inventory = files(selected)
    changed = {k for k in inventory if inventory[k] != m["runtime_files"].get(k)}
    assert changed == {"d4_mopd/node.sh", "d4_mopd/control.py", "d4_mopd/job_exit.py",
                       "d4_mopd/cuda_runtime.sh", "d4_mopd/cuda_runtime.py"}
    assert files(tmp_path / "runtime") == m["runtime_files"]
    assert (tmp_path / "manifest.json").read_bytes() == before_manifest
    assert read(tmp_path / "states.json") == before_state
    assert files(Path(previous["folder"]) / "runtime") == old_inventory
    assert read(selected.parent / "receipt.json")["previous_pointer"] == previous
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    assert result["gpu_execution_tested"] is False
    assert cuda_repair.repair(m["manifest_path"])["revision"] == result["revision"]
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == 2
    assert cuda_repair.CUDA_SOURCE in (ROOT / "d4_mopd/node.sh").read_text()
    assert str(selected) in control.entry_command(m)[-1]


@pytest.mark.parametrize("change", ["update", "checkpoint", "resume", "unrelated", "identity", "only_code", "active", "unknown"])
def test_cuda_repair_refuses_unrelated_failure_or_training_progress(cuda_case, tmp_path, monkeypatch, change):
    state = read(tmp_path / "states.json")
    mix = Path(state["qualification:domain_mix"]["job"]["folder"])
    only = Path(state["qualification:domain_only"]["job"]["folder"])
    if change == "update": write(mix / "update_1.json", {"step": 1})
    if change == "checkpoint": (mix.parent.parent / "checkpoints/global_step_4").mkdir(parents=True)
    if change == "resume": write(mix / "resume.json", {"step": 0})
    if change == "unrelated": (mix / "pbs.log").write_text("RuntimeError: CUDA out of memory\n")
    if change in ("identity", "only_code"):
        path = (mix if change == "identity" else only) / "result.json"
        result = read(path)
        result["identity" if change == "identity" else "traceback"] = "different"
        write(path, result)
    if change in ("active", "unknown"):
        class PBS:
            def observe(self, job):
                if change == "unknown": raise Unknown("temporary qstat outage")
                return {"job_state": "R"}
        monkeypatch.setattr(control, "PBS", PBS)
    before = (tmp_path / "launcher_repair.json").read_bytes()
    with pytest.raises((RuntimeError, Unknown)):
        cuda_repair.repair(cuda_case["manifest_path"])
    assert (tmp_path / "launcher_repair.json").read_bytes() == before
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == 1


def test_cuda_snapshot_tamper_is_rejected(cuda_case, tmp_path):
    cuda_repair.repair(cuda_case["manifest_path"])
    selected = launcher_repair.active_runtime(cuda_case)
    (selected / "d4_mopd/cuda_runtime.sh").write_text("skip checks\n")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        launcher_repair.active_runtime(cuda_case)


def test_cuda_frozen_controller_uses_repaired_launcher_for_all_stages(cuda_case, tmp_path):
    cuda_repair.repair(cuda_case["manifest_path"])
    selected = launcher_repair.active_runtime(cuda_case)
    ctl = load_controller(selected / "d4_mopd/control.py")
    for arm in ARMS:
        write(tmp_path / "runs" / arm / "receipt.json",
              dict(status="PASS", identity=cuda_case["identity"], selection={"step": 10}))
        for phase in ("qualification", "train", "test"):
            job = ctl.new_attempt(cuda_case, phase, arm)
            folder = Path(job["folder"])
            assert str(selected / "d4_mopd/node.sh") in (folder / "stage.pbs").read_text()
            entry = (folder / "entry.py").read_text()
            assert 'sys.path[:0]=' + repr([str(tmp_path / "runtime"), cuda_case["verl_src"]]) in entry
            assert str(selected / "d4_mopd/job_exit.py") in entry
    states = read(tmp_path / "states.json")
    for arm in ARMS:
        states["qualification:" + arm]["status"] = "DONE"
        write(tmp_path / "qualification" / arm / "receipt.json",
              dict(status="PASS", identity=cuda_case["identity"], arm=arm,
                   scope="qualification", step=8, resumed_from=4))
        ctl.advance_pipeline(cuda_case, states)
        assert any(s["phase"] == "train" for s in states.values()) == (arm == ARMS[-1])


def load_controller(path):
    spec = importlib.util.spec_from_file_location("d4_mopd._cuda_controller_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("controller_kind", ["fresh", "repaired"])
def test_controller_requires_signal_message_and_preserves_real_failures(cuda_case, tmp_path, controller_kind):
    cuda_repair.repair(cuda_case["manifest_path"])
    selected = launcher_repair.active_runtime(cuda_case)
    ctl = control if controller_kind == "fresh" else load_controller(selected / "d4_mopd/control.py")
    for variant in ("legacy", "sigterm", "different-message", "wrong-attempt", "non-admin", "app-error", "node-fail"):
        state = read(tmp_path / "states.json")
        s = state["qualification:domain_only"]
        s.update(status="R")
        folder = Path(s["job"]["folder"])
        result = dict(status="FAIL", identity=cuda_case["identity"], error="1", traceback=TRACE)
        write(folder / "result.json", result)
        (folder / "cuda_runtime_1.json").unlink(missing_ok=True)
        exc = SystemExit(1)
        exc.ray_terminate_msg = job_exit.RAY_DRIVER_SIGTERM if variant != "different-message" else "worker failed"
        if variant != "legacy": job_exit.record_ray_exit(exc, folder / "request.json")
        result = read(folder / "result.json")
        if variant == "wrong-attempt": result["termination_evidence"]["attempt"] = "/other"
        if variant == "app-error": result.update(error="OOM", traceback="RuntimeError: out of memory")
        write(folder / "result.json", result)
        if variant == "node-fail": write(folder / "cuda_runtime_1.json", {"status": "FAIL", "error": "nvcc not found"})
        class PBS:
            def observe(self, job):
                return dict(job_state="F", Exit_status=271,
                    comment="terminated by " + ("user@fixture" if variant == "non-admin" else "root@fixture"))
        ctl.tick(cuda_case, state, PBS())
        assert s["status"] == ("READY" if variant == "sigterm" else "BLOCKED"), variant
        assert not any(x["phase"] == "train" for x in state.values())
        # Keep the original job for the next scenario.
        if s.get("job") is None: s["job"] = s["previous_job"]
        write(tmp_path / "states.json", state)


def test_entry_wrapper_keeps_learner_path_and_captures_ray_attribute(tmp_path):
    original = tmp_path / "runtime/d4_mopd"
    original.mkdir(parents=True)
    (original / "__init__.py").touch()
    request = tmp_path / "request.json"
    write(request, dict(folder=str(tmp_path), identity="fixture"))
    write(tmp_path / "result.json", dict(status="FAIL", identity="fixture", error="1", traceback=TRACE))
    (original / "cli.py").write_text("def main():\n    e = SystemExit(1)\n    e.ray_terminate_msg = "
        + repr(job_exit.RAY_DRIVER_SIGTERM) + "\n    raise e\n")
    m = dict(manifest_path=str(tmp_path / "manifest.json"), verl_src="/fixture/upstream")
    entry = tmp_path / "entry.py"
    entry.write_text(job_exit.entry_script(m, ROOT / "d4_mopd/job_exit.py", request))
    p = subprocess.run([sys.executable, "-I", str(entry)], capture_output=True, text=True, timeout=15)
    assert p.returncode == 1
    result = read(tmp_path / "result.json")
    assert result["traceback"] == TRACE and result["status"] == "FAIL"
    assert result["termination_evidence"]["signal"] == 15
    assert repr(str(tmp_path / "runtime")) in entry.read_text()


def make_tools(tmp_path, *, release="13.0", ninja_ok=True):
    bins, cuda = tmp_path / "venv/bin", tmp_path / "cuda"
    bins.mkdir(parents=True)
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include").mkdir()
    (cuda / "include/cuda_runtime.h").touch()
    for path, code in {
        cuda / "bin/nvcc": f'''import pathlib, sys
if '--version' in sys.argv: print('Cuda compilation tools, release {release}, V{release}.88')
else: pathlib.Path('probe.o').write_bytes(b'compiled-fixture')
''',
        bins / "ninja": '''import pathlib, shlex, subprocess, sys
if '--version' in sys.argv: print('1.13.0')
else:
    line = next(s for s in pathlib.Path('build.ninja').read_text().splitlines() if s.startswith('  command = '))
    raise SystemExit(subprocess.call(shlex.split(line.split(' = ',1)[1].replace('$$','$'))))
''',
        bins / "g++": "print('13.2.0')\n",
    }.items():
        path.write_text("#!" + sys.executable + "\n" + code)
        path.chmod(0o755)
    if not ninja_ok: (bins / "ninja").unlink()
    return bins, cuda


@pytest.mark.parametrize("issue", [None, "cuda-version", "missing-ninja", "stubs", "headers", "torch-build", "compile", "ninja-suffix"])
def test_runtime_compiler_probe(tmp_path, monkeypatch, issue):
    bins, cuda = make_tools(tmp_path, release="12.8" if issue == "cuda-version" else "13.0",
                            ninja_ok=issue != "missing-ninja")
    monkeypatch.setattr(sys, "executable", str(bins / "python"))
    monkeypatch.setattr(cuda_runtime.md, "version", lambda n: {"torch": "2.13.0+cu128" if issue == "torch-build" else "2.13.0+cu130", "ninja": "1.13.0"}[n])
    for k, v in dict(PATH=str(bins) + ":" + str(cuda / "bin"), CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
                     CUDACXX=str(cuda / "bin/nvcc"), PBS_LOCALDIR=str(tmp_path / "local"),
                     LD_LIBRARY_PATH="/cuda/lib64/stubs" if issue == "stubs" else "").items():
        monkeypatch.setenv(k, v)
    if issue == "headers": (cuda / "include/cuda_runtime.h").unlink()
    if issue == "compile":
        nvcc = cuda / "bin/nvcc"
        nvcc.write_text(nvcc.read_text().replace("pathlib.Path('probe.o').write_bytes(b'compiled-fixture')", "sys.exit(7)"))
    if issue == "ninja-suffix":
        ninja = bins / "ninja"
        ninja.write_text(ninja.read_text().replace("print('1.13.0')", "print('1.13.0.vendor')"))
    if issue and issue != "ninja-suffix":
        with pytest.raises(RuntimeError): cuda_runtime.probe(tmp_path, 0)
    else:
        proof = cuda_runtime.probe(tmp_path, 0)
        assert proof["compile_probe"]["status"] == "PASS" and proof["gpu_execution_tested"] is False
        assert "-arch=sm_90" in (tmp_path / "local/cuda_probe_0/build.ninja").read_text()


@pytest.mark.parametrize("rank,modules", [(0, True), (1, True), (0, False), (1, False)])
def test_each_node_exports_tools_before_gpu_import_and_ray(tmp_path, rank, modules):
    bins, cuda = make_tools(tmp_path)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write(attempt / "request.json", {"identity": "fixture"})
    python = bins / "python"
    python.write_text("#!" + sys.executable + "\n" + r'''
import importlib.metadata as md, os, pathlib, runpy, subprocess, sys
a = sys.argv[1:]
root = pathlib.Path(os.environ['TEST_ATTEMPT'])
if a[:2] == ['-I', '-'] or a[0] == '-c':
    raise SystemExit(subprocess.call([sys.executable] + a))
if a[0] == '-I' and a[1].endswith('/cuda_runtime.py'):
    sys.executable = __file__
    md.version = lambda n: {'torch':'2.13.0+cu130', 'ninja':'1.13.0'}[n]
    sys.argv = a[1:]
    runpy.run_path(a[1], run_name='__main__')
    raise SystemExit(0)
if 'paths' in a:
    print('/fixture/original/runtime')
elif 'gpu-node' in a or a[:3] == ['-m', 'ray.scripts.scripts', 'start'] or 'execute' in a:
    assert os.environ['CUDA_HOME'] == os.environ['MOCK_CUDA']
    assert os.environ['CUDA_PATH'] == os.environ['MOCK_CUDA']
    assert pathlib.Path(os.environ['CUDACXX']).is_file()
    assert os.environ['PATH'].split(':')[0] == str(pathlib.Path(__file__).parent)
    import json
    rank = os.environ['TEST_RANK']
    assert json.loads((root / ('cuda_runtime_'+rank+'.json')).read_text())['status'] == 'PASS'
    with (root / 'events').open('a') as f: f.write('gpu\n' if 'gpu-node' in a else 'ray\n')
elif a[:3] != ['-m', 'ray.scripts.scripts', 'stop']:
    raise RuntimeError(a)
''')
    python.chmod(0o755)
    (attempt / "python.txt").write_text(str(python))
    bashenv = tmp_path / "modules.sh"
    bashenv.write_text('''module() {
  case "$1" in
    show) [[ $MOCK_MODULES == yes ]] ;;
    load) export PATH="$MOCK_CUDA/bin:$PATH" ;;
    list) return 0 ;;
    *) return 1 ;;
  esac
}
''')
    if rank:
        (attempt / "head_address").write_text("127.0.0.1:6379")
        (attempt / "finished").write_text("0")
    with tempfile.TemporaryDirectory(prefix="cu", dir="/tmp") as local:
        env = dict(os.environ, BASH_ENV=str(bashenv), MOCK_CUDA=str(cuda), MOCK_MODULES="yes" if modules else "no",
                   TEST_ATTEMPT=str(attempt), TEST_RANK=str(rank), PBS_LOCALDIR=local,
                   OMPI_COMM_WORLD_RANK=str(rank), LD_LIBRARY_PATH="")
        p = subprocess.run(["bash", str(ROOT / "d4_mopd/node.sh"), str(attempt / "request.json")],
                           env=env, capture_output=True, text=True, timeout=20)
    assert (p.returncode == 0) == modules, p.stdout + p.stderr
    proof = read(attempt / f"cuda_runtime_{rank}.json")
    assert proof["status"] == ("PASS" if modules else "FAIL")
    assert (attempt / "events").exists() == modules
    if modules: assert (attempt / "events").read_text().splitlines()[:2] == ["gpu", "ray"]
