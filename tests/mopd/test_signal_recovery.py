"""Ray's compiled exception bridge, cleanup interruption, and frozen recovery."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from d4_mopd import ARMS, cli, control, job_exit, job_exit_v2 as exits, launcher_repair, metadata_repair, sigterm_repair
from d4_teachers.common import files, read, write
from d4_training_continuation.scheduler import Unknown
from test_batch_metadata import metadata_case
from test_cuda_runtime import TRACE, load_controller
from test_launcher_repair import frozen


ROOT = Path(__file__).resolve().parents[2]
PBS_SIGNAL = dict(job_state="F", Exit_status=271, comment="Job run at example and terminated by root@fixture")


@pytest.mark.parametrize("variant,allowed", [
    ("legacy", True), ("old-marker", True), ("early-details", True), ("direct-signal", True),
    ("generic-exit", False), ("app-error", False), ("chained-error", False),
    ("wrong-identity", False), ("non-root", False), ("exit-one", False),
    ("wrong-message", False), ("wrong-attempt", False), ("null-details", False),
    ("contradictory-markers", False), ("null-marker", False),
])
def test_retry_requires_joint_pbs_and_ray_evidence(tmp_path, variant, allowed):
    row, job = dict(PBS_SIGNAL), dict(folder=str(tmp_path))
    result = dict(status="FAIL", identity="fixture", error="1", traceback=TRACE)
    details = dict(schema=exits.EXIT_SCHEMA, type="SystemExit", code=1, ray_terminate_msg=job_exit.RAY_DRIVER_SIGTERM)
    marker = dict(schema=job_exit.EVIDENCE_SCHEMA, signal=15, ray_terminate_msg=job_exit.RAY_DRIVER_SIGTERM,
                  identity="fixture", attempt=str(tmp_path.resolve()))
    if variant in ("old-marker", "wrong-attempt", "contradictory-markers"):
        result["termination_evidence"] = dict(marker)
    if variant in ("early-details", "wrong-message", "contradictory-markers"):
        result["exit_details"] = dict(details)
    if variant == "direct-signal":
        result.update(error="15", traceback='''Traceback (most recent call last):
  File "worker.py", line 1, in sigterm_handler
    sys.exit(signum)
SystemExit: 15
''', exit_details=dict(details, code=15, ray_terminate_msg=None))
    if variant == "generic-exit": result["traceback"] = "Traceback (most recent call last):\nSystemExit: 1\n"
    if variant == "app-error": result.update(error="index", traceback="KeyError: 'index'\n")
    if variant == "chained-error": result["traceback"] = "Traceback (most recent call last):\nRuntimeError: OOM\n" + TRACE
    if variant == "wrong-identity": result["identity"] = "other"
    if variant == "non-root": row["comment"] = "terminated by user@fixture"
    if variant == "exit-one": row["Exit_status"] = 1
    if variant in ("wrong-message", "contradictory-markers"): result["exit_details"]["ray_terminate_msg"] = "application exit"
    if variant == "wrong-attempt": result["termination_evidence"]["attempt"] = "/different/attempt"
    if variant == "null-details": result["exit_details"] = None
    if variant == "null-marker": result["termination_evidence"] = None
    assessment = exits.retry_assessment(row, result, job, "fixture")
    assert assessment["allowed"] == allowed, assessment
    assert assessment["reason"]


@pytest.fixture
def signal_case(metadata_case, monkeypatch):
    monkeypatch.setattr(sigterm_repair, "verify", lambda _: metadata_case)
    return metadata_case


@pytest.mark.parametrize("previous_version", [2, 3])
def test_migration_preserves_prior_snapshots_and_recipe(signal_case, tmp_path, previous_version):
    m = signal_case
    if previous_version == 3: metadata_repair.repair(m["manifest_path"])
    previous = launcher_repair.active_runtime(m)
    old_files = files(previous)
    before_manifest, before_states = (tmp_path / "manifest.json").read_bytes(), read(tmp_path / "states.json")
    result = sigterm_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    assert files(previous) == old_files and files(tmp_path / "runtime") == m["runtime_files"]
    assert (tmp_path / "manifest.json").read_bytes() == before_manifest
    assert read(tmp_path / "states.json") == before_states
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    assert result["failure_evidence"] == dict(zip(
        ["qualification:" + a for a in ARMS],
        ["pbs_root_sigterm_with_ray_trace", "top_level_index_removed_before_first_update"]))
    _, v3_files, _, _ = metadata_repair.revision_plan(m)
    assert {p for p, h in files(selected).items() if v3_files.get(p) != h} == {
        "d4_mopd/control.py", "d4_mopd/cli.py", "d4_mopd/job_exit_v2.py"}
    assert result["gpu_execution_tested"] is False
    assert sigterm_repair.repair(m["manifest_path"])["revision"] == result["revision"]
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == previous_version
    for name in ("config.py", "spec.py", "integration.py", "checkpoints.py", "sampler.py"):
        if (previous / "d4_mopd" / name).exists() and (previous_version == 3 or name != "integration.py"):
            assert (selected / "d4_mopd" / name).read_bytes() == (previous / "d4_mopd" / name).read_bytes()
    assert (selected / "config/mopd/baseline.json").read_bytes() == (previous / "config/mopd/baseline.json").read_bytes()


@pytest.mark.parametrize("controller_kind", ["fresh", "repaired"])
def test_mix_signal_retries_without_releasing_training_barrier(signal_case, tmp_path, controller_kind):
    m = signal_case
    sigterm_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    ctl = control if controller_kind == "fresh" else load_controller(selected / "d4_mopd/control.py")
    state = read(tmp_path / "states.json")
    mix = state["qualification:domain_mix"]
    mix.update(status="R", external_interruptions=24)
    original_job = mix["job"]
    original_result = (Path(original_job["folder"]) / "result.json").read_bytes()
    class PBS:
        submitted = []
        def observe(self, job): return dict(PBS_SIGNAL)
        def submit(self, folder, job_name):
            job = dict(folder=folder, job_name=job_name, job_id="new.pbs1")
            self.submitted.append(job)
            return job
    pbs = PBS()
    ctl.tick(m, state, pbs)
    assert mix["status"] == "READY" and mix["job"] is None and mix["previous_job"] == original_job
    assert mix["external_interruptions"] == 25 and mix["retry_reason"] == "pbs_271"
    assert mix["retry_assessment"] == dict(allowed=True, reason="pbs_root_sigterm_with_ray_trace",
                                            pbs_reason="pbs_271", job_id=original_job["job_id"])
    assert not pbs.submitted and not any(s["phase"] == "train" for s in state.values())
    ctl.tick(m, state, pbs)
    assert mix["status"] == "SUBMITTED" and len(pbs.submitted) == 1
    assert state["qualification:domain_only"]["status"] == "BLOCKED"
    assert not any(s["phase"] == "train" for s in state.values())
    assert (Path(original_job["folder"]) / "result.json").read_bytes() == original_result


def test_real_node_failure_overrides_driver_signal(signal_case, tmp_path):
    state = read(tmp_path / "states.json")
    mix = state["qualification:domain_mix"]
    mix["status"] = "R"
    write(Path(mix["job"]["folder"]) / "gpu_node_1.json", dict(status="FAIL", error="native ABI failed"))
    pbs = type("PBS", (), {"observe": lambda *a: PBS_SIGNAL})()
    control.tick(signal_case, state, pbs)
    assert mix["status"] == "BLOCKED" and "native ABI" in mix["error"]
    assert mix["retry_assessment"]["reason"] == "unbound_failure_or_failed_node_preflight"


@pytest.mark.parametrize("driver_kind", ["fresh", "repaired"])
def test_real_ray_exception_is_saved_even_if_cleanup_interrupts(signal_case, tmp_path, monkeypatch, driver_kind):
    import ray
    import ray._raylet
    from verl.trainer import main_ppo
    from d4_mopd import config, preflight
    m = signal_case
    module = cli
    if driver_kind == "repaired":
        sigterm_repair.repair(m["manifest_path"])
        path = launcher_repair.active_runtime(m) / "d4_mopd/cli.py"
        spec = importlib.util.spec_from_file_location("d4_mopd._signal_driver_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    folder = tmp_path / "driver_attempt"
    request = folder / "request.json"
    write(request, dict(manifest=m["manifest_path"], identity=m["identity"], folder=str(folder),
                        run_dir=str(tmp_path / "driver_run"), job_name="fixture", phase="qualification", arm="domain_mix"))
    for rank in (0, 1): write(folder / f"gpu_node_{rank}.json", dict(status="PASS", identity=m["identity"]))
    write(tmp_path / "cpu_preflight.json", dict(versions={"fixture": "version"}))
    monkeypatch.setattr(module, "verify", lambda _: m)
    monkeypatch.setattr(control, "PBS", lambda: type("PBS", (), {"observe": lambda *a: {"job_state": "R"}})())
    monkeypatch.setattr(preflight, "environment", lambda **kw: {"fixture": "version"})
    monkeypatch.setattr(preflight, "validate_native_config", lambda _: None)
    monkeypatch.setattr(config, "compose", lambda *a, **kw: OmegaConf.create(dict(mopd=dict(test_checkpoint=None), trainer={})))
    monkeypatch.setattr(ray, "init", lambda **kw: None)
    monkeypatch.setattr(ray, "nodes", lambda: [dict(Alive=True, Resources={"GPU": 8})] * 2)
    for key in ("PBS_JOBID", "RAY_ADDRESS", "PYTHONPATH"): monkeypatch.setenv(key, "fixture")
    def terminate(*args):
        # Use the installed compiled Ray bridge, without starting a local cluster.
        ray._raylet.raise_sys_exit_with_custom_error_message(job_exit.RAY_DRIVER_SIGTERM, exit_code=1)
    def interrupted_cleanup():
        result = read(folder / "result.json")
        assert result["status"] == "FAIL" and result["identity"] == m["identity"]
        assert result["exit_details"] == dict(schema=exits.EXIT_SCHEMA, type="SystemExit", code=1,
                                               ray_terminate_msg=job_exit.RAY_DRIVER_SIGTERM)
        assert "raise_sys_exit_with_custom_error_message" in result["traceback"]
        raise RuntimeError("cleanup interrupted before outer entry handler")
    monkeypatch.setattr(main_ppo, "run_ppo", terminate)
    monkeypatch.setattr(ray, "shutdown", interrupted_cleanup)
    with pytest.raises(RuntimeError, match="cleanup interrupted"):
        module.execute(request)
    assert read(folder / "result.json")["error"] == "1"


def test_repaired_guardian_driver_and_workers_share_checked_snapshot(signal_case, tmp_path):
    m = signal_case
    sigterm_repair.repair(m["manifest_path"])
    selected = launcher_repair.active_runtime(m)
    for arm in ARMS:
        write(tmp_path / "runs" / arm / "receipt.json", dict(status="PASS", identity=m["identity"], selection={"step": 10}))
    code = ("import sys,json; sys.path[:0]=" + repr([str(selected), m["verl_src"]])
        + "; from d4_mopd.control import new_attempt; m=" + repr(m)
        + "; print(json.dumps([new_attempt(m,phase,arm) for arm in " + repr(ARMS)
        + " for phase in ('qualification','train','test')]))")
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True, timeout=30)
    for job in json.loads(p.stdout):
        folder = Path(job["folder"])
        assert str(selected / "d4_mopd/node.sh") in (folder / "stage.pbs").read_text()
        driver = subprocess.run([sys.executable, "-I", str(folder / "entry.py"), "paths", "--request", str(folder / "request.json")],
                                capture_output=True, text=True, check=True, timeout=30)
        assert driver.stdout.strip() == str(selected) + os.pathsep + m["verl_src"]
        assert "runpy" not in (folder / "entry.py").read_text()
    worker = subprocess.run([sys.executable, "-c", "import d4_mopd.job_exit_v2 as e; print(e.__file__)"],
        env={**os.environ, "PYTHONPATH": driver.stdout.strip()}, cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=30)
    assert worker.stdout.strip() == str(selected / "d4_mopd/job_exit_v2.py")


@pytest.mark.parametrize("change", ["update", "checkpoint", "resume", "generic-exit", "app-error", "identity",
                                    "conflict", "non-root", "node-fail", "missing-proof", "active", "unknown"])
def test_migration_refuses_unrelated_failures_and_progress(signal_case, tmp_path, monkeypatch, change):
    m = signal_case
    state = read(tmp_path / "states.json")
    mix = Path(state["qualification:domain_mix"]["job"]["folder"])
    if change == "update": write(mix / "update_1.json", dict(step=1))
    if change == "checkpoint": (mix.parent.parent / "checkpoints/global_step_4").mkdir(parents=True)
    if change == "resume": write(mix / "resume.json", dict(identity=m["identity"], step=4, checkpoint="saved"))
    result = read(mix / "result.json")
    if change == "generic-exit": result["traceback"] = "SystemExit: 1"
    if change == "app-error": result.update(error="OOM", traceback="RuntimeError: OOM")
    if change == "identity": result["identity"] = "other"
    if change == "conflict": result["exit_details"] = dict(type="SystemExit", code=1, ray_terminate_msg="application exit")
    write(mix / "result.json", result)
    if change == "node-fail": write(mix / "cuda_runtime_1.json", dict(status="FAIL", error="compile failed"))
    if change == "missing-proof": (mix / "gpu_node_1.json").unlink()
    if change in ("active", "unknown", "non-root"):
        class PBS:
            def observe(self, job):
                if change == "unknown": raise Unknown("qstat timeout")
                return dict(job_state="R" if change == "active" else "F", Exit_status=271, comment="terminated by user@fixture")
        monkeypatch.setattr(control, "PBS", PBS)
    pointer = (tmp_path / "launcher_repair.json").read_bytes()
    with pytest.raises((RuntimeError, FileNotFoundError, Unknown)):
        sigterm_repair.repair(m["manifest_path"])
    assert (tmp_path / "launcher_repair.json").read_bytes() == pointer
    assert read(tmp_path / "states.json") == state
    assert len(list((tmp_path / "launcher_revisions").iterdir())) == 1
