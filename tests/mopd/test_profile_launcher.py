"""Profile submission boundaries and unchanged scientific config on real Hydra."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from d4_mopd import control
from d4_mopd.config import compose
from d4_mopd.execution import make_record, profile
from d4_mopd.preflight import validate_native_config
from d4_teachers.common import read, write
from d4_training_continuation.scheduler import Unknown


@pytest.fixture
def execution_manifest(manifest):
    m = deepcopy(manifest)
    m["execution"] = make_record(m, "single_node", {"d4_mopd/cli.py": "0" * 64},
                                 qualification_transition=("dual_node", "single_node"))
    Path(m["execution"]["control_root"]).mkdir(parents=True)
    return m


@pytest.mark.parametrize("arm", ["domain_mix", "domain_only"])
@pytest.mark.parametrize("name", ["single_node", "dual_node"])
def test_full_qualification_preserves_batches_lengths_sampler_and_native_loss(execution_manifest, tmp_path, arm, name):
    m = execution_manifest
    c = compose(m, arm, tmp_path / arm, qualification=True, execution_profile=name)
    native = validate_native_config(c)
    assert native["teacher_gpus"] == native["student_gpus"] == 8
    assert c.data.train_batch_size == c.actor_rollout_ref.actor.ppo_mini_batch_size == 2048
    assert c.data.max_response_length == 65535
    assert c.actor_rollout_ref.rollout.max_model_len == 65536
    assert c.actor_rollout_ref.rollout.n == 1
    assert c.actor_rollout_ref.rollout.max_num_seqs == 16
    assert c.actor_rollout_ref.rollout.max_num_batched_tokens == 8192
    assert c.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 65536
    assert c.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu == 65536
    assert c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert c.mopd.effective_spec == m["spec"]
    assert c.mopd.effective_spec.updates == 200
    assert c.trainer.total_training_steps == 8 and c.trainer.save_freq == c.trainer.test_freq == 4
    assert c.mopd.qualification and c.mopd.full_qualification
    assert not c.data.mopd_qualification
    assert c.distillation.enabled and c.mopd.execution_profile == name
    assert c.mopd.qualification_profile_sequence == ["dual_node", "single_node"]
    if name == "single_node":
        assert c.actor_rollout_ref.rollout.agent.agent_loop_manager_class.endswith("PhasedAgentLoopManager")
        assert c.actor_rollout_ref.rollout.checkpoint_manager_class.endswith("PhasedCheckpointEngineManager")


def test_profile_preflight_does_not_change_formal_schedule(execution_manifest, tmp_path):
    c = compose(execution_manifest, "domain_only", tmp_path)
    assert c.trainer.total_training_steps == 200
    assert c.trainer.save_freq == 5 and c.trainer.test_freq == 10
    assert not c.mopd.full_qualification
    assert c.data.max_response_length == 65535 and c.data.train_batch_size == 2048


@pytest.mark.parametrize("name,nodes", [("single_node", 1), ("dual_node", 2)])
def test_pbs_uses_reserved_queue_and_exact_physical_nodes(execution_manifest, tmp_path, monkeypatch, name, nodes):
    import d4_mopd.launcher_repair as launcher
    monkeypatch.setattr(launcher, "active_runtime", lambda m: Path(m["execution"]["runtime"]))
    script = control.render_pbs(execution_manifest, tmp_path / "spaces here", "0390_mp_qmix", name)
    assert "#PBS -q R9920261000" in script and "#PBS -P gcg51557" in script
    assert "#PBS -N 0390_mp_qmix" in script
    assert f"#PBS -l select={nodes}" in script
    assert f"mpirun -np {nodes} --map-by ppr:1:node" in script
    assert f") -eq {nodes} ]]" in script
    assert execution_manifest["execution"]["runtime"] in script


def test_qualification_isolated_but_formal_checkpoint_paths_shared(execution_manifest):
    m = execution_manifest
    base = Path(m["manifest_path"]).parent
    assert control.task_root(m, "qualification", "domain_mix") == Path(m["execution"]["control_root"]) / "qualification/domain_mix"
    assert control.task_root(m, "train", "domain_mix") == base / "runs/domain_mix"
    assert control.task_root(m, "test", "domain_only") == base / "final_tests/domain_only"


def test_new_attempt_changes_profile_only_at_committed_boundary(execution_manifest, monkeypatch):
    import d4_mopd.launcher_repair as launcher
    m = execution_manifest
    monkeypatch.setattr(launcher, "active_runtime", lambda m: Path(m["execution"]["runtime"]))
    run = control.task_root(m, "qualification", "domain_mix")
    incomplete = run / "checkpoints/global_step_4"
    incomplete.mkdir(parents=True)
    a = control.new_attempt(m, "qualification", "domain_mix")
    ra = read(Path(a["folder"]) / "request.json")
    assert ra["execution_profile"] == "dual_node"
    write(incomplete / "mopd_complete.json", {"step": 4})
    b = control.new_attempt(m, "qualification", "domain_mix")
    rb = read(Path(b["folder"]) / "request.json")
    assert rb["execution_profile"] == "single_node"
    assert rb["execution_identity"] == m["execution"]["execution_identity"]
    entry = (Path(b["folder"]) / "entry.py").read_text()
    assert "D4_MOPD_EXECUTION_RECORD" in entry
    assert m["execution"]["runtime"] in entry


def test_legacy_short_receipt_cannot_admit_execution_training(execution_manifest):
    m = execution_manifest
    write(control.task_root(m, "qualification", "domain_mix") / "receipt.json",
          {"status": "PASS", "identity": m["identity"], "scope": "qualification", "arm": "domain_mix", "step": 8, "resumed_from": 4})
    with pytest.raises(RuntimeError):
        control.qualification_passed(m, "domain_mix")


@pytest.mark.parametrize("failure", ["guardian", "pending", "running", "unknown", "writer"])
def test_handoff_requires_retired_controller_and_known_terminal_pbs(execution_manifest, monkeypatch, failure):
    m = execution_manifest
    base = Path(m["manifest_path"]).parent
    write(base / "STOP", {"reason": "handoff"})
    write(base / "states.json", {"train:domain_mix": {
        "phase": "train", "arm": "domain_mix", "status": "R", "job": {"job_id": "1"},
        "pending": {"folder": "ambiguous"} if failure == "pending" else None}})
    def active(path):
        return (failure == "guardian" and path == base / "guardian.lock"
                or failure == "writer" and path == base / "runs/domain_mix/writer.lock")
    monkeypatch.setattr(control, "active", active)
    def observe(job):
        if failure == "unknown":
            raise Unknown("PBS unavailable")
        return {"job_state": "R" if failure == "running" else "F"}
    with pytest.raises((RuntimeError, Unknown)):
        with control.formal_submission(m, "train", SimpleNamespace(observe=observe)):
            pytest.fail("unsafe handoff admitted")


def test_stopped_terminal_legacy_controller_does_not_need_state_deletion(execution_manifest, monkeypatch):
    m = execution_manifest
    base = Path(m["manifest_path"]).parent
    write(base / "STOP", {"reason": "handoff"})
    write(base / "states.json", {"train:domain_mix": {
        "phase": "train", "arm": "domain_mix", "status": "R", "job": {"job_id": "1"}, "pending": None}})
    monkeypatch.setattr(control, "active", lambda path: False)
    with control.formal_submission(m, "train", SimpleNamespace(observe=lambda job: {"job_state": "F"})):
        pass


def test_cli_reexecutes_exact_snapshot_and_preserves_arguments(execution_manifest, monkeypatch):
    import os
    import sys
    from d4_mopd import cli
    m = execution_manifest
    record = m["execution"]["record_path"]
    monkeypatch.setattr(cli, "verify", lambda path, models=False: m)
    expected_bootstrap = [m["python"], "-I", "-c", "snapshot bootstrap"]
    monkeypatch.setattr(control, "entry_command", lambda selected: expected_bootstrap)
    args = ["start", "--manifest", m["manifest_path"], "--phase", "qualification",
            "--arm", "domain_only", "--execution-record", record]
    monkeypatch.setattr(sys, "argv", ["scripts/mopd.py"] + args)
    monkeypatch.setenv("D4_MOPD_EXECUTION_RECORD", "/wrong/inherited/record.json")
    class Reexecuted(Exception):
        pass
    def execv(executable, command):
        assert executable == m["python"]
        assert command == expected_bootstrap + args
        assert os.environ["D4_MOPD_EXECUTION_RECORD"] == record
        raise Reexecuted
    monkeypatch.setattr(os, "execv", execv)
    with pytest.raises(Reexecuted):
        cli.main()


def test_request_runtime_wins_over_inherited_environment(monkeypatch):
    import os
    from d4_mopd.cli import select_request_execution
    monkeypatch.setenv("D4_MOPD_EXECUTION_RECORD", "/wrong/record.json")
    select_request_execution({"execution_record": "/request/record.json"})
    assert os.environ["D4_MOPD_EXECUTION_RECORD"] == "/request/record.json"
    select_request_execution({})
    assert "D4_MOPD_EXECUTION_RECORD" not in os.environ


def test_execution_preflight_fails_closed_in_its_own_directory(execution_manifest, monkeypatch):
    from d4_mopd import preflight, phase_runtime, launcher_repair
    m = execution_manifest
    base = Path(m["manifest_path"]).parent
    root = Path(m["execution"]["control_root"])
    write(base / "cpu_preflight.json", {"status": "PASS", "identity": "original"})
    write(root / "cpu_preflight.json", {"status": "PASS", "identity": m["identity"]})
    monkeypatch.setenv("D4_MOPD_EXECUTION_RECORD", m["execution"]["record_path"])
    monkeypatch.setattr(preflight, "verify", lambda path: m)
    monkeypatch.setattr(launcher_repair, "active_runtime", lambda value: Path(m["execution"]["runtime"]))
    seen = []
    def reject_changed_source(path):
        seen.append(path)
        raise phase_runtime.PhaseContractError("Pinned upstream source changed")
    monkeypatch.setattr(phase_runtime, "verify_upstream_contract", reject_changed_source)
    monkeypatch.setattr(preflight, "environment", lambda **kw: pytest.fail("Source verification must precede native checks"))
    with pytest.raises(phase_runtime.PhaseContractError, match="Pinned upstream"):
        preflight.cpu(m["manifest_path"])
    assert seen == [Path(m["verl_src"]) / "verl"]
    assert read(root / "cpu_preflight.json")["status"] == "FAIL"
    assert read(base / "cpu_preflight.json") == {"status": "PASS", "identity": "original"}


def test_direct_profile_final_test_requires_its_full_qualification(execution_manifest, monkeypatch):
    m = execution_manifest
    root = Path(m["execution"]["control_root"])
    monkeypatch.setattr(control, "verify", lambda path: m)
    write(root / "cpu_preflight.json", {"status": "PASS", "identity": m["identity"],
          "execution_identity": m["execution"]["execution_identity"],
          "runtime_identity": m["execution"]["runtime_identity"], "native_abi": {"status": "PASS"}})
    calls = []
    def gate(selected, arm):
        calls.append((selected, arm))
        raise RuntimeError("No full qualification for this execution")
    monkeypatch.setattr(control, "qualification_passed", gate)
    with pytest.raises(RuntimeError, match="No full qualification"):
        control.start(m["manifest_path"], "test", ("domain_mix",))
    assert calls == [(m, "domain_mix")]
