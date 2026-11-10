from pathlib import Path
import copy
import pytest
import torch

from d4_mopd import checkpoints, control
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import read, write
from d4_training_continuation.scheduler import Unknown


@pytest.mark.parametrize("available,release,load_fails,selected,success", [
    ("nested", "13.0", False, "cuda/13.0/13.0.1", True),
    ("flat", "13.0", False, "cuda/13.0.1", True),
    ("both", "13.0", False, "cuda/13.0/13.0.1", True),
    ("neither", "13.0", False, None, False),
    ("nested", "12.8", False, "cuda/13.0/13.0.1", False),
    ("nested", "13.1", False, "cuda/13.0/13.0.1", False),
    ("nested", "13.0", True, "cuda/13.0/13.0.1", False),
])
def test_te_toolchain_resolves_abci_module_and_refuses_wrong_cuda(
        tmp_path, available, release, load_fails, selected, success):
    """Run the production shell check against a synthetic Modules installation."""
    import os
    import subprocess
    import sys
    cuda = tmp_path / "cuda" / "bin"
    cuda.mkdir(parents=True)
    nvcc = cuda / "nvcc"
    nvcc.write_text(f'#!/bin/sh\nprintf "Cuda compilation tools, release {release}, V{release}.88\\n"\n')
    nvcc.chmod(0o755)
    audit = tmp_path / "audit"
    audit.mkdir()
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    env = dict(os.environ, MOCK_AVAILABLE=available, MOCK_LOAD_FAIL=str(int(load_fails)),
               MOCK_CUDA=str(cuda), MOCK_AUDIT=str(audit))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", r'''
module() {
  case "$1" in
    -t) printf '/apps/modules:\ncuda/12.8/12.8.1\ncuda/13.0/13.0.1\n' >&2 ;;
    show)
      if [[ $2 == cuda/13.0/13.0.1 && ( $MOCK_AVAILABLE == nested || $MOCK_AVAILABLE == both ) ]] ||
         [[ $2 == cuda/13.0.1 && ( $MOCK_AVAILABLE == flat || $MOCK_AVAILABLE == both ) ]]; then
        printf 'modulefile %s\n' "$2" >&2
      else
        printf 'ERROR: Unable to locate a modulefile for %s\n' "$2" >&2
        return 1
      fi ;;
    load)
      printf '%s\n' "$*" > "$MOCK_AUDIT/load_called.txt"
      [[ $MOCK_LOAD_FAIL == 0 ]] || return 1
      export PATH="$MOCK_CUDA:$PATH" ;;
    list) printf 'loaded modules\n' >&2 ;;
    *) return 2 ;;
  esac
}
source "$1"
mopd_te_toolchain "$2" "$3"
touch "$3/build_permitted"
''', "toolchain-test", str(script), sys.executable, str(audit)],
        env=env, capture_output=True, text=True, timeout=20)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    assert (audit / "build_permitted").exists() == success
    assert "cuda/12.8/12.8.1" in (audit / "cuda_modules.txt").read_text()
    if selected:
        assert (audit / "load_called.txt").read_text().strip() == f"load gcc/13.2.0 {selected} cmake/4.1.1"
    else:
        assert not (audit / "load_called.txt").exists()
        assert "CUDA 13.0.1 module not found" in result.stderr
    if selected and not load_fails:
        proof = read(audit / "toolchain.json")
        assert proof["status"] == ("PASS" if success else "FAIL")
        assert proof["cuda_module"] == selected
        assert proof["cuda_home"] == str(cuda.parent)
        assert proof["nvcc_path"] == str(nvcc)
        if not success:
            assert "requires nvcc release 13.0" in result.stderr


def blocked_qualifications(manifest, tmp_path, monkeypatch):
    from d4_mopd import preflight, ARMS
    states = {}
    for i, arm in enumerate(ARMS):
        folder = tmp_path / "qualification" / arm / "attempts" / "failed"
        folder.mkdir(parents=True)
        (folder / "pbs.log").write_text("OSError: GLIBC_2.38 not found")
        states["qualification:"+arm] = dict(phase="qualification", arm=arm, status="BLOCKED", pending=None,
            error="Unclassified terminal job; inspect logs",
            job=dict(folder=str(folder), job_name="0390_mp_q"+arm, job_id=f"{10+i}.pbs1"))
    abi = dict(status="PASS", files=[dict(sha256="repaired")], incompatible=[])
    write(tmp_path / "states.json", states)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], versions={}, native_abi=abi))
    write(tmp_path / "pipeline.json", dict(schema="d4-mopd-pipeline-v1", identity=manifest["identity"], arms=list(ARMS)))
    monkeypatch.setattr(control, "verify", lambda _:manifest)
    monkeypatch.setattr(preflight, "environment", lambda **_: {})
    monkeypatch.setattr(preflight, "native_abi", lambda:abi)
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    return states, abi


def test_explicit_retry_preserves_failures_and_parallel_pipeline_barrier(manifest, tmp_path, monkeypatch):
    from d4_mopd import ARMS
    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(control, "start", lambda *args:calls.append(args) or dict(status="REGISTERED"))
    control.retry_qualification(manifest["manifest_path"], ARMS)
    current = read(tmp_path / "states.json")
    assert calls == [(manifest["manifest_path"], "pipeline", list(ARMS))]
    assert {s["status"] for s in current.values()} == {"READY"}
    for key, old in states.items():
        assert current[key]["previous_job"] == old["job"]
        assert Path(old["job"]["folder"], "pbs.log").read_text().startswith("OSError")
        assert read(current[key]["explicit_retry"])["before"] == states
    control.advance_pipeline(manifest, current)
    assert not any(s["phase"] == "train" for s in current.values())
    # A repeated command cannot erase a fresh pending/running submission.
    with pytest.raises(RuntimeError, match="BLOCKED"):
        control.retry_qualification(manifest["manifest_path"], ARMS)
    assert len(calls) == 1


@pytest.mark.parametrize("condition", ["active", "unknown", "pending", "checkpoint", "old_proof", "binary_change", "guardian"])
def test_explicit_retry_refuses_ambiguous_or_changed_state(manifest, tmp_path, monkeypatch, condition):
    from contextlib import ExitStack
    from d4_mopd import ARMS, preflight
    from d4_training_continuation.common import locked
    states, abi = blocked_qualifications(manifest, tmp_path, monkeypatch)
    key = "qualification:domain_mix"
    if condition in ("active", "unknown"):
        class PBS:
            def observe(self, job):
                if condition == "unknown": raise Unknown("qstat timeout")
                return dict(job_state="R")
        monkeypatch.setattr(control, "PBS", PBS)
    if condition == "pending":
        states[key]["pending"] = {"folder":"ambiguous"}
        write(tmp_path / "states.json", states)
    if condition == "checkpoint":
        (tmp_path / "qualification/domain_mix/checkpoints/global_step_4").mkdir(parents=True)
    if condition == "old_proof":
        write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], versions={}))
    if condition == "binary_change":
        monkeypatch.setattr(preflight, "native_abi", lambda:dict(abi, files=[dict(sha256="changed")]))
    monkeypatch.setattr(control, "start", lambda *a:pytest.fail("must not launch"))
    with ExitStack() as stack:
        if condition == "guardian": stack.enter_context(locked(tmp_path / "guardian.lock"))
        with pytest.raises((RuntimeError, BlockingIOError, Unknown)):
            control.retry_qualification(manifest["manifest_path"], ARMS)
    assert read(tmp_path / "states.json") == states
    assert not list((tmp_path / "qualification_retries").glob("*.json"))


def test_structured_node_error_blocks_instead_of_external_retry(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "gpu_node_1.json", dict(status="FAIL", error="GLIBC_2.38 not found"))
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=271, comment="terminated by root@login3")
    control.tick(manifest, state, PBS())
    s = state["train:domain_mix"]
    assert s["status"] == "BLOCKED" and "GLIBC_2.38" in s["error"]


@pytest.mark.parametrize("build_fails", [False, True])
def test_environment_repair_keeps_states_frozen_and_requires_new_preflight(manifest, tmp_path, monkeypatch, build_fails):
    import subprocess
    import importlib.metadata
    from types import SimpleNamespace
    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    before = read(tmp_path / "cpu_preflight.json")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda:[SimpleNamespace(metadata={"Name":"torch"}, version="2.13.0")])
    def build(args, **kwargs):
        assert args[3] == "--rebuild-te"
        assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
        assert control.active(tmp_path / "guardian.lock")
        if build_fails: raise subprocess.CalledProcessError(1, args)
        write(Path(args[4]) / "build_receipt.json", dict(source_revision="pinned", sha256="wheel"))
    monkeypatch.setattr(control.subprocess, "run", build)
    if build_fails:
        with pytest.raises(subprocess.CalledProcessError): control.repair_te(manifest["manifest_path"])
    else:
        assert control.repair_te(manifest["manifest_path"])["status"] == "PASS"
    assert read(tmp_path / "states.json") == states
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    audits = list((tmp_path / "environment_repairs").iterdir())
    assert len(audits) == 1
    assert read(audits[0] / "previous_cpu_preflight.json") == before
    assert read(audits[0] / "result.json")["status"] == ("FAIL" if build_fails else "PASS")
    assert not control.active(tmp_path / "guardian.lock")


def checkpoint(root, step, identity="id", arm="domain_mix"):
    folder = root / f"global_step_{step}"
    actor = folder / "actor"
    contents = {}
    for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
        path = actor / key / "shard.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-native-shard")
        contents[key] = {"path": key}
    write(actor / "ckpt_contents.json", dict(framework="megatron",global_step=step,world_size=8,contents=contents))
    (folder / "data.pt").write_bytes(b"loader")
    (folder / "driver.pt").write_bytes(b"driver")
    checkpoints.commit(folder, identity, arm, step, "sampler")
    return folder


def test_partial_newer_checkpoint_is_never_resumed(tmp_path):
    complete = checkpoint(tmp_path, 4)
    (tmp_path / "global_step_8/actor").mkdir(parents=True)
    assert checkpoints.latest(tmp_path, "id", "domain_mix", "sampler") == complete
    (complete / "data.pt").write_bytes(b"changed-loader")
    with pytest.raises(RuntimeError, match="changed"):
        checkpoints.latest(tmp_path, "id", "domain_mix", "sampler")


def test_weights_only_and_wrong_arm_are_rejected(tmp_path):
    p = checkpoint(tmp_path, 4)
    with pytest.raises(RuntimeError, match="another experiment"):
        checkpoints.latest(tmp_path, "id", "domain_only")
    (p / "actor/optimizer/shard.pt").unlink()
    with pytest.raises(RuntimeError, match="Missing checkpoint"):
        checkpoints.latest(tmp_path, "id", "domain_mix")


def test_adam_scheduler_rng_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(123)
    net = torch.nn.Linear(3, 2)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
    def step(model, opt, sched):
        x = torch.randn(5, 3)
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step(); sched.step()
    step(net, optim, scheduler)
    state = dict(model=net.state_dict(), optimizer=optim.state_dict(), scheduler=scheduler.state_dict(), rng=torch.get_rng_state())
    path = tmp_path / "toy.pt"
    torch.save(state, path)
    step(net, optim, scheduler)
    expected = copy.deepcopy(net.state_dict())
    resumed = torch.nn.Linear(3, 2)
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1.0)
    state = torch.load(path, weights_only=False)
    resumed.load_state_dict(state["model"]); opt2.load_state_dict(state["optimizer"])
    sched2.load_state_dict(state["scheduler"]); torch.set_rng_state(state["rng"])
    step(resumed, opt2, sched2)
    for key in expected:
        torch.testing.assert_close(resumed.state_dict()[key], expected[key], rtol=0, atol=0)


def running(manifest, tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    job = dict(folder=str(folder), job_name="0390_mp_tmix", job_id="12.pbs1")
    state = {"train:domain_mix": dict(phase="train", arm="domain_mix", status="R", job=job, pending=None)}
    return folder, job, state


@pytest.mark.parametrize("code,comment,retry", [(271,"terminated by root@login3",True),
    (265,"terminated by root@login1",True), (265,"application failure",False), (1,"error",False)])
def test_pbs_external_interruptions_only(manifest, tmp_path, code, comment, retry):
    folder, job, state = running(manifest, tmp_path)
    class PBS:
        def observe(self, j): return {"job_state":"F","Exit_status":code,"comment":comment}
    result = control.tick(manifest, state, PBS())["train:domain_mix"]
    assert result["status"] == ("READY" if retry else "BLOCKED")
    assert bool(result["job"]) != retry


def test_timeout_keeps_registered_job_without_resubmission(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    class PBS:
        def observe(self, j): raise Unknown("qstat timeout")
        def submit(self, *a): raise AssertionError("must not submit")
    result = control.tick(manifest, state, PBS())["train:domain_mix"]
    assert result["job"] == job
    assert "timeout" in result["observation_error"]


def test_ambiguous_qsub_reuses_same_pending_directory(manifest, tmp_path, monkeypatch):
    calls = []
    folder = tmp_path / "pending"
    folder.mkdir()
    pending = dict(folder=str(folder), job_name="0390_mp_tmix")
    monkeypatch.setattr(control, "new_attempt", lambda *args: pending)
    class PBS:
        def submit(self, folder, name):
            calls.append(folder)
            raise Unknown("ambiguous qsub")
    state = {"train:domain_mix": dict(phase="train", arm="domain_mix", status="READY", job=None)}
    control.tick(manifest, state, PBS()); control.tick(manifest, state, PBS())
    assert calls == [str(folder), str(folder)]
    assert state["train:domain_mix"]["pending"] == pending


def test_application_failure_does_not_become_infinite_271_retry(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "result.json", {"status":"FAIL", "error":"CUDA out of memory"})
    class PBS:
        def observe(self, j): return {"job_state":"F","Exit_status":271,"comment":"terminated"}
    assert control.tick(manifest, state, PBS())["train:domain_mix"]["status"] == "BLOCKED"


def test_qualification_requires_second_process_and_bound_success(manifest, tmp_path, monkeypatch):
    folder, job, _ = running(manifest, tmp_path)
    state = {"qualification:domain_mix":dict(phase="qualification", arm="domain_mix",status="R",job=job)}
    write(folder / "result.json", dict(status="RESUME_REQUIRED",step=4,identity=manifest["identity"],arm="domain_mix"))
    class PBS:
        def observe(self, job): return {"job_state":"F", "Exit_status":0}
        def submit(self, folder, name): return dict(folder=folder,job_name=name,job_id="13.pbs1")
    pbs = PBS()
    control.tick(manifest,state,pbs)
    assert state["qualification:domain_mix"]["status"] == "READY"
    assert not (tmp_path / "qualification/domain_mix/receipt.json").exists()
    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setattr(control,"new_attempt",lambda *a:dict(folder=str(second),job_name="0390_mp_qmix"))
    control.tick(manifest,state,pbs)
    write(second / "result.json",dict(status="PASS",step=8,resumed_from=4,identity=manifest["identity"],arm="domain_mix"))
    control.tick(manifest,state,pbs)
    receipt = read(tmp_path / "qualification/domain_mix/receipt.json")
    assert receipt["resumed_from"] == 4 and receipt["pbs_job"]["job_id"] == "13.pbs1"
    assert state["qualification:domain_mix"]["status"] == "DONE"


def test_stopped_training_is_rearmed_after_control_resumes(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "result.json",dict(status="CONTROL_PAUSED",identity=manifest["identity"]))
    class PBS:
        def observe(self, job): return {"job_state":"F", "Exit_status":0}
    control.tick(manifest,state,PBS())
    assert state["train:domain_mix"]["status"] == "READY"


def test_completed_resume_provenance_survives_lost_result(tmp_path):
    p = checkpoint(tmp_path,8)
    checkpoints.commit(p,"id","domain_mix",8,"sampler",resumed_from=4)
    loaded = checkpoints.latest(tmp_path,"id","domain_mix","sampler")
    assert read(loaded / "mopd_complete.json")["resumed_from"] == 4


def validation(run, step, score, identity="id", arm="domain_mix"):
    from d4_mopd.metrics import SELECTION_KEY
    ref = checkpoints.evaluation_reference(run / "checkpoints", identity, arm, step, "sampler")
    row = dict(status="PASS", split="validation", identity=identity, arm=arm, step=step,
               metrics={SELECTION_KEY:score}, **ref)
    write(run / "validation" / f"step_{step}.json", row)
    return row


def test_best_checkpoint_not_latest_and_never_selected_by_test(tmp_path):
    from d4_mopd.metrics import select_best
    for step, score in [(0,.4),(20,.7),(40,.7),(60,.6)]:
        if step:
            checkpoint(tmp_path / "checkpoints", step)
        validation(tmp_path, step, score)
    write(tmp_path / "test/step_60.json", dict(score=.99))
    best = select_best(tmp_path, "id", "domain_mix")
    assert best["step"] == 20 and best["score"] == .7
    assert checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler").name == "global_step_20"
    (tmp_path / "checkpoints/global_step_20/actor/model/shard.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler")


def test_base_can_win_selection_and_receipt_changes_are_rejected(tmp_path):
    from d4_mopd.metrics import select_best
    validation(tmp_path, 0, .8)
    best = select_best(tmp_path, "id", "domain_mix")
    assert checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler") is None
    validation(tmp_path, 0, .9)
    with pytest.raises(RuntimeError, match="receipt changed"):
        checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler")


def test_training_success_registers_test_and_test_271_retries_same_selection(manifest, tmp_path):
    from d4_mopd.metrics import select_best
    folder, job, state = running(manifest, tmp_path)
    train = tmp_path / "runs/domain_mix"
    validation(train, 0, .8, identity=manifest["identity"])
    best = select_best(train, manifest["identity"], "domain_mix")
    write(folder / "result.json", dict(status="PASS", step=200, identity=manifest["identity"], selection=best))
    class PBS:
        code = 0
        submissions = []
        def observe(self, job):
            return dict(job_state="F", Exit_status=self.code, comment="terminated by root@login3")
        def submit(self, folder, name):
            self.submissions.append(folder)
            return dict(folder=folder, job_name=name, job_id="13.pbs1")
    pbs = PBS()
    control.tick(manifest, state, pbs)
    assert state["train:domain_mix"]["status"] == "DONE"
    assert state["test:domain_mix"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "SUBMITTED", state["test:domain_mix"]
    attempt = Path(state["test:domain_mix"]["job"]["folder"])
    assert read(attempt / "request.json")["selection"] == best
    pbs.code = 271
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    second = Path(state["test:domain_mix"]["job"]["folder"])
    assert second != attempt and read(second / "request.json")["selection"] == best
    receipt = tmp_path / "final_tests/domain_mix/test/step_0.json"
    write(receipt, dict(status="PASS", split="test", identity=manifest["identity"], arm="domain_mix", step=0,
                        checkpoint_identity=best["checkpoint_identity"]))
    write(second / "result.json", dict(status="PASS", identity=manifest["identity"], step=0, selection=best,
                                       test_receipt=str(receipt)))
    pbs.code = 0
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "DONE"
    assert read(tmp_path / "final_tests/domain_mix/receipt.json")["selection"] == best
    count = len(pbs.submissions)
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == count


def test_full_validation_cache_survives_lost_best_pointer(manifest, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.metrics import SELECTION_KEY
    from d4_mopd.config import compose
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    calls = []
    def validate(_):
        calls.append(1)
        return {SELECTION_KEY:.6}
    monkeypatch.setattr(RayPPOTrainer, "_validate", validate)
    trainer._validate()
    (tmp_path / "best_validation.json").unlink()
    assert trainer._validate()[SELECTION_KEY] == .6
    assert len(calls) == 1 and read(tmp_path / "best_validation.json")["step"] == 0


def test_test_evaluation_does_not_write_validation_selection(manifest, tmp_path, monkeypatch):
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    from d4_mopd.metrics import SELECTION_KEY
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path,
                             test_checkpoint=dict(step=20,checkpoint="chosen",checkpoint_identity="bound"))
    trainer.global_steps = 20
    monkeypatch.setattr(RayPPOTrainer, "_validate", lambda _: {SELECTION_KEY:.99})
    trainer._validate()
    assert read(tmp_path / "test/step_20.json")["split"] == "test"
    assert not (tmp_path / "best_validation.json").exists()
    assert not (tmp_path / "validation").exists()


@pytest.mark.parametrize("evaluation", [False, True])
def test_native_load_adapter_uses_selected_step_and_skips_off_grid_validation(manifest, tmp_path, monkeypatch, evaluation):
    import random
    import numpy as np
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    from d4_mopd.metrics import select_best
    run = tmp_path / "runs/domain_mix"
    for step in (20,56):
        folder = checkpoint(run / "checkpoints", step, identity=manifest["identity"])
        torch.save(dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), folder / "driver.pt")
        checkpoints.commit(folder, manifest["identity"], "domain_mix", step, "sampler")
    validation(run, 20, .8, identity=manifest["identity"])
    selected = select_best(run, manifest["identity"], "domain_mix")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path / "test" if evaluation else run,
                             test_checkpoint=selected if evaluation else None)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    def native_load(self):
        assert self.config.trainer.resume_mode == "resume_path"
        self.global_steps = int(self.config.trainer.resume_from_path.rsplit("_",1)[-1])
    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", native_load)
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer._load_checkpoint()
    assert trainer.global_steps == (20 if evaluation else 56)
    assert trainer.config.trainer.val_before_train == evaluation


def pipeline(manifest):
    from d4_mopd import ARMS
    root = Path(manifest["manifest_path"]).parent
    write(root / "pipeline.json", dict(schema="d4-mopd-pipeline-v1", identity=manifest["identity"], arms=list(ARMS)))


def qualification_receipt(manifest, target_arm, **overrides):
    proof = dict(status="PASS", identity=manifest["identity"], scope="qualification", arm=target_arm, step=8, resumed_from=4)
    proof.update(overrides)
    write(control.task_root(manifest, "qualification", target_arm) / "receipt.json", proof)


@pytest.mark.parametrize("finish_order", [("domain_mix", "domain_only"), ("domain_only", "domain_mix")])
def test_parallel_qualifications_gate_both_fresh_training_arms(manifest, tmp_path, finish_order):
    pipeline(manifest)
    class PBS:
        submissions = []
        def submit(self, folder, name):
            self.submissions.append((folder, name))
            return dict(folder=folder, job_name=name, job_id=f"{len(self.submissions)}.pbs1")
        def observe(self, job):
            path = Path(job["folder"]) / "result.json"
            return dict(job_state="F", Exit_status=0) if path.exists() else dict(job_state="R")
    state, pbs = {}, PBS()
    control.tick(manifest, state, pbs)
    first_jobs = {arm: copy.deepcopy(state["qualification:" + arm]["job"]) for arm in finish_order}
    assert [n for _, n in pbs.submissions] == ["0390_mp_qmix", "0390_mp_qonly"]
    for arm in finish_order:
        key = "qualification:" + arm
        job = state[key]["job"]
        write(Path(job["folder"]) / "result.json", dict(status="RESUME_REQUIRED", step=4, identity=manifest["identity"], arm=arm))
        control.tick(manifest, state, pbs)
        assert state[key]["status"] == "READY"
        assert not any(k.startswith("train:") for k in state)
        control.tick(manifest, state, pbs)
        second = state[key]["job"]
        assert second != job
        write(Path(second["folder"]) / "result.json", dict(status="PASS", step=8, resumed_from=4, identity=manifest["identity"], arm=arm))
        control.tick(manifest, state, pbs)
        assert state[key]["status"] == "DONE"
        if arm == finish_order[0]:
            peer = state["qualification:" + finish_order[1]]
            assert peer["status"] == "R" and peer["job"] == first_jobs[finish_order[1]]
            assert not any(k.startswith("train:") for k in state)
    assert len(pbs.submissions) == 4
    assert state["train:domain_mix"]["status"] == state["train:domain_only"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    resumed_names = ["0390_mp_qmix" if a == "domain_mix" else "0390_mp_qonly" for a in finish_order]
    assert [n for _, n in pbs.submissions] == ["0390_mp_qmix", "0390_mp_qonly"] + resumed_names + ["0390_mp_tmix", "0390_mp_tonly"]
    # Formal attempts use separate empty checkpoint roots, never the qualification weights.
    for arm in ("domain_mix", "domain_only"):
        request = read(Path(state["train:"+arm]["job"]["folder"]) / "request.json")
        assert request["run_dir"] == str(tmp_path / "runs" / arm)
        assert not (Path(request["run_dir"]) / "checkpoints").exists()
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == 6
    # Finish both formal runs and exercise the automatic selected-model tests.
    from d4_mopd.metrics import select_best
    for arm in ("domain_mix", "domain_only"):
        train = tmp_path / "runs" / arm
        validation(train, 0, .8, identity=manifest["identity"], arm=arm)
        best = select_best(train, manifest["identity"], arm)
        folder = Path(state["train:"+arm]["job"]["folder"])
        write(folder / "result.json", dict(status="PASS", step=200, identity=manifest["identity"], arm=arm, selection=best))
    control.tick(manifest, state, pbs)
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == 8
    for arm in ("domain_mix", "domain_only"):
        folder = Path(state["test:"+arm]["job"]["folder"])
        best = read(folder / "request.json")["selection"]
        receipt = tmp_path / "final_tests" / arm / "test/step_0.json"
        write(receipt, dict(status="PASS", split="test", identity=manifest["identity"], arm=arm, step=0,
                            checkpoint_identity=best["checkpoint_identity"]))
        write(folder / "result.json", dict(status="PASS", step=0, identity=manifest["identity"], arm=arm,
                                            selection=best, test_receipt=str(receipt)))
    control.tick(manifest, state, pbs)
    control.tick(manifest, state, pbs)
    assert all(s["status"] == "DONE" for s in state.values()) and len(pbs.submissions) == 8


def test_pipeline_restart_repairs_gap_and_blocked_qualification_never_promotes(manifest):
    pipeline(manifest)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="BLOCKED")}
    control.advance_pipeline(manifest, state)
    assert len(state) == 2 and state["qualification:domain_only"]["status"] == "READY"
    state["qualification:domain_mix"]["status"] = "DONE"
    qualification_receipt(manifest, "domain_mix")
    control.advance_pipeline(manifest, state)
    assert state["qualification:domain_only"]["status"] == "READY"
    qualification_receipt(manifest, "domain_only")
    state["qualification:domain_only"]["status"] = "DONE"
    control.advance_pipeline(manifest, state)
    state["train:domain_mix"].update(status="R", job={"job_id":"keep-this-job"})
    control.advance_pipeline(manifest, state)
    assert state["train:domain_mix"]["job"]["job_id"] == "keep-this-job"


@pytest.mark.parametrize("bad", [{"identity":"other"}, {"resumed_from":0}, {"step":4}, {"arm":"domain_only"}])
def test_pipeline_requires_bound_full_qualification_receipt(manifest, bad):
    pipeline(manifest)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="DONE")}
    qualification_receipt(manifest, "domain_mix", **bad)
    with pytest.raises(RuntimeError, match="qualification/reload"):
        control.advance_pipeline(manifest, state)
    assert set(state) == {"qualification:domain_mix", "qualification:domain_only"}


def test_pipeline_stop_prevents_submissions_and_automatic_transitions(manifest, tmp_path):
    pipeline(manifest)
    write(tmp_path / "STOP", {})
    class PBS:
        def submit(self, *args): raise AssertionError("stopped")
    state = {}
    control.tick(manifest, state, PBS())
    assert state == {}


def test_wrong_arm_job_success_is_not_a_qualification_pass(manifest, tmp_path):
    folder, job, _ = running(manifest, tmp_path)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="R", job=job)}
    write(folder / "result.json", dict(status="PASS", identity=manifest["identity"], arm="domain_only", step=8, resumed_from=4))
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=0)
    control.tick(manifest, state, PBS())
    assert state["qualification:domain_mix"]["status"] == "BLOCKED"
    assert not (tmp_path / "qualification/domain_mix/receipt.json").exists()


def test_pipeline_start_persists_intent_is_idempotent_and_respects_stop(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    monkeypatch.setattr(control, "active", lambda _: True)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], native_abi={"status":"PASS"}))
    for _ in range(2):
        result = control.start(manifest["manifest_path"], "pipeline", ("domain_mix", "domain_only"))
        assert result["automatic_pipeline"]
    assert list(read(tmp_path / "states.json")) == ["qualification:domain_mix", "qualification:domain_only"]
    assert read(tmp_path / "pipeline.json")["arms"] == ["domain_mix", "domain_only"]
    with pytest.raises(RuntimeError, match="another identity/arms"):
        control.start(manifest["manifest_path"], "pipeline", ("domain_mix",))
    write(tmp_path / "STOP", {})
    with pytest.raises(RuntimeError, match="stopped"):
        control.start(manifest["manifest_path"], "pipeline", ("domain_mix", "domain_only"))


def legacy_advance_pipeline_v1(m, state):
    # Frozen 63d59e2 dispatcher. tick/guardian are unchanged by the parallel
    # registration fix; exercise an existing guardian without replacing runtime.
    from d4_mopd import ARMS
    path = Path(m["manifest_path"]).parent / "pipeline.json"
    if not path.exists():
        return
    p = read(path)
    control.need(p["schema"] == "d4-mopd-pipeline-v1" and p["identity"] == m["identity"]
                 and p["arms"] == [a for a in ARMS if a in p["arms"]] and p["arms"],
                 "Pipeline identity or arm order changed")
    for arm in p["arms"]:
        s = state.setdefault("qualification:" + arm,
                             dict(phase="qualification", arm=arm, status="READY", job=None, pending=None))
        control.need(s["phase"] == "qualification" and s["arm"] == arm, "Pipeline qualification task changed")
        if s["status"] != "DONE":
            return
        control.qualification_passed(m, arm)
    for arm in p["arms"]:
        state.setdefault("train:" + arm, dict(phase="train", arm=arm, status="READY", job=None, pending=None))


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("registration", ["pipeline", "qualification"])
def test_existing_frozen_guardian_adds_only_without_restarting_mix(manifest, tmp_path, monkeypatch, pending, registration):
    pipeline(manifest)
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], native_abi={"status":"PASS"}))
    frozen = tmp_path / "runtime/d4_mopd/control.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen runtime must remain unchanged\n")
    protected = [tmp_path / n for n in ("manifest.json", "pipeline.json", "cpu_preflight.json")] + [frozen]
    original = {p: p.read_bytes() for p in protected}
    attempt = control.new_attempt(manifest, "qualification", "domain_mix")
    job = dict(attempt, job_id="2893794.pbs1")
    Path(attempt["folder"], "job_id.txt").write_text(job["job_id"])
    mix = dict(phase="qualification", arm="domain_mix", status="READY" if pending else "Q",
               job=None if pending else job, pending=attempt if pending else None)
    write(tmp_path / "states.json", {"qualification:domain_mix": mix})
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    monkeypatch.setattr(control, "active", lambda _: True)
    def no_restart(*args, **kwargs):
        raise AssertionError("must reuse the existing guardian")
    monkeypatch.setattr(control.subprocess, "Popen", no_restart)
    for _ in range(2):
        result = control.start(manifest["manifest_path"], registration, ("domain_mix", "domain_only"))
        assert result["automatic_pipeline"]
        assert read(tmp_path / "states.json")["qualification:domain_mix"] == mix
    class PBS:
        submissions = []
        def submit(self, folder, name):
            # An accepted-but-pending qsub is recovered by its original path.
            identity = Path(folder) / "job_id.txt"
            if identity.exists():
                return dict(folder=folder, job_name=name, job_id=identity.read_text())
            self.submissions.append(name)
            return dict(folder=folder, job_name=name, job_id=f"new-{len(self.submissions)}.pbs1")
        def observe(self, job):
            return dict(job_state="Q")
    # The updated checkout registers tasks; the already-running guardian keeps
    # its frozen dispatcher and consumes the same shared state table.
    monkeypatch.setattr(control, "advance_pipeline", legacy_advance_pipeline_v1)
    state, pbs = read(tmp_path / "states.json"), PBS()
    control.tick(manifest, state, pbs)
    assert state["qualification:domain_mix"]["job"] == job
    only_job = copy.deepcopy(state["qualification:domain_only"]["job"])
    control.tick(manifest, state, pbs)
    assert state["qualification:domain_only"]["job"] == only_job
    assert pbs.submissions == ["0390_mp_qonly"]
    # Only may finish first under the old guardian; formal training still waits.
    qualification_receipt(manifest, "domain_only")
    state["qualification:domain_only"]["status"] = "DONE"
    control.tick(manifest, state, pbs)
    assert not any(k.startswith("train:") for k in state)
    qualification_receipt(manifest, "domain_mix")
    state["qualification:domain_mix"]["status"] = "DONE"
    control.tick(manifest, state, pbs)
    assert pbs.submissions == ["0390_mp_qonly", "0390_mp_tmix", "0390_mp_tonly"]
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("outcome", ["unknown", "271", "265", "application_fail"])
def test_parallel_peer_continues_during_interruption_or_observation_error(manifest, tmp_path, outcome):
    pipeline(manifest)
    class PBS:
        submissions = []
        interrupted = False
        def submit(self, folder, name):
            self.submissions.append(name)
            return dict(folder=folder, job_name=name, job_id=f"{len(self.submissions)}.pbs1")
        def observe(self, job):
            if job["job_id"] == "1.pbs1":
                if outcome == "unknown":
                    raise Unknown("qstat timeout")
                if not self.interrupted:
                    self.interrupted = True
                    return dict(job_state="F", Exit_status=271 if outcome == "application_fail" else int(outcome),
                                comment="terminated by root@login3")
            return dict(job_state="R")
    state, pbs = {}, PBS()
    control.tick(manifest, state, pbs)
    mix, only = state["qualification:domain_mix"], state["qualification:domain_only"]
    old_mix, old_only = copy.deepcopy(mix["job"]), copy.deepcopy(only["job"])
    if outcome == "application_fail":
        write(Path(old_mix["folder"]) / "result.json", dict(status="FAIL", error="explicit failure"))
    control.tick(manifest, state, pbs)
    assert only["status"] == "R" and only["job"] == old_only
    if outcome in ("271", "265"):
        assert mix["status"] == "READY" and mix["job"] is None
        assert mix["external_interruptions"] == 1
    else:
        assert mix["job"] == old_mix
        assert mix["status"] == ("BLOCKED" if outcome == "application_fail" else "SUBMITTED")
    control.tick(manifest, state, pbs)
    assert pbs.submissions == ["0390_mp_qmix", "0390_mp_qonly"] + (["0390_mp_qmix"] if outcome in ("271", "265") else [])
    assert only["job"] == old_only
    assert not any(k.startswith("train:") for k in state)


def test_retention_keeps_two_latest_and_best_with_complete_history(tmp_path):
    from d4_mopd.metrics import select_best
    root = tmp_path / "checkpoints"
    validation(tmp_path, 0, .4)
    for step in range(5, 36, 5):
        checkpoint(root, step)
        if step % 10 == 0:
            validation(tmp_path, step, .8 if step == 10 else .6)
    # Simulate interruption before the best pointer was published.
    assert not (tmp_path / "best_validation.json").exists()
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [10, 30, 35]
    assert result["deleted_steps"] == [5, 15, 20, 25]
    assert len(list((tmp_path / "validation").glob("*.json"))) == 4
    assert checkpoints.latest(root, "id", "domain_mix", "sampler").name == "global_step_35"
    best = select_best(tmp_path, "id", "domain_mix")
    assert checkpoints.restore_selected(best, root, "id", "domain_mix", "sampler").name == "global_step_10"
    checkpoint(root, 40)
    validation(tmp_path, 40, .9)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [35, 40]
    assert 10 in result["deleted_steps"]


def test_retention_base_best_and_protected_final_test_selection(tmp_path):
    from d4_mopd.metrics import select_best
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15, 20):
        checkpoint(root, step)
    validation(tmp_path, 10, .8)
    selected = select_best(tmp_path, "id", "domain_mix")
    write(tmp_path / "receipt.json", dict(status="PASS", identity="id", arm="domain_mix", selection=selected))
    # A frozen final-test request must remain loadable even if a later receipt
    # changes the current best; this is defensive and not the normal train order.
    validation(tmp_path, 0, .9)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 20)
    assert result["retained_steps"] == [10, 15, 20] and result["best_step"] == 0
    assert checkpoints.restore_selected(selected, root, "id", "domain_mix", "sampler").name == "global_step_10"


def test_retention_interrupted_deletion_cannot_corrupt_resume_and_is_finished_on_restart(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15):
        checkpoint(root, step)
    validation(tmp_path, 0, .8)
    original = checkpoints.shutil.rmtree
    def interrupted(path):
        (path / "actor/model/shard.pt").unlink()
        raise InterruptedError("simulated external kill during deletion")
    monkeypatch.setattr(checkpoints.shutil, "rmtree", interrupted)
    with pytest.raises(InterruptedError):
        checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert not (root / "global_step_5").exists()
    assert checkpoints.latest(root, "id", "domain_mix", "sampler").name == "global_step_15"
    monkeypatch.setattr(checkpoints.shutil, "rmtree", original)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [10, 15] and result["deleted_steps"] == [5]
    assert not list((root / ".retention_trash").iterdir())


def test_native_restart_retries_interrupted_first_validation(manifest, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    (tmp_path / "validation").mkdir()
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", lambda _: None)
    trainer._load_checkpoint()
    assert trainer.config.trainer.resume_mode == "disable" and trainer.config.trainer.val_before_train
    assert not (tmp_path / "retention.json").exists()


@pytest.mark.parametrize("corruption", ["wrong_arm", "latest_state", "missing_best", "symlink"])
def test_retention_refuses_deletion_when_identity_or_protected_state_is_invalid(tmp_path, corruption):
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15, 20):
        checkpoint(root, step)
    validation(tmp_path, 5, .8)
    if corruption == "wrong_arm":
        path = root / "global_step_10/mopd_complete.json"
        write(path, dict(read(path), arm="domain_only"))
    elif corruption == "latest_state":
        (root / "global_step_20/data.pt").write_bytes(b"corrupted")
    elif corruption == "missing_best":
        (root / "global_step_5/mopd_complete.json").unlink()
    else:
        (root / "global_step_10").rename(tmp_path / "outside")
        (root / "global_step_10").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert all((root / f"global_step_{s}/data.pt").exists() for s in (5,10,15,20))
