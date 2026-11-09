from pathlib import Path
import copy
import pytest
import torch

from d4_mopd import checkpoints, control
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import read, write
from d4_training_continuation.scheduler import Unknown


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
    write(folder / "result.json", dict(status="RESUME_REQUIRED",step=4,identity=manifest["identity"]))
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
    write(second / "result.json",dict(status="PASS",step=8,resumed_from=4,identity=manifest["identity"]))
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
