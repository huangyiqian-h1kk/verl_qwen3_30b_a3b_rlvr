"""Native CPU boundary tests for full qualification and restart control flow.

Only the remote GPU worker is replaced by a file-writing double. The pinned
Hydra config, dataset factory, native save/load/fit control flow, RNG objects,
and MOPD commit/receipt functions execute directly; no Ray cluster is started.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from d4_mopd import ARMS, DOMAINS, checkpoints, execution, integration, phase_runtime
from d4_mopd.config import compose
from d4_mopd.integration import MOPDTrainer, QualificationPause
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import read, write
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, SkipManager
from verl.trainer.ppo.utils import create_rl_dataset


@pytest.fixture
def executed(manifest):
    write(manifest["manifest_path"], manifest)
    record = execution.make_record(manifest, "single_node", {"fixture.py": "a" * 64},
                                   qualification_transition=["dual_node", "single_node"])
    write(record["record_path"], record)
    return dict(manifest, execution=record)


class NativeActorFiles:
    def __init__(self):
        self.loaded = []

    def save_checkpoint(self, local, remote, step, max_ckpt_to_keep=None):
        actor = Path(local)
        contents = {}
        for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
            path = actor / key / "rank0.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((key + " CPU stand-in for GPU native state").encode())
            contents[key] = dict(path=key)
        write(actor / "ckpt_contents.json",
              dict(framework="megatron", global_step=step, world_size=8, contents=contents))

    def load_checkpoint(self, path, del_local_after_load=False):
        assert not del_local_after_load
        self.loaded.append(path)


class Loader:
    def __init__(self, sampler):
        self.sampler = sampler
        self.loaded = []

    def __len__(self):
        return 200

    def __iter__(self):
        raise AssertionError("Endpoint recovery must never consume another training batch")

    def state_dict(self):
        return dict(sampler=self.sampler.state_dict(), fixture="native-loader-state")

    def load_state_dict(self, state):
        self.loaded.append(state)
        self.sampler.load_state_dict(state["sampler"])


def trainer_for(config, rows, loaded=0):
    trainer = object.__new__(MOPDTrainer)
    trainer.config = config
    trainer.global_steps = loaded
    trainer.mopd_loaded_step = loaded
    trainer.use_critic = False
    trainer.actor_rollout_wg = NativeActorFiles()
    spec = OmegaConf.to_container(config.mopd.effective_spec, resolve=True)
    trainer.train_dataloader = Loader(PlannedSampler(rows, spec, config.mopd.arm))
    trainer.phase_runtime = SimpleNamespace(assert_training_ready=lambda: None)
    return trainer


def save_boundary(executed, tmp_path, rows, monkeypatch, step, attempt):
    selected = "dual_node" if step == 4 else "single_node"
    c = compose(executed, "domain_mix", tmp_path / "qualified", qualification=True,
                execution_profile=selected)
    trainer = trainer_for(c, rows, loaded=0 if step == 4 else 4)
    trainer.global_steps = step
    trainer.train_dataloader.sampler.cursor = step * 2048
    monkeypatch.setenv("D4_MOPD_ATTEMPT", attempt)
    if step == 4:
        with pytest.raises(QualificationPause):
            trainer._save_checkpoint()
    else:
        trainer._save_checkpoint()
    return trainer


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("name", execution.PROFILE_NAMES)
def test_full_qualification_preserves_real_batch_budgets_and_sampler(executed, tmp_path, rows, arm, name):
    c = compose(executed, arm, tmp_path, qualification=True, execution_profile=name)
    formal = compose(executed, arm, tmp_path, execution_profile=name)
    assert c.data.train_batch_size == c.actor_rollout_ref.actor.ppo_mini_batch_size == 2048
    assert c.actor_rollout_ref.rollout.n == 1
    assert c.data.max_response_length == 65535 and not c.data.mopd_qualification
    assert (c.trainer.total_training_steps, c.trainer.save_freq, c.trainer.test_freq) == (8, 4, 4)
    assert c.mopd.effective_spec.updates == 200
    assert OmegaConf.to_container(c.mopd.effective_spec) == executed["spec"]
    assert PlannedSampler(rows, OmegaConf.to_container(c.mopd.effective_spec), arm).identity == PlannedSampler(
        rows, OmegaConf.to_container(formal.mopd.effective_spec), arm).identity
    for key in ("ppo_micro_batch_size_per_gpu", "ppo_max_token_len_per_gpu", "ppo_epochs", "loss_agg_mode"):
        assert c.actor_rollout_ref.actor[key] == formal.actor_rollout_ref.actor[key]
    for key in ("max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "max_model_len"):
        assert c.actor_rollout_ref.rollout[key] == formal.actor_rollout_ref.rollout[key]
    assert set(c.distillation.teacher_models) - {"teacher_model"} == set(DOMAINS)
    assert phase_runtime.DOMAINS == DOMAINS


def test_native_dataset_keeps_near_64k_budget_for_full_qualification(executed, tmp_path):
    rows = [dict(domain=d, sample_id=d, source="source", messages_json='[{"role":"user","content":"x"}]',
                 tools_json="[]", verifier_json="{}", prompt_tokens=12, prompt_token_hash="hash",
                 generation_max_tokens=65524, mopd_split="train") for d in DOMAINS]
    pq.write_table(pa.Table.from_pylist(rows), executed["data"]["train"][0])
    full = compose(executed, "domain_mix", tmp_path / "full", qualification=True)
    actual = create_rl_dataset(full.data.train_files, full.data, None, None)
    assert [actual[i]["extra_info"]["generation_max_tokens"] for i in range(len(actual))] == [65524] * 4
    legacy = compose({k: v for k, v in executed.items() if k != "execution"}, "domain_mix",
                     tmp_path / "legacy", qualification=True)
    old = create_rl_dataset(legacy.data.train_files, legacy.data, None, None)
    assert legacy.data.train_batch_size == 8
    assert old[0]["extra_info"]["generation_max_tokens"] == 256


def test_save4_pause_native_reload_preserves_cursor_rng_and_real_profile_trace(executed, tmp_path, rows, monkeypatch):
    random.seed(19)
    np.random.seed(20)
    torch.manual_seed(21)
    first = save_boundary(executed, tmp_path, rows, monkeypatch, 4, "source-attempt")
    folder = Path(first.config.trainer.default_local_dir) / "global_step_4"
    marker = read(folder / "mopd_complete.json")
    assert marker["status"] == "PASS" and marker["resumed_from"] == 0
    assert "qualification_segment.json" in marker["contents"]
    trace = read(folder / "qualification_segment.json")
    assert (trace["profile"], trace["attempt_id"], trace["loaded_step"], trace["completed_step"]) == (
        "dual_node", "source-attempt", 0, 4)
    saved_torch = torch.get_rng_state().clone()
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    random.random()
    np.random.rand()
    torch.rand(1)
    resumed_config = compose(executed, "domain_mix", tmp_path / "qualified", qualification=True,
                             execution_profile="single_node")
    resumed = trainer_for(resumed_config, rows)
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "target-attempt")
    resumed._load_checkpoint()
    assert resumed.global_steps == resumed.mopd_loaded_step == 4
    assert resumed.train_dataloader.sampler.cursor == 4 * 2048
    assert len(resumed.actor_rollout_wg.loaded) == len(resumed.train_dataloader.loaded) == 1
    assert resumed.config.trainer.val_before_train and not resumed.config.trainer.val_only
    assert torch.equal(torch.get_rng_state(), saved_torch)
    assert random.getstate() == saved_python
    assert np.array_equal(np.random.get_state()[1], saved_numpy[1])


def test_partial8_reloads_committed4_and_rejects_wrong_profile(executed, tmp_path, rows, monkeypatch):
    first = save_boundary(executed, tmp_path, rows, monkeypatch, 4, "A")
    (Path(first.config.trainer.default_local_dir) / "global_step_8/actor").mkdir(parents=True)
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "retry-B")
    c = compose(executed, "domain_mix", tmp_path / "qualified", qualification=True,
                execution_profile="single_node")
    retried = trainer_for(c, rows)
    retried._load_checkpoint()
    assert retried.global_steps == 4
    wrong = compose(executed, "domain_mix", tmp_path / "qualified", qualification=True,
                    execution_profile="dual_node")
    with pytest.raises(RuntimeError, match="Wrong profile"):
        trainer_for(wrong, rows)._load_checkpoint()


def test_complete8_retry_native_fit_only_validates_and_keeps_original_segments(executed, tmp_path, rows, monkeypatch):
    save_boundary(executed, tmp_path, rows, monkeypatch, 4, "A")
    second = save_boundary(executed, tmp_path, rows, monkeypatch, 8, "B")
    before = integration.qualification_segments(second.config)
    retry = trainer_for(second.config, rows)
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "C-after-final-commit")
    retry._dump_executor = SimpleNamespace(_shutdown=False)
    events = []
    retry.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append(("sync", step)))
    retry._validate = lambda: events.append(("validation", retry.global_steps)) or {"score": .5}
    retry._shutdown_dump_executor = lambda: events.append("shutdown")
    from verl.utils import tracking
    monkeypatch.setattr(tracking, "Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(SkipManager, "init", lambda *args: None)
    retry.fit()  # Actual pinned fit reaches val_only before the training iterator.
    assert events == [("sync", 8), ("validation", 8), "shutdown"]
    assert retry.global_steps == 8 and retry.config.trainer.val_only
    after = integration.qualification_segments(retry.config)
    assert after == before
    assert [s["attempt_id"] for s in after["qualification_segments"]] == ["A", "B"]


def test_complete200_formal_retry_cannot_execute_update201(executed, tmp_path, rows, monkeypatch):
    c = compose(executed, "domain_mix", tmp_path / "formal", execution_profile="single_node")
    trainer = trainer_for(c, rows)
    folder = Path(c.trainer.default_local_dir) / "global_step_200"
    trainer.actor_rollout_wg.save_checkpoint(folder / "actor", None, 200)
    trainer.train_dataloader.sampler.cursor = 200 * 2048
    torch.save(trainer.train_dataloader.state_dict(), folder / "data.pt")
    torch.save(dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()),
               folder / "driver.pt")
    checkpoints.commit(folder, c.mopd.identity, c.mopd.arm, 200, trainer.train_dataloader.sampler.identity)
    reference = checkpoints.evaluation_reference(c.trainer.default_local_dir, c.mopd.identity,
                                                  c.mopd.arm, 0, trainer.train_dataloader.sampler.identity)
    write(Path(c.mopd.run_dir) / "validation/step_0.json",
          dict(status="PASS", identity=c.mopd.identity, arm=c.mopd.arm, step=0,
               split="validation", test_used_for_selection=False,
               metrics={integration.SELECTION_KEY: .5}, **reference))
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "after-final200")
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    events = []
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append(("sync", step)))
    trainer._validate = lambda: events.append(("validation", trainer.global_steps)) or {"score": .5}
    trainer._shutdown_dump_executor = lambda: events.append("shutdown")
    from verl.utils import tracking
    monkeypatch.setattr(tracking, "Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(SkipManager, "init", lambda *args: None)
    trainer.fit()
    assert events == [("sync", 200), ("validation", 200), "shutdown"]
    assert trainer.global_steps == 200 and trainer.config.trainer.val_only
    assert (Path(c.mopd.run_dir) / "attempts/after-final200/resume_lineage.json").is_file()


@pytest.mark.parametrize("corruption", ["trace", "status", "sampler", "missing_native_optimizer"])
def test_qualification_segments_reject_incompatible_or_incomplete_first_save(
        executed, tmp_path, rows, monkeypatch, corruption):
    first = save_boundary(executed, tmp_path, rows, monkeypatch, 4, "A")
    second = save_boundary(executed, tmp_path, rows, monkeypatch, 8, "B")
    folder = Path(first.config.trainer.default_local_dir) / "global_step_4"
    marker = folder / "mopd_complete.json"
    proof = read(marker)
    if corruption == "trace":
        segment = read(folder / "qualification_segment.json")
        write(folder / "qualification_segment.json", dict(segment, profile="single_node"))
    elif corruption == "status":
        write(marker, dict(proof, status="FAIL"))
    elif corruption == "sampler":
        write(marker, dict(proof, sampler_identity="unrelated-data-sampler"))
    else:
        (folder / "actor/optimizer/rank0.bin").unlink()
        write(marker, dict(proof, contents=checkpoints.inventory(folder)))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        integration.qualification_segments(second.config)


@pytest.mark.parametrize("loaded", [4, 8])
def test_runner_final_result_uses8_endpoint_not_scientific200_and_keeps_reload_proof(
        executed, tmp_path, rows, monkeypatch, loaded):
    save_boundary(executed, tmp_path, rows, monkeypatch, 4, "A")
    second = save_boundary(executed, tmp_path, rows, monkeypatch, 8, "B")
    config = second.config
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "result-retry")
    actual_rows_seen = []

    class CompletedTrainer:
        def __init__(self, **kwargs):
            self.mopd_loaded_step = loaded
            self.global_steps = 9 if loaded == 4 else 8
            actual_rows_seen.append(list(kwargs["train_dataset"].rows))

        def init_workers(self):
            pass

        def fit(self):
            pass

    monkeypatch.setattr(integration, "MOPDTrainer", CompletedTrainer)
    monkeypatch.setattr(integration, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(integration, "omega_conf_to_dataclass", lambda *args: SimpleNamespace(tokenizer=None, processor=None))
    monkeypatch.setattr(integration, "create_rl_dataset", lambda *args, **kwargs: SimpleNamespace(rows=deepcopy(rows)))
    cls = integration.MOPDTaskRunner.__ray_metadata__.modified_class
    runner = object.__new__(cls)
    runner.add_actor_rollout_worker = lambda _: (None, object)
    runner.add_teacher_model_resource_pool = lambda _: None
    runner.add_reward_model_resource_pool = lambda _: None
    runner.init_resource_pool_mgr = lambda _: None
    runner.role_worker_mapping = {}
    result = runner.run(config)
    assert result["status"] == "PASS" and result["step"] == 8 and result["resumed_from"] == 4
    assert config.mopd.effective_spec.updates == 200
    assert actual_rows_seen == [rows]  # Full qualifier did not truncate to eight prompts.
    assert [s["attempt_id"] for s in result["qualification_segments"]] == ["A", "B"]
    execution.validate_qualification(dict(result, scope="qualification"), executed, "domain_mix")


@pytest.mark.parametrize("profile", ["dual_node", "single_node"])
def test_worker_initialization_wraps_only_shared_profile_and_binds_after_native_init(
        executed, tmp_path, monkeypatch, profile):
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "worker-init")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(executed, "domain_mix", tmp_path, execution_profile=profile)
    events = []

    @contextmanager
    def hook(config):
        events.append("park-hook-enter")
        yield
        events.append("park-hook-exit")

    monkeypatch.setattr(phase_runtime, "teacher_initialization_hook", hook)
    monkeypatch.setattr(phase_runtime, "bind_phase_runtime", lambda _: events.append("bind"))
    monkeypatch.setattr(RayPPOTrainer, "init_workers", lambda _: events.append("native-init"))
    trainer.init_workers()
    assert events == (["native-init"] if profile == "dual_node" else [
        "park-hook-enter", "native-init", "park-hook-exit", "bind"])
    trace = tmp_path / "attempts/worker-init/phase_events.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [(row["stage"], row["event"]) for row in records] == [
        ("worker_initialization", "START"), ("worker_initialization", "FINISH")]
    assert records[1]["monotonic_ns"] >= records[0]["monotonic_ns"]
    assert records[1]["elapsed_seconds"] >= 0


def test_legacy_qualification_save_stays_short_and_has_no_execution_segment(manifest, tmp_path, rows, monkeypatch):
    c = compose(manifest, "domain_mix", tmp_path / "legacy", qualification=True)
    trainer = trainer_for(c, rows)
    trainer.global_steps = 4
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "old-A")
    with pytest.raises(QualificationPause):
        trainer._save_checkpoint()
    folder = Path(c.trainer.default_local_dir) / "global_step_4"
    assert not (folder / "qualification_segment.json").exists()
    assert integration.qualification_segments(c) == integration.execution_result(c) == {}
    assert c.mopd.effective_spec.train_batch_size == c.data.train_batch_size == 8
    assert c.mopd.effective_spec.updates == c.trainer.total_training_steps == 8
