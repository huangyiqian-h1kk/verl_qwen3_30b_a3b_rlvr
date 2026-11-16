"""Small TaskRunner/Trainer adapters; native generation, scoring, loss and optimizer."""
from collections import Counter
from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from . import DOMAINS
from . import checkpoints
from .execution import qualification_fields, inspect_resume
from .spec import PlannedSampler, domain_order
from .metrics import SELECTION_KEY, select_best
from d4_teachers.common import digest, need, now, read, write
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import create_rl_dataset
from verl.utils.config import omega_conf_to_dataclass, validate_config


class QualificationPause(Exception):
    """A deliberate checkpoint boundary; a second process must prove reload."""


class ControlPause(Exception):
    """Stop at an update boundary and resume from the last committed save."""


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def execution_manifest(config):
    """The driver receives the execution identity already checked by execute()."""
    c = config.mopd
    m = read(c.manifest_path)
    if c.get("execution_identity"):
        record = read(Path(c.control_root) / "record.json")
        need(record["identity"] == c.identity and record["execution_identity"] == c.execution_identity
             and record["runtime_identity"] == c.runtime_identity, "Driver execution identity changed")
        m["execution"] = record
    return m


def execution_result(config):
    c = config.mopd
    if not c.get("execution_identity"):
        return {}
    return dict(execution_identity=c.execution_identity, runtime_identity=c.runtime_identity,
                execution_profile=c.execution_profile)


def profile_stage(trainer, stage):
    c = trainer.config.mopd
    if not getattr(c, "execution_identity", None):
        return nullcontext()
    from .phase_runtime import StageRecorder
    path = Path(c.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "phase_events.jsonl"
    return StageRecorder(path).stage(stage, step=getattr(trainer, "global_steps", 0),
                                    execution_profile=c.execution_profile)


def qualification_segments(config):
    """Read actual segment evidence committed atomically with the native states."""
    c = config.mopd
    if not c.get("full_qualification", False):
        return {}
    m = execution_manifest(config)
    segments = []
    sampler_identity = None
    for step, start, expected in zip((4, 8), (0, 4), m["execution"]["qualification_transition"]):
        folder = Path(config.trainer.default_local_dir) / f"global_step_{step}"
        proof = read(folder / "mopd_complete.json")
        checkpoints.validate_native(folder, step)
        if sampler_identity is None:
            sampler_identity = proof["sampler_identity"]
        need(proof.get("status") == "PASS" and proof["identity"] == c.identity and proof["arm"] == c.arm
             and proof["step"] == step and proof["resumed_from"] == start
             and proof["sampler_identity"] == sampler_identity
             and checkpoints.inventory(folder) == proof["contents"],
             "Qualification checkpoint evidence changed")
        segment = read(folder / "qualification_segment.json")
        need(segment["execution_identity"] == c.execution_identity
             and segment["runtime_identity"] == c.runtime_identity
             and segment["profile"] == expected and segment["loaded_step"] == start
             and segment["completed_step"] == step, "Qualification did not execute its selected transition")
        segments.append(segment)
    need(segments[0]["attempt_id"] != segments[1]["attempt_id"], "Qualification did not restart")
    return dict(**qualification_fields(m), qualification_segments=segments)


def check_batch(batch, spec, arm, step, expected_ids):
    domains = list(batch.non_tensor_batch["data_source"])
    b = spec["train_batch_size"]
    need(len(domains) == b, "Unexpected optimizer batch size")
    counts = Counter(domains)
    if arm == "domain_mix":
        need(counts == {d: b//4 for d in DOMAINS}, "Mixed batch changed domain proportions")
    else:
        target = domain_order(spec)[step-1]
        need(counts == {target: b}, "Domain-only optimizer update mixes domains or changes schedule")
    # verl moves top-level index into the generation request. extra_info stays
    # with the optimizer batch through rollout, repeat and token balancing.
    extra = batch.non_tensor_batch.get("extra_info", ())
    need(len(extra) == b and all(isinstance(v, dict) and isinstance(v.get("sample_id"), str)
                                and v["sample_id"] for v in extra), "Missing stable prompt IDs")
    ids = [v["sample_id"] for v in extra]
    need(Counter(ids) == Counter(expected_ids), "DataLoader resumed at the wrong prompt position")
    mask = batch.batch["response_mask"].bool()
    need(bool(mask.any(dim=-1).all()), "Fully masked response")
    # Native teacher outputs have already been shifted to predict the next token.
    p, r = batch.batch["prompts"].shape[1], mask.shape[1]
    lp = batch.batch["teacher_logprobs"][:, p-1:p+r-1, 0]
    token = batch.batch["teacher_ids"][:, p-1:p+r-1, 0]
    need(lp.shape == mask.shape and bool(torch.isfinite(lp[mask]).all()), "Missing/nonfinite teacher signal")
    need(bool((token[mask] == batch.batch["responses"][mask]).all()), "Teacher/student token alignment mismatch")
    old = batch.batch["old_log_probs"]
    need(bool(torch.isfinite(old[mask]).all()), "Nonfinite student signal")
    need(bool((batch.batch["token_level_rewards"] == 0).all()), "Task reward entered pure MOPD batch")
    gap = lp - old
    stats = {}
    for d in counts:
        dm = torch.tensor([v == d for v in domains], device=mask.device)
        valid = mask[dm]
        stats[d] = dict(prompts=counts[d], response_tokens=int(valid.sum()),
                        mean_response_length=float(valid.sum(-1).float().mean()),
                        mean_teacher_minus_student=float(gap[dm][valid].mean()),
                        pg_clipped_fraction=float((gap[dm][valid].abs() > 5).float().mean()))
    return dict(domains=stats, sample_ids_sha256=digest(sorted(ids)))


class MOPDTrainer(RayPPOTrainer):
    def init_workers(self):
        from .phase_runtime import is_phased_config, teacher_initialization_hook, bind_phase_runtime
        with profile_stage(self, "worker_initialization"):
            if not is_phased_config(self.config):
                return super().init_workers()
            with teacher_initialization_hook(self.config):
                super().init_workers()
        bind_phase_runtime(self)

    def _compute_old_log_prob(self, batch):
        from .phase_runtime import assert_training_ready
        assert_training_ready(self)
        with profile_stage(self, "student_old_logprob"):
            return super()._compute_old_log_prob(batch)

    def _prune_checkpoints(self):
        c = self.config.mopd
        if not c.qualification and c.test_checkpoint is None:
            checkpoints.prune(c.run_dir, c.identity, c.arm, self.train_dataloader.sampler.identity,
                              c.effective_spec.checkpoint_keep_recent, c.effective_spec.updates)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        from .metrics import source_metrics
        metrics = super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        metrics.update(source_metrics(reward_extra_infos_dict))
        return metrics

    def _load_checkpoint(self):
        from .phase_runtime import assert_training_ready
        assert_training_ready(self)
        cfg = self.config.mopd
        self.mopd_started = time.monotonic()
        self.mopd_loaded_step = 0
        sampler = self.train_dataloader.sampler
        if cfg.test_checkpoint is not None:
            train_root = Path(cfg.manifest_path).parent / "runs" / cfg.arm / "checkpoints"
            selection = OmegaConf.to_container(cfg.test_checkpoint, resolve=True)
            selected = checkpoints.restore_selected(selection, train_root, cfg.identity, cfg.arm, sampler.identity)
        else:
            if cfg.get("execution_identity") and not cfg.qualification:
                evidence = inspect_resume(cfg.run_dir, execution_manifest(self.config), cfg.arm, sampler.identity)
                write(Path(cfg.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "resume_lineage.json", evidence)
            selected = checkpoints.latest(self.config.trainer.default_local_dir, cfg.identity, cfg.arm, sampler.identity)
        if cfg.get("full_qualification", False):
            step = read(selected / "mopd_complete.json")["step"] if selected else 0
            need(step in (0, 4, 8), "Unexpected full qualification resume boundary")
            route = execution_manifest(self.config)["execution"]["qualification_transition"]
            need(cfg.execution_profile == route[int(step >= 4)], "Wrong profile for qualification segment")
        if selected is None:
            self.config.trainer.resume_mode = "disable"
        else:
            self.config.trainer.resume_mode = "resume_path"
            self.config.trainer.resume_from_path = str(selected)
        with profile_stage(self, "checkpoint_load"):
            super()._load_checkpoint()
        if selected:
            driver = torch.load(selected / "driver.pt", weights_only=False)
            random.setstate(driver["random"])
            np.random.set_state(driver["numpy"])
            torch.set_rng_state(driver["torch"])
            self.mopd_loaded_step = self.global_steps
        # A scheduler interruption must not introduce extra selection checkpoints.
        validation_interval = self.config.trainer.test_freq
        if cfg.test_checkpoint is None and validation_interval > 0 and self.global_steps % validation_interval:
            self.config.trainer.val_before_train = False
        # A termination after the final save must not execute update 9/201 on retry.
        if cfg.test_checkpoint is None and self.global_steps == self.config.trainer.total_training_steps:
            self.config.trainer.val_only = True
            self.config.trainer.val_before_train = True
        write(Path(cfg.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "resume.json",
              dict(step=self.global_steps, checkpoint=str(selected) if selected else None,
                   model_optimizer_scheduler_rng_and_loader=bool(selected),
                   vllm_sampling_rng_bitwise_restored=False, identity=cfg.identity,
                   **execution_result(self.config)))
        if any((Path(cfg.run_dir) / "validation").glob("step_*.json")):
            self._prune_checkpoints()

    def _save_checkpoint(self):
        from .phase_runtime import assert_training_ready
        assert_training_ready(self)
        with profile_stage(self, "checkpoint_save"):
            super()._save_checkpoint()
        c = self.config.mopd
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        torch.save(dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), folder / "driver.pt")
        if c.get("full_qualification", False):
            need(self.global_steps in (4, 8), "Qualification saved outside the declared boundary")
            write(folder / "qualification_segment.json", dict(
                profile=c.execution_profile, attempt_id=os.environ["D4_MOPD_ATTEMPT"],
                loaded_step=self.mopd_loaded_step, completed_step=self.global_steps,
                execution_identity=c.execution_identity, runtime_identity=c.runtime_identity))
        checkpoints.commit(folder, c.identity, c.arm, self.global_steps, self.train_dataloader.sampler.identity,
                           resumed_from=self.mopd_loaded_step)
        self._prune_checkpoints()
        if c.qualification and self.mopd_loaded_step == 0 and self.global_steps == 4:
            raise QualificationPause("Step 4 committed; restart another PBS process to check native resume")

    def _update_actor(self, batch):
        from .phase_runtime import assert_training_ready
        assert_training_ready(self)
        c = self.config.mopd
        run = Path(c.run_dir)
        if (Path(c.get("control_root", str(Path(c.manifest_path).parent))) / "STOP").exists():
            raise ControlPause("Experiment STOP requested")
        spec = OmegaConf.to_container(c.effective_spec, resolve=True)
        b = spec["train_batch_size"]
        planned = self.train_dataloader.sampler.indices[(self.global_steps-1)*b:self.global_steps*b]
        expected = [self.train_dataset.rows[i]["sample_id"] for i in planned]
        record = check_batch(batch, spec, c.arm, self.global_steps, expected)
        if c.get("execution_identity"):
            p = batch.batch["prompts"].shape[1]
            attention = batch.batch["attention_mask"]
            for domain, stats in record["domains"].items():
                rows = torch.tensor([v == domain for v in batch.non_tensor_batch["data_source"]],
                                    device=attention.device)
                stats["max_response_length"] = int(attention[rows, p:].sum(-1).max())
                stats["max_total_tokens"] = int(attention[rows].sum(-1).max())
        started = time.monotonic()
        with profile_stage(self, "student_update"):
            result = super()._update_actor(batch)
        metrics = plain(result.meta_info["metrics"])
        for name, value in metrics.items():
            if "grad_norm" in name or "distillation/loss" in name:
                need(bool(np.isfinite(np.asarray(value)).all()), "Nonfinite update metric: " + name)
        record.update(step=self.global_steps, at_utc=now(), identity=c.identity,
                      actor_update_seconds=time.monotonic()-started, metrics=metrics,
                      optimizer_updates_per_rollout_batch=1,
                      elapsed_attempt_seconds=time.monotonic()-self.mopd_started,
                      **execution_result(self.config))
        write(run / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / f"update_{self.global_steps}.json", record)
        return result

    def _validate(self):
        c = self.config.mopd
        run = Path(c.run_dir)
        is_test = c.test_checkpoint is not None
        split = "test" if is_test else "validation"
        canonical = run / split / f"step_{self.global_steps}.json"
        reference = None
        if not c.qualification:
            if is_test:
                reference = {k: c.test_checkpoint[k] for k in ("checkpoint", "checkpoint_identity")}
            else:
                reference = checkpoints.evaluation_reference(self.config.trainer.default_local_dir,
                    c.identity, c.arm, self.global_steps, self.train_dataloader.sampler.identity)
            if canonical.exists():
                old = read(canonical)
                need(old["identity"] == c.identity and old["arm"] == c.arm and old["split"] == split
                     and old["step"] == self.global_steps and old["status"] == "PASS"
                     and all(old[k] == v for k, v in reference.items()), "Cached evaluation identity mismatch")
                if not is_test:
                    select_best(run, c.identity, c.arm)
                    self._prune_checkpoints()
                return old["metrics"]
        metrics = plain(super()._validate())
        row = dict(status="PASS", step=self.global_steps, identity=c.identity, arm=c.arm, metrics=metrics,
                   split=split, test_used_for_selection=False, at_utc=now())
        if reference is not None:
            need(SELECTION_KEY in metrics, "Full four-domain/two-source evaluation is incomplete")
            row.update(reference)
            write(canonical, row)
            if not is_test:
                select_best(run, c.identity, c.arm)
                self._prune_checkpoints()
        write(Path(c.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / f"{split}_{self.global_steps}.json",
              row)
        return metrics


@ray.remote
class MOPDTaskRunner(BaseTaskRunner):
    def init_resource_pool_mgr(self, config):
        from .phase_runtime import is_phased_config, shared_resource_pool_manager
        if is_phased_config(config):
            return shared_resource_pool_manager(config, self.mapping)
        return super().init_resource_pool_mgr(config)

    def run(self, config):
        os.environ["VERL_FILE_LOGGER_PATH"] = str(Path(config.mopd.run_dir) / "attempts"
                                                 / os.environ["D4_MOPD_ATTEMPT"] / "metrics.jsonl")
        random.seed(config.mopd.effective_spec.seed)
        np.random.seed(config.mopd.effective_spec.seed)
        torch.manual_seed(config.mopd.effective_spec.seed)
        _, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_teacher_model_resource_pool(config)
        self.add_reward_model_resource_pool(config)
        validate_config(config, use_reference_policy=False, use_critic=False)
        model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        from verl.utils.dataset.rl_dataset import collate_fn
        train = create_rl_dataset(config.data.train_files, config.data, model.tokenizer, model.processor, is_train=True)
        val = create_rl_dataset(config.data.val_files, config.data, model.tokenizer, model.processor, is_train=False)
        if config.mopd.qualification and not config.mopd.get("full_qualification", False):
            # Four-domain routing + full native update/save/reload. Not a throughput
            # or worst-case-64K capacity qualification; use separate paths/receipts.
            train.rows = [r for d in DOMAINS for r in sorted(
                [r for r in train.rows if r["domain"] == d], key=lambda r: r["prompt_tokens"], reverse=True)[:2]]
            val.rows = [next(r for r in val.rows if r["domain"] == d) for d in DOMAINS]
        sampler = PlannedSampler(train.rows, OmegaConf.to_container(config.mopd.effective_spec, resolve=True), config.mopd.arm)
        trainer = MOPDTrainer(config=config, tokenizer=model.tokenizer, processor=model.processor,
                             role_worker_mapping=self.role_worker_mapping,
                             resource_pool_manager=self.init_resource_pool_mgr(config),
                             ray_worker_group_cls=ray_worker_group_cls, train_dataset=train, val_dataset=val,
                             collate_fn=collate_fn, train_sampler=sampler)
        trainer.init_workers()
        try:
            trainer.fit()
        except QualificationPause:
            result = {"status": "RESUME_REQUIRED", "step": trainer.global_steps,
                      "identity": config.mopd.identity, "arm": config.mopd.arm,
                      **execution_result(config)}
            write(Path(config.mopd.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "result.json", result)
            return result
        except ControlPause:
            result = {"status": "CONTROL_PAUSED", "identity": config.mopd.identity,
                      "arm": config.mopd.arm,
                      "uncommitted_next_step": trainer.global_steps, **execution_result(config)}
            write(Path(config.mopd.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "result.json", result)
            return result
        if config.mopd.test_checkpoint is not None:
            selection = OmegaConf.to_container(config.mopd.test_checkpoint, resolve=True)
            receipt = Path(config.mopd.run_dir) / "test" / f"step_{selection['step']}.json"
            need(read(receipt)["status"] == "PASS", "Final test did not complete")
            result = dict(status="PASS", identity=config.mopd.identity, step=selection["step"],
                          arm=config.mopd.arm, selection=selection, test_receipt=str(receipt), split="test",
                          **execution_result(config))
            write(Path(config.mopd.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "result.json", result)
            return result
        final = checkpoints.latest(config.trainer.default_local_dir, config.mopd.identity, config.mopd.arm, sampler.identity)
        need(final is not None and read(final / "mopd_complete.json")["step"] == config.trainer.total_training_steps,
             "fit returned without the final complete checkpoint")
        resumed_from = trainer.mopd_loaded_step
        if config.mopd.qualification:
            # A kill after the final commit but before publishing the PASS result
            # must not erase the already completed fresh-process resume proof.
            resumed_from = read(final / "mopd_complete.json")["resumed_from"]
            need(resumed_from == 4, "Qualification did not reload in a fresh process")
        result = {"status": "PASS", "step": config.trainer.total_training_steps,
                  "arm": config.mopd.arm,
                  "resumed_from": resumed_from, "checkpoint": str(final),
                  "identity": config.mopd.identity, "qualification": config.mopd.qualification,
                  **execution_result(config), **qualification_segments(config)}
        if not config.mopd.qualification:
            need((Path(config.mopd.run_dir) / "validation" / f"step_{config.mopd.effective_spec.updates}.json").is_file(),
                 "Missing fixed-budget endpoint validation")
            result["selection"] = select_best(config.mopd.run_dir, config.mopd.identity, config.mopd.arm)
        write(Path(config.mopd.run_dir) / "attempts" / os.environ["D4_MOPD_ATTEMPT"] / "result.json", result)
        return result
