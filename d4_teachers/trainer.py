"""A 200-step teacher stream with audited full saves and validation-only selection."""

from collections import defaultdict
from pathlib import Path
import shutil
import time
import uuid
from .common import digest, is_fenced, need, now, read, write
from .sampling import verify_batch
from . import evaluation
from d4_training_continuation import checkpoints as ck
from d4_training_continuation.validation_journal import plain


class TeacherComplete(Exception):
    pass


def summarize(batch):
    import torch

    extras = batch.non_tensor_batch["extra_info"]
    scores = batch.batch["token_level_scores"].sum(-1).detach().float().cpu()
    lengths = batch.batch["response_mask"].sum(-1).detach().cpu()
    need(
        len(batch) == 2048
        and bool(scores.isfinite().all())
        and bool(((scores >= 0) & (scores <= 1)).all()),
        "Invalid GRPO response count/rewards",
    )
    groups, strata = defaultdict(list), defaultdict(list)
    budgets = []
    for extra, score, length in zip(extras, scores.tolist(), lengths.tolist()):
        budget = min(
            65536 - int(extra["prompt_tokens"]), int(extra["generation_max_tokens"])
        )
        need(0 < length <= budget, "Generated response exceeded frozen token budget")
        budgets.append(budget)
        groups[str(extra["sample_id"])].append(score)
        for k in ("source", "category", "task"):
            strata[k + "/" + str(extra.get(k, ""))].append(score)
    need(
        len(groups) == 256 and all(len(v) == 8 for v in groups.values()),
        "GRPO prompt grouping changed",
    )
    return dict(
        prompts=256,
        responses=2048,
        reward_mean=float(scores.mean()),
        reward_min=float(scores.min()),
        reward_max=float(scores.max()),
        zero_advantage_group_fraction=sum(max(v) == min(v) for v in groups.values())
        / len(groups),
        response_length_mean=float(lengths.float().mean()),
        response_length_max=int(lengths.max()),
        response_at_budget_fraction=float(
            (lengths == torch.tensor(budgets)).float().mean()
        ),
        reward_by_stratum={
            k: dict(n=len(v), mean=sum(v) / len(v)) for k, v in strata.items()
        },
    )


def trainer_class(base):
    class TeacherTrainer(base):
        def configure_teacher(self, request, phase, resume, plan):
            self.teacher_request, self.teacher_phase = request, Path(phase)
            self.teacher_resume, self.teacher_plan = resume, plan
            self.teacher_pending = None
            self.teacher_saved = None
            self.teacher_validating = self.teacher_stop = False
            self.teacher_step_started = None
            need(
                self.total_training_steps == request["policy"]["max_steps"],
                "Teacher step budget changed",
            )
            need(
                len(self.train_dataloader) == self.total_training_steps,
                "Teacher stream was capped by epoch length",
            )
            original = self.checkpoint_manager.update_weights

            def synchronized(step):
                result = original(step)
                if self.teacher_pending is not None:
                    need(
                        self.teacher_pending["step"] == step,
                        "Update/synchronization step mismatch",
                    )
                    record = dict(
                        self.teacher_pending,
                        weights_synchronized=True,
                        status="PASS",
                        at_utc=now(),
                        train_and_save_seconds=time.monotonic()
                        - self.teacher_step_started,
                    )
                    write(self.teacher_phase / f"step_{step}.json", record)
                    write(
                        Path(request["run_dir"]) / "progress.json",
                        dict(
                            step=step,
                            latest_full_step=self.teacher_saved or resume["step"],
                            at_utc=now(),
                        ),
                    )
                    self.teacher_pending = None
                return result

            self.checkpoint_manager.update_weights = synchronized

        def weights_identity(self, checkpoint):
            if self.teacher_request["role"] == "test":
                return self.teacher_request["evaluation_model"]["weights_identity"]
            if not checkpoint["step"]:
                return self.teacher_request["base_identity"]
            return digest({k: checkpoint[k] for k in ("lineage", "step", "files")})

        def current_checkpoint(self, step):
            if self.teacher_request["role"] == "test" or step == 0:
                return dict(
                    step=step,
                    files={},
                    world_size=8,
                    path=self.teacher_request["base_model"],
                )
            checkpoint = ck.latest(
                self.teacher_request["run_dir"], self.teacher_request["lineage"]
            )
            need(
                checkpoint and checkpoint["step"] == step,
                "Validation requires a committed full checkpoint",
            )
            return checkpoint

        def _get_gen_batch(self, batch):
            if not self.teacher_validating:
                need(
                    self.teacher_request["role"] == "train",
                    "Updates forbidden outside training",
                )
                need(
                    not is_fenced(
                        self.teacher_request["campaign_dir"],
                        self.teacher_request["domain"],
                    ),
                    "Teacher stopped",
                )
                if self.teacher_stop:
                    raise TeacherComplete()
                ids = [
                    self.train_dataset.original.rows[i]["sample_id"]
                    for i in self.teacher_plan.batch(self.global_steps)
                ]
                verify_batch(batch, self.teacher_request["domain"], ids, 1)
                self.teacher_step_started = time.monotonic()
            return super()._get_gen_batch(batch)

        def _update_actor(self, batch):
            need(
                self.teacher_request["role"] == "train",
                "Updates forbidden outside training",
            )
            ids = [
                self.train_dataset.original.rows[i]["sample_id"]
                for i in self.teacher_plan.batch(self.global_steps)
            ]
            verify_batch(batch, self.teacher_request["domain"], ids, 8)
            summary = summarize(batch)
            result = super()._update_actor(batch)
            metrics = plain(result.meta_info["metrics"])
            self.teacher_pending = dict(
                summary,
                step=self.global_steps,
                domain=self.teacher_request["domain"],
                actor_update_completed=True,
                optimizer_minibatches_per_grpo_step=8,
                actor_metrics=metrics,
                prompt_ids_sha256=digest(ids),
                sampling_sha256=self.teacher_plan.metadata()["plan_sha256"],
            )
            return result

        def _validate(self, merged=False):
            self.teacher_validating = True
            try:
                return evaluation.validate(self, super()._validate, merged)
            finally:
                self.teacher_validating = False

        def _load_checkpoint(self):
            source = self.teacher_resume
            if self.teacher_request["role"] == "test":
                # HF weights were loaded during worker construction. No optimizer restore.
                self.global_steps = source["step"]
                return
            if source["step"]:
                ck.validate(source)
                need(
                    read(Path(source["path"]) / "sampling.json")
                    == self.teacher_plan.metadata(),
                    "Sampler identity changed",
                )
                need(
                    read(Path(source["path"]) / "teacher.json")["campaign_identity"]
                    == self.teacher_request["campaign_identity"],
                    "Foreign teacher checkpoint",
                )
            super()._load_checkpoint()
            need(self.global_steps == source["step"], "Wrong restored training step")
            if source["step"]:
                import torch
                from d4_training_continuation.pinned_audit import state_digest

                boundary = self.global_steps == self.total_training_steps
                if not boundary:
                    saved = torch.load(
                        Path(source["path"]) / "data.pt", weights_only=False
                    )
                    need(
                        state_digest(saved)
                        == state_digest(self.train_dataloader.state_dict()),
                        "Dataloader cursor restore failed",
                    )
                need(
                    not boundary or self.config.trainer.val_only,
                    "Training beyond the final cursor",
                )
            write(
                self.teacher_phase / "resume.json",
                dict(
                    status="PASS",
                    step=source["step"],
                    sampling=self.teacher_plan.metadata(),
                    full_restore=bool(source["step"]),
                ),
            )

        def _save_checkpoint(self):
            req, step = self.teacher_request, self.global_steps
            need(req["role"] == "train", "Checkpoint writes forbidden outside training")
            if self.teacher_saved == step:
                return
            root = Path(self.config.trainer.default_local_dir)
            staging = root / (".staging_" + uuid.uuid4().hex)
            staging.mkdir(parents=True)
            try:
                self.config.trainer.default_local_dir = str(staging)
                super()._save_checkpoint()
            finally:
                self.config.trainer.default_local_dir = str(root)
            path = staging / f"global_step_{step}"
            write(path / "sampling.json", self.teacher_plan.metadata())
            write(
                path / "teacher.json",
                dict(
                    domain=req["domain"],
                    lineage=req["lineage"],
                    campaign_identity=req["campaign_identity"],
                ),
            )
            parent = ck.latest(req["run_dir"], req["lineage"]) or self.teacher_resume
            ck.commit(
                path,
                req["run_dir"],
                step,
                8,
                req["lineage"],
                "teacher",
                parent,
                "train",
            )
            self.teacher_saved = step
            ck.retain(req["run_dir"], req["lineage"], keep=2)
            shutil.rmtree(staging)

    return TeacherTrainer
