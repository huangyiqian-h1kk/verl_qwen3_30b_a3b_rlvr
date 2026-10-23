"""Preserve the original GRPO/sampling schedule, add transactional save/handoff."""
import math
import os
from pathlib import Path
import shutil
import time
import uuid
from . import TOTAL_STEPS
from .common import digest, expired, need, now, read, sha, write
from . import checkpoints as ck
from .evidence import phase_pass, verify_qualification


class PhaseComplete(Exception):
    pass


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if hasattr(value, "tolist"):
        return finite(value.tolist())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (float, int)) or math.isfinite(float(value))


def trainer_class(base):
    class ContinuationTrainer(base):
        def configure_continuation(self, request, phase, resume, plan=None, rows=None):
            self.cont_req, self.cont_phase, self.cont_resume = request, Path(phase), resume
            self.cont_plan, self.cont_rows = plan, rows
            self.cont_pending, self.cont_saved, self.cont_validating = None, None, False
            self.cont_stop_next_batch = False
            need(self.total_training_steps == TOTAL_STEPS, "Training/LR step budget changed")
            need(self.config.data.dataloader_num_workers == 0, "Loader worker count changed")
            if plan is not None:
                need(len(self.train_dataloader) == TOTAL_STEPS and plan.metadata() == request["sampling"], "Domain plan changed")
            original_sync = self.checkpoint_manager.update_weights
            def sync(step):
                updated = self.cont_pending is not None and step > resume["step"]
                release_file = Path(request["job_dir"]) / "release.json"
                target = read(release_file)["target"] if release_file.exists() else None
                if updated:
                    need(self.cont_pending["step"] == step, "Actor update/sync step mismatch")
                    if step == resume["step"] + 1 or step == target:
                        self._save_checkpoint()  # before vLLM wake-up, never after
                result = original_sync(step)
                if not updated:
                    return result
                record = dict(self.cont_pending, status="PASS", weights_synchronized=True,
                              saved_checkpoint_step=self.cont_saved, finished_utc=now())
                write(self.cont_phase / f"step_{step}.json", record)
                self.cont_pending = None
                if step == resume["step"] + 1:
                    checkpoint = ck.latest(request["run_dir"], request["lineage"])
                    phase_pass(request, self.cont_phase, resume, checkpoint, step)
                    if self.cont_phase.name in ("A", "B", "C", "D"):
                        # Finish upstream validation/metrics/logging for this step.
                        # Exit before generation of the NEXT training batch.
                        self.cont_stop_next_batch = True
                    elif self.cont_phase.name == "E":
                        self.publish_ready(checkpoint)
                        self.wait_for(release_file)
                if release_file.exists() and request["role"] == "candidate":
                    self.catch_up(step, read(release_file))
                return result
            self.checkpoint_manager.update_weights = sync

        def _expected_domain(self):
            indices = self.cont_plan.batch(self.global_steps)
            return self.cont_plan.domains[self.global_steps - 1], [self.cont_rows[i]["sample_id"] for i in indices]

        def _validate(self, merged=False):
            previous, self.cont_validating = self.cont_validating, True
            try:
                if self.cont_req.get("training_policy"):
                    from . import evaluation
                    return evaluation.validate(self, super()._validate, merged=merged)
                return super()._validate(merged=merged)
            finally:
                self.cont_validating = previous

        def _get_gen_batch(self, batch):
            if not self.cont_validating:
                if self.cont_stop_next_batch:
                    raise PhaseComplete()
                if self.cont_plan is not None:
                    from d4_domain_batch.trainer import batch_identity
                    batch_identity(batch, *self._expected_domain(), repeats=1)
            return super()._get_gen_batch(batch)

        def _update_actor(self, batch):
            need(self.cont_req["role"] != "evaluation", "Actor updates are forbidden in evaluation-only jobs")
            from d4_two_node_resumefix.trainer import summarize_batch
            summary = summarize_batch(batch)
            if self.cont_plan is not None:
                from d4_domain_batch.trainer import batch_identity
                domain, ids = self._expected_domain()
                batch_identity(batch, domain, ids, repeats=8)
                summary.update(domain=domain, expected_ids_sha256=digest(ids),
                               actual_ids_sha256=digest(sorted(summary["prompt_ids"])),
                               sampling_sha256=self.cont_plan.metadata()["plan_sha256"])
            result = super()._update_actor(batch)
            metrics = result.meta_info["metrics"]
            need(finite(metrics), "Non-finite actor metrics")
            if self.cont_plan is not None:
                for candidate in ("math", "if", "conversational_pivot", "logic_algorithmic"):
                    metrics["sampling/domain/" + candidate] = float(domain == candidate)
                metrics["sampling/budget_epoch"] = (self.global_steps - 1) // 156 + 1
            # JSON-safe metrics, including entropy/loss; raw upstream logging remains.
            plain = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
            self.cont_pending = dict(summary, step=self.global_steps, actor_update_completed=True,
                                     optimizer_minibatches_per_grpo_step=8, actor_metrics=plain)
            return result

        def _load_checkpoint(self):
            from .pinned_audit import state_digest
            import torch
            source = ck.validate(self.cont_resume)
            if self.cont_plan is not None:
                need(read(Path(source["path"]) / "sampling.json") == self.cont_plan.metadata(), "Source sampler changed")
            previous = self.config.trainer.default_local_dir
            self.config.trainer.default_local_dir = str(Path(source["path"]).parent)
            try:
                super()._load_checkpoint()
            finally:
                self.config.trainer.default_local_dir = previous
            need(self.global_steps == source["step"], "Unexpected restored step")
            boundary = self.global_steps % len(self.train_dataloader) == 0
            saved = torch.load(Path(source["path"]) / "data.pt", weights_only=False)
            if not boundary:
                need(state_digest(saved) == state_digest(self.train_dataloader.state_dict()), "Dataloader state differs")
            need(not boundary or self.cont_plan is None or self.config.trainer.val_only,
                 "Unexpected domain epoch boundary")
            write(self.cont_phase / "resume.json", dict(status="PASS", step=self.global_steps,
                  dataloader_restored=not boundary, epoch_boundary=boundary,
                  epoch_boundary_policy="pinned_verl_restart_exhausted_iterator" if boundary else None,
                  sampling_sha256=self.cont_plan.metadata()["plan_sha256"] if self.cont_plan is not None else None))

        def _save_checkpoint(self):
            need(self.cont_req["role"] != "evaluation", "Checkpoint writes are forbidden in evaluation-only jobs")
            step = self.global_steps
            if self.cont_saved == step:
                return
            req = self.cont_req
            checkpoints = Path(self.config.trainer.default_local_dir)
            staging = checkpoints / (".staging_" + uuid.uuid4().hex)
            staging.mkdir(parents=True)
            keep = self.config.trainer.max_actor_ckpt_to_keep
            try:
                self.config.trainer.default_local_dir = str(staging)
                self.config.trainer.max_actor_ckpt_to_keep = None
                base._save_checkpoint(self)
            finally:
                self.config.trainer.default_local_dir = str(checkpoints)
                self.config.trainer.max_actor_ckpt_to_keep = keep
            path = staging / f"global_step_{step}"
            if self.cont_plan is not None:
                write(path / "sampling.json", self.cont_plan.metadata())
            execution = read(self.cont_phase / "phase.json")
            ck.commit(path, req["run_dir"], step, execution["target_world"], req["lineage"], req["kind"],
                      self.cont_resume, self.cont_phase.name)
            self.cont_saved = step
            ck.retain(req["run_dir"], req["lineage"], keep=2)
            shutil.rmtree(staging)

        def wait_for(self, path):
            while not Path(path).exists():
                state = read(Path(self.cont_req["control_dir"]) / "state.json")
                if self.cont_req.get("opportunity_id"):
                    from .opportunities import authorized
                    need(authorized(state, self.cont_req, allow_committed=True), "Opportunity is no longer authorized")
                committed = (self.cont_req.get("opportunity_id")
                             and state.get("opportunity", {}).get("id") == self.cont_req["opportunity_id"]
                             and state["opportunity"]["phase"] == "COMMITTED")
                need(committed or (not state.get("handoff_failed") and not expired(state["deadline_utc"])),
                     "Handoff expired or failed; candidate will not continue without release")
                time.sleep(2)
            result = read(path)
            allocation = read(Path(self.cont_req["job_dir"]) / "allocation.json")
            need(result["identity"] == self.cont_req["identity"] and result["execution_id"] == allocation["execution_id"],
                 "Handoff acknowledgement belongs to another execution")
            return result

        def publish_ready(self, checkpoint):
            req, folder = self.cont_req, Path(self.cont_req["job_dir"])
            allocation = read(folder / "allocation.json")
            record = dict(identity=req["identity"], job_id=os.environ["PBS_JOBID"], checkpoint=checkpoint,
                          execution_id=allocation["execution_id"], evidence={})
            verify_qualification(folder, record)
            for name in ("A", "B", "C", "D", "E"):
                record["evidence"][name + "/qualification.json"] = sha(folder / name / "qualification.json")
            write(folder / "ready.json", record)

        def catch_up(self, step, decision):
            req, folder = self.cont_req, Path(self.cont_req["job_dir"])
            if (folder / "committed.json").exists():
                self.wait_for(folder / "committed.json")
                return
            if step < decision["target"]:
                return
            need(step == decision["target"] and self.cont_saved == step, "Catch-up checkpoint was not saved before sync")
            cp = ck.latest(req["run_dir"], req["lineage"])
            proof = dict(identity=req["identity"], job_id=os.environ["PBS_JOBID"], checkpoint=cp,
                         execution_id=decision["execution_id"],
                         evidence={"E/step_" + str(step) + ".json": sha(self.cont_phase / f"step_{step}.json")})
            write(folder / "matched.json", proof)
            self.wait_for(folder / "committed.json")
    return ContinuationTrainer
