"""Reuse audited full training restores; model-only restore is export-only."""

import os
from pathlib import Path
import torch
from verl.single_controller.base.decorator import register, Dispatch
from d4_training_continuation.worker import ContinuationActor
from d4_training_continuation.optimizer_repair import context
from .common import need, read, write
from .selection import validate_best


class TeacherActor(ContinuationActor):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_selected_teacher(self, selected, destination):
        job, phase, req, execution = context()
        need(
            req["role"] == "export",
            "Model-only loading is allowed only in a separate export allocation",
        )
        need(
            selected == read(Path(req["run_dir"]) / "best.json"),
            "Export selection changed",
        )
        validate_best(selected)
        engine = self.actor.engine
        manager = engine.checkpoint_mananager
        need(
            manager.use_dist_checkpointing and not manager.checkpoint_config.async_save,
            "Export requires synchronous native weights",
        )
        # The repaired full-restore wrapper remains installed for training. This
        # separate RPC exits its process after export and never updates an actor.
        original_load = type(manager).load_checkpoint.__wrapped__
        manager.checkpoint_load_contents = ["model"]
        manager.checkpoint_save_contents = ["hf_model"]
        with engine.train_mode():
            original_load(
                manager,
                str(Path(selected["path"]) / "actor"),
                del_local_after_load=False,
            )
            manager.save_checkpoint(
                str(destination), global_step=selected["step"], max_ckpt_to_keep=None
            )
        write(
            phase / f"export_rank_{torch.distributed.get_rank()}.json",
            dict(
                status="PASS",
                pid=os.getpid(),
                selected_identity=selected["weights_identity"],
                step=selected["step"],
                optimizer_updates=0,
            ),
        )
        return dict(status="PASS", rank=torch.distributed.get_rank())
