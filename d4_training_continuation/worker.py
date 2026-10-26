"""Actual per-rank backend, LR, RNG and same-layout restore evidence."""
import os
from pathlib import Path
import socket
import torch
from verl.single_controller.base.decorator import register, Dispatch
from multidomain.worker import MultiDomainWorker
from .common import need, read, write
from .pinned_audit import checkpoint_engine_state, compare_restored_state, require_audit_binding, state_digest
from .optimizer_repair import context, worker_check


class ContinuationActor(MultiDomainWorker):
    # Unique name is required by pinned verl's colocated worker-class factory.
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def continuation_topology(self):
        from megatron.core import parallel_state as mpu
        require_audit_binding()
        return dict(rank=torch.distributed.get_rank(), host=socket.gethostname().split('.')[0],
                    world_size=torch.distributed.get_world_size(), optimizer_repair=worker_check(self.actor.engine),
                    tp=mpu.get_tensor_model_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size(),
                    ep=mpu.get_expert_model_parallel_world_size(), cp=mpu.get_context_parallel_world_size(),
                    dp=mpu.get_data_parallel_world_size())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        _, phase, req, execution = context()
        need(not del_local_after_load, "Source retention required")
        engine = self.actor.engine
        manager = engine.checkpoint_mananager
        manager.use_checkpoint_opt_param_scheduler = True
        need(manager.use_dist_checkpointing and manager.should_load_dist_ckpt_model
             and manager.should_load_optimizer and manager.should_load_extra, "Incomplete restore flags")
        scheduler = manager.lr_scheduler
        original_lr, original_rng = scheduler.load_state_dict, manager.load_rng_states
        observed = {}
        def load_lr(state):
            observed["lr"] = state_digest(state)
            return original_lr(state)
        def load_rng(*args, **kwargs):
            result = original_rng(*args, **kwargs)
            observed["rng"] = True
            return result
        scheduler.load_state_dict, manager.load_rng_states = load_lr, load_rng
        try:
            result = super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        finally:
            scheduler.load_state_dict, manager.load_rng_states = original_lr, original_rng
        actual = checkpoint_engine_state(engine)
        need(observed.get("lr") == actual["lr_scheduler"] and observed.get("rng"), "LR/RNG restore failed")
        rank, world = torch.distributed.get_rank(), execution["target_world"]
        source = read(phase / "resume_source.json")
        same = source["world_size"] == world
        comparison = "backend_full_restore_and_full_LR_RNG"
        if same and source.get("lineage"):
            compare_restored_state(Path(local_path) / f"continuation_state_rank_{rank}.json", actual)
            comparison = "same_layout_sampled_model_optimizer_and_full_LR"
        elif same:
            # Only the already-running legacy single stream can enter this path
            # during monitor adoption. Qualification B/D always have new sidecars.
            need(req["role"] == "formal" and world == 8, "Missing same-layout audit evidence")
            comparison = "legacy_single_full_backend_restore_no_new_sidecar"
        write(phase / f"restore_rank_{rank}.json", dict(status="PASS", rank=rank,
              pid=os.getpid(), host=socket.gethostname().split('.')[0], token=execution["token"],
              source_world=source["world_size"], target_world=world, comparison=comparison,
              full_model_optimizer_load_returned=True, lr_state_equal=True, rng_restored=True, state=actual))
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        need(not self.actor.engine.checkpoint_mananager.checkpoint_config.async_save, "Synchronous full save required")
        result = super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        write(Path(local_path) / f"continuation_state_rank_{torch.distributed.get_rank()}.json",
              checkpoint_engine_state(self.actor.engine))
        return result
