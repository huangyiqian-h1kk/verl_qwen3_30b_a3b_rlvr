"""Real Megatron restore evidence, produced inside each allocated actor rank."""
import os
from pathlib import Path
import torch
from verl.single_controller.base.decorator import register, Dispatch
from d4_two_node_resumefix.worker import TwoNodeWorker
from d4_two_node_resumefix.state_audit import state_digest
from d4_two_node_resumefix.audit_offload import checkpoint_engine_state
from .common import need, read, write


class Worker(TwoNodeWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def two_node_topology(self):
        import socket
        from megatron.core import parallel_state as mpu
        from d4_two_node_resumefix.state_audit import AUDIT_INDEX_VERSION
        from .optimizer_repair import worker_check
        return dict(optimizer_repair_version=worker_check(self.actor.engine),
            audit_index_version=AUDIT_INDEX_VERSION, rank=torch.distributed.get_rank(),
            host=socket.gethostname().split(".")[0], world_size=torch.distributed.get_world_size(),
            tp=mpu.get_tensor_model_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size(),
            ep=mpu.get_expert_model_parallel_world_size(), cp=mpu.get_context_parallel_world_size(),
            dp=mpu.get_data_parallel_world_size())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        need(not del_local_after_load, "Do not delete the source during restore")
        engine = self.actor.engine
        manager = engine.checkpoint_mananager
        need(manager.use_dist_checkpointing and manager.should_load_dist_ckpt_model
             and manager.should_load_optimizer and manager.should_load_extra,
             "Require model, optimizer, LR and RNG distributed restore")
        # In this pinned verl version LR is loaded only when this flag is true.
        manager.use_checkpoint_opt_param_scheduler = True
        scheduler = manager.lr_scheduler
        original_lr, original_rng = scheduler.load_state_dict, manager.load_rng_states
        observed = {}
        def load_lr(state):
            observed["expected_lr"] = state_digest(state)
            return original_lr(state)
        def load_rng(*args, **kwargs):
            result = original_rng(*args, **kwargs)
            observed["rng_loaded"] = True
            return result
        scheduler.load_state_dict, manager.load_rng_states = load_lr, load_rng
        try:
            result = super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
        finally:
            scheduler.load_state_dict, manager.load_rng_states = original_lr, original_rng
        actual = checkpoint_engine_state(engine)
        need(observed.get("expected_lr") == actual["lr_scheduler"], "LR scheduler did not restore the saved full state")
        need(observed.get("rng_loaded"), "RNG restore was not called")
        rank = torch.distributed.get_rank()
        saved = Path(local_path) / f"db_dual_state_rank_{rank}.json"
        source_world = read(Path(local_path) / "ckpt_contents.json")["world_size"]
        if source_world == 16:
            need(saved.exists() and read(saved) == actual, "Same-topology sampled model/optimizer or full LR state mismatch")
        else:
            need(source_world == 8, "Unexpected source world size")
        report = dict(status="PASS", rank=rank, source_world_size=source_world, target_world_size=16,
                      pid=os.getpid(), host=__import__('socket').gethostname().split('.')[0],
                      full_model_optimizer_load_returned=True, lr_state_equal=True, rng_restored=True,
                      state=actual,
                      comparison=("strict backend restore + initialized optimizer + full LR equality; "
                                  "8-to-16 elementwise optimizer equality is not separately proven"))
        write(Path(os.environ["DB_SWITCH_PHASE_DIR"]) / f"restore_rank_{rank}.json", report)
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        need(not self.actor.engine.checkpoint_mananager.checkpoint_config.async_save,
             "This handoff requires synchronous checkpoint publication")
        result = super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        write(Path(local_path) / f"db_dual_state_rank_{torch.distributed.get_rank()}.json", checkpoint_engine_state(self.actor.engine))
        return result
