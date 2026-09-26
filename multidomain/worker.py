"""Small pinned-verl extension: full distributed checkpoints and a no-step capacity probe."""
from functools import partial
import torch
from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.engine_workers import ActorRolloutRefWorker


class MultiDomainWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_checkpoints(self, final_step):
        self.md_final_step = final_step
        # Load initial HF weights via mbridge first, then save/load distributed resume checkpoints.
        self.actor.engine.checkpoint_mananager.use_dist_checkpointing = True
        engine = self.actor.engine
        first_batch = True
        first_step = True
        original_batch, original_step = engine.train_batch, engine.optimizer_step
        def measured_batch(*args, **kwargs):
            nonlocal first_batch
            if first_batch:
                torch.cuda.reset_peak_memory_stats()
                first_batch = False
            return original_batch(*args, **kwargs)
        def measured_step(*args, **kwargs):
            nonlocal first_step
            value = original_step(*args, **kwargs)
            if first_step:
                import os
                from pathlib import Path
                from multidomain.common import write_json
                torch.cuda.synchronize()
                report_dir = os.environ.get('MD_RUN_DIR')
                if report_dir:
                    write_json(Path(report_dir) / 'reports' / f'first_optimizer_step_rank_{torch.distributed.get_rank()}.json',
                        {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'optimizer_step_completed': True})
                first_step = False
            return value
        engine.train_batch = measured_batch
        engine.optimizer_step = measured_step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        manager = self.actor.engine.checkpoint_mananager
        final = global_step == self.md_final_step
        manager.checkpoint_save_contents = ['model', 'optimizer', 'extra'] + (['hf_model'] if final else [])
        return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def backward_capacity(self, token_ids):
        """Real actor engine forward/backward; optimizer.step is never called."""
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.losses import sft_loss
        engine = self.actor.engine
        ids = torch.tensor(token_ids, dtype=torch.long, device='cuda')
        mask = torch.ones_like(ids)
        mask[0] = 0
        def nested(t):
            return torch.nested.as_nested_tensor([t], layout=torch.jagged)
        td = TensorDict({'input_ids': nested(ids), 'position_ids': nested(torch.arange(len(ids), device='cuda')),
                         'loss_mask': nested(mask), 'prompts': nested(ids[:1]),
                         'responses': nested(ids[1:]), 'response_mask': nested(mask[1:])}, batch_size=[1])
        tu.assign_non_tensor(td, use_remove_padding=True, use_dynamic_bsz=True,
            max_token_len_per_gpu=65536, micro_batch_size_per_gpu=1, use_fused_kernels=True,
            global_batch_size=engine.get_data_parallel_size(), calculate_entropy=False)
        torch.cuda.reset_peak_memory_stats()
        with engine.train_mode():
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(td, partial(sft_loss, config=self.config.actor), forward_only=False)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            engine.optimizer_zero_grad()
        return {'rank': torch.distributed.get_rank(), 'tokens': len(token_ids),
                'peak_allocated_bytes': peak, 'optimizer_steps': 0, 'loss': 'token NLL capacity probe'}
