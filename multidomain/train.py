"""Run the fixed engine through an explicit TaskRunner; no edits to the shared verl checkout."""
from pathlib import Path
import json
from multidomain.common import write_json


def run(config, stage, run_dir, data_dir):
    import os
    os.environ['MD_RUN_DIR'] = str(run_dir)
    os.environ['MD_STAGE'] = stage
    from omegaconf import OmegaConf
    OmegaConf.update(config, 'ray_kwargs.ray_init.runtime_env.env_vars.MD_RUN_DIR', str(run_dir), force_add=True)
    OmegaConf.update(config, 'ray_kwargs.ray_init.runtime_env.env_vars.MD_STAGE', stage, force_add=True)
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    from verl.trainer.ppo.utils import create_rl_sampler
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from multidomain.dataset import MultiDomainDataset
    from multidomain.worker import MultiDomainWorker

    @ray.remote
    class Runner(BaseTaskRunner):
        def run(self, config):
            OmegaConf.resolve(config)
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(MultiDomainWorker)
            self.mapping[Role.ActorRollout] = 'global_pool'
            self.add_reward_model_resource_pool(config)
            tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
            train_ds = MultiDomainDataset(config.data.train_files, tokenizer, config.data)
            val_ds = MultiDomainDataset(config.data.val_files, tokenizer, config.data)
            trainer = RayPPOTrainer(config=config, tokenizer=tokenizer, processor=None,
                role_worker_mapping=self.role_worker_mapping, resource_pool_manager=self.init_resource_pool_mgr(config),
                ray_worker_group_cls=RayWorkerGroup, train_dataset=train_ds, val_dataset=val_ds,
                collate_fn=collate_fn, train_sampler=create_rl_sampler(config.data, train_ds))
            trainer.init_workers()
            trainer.actor_rollout_wg.configure_checkpoints(trainer.total_training_steps)
            if stage == 'backward':
                from multidomain.template import render
                longest = max(train_ds.rows, key=lambda r: r['prompt_tokens'])
                ids = render(tokenizer, json.loads(longest['messages_json']), json.loads(longest['tools_json']))
                filler = tokenizer.encode('\nThe answer follows from the stated conditions.', add_special_tokens=False)
                cases = {'longest_prompt_plus_one': ids + [tokenizer.eos_token_id],
                         'synthetic_64k_capacity': (ids + filler * (65536 // len(filler) + 1))[:65536]}
                reports = {}
                for name, tokens in cases.items():
                    reports[name] = trainer.actor_rollout_wg.backward_capacity(tokens)
                write_json(Path(run_dir) / 'reports/backward_detail.json',
                           {'cases': reports, 'optimizer_steps': 0, 'scope': 'forward_backward_NLL; optimizer-step peak untested'})
            else:
                trainer.fit()
    run_ppo(config, Runner)
