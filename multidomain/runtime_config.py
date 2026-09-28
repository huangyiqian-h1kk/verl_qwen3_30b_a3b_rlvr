"""Compile the experiment YAML to pinned verl Hydra overrides."""
import json
from pathlib import Path
from multidomain.common import ROOT


def overrides(c, data_dir, run_dir, stage, model_path=None):
    t, j, dc = c['training'], c['judge'], c['data']
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    validate = stage != 'train'
    val_name = {'smoke': 'smoke', 'baseline-train': 'baseline_train', 'evaluate': 'test'}.get(stage, 'validation')
    sampled = stage in ('smoke', 'baseline-train')
    n = c['acceptance']['model_smoke_n'] if stage == 'smoke' else c['acceptance']['baseline_n'] if stage == 'baseline-train' else 1
    d = {
        'trainer.use_v1': False,
        'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False,
        'data.train_files': str(data_dir / 'train.parquet'), 'data.val_files': str(data_dir / f'{val_name}.parquet'),
        'data.train_batch_size': t['train_batch_size'], 'data.val_batch_size': 8,
        'data.max_prompt_length': dc['max_prompt_tokens'], 'data.max_response_length': dc['max_response_tokens'],
        'data.filter_overlong_prompts': False, 'data.truncation': 'error', 'data.seed': c['seed'],
        'data.custom_cls.path': str(ROOT / 'multidomain/dataset.py'), 'data.custom_cls.name': 'MultiDomainDataset',
        'data.apply_chat_template_kwargs.enable_thinking': False, 'data.dataloader_num_workers': 0,
        'actor_rollout_ref.model.path': model_path or t['model_path'],
        'actor_rollout_ref.model.use_fused_kernels': True, 'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.optim.lr': t['learning_rate'],
        'actor_rollout_ref.actor.ppo_mini_batch_size': t['ppo_mini_batch_size'],
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': t['ppo_micro_batch_size_per_gpu'],
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': t['ppo_max_token_len_per_gpu'],
        'actor_rollout_ref.actor.use_dynamic_bsz': True, 'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.entropy_coeff': t['entropy_coeff'], 'actor_rollout_ref.actor.ppo_epochs': t['ppo_epochs'],
        'actor_rollout_ref.actor.checkpoint.save_contents': ['model', 'optimizer', 'extra'],
        'actor_rollout_ref.actor.checkpoint.load_contents': ['model', 'optimizer', 'extra'],
        'actor_rollout_ref.actor.checkpoint.async_save': False,
        'actor_rollout_ref.rollout.name': 'vllm', 'actor_rollout_ref.rollout.tensor_model_parallel_size': t['rollout_tp'],
        'actor_rollout_ref.rollout.n': t['rollout_n'], 'actor_rollout_ref.rollout.temperature': t['rollout_temperature'],
        'actor_rollout_ref.rollout.top_p': t['rollout_top_p'],
        'actor_rollout_ref.rollout.gpu_memory_utilization': t['rollout_gpu_memory_utilization'],
        'actor_rollout_ref.rollout.enable_chunked_prefill': True,
        'actor_rollout_ref.rollout.max_num_seqs': t['rollout_max_num_seqs'],
        'actor_rollout_ref.rollout.max_num_batched_tokens': t['rollout_max_num_batched_tokens'],
        'actor_rollout_ref.rollout.max_model_len': dc['max_context_tokens'],
        'actor_rollout_ref.rollout.prompt_length': dc['max_prompt_tokens'],
        'actor_rollout_ref.rollout.response_length': dc['max_response_tokens'],
        'actor_rollout_ref.rollout.calculate_log_probs': True,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': True,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': t['log_prob_max_token_len_per_gpu'],
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': t['log_prob_micro_batch_size_per_gpu'],
        'actor_rollout_ref.rollout.val_kwargs.n': n,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': sampled,
        'actor_rollout_ref.rollout.val_kwargs.temperature': t['rollout_temperature'] if sampled else 0.0,
        'actor_rollout_ref.rollout.val_kwargs.top_p': 1.0,
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(ROOT / 'config/multidomain/agent.yaml'),
        'actor_rollout_ref.rollout.agent.num_workers': 8,
        'reward.num_workers': 1, 'reward.reward_manager.name': 'naive',
        'reward.custom_reward_function.path': 'pkg://multidomain.reward',
        'reward.custom_reward_function.name': 'compute_score',
        'reward.reward_model.enable': c['resolved']['judge_enabled'],
        'reward.reward_model.enable_resource_pool': c['resolved']['judge_enabled'],
        'reward.reward_model.nnodes': j['nodes'] if c['resolved']['judge_enabled'] else 0,
        'reward.reward_model.n_gpus_per_node': j['gpus_per_node'],
        'reward.reward_model.model_path': j['model_path'],
        'reward.reward_model.rollout.name': 'vllm',
        'reward.reward_model.rollout.tensor_model_parallel_size': j['tensor_parallel_size'],
        'reward.reward_model.rollout.expert_parallel_size': j.get('expert_parallel_size', 1),
        'reward.reward_model.rollout.data_parallel_size': j['data_parallel_size'],
        'reward.reward_model.rollout.pipeline_model_parallel_size': j['pipeline_parallel_size'],
        'reward.reward_model.rollout.dtype': j['activation_dtype'],
        'reward.reward_model.rollout.quantization': 'fp8',
        'reward.reward_model.rollout.free_cache_engine': False,
        'reward.reward_model.rollout.gpu_memory_utilization': j['gpu_memory_utilization'],
        'reward.reward_model.rollout.max_model_len': j['max_model_len'],
        'reward.reward_model.rollout.prompt_length': j['max_model_len'] - j['max_output_tokens'],
        'reward.reward_model.rollout.response_length': j['max_output_tokens'],
        'reward.reward_model.rollout.max_num_seqs': j['max_num_seqs'],
        'reward.reward_model.rollout.max_num_batched_tokens': j['max_num_batched_tokens'],
        'reward.reward_model.rollout.enable_chunked_prefill': True,
        'trainer.nnodes': t['train_nodes'], 'trainer.n_gpus_per_node': t['gpus_per_node'],
        'trainer.total_epochs': t['total_epochs'], 'trainer.total_training_steps': t['total_training_steps'],
        'trainer.save_freq': t['save_freq'], 'trainer.test_freq': t['test_freq'],
        'trainer.max_actor_ckpt_to_keep': t['max_actor_checkpoints'],
        'trainer.default_local_dir': str(run_dir / 'checkpoints'),
        'trainer.validation_data_dir': str(run_dir / 'generations' / stage),
        'trainer.project_name': 'data_mixture_rl', 'trainer.experiment_name': run_dir.name,
        'trainer.logger': ['console', 'wandb'], 'trainer.log_val_generations': 0,
        'trainer.val_before_train': validate, 'trainer.val_only': validate,
        'trainer.resume_mode': 'auto' if stage == 'train' else 'disable',
    }
    for role in ('actor', 'ref'):
        pre = f'actor_rollout_ref.{role}.megatron.'
        d.update({pre + 'tensor_model_parallel_size': t['actor_tp'], pre + 'pipeline_model_parallel_size': t['actor_pp'],
                  pre + 'expert_model_parallel_size': t['actor_ep'], pre + 'context_parallel_size': t['actor_cp'],
                  pre + 'virtual_pipeline_model_parallel_size': None, pre + 'param_offload': True,
                  pre + 'use_mbridge': True})
    pre = 'actor_rollout_ref.actor.megatron.'
    d.update({pre + 'optimizer_offload': True, pre + 'grad_offload': True})
    for key, value in {'recompute_method': 'uniform', 'recompute_granularity': 'full', 'recompute_num_layers': 1,
                       'gradient_accumulation_fusion': False, 'moe_grouped_gemm': True,
                       'moe_permute_fusion': True}.items():
        d[pre + 'override_transformer_config.' + key] = value
    return ['model_engine=megatron'] + ['++' + k + '=' + json.dumps(v, ensure_ascii=False, separators=(',', ':')) for k, v in d.items()]


def compose(c, data_dir, run_dir, stage, model_path=None):
    from hydra import compose as hydra_compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(c['verl_src']) / 'verl/trainer/config'), version_base=None):
        return hydra_compose(config_name='ppo_trainer', overrides=overrides(c, data_dir, run_dir, stage, model_path))
