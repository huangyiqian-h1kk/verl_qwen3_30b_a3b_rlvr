"""Compose upstream Hydra config rather than copying verl's trainer/loss."""
from pathlib import Path
from . import DOMAINS


def compose(manifest, arm, run_dir, *, qualification=False):
    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import OmegaConf
    s = dict(manifest["spec"])
    if qualification:
        s.update(train_batch_size=8, updates=8, save_every=4, validate_every=4)
    with initialize_config_dir(version_base=None, config_dir=str(Path(manifest["verl_src"]).resolve() / "verl/trainer/config")):
        c = hydra_compose(config_name="ppo_trainer", overrides=["model_engine=megatron"])
    root = Path(run_dir).resolve()
    # Upper bounds for separately padded prompt and response tensors. Actual per-row
    # prompt+response remains <=64K; teacher prompt-logprob API needs one extra token.
    prompt = manifest["max_prompt_tokens"]
    response = 256 if qualification else s["max_context_tokens"] - 1
    vals = {
        "trainer.use_v1": False,
        "trainer.project_name": "d4_plain_mopd",
        "trainer.experiment_name": arm,
        "trainer.logger": ["console", "file"],
        "trainer.nnodes": s["student_nodes"],
        "trainer.n_gpus_per_node": 8,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": s["updates"],
        "trainer.save_freq": s["save_every"],
        "trainer.test_freq": s["validate_every"],
        "trainer.val_before_train": True,
        "trainer.balance_batch": True,
        "trainer.default_local_dir": str(root / "checkpoints"),
        "trainer.validation_data_dir": str(root / "validation_generations"),
        "trainer.resume_mode": "disable",  # strict marker selection in integration.py
        "trainer.max_actor_ckpt_to_keep": None,
        "trainer.del_local_ckpt_after_load": False,
        "data.train_files": manifest["data"]["train"],
        "data.val_files": manifest["data"]["validation"],
        "data.train_batch_size": s["train_batch_size"],
        "data.val_batch_size": 8,
        "data.max_prompt_length": prompt,
        "data.max_response_length": response,
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": False,
        "data.validation_shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.custom_cls.path": "pkg://d4_mopd.dataset",
        "data.custom_cls.name": "MOPDDataset",
        "data.mopd_qualification": qualification,
        "actor_rollout_ref.model.path": manifest["student"]["path"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": s["train_batch_size"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 65536,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.loss_agg_mode": s["loss_reduction"],
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.optim.lr": s["learning_rate"],
        "actor_rollout_ref.actor.optim.weight_decay": s["weight_decay"],
        "actor_rollout_ref.actor.optim.betas": s["adam_betas"],
        "actor_rollout_ref.actor.optim.clip_grad": s["grad_clip"],
        "actor_rollout_ref.actor.optim.optimizer": "adam",
        "actor_rollout_ref.actor.optim.lr_decay_style": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler": True,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": s["actor_tp"],
        "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": s["actor_pp"],
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size": s["actor_ep"],
        "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size": 1,
        "actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size": None,
        "actor_rollout_ref.actor.megatron.param_offload": True,
        "actor_rollout_ref.actor.megatron.optimizer_offload": True,
        "actor_rollout_ref.actor.megatron.use_mbridge": True,
        "actor_rollout_ref.actor.megatron.seed": s["seed"],
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method": "uniform",
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity": "full",
        "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers": 1,
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype": "fp32",
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff": 0.0,
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff": None,
        "actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout": 0.0,
        "actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout": 0.0,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.seed": s["seed"],
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": s["rollout_tp"],
        "actor_rollout_ref.rollout.max_model_len": s["max_context_tokens"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": s["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.max_num_seqs": s["max_num_seqs"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": s["max_num_batched_tokens"],
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 65536,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(root / "agent.yaml"),
        "algorithm.adv_estimator": "grpo",  # upstream plumbing only; never the distillation signal
        "algorithm.use_kl_in_reward": False,
        "algorithm.filter_groups.enable": False,
        "algorithm.rollout_correction.rollout_is": None,
        "algorithm.rollout_correction.rollout_rs": None,
        "algorithm.rollout_correction.bypass_mode": False,
        "reward.custom_reward_function.path": "pkg://d4_mopd.reward",
        "reward.custom_reward_function.name": "compute_score",
        "reward.custom_reward_function.reward_kwargs.manifest_path": manifest["manifest_path"],
        "distillation.enabled": True,
        "distillation.n_gpus_per_node": 8,
        "distillation.nnodes": 1,
        "distillation.teacher_key": "data_source",
        "distillation.distillation_loss._target_": "d4_mopd.compat.BasicPGConfig",
        "distillation.distillation_loss.loss_mode": "k1",
        "distillation.distillation_loss.use_policy_gradient": True,
        "distillation.distillation_loss.policy_loss_mode": "gpg",
        "distillation.distillation_loss.use_task_rewards": False,
        "distillation.distillation_loss.loss_max_clamp": 5.0,
        "distillation.distillation_loss.log_prob_min_clamp": None,
        "ray_kwargs.ray_init.runtime_env.py_executable": None,
    }
    OmegaConf.set_struct(c, False)
    for key, value in vals.items():
        OmegaConf.update(c, key, value, force_add=True)
    teacher_default = OmegaConf.to_container(c.distillation.teacher_models.teacher_model, resolve=True)
    # Native DistillationConfig requires the template entry and removes it when
    # resolving multiple teachers. It is not an allocated fifth teacher.
    for d in DOMAINS:
        t = OmegaConf.create(teacher_default)
        t.key, t.model_path, t.num_replicas = d, manifest["teachers"][d]["path"], 1
        t.inference.tensor_model_parallel_size = s["teacher_tp"]
        t.inference.max_model_len = prompt + response + 1
        t.inference.max_num_seqs = s["max_num_seqs"]
        t.inference.max_num_batched_tokens = s["max_num_batched_tokens"]
        t.inference.gpu_memory_utilization = s["gpu_memory_utilization"]
        c.distillation.teacher_models[d] = t
    c.mopd = {"manifest_path": manifest["manifest_path"], "identity": manifest["identity"],
              "arm": arm, "run_dir": str(root), "qualification": qualification,
              "effective_spec": s}
    OmegaConf.resolve(c)
    return c
