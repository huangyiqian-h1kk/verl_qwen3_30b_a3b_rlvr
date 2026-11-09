from types import SimpleNamespace
import asyncio
import copy
import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from d4_mopd import ARMS, DOMAINS
from d4_mopd.config import compose
from d4_mopd.compat import BasicPGConfig
from d4_mopd.preflight import validate_native_config
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.core_algos import compute_policy_loss_gpg
from verl.workers.utils.padding import left_right_2_no_padding


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("qualification", [False, True])
def test_real_hydra_and_native_dataclasses(manifest, tmp_path, arm, qualification):
    c = compose(manifest, arm, tmp_path / arm, qualification=qualification)
    assert validate_native_config(c)["teacher_gpus"] == 8
    assert c.actor_rollout_ref.rollout.n == 1
    assert c.data.train_batch_size == c.actor_rollout_ref.actor.ppo_mini_batch_size
    assert c.actor_rollout_ref.actor.ppo_epochs == 1
    assert c.trainer.test_freq == (4 if qualification else 10)
    assert c.trainer.save_freq == (4 if qualification else 5)
    assert c.distillation.distillation_loss.log_prob_min_clamp is None
    assert not c.distillation.distillation_loss.use_task_rewards
    assert not c.algorithm.filter_groups.enable
    assert c.algorithm.rollout_correction.rollout_is is None
    assert not c.actor_rollout_ref.actor.use_kl_loss
    overrides = c.actor_rollout_ref.actor.optim.override_optimizer_config
    assert (overrides.adam_beta1, overrides.adam_beta2, overrides.adam_eps) == (.9, .999, 1e-8)
    assert c.actor_rollout_ref.actor.entropy_coeff == 0
    assert c.actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff == 0
    assert c.actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff is None
    from verl.utils.config import omega_conf_to_dataclass
    assert set(omega_conf_to_dataclass(c.distillation).teacher_models) == set(DOMAINS)


def test_test_config_uses_frozen_test_rows_and_native_val_only(manifest, tmp_path):
    selected = {"step":20,"checkpoint":"/synthetic/global_step_20"}
    c = compose(manifest, "domain_mix", tmp_path / "test", test_checkpoint=selected)
    validate_native_config(c)
    assert c.trainer.val_only and c.trainer.val_before_train
    assert list(c.data.val_files) == manifest["data"]["test"]
    assert c.mopd.test_checkpoint.step == 20


def test_native_megatron_constructor_receives_explicit_adam_parameters(manifest, tmp_path):
    # Execute the pinned upstream argument-builder, intercepting OptimizerConfig.
    # This audits kwargs without pretending to instantiate a GPU/MCore optimizer.
    import ast
    from pathlib import Path
    path = Path(manifest["verl_src"]) / "verl/utils/megatron/optimizer.py"
    source = ast.parse(path.read_text())
    func = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "init_megatron_optim_config")
    namespace = dict(torch=torch, OptimizerConfig=SimpleNamespace, _MUON_ALGORITHMS=set(), print_rank_0=lambda _: None)
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)
    changed = dict(manifest, spec=dict(manifest["spec"], adam_betas=[.81,.97], adam_epsilon=2e-8))
    c = compose(changed, "domain_mix", tmp_path)
    actual = namespace["init_megatron_optim_config"](c.actor_rollout_ref.actor.optim)
    assert (actual.adam_beta1, actual.adam_beta2, actual.adam_eps) == (.81,.97,2e-8)
    assert actual.lr == changed["spec"]["learning_rate"]


def test_native_val_only_fit_never_iterates_training_or_updates(manifest, tmp_path, monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, SkipManager
    from verl.utils import tracking
    monkeypatch.setattr(tracking, "Tracking", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(SkipManager, "init", lambda *args: None)
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path, test_checkpoint={"step":20})
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    events = []
    def load():
        trainer.global_steps = 20
        events.append("load20")
    trainer._load_checkpoint = load
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: events.append(("sync",step)))
    class Loader:
        def __len__(self): return 200
        def __iter__(self): raise AssertionError("test must not train")
    trainer.train_dataloader = Loader()
    trainer._validate = lambda: {"test_score":.5}
    trainer._shutdown_dump_executor = lambda: events.append("done")
    trainer.fit()
    assert events == ["load20", ("sync",20), "done"]


def pg_config():
    return BasicPGConfig(loss_mode="k1", use_policy_gradient=True, policy_loss_mode="gpg",
                         use_task_rewards=False, loss_max_clamp=5, log_prob_min_clamp=None)


def test_native_distillation_gradient_sign_detach_and_sequence_mean():
    # Two different response lengths and both signs of the clipped teacher signal.
    # Follow actual verl packing/left-shift code, with one prompt token per sample.
    student = torch.tensor([-1., -3., -2., 0., -2., 0.], requires_grad=True)
    teacher = torch.tensor([[-.5, -20., -1., 0.], [-.2, 0., 0., 0.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    td = TensorDict({
        "prompts": torch.tensor([[8], [9]]), "responses": torch.tensor([[1, 2, 3], [4, 0, 0]]),
        "input_ids": torch.tensor([[8, 1, 2, 3], [9, 4, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "response_mask": mask, "old_log_probs": torch.zeros(2, 3),
        "teacher_ids": torch.zeros(2, 4, 1, dtype=torch.long), "teacher_logprobs": teacher.unsqueeze(-1),
    }, batch_size=[2])
    td = left_right_2_no_padding(td)
    from verl.utils import tensordict_utils as tu
    tu.assign_non_tensor(td, dp_size=1, batch_num_tokens=4, global_batch_size=2)
    actor = SimpleNamespace(loss_agg_mode="seq-mean-token-mean", global_batch_info={}, loss_scale_factor=None)
    loss, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=pg_config()), {"log_probs": student}, td)
    loss.backward()
    advantage = torch.tensor([.5, -5., 1., 0., 1.8, 0.])
    expected = -advantage * torch.tensor([1/6, 1/6, 1/6, 0., 1/2, 0.])
    torch.testing.assert_close(student.grad, expected)
    assert teacher.grad is None  # no teacher optimization, no gradient through the signal
    assert loss.detach().item() == pytest.approx(float((-advantage*student.detach()*torch.tensor([1/6,1/6,1/6,0,1/2,0])).sum()))


def test_native_microbatch_and_dp_gradient_matches_full_batch():
    torch.manual_seed(42)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    advantages = torch.randn(4, 3).clamp(-5, 5)
    initial = torch.randn(4, 3, 7)
    chosen = torch.tensor([[0, 1, 2], [3, 4, 5], [2, 1, 0], [1, 1, 1]])
    def grad(partitions, dp_size):
        logits = initial.clone().requires_grad_()
        for sl in partitions:
            lp = logits[sl].log_softmax(-1).gather(-1, chosen[sl].unsqueeze(-1)).squeeze(-1)
            cfg = SimpleNamespace(global_batch_info={"global_batch_size": 4, "dp_size": dp_size})
            loss, _ = compute_policy_loss_gpg(lp.detach(), lp, advantages[sl], mask[sl], "seq-mean-token-mean", cfg)
            (loss / dp_size).backward()  # DDP averages ranks after microbatch accumulation
        return logits.grad
    reference = grad([slice(None)], 1)
    torch.testing.assert_close(grad([slice(i,i+1) for i in range(4)], 1), reference)
    torch.testing.assert_close(grad([slice(0,2), slice(2,4)], 2), reference)


def test_actual_teacher_manager_routes_same_student_prefix(manifest, tmp_path):
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    c = compose(manifest, "domain_mix", tmp_path)
    seen = []
    class Client:
        def __init__(self, key): self.key = key
        async def generate(self, **kwargs):
            seen.append((self.key, kwargs))
            n = len(kwargs["prompt_ids"])
            return SimpleNamespace(extra_fields={"prompt_ids": [[v] for v in kwargs["prompt_ids"]],
                                                "prompt_logprobs": [[-1.0]]*n})
    manager = AsyncTeacherLLMServerManager(c, {d: Client(d) for d in DOMAINS})
    async def score():
        for d in DOMAINS:
            await manager.compute_teacher_logprobs_single([7, 8, 9], routing_key=d)
    asyncio.run(score())
    assert [x[0] for x in seen] == list(DOMAINS)
    assert all(x[1]["prompt_ids"] == [7,8,9] for x in seen)
    assert all(x[1]["sampling_params"]["prompt_logprobs"] == 0 for x in seen)
    with pytest.raises(ValueError, match="No teacher"):
        manager._resolve_teacher_key("unknown_domain")
    with pytest.raises(ValueError, match="required"):
        manager._resolve_teacher_key(None)


def test_native_imports_no_trainer_fork():
    from d4_mopd.integration import MOPDTrainer, MOPDTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    assert issubclass(MOPDTrainer, RayPPOTrainer)
    assert MOPDTaskRunner is not None
