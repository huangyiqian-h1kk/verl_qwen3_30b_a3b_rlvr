"""Single-domain validation without the legacy mandatory Math+IF rule."""

import copy
from pathlib import Path
from . import VERSION, DOMAINS
from .common import need, read, safe_id

DEFAULT = Path(__file__).resolve().parents[1] / "config/domain_teachers/campaign.json"


def load(path=DEFAULT):
    c = read(path)
    need(c["version"] == VERSION, "Unknown teacher configuration version")
    safe_id(c["campaign"])
    for k in (
        "seed",
        "max_steps",
        "save_every",
        "validate_every",
        "early_stop_min_step",
        "early_stop_patience_steps",
        "max_concurrent_jobs",
        "poll_seconds",
    ):
        need(type(c[k]) is int and c[k] >= 0, "Invalid integer: " + k)
    need(1 <= c["max_concurrent_jobs"] <= 4, "Campaign concurrency must be 1..4")
    need(c["max_steps"] > 0 and c["poll_seconds"] >= 5, "Invalid budget/poll interval")
    need(
        c["max_steps"] == 200
        and c["early_stop_min_step"] == 50
        and c["early_stop_patience_steps"] == 25,
        "Teacher budget/patience must remain 200/50/25",
    )
    need(
        c["save_every"] == 5 and c["validate_every"] == 10,
        "Teacher policy is fixed: save every 5; validate every 10 (no adaptive interval)",
    )
    need(
        0 < c["early_stop_patience_steps"]
        and c["early_stop_min_step"] <= c["max_steps"],
        "Invalid early stopping policy",
    )
    need(
        c["retain_full_during_training"] == 2 and c["retain_full_after_training"] == 1,
        "Unexpected checkpoint retention",
    )
    need(
        not c["auto_test"] or c["auto_export"],
        "Automatic test requires automatic export",
    )
    return c


def teacher_config(base, policy, domain, runtime_root):
    need(domain in DOMAINS, "Unknown teacher domain")
    c = copy.deepcopy(base)
    c["project_root"] = str(runtime_root)
    c["seed"] = policy["seed"]
    for name, value in c["domains"].items():
        value.update(enabled=name == domain, required=name == domain)
    c["data"]["balance"] = "frozen_parent_domain_slice"
    c["training"].update(
        train_nodes=1,
        gpus_per_node=8,
        total_epochs=1,
        total_training_steps=policy["max_steps"],
        save_freq=5,
        test_freq=10,
        max_actor_checkpoints=None,
        final_hf_export=False,
        seed=policy["seed"],
    )
    c["scheduler"]["walltime"] = "24:00:00"
    c["acceptance"]["infrastructure_nodes"] = 1
    c["resolved"].update(
        enabled_domains=[domain],
        domain_count=1,
        domain_target_fraction={domain: 1.0},
        judge_enabled=False,
        judge_gpus=0,
        judge_replicas=0,
        pbs_nodes_for_training_job=1,
        train_gpus=8,
        training_rollout_replicas=2,
        target_train_rows=10000,
    )
    validate_teacher(c, domain)
    return c


def validate_teacher(c, domain):
    need(
        [k for k, v in c["domains"].items() if v["enabled"]] == [domain],
        "Exactly the requested domain must be enabled",
    )
    t = c["training"]
    expected = dict(
        train_nodes=1,
        gpus_per_node=8,
        actor_tp=2,
        actor_pp=2,
        actor_ep=2,
        actor_cp=1,
        rollout_tp=4,
        train_batch_size=256,
        rollout_n=8,
        ppo_mini_batch_size=32,
        ppo_epochs=1,
        learning_rate=1e-6,
        entropy_coeff=0.0,
        use_kl_loss=False,
        use_kl_in_reward=False,
        save_freq=5,
        test_freq=10,
        val_n=1,
        val_temperature=0.0,
    )
    expected.update(
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        ppo_micro_batch_size_per_gpu=1,
        param_optimizer_grad_offload=True,
        gradient_checkpointing=True,
    )
    need(
        all(t.get(k) == v for k, v in expected.items()),
        "Teacher training recipe changed",
    )
    need(
        not c["resolved"]["judge_enabled"] and c["data"]["max_context_tokens"] == 65536,
        "Teacher requires local verifiers and the frozen 64K context policy",
    )
