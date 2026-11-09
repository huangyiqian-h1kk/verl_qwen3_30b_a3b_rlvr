"""Separate CPU evidence, local-node GPU evidence, and end-to-end qualification."""
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

from . import ARMS, DOMAINS
from .assets import verify
from .config import compose
from d4_teachers.common import digest, file_sha, need, now, read, write

LOCKED = {"torch": "2.13.0", "ray": "2.55.1", "tensordict": "0.10.0",
          "torchdata": "0.11.0", "transformers": "5.16.1", "vllm": "0.32.0",
          "megatron-core": "0.19.2", "megatron-bridge": "0.6.2", "transformer-engine": "2.20.2"}


def environment(*, gpu_packages=False):
    versions = {}
    for name, expected in LOCKED.items():
        if not gpu_packages and name in ("vllm", "megatron-core", "megatron-bridge", "transformer-engine"):
            continue
        version = importlib.metadata.version(name)
        need(version.split("+")[0] == expected, f"{name} {version} != upstream uv.lock {expected}")
        versions[name] = version
    return versions


def validate_native_config(config):
    from verl.utils.config import omega_conf_to_dataclass, validate_config
    validate_config(config, use_critic=False, use_reference_policy=False)
    dist = omega_conf_to_dataclass(config.distillation)
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    need(set(dist.teacher_models) == set(DOMAINS), "Four-teacher routing incomplete")
    need(dist.distillation_loss.policy_loss_mode == "gpg" and not dist.distillation_loss.use_task_rewards,
         "Unexpected instantiated distillation objective")
    need(actor.ppo_mini_batch_size == config.data.train_batch_size and actor.ppo_epochs == 1,
         "More than one optimizer update per rollout batch")
    need(actor.checkpoint.save_contents == ["model", "optimizer", "extra"], "Incomplete native checkpoint policy")
    return dict(teacher_gpus=sum(t.world_size for t in dist.teacher_models.values()), student_gpus=8,
                optimizer_updates_per_rollout_batch=1, loss="native k1 -> detach/clip5 -> native gpg")


def cpu(manifest_path):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from multidomain.template import render, tokenizer_fingerprint
    from .spec import plan_indices
    m = verify(manifest_path)
    need(Path(sys.executable).absolute() == Path(m["python"]).absolute(),
         "Run CPU preflight with the NEW pinned verl interpreter recorded at prepare")
    versions = environment(gpu_packages=True)
    import verl
    need(Path(verl.__file__).resolve().parent.parent == Path(m["verl_src"]), "Another verl shadows the pinned checkout")
    base = AutoTokenizer.from_pretrained(m["student"]["path"], local_files_only=True)
    fingerprint = tokenizer_fingerprint(base)
    for t in m["teachers"].values():
        teacher_tokenizer = AutoTokenizer.from_pretrained(t["path"], local_files_only=True)
        need(tokenizer_fingerprint(teacher_tokenizer) == fingerprint, "Teacher vocabulary/chat template differs")
        conf = read(Path(t["path"]) / "config.json")
        need(conf.get("model_type") == "qwen3_moe", "Expected Qwen3 MoE teacher")
    train = None
    for split, paths in m["data"].items():
        rows = pq.read_table(paths[0]).to_pylist()
        for row in rows:
            ids = render(base, json.loads(row["messages_json"]), json.loads(row["tools_json"]))
            need(len(ids) == row["prompt_tokens"] and digest(ids) == row["prompt_token_hash"],
                 "Frozen prompt tokenization changed: " + row["sample_id"])
        if split == "train":
            train = rows
        elif split == "validation":
            from collections import Counter
            counts = Counter(r["domain"] for r in rows)
            need(counts == {d: m["spec"]["validation_per_domain"] for d in DOMAINS},
                 "Validation domain sizes differ from the frozen recipe")
            math_sources = Counter("deepmath" if r["source"].startswith("deepmath_") else "nemotron"
                                   for r in rows if r["domain"] == "math")
            need(math_sources == {s: m["spec"]["validation_per_domain"] // 2 for s in ("nemotron", "deepmath")},
                 "Math validation must contain equal Nemotron and DeepMath counts")
    plans = {arm: plan_indices(train, m["spec"], arm) for arm in ARMS}
    b = m["spec"]["train_batch_size"]
    from collections import Counter
    exposure = {}
    for arm, indices in plans.items():
        need(len(indices) == m["spec"]["updates"] * b, "Incorrect prompt budget")
        exposure[arm] = dict(Counter(train[i]["domain"] for i in indices))
        for start in range(0, len(indices), b):
            counts = Counter(train[i]["domain"] for i in indices[start:start+b])
            need((len(counts) == 1 if arm == "domain_only" else counts == {d:b//4 for d in DOMAINS}),
                 "Incorrect optimizer batch domain composition")
    configs = {}
    for arm in ARMS:
        for qual in (False, True):
            run = Path(manifest_path).parent / ("qualification" if qual else "runs") / arm
            run.mkdir(parents=True, exist_ok=True)
            (run / "agent.yaml").write_text("- name: d4_plain_mopd\n  _target_: d4_mopd.agent_loop.MOPDAgentLoop\n")
            cfg = compose(m, arm, run, qualification=qual)
            configs[f"{arm}:{qual}"] = validate_native_config(cfg)
            OmegaConf.save(cfg, run / "resolved.yaml")
    p = subprocess.run([m["verifier_python"], "-I", str(Path(__file__).with_name("verifier_worker.py")), str(manifest_path)],
                       input='{"import_only":true}', capture_output=True, text=True, timeout=120)
    need(p.returncode == 0, "Original verifier environment failed to import: " + p.stderr[-3000:])
    proof = dict(status="PASS", identity=m["identity"], scope="cpu-assets-config-tokenization",
                 versions=versions, configs=configs, tokenizer_fingerprint=fingerprint,
                 prompt_budget_matched=True, domain_exposure=exposure,
                 validation=m["data_summary"]["validation"],
                 domain_only_schedule="iid_uniform; exposure equal only in expectation",
                 gpu_execution_tested=False, at_utc=now())
    weight_bytes = sum(v["bytes"] for k, v in m["student"]["files"].items()
                       if k.endswith((".safetensors", ".bin")))
    saves_per_arm = (m["spec"]["updates"] + m["spec"]["save_every"]-1) // m["spec"]["save_every"]
    retained_per_arm = min(saves_per_arm, m["spec"]["checkpoint_keep_recent"] + 1)
    proof["storage"] = dict(filesystem_free_bytes=shutil.disk_usage(Path(manifest_path).parent).free,
        created_formal_checkpoints=2*saves_per_arm,
        retained_formal_checkpoints_upper_bound=2*retained_per_arm, retained_qualification_checkpoints=4,
        checkpoint_keep_recent=m["spec"]["checkpoint_keep_recent"], keep_validation_best=True,
        estimated_retained_checkpoint_bytes=(2*retained_per_arm+4)*7*weight_bytes,
        estimated_peak_checkpoint_bytes=(2*(retained_per_arm+1)+4)*7*weight_bytes,
        estimate_basis="BF16 HF weights x7 for model/FP32 master/Adam moments; peak includes one new save per arm; excludes incomplete retries and overhead",
        project_quota_checked=False)
    write(Path(manifest_path).parent / "cpu_preflight.json", proof)
    return proof


def gpu_node(manifest_path, folder, rank):
    import torch
    m = verify(manifest_path, models=False)
    versions = environment(gpu_packages=True)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 8, "Allocation must expose eight GPUs per node")
    # Imports execute extension/ABI checks; driver/runtime incompatibilities fail here.
    import flash_attn  # noqa: F401
    import transformer_engine.pytorch  # noqa: F401
    import megatron.core  # noqa: F401
    import megatron.bridge  # noqa: F401
    from vllm.platforms import current_platform
    capabilities = [str(current_platform.get_device_capability(i)) for i in range(8)]
    devices = []
    for i in range(8):
        p = torch.cuda.get_device_properties(i)
        need(p.total_memory >= 120 * 1024**3, "Default topology requires H200-class >=120GiB GPUs")
        with torch.cuda.device(i):
            a = torch.ones((32, 32), device=f"cuda:{i}", dtype=torch.bfloat16)
            need(bool(torch.isfinite(a @ a).all()), "GPU BF16 matmul failed")
        devices.append(dict(name=p.name, memory=p.total_memory))
    # Exercise the cross-node prerequisite before launching Ray/NCCL workers.
    need(os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize > 8 * 1024**3,
         "Insufficient shared memory")
    proof = dict(status="PASS", identity=m["identity"], devices=devices, capabilities=capabilities,
                 versions=versions, cuda=torch.version.cuda, rank=rank, at_utc=now(),
                 scope="node-imports-bf16-only; distributed training requires qualification")
    write(Path(folder) / f"gpu_node_{rank}.json", proof)
    return proof
