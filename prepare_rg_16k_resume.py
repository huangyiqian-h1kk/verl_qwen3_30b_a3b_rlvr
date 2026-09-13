#!/usr/bin/env python3
"""Prepare the r08 16K phase from the committed step-25 checkpoint.

Run in the repository root on ABCI:
    python prepare_rg_16k_resume.py

Uses only Python's standard library. Reads the *installed* r08 PBS and creates
a separate PBS/policy/run directory. Existing jobs, data, reward and launcher
are not rewritten. Checkpoint files are hard-linked on the same filesystem;
deleting the original checkpoint cannot remove the retained links.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


SOURCE_PBS = "jobs/0390_q3_rg_sixcat_r08_full500_k8.pbs"
OUTPUT_PBS = "jobs/0390_q3_rg_sixcat_r08_16k_from25_full500_k8.pbs"
SOURCE_POLICY = "config/reasoning_gym_curve_steps_v1.txt"
OUTPUT_POLICY = "config/reasoning_gym_curve_steps_16k_from25.txt"
DEFAULT_SOURCE_RUN = "inst2507_rg_sixcat_r08_k8_full_v1"
DEFAULT_VERSION = "v1_len16k"


def replace_one(text: str, pattern: str, replacement: str) -> str:
    text, count = re.subn(pattern, lambda _m: replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one match for {pattern!r}; found {count}. No existing source was edited.")
    return text


def make_pbs(source: str, version: str) -> str:
    if "inst2507_rg_sixcat_r08_k8_full_" not in source:
        raise ValueError("The source PBS is not the installed r08 experiment.")
    for anchor in ("DATA_DIR=", 'bash "$LAUNCHER"', 'python "$PREFLIGHT"'):
        if anchor not in source:
            raise ValueError(f"Source PBS is missing {anchor!r}")
    if "trainer.total_training_steps=500" not in source:
        raise ValueError("Expected the existing global 500-step budget.")

    source = replace_one(source, r"^#PBS -N .+$", "#PBS -N rg_r08_16k")
    source = replace_one(source, r"^RUN_VERSION=.*$", f"RUN_VERSION=${{RUN_VERSION:-{version}}}")
    source = replace_one(
        source, r"^EXPERIMENT_NAME=inst2507_rg_sixcat_r08_k8_full_.*$",
        f'[[ "$RUN_VERSION" == "{version}" ]] || {{ echo "[FAIL] This prepared phase uses RUN_VERSION={version}" >&2; exit 2; }}\n'
        'EXPERIMENT_NAME=inst2507_rg_sixcat_r08_k8_full_${RUN_VERSION}',
    )
    source = replace_one(source, r"^LOG=.*$",
                         'LOG=$PROJ/logs/0390_q3_rg6_r08_16k_${RUN_VERSION}_${PBS_JOBID:-manual}_$(date +%F_%H%M%S).log')
    source = replace_one(source, r"^SNAPSHOT_POLICY=.*$", f"SNAPSHOT_POLICY=$PROJ/{OUTPUT_POLICY}")
    source = replace_one(source, r"^MAX_RESPONSE_LENGTH=\d+ \\$", "MAX_RESPONSE_LENGTH=16384 \\")
    source = replace_one(source, r"^ROLLOUT_MAX_MODEL_LEN=\d+ \\$", "ROLLOUT_MAX_MODEL_LEN=20480 \\")
    source = replace_one(source, r'^  --item "max_response_length=\d+" \\$',
                         '  --item "max_response_length=16384" \\')
    source = replace_one(source, r'^  --item "rollout_max_model_len=\d+" \\$',
                         '  --item "rollout_max_model_len=20480" \\')
    source = replace_one(source, r"^TOTAL_EPOCHS=\d+ \\$", "TOTAL_EPOCHS=3 \\")
    source = replace_one(source, r"^  trainer.total_epochs=\d+ \\$", "  trainer.total_epochs=3 \\")
    source = replace_one(source, r"^  \+\+reward_model.reward_kwargs.max_resp_len=\d+ \\$",
                         "  ++reward_model.reward_kwargs.max_resp_len=16384 \\")
    # Keep the reward behaviour actually used by the V1 r08 run: naive manager,
    # binary correctness + 0.05 format. Do not activate a new length penalty.
    source = replace_one(source, r"^  \+\+reward_model.reward_kwargs.overlong_buffer_cfg.enable=\w+ \\$",
                         "  ++reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \\")

    # Remove the inherited claim that this new phase starts from the base model.
    # The installation creates phase_start_step_25.json instead.
    baseline_start = source.find("# Record step 0")
    baseline_end = source.find("\nWATCHER_LOG=", baseline_start)
    if baseline_start < 0 or baseline_end < 0:
        raise ValueError("Cannot locate the baseline/resume block in the installed PBS.")
    resume_block = r'''# This phase was seeded with a complete step-25 checkpoint by the preparer.
# On resubmission select this phase's latest committed checkpoint, never restart.
TRACKER=$FULL_CKPT_DIR/latest_checkpointed_iteration.txt
[[ -f "$FULL_CKPT_DIR/phase_start_step_25.json" && -f "$TRACKER" ]] || {
  echo "[FAIL] Run prepare_rg_16k_resume.py in the repository before qsub." >&2; exit 2;
}
RESUME_STEP=$(tr -d '[:space:]' < "$TRACKER")
case "$RESUME_STEP" in ''|*[!0-9]*) echo "[FAIL] Invalid checkpoint tracker: $RESUME_STEP" >&2; exit 2 ;; esac
[[ "$RESUME_STEP" -ge 25 ]] || { echo "[FAIL] Expected a checkpoint at step >= 25" >&2; exit 2; }
RESUME_FROM_PATH=$FULL_CKPT_DIR/global_step_$RESUME_STEP
[[ -f "$RESUME_FROM_PATH/data.pt" && -f "$RESUME_FROM_PATH/actor/ckpt_contents.json" ]] || {
  echo "[FAIL] Missing full checkpoint/data.pt at $RESUME_FROM_PATH" >&2; exit 2;
}
VAL_BEFORE_TRAIN=False
[[ "$RESUME_STEP" -eq 25 ]] && VAL_BEFORE_TRAIN=True
echo "[PASS] Resume $RESUME_FROM_PATH; response=16384; context=20480; val_before_train=$VAL_BEFORE_TRAIN"
'''
    source = source[:baseline_start] + resume_block + source[baseline_end:]
    source = replace_one(
        source, r"^  trainer.resume_mode=auto \\$",
        '  trainer.resume_mode=resume_path \\\n'
        '  trainer.resume_from_path="$RESUME_FROM_PATH" \\\n'
        '  trainer.del_local_ckpt_after_load=False \\\n'
        "  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \\\n"
        '  actor_rollout_ref.actor.checkpoint.async_save=False \\\n'
        '  reward.reward_manager.name=naive \\\n'
        '  reward_model.reward_manager=naive \\',
    )
    # The first comments are instructions for this derived job, not the 8K job.
    source = re.sub(r"^#   qsub.*$", f"#   qsub -v RTYPE=rt_HF,ACTOR_LR=1e-6,RUN_VERSION={version} {OUTPUT_PBS}",
                    source, flags=re.MULTILINE)
    source = source.replace("# Resubmit the identical command after walltime/SIGTERM; resume_mode=auto.",
                            "# Resubmit the identical command to continue from this phase's latest checkpoint.")
    subprocess.run(["bash", "-n"], input=source, text=True, check=True)
    return source


def check_checkpoint(path: Path, step: int) -> dict:
    """Check the small manifest and saved-file inventory without loading weights."""
    if not (path / "data.pt").is_file():
        raise ValueError(f"Missing dataloader state: {path / 'data.pt'}")
    actor = path / "actor"
    manifest = json.loads((actor / "ckpt_contents.json").read_text())
    if int(manifest.get("global_step", -1)) != step:
        raise ValueError(f"Checkpoint manifest does not describe step {step}: {actor}")
    if not {"model", "optimizer", "extra"} <= set(manifest.get("save_contents", [])):
        raise ValueError(f"This is not a full model/optimizer/extra checkpoint: {actor}")
    for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
        info = manifest.get("contents", {}).get(name)
        if not info or "path" not in info:
            raise ValueError(f"Checkpoint manifest lacks {name}: {actor}")
        directory = actor / info["path"]
        if not directory.is_dir() or not any(p.is_file() for p in directory.rglob("*")):
            raise ValueError(f"Missing/empty {name} checkpoint: {directory}")
        if info.get("format") == "huggingface":
            index = directory / "model.safetensors.index.json"
            if index.is_file():
                weights = set(json.loads(index.read_text())["weight_map"].values())
            else:
                weights = {p.name for p in directory.glob("*.safetensors")}
            if not (directory / "config.json").is_file() or not weights:
                raise ValueError(f"Missing model config/weights: {directory}")
            if any(not (directory / name).is_file() for name in weights):
                raise ValueError(f"Missing model weight shard: {directory}")
    return manifest


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: dict) -> None:
    write_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def link_checkpoint(source: Path, destination: Path) -> None:
    """Link immutable, committed files while excluding the old run's pruner."""
    prune_lock = source.parent / ".full_ckpt_prune_lock"
    try:
        prune_lock.mkdir()
    except FileExistsError as exc:
        raise ValueError("The original checkpoint pruner is active; rerun this command shortly.") from exc
    tmp = destination.with_name(f".{destination.name}.linking.{os.getpid()}")
    try:
        latest = int((source.parent / "latest_checkpointed_iteration.txt").read_text().strip())
        if latest < 25:
            raise ValueError("Step 25 has not been committed yet. Wait for its checkpoint and validation.")
        check_checkpoint(source, 25)

        def link_file(src: str, dst: str) -> str:
            # Dereference source symlinks so pruning the old tree cannot break them.
            os.link(Path(src).resolve(strict=True), dst)
            return dst

        shutil.copytree(source, tmp, copy_function=link_file, symlinks=False)
        check_checkpoint(tmp, 25)
        os.rename(tmp, destination)
    except OSError as exc:
        raise ValueError(f"Cannot retain checkpoint using hard links: {exc}. Source files were not changed.") from exc
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
        prune_lock.rmdir()


def prepare(root: Path, version: str, source_run: str) -> Path:
    root = root.resolve()
    new_run = f"inst2507_rg_sixcat_r08_k8_full_{version}"
    if new_run == source_run:
        raise ValueError("Use a new run version for the 16K phase.")
    # Read the server's working files, including all fixes that are not on GitHub.
    pbs_text = make_pbs((root / SOURCE_PBS).read_text(), version)
    policy = (root / SOURCE_POLICY).read_text()
    policy_steps = {int(l.split("#", 1)[0].strip()) for l in policy.splitlines() if l.split("#", 1)[0].strip()}
    policy_steps.add(25)
    policy_text = "# 16K phase: retain the step-25 starting model and the existing curve steps.\n"
    policy_text += "\n".join(str(x) for x in sorted(policy_steps)) + "\n"
    for rel in ("jobs/run_qwen3_30b_a3b_megatron.sh", "scripts/preflight_reasoning_gym_training.py",
                "data/reasoning_gym_sixcat_r08/BUILD_COMPLETE"):
        if not (root / rel).is_file():
            raise ValueError(f"Missing installed dependency: {root / rel}")

    full_root = root / "ckpts/verl_full"
    old_checkpoint = full_root / source_run / "global_step_25"
    new_dir = full_root / new_run
    phase = {
        "initial_step": 25, "source_checkpoint": str(old_checkpoint),
        "experiment": new_run, "source_response_length": 8192,
        "max_response_length": 16384, "max_model_len": 20480,
        "data_dir": str(root / "data/reasoning_gym_sixcat_r08"),
        "checkpoint_retention_method": "hardlinks to committed files; existing source unmodified",
    }
    new_dir.mkdir(parents=True, exist_ok=True)
    with (new_dir / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("The 16K phase is already running; its files were not changed.") from exc
        marker = new_dir / "phase_start_step_25.json"
        tracker = new_dir / "latest_checkpointed_iteration.txt"
        if marker.exists() and json.loads(marker.read_text()) != phase:
            raise ValueError("This output directory belongs to a different prepared phase.")
        if not marker.exists() and (tracker.exists() or (new_dir / "run_fingerprint.json").exists()):
            raise ValueError("The new run directory already contains another run; use a new --run-version.")
        for path, text in ((root / OUTPUT_PBS, pbs_text), (root / OUTPUT_POLICY, policy_text)):
            if path.exists() and path.read_text() != text:
                raise ValueError(f"A different generated file already exists: {path}. Keep it or rename it first.")

        seed = new_dir / "global_step_25"
        if not tracker.exists():
            if not seed.exists():
                if not old_checkpoint.is_dir():
                    raise ValueError(f"Full step-25 checkpoint is missing: {old_checkpoint}. A model-only snapshot cannot resume the optimizer.")
                print(f"[INFO] Retaining full checkpoint with hard links: {old_checkpoint}", flush=True)
                link_checkpoint(old_checkpoint, seed)
            check_checkpoint(seed, 25)
            write_json(marker, phase)
            write_atomic(tracker, "25\n")
        else:
            latest = int(tracker.read_text().strip())
            if latest < 25:
                raise ValueError(f"Invalid phase checkpoint step: {latest}")
            check_checkpoint(new_dir / f"global_step_{latest}", latest)
        write_atomic(root / OUTPUT_POLICY, policy_text)
        write_atomic(root / OUTPUT_PBS, pbs_text)
        (root / OUTPUT_PBS).chmod(0o755)
    print(f"[PASS] Full checkpoint retained; resume step = {tracker.read_text().strip()}")
    print(f"[PASS] Job: {root / OUTPUT_PBS}")
    print(f"[PASS] Snapshot policy includes step 25: {root / OUTPUT_POLICY}")
    print("[PASS] response=16384, context=20480, total_steps=500, epochs=3, eval_freq=25, save_freq=5")
    print("[PASS] First launch evaluates the restored step-25 model at 16K before updating weights.")
    print("[PASS] Existing data, launcher, preflight and reward source were not rewritten.")
    print("Stop the old 8K job after its step-25 evaluation, then submit:")
    print(f"qsub -v RTYPE=rt_HF,ACTOR_LR=1e-6,RUN_VERSION={version} {OUTPUT_PBS}")
    return root / OUTPUT_PBS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    for name in (args.run_version, args.source_run):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            parser.error(f"Invalid run name: {name!r}")
    try:
        prepare(args.repo_root, args.run_version, args.source_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc


if __name__ == "__main__":
    main()
