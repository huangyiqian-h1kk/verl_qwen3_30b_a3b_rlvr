# Checkpoint rotation fix v3

This patch makes `/home` full-checkpoint retention safe across PBS restarts.
It does not rely on verl's process-local retention state.

## Files to install

Copy these files into the project, preserving their relative paths:

- `compat/sitecustomize.py`
- `scripts/prune_full_checkpoints_v3.sh`
- `scripts/snapshot_watcher_v3.sh`
- `scripts/finalize_ckpt_rehearsal_v3.sh`
- `scripts/check_home_quota_v3.sh`
- `jobs/q3_ckpt_scheme_verify_5step_v3.pbs`
- `jobs/q3_grpo_train_v3.pbs`

Make the shell scripts executable:

```bash
chmod +x \
  scripts/prune_full_checkpoints_v3.sh \
  scripts/snapshot_watcher_v3.sh \
  scripts/finalize_ckpt_rehearsal_v3.sh \
  scripts/check_home_quota_v3.sh
```

## Finish the existing rehearsal without another GPU job

The existing run already has tracker 15 and complete group snapshots at steps
5, 10, and 15. Run this once on an ABCI login node:

```bash
cd /groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr
bash scripts/finalize_ckpt_rehearsal_v3.sh
```

The script checks every safety condition, removes only the old full checkpoint
`global_step_5`, retains full checkpoints 10 and 15, and writes
`resume_to_step_15.PASS`.

## Independent snapshot load test

After the finalizer passes:

```bash
qsub -v RTYPE=rt_HF,SNAPSHOT_DIR=/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr/ckpts/model_snapshots/inst2507_dapo_unicode_ckptverify_5step_v3/global_step_15 \
  jobs/q3_validate_snapshot_v2.pbs
```

## How automatic rotation works

`snapshot_watcher_v3.sh` first copies and verifies a Hugging Face snapshot. It
creates `.complete` only after the model shards, config, and tokenizer checks
pass. It then invokes `prune_full_checkpoints_v3.sh`.

The pruner:

1. reads `latest_checkpointed_iteration.txt`;
2. keeps the newest two committed full checkpoints;
3. refuses to delete an older checkpoint unless its group snapshot has
   `.complete`, `.snapshot_meta`, and `config.json`;
4. refuses paths outside `$HOME/ckpts/verl_full`;
5. logs each deletion and uses a lock to prevent concurrent pruning.

The v3 PBS files set `KEEP_FULL_CKPTS=2` and make this external pruner
authoritative. `trainer.max_actor_ckpt_to_keep` is set to `null`, avoiding an
unverified deletion before snapshot harvest.

## Submitting formal training

Immediately before each submission, capture current home free space:

```bash
HOME_FREE_GIB_AT_SUBMIT=$(
  show_quota -b G |
  awk -v home_path="$HOME" '$1 == home_path {print $3 - $2; exit}'
)
test -n "$HOME_FREE_GIB_AT_SUBMIT"
echo "$HOME_FREE_GIB_AT_SUBMIT"
```

Pass that value through `qsub`, together with the normal formal-training
variables. Use `jobs/q3_grpo_train_v3.pbs`, not the v2 file.
