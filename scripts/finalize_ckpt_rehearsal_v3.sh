#!/usr/bin/env bash
# Finish the already-trained 5->10->15 rehearsal without another GPU job.

set -euo pipefail

PROJ=${PROJ:-/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-inst2507_dapo_unicode_ckptverify_5step_v3}
CKPT_DIR=${CKPT_DIR:-"$HOME/ckpts/verl_full/$EXPERIMENT_NAME"}
SNAP_DIR=${SNAP_DIR:-"$PROJ/ckpts/model_snapshots/$EXPERIMENT_NAME"}
PRUNER=${PRUNER:-"$PROJ/scripts/prune_full_checkpoints_v3.sh"}

[ -x "$PRUNER" ] || {
  echo "[FAIL] pruner is missing or not executable: $PRUNER" >&2
  exit 2
}

tracker="$CKPT_DIR/latest_checkpointed_iteration.txt"
latest=$(tr -d '[:space:]' < "$tracker" 2>/dev/null || true)
[ "$latest" = 15 ] || {
  echo "[FAIL] expected tracker=15, got: ${latest:-missing}" >&2
  exit 3
}

for step in 5 10 15; do
  [ -f "$SNAP_DIR/global_step_$step/.complete" ] || {
    echo "[FAIL] snapshot global_step_$step is not complete" >&2
    exit 3
  }
done

bash "$PRUNER" "$CKPT_DIR" "$SNAP_DIR" 2

[ ! -d "$CKPT_DIR/global_step_5" ] || {
  echo "[FAIL] external rotation did not remove global_step_5" >&2
  exit 4
}
for step in 10 15; do
  [ -d "$CKPT_DIR/global_step_$step" ] || {
    echo "[FAIL] retained full checkpoint global_step_$step is missing" >&2
    exit 4
  }
done

printf 'completed=%s source_checkpoint=global_step_10 target_step=15 rotation=external_safe_pruner\n' \
  "$(date --iso-8601=seconds)" > "$CKPT_DIR/resume_to_step_15.PASS"

echo "================ REHEARSAL FINALIZED ================"
echo "tracker: $latest"
echo "home checkpoints:"
find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' \
  -printf '  %f\n' | sort -V
echo "group snapshots:"
find "$SNAP_DIR" -maxdepth 2 -name .complete \
  -printf '  %h\n' | sort -V
echo "[PASS] save, resume, external rotation, and harvest"
echo "======================================================"
