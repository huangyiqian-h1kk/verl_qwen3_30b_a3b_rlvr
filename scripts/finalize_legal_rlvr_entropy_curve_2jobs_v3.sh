#!/bin/bash
# Validate both partitions, then aggregate on the login node (no GPU needed).

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}
MODE=${MODE:-full}
SEED=${SEED:-1}

case "$MODE" in
  smoke)
    STEPS=${STEPS:-"0 100"}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b_smoke_unicode}
    ;;
  full)
    STEPS=${STEPS:-"0 10 20 30 40 50 60 70 80 90 100"}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b_unicode_none}
    ;;
  *)
    echo "[FAIL] MODE must be smoke or full" >&2
    exit 2
    ;;
esac

test -f "$OUT_ROOT/.part_A_complete" || {
  echo "[FAIL] part A is not complete: $OUT_ROOT/.part_A_complete" >&2
  exit 3
}
test -f "$OUT_ROOT/.part_B_complete" || {
  echo "[FAIL] part B is not complete: $OUT_ROOT/.part_B_complete" >&2
  exit 3
}

for step in $STEPS; do
  step_padded=$(printf '%03d' "$step")
  run_dir=$OUT_ROOT/step_$step_padded/seed_$SEED
  test -f "$run_dir/.complete" || {
    echo "[FAIL] incomplete checkpoint result: $run_dir" >&2
    exit 3
  }
  for file in case_tests.jsonl case_evals.jsonl trajectory_entropy.jsonl token_entropy.jsonl; do
    test -s "$run_dir/$file" || {
      echo "[FAIL] missing or empty: $run_dir/$file" >&2
      exit 3
    }
  done
done

# shellcheck disable=SC1091
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
python - <<'PY'
import matplotlib
import numpy
import pandas
print(f"[PASS] aggregation dependencies: matplotlib={matplotlib.__version__}")
PY

OUT_ROOT="$OUT_ROOT" \
  bash "$PROJ/scripts/run_legal_rlvr_entropy_aggregate_v2.sh"

test -f "$OUT_ROOT/curves/.complete" || {
  echo "[FAIL] aggregation returned without curves/.complete" >&2
  exit 4
}
touch "$OUT_ROOT/.two_jobs_complete"
echo "[PASS] experiment 0390 two-job entropy curve complete: $OUT_ROOT"
