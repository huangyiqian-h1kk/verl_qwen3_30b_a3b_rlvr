#!/bin/bash
# Submit one ABCI PBS job per checkpoint and optionally a dependent aggregate.

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}

LEGAL_DATA=${LEGAL_DATA:-$PROJ/data/remain_all_cases.jsonl}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT to the DeepMath RLVR snapshot root}
BASE_MODEL=${BASE_MODEL:-$WORK/models/Qwen3-30B-A3B-Instruct-2507}
OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b}
PROBE=${PROBE:-$PROJ/data/legal_entropy_probe/legal_probe_n100_seed1.jsonl}
STEPS=${STEPS:-"0 10 20 30 40 50 60 70 80 90 100"}
SEED=${SEED:-1}
PROBE_SEED=${PROBE_SEED:-1}
SAMPLE_SIZE=${SAMPLE_SIZE:-100}
HINT_LEVELS=${HINT_LEVELS:-none:weak:strong}
SUBMIT_AGGREGATE=${SUBMIT_AGGREGATE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32000}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-1}
ENTROPY_CHUNK_TOKENS=${ENTROPY_CHUNK_TOKENS:-128}
SCORER_MAX_MEMORY_GIB_PER_GPU=${SCORER_MAX_MEMORY_GIB_PER_GPU:-16}

test -f "$LEGAL_DATA" || { echo "[FAIL] LEGAL_DATA missing: $LEGAL_DATA" >&2; exit 2; }
test -d "$SNAPSHOT_ROOT" || {
  echo "[FAIL] SNAPSHOT_ROOT missing: $SNAPSHOT_ROOT" >&2
  exit 2
}
test -d "$BASE_MODEL" || { echo "[FAIL] BASE_MODEL missing: $BASE_MODEL" >&2; exit 2; }

mkdir -p "$(dirname "$PROBE")" "$OUT_ROOT" "$PROJ/logs"

# shellcheck disable=SC1091
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
python "$PROJ/scripts/prepare_legal_entropy_probe_v1.py" \
  --input "$LEGAL_DATA" \
  --output "$PROBE" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$PROBE_SEED" \
  --reuse-if-matches

job_ids=()
submission_manifest=$OUT_ROOT/submitted_jobs_$(date +%F_%H%M%S).txt

for step in $STEPS; do
  case "$step" in
    ''|*[!0-9]*) echo "[FAIL] invalid step in STEPS: $step" >&2; exit 2 ;;
  esac
  if [ "$step" -gt 0 ]; then
    snapshot=$SNAPSHOT_ROOT/global_step_$step
    test -f "$snapshot/.complete" || {
      echo "[FAIL] incomplete or missing snapshot: $snapshot" >&2
      exit 3
    }
  fi

  step_padded=$(printf '%03d' "$step")
  pbs_log=$PROJ/logs/legal_entropy_step${step_padded}_seed${SEED}.pbs.log
  variables="RTYPE=rt_HF,STEP=$step,SNAPSHOT_ROOT=$SNAPSHOT_ROOT,BASE_MODEL=$BASE_MODEL"
  variables="$variables,PROBE=$PROBE,OUT_ROOT=$OUT_ROOT,SEED=$SEED"
  variables="$variables,HINT_LEVELS=$HINT_LEVELS"
  variables="$variables,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,MAX_MODEL_LEN=$MAX_MODEL_LEN"
  variables="$variables,TEMPERATURE=$TEMPERATURE,TOP_P=$TOP_P,TOP_K=$TOP_K"
  variables="$variables,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  variables="$variables,GEN_BATCH_SIZE=$GEN_BATCH_SIZE"
  variables="$variables,SCORE_BATCH_SIZE=$SCORE_BATCH_SIZE"
  variables="$variables,ENTROPY_CHUNK_TOKENS=$ENTROPY_CHUNK_TOKENS"
  variables="$variables,SCORER_MAX_MEMORY_GIB_PER_GPU=$SCORER_MAX_MEMORY_GIB_PER_GPU"

  job_id=$(
    qsub \
      -N "le_s${step_padded}" \
      -o "$pbs_log" \
      -v "$variables" \
      "$PROJ/jobs/q3_legal_rlvr_entropy_probe_v1.pbs"
  )
  job_ids+=("$job_id")
  printf 'step=%s job_id=%s log=%s\n' "$step" "$job_id" "$pbs_log" \
    | tee -a "$submission_manifest"
done

if [ "$SUBMIT_AGGREGATE" = 1 ]; then
  dependency=$(IFS=:; echo "${job_ids[*]}")
  aggregate_log=$PROJ/logs/legal_entropy_aggregate.pbs.log
  aggregate_job=$(
    qsub \
      -W "depend=afterok:$dependency" \
      -o "$aggregate_log" \
      -v "RTYPE=rt_HF,OUT_ROOT=$OUT_ROOT" \
      "$PROJ/jobs/q3_legal_rlvr_entropy_aggregate_v1.pbs"
  )
  printf 'aggregate_job_id=%s depends_on=%s log=%s\n' \
    "$aggregate_job" "$dependency" "$aggregate_log" \
    | tee -a "$submission_manifest"
fi

echo "[PASS] submitted ${#job_ids[@]} checkpoint jobs"
echo "[PASS] submission manifest: $submission_manifest"
