#!/bin/bash
# Prepare one frozen legal probe and submit exactly one serial PBS job.

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}

MODE=${MODE:-smoke}
LEGAL_DATA=${LEGAL_DATA:?set LEGAL_DATA to the original legal JSONL}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT to the RLVR model-only snapshot root}
BASE_MODEL=${BASE_MODEL:-$WORK/models/Qwen3-30B-A3B-Instruct-2507}
SEED=${SEED:-1}
PROBE_SEED=${PROBE_SEED:-1}
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
OVERWRITE=${OVERWRITE:-0}

case "$MODE" in
  smoke)
    SAMPLE_SIZE=${SAMPLE_SIZE:-2}
    STEPS=${STEPS:-"0 100"}
    PROBE=${PROBE:-$PROJ/data/legal_entropy_probe/legal_probe_n${SAMPLE_SIZE}_seed${PROBE_SEED}.jsonl}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b_smoke}
    ;;
  full)
    SAMPLE_SIZE=${SAMPLE_SIZE:-100}
    STEPS=${STEPS:-"0 10 20 30 40 50 60 70 80 90 100"}
    PROBE=${PROBE:-$PROJ/data/legal_entropy_probe/legal_probe_n${SAMPLE_SIZE}_seed${PROBE_SEED}.jsonl}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b}
    ;;
  *)
    echo "[FAIL] MODE must be smoke or full" >&2
    exit 2
    ;;
esac

test -f "$LEGAL_DATA" || { echo "[FAIL] LEGAL_DATA missing: $LEGAL_DATA" >&2; exit 2; }
test -d "$SNAPSHOT_ROOT" || {
  echo "[FAIL] SNAPSHOT_ROOT missing: $SNAPSHOT_ROOT" >&2
  exit 2
}
test -d "$BASE_MODEL" || { echo "[FAIL] BASE_MODEL missing: $BASE_MODEL" >&2; exit 2; }

STEP_LIST=$(printf '%s' "$STEPS" | tr ' ,' '::' | tr -s ':' | sed 's/^://;s/:$//')
test -n "$STEP_LIST" || { echo "[FAIL] STEPS is empty" >&2; exit 2; }

for step in $(printf '%s' "$STEP_LIST" | tr ':' ' '); do
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
done

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

PBS_LOG=$PROJ/logs/legal_entropy_serial_${MODE}.pbs.log
VARIABLES="RTYPE=rt_HF,SNAPSHOT_ROOT=$SNAPSHOT_ROOT,BASE_MODEL=$BASE_MODEL"
VARIABLES="$VARIABLES,PROBE=$PROBE,OUT_ROOT=$OUT_ROOT,STEP_LIST=$STEP_LIST"
VARIABLES="$VARIABLES,SEED=$SEED,HINT_LEVELS=none"
VARIABLES="$VARIABLES,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,MAX_MODEL_LEN=$MAX_MODEL_LEN"
VARIABLES="$VARIABLES,TEMPERATURE=$TEMPERATURE,TOP_P=$TOP_P,TOP_K=$TOP_K"
VARIABLES="$VARIABLES,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
VARIABLES="$VARIABLES,GEN_BATCH_SIZE=$GEN_BATCH_SIZE,SCORE_BATCH_SIZE=$SCORE_BATCH_SIZE"
VARIABLES="$VARIABLES,ENTROPY_CHUNK_TOKENS=$ENTROPY_CHUNK_TOKENS"
VARIABLES="$VARIABLES,SCORER_MAX_MEMORY_GIB_PER_GPU=$SCORER_MAX_MEMORY_GIB_PER_GPU"
VARIABLES="$VARIABLES,OVERWRITE=$OVERWRITE,RUN_AGGREGATE=1"

JOB_ID=$(
  qsub \
    -N "0390_legal_ent_${MODE}" \
    -o "$PBS_LOG" \
    -v "$VARIABLES" \
    "$PROJ/jobs/q3_legal_rlvr_entropy_curve_serial_v2.pbs"
)

MANIFEST=$OUT_ROOT/submitted_serial_job_$(date +%F_%H%M%S).txt
{
  echo "mode=$MODE"
  echo "job_id=$JOB_ID"
  echo "steps=$STEP_LIST"
  echo "hint_levels=none"
  echo "probe=$PROBE"
  echo "out_root=$OUT_ROOT"
  echo "pbs_log=$PBS_LOG"
} | tee "$MANIFEST"

echo "[PASS] submitted exactly one serial job: $JOB_ID"
echo "[PASS] manifest: $MANIFEST"
