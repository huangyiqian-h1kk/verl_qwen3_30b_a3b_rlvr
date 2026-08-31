#!/bin/bash
# Internal single-checkpoint runner.
# Do not qsub this file; q3_legal_rlvr_entropy_curve_serial_v2.pbs calls it.

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}

STEP=${STEP:?qsub must set STEP}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:?qsub must set SNAPSHOT_ROOT}
PROBE=${PROBE:?qsub must set PROBE}
OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b}
BASE_MODEL=${BASE_MODEL:-$WORK/models/Qwen3-30B-A3B-Instruct-2507}
SEED=${SEED:-1}
HINT_LEVELS=${HINT_LEVELS:-none}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32000}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
TP=${TP:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-1}
ENTROPY_CHUNK_TOKENS=${ENTROPY_CHUNK_TOKENS:-128}
SCORER_MAX_MEMORY_GIB_PER_GPU=${SCORER_MAX_MEMORY_GIB_PER_GPU:-16}
OVERWRITE=${OVERWRITE:-0}

case "$STEP" in
  ''|*[!0-9]*) echo "[FAIL] STEP must be a non-negative integer" >&2; exit 2 ;;
esac
case "$SEED" in
  ''|*[!0-9]*) echo "[FAIL] SEED must be a non-negative integer" >&2; exit 2 ;;
esac
if [ "$HINT_LEVELS" != none ]; then
  echo "[FAIL] serial v2 is intentionally restricted to HINT_LEVELS=none" >&2
  exit 2
fi

if [ "$STEP" -eq 0 ]; then
  MODEL=$BASE_MODEL
else
  MODEL=$SNAPSHOT_ROOT/global_step_$STEP
  test -f "$MODEL/.complete" || {
    echo "[FAIL] snapshot is not marked complete: $MODEL/.complete" >&2
    exit 3
  }
fi

test -d "$MODEL" || { echo "[FAIL] model directory missing: $MODEL" >&2; exit 3; }
test -f "$MODEL/config.json" || {
  echo "[FAIL] config.json missing from model: $MODEL" >&2
  exit 3
}
if ! find "$MODEL" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) \
    -print -quit | grep -q .; then
  echo "[FAIL] no safetensors weights found under $MODEL" >&2
  exit 3
fi
test -s "$PROBE" || { echo "[FAIL] frozen probe missing: $PROBE" >&2; exit 3; }

STEP_PADDED=$(printf '%03d' "$STEP")
RUN_DIR=$OUT_ROOT/step_$STEP_PADDED/seed_$SEED
mkdir -p "$RUN_DIR" "$PROJ/logs"

exec 9>"$RUN_DIR/.job.lock"
if ! flock -n 9; then
  echo "[FAIL] another job owns $RUN_DIR" >&2
  exit 4
fi

LOG=$PROJ/logs/legal_entropy_step${STEP_PADDED}_seed${SEED}_${PBS_JOBID:-manual}_$(date +%F_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "[run] step=$STEP model=$MODEL"
echo "[run] probe=$PROBE"
echo "[run] out=$RUN_DIR"
echo "[run] hints=$HINT_LEVELS seed=$SEED"

if [ -f "$RUN_DIR/.complete" ] && [ "$OVERWRITE" != 1 ]; then
  echo "[PASS] run is already complete: $RUN_DIR"
  exit 0
fi
if [ "$OVERWRITE" = 1 ]; then
  rm -f -- "$RUN_DIR/.complete"
fi

if [ -f "$PROJ/scripts/abci_verl_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$PROJ/scripts/abci_verl_env.sh"
fi
# shellcheck disable=SC1091
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROBE_SCRIPT=$PROJ/scripts/legal_rlvr_entropy_probe_v1.py
ORIGINAL_SCRIPT=$PROJ/scripts/legal_reasoning_probe_vllm_two_stage_entropy_noChunk.py
test -f "$PROBE_SCRIPT" || { echo "[FAIL] missing $PROBE_SCRIPT" >&2; exit 3; }
test -f "$ORIGINAL_SCRIPT" || { echo "[FAIL] missing $ORIGINAL_SCRIPT" >&2; exit 3; }

OVERWRITE_ARG=()
if [ "$OVERWRITE" = 1 ]; then
  OVERWRITE_ARG=(--overwrite)
fi

if [ ! -s "$RUN_DIR/case_tests.jsonl" ] || [ ! -s "$RUN_DIR/case_evals.jsonl" ]; then
  if [ -e "$RUN_DIR/case_tests.jsonl" ] || [ -e "$RUN_DIR/case_evals.jsonl" ]; then
    if [ "$OVERWRITE" != 1 ]; then
      echo "[FAIL] partial generation output exists; set OVERWRITE=1 intentionally" >&2
      exit 5
    fi
  fi
  python "$PROBE_SCRIPT" generate \
    --model "$MODEL" \
    --probe "$PROBE" \
    --run-dir "$RUN_DIR" \
    --checkpoint-step "$STEP" \
    --hint-levels "$HINT_LEVELS" \
    --seed "$SEED" \
    --batch-size "$GEN_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --trust-remote-code \
    "${OVERWRITE_ARG[@]}"
else
  echo "[resume] reusing complete generation output"
fi

# A new Python process guarantees that vLLM/Ray GPU allocations are gone
# before the HF full-logit scorer loads the same checkpoint.
if [ ! -s "$RUN_DIR/trajectory_entropy.jsonl" ] || [ ! -s "$RUN_DIR/token_entropy.jsonl" ]; then
  if [ -e "$RUN_DIR/trajectory_entropy.jsonl" ] || [ -e "$RUN_DIR/token_entropy.jsonl" ]; then
    if [ "$OVERWRITE" != 1 ]; then
      echo "[FAIL] partial entropy output exists; set OVERWRITE=1 intentionally" >&2
      exit 5
    fi
  fi
  python "$PROBE_SCRIPT" score \
    --model "$MODEL" \
    --run-dir "$RUN_DIR" \
    --checkpoint-step "$STEP" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --scorer-dtype bfloat16 \
    --entropy-chunk-tokens "$ENTROPY_CHUNK_TOKENS" \
    --scorer-max-memory-gib-per-gpu "$SCORER_MAX_MEMORY_GIB_PER_GPU" \
    --trust-remote-code \
    "${OVERWRITE_ARG[@]}"
else
  echo "[resume] reusing complete entropy output"
fi

N_CASES=$(grep -cve '^[[:space:]]*$' "$PROBE")
N_HINTS=$(printf '%s\n' "$HINT_LEVELS" | tr ':' '\n' | grep -cve '^[[:space:]]*$')
EXPECTED=$((N_CASES * N_HINTS))

for file in case_tests.jsonl case_evals.jsonl trajectory_entropy.jsonl; do
  ACTUAL=$(grep -cve '^[[:space:]]*$' "$RUN_DIR/$file")
  if [ "$ACTUAL" -ne "$EXPECTED" ]; then
    echo "[FAIL] $file rows=$ACTUAL expected=$EXPECTED" >&2
    exit 6
  fi
done
test -s "$RUN_DIR/token_entropy.jsonl" || {
  echo "[FAIL] token_entropy.jsonl is empty" >&2
  exit 6
}

touch "$RUN_DIR/.complete"
echo "[PASS] legal entropy step=$STEP seed=$SEED trajectories=$EXPECTED"
echo "[PASS] run_dir=$RUN_DIR"
