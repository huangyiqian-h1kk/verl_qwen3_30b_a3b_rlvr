#!/bin/bash
# Prepare one frozen probe and submit two checkpoint-partitioned PBS jobs.

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}

MODE=${MODE:-smoke}
PARTS=${PARTS:-both}
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
    STEP_LIST_A=${STEP_LIST_A:-0}
    STEP_LIST_B=${STEP_LIST_B:-100}
    PROBE=${PROBE:-$PROJ/data/legal_entropy_probe/legal_probe_n${SAMPLE_SIZE}_seed${PROBE_SEED}.jsonl}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b_smoke_unicode}
    JOB_NAME_A=0390_lesm_a
    JOB_NAME_B=0390_lesm_b
    ;;
  full)
    SAMPLE_SIZE=${SAMPLE_SIZE:-100}
    STEP_LIST_A=${STEP_LIST_A:-0:10:20:30:40:50}
    STEP_LIST_B=${STEP_LIST_B:-60:70:80:90:100}
    PROBE=${PROBE:-$PROJ/data/legal_entropy_probe/legal_probe_n${SAMPLE_SIZE}_seed${PROBE_SEED}.jsonl}
    OUT_ROOT=${OUT_ROOT:-$PROJ/outputs/legal_entropy_curve_qwen3_30b_unicode_none}
    JOB_NAME_A=0390_lefl_a
    JOB_NAME_B=0390_lefl_b
    ;;
  *)
    echo "[FAIL] MODE must be smoke or full" >&2
    exit 2
    ;;
esac

case "$PARTS" in
  both|A|B) ;;
  *) echo "[FAIL] PARTS must be both, A, or B" >&2; exit 2 ;;
esac

test -f "$LEGAL_DATA" || { echo "[FAIL] LEGAL_DATA missing: $LEGAL_DATA" >&2; exit 2; }
test -d "$SNAPSHOT_ROOT" || {
  echo "[FAIL] SNAPSHOT_ROOT missing: $SNAPSHOT_ROOT" >&2
  exit 2
}
test -d "$BASE_MODEL" || { echo "[FAIL] BASE_MODEL missing: $BASE_MODEL" >&2; exit 2; }

for step in $(printf '%s:%s' "$STEP_LIST_A" "$STEP_LIST_B" | tr ':,' '  '); do
  case "$step" in
    ''|*[!0-9]*) echo "[FAIL] invalid checkpoint step: $step" >&2; exit 2 ;;
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

MANIFEST=$OUT_ROOT/submitted_2jobs_$(date +%F_%H%M%S).txt

submit_part() {
  part_id=$1
  step_list=$2
  job_name=$3
  part_lower=$(printf '%s' "$part_id" | tr '[:upper:]' '[:lower:]')
  pbs_log=$PROJ/logs/${job_name}.pbs.log

  variables="RTYPE=rt_HF,PART_ID=$part_id,STEP_LIST=$step_list"
  variables="$variables,SNAPSHOT_ROOT=$SNAPSHOT_ROOT,BASE_MODEL=$BASE_MODEL"
  variables="$variables,PROBE=$PROBE,OUT_ROOT=$OUT_ROOT,SEED=$SEED,HINT_LEVELS=none"
  variables="$variables,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,MAX_MODEL_LEN=$MAX_MODEL_LEN"
  variables="$variables,TEMPERATURE=$TEMPERATURE,TOP_P=$TOP_P,TOP_K=$TOP_K"
  variables="$variables,GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  variables="$variables,GEN_BATCH_SIZE=$GEN_BATCH_SIZE,SCORE_BATCH_SIZE=$SCORE_BATCH_SIZE"
  variables="$variables,ENTROPY_CHUNK_TOKENS=$ENTROPY_CHUNK_TOKENS"
  variables="$variables,SCORER_MAX_MEMORY_GIB_PER_GPU=$SCORER_MAX_MEMORY_GIB_PER_GPU"
  variables="$variables,OVERWRITE=$OVERWRITE"

  job_id=$(
    qsub \
      -N "$job_name" \
      -o "$pbs_log" \
      -v "$variables" \
      "$PROJ/jobs/q3_legal_rlvr_entropy_curve_part_v3.pbs"
  )

  printf 'part=%s job_name=%s job_id=%s steps=%s log=%s\n' \
    "$part_id" "$job_name" "$job_id" "$step_list" "$pbs_log" \
    | tee -a "$MANIFEST"
}

case "$PARTS" in
  both)
    submit_part A "$STEP_LIST_A" "$JOB_NAME_A"
    submit_part B "$STEP_LIST_B" "$JOB_NAME_B"
    ;;
  A)
    submit_part A "$STEP_LIST_A" "$JOB_NAME_A"
    ;;
  B)
    submit_part B "$STEP_LIST_B" "$JOB_NAME_B"
    ;;
esac

{
  echo "mode=$MODE"
  echo "out_root=$OUT_ROOT"
  echo "probe=$PROBE"
  echo "hint_levels=none"
  echo "walltime=16:00:00"
} | tee -a "$MANIFEST"

echo "[PASS] submitted requested part(s): $PARTS"
echo "[PASS] manifest: $MANIFEST"
echo "[NEXT] after both parts finish, run finalize_legal_rlvr_entropy_curve_2jobs_v3.sh"
