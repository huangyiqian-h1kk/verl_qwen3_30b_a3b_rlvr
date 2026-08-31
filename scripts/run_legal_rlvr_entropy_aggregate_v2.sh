#!/bin/bash
# CPU-only aggregation invoked inside the serial GPU job after every step passes.

set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
CURVE_DIR=${CURVE_DIR:-$OUT_ROOT/curves}
KEYWORDS=${KEYWORDS:-$PROJ/config/legal_entropy_keywords_zh.txt}

mkdir -p "$CURVE_DIR"

if [ -f "$PROJ/scripts/abci_verl_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$PROJ/scripts/abci_verl_env.sh"
fi
# shellcheck disable=SC1091
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export MPLBACKEND=Agg
export MPLCONFIGDIR=$CURVE_DIR/.matplotlib
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

python "$PROJ/scripts/aggregate_legal_rlvr_entropy_curve_v1.py" \
  --input-root "$OUT_ROOT" \
  --out-dir "$CURVE_DIR"

python "$PROJ/scripts/aggregate_legal_keyword_entropy_curve_v1.py" \
  --input-root "$OUT_ROOT" \
  --keywords "$KEYWORDS" \
  --out-dir "$CURVE_DIR/keywords"

touch "$CURVE_DIR/.complete"
echo "[PASS] aggregate complete: $CURVE_DIR"
