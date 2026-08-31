#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}
RG_VERSION=${RG_VERSION:-0.1.25}
WHEELHOUSE=${WHEELHOUSE:-$PROJ/vendor/reasoning_gym_wheels}
TARGET=${TARGET:-$PROJ/vendor/reasoning_gym_py}
ACTION=${1:-}

case "$ACTION" in
  download)
    # Run once on an internet-connected host with the same Python major/minor.
    mkdir -p "$WHEELHOUSE"
    CC=${CC:-gcc} CXX=${CXX:-g++} PIP_CACHE_DIR=/tmp/0390_reasoning_gym_pip_cache \
      python -m pip wheel --wheel-dir "$WHEELHOUSE" "reasoning-gym==$RG_VERSION"
    ;;
  install)
    # Safe project-local install; the shared conda environment is not mutated.
    source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
    conda activate "$ENV_PREFIX"
    mkdir -p "$TARGET"
    python -m pip install --no-index --find-links "$WHEELHOUSE" \
      --target "$TARGET" "reasoning-gym==$RG_VERSION"
    PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib.metadata
import reasoning_gym
print("[PASS] reasoning-gym", importlib.metadata.version("reasoning-gym"))
print("[PASS] registered tasks", len(reasoning_gym.factory.DATASETS))
PY
    ;;
  *)
    echo "usage: $0 download|install" >&2
    exit 2
    ;;
esac
