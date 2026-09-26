#!/bin/bash
# Source this after conda activation on login or compute nodes.
set -euo pipefail
export MD_ROOT
MD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export VERL_SRC=${VERL_SRC:-/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr/src/verl}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MD_ROOT:$MD_ROOT/compat:$VERL_SRC"
if [[ -f "$MD_ROOT/.venv-multidomain/bin/activate" ]]; then
  source "$MD_ROOT/.venv-multidomain/bin/activate"
fi
if [[ -n "${PBS_JOBID:-}" ]]; then
  export TMPDIR=${PBS_LOCALDIR:?PBS_LOCALDIR is required}/multidomain
  export RAY_TMPDIR=$TMPDIR/ray
  export TRITON_CACHE_DIR=$TMPDIR/triton
  export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_extensions
  export XDG_CACHE_HOME=$TMPDIR/cache
  mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
fi
export HF_HOME=$MD_ROOT/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WANDB_MODE=offline WANDB_DIR=$MD_ROOT/wandb
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_USE_V1=1
export NCCL_IB_DISABLE=0 NCCL_DEBUG=WARN RAY_DEDUP_LOGS=0
mkdir -p "$HF_HOME" "$WANDB_DIR"
