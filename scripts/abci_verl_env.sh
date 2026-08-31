#!/bin/bash
set -euo pipefail

module purge
module load gcc/13.2.0
module load cuda/12.8/12.8.1
module load cudnn/9.10/9.10.2
module load nccl/2.29/2.29.7-1

export WORK=/groups/gcg51557/experiments/0390_rlsd
export PROJ=$WORK/RLVR/verl_qwen3_30b_a3b_rlvr
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ENV_PREFIX=$WORK/envs/verl_qwen3_moe_megatron_py312_cu128
export VERL_SRC=$PROJ/src/verl

export HF_HOME=$PROJ/hf_home
export HF_HUB_CACHE=$PROJ/hf_home/hub
export HF_DATASETS_CACHE=$PROJ/hf_home/datasets
export TRANSFORMERS_CACHE=$PROJ/hf_home/transformers
export XDG_CACHE_HOME=$PROJ/.cache

export TRITON_CACHE_DIR=$PROJ/.cache/triton
export TORCH_EXTENSIONS_DIR=$PROJ/.cache/torch_extensions
export WANDB_DIR=$PROJ/wandb

export TMPDIR=${TMPDIR:-$PROJ/tmp/$USER}
export RAY_TMPDIR=$TMPDIR/ray

mkdir -p \
  "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_DIR" \
  "$TMPDIR" "$RAY_TMPDIR" "$PROJ/logs"

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export VLLM_USE_V1=1

export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=${MAX_JOBS:-8}
