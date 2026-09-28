#!/bin/bash
# Exactly one MPI launcher process per allocated node.
set -euo pipefail
MD_RANK=${OMPI_COMM_WORLD_RANK:?}
MD_ROOT=${1:?}
MD_REQUEST=${2:?}
MD_JOB_DIR=${3:?}
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load gcc/13.2.0 cuda/12.8/12.8.1 cudnn/9.10/9.10.2 hpcx/2.20 nccl/2.29/2.29.7-1
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate /groups/gcg51557/experiments/0390_rlsd/envs/verl_qwen3_moe_megatron_py312_cu128
source "$MD_ROOT/scripts/multidomain/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$MD_ROOT"
# MPI only starts one shell per node; Ray/torch workers establish their own ranks.
while IFS='=' read -r MD_ENV_NAME _; do
  case "$MD_ENV_NAME" in OMPI_*|PMI_*|PMIX_*|MPI_*) unset "$MD_ENV_NAME" ;; esac
done < <(env)
MD_IP=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
MD_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ "$MD_GPUS" == 8 ]] || { echo "Expected 8 visible GPUs; found $MD_GPUS" >&2; exit 2; }
MD_STARTED=0
cleanup() {
  MD_STATUS=$?
  trap - EXIT INT TERM
  if [[ "$MD_RANK" == 0 ]]; then
    printf '%s\n' "$MD_STATUS" > "$MD_JOB_DIR/finished.tmp"
    mv "$MD_JOB_DIR/finished.tmp" "$MD_JOB_DIR/finished"
  fi
  if [[ "$MD_STARTED" == 1 ]]; then ray stop --force > "$MD_JOB_DIR/ray_stop_${MD_RANK}.log" 2>&1 || true; fi
  exit "$MD_STATUS"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$MD_RANK" == 0 ]]; then
  MD_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  ray start --head --node-ip-address="$MD_IP" --port="$MD_PORT" --num-gpus="$MD_GPUS" \
    --temp-dir="$RAY_TMPDIR" --include-dashboard=false --disable-usage-stats > "$MD_JOB_DIR/ray_start_0.log" 2>&1
  MD_STARTED=1
  printf '%s:%s\n' "$MD_IP" "$MD_PORT" > "$MD_JOB_DIR/head_address.tmp"
  mv "$MD_JOB_DIR/head_address.tmp" "$MD_JOB_DIR/head_address"
  export RAY_ADDRESS="$MD_IP:$MD_PORT"
  python -m multidomain.stage --request "$MD_REQUEST"
else
  MD_DEADLINE=$((SECONDS + 600))
  while [[ ! -f "$MD_JOB_DIR/head_address" ]]; do
    [[ ! -f "$MD_JOB_DIR/finished" && $SECONDS -lt $MD_DEADLINE ]] || exit 2
    sleep 2
  done
  export RAY_ADDRESS
  RAY_ADDRESS=$(cat "$MD_JOB_DIR/head_address")
  ray start --address="$RAY_ADDRESS" --node-ip-address="$MD_IP" --num-gpus="$MD_GPUS" \
    --disable-usage-stats > "$MD_JOB_DIR/ray_start_${MD_RANK}.log" 2>&1
  MD_STARTED=1
  while [[ ! -f "$MD_JOB_DIR/finished" ]]; do sleep 2; done
  exit "$(cat "$MD_JOB_DIR/finished")"
fi
