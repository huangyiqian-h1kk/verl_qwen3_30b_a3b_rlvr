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
# Launch Ray itself with the overlay interpreter. An inherited `ray` console
# script can have a base-conda shebang even after venv activation.
MD_PYTHON="$MD_ROOT/.venv-multidomain/bin/python"
[[ -x "$MD_PYTHON" ]] || { echo '[FAIL] Missing project Python overlay' >&2; exit 2; }
"$MD_PYTHON" -c 'import sys; print("[0390] Ray launch Python:", sys.executable, "prefix:", sys.prefix, flush=True)'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$MD_ROOT"
"$MD_PYTHON" "$MD_ROOT/scripts/multidomain/check_shared_memory.py" --request "$MD_REQUEST" --rank "$MD_RANK"
# MPI only starts one shell per node; Ray/torch workers establish their own ranks.
while IFS='=' read -r MD_ENV_NAME _; do
  case "$MD_ENV_NAME" in OMPI_*|PMI_*|PMIX_*|MPI_*) unset "$MD_ENV_NAME" ;; esac
done < <(env)
# BEGIN 0390 FULL-NODE CUDA INDEX COMPATIBILITY
# vLLM 0.11.0 parses CUDA_VISIBLE_DEVICES entries as integers. ABCI may
# supply UUIDs. Restore the established rt_HF recipe only after confirming
# that this process already sees all eight physical GPUs on this node.
printf '[0390] inherited CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
[[ "${CUDA_VISIBLE_DEVICES:-}" != *MIG-* ]] || { echo '[FAIL] This launcher requires full GPUs, not MIG devices.' >&2; exit 2; }
MD_GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, - | tr -d '[:space:]')
[[ "$MD_GPU_INDICES" == 0,1,2,3,4,5,6,7 ]] || { echo "[FAIL] Expected exactly eight physical GPUs; found indices: $MD_GPU_INDICES" >&2; exit 2; }
MD_INHERITED_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ "$MD_INHERITED_GPUS" == 8 ]] || { echo "[FAIL] Inherited allocation exposes $MD_INHERITED_GPUS GPUs, expected 8; CUDA_VISIBLE_DEVICES was not changed." >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
printf '[0390] CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
# END 0390 FULL-NODE CUDA INDEX COMPATIBILITY
MD_IP=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
MD_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ "$MD_GPUS" == 8 ]] || { echo "Expected 8 visible GPUs; found $MD_GPUS" >&2; exit 2; }
# Exercise the precise vLLM capability query that previously failed, before Ray.
python - <<'PY'
from vllm.platforms import current_platform
capabilities = [current_platform.get_device_capability(i) for i in range(8)]
if any(value is None for value in capabilities):
    raise RuntimeError(f'Could not query every GPU capability: {capabilities}')
print('[0390] vLLM GPU capability check PASS:', capabilities, flush=True)
PY
# BEGIN 0390 SMOKE DIAGNOSTICS
# Export before Ray starts so actor and EngineCore children inherit these.
MD_DIAGNOSTICS=$("$MD_PYTHON" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["stage"] in ("smoke", "accept-four")))' "$MD_REQUEST")
if [[ "$MD_DIAGNOSTICS" == 1 ]]; then
  export VLLM_LOGGING_LEVEL=DEBUG PYTHONFAULTHANDLER=1
fi
MD_DIAG_DIR="$MD_JOB_DIR/diagnostics/$(hostname -s)_rank${MD_RANK}"
MD_DIAG_PID=""
MD_DIAG_SCRIPT="$MD_ROOT/scripts/multidomain/collect_node_diagnostics.py"
MD_DIAG_ARGS=(--output "$MD_DIAG_DIR" --local-root "$PBS_LOCALDIR" --ray-root "$RAY_TMPDIR" --rank "$MD_RANK")
if [[ "$MD_DIAGNOSTICS" == 1 ]]; then mkdir -p "$MD_DIAG_DIR"; fi
# END 0390 SMOKE DIAGNOSTICS
MD_STARTED=0
cleanup() {
  MD_STATUS=$?
  trap - EXIT INT TERM
  # Diagnostic errors must never change the computation's exit status.
  set +e
  if [[ -n "$MD_DIAG_PID" ]]; then
    kill -TERM "$MD_DIAG_PID" 2>/dev/null
    wait "$MD_DIAG_PID" 2>/dev/null
  fi
  if [[ "$MD_DIAGNOSTICS" == 1 ]]; then
    "$MD_PYTHON" "$MD_DIAG_SCRIPT" "${MD_DIAG_ARGS[@]}" --phase before_ray_stop >> "$MD_DIAG_DIR/collector.log" 2>&1
  fi
  if [[ "$MD_RANK" == 0 ]]; then
    printf '%s\n' "$MD_STATUS" > "$MD_JOB_DIR/finished.tmp"
    mv "$MD_JOB_DIR/finished.tmp" "$MD_JOB_DIR/finished"
  fi
  if [[ "$MD_STARTED" == 1 ]]; then "$MD_PYTHON" -m ray.scripts.scripts stop --force > "$MD_JOB_DIR/ray_stop_${MD_RANK}.log" 2>&1 || true; fi
  if [[ "$MD_DIAGNOSTICS" == 1 ]]; then
    "$MD_PYTHON" "$MD_DIAG_SCRIPT" "${MD_DIAG_ARGS[@]}" --phase after_ray_stop >> "$MD_DIAG_DIR/collector.log" 2>&1
  fi
  if [[ "$MD_DIAGNOSTICS" == 1 ]]; then printf '[0390] Node diagnostics saved: %s\n' "$MD_DIAG_DIR"; fi
  exit "$MD_STATUS"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$MD_DIAGNOSTICS" == 1 ]]; then
  "$MD_PYTHON" "$MD_DIAG_SCRIPT" "${MD_DIAG_ARGS[@]}" --watch --parent-pid "$$" >> "$MD_DIAG_DIR/collector.log" 2>&1 &
  MD_DIAG_PID=$!
fi
if [[ "$MD_RANK" == 0 ]]; then
  MD_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  "$MD_PYTHON" -m ray.scripts.scripts start --head --node-ip-address="$MD_IP" --port="$MD_PORT" --num-gpus="$MD_GPUS" \
    --temp-dir="$RAY_TMPDIR" --include-dashboard=false --disable-usage-stats > "$MD_JOB_DIR/ray_start_0.log" 2>&1
  MD_STARTED=1
  printf '%s:%s\n' "$MD_IP" "$MD_PORT" > "$MD_JOB_DIR/head_address.tmp"
  mv "$MD_JOB_DIR/head_address.tmp" "$MD_JOB_DIR/head_address"
  export RAY_ADDRESS="$MD_IP:$MD_PORT"
  "$MD_PYTHON" "$MD_ROOT/scripts/multidomain/check_ray_environment.py" --request "$MD_REQUEST"
  "$MD_PYTHON" -m multidomain.stage --request "$MD_REQUEST"
else
  MD_DEADLINE=$((SECONDS + 600))
  while [[ ! -f "$MD_JOB_DIR/head_address" ]]; do
    [[ ! -f "$MD_JOB_DIR/finished" && $SECONDS -lt $MD_DEADLINE ]] || exit 2
    sleep 2
  done
  export RAY_ADDRESS
  RAY_ADDRESS=$(cat "$MD_JOB_DIR/head_address")
  "$MD_PYTHON" -m ray.scripts.scripts start --address="$RAY_ADDRESS" --node-ip-address="$MD_IP" --num-gpus="$MD_GPUS" \
    --disable-usage-stats > "$MD_JOB_DIR/ray_start_${MD_RANK}.log" 2>&1
  MD_STARTED=1
  while [[ ! -f "$MD_JOB_DIR/finished" ]]; do sleep 2; done
  exit "$(cat "$MD_JOB_DIR/finished")"
fi
