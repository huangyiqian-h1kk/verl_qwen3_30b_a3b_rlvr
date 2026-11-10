#!/bin/bash
# One process per exclusive eight-GPU node; never run this on a login node.
set -euo pipefail
mopd_request=${1:?}
mopd_folder=$(dirname "$mopd_request")
mopd_rank=${OMPI_COMM_WORLD_RANK:?MPI allocation required}
mopd_python=$(cat "$mopd_folder/python.txt")
mopd_ray_started=0
cleanup() {
  mopd_status=$?
  trap - EXIT INT TERM
  if [[ "$mopd_rank" == 0 ]]; then
    printf '%s\n' "$mopd_status" > "$mopd_folder/finished.tmp"
    mv "$mopd_folder/finished.tmp" "$mopd_folder/finished"
  fi
  if [[ "$mopd_ray_started" == 1 ]]; then
    "$mopd_python" -m ray.scripts.scripts stop --force > "$mopd_folder/ray_stop_$mopd_rank.log" 2>&1 || true
  fi
  exit "$mopd_status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export D4_MOPD_ATTEMPT
D4_MOPD_ATTEMPT=$(basename "$mopd_folder")
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Both packages come from this experiment's immutable snapshot/pinned upstream.
export PYTHONPATH
PYTHONPATH=$("$mopd_python" -I "$mopd_folder/entry.py" paths --request "$mopd_request")
unset PYTHONHOME
export RAY_TMPDIR="${PBS_LOCALDIR:?}/mopd_${D4_MOPD_ATTEMPT}"
mkdir -p "$RAY_TMPDIR"
# Drop MPI rank variables before spawning Ray/torch distributed workers.
while IFS='=' read -r mopd_env _; do
  case "$mopd_env" in OMPI_*|PMI_*|PMIX_*|MPI_*) unset "$mopd_env" ;; esac
done < <(env)
"$mopd_python" -I "$mopd_folder/entry.py" gpu-node --request "$mopd_request" --rank "$mopd_rank"
mopd_ip=$("$mopd_python" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
if [[ "$mopd_rank" == 0 ]]; then
  mopd_port=$("$mopd_python" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  "$mopd_python" -m ray.scripts.scripts start --head --node-ip-address="$mopd_ip" --port="$mopd_port" \
    --num-gpus=8 --temp-dir="$RAY_TMPDIR" --include-dashboard=false --disable-usage-stats > "$mopd_folder/ray_start_0.log" 2>&1
  mopd_ray_started=1
  printf '%s:%s\n' "$mopd_ip" "$mopd_port" > "$mopd_folder/head_address.tmp"
  mv "$mopd_folder/head_address.tmp" "$mopd_folder/head_address"
  export RAY_ADDRESS="$mopd_ip:$mopd_port"
  "$mopd_python" -I "$mopd_folder/entry.py" execute --request "$mopd_request"
else
  mopd_deadline=$((SECONDS + 900))
  until [[ -f "$mopd_folder/head_address" ]]; do
    [[ ! -f "$mopd_folder/finished" && $SECONDS -lt $mopd_deadline ]] || exit 2
    sleep 2
  done
  export RAY_ADDRESS
  RAY_ADDRESS=$(cat "$mopd_folder/head_address")
  "$mopd_python" -m ray.scripts.scripts start --address="$RAY_ADDRESS" --node-ip-address="$mopd_ip" \
    --num-gpus=8 --disable-usage-stats > "$mopd_folder/ray_start_$mopd_rank.log" 2>&1
  mopd_ray_started=1
  until [[ -f "$mopd_folder/finished" ]]; do sleep 2; done
  exit "$(cat "$mopd_folder/finished")"
fi
