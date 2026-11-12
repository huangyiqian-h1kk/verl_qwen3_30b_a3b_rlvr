# Sourced on every compute node before importing vLLM or starting Ray.
mopd_cuda_tools() {
  local mopd_support mopd_candidate mopd_cuda_module='' mopd_nvcc
  mopd_support=$(dirname "${BASH_SOURCE[0]}")
  "$mopd_python" -I "$mopd_support/cuda_runtime.py" begin "$mopd_folder" "$mopd_rank" || return 1
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh || return 1
  fi
  for mopd_candidate in cuda/13.0/13.0.1 cuda/13.0.1; do
    if module show "$mopd_candidate" >> "$mopd_folder/cuda_modules_$mopd_rank.log" 2>&1; then
      mopd_cuda_module=$mopd_candidate
      break
    fi
  done
  [[ -n "$mopd_cuda_module" ]] || { printf 'CUDA 13.0.1 module is unavailable\n' >&2; return 1; }
  module load gcc/13.2.0 "$mopd_cuda_module" || return 1
  module list >> "$mopd_folder/cuda_modules_$mopd_rank.log" 2>&1 || return 1
  mopd_nvcc=$(command -v nvcc) || { printf 'CUDA module did not expose nvcc\n' >&2; return 1; }
  export CUDACXX CUDA_HOME CUDA_PATH
  CUDACXX=$(readlink -f "$mopd_nvcc") || return 1
  CUDA_HOME=$(dirname "$(dirname "$CUDACXX")")
  CUDA_PATH=$CUDA_HOME
  # The batch shell does not activate the venv: expose its pinned ninja explicitly.
  export PATH="$(dirname "$mopd_python"):$CUDA_HOME/bin:$PATH"
  printf '[0390] CUDA runtime tools rank=%s module=%s home=%s\n' "$mopd_rank" "$mopd_cuda_module" "$CUDA_HOME"
  "$mopd_python" -I "$mopd_support/cuda_runtime.py" probe "$mopd_folder" "$mopd_rank"
}
if ! mopd_cuda_tools; then
  "$mopd_python" -I "$(dirname "${BASH_SOURCE[0]}")/cuda_runtime.py" fail "$mopd_folder" "$mopd_rank"
  exit 1
fi
