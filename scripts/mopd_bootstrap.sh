#!/bin/bash
# Separate upstream installation: do not upgrade the deployed teacher environment.
set -euo pipefail
mopd_upstream=${1:?Usage: bash scripts/mopd_bootstrap.sh NEW_VERL_DIRECTORY}
mopd_revision=482f86fd09959912c99c0acf2b90329f9e061da7
if [[ ${2:-} == --rebuild-te ]]; then
  # Invoked by repair-te while the experiment's shared control locks are held.
  # Build a complete replacement wheel first; do not overwrite individual .so
  # files or update the system glibc. No dependency resolver upgrades are allowed.
  mopd_audit=${3:?repair audit directory required}
  exec > >(tee -a "$mopd_audit/build.log") 2>&1
  mopd_python="$mopd_upstream/.venv/bin/python"
  [[ $(git -C "$mopd_upstream" rev-parse HEAD) == "$mopd_revision" ]]
  [[ -z $(git -C "$mopd_upstream" status --porcelain) ]]
  if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
  module load gcc/13.2.0 cuda/13.0.1 cmake/4.1.1
  export PATH="$mopd_upstream/.venv/bin:$PATH"
  export CUDA_HOME CUDA_PATH CUDNN_PATH CUDNN_HOME NCCL_HOME
  CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
  CUDA_PATH=$CUDA_HOME
  nvcc --version
  "$mopd_python" -I - <<'PY'
import os, torch, importlib.metadata as md
assert tuple(map(int, os.confstr('CS_GNU_LIBC_VERSION').split()[1].split('.'))) <= (2, 34), 'Build on an ABCI glibc <=2.34 host'
assert torch.__version__.split('+')[0] == '2.13.0' and torch.version.cuda == '13.0'
assert md.version('transformer-engine').split('+')[0] == '2.20.2'
assert md.version('nvidia-cudnn-frontend') == '1.31.0'
PY
  mopd_header_root='import importlib.metadata as m,sys; from pathlib import Path; d=m.distribution(sys.argv[1]); p=[Path(d.locate_file(f)).resolve().parent.parent for f in d.files if Path(f).name==sys.argv[2]]; assert len(p)==1, p; print(p[0])'
  CUDNN_PATH=$("$mopd_python" -I -c "$mopd_header_root" nvidia-cudnn-cu13 cudnn.h)
  CUDNN_HOME=$CUDNN_PATH
  NCCL_HOME=$("$mopd_python" -I -c "$mopd_header_root" nvidia-nccl-cu13 nccl.h)
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # Only build tools absent from the lock may be added. Existing versions stay.
  for mopd_tool in wheel==0.48.0 pybind11==3.1.0; do
    if ! "$mopd_python" -I -c 'import importlib.metadata as m,sys; m.version(sys.argv[1])' "${mopd_tool%%==*}"; then
      uv pip install --python "$mopd_python" --no-deps "$mopd_tool"
    fi
  done
  mopd_te_revision=6ea2a74a9e98c99e6d7b164a33775cc457520027
  mopd_source="$mopd_audit/TransformerEngine"
  git clone --depth 1 --branch v2.20.2 https://github.com/NVIDIA/TransformerEngine.git "$mopd_source"
  [[ $(git -C "$mopd_source" rev-parse HEAD) == "$mopd_te_revision" ]]
  git -C "$mopd_source" submodule update --init --recursive --depth 1
  # Match the verl cu130 wheelhouse build: PyTorch only, NCCL EP disabled.
  # H200 uses SM90. Bound compiler parallelism on shared development hosts.
  export NVTE_FRAMEWORK=pytorch NVTE_NO_LOCAL_VERSION=1 NVTE_RELEASE_BUILD=0
  export NVTE_BUILD_METAPACKAGE=0 NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS=90
  export NVTE_WITH_CUTEDSL=0 NVTE_UB_WITH_MPI=0 NVTE_ENABLE_NVSHMEM=0
  export NVTE_WITH_CUBLASMP=0 NVTE_WITH_CUSOLVERMP=0 NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=0
  export MAX_JOBS=1 NVTE_BUILD_MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 NVTE_BUILD_THREADS_PER_JOB=1
  export NVTE_CMAKE_EXTRA_ARGS="-DNCCL_INCLUDE_DIR=$NCCL_HOME/include"
  export CC=gcc CXX=g++
  cd "$mopd_source"
  uv build --python "$mopd_python" --wheel --no-build-isolation --out-dir "$mopd_audit/wheels" .
  mopd_wheels=("$mopd_audit"/wheels/transformer_engine-2.20.2-*.whl)
  [[ ${#mopd_wheels[@]} == 1 && -f ${mopd_wheels[0]} ]]
  "$mopd_python" -I - "$mopd_audit" "${mopd_wheels[0]}" "$mopd_te_revision" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
audit, wheel = map(Path, sys.argv[1:3])
record = dict(source_revision=sys.argv[3], wheel=str(wheel), sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
              host=platform.node(), glibc=os.confstr('CS_GNU_LIBC_VERSION'), python=sys.version,
              compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
              nvcc=subprocess.check_output(['nvcc','--version'],text=True),
              flags={k:v for k,v in os.environ.items() if k.startswith('NVTE_') or k in ('MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL')})
(audit/'build_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
PY
  uv pip install --python "$mopd_python" --no-deps --reinstall "${mopd_wheels[0]}"
  cp "$mopd_audit/build_receipt.json" "$mopd_upstream/.venv/mopd_te_source_build.json"
  exit 0
fi
[[ $# == 1 ]]
if [[ -f "$mopd_upstream/.venv/mopd_te_source_build.json" ]]; then
  printf 'Source-built TE is installed. Refusing uv sync, which would restore the incompatible wheelhouse binary.\n' >&2
  exit 1
fi
if [[ ! -e "$mopd_upstream" ]]; then
  git clone https://github.com/verl-project/verl.git "$mopd_upstream"
fi
[[ -d "$mopd_upstream/.git" ]]
[[ -z $(git -C "$mopd_upstream" status --porcelain) ]]
git -C "$mopd_upstream" fetch origin "$mopd_revision"
git -C "$mopd_upstream" checkout --detach "$mopd_revision"
cd "$mopd_upstream"
# CUDA 13 build in this exact lock. Driver/extension compatibility is checked in
# CPU native-ABI and gpu-node preflights; sync alone is not compatibility proof.
uv sync --python 3.12 --frozen --all-packages --extra vllm --extra megatron
printf 'New interpreter: %s/.venv/bin/python\n' "$PWD"
