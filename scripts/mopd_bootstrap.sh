#!/bin/bash
# Separate upstream installation: do not upgrade the deployed teacher environment.

mopd_resolve_uv() {
  local mopd_upstream=$1 mopd_python=${2:-} mopd_audit=${3:-} mopd_user_home=${4:-$HOME}
  local mopd_candidate
  mopd_uv=''
  if [[ -n ${MOPD_UV:-} ]]; then
    [[ $MOPD_UV == /* && -x $MOPD_UV && ! -d $MOPD_UV ]] || {
      printf 'MOPD_UV must be an executable absolute path: %s\n' "$MOPD_UV" >&2; return 1;
    }
    mopd_uv=$MOPD_UV
  else
    for mopd_candidate in "$(type -P uv || true)" \
      "${UV_INSTALL_DIR:-$mopd_user_home/.local/bin}/uv" \
      "${UV_UNMANAGED_INSTALL:-$mopd_user_home/.local/bin}/uv" \
      "$mopd_user_home/.local/bin/uv" "$mopd_user_home/.cargo/bin/uv" \
      "$mopd_upstream/.venv/bin/uv"; do
      if [[ -f $mopd_candidate && -x $mopd_candidate ]]; then
        mopd_uv=$mopd_candidate
        break
      fi
    done
  fi
  if [[ -z $mopd_uv && -n $mopd_python && -n $mopd_audit ]]; then
    # The uv-managed training environment may have neither uv nor pip. Keep
    # this CLI's bootstrap outside it and do not modify shell startup files.
    printf 'uv not found; installing uv==0.13.6 in %s/uv-tool\n' "$mopd_audit"
    "$mopd_python" -I -m venv "$mopd_audit/uv-tool" || return 1
    "$mopd_audit/uv-tool/bin/python" -I -m pip --isolated install \
      --disable-pip-version-check --no-deps uv==0.13.6 || return 1
    mopd_uv="$mopd_audit/uv-tool/bin/uv"
  fi
  [[ -n $mopd_uv ]] || {
    printf 'uv was not found on PATH or in user install directories. Set MOPD_UV to its absolute path.\n' >&2
    return 1
  }
  mopd_uv=$(readlink -f "$mopd_uv")
  printf 'uv executable: %s\n' "$mopd_uv"
  "$mopd_uv" --version || return 1
  "$mopd_uv" build --help >/dev/null || return 1
  if [[ -n $mopd_audit ]]; then
    "$mopd_python" -I - "$mopd_audit" "$mopd_uv" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
path = Path(sys.argv[2])
record = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              version=subprocess.check_output([str(path), '--version'], text=True).strip())
Path(sys.argv[1], 'uv.json').write_text(json.dumps(record, indent=2)+'\n')
PY
  fi
}

mopd_te_build_tools() {
  local mopd_python=$1 mopd_audit=$2 mopd_missing mopd_tool
  # Preserve installed versions. For absent tools, use upstream lock versions;
  # wheel is the extra build requirement absent from that runtime lock.
  mopd_missing=$("$mopd_python" -I - "$mopd_audit" <<'PY'
import importlib.metadata as md, json, sys
from pathlib import Path
pins = {'setuptools': '80.10.2', 'wheel': '0.48.0', 'pybind11': '3.0.4', 'ninja': '1.13.0'}
installed, missing = {}, []
for name, version in pins.items():
    try:
        installed[name] = md.version(name)
    except md.PackageNotFoundError:
        missing.append(f'{name}=={version}')
Path(sys.argv[1], 'build_tools_plan.json').write_text(json.dumps(dict(installed=installed, install=missing), indent=2)+'\n')
print('\n'.join(missing))
PY
  ) || return 1
  while IFS= read -r mopd_tool; do
    [[ -n $mopd_tool ]] || continue
    printf 'Installing missing build tool: %s\n' "$mopd_tool"
    "$mopd_uv" pip install --python "$mopd_python" --no-deps "$mopd_tool" || return 1
  done <<< "$mopd_missing"
  "$mopd_python" -I - <<'PY'
from setuptools import build_meta
from wheel.bdist_wheel import bdist_wheel
import pybind11
import ninja
PY
}

mopd_te_build_preflight() {
  "$1" -I - "$2" <<'PY'
import importlib.metadata as md, json, os, re, shutil, subprocess, sys, sysconfig
from pathlib import Path
errors, commands = [], {}
for name in ('git', 'g++', 'cmake', 'ninja'):
    path = shutil.which(name)
    try:
        if path is None:
            raise FileNotFoundError(name)
        output = subprocess.check_output([path, '--version'], text=True, stderr=subprocess.STDOUT)
        commands[name] = dict(path=path, version=output.splitlines()[0])
        if name == 'cmake':
            version = re.search(r'version\s+(\d+)\.(\d+)', output)
            if not version or tuple(map(int, version.groups())) < (3, 21):
                errors.append('CMake >=3.21 is required')
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f'{name}: {exc}')
paths = {
    'Python.h': Path(sysconfig.get_path('include')) / 'Python.h',
    'cuda_runtime.h': Path(os.environ['CUDA_HOME']) / 'include/cuda_runtime.h',
    'cudnn.h': Path(os.environ['CUDNN_PATH']) / 'include/cudnn.h',
    'libcudnn.so.9': Path(os.environ['CUDNN_PATH']) / 'lib/libcudnn.so.9',
    'nccl.h': Path(os.environ['NCCL_HOME']) / 'include/nccl.h',
    'libnccl.so.2': Path(os.environ['NCCL_HOME']) / 'lib/libnccl.so.2',
    'cudnn_frontend.h': Path(md.distribution('nvidia-cudnn-frontend').locate_file('include')) / 'cudnn_frontend.h',
}
errors.extend(f'Missing build input: {path}' for path in paths.values() if not path.is_file())
record = dict(status='FAIL' if errors else 'PASS', commands=commands,
              files={k: str(v) for k, v in paths.items()}, errors=errors,
              packages={k: md.version(k) for k in ('setuptools', 'wheel', 'pybind11', 'ninja')})
Path(sys.argv[1], 'build_prerequisites.json').write_text(json.dumps(record, indent=2)+'\n')
if errors:
    raise SystemExit('TE build prerequisites failed:\n'+'\n'.join(errors))
print('TE build prerequisites: PASS (build tools, Python/CUDA/cuDNN/NCCL headers and libraries)', flush=True)
PY
}

mopd_te_toolchain() {
  local mopd_python=$1 mopd_audit=$2 mopd_candidate mopd_cuda_module='' mopd_nvcc
  if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
  module -t avail cuda > "$mopd_audit/cuda_modules.txt" 2>&1 || true
  : > "$mopd_audit/module_probe.log"
  # ABCI uses cuda/<major.minor>/<toolkit version>. Some installations also
  # expose the flat name. Probe exact names; never select an unpinned default.
  for mopd_candidate in cuda/13.0/13.0.1 cuda/13.0.1; do
    if module show "$mopd_candidate" >> "$mopd_audit/module_probe.log" 2>&1; then
      mopd_cuda_module=$mopd_candidate
      break
    fi
  done
  if [[ -z $mopd_cuda_module ]]; then
    printf 'CUDA 13.0.1 module not found. Checked cuda/13.0/13.0.1 and cuda/13.0.1. Available CUDA modules:\n' >&2
    cat "$mopd_audit/cuda_modules.txt" >&2
    return 1
  fi
  printf 'TE build modules: gcc/13.2.0 %s cmake/4.1.1\n' "$mopd_cuda_module"
  module load gcc/13.2.0 "$mopd_cuda_module" cmake/4.1.1 || return 1
  module list > "$mopd_audit/loaded_modules.txt" 2>&1
  mopd_nvcc=$(command -v nvcc) || { printf 'Loaded CUDA module has no nvcc on PATH\n' >&2; return 1; }
  export CUDACXX CUDA_HOME CUDA_PATH
  CUDACXX=$(readlink -f "$mopd_nvcc")
  CUDA_HOME=$(dirname "$(dirname "$CUDACXX")")
  CUDA_PATH=$CUDA_HOME
  "$mopd_python" -I - "$mopd_audit" "$mopd_cuda_module" <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path
nvcc = subprocess.check_output([os.environ['CUDACXX'], '--version'], text=True)
print(nvcc, flush=True)
release = re.search(r'\brelease\s+(\d+\.\d+),', nvcc)
record = dict(cuda_module=sys.argv[2], cuda_home=os.environ['CUDA_HOME'],
              nvcc_path=os.environ['CUDACXX'], nvcc=nvcc,
              status='PASS' if release and release[1] == '13.0' else 'FAIL')
Path(sys.argv[1], 'toolchain.json').write_text(json.dumps(record, indent=2)+'\n')
if record['status'] != 'PASS':
    raise SystemExit('TE build requires nvcc release 13.0 to match pinned PyTorch CUDA 13.0; refusing this compiler')
PY
}

# Allow the toolchain check to be exercised without cloning/building/installing.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then return; fi
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
  mopd_resolve_uv "$mopd_upstream" "$mopd_python" "$mopd_audit"
  mopd_te_toolchain "$mopd_python" "$mopd_audit"
  export PATH="$CUDA_HOME/bin:$mopd_upstream/.venv/bin:$PATH"
  export CUDNN_PATH CUDNN_HOME NCCL_HOME
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
  mopd_te_build_tools "$mopd_python" "$mopd_audit"
  mopd_te_build_preflight "$mopd_python" "$mopd_audit"
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
  "$mopd_uv" build --python "$mopd_python" --wheel --no-build-isolation --out-dir "$mopd_audit/wheels" .
  mopd_wheels=("$mopd_audit"/wheels/transformer_engine-2.20.2-*.whl)
  [[ ${#mopd_wheels[@]} == 1 && -f ${mopd_wheels[0]} ]]
  "$mopd_python" -I - "$mopd_audit" "${mopd_wheels[0]}" "$mopd_te_revision" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
audit, wheel = map(Path, sys.argv[1:3])
record = dict(source_revision=sys.argv[3], wheel=str(wheel), sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
              host=platform.node(), glibc=os.confstr('CS_GNU_LIBC_VERSION'), python=sys.version,
              toolchain=json.loads((audit/'toolchain.json').read_text()),
              uv=json.loads((audit/'uv.json').read_text()),
              build_prerequisites=json.loads((audit/'build_prerequisites.json').read_text()),
              compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
              nvcc=subprocess.check_output(['nvcc','--version'],text=True),
              flags={k:v for k,v in os.environ.items() if k.startswith('NVTE_') or k in ('MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL')})
(audit/'build_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
PY
  "$mopd_uv" pip install --python "$mopd_python" --no-deps --reinstall "${mopd_wheels[0]}"
  cp "$mopd_audit/build_receipt.json" "$mopd_upstream/.venv/mopd_te_source_build.json"
  exit 0
fi
[[ $# == 1 ]]
if [[ -f "$mopd_upstream/.venv/mopd_te_source_build.json" ]]; then
  printf 'Source-built TE is installed. Refusing uv sync, which would restore the incompatible wheelhouse binary.\n' >&2
  exit 1
fi
mopd_resolve_uv "$mopd_upstream"
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
"$mopd_uv" sync --python 3.12 --frozen --all-packages --extra vllm --extra megatron
printf 'New interpreter: %s/.venv/bin/python\n' "$PWD"
