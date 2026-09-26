#!/bin/bash
# Run with the existing training conda environment activated and Internet access.
set -euo pipefail
MD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "${CONDA_PREFIX:-}" == */verl_qwen3_moe_megatron_py312_cu128 ]] || {
  echo 'Activate verl_qwen3_moe_megatron_py312_cu128 first.' >&2; exit 2;
}
MD_BASE_PYTHON="$CONDA_PREFIX/bin/python"
MD_VENV="$MD_ROOT/.venv-multidomain"
export PYTHONNOUSERSITE=1
"$MD_BASE_PYTHON" -m venv --system-site-packages "$MD_VENV"
"$MD_BASE_PYTHON" "$MD_ROOT/multidomain/dependencies.py" capture \
  --python "$MD_BASE_PYTHON" --output "$MD_VENV/base.before.json" \
  --constraints "$MD_VENV/base.constraints.txt"
"$MD_VENV/bin/python" -m pip install \
  -c "$MD_VENV/base.constraints.txt" \
  -r "$MD_ROOT/config/multidomain/requirements.txt"
"$MD_BASE_PYTHON" "$MD_ROOT/multidomain/dependencies.py" check \
  --base-python "$MD_BASE_PYTHON" --project-python "$MD_VENV/bin/python" \
  --before "$MD_VENV/base.before.json" --output "$MD_VENV/dependency_audit.json"
