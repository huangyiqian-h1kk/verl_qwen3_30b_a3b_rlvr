#!/bin/bash
# Run once with the existing training conda environment activated and Internet access.
set -euo pipefail
MD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ "${CONDA_PREFIX:-}" == */verl_qwen3_moe_megatron_py312_cu128 ]] || {
  echo 'Activate verl_qwen3_moe_megatron_py312_cu128 first.' >&2; exit 2;
}
python -m venv --system-site-packages "$MD_ROOT/.venv-multidomain"
python -m pip freeze > "$MD_ROOT/.venv-multidomain/base.constraints.txt"
"$MD_ROOT/.venv-multidomain/bin/python" -m pip install \
  -c "$MD_ROOT/.venv-multidomain/base.constraints.txt" \
  -r "$MD_ROOT/config/multidomain/requirements.txt"
"$MD_ROOT/.venv-multidomain/bin/python" -m pip check
