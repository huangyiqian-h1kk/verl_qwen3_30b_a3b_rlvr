#!/usr/bin/env bash
set -euo pipefail

# Read-only compatibility check for the already-installed thin overlay.
# This script never invokes pip and never modifies the base Conda environment.

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}
MODEL_PATH=${MODEL_PATH:-$WORK/models/Qwen3-30B-A3B-Instruct-2507}
RG_PY=${RG_PY:-$PROJ/vendor/reasoning_gym_compat_py}
OLD_RG_PY=${OLD_RG_PY:-$PROJ/vendor/reasoning_gym_py}

export PYTHONNOUSERSITE=1

source "$PROJ/scripts/abci_verl_env.sh"
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

[[ -d "$RG_PY" ]] || {
  echo "[FAIL] thin overlay is absent: $RG_PY" >&2
  echo "Create it once with scripts/prepare_reasoning_gym_compat_v2.sh install." >&2
  exit 2
}
export PYTHONPATH="$RG_PY:$PROJ/compat${PYTHONPATH:+:$PYTHONPATH}"

RG_PY="$RG_PY" OLD_RG_PY="$OLD_RG_PY" python - <<'PY'
from importlib import import_module, metadata
from pathlib import Path
from packaging.version import Version
import json
import os
import sys

overlay = Path(os.environ["RG_PY"]).resolve()
old_overlay = Path(os.environ["OLD_RG_PY"]).resolve()

base = {}
for name in ("numpy", "packaging", "sympy", "matplotlib", "tabulate"):
    module = import_module(name)
    path = Path(module.__file__).resolve()
    if path == overlay or overlay in path.parents:
        raise RuntimeError(f"{name} is shadowed by the thin overlay: {path}")
    if path == old_overlay or old_overlay in path.parents:
        raise RuntimeError(f"{name} is shadowed by the obsolete overlay: {path}")
    base[name] = {"version": getattr(module, "__version__", metadata.version(name)), "path": str(path)}

if Version(base["numpy"]["version"]) >= Version("2.3"):
    raise RuntimeError(f"numpy must remain <2.3: {base['numpy']}")
if Version(base["packaging"]["version"]) >= Version("26"):
    raise RuntimeError(f"packaging must remain <26: {base['packaging']}")
if Version(base["sympy"]["version"]) < Version("1.13.1"):
    raise RuntimeError(f"sympy must be >=1.13.1: {base['sympy']}")

for name in ("pytz", "torch", "transformers", "vllm", "verl", "megatron.core", "pandas", "pyarrow"):
    import_module(name)

import reasoning_gym

rg_path = Path(reasoning_gym.__file__).resolve()
if not (rg_path == overlay or overlay in rg_path.parents):
    raise RuntimeError(f"reasoning_gym did not load from the thin overlay: {rg_path}")
if Version(metadata.version("reasoning-gym")).base_version != Version("0.1.25").base_version:
    raise RuntimeError(f"unexpected reasoning-gym version: {metadata.version('reasoning-gym')}")

print(json.dumps({
    "status": "PASS",
    "python": sys.executable,
    "reasoning_gym": {"version": metadata.version("reasoning-gym"), "path": str(rg_path)},
    "registered_tasks": len(reasoning_gym.factory.DATASETS),
    "base_modules": base,
}, indent=2))
PY

python "$PROJ/rewards/unicode_reasoning_gym_reward_v2.py"
python "$PROJ/scripts/build_reasoning_gym_sixcat_parquet.py" \
  --config "$PROJ/config/reasoning_gym_sixcat_profiles_v1.yaml" \
  --system-prompt-file "$PROJ/config/unicode_system_prompt.txt" \
  --tokenizer-path "$MODEL_PATH" \
  --output-dir "$PROJ/data/reasoning_gym_sixcat_v1" \
  --plan-only

echo "[PASS] thin overlay and six-category training plan are compatible"
