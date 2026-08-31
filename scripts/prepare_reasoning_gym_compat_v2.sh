#!/usr/bin/env bash
set -euo pipefail

# Build a thin Reasoning Gym overlay that reuses the existing verl environment's
# numerical/runtime stack.  It deliberately does NOT install dependencies such
# as numpy, packaging, sympy, or matplotlib into the overlay.

WORK=${WORK:-/groups/gcg51557/experiments/0390_rlsd}
PROJ=${PROJ:-$WORK/RLVR/verl_qwen3_30b_a3b_rlvr}
ENV_PREFIX=${ENV_PREFIX:-$WORK/envs/verl_qwen3_moe_megatron_py312_cu128}
WHEELHOUSE=${WHEELHOUSE:-$PROJ/vendor/reasoning_gym_wheels}
TARGET=${TARGET:-$PROJ/vendor/reasoning_gym_compat_py}
ACTION=${1:-}

export PYTHONNOUSERSITE=1

source "$PROJ/scripts/abci_verl_env.sh"
source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

audit_base_env() {
  RG_OLD_TARGET="$PROJ/vendor/reasoning_gym_py" python - <<'PY'
from importlib import metadata
from pathlib import Path
from packaging.version import Version
import importlib
import json
import os
import sys

old_target = Path(os.environ["RG_OLD_TARGET"]).resolve()

requirements = {
    # Preserve the already operational training stack.  NumPy 2.2.6 is used by
    # this environment and satisfies numba's <2.3 and mistral-common's <2.4
    # bounds.  Do not replace it with the wheelhouse's NumPy 2.5.2.
    "numpy": (None, Version("2.3")),
    "packaging": (None, Version("26")),
    "sympy": (Version("1.13.1"), None),
    "PyYAML": (Version("6.0.2"), None),
    # Reasoning Gym pins 0.9.0, but the existing 0.10.0 is tested directly
    # against all selected strata instead of being downgraded or shadowed.
    "tabulate": (Version("0.9.0"), None),
    "matplotlib": (Version("3.0.2"), None),
}

rows = []
failures = []
for dist, (minimum, maximum) in requirements.items():
    try:
        raw = metadata.version(dist)
        version = Version(raw)
        dist_obj = metadata.distribution(dist)
        location = str(Path(dist_obj.locate_file("")).resolve())
    except metadata.PackageNotFoundError:
        failures.append(f"missing required base distribution: {dist}")
        rows.append({"distribution": dist, "version": "MISSING", "location": ""})
        continue
    rows.append({"distribution": dist, "version": raw, "location": location})
    location_path = Path(location)
    if location_path == old_target or old_target in location_path.parents:
        failures.append(f"{dist} is loading from the unsafe old overlay: {location}")
    if minimum is not None and version < minimum:
        failures.append(f"{dist}=={raw} is below required {minimum}")
    if maximum is not None and version >= maximum:
        failures.append(f"{dist}=={raw} must be < {maximum}")

for module_name in ("rich", "mpmath", "six", "dateutil", "PIL", "pyparsing"):
    try:
        module = importlib.import_module(module_name)
        rows.append({
            "module": module_name,
            "version": getattr(module, "__version__", "available"),
            "location": str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else "",
        })
    except Exception as exc:
        failures.append(f"cannot import base module {module_name}: {type(exc).__name__}: {exc}")

print(json.dumps({"python": sys.executable, "base_packages": rows}, indent=2))
if failures:
    print("[FAIL] base environment is not yet suitable for a thin overlay:", file=sys.stderr)
    for item in failures:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(2)
try:
    pytz_version = metadata.version("pytz")
    print(f"[INFO] base pytz is available: {pytz_version}")
except metadata.PackageNotFoundError:
    print("[INFO] base pytz is absent; the thin overlay will supply pytz from the wheelhouse")
print("[PASS] base environment versions are suitable for compatibility testing")
PY
}

verify_overlay() {
  if [[ ! -d "$TARGET" ]]; then
    echo "[FAIL] overlay is absent: $TARGET" >&2
    exit 2
  fi
  RG_TARGET="$TARGET" RG_OLD_TARGET="$PROJ/vendor/reasoning_gym_py" \
    PYTHONPATH="$TARGET:$PROJ/compat${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from importlib import metadata
from pathlib import Path
from packaging.version import Version
import importlib
import json
import os
import sys

target = Path(os.environ["RG_TARGET"]).resolve()
old_target = Path(os.environ["RG_OLD_TARGET"]).resolve()

def module_info(name):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    return module, path

base_modules = {}
for name in ("numpy", "packaging", "sympy", "matplotlib"):
    module, path = module_info(name)
    base_modules[name] = {"version": getattr(module, "__version__", "unknown"), "path": str(path)}
    if path is not None and (path == target or target in path.parents):
        raise RuntimeError(f"{name} is incorrectly shadowed by the new Reasoning Gym overlay: {path}")
    if path is not None and (path == old_target or old_target in path.parents):
        raise RuntimeError(f"{name} is incorrectly shadowed by the unsafe old Reasoning Gym overlay: {path}")

if Version(base_modules["numpy"]["version"]) >= Version("2.3"):
    raise RuntimeError(f"unexpected numpy loaded (must preserve the working <2.3 stack): {base_modules['numpy']}")
if Version(base_modules["packaging"]["version"]) >= Version("26"):
    raise RuntimeError(f"pyvers-incompatible packaging loaded: {base_modules['packaging']}")

for name in ("torch", "transformers", "vllm", "verl", "megatron.core"):
    importlib.import_module(name)

import reasoning_gym
rg_path = Path(reasoning_gym.__file__).resolve()
if not (rg_path == target or target in rg_path.parents):
    raise RuntimeError(f"reasoning_gym did not load from overlay: {rg_path}")
if metadata.version("reasoning-gym") != "0.1.25":
    raise RuntimeError(f"unexpected reasoning-gym version: {metadata.version('reasoning-gym')}")

print(json.dumps({
    "python": sys.executable,
    "reasoning_gym": {"version": metadata.version("reasoning-gym"), "path": str(rg_path)},
    "base_modules": base_modules,
    "registered_tasks": len(reasoning_gym.factory.DATASETS),
}, indent=2))
print("[PASS] Reasoning Gym and verl/vLLM/Megatron import together without shadowing base packages")
PY

  RG_TARGET="$TARGET" PYTHONPATH="$TARGET:$PROJ/compat${PYTHONPATH:+:$PYTHONPATH}" \
    python "$PROJ/rewards/unicode_reasoning_gym_reward_v1.py"

  smoke_output="/tmp/0390_rg_compat_smoke_${$}.parquet"
  RG_TARGET="$TARGET" PYTHONPATH="$TARGET:$PROJ/compat${PYTHONPATH:+:$PYTHONPATH}" \
    python "$PROJ/scripts/build_reasoning_gym_parquet.py" \
      --config "$PROJ/config/reasoning_gym_calibration_v1.yaml" \
      --system-prompt-file "$PROJ/config/unicode_system_prompt.txt" \
      --output "$smoke_output" \
      --samples-per-stratum 1 \
      --split compat_smoke
  rm -f "$smoke_output" "${smoke_output}.manifest.json"
  echo "[PASS] all 16 task/tier strata generated and their stored oracles passed native verification"
}

one_wheel() {
  local pattern=$1
  local -a matches=()
  mapfile -t matches < <(compgen -G "$WHEELHOUSE/$pattern" || true)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "[FAIL] expected exactly one wheel for $pattern under $WHEELHOUSE; found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

install_overlay() {
  audit_base_env
  if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "[FAIL] wheelhouse is absent: $WHEELHOUSE" >&2
    exit 2
  fi
  if [[ -e "$TARGET" ]]; then
    echo "[FAIL] target already exists; refusing to overwrite: $TARGET" >&2
    echo "       Move it aside explicitly only if you intend to replace it." >&2
    exit 2
  fi

  local -a wheels=(
    "$(one_wheel 'reasoning_gym-0.1.25-*.whl')"
    "$(one_wheel 'arckit-0.1.0-*.whl')"
    "$(one_wheel 'bfi-1.0.4-*.whl')"
    "$(one_wheel 'cellpylib-2.4.0-*.whl')"
    "$(one_wheel 'drawsvg-2.4.2-*.whl')"
    "$(one_wheel 'magiccube-0.3.0-*.whl')"
    "$(one_wheel 'pycosat-0.6.6-*.whl')"
    "$(one_wheel 'pyfiglet-1.0.2-*.whl')"
    "$(one_wheel 'pytz-*.whl')"
    "$(one_wheel 'zss-1.2.0-*.whl')"
  )

  mkdir -p "$(dirname "$TARGET")"
  local tmp_target
  tmp_target=$(mktemp -d "$(dirname "$TARGET")/.reasoning_gym_compat_tmp.XXXXXX")
  trap 'rm -rf "$tmp_target"' EXIT

  python -m pip install --no-index --no-deps --target "$tmp_target" "${wheels[@]}"
  mv "$tmp_target" "$TARGET"
  trap - EXIT
  echo "[PASS] thin overlay created at $TARGET"
  verify_overlay
}

case "$ACTION" in
  audit)
    audit_base_env
    ;;
  install)
    install_overlay
    ;;
  verify)
    audit_base_env
    verify_overlay
    ;;
  *)
    echo "usage: $0 audit|install|verify" >&2
    exit 2
    ;;
esac
