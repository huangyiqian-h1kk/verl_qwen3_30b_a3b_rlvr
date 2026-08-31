#!/usr/bin/env bash
# Guard the transient three-checkpoint write peak against the user's /home quota.

set -euo pipefail

CKPT_DIR=${1:?usage: check_home_quota.sh CKPT_DIR [FRESH_MIN_GIB] [RESUME_MIN_GIB]}
FRESH_MIN_GIB=${2:-1500}
RESUME_MIN_GIB=${3:-600}

for value in "$FRESH_MIN_GIB" "$RESUME_MIN_GIB"; do
  case "$value" in
    ''|*[!0-9]*) echo "[FAIL] quota thresholds must be integer GiB" >&2; exit 2 ;;
  esac
done

if ! command -v show_quota >/dev/null; then
  echo "[FAIL] show_quota is unavailable; cannot enforce /home space guard" >&2
  exit 3
fi

quota_output=$(show_quota -b G)
line=$(awk -v home_path="$HOME" '$1 == home_path {print $2, $3; exit}' <<<"$quota_output")
if [ -z "$line" ]; then
  echo "$quota_output"
  echo "[FAIL] could not parse the user /home row from show_quota -b G" >&2
  exit 4
fi
read -r used_gib limit_gib <<<"$line"
case "$used_gib:$limit_gib" in
  *[!0-9:]*|:*) echo "[FAIL] non-integer quota values: $line" >&2; exit 4 ;;
esac
free_gib=$((limit_gib - used_gib))

shopt -s nullglob
steps=("$CKPT_DIR"/global_step_*)
if [ "${#steps[@]}" -eq 0 ]; then
  required_gib=$FRESH_MIN_GIB
  phase=fresh
else
  required_gib=$RESUME_MIN_GIB
  phase=resume
fi

echo "[quota] used=${used_gib}GiB limit=${limit_gib}GiB free=${free_gib}GiB"
echo "[quota] phase=$phase required_free=${required_gib}GiB checkpoint_dir=$CKPT_DIR"
if [ "$free_gib" -lt "$required_gib" ]; then
  echo "[FAIL] insufficient /home quota headroom for checkpoint save peak" >&2
  exit 5
fi
echo "[PASS] /home quota guard"
