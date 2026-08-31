#!/usr/bin/env bash
# Guard the temporary third-checkpoint write peak.  ABCI compute nodes cannot
# run show_quota, so they consume a login-node snapshot passed via qsub.

set -euo pipefail

CKPT_DIR=${1:?usage: check_home_quota_v3.sh CKPT_DIR [FRESH_MIN_GIB] [RESUME_MIN_GIB]}
FRESH_MIN_GIB=${2:-1500}
RESUME_MIN_GIB=${3:-600}
HOME_FREE_GIB_AT_SUBMIT=${HOME_FREE_GIB_AT_SUBMIT:-}

for value in "$FRESH_MIN_GIB" "$RESUME_MIN_GIB"; do
  case "$value" in
    ''|*[!0-9]*) echo "[FAIL] quota thresholds must be integer GiB" >&2; exit 2 ;;
  esac
done

if command -v show_quota >/dev/null 2>&1; then
  quota_output=$(show_quota -b G)
  line=$(awk -v home_path="$HOME" '$1 == home_path {print $2, $3; exit}' <<<"$quota_output")
  [ -n "$line" ] || {
    echo "$quota_output"
    echo "[FAIL] could not parse the /home row from show_quota -b G" >&2
    exit 3
  }
  read -r used_gib limit_gib <<<"$line"
  case "$used_gib:$limit_gib" in
    *[!0-9:]*|:*) echo "[FAIL] non-integer quota values: $line" >&2; exit 3 ;;
  esac
  free_gib=$((limit_gib - used_gib))
  source_label=live-login-query
else
  case "$HOME_FREE_GIB_AT_SUBMIT" in
    ''|*[!0-9]*)
      echo "[FAIL] show_quota is unavailable on this compute node." >&2
      echo "[FAIL] Query it immediately before qsub and pass:" >&2
      echo "[FAIL]   HOME_FREE_GIB_AT_SUBMIT=<free_GiB>" >&2
      exit 3
      ;;
  esac
  free_gib=$HOME_FREE_GIB_AT_SUBMIT
  source_label=login-node-submission-snapshot
fi

shopt -s nullglob
steps=("$CKPT_DIR"/global_step_*)
if [ "${#steps[@]}" -eq 0 ]; then
  required_gib=$FRESH_MIN_GIB
  phase=fresh
else
  largest_ckpt_gib=0
  for directory in "${steps[@]}"; do
    [ -d "$directory" ] || continue
    size_gib=$(du -s --block-size=1G "$directory" | awk '{print $1}')
    if [ "$size_gib" -gt "$largest_ckpt_gib" ]; then
      largest_ckpt_gib=$size_gib
    fi
  done
  dynamic_required=$((largest_ckpt_gib + 100))
  required_gib=$RESUME_MIN_GIB
  if [ "$dynamic_required" -gt "$required_gib" ]; then
    required_gib=$dynamic_required
  fi
  phase=resume
  echo "[quota] largest_checkpoint=${largest_ckpt_gib}GiB dynamic_requirement=${dynamic_required}GiB"
fi

echo "[quota] source=$source_label free=${free_gib}GiB"
echo "[quota] phase=$phase required_free=${required_gib}GiB checkpoint_dir=$CKPT_DIR"
if [ "$free_gib" -lt "$required_gib" ]; then
  echo "[FAIL] insufficient /home quota headroom for checkpoint save peak" >&2
  exit 5
fi
echo "[PASS] /home quota guard"
