#!/usr/bin/env bash
# Retain the newest N committed full checkpoints.  A curve-selected checkpoint
# can be deleted only after its verified model-only snapshot exists.

set -euo pipefail

CKPT_DIR=${1:?usage: prune_full_checkpoints_v4_selected.sh CKPT_DIR SNAP_DIR POLICY_FILE [KEEP]}
SNAP_DIR=${2:?missing SNAP_DIR}
POLICY_FILE=${3:?missing POLICY_FILE}
KEEP=${4:-2}
EXPECTED_ROOT=${EXPECTED_FULL_CKPT_ROOT:?set EXPECTED_FULL_CKPT_ROOT explicitly}

log() { echo "[pruner $(date '+%F %T')] $*"; }
fail() { log "FAIL: $*" >&2; exit 1; }

case "$KEEP" in ''|*[!0-9]*|0) fail "KEEP must be an integer >= 1" ;; esac
[[ -f "$POLICY_FILE" ]] || fail "policy file missing: $POLICY_FILE"

resolved_ckpt=$(realpath -m "$CKPT_DIR")
resolved_snap=$(realpath -m "$SNAP_DIR")
resolved_root=$(realpath -m "$EXPECTED_ROOT")
case "$resolved_ckpt" in "$resolved_root"/*) ;; *) fail "checkpoint dir outside $resolved_root: $resolved_ckpt" ;; esac
[[ -d "$resolved_ckpt" ]] || fail "checkpoint directory missing"
[[ -d "$resolved_snap" ]] || fail "snapshot directory missing"

policy_has_step() {
  local target=$1
  awk -v target="$target" '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    { gsub(/[[:space:]]/, "", $0); if ($0 == target) found=1 }
    END { exit(found ? 0 : 1) }
  ' "$POLICY_FILE"
}

tracker="$resolved_ckpt/latest_checkpointed_iteration.txt"
[[ -f "$tracker" ]] || { log "nothing committed yet"; exit 0; }
latest=$(tr -d '[:space:]' < "$tracker")
case "$latest" in ''|*[!0-9]*) fail "invalid checkpoint tracker: $latest" ;; esac

lock_dir="$resolved_ckpt/.full_ckpt_prune_lock"
mkdir "$lock_dir" 2>/dev/null || fail "another pruner is active: $lock_dir"
cleanup_lock() { rmdir "$lock_dir" 2>/dev/null || true; }
trap cleanup_lock EXIT

records=()
while IFS= read -r directory; do
  name=${directory##*/}; step=${name#global_step_}
  case "$step" in ''|*[!0-9]*) continue ;; esac
  [[ "$step" -le "$latest" ]] || continue
  records+=("$(printf '%020d\t%s\t%s' "$step" "$step" "$directory")")
done < <(find "$resolved_ckpt" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print)

[[ "${#records[@]}" -gt "$KEEP" ]] || { log "nothing to prune; found=${#records[@]} keep=$KEEP"; exit 0; }
mapfile -t records < <(printf '%s\n' "${records[@]}" | sort -n)
delete_count=$((${#records[@]} - KEEP))
candidates=()

for ((index=0; index<delete_count; index++)); do
  IFS=$'\t' read -r _key step directory <<<"${records[$index]}"
  resolved_dir=$(realpath -m "$directory")
  [[ "$resolved_dir" == "$resolved_ckpt/global_step_$step" ]] || fail "candidate path mismatch: $resolved_dir"
  [[ -d "$resolved_dir" ]] || fail "candidate disappeared: $resolved_dir"
  if policy_has_step "$step"; then
    snapshot="$resolved_snap/global_step_$step"
    [[ -f "$snapshot/.complete" ]] || fail "selected step $step has no complete snapshot"
    [[ -f "$snapshot/.snapshot_meta" ]] || fail "selected step $step has no snapshot metadata"
    [[ -f "$snapshot/config.json" ]] || fail "selected step $step snapshot has no config.json"
  fi
  candidates+=("$resolved_dir")
done

for directory in "${candidates[@]}"; do
  step=${directory##*/global_step_}
  log "deleting committed full checkpoint step $step"
  rm -rf -- "$directory"
  [[ ! -e "$directory" ]] || fail "failed to remove $directory"
done
log "prune complete: latest=$latest deleted=$delete_count keep=$KEEP"
