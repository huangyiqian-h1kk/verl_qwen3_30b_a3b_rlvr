#!/usr/bin/env bash
# Safely retain only the newest N committed full checkpoints in /home.
#
# A checkpoint is eligible for deletion only when its corresponding long-term
# Hugging Face snapshot in /groups has a .complete marker.  The marker is
# created by snapshot_watcher_v3.sh only after shard and tokenizer checks pass.

set -euo pipefail

CKPT_DIR=${1:?usage: prune_full_checkpoints_v3.sh CKPT_DIR SNAP_DIR [KEEP]}
SNAP_DIR=${2:?missing SNAP_DIR}
KEEP=${3:-2}
EXPECTED_ROOT=${EXPECTED_FULL_CKPT_ROOT:-"$HOME/ckpts/verl_full"}

log() { echo "[pruner $(date '+%F %T')] $*"; }
fail() { log "FAIL: $*" >&2; exit 1; }

case "$KEEP" in
  ''|*[!0-9]*) fail "KEEP must be an integer >= 1" ;;
  0) fail "KEEP must be >= 1" ;;
esac

resolved_ckpt=$(realpath -m "$CKPT_DIR")
resolved_snap=$(realpath -m "$SNAP_DIR")
resolved_root=$(realpath -m "$EXPECTED_ROOT")

case "$resolved_ckpt" in
  "$resolved_root"/*) ;;
  *) fail "refusing checkpoint directory outside $resolved_root: $resolved_ckpt" ;;
esac

[ -d "$resolved_ckpt" ] || fail "checkpoint directory is missing: $resolved_ckpt"
[ -d "$resolved_snap" ] || fail "snapshot directory is missing: $resolved_snap"

tracker="$resolved_ckpt/latest_checkpointed_iteration.txt"
[ -f "$tracker" ] || fail "committed checkpoint tracker is missing: $tracker"
latest=$(tr -d '[:space:]' < "$tracker")
case "$latest" in
  ''|*[!0-9]*) fail "checkpoint tracker is not a non-negative integer: $latest" ;;
esac

lock_dir="$resolved_ckpt/.full_ckpt_prune_lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  fail "another full-checkpoint pruner is active: $lock_dir"
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

records=()
while IFS= read -r directory; do
  name=${directory##*/}
  step=${name#global_step_}
  case "$step" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$step" -le "$latest" ]; then
    records+=("$(printf '%020d\t%s\t%s' "$step" "$step" "$directory")")
  fi
done < <(find "$resolved_ckpt" -maxdepth 1 -mindepth 1 -type d \
  -name 'global_step_*' -print)

if [ "${#records[@]}" -le "$KEEP" ]; then
  log "nothing to prune: committed=$latest found=${#records[@]} keep=$KEEP"
  exit 0
fi

mapfile -t records < <(printf '%s\n' "${records[@]}" | sort -n)
delete_count=$((${#records[@]} - KEEP))
candidates=()

# Validate every candidate before deleting any candidate.
for ((index = 0; index < delete_count; index++)); do
  IFS=$'\t' read -r _sort_key step directory <<<"${records[$index]}"
  expected_dir="$resolved_ckpt/global_step_$step"
  resolved_dir=$(realpath -m "$directory")
  [ "$resolved_dir" = "$expected_dir" ] ||
    fail "candidate path mismatch: expected=$expected_dir actual=$resolved_dir"
  [ -d "$resolved_dir" ] || fail "candidate disappeared before pruning: $resolved_dir"

  snapshot="$resolved_snap/global_step_$step"
  [ -f "$snapshot/.complete" ] ||
    fail "refusing to delete step $step: snapshot .complete is missing"
  [ -f "$snapshot/.snapshot_meta" ] ||
    fail "refusing to delete step $step: snapshot metadata is missing"
  [ -f "$snapshot/config.json" ] ||
    fail "refusing to delete step $step: snapshot config.json is missing"

  candidates+=("$resolved_dir")
done

for directory in "${candidates[@]}"; do
  step=${directory##*/global_step_}
  log "deleting full checkpoint global_step_$step after verified snapshot harvest"
  rm -rf -- "$directory"
  [ ! -e "$directory" ] || fail "failed to remove $directory"
done

log "prune complete: committed=$latest deleted=$delete_count keep=$KEEP"
