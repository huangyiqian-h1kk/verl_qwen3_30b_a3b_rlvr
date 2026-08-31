#!/usr/bin/env bash
# Safely quarantine checkpoint directories newer than verl's committed tracker.
# Nothing is deleted. A missing/corrupt tracker with existing checkpoints is a
# hard stop rather than a reason to remove user data.

set -euo pipefail

CKPT_DIR=${1:?usage: checkpoint_preflight.sh CKPT_DIR}
EXPECTED_ROOT=${2:-"$HOME/ckpts/verl_full"}

resolved_ckpt=$(realpath -m "$CKPT_DIR")
resolved_root=$(realpath -m "$EXPECTED_ROOT")
case "$resolved_ckpt" in
  "$resolved_root"/*) ;;
  *)
    echo "[FAIL] checkpoint path is outside the allowed root: $resolved_ckpt" >&2
    exit 2
    ;;
esac

mkdir -p "$resolved_ckpt"
tracker="$resolved_ckpt/latest_checkpointed_iteration.txt"

shopt -s nullglob
steps=("$resolved_ckpt"/global_step_*)
if [ ! -f "$tracker" ]; then
  if [ "${#steps[@]}" -gt 0 ]; then
    echo "[FAIL] checkpoint directories exist but tracker is missing: $tracker" >&2
    printf '  %s\n' "${steps[@]}" >&2
    exit 3
  fi
  echo "[PASS] fresh checkpoint directory"
  exit 0
fi

raw_tracker=$(tr -d '[:space:]' < "$tracker")
case "$raw_tracker" in
  ''|*[!0-9]*)
    echo "[FAIL] tracker is not a strict non-negative integer: $tracker" >&2
    exit 4
    ;;
esac
latest=$raw_tracker

quarantine="$resolved_ckpt/_orphaned/$(date +%Y%m%d_%H%M%S)"
moved=0
for directory in "${steps[@]}"; do
  name=$(basename "$directory")
  step=${name#global_step_}
  case "$step" in
    ''|*[!0-9]*) continue ;;
  esac
  if [ "$step" -gt "$latest" ]; then
    mkdir -p "$quarantine"
    echo "[WARN] quarantining uncommitted checkpoint newer than tracker: $directory"
    mv "$directory" "$quarantine/"
    moved=1
  fi
done

echo "[PASS] committed tracker step: $latest"
if [ "$moved" -eq 1 ]; then
  echo "[WARN] orphaned data was moved, not deleted: $quarantine"
  echo "[WARN] inspect it and remove it manually only after a successful resume."
fi
