#!/usr/bin/env bash
# Harvest only curve-selected HF snapshots, then invoke the safe full-checkpoint
# pruner.  Full checkpoints may be saved more frequently than curve snapshots.

set -uo pipefail

MODE=${1:?usage: snapshot_watcher_v4_selected.sh watch|oneshot CKPT_DIR SNAP_DIR POLICY_FILE [POLL_SEC]}
CKPT_DIR=${2:?missing CKPT_DIR}
SNAP_DIR=${3:?missing SNAP_DIR}
POLICY_FILE=${4:?missing POLICY_FILE}
POLL_SEC=${5:-30}
TOKENIZER_FALLBACK_DIR=${TOKENIZER_FALLBACK_DIR:-}
KEEP_FULL_CKPTS=${KEEP_FULL_CKPTS:-2}
EXPECTED_FULL_CKPT_ROOT=${EXPECTED_FULL_CKPT_ROOT:-}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FULL_CKPT_PRUNER=${FULL_CKPT_PRUNER:-"$SCRIPT_DIR/prune_full_checkpoints_v4_selected.sh"}

log() { echo "[watcher $(date '+%F %T')] $*"; }

case "$KEEP_FULL_CKPTS" in
  ''|*[!0-9]*|0) log "ERROR: KEEP_FULL_CKPTS must be an integer >= 1"; exit 2 ;;
esac
[[ -f "$POLICY_FILE" ]] || { log "ERROR: snapshot policy is missing: $POLICY_FILE"; exit 2; }
[[ -x "$FULL_CKPT_PRUNER" ]] || { log "ERROR: pruner is not executable: $FULL_CKPT_PRUNER"; exit 2; }

policy_has_step() {
  local target=$1
  awk -v target="$target" '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    { gsub(/[[:space:]]/, "", $0); if ($0 == target) found=1 }
    END { exit(found ? 0 : 1) }
  ' "$POLICY_FILE"
}

mkdir -p "$SNAP_DIR"
lock_dir="$SNAP_DIR/.watcher_lock"

acquire_lock() {
  if ! mkdir "$lock_dir" 2>/dev/null; then
    log "ERROR: another watcher appears active: $lock_dir"
    return 1
  fi
  echo "$$" > "$lock_dir/pid"
}

release_lock() {
  rm -f -- "$lock_dir/pid"
  rmdir "$lock_dir" 2>/dev/null || true
}

resolve_hf_subpath() {
  local step_dir=$1 candidate
  for candidate in actor/model/huggingface actor/huggingface; do
    [[ -d "$step_dir/$candidate" ]] && { echo "$candidate"; return 0; }
  done
  return 1
}

verify_hf_tree() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit("missing config.json")
index = root / "model.safetensors.index.json"
if index.is_file():
    data = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted(set(data.get("weight_map", {}).values()))
    if not shards:
        raise SystemExit("empty safetensor weight map")
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing safetensor shards: {missing[:5]}")
elif not list(root.glob("*.safetensors")):
    raise SystemExit("no safetensor weights")
PY
}

backfill_tokenizer() {
  local directory=$1 name
  [[ -n "$TOKENIZER_FALLBACK_DIR" ]] || return 0
  for name in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja generation_config.json vocab.json merges.txt; do
    [[ -e "$directory/$name" ]] && continue
    [[ -e "$TOKENIZER_FALLBACK_DIR/$name" ]] || continue
    cp -L "$TOKENIZER_FALLBACK_DIR/$name" "$directory/"
  done
}

harvest_one() {
  local step_dir=$1 step_name step src_subpath src dest incomplete
  step_name=$(basename "$step_dir")
  step=${step_name#global_step_}
  policy_has_step "$step" || return 0
  dest="$SNAP_DIR/$step_name"
  incomplete="$SNAP_DIR/.${step_name}.incomplete"

  [[ -f "$dest/.complete" ]] && return 0
  [[ ! -e "$dest" ]] || { log "ERROR: refusing to overwrite incomplete destination: $dest"; return 1; }
  src_subpath=$(resolve_hf_subpath "$step_dir") || { log "WARN: HF tree not found for $step_name"; return 1; }
  src="$step_dir/$src_subpath"
  verify_hf_tree "$src" || { log "WARN: source HF tree incomplete: $src"; return 1; }

  mkdir -p "$incomplete"
  log "harvesting selected $step_name"
  rsync -a --partial "$src/" "$incomplete/" || return 1
  backfill_tokenizer "$incomplete"
  verify_hf_tree "$incomplete" || return 1
  [[ -f "$incomplete/tokenizer_config.json" ]] || { log "ERROR: tokenizer_config.json missing"; return 1; }
  {
    echo "experiment_src=$CKPT_DIR"
    echo "step_dir=$step_name"
    echo "policy_file=$POLICY_FILE"
    echo "policy_sha256=$(sha256sum "$POLICY_FILE" | awk '{print $1}')"
    echo "harvested_at=$(date --iso-8601=seconds)"
    echo "file_count=$(find "$incomplete" -type f | wc -l)"
    echo "size_bytes=$(du -sb "$incomplete" | cut -f1)"
  } > "$incomplete/.snapshot_meta"
  mv "$incomplete" "$dest"
  touch "$dest/.complete"
  log "complete: $dest"
}

sweep() {
  local tracker="$CKPT_DIR/latest_checkpointed_iteration.txt" latest directory step failures=0
  [[ -f "$tracker" ]] || { log "no committed checkpoint tracker yet"; return 0; }
  latest=$(tr -d '[:space:]' < "$tracker")
  case "$latest" in ''|*[!0-9]*) log "ERROR: invalid tracker: $tracker"; return 1 ;; esac

  while IFS= read -r directory; do
    [[ -d "$directory" ]] || continue
    step=${directory##*/global_step_}
    case "$step" in ''|*[!0-9]*) continue ;; esac
    [[ "$step" -le "$latest" ]] || continue
    harvest_one "$directory" || failures=1
  done < <(find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V)

  if [[ "$failures" -eq 0 ]]; then
    EXPECTED_FULL_CKPT_ROOT="$EXPECTED_FULL_CKPT_ROOT" bash "$FULL_CKPT_PRUNER" \
      "$CKPT_DIR" "$SNAP_DIR" "$POLICY_FILE" "$KEEP_FULL_CKPTS" || failures=1
  fi
  return "$failures"
}

case "$MODE" in
  watch)
    acquire_lock || exit 2
    trap release_lock EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    while true; do sweep || true; sleep "$POLL_SEC"; done
    ;;
  oneshot)
    acquire_lock || exit 2
    trap release_lock EXIT
    sweep
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
