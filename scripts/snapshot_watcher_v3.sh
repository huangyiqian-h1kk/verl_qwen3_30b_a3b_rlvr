#!/usr/bin/env bash
# Harvest completed Hugging Face actor snapshots and safely rotate full
# checkpoints across fresh and resumed PBS jobs.
#
# Supports both layouts:
#   global_step_N/actor/huggingface
#   global_step_N/actor/model/huggingface
# and prefers actor/ckpt_contents.json when present.

set -uo pipefail

MODE=${1:?usage: snapshot_watcher_v2.sh watch|oneshot CKPT_DIR SNAP_DIR [POLL_SEC]}
CKPT_DIR=${2:?missing CKPT_DIR}
SNAP_DIR=${3:?missing SNAP_DIR}
POLL_SEC=${4:-30}
TOKENIZER_FALLBACK_DIR=${TOKENIZER_FALLBACK_DIR:-}
KEEP_FULL_CKPTS=${KEEP_FULL_CKPTS:-0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FULL_CKPT_PRUNER=${FULL_CKPT_PRUNER:-"$SCRIPT_DIR/prune_full_checkpoints_v3.sh"}

log() { echo "[watcher $(date '+%F %T')] $*"; }

case "$KEEP_FULL_CKPTS" in
  ''|*[!0-9]*)
    log "ERROR: KEEP_FULL_CKPTS must be a non-negative integer"
    exit 2
    ;;
esac
if [ "$KEEP_FULL_CKPTS" -gt 0 ] && [ ! -x "$FULL_CKPT_PRUNER" ]; then
  log "ERROR: full-checkpoint pruner is missing or not executable: $FULL_CKPT_PRUNER"
  exit 2
fi

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
  rm -rf "$lock_dir"
}

resolve_hf_subpath() {
  local step_dir=$1
  local manifest="$step_dir/actor/ckpt_contents.json"
  if [ -f "$manifest" ]; then
    local manifest_path
    if manifest_path=$(python3 - "$manifest" <<'PY'
import json, sys
from pathlib import Path

manifest = Path(sys.argv[1])
try:
    data = json.loads(manifest.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
contents = data.get("contents", {})
for key in ("hf_model", "model", "hf_config"):
    item = contents.get(key)
    if not isinstance(item, dict):
        continue
    path = item.get("path")
    fmt = str(item.get("format", "")).lower()
    if path and ("huggingface" in fmt or "huggingface" in str(path)):
        print(f"actor/{path}")
        raise SystemExit(0)
raise SystemExit(1)
PY
    ); then
      echo "$manifest_path"
      return 0
    fi
    log "WARN: manifest did not resolve an HF path; trying known layouts" >&2
  fi

  local candidate
  for candidate in actor/model/huggingface actor/huggingface; do
    if [ -d "$step_dir/$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

verify_hf_tree() {
  local directory=$1
  python3 - "$directory" <<'PY'
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
  local directory=$1
  [ -n "$TOKENIZER_FALLBACK_DIR" ] || return 0
  [ -f "$directory/tokenizer_config.json" ] && return 0
  log "backfilling tokenizer/config artifacts from $TOKENIZER_FALLBACK_DIR"
  local name
  for name in \
    tokenizer.json tokenizer_config.json special_tokens_map.json \
    chat_template.jinja generation_config.json vocab.json merges.txt; do
    [ -e "$TOKENIZER_FALLBACK_DIR/$name" ] || continue
    cp -L "$TOKENIZER_FALLBACK_DIR/$name" "$directory/"
  done
}

harvest_one() {
  local step_dir=$1
  local step_name src_subpath src dest incomplete
  step_name=$(basename "$step_dir")
  dest="$SNAP_DIR/$step_name"
  incomplete="$SNAP_DIR/.${step_name}.incomplete"

  [ -f "$dest/.complete" ] && return 0
  if ! src_subpath=$(resolve_hf_subpath "$step_dir"); then
    log "WARN: cannot locate HF model for $step_name"
    find "$step_dir/actor" -maxdepth 3 -mindepth 1 2>/dev/null | sed 's/^/[watcher] /'
    return 1
  fi
  src="$step_dir/$src_subpath"
  if ! verify_hf_tree "$src"; then
    log "WARN: source HF tree is incomplete: $src"
    return 1
  fi

  mkdir -p "$incomplete"
  log "harvesting $step_name from $src_subpath"
  if ! rsync -a --partial "$src/" "$incomplete/"; then
    log "ERROR: rsync failed; retaining resumable partial directory $incomplete"
    return 1
  fi
  backfill_tokenizer "$incomplete"
  if ! verify_hf_tree "$incomplete"; then
    log "ERROR: copied snapshot failed shard verification: $incomplete"
    return 1
  fi
  if [ ! -f "$incomplete/tokenizer_config.json" ]; then
    log "ERROR: tokenizer_config.json is still missing: $incomplete"
    return 1
  fi

  {
    echo "experiment_src=$CKPT_DIR"
    echo "step_dir=$step_name"
    echo "hf_source_subpath=$src_subpath"
    echo "harvested_at=$(date --iso-8601=seconds)"
    echo "harvested_on=$(hostname)"
    echo "file_count=$(find "$incomplete" -type f | wc -l)"
    echo "size_bytes=$(du -sb "$incomplete" | cut -f1)"
  } > "$incomplete/.snapshot_meta"

  rm -rf "$dest"
  mv "$incomplete" "$dest"
  touch "$dest/.complete"
  log "complete: $dest ($(du -sh "$dest" | cut -f1))"
}

sweep() {
  local tracker="$CKPT_DIR/latest_checkpointed_iteration.txt"
  [ -f "$tracker" ] || { log "no tracker yet"; return 0; }
  local latest
  latest=$(tr -d '[:space:]' < "$tracker" 2>/dev/null)
  case "$latest" in
    ''|*[!0-9]*) log "ERROR: unreadable tracker: $tracker"; return 1 ;;
  esac

  local directory step failures=0
  while IFS= read -r directory; do
    [ -d "$directory" ] || continue
    step=${directory##*/global_step_}
    case "$step" in ''|*[!0-9]*) continue ;; esac
    if [ "$step" -le "$latest" ]; then
      harvest_one "$directory" || failures=1
    fi
  done < <(find "$CKPT_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort -V)

  # The built-in verl retention queue is not reliable across PBS process
  # restarts.  Prune only after every eligible snapshot in this sweep has
  # completed, and let the pruner fail closed on any missing safety marker.
  if [ "$failures" -eq 0 ] && [ "$KEEP_FULL_CKPTS" -gt 0 ]; then
    if ! bash "$FULL_CKPT_PRUNER" \
      "$CKPT_DIR" "$SNAP_DIR" "$KEEP_FULL_CKPTS"; then
      log "ERROR: safe full-checkpoint pruning failed"
      failures=1
    fi
  fi
  return "$failures"
}

case "$MODE" in
  watch)
    acquire_lock || exit 2
    trap release_lock EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    log "watch mode: poll=${POLL_SEC}s"
    while true; do
      sweep || true
      sleep "$POLL_SEC"
    done
    ;;
  oneshot)
    acquire_lock || exit 2
    trap release_lock EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    log "oneshot"
    sweep
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
