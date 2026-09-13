#!/usr/bin/env python3
"""Install the RG v5 watcher/PBS repair; optionally recover saved snapshots.

Usage (from the repository root):
  python3 repair_rg_snapshot_watcher.py
  python3 repair_rg_snapshot_watcher.py --recover

The first command backs up and patches scripts only. It does not submit/stop
jobs or modify checkpoints. --recover additionally exports existing selected
committed steps and retains the newest two complete full checkpoints; it
defers if training holds the lock. Existing snapshots and policy are preserved.
Python standard library only; no torch import and no model loaded into RAM.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PAYLOADS = {'scripts/rg_checkpoint_maintenance_v5.py': '#!/usr/bin/env python3\n"""Checkpoint maintenance v5: process locks, atomic HF snapshots, guarded pruning.\nOnly the standard library is used. Weight validation reads headers, not tensors.\n"""\nfrom __future__ import annotations\nimport argparse\nfrom contextlib import ExitStack, contextmanager\nfrom datetime import datetime\nimport fcntl\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport shutil\nimport signal\nimport socket\nimport stat\nimport struct\nimport sys\nimport tempfile\nimport time\n\nHF_LAYOUTS = ("actor/model/huggingface", "actor/huggingface")\nTOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",\n    "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",\n    "chat_template.jinja", "generation_config.json")\n\ndef log(level, message):\n    print(f"[watcher {datetime.now().astimezone().isoformat(timespec=\'seconds\')}] {level}: {message}", flush=True)\n\ndef json_object(path: Path) -> dict:\n    value = json.loads(path.read_text(encoding="utf-8"))\n    if not isinstance(value, dict):\n        raise ValueError(f"Expected a JSON object: {path}")\n    return value\n\n\ndef check_tensor_file(path: Path) -> set[str]:\n    """Check safetensors header/offsets/file length, without reading tensor data.\n\n    This is a structural check, not a full checksum of weight values.\n    """\n    size = path.stat().st_size\n    with path.open("rb") as stream:\n        prefix = stream.read(8)\n        if len(prefix) != 8:\n            raise ValueError(f"Missing safetensors header: {path}")\n        length = struct.unpack("<Q", prefix)[0]\n        if not 2 <= length <= 100_000_000 or 8 + length > size:\n            raise ValueError(f"Invalid/truncated safetensors header: {path}")\n        header = json.loads(stream.read(length))\n    if not isinstance(header, dict):\n        raise ValueError(f"Invalid safetensors header object: {path}")\n    tensors = {key: value for key, value in header.items() if key != "__metadata__"}\n    if not tensors:\n        raise ValueError(f"No tensors in {path}")\n    intervals = []\n    for name, value in tensors.items():\n        offsets = value.get("data_offsets") if isinstance(value, dict) else None\n        if (not isinstance(offsets, list) or len(offsets) != 2\n                or any(type(item) is not int for item in offsets)\n                or not 0 <= offsets[0] <= offsets[1]):\n            raise ValueError(f"Invalid offsets for tensor {name!r}: {path}")\n        intervals.append(tuple(offsets))\n    end = 0\n    for start, next_end in sorted(intervals):\n        if start != end:\n            raise ValueError(f"Non-contiguous/overlapping tensor data: {path}")\n        end = next_end\n    if 8 + length + end != size:\n        raise ValueError(f"Tensor data length does not match header: {path}")\n    return set(tensors)\n\n\ndef inspect_weights(directory: Path) -> tuple[list[str], dict]:\n    config = json_object(directory / "config.json")\n    index_path = directory / "model.safetensors.index.json"\n    if index_path.is_file():\n        weight_map = json_object(index_path).get("weight_map")\n        if not isinstance(weight_map, dict) or not weight_map:\n            raise ValueError(f"Missing/empty weight_map: {index_path}")\n        names = set()\n        for name in weight_map.values():\n            if (not isinstance(name, str) or Path(name).is_absolute()\n                    or ".." in Path(name).parts or not name.endswith(".safetensors")):\n                raise ValueError(f"Invalid model shard name in {index_path}: {name!r}")\n            names.add(name)\n        names = sorted(names)\n    else:\n        names = ["model.safetensors"]\n        weight_map = None\n    headers = {name: check_tensor_file(directory / name) for name in names}\n    if weight_map is not None:\n        for tensor, shard in weight_map.items():\n            if tensor not in headers[shard]:\n                raise ValueError(f"Index tensor {tensor!r} is absent from {directory / shard}")\n    return names, config\n\n\ndef tokenizer_is_usable(directory: Path) -> None:\n    json_object(directory / "tokenizer_config.json")\n    if (directory / "tokenizer.json").is_file():\n        json_object(directory / "tokenizer.json")\n    elif not any((directory / name).is_file() for name in ("tokenizer.model", "spiece.model")):\n        if not all((directory / name).is_file() for name in ("vocab.json", "merges.txt")):\n            raise ValueError(f"Tokenizer data files are missing: {directory}")\n\n\ndef verify_snapshot(directory: Path) -> None:\n    inspect_weights(directory)\n    tokenizer_is_usable(directory)\n    if not (directory / ".complete").is_file() or not (directory / ".snapshot_meta").is_file():\n        raise ValueError(f"Existing snapshot lacks completion metadata: {directory}")\n\n\ndef resolve_source(step_dir: Path) -> Path:\n    for layout in HF_LAYOUTS:\n        directory = step_dir / layout\n        if (directory / "config.json").is_file():\n            try:\n                inspect_weights(directory)\n            except (OSError, ValueError) as error:\n                log("WARN", f"Cannot use {directory}: {error}")\n                continue\n            return directory\n    raise ValueError(\n        f"No complete HF safetensors export in {step_dir}; expected one of {HF_LAYOUTS}. "\n        "This script does not convert Megatron distributed shards. Full checkpoint retained."\n    )\n\n\n@contextmanager\ndef locked(path, label):\n    # Never unlink lock files: unlinking would create two independent lock inodes.\n    if path.is_symlink():\n        raise ValueError(f"Refusing symlink lock: {path}")\n    with path.open("a+") as handle:\n        try:\n            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        except BlockingIOError:\n            raise ValueError(f"{label} is active: {path}") from None\n        handle.seek(0)\n        handle.truncate()\n        json.dump({"pid": os.getpid(), "host": socket.gethostname(),\n                   "job": os.environ.get("PBS_JOBID"), "started": time.time()}, handle)\n        handle.flush()\n        yield\n\n\ndef reject_local_only_lustre_locks(path):\n    # localflock cannot exclude a process on a different compute/login node.\n    entries = []\n    for line in Path(\'/proc/self/mountinfo\').read_text().splitlines():\n        left, right = line.split(\' - \', 1)\n        fields, extra = left.split(), right.split()\n        mount = Path(re.sub(r\'\\\\([0-7]{3})\', lambda m: chr(int(m[1], 8)), fields[4]))\n        if path == mount or mount in path.parents:\n            entries.append((len(mount.parts), extra[0], fields[5] + \',\' + extra[2]))\n    if entries:\n        _, fs, options = max(entries)\n        if fs == \'lustre\' and {\'localflock\', \'noflock\'} & set(options.split(\',\')):\n            raise ValueError(f"Shared cross-node locking is unavailable on {path}: {options}")\n\n\ndef atomic_text(path, value):\n    fd, temporary = tempfile.mkstemp(prefix=\'.\' + path.name + \'.\', dir=path.parent)\n    try:\n        with os.fdopen(fd, \'w\') as output:\n            output.write(value)\n            output.flush()\n            os.fsync(output.fileno())\n        os.replace(temporary, path)\n    finally:\n        if os.path.exists(temporary):\n            os.unlink(temporary)\n\n\ndef migrate_legacy_locks(full, snap):\n    """Exclude old PBS descendants using the training lock; preserve old dirs.\n\n    The old PBS passes fd 9 to its watcher/pruner. A new PBS can acquire that\n    lock only once those old processes have exited. A login-node invocation\n    instead takes the training lock itself for the migration.\n    """\n    marker = snap / \'.watcher_v5_migration.json\'\n    if marker.is_file():\n        return\n    training_lock = full / \'.training.lock\'\n    training_fd = int(os.environ.get(\'RG_TRAINING_LOCK_FD\', \'9\'))\n    with ExitStack() as stack:\n        inherited = False\n        try:\n            inherited = os.path.samestat(os.fstat(training_fd), training_lock.stat())\n        except OSError:\n            pass\n        if inherited:\n            fcntl.flock(training_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        else:\n            stack.enter_context(locked(training_lock, \'Training (legacy-lock migration deferred)\'))\n        # These guard directories also prevent an old v4 script from starting.\n        # No PID-only test is used: PIDs have no cross-node identity.\n        for directory in (snap / \'.watcher_lock\', full / \'.full_ckpt_prune_lock\'):\n            if directory.is_symlink():\n                raise ValueError(f"Unexpected legacy lock symlink: {directory}")\n            directory.mkdir(exist_ok=True)\n            atomic_text(directory / \'v5_guard\', \'Legacy directory retained; v5 uses flock.\\n\')\n        atomic_text(marker, json.dumps({\'host\': socket.gethostname(), \'pid\': os.getpid(),\n                                      \'migrated_at\': time.time()}) + \'\\n\')\n        log(\'PASS\', \'legacy locks migrated under the training lock; no lock directories deleted\')\n\n\ndef policy_steps(path):\n    steps = set()\n    for line in path.read_text().splitlines():\n        value = line.split(\'#\', 1)[0].strip()\n        if not value:\n            continue\n        if not re.fullmatch(r\'[0-9]+\', value):\n            raise ValueError(f"Invalid snapshot policy line: {line!r}")\n        steps.add(int(value))\n    if not steps:\n        raise ValueError(f"Empty snapshot policy: {path}")\n    return steps\n\n\ndef latest_step(full):\n    text = (full / \'latest_checkpointed_iteration.txt\').read_text().strip()\n    if not re.fullmatch(r\'[0-9]+\', text):\n        raise ValueError(\'Checkpoint tracker is empty/invalid; no pruning this sweep\')\n    return int(text)\n\n\ndef committed_dirs(full, latest):\n    records = []\n    for path in full.iterdir():\n        match = re.fullmatch(r\'global_step_(0|[1-9][0-9]*)\', path.name)\n        if match and int(match[1]) <= latest:\n            if path.is_symlink() or not path.is_dir():\n                raise ValueError(f"Unexpected checkpoint path: {path}")\n            records.append((int(match[1]), path))\n    return sorted(records)\n\n\ndef component_directory(actor, manifest, key):\n    entry = manifest.get(\'contents\', {}).get(key)\n    if not isinstance(entry, dict) or not isinstance(entry.get(\'path\'), str):\n        raise ValueError(f"Manifest component {key} missing: {actor}")\n    relative = Path(entry[\'path\'])\n    if relative.is_absolute() or \'..\' in relative.parts:\n        raise ValueError(f"Unsafe component path: {relative}")\n    directory = (actor / relative).resolve(strict=True)\n    if actor.resolve() not in directory.parents or not directory.is_dir():\n        raise ValueError(f"Component outside actor directory: {directory}")\n    return directory, entry\n\n\ndef state_payload(directory, kind):\n    payloads = []\n    for base, dirs, files in os.walk(directory, followlinks=False):\n        for name in dirs + files:\n            path = Path(base) / name\n            if path.is_symlink():\n                raise ValueError(f"Unexpected state symlink: {path}")\n        payloads.extend(Path(base) / n for n in files if Path(n).suffix in (\'.pt\', \'.distcp\', \'.bin\'))\n    if not payloads or any(p.stat().st_size == 0 for p in payloads):\n        raise ValueError(f"Missing/empty {kind} payloads: {directory}")\n    if kind == \'optimizer\' and all(p.name == \'common.pt\' for p in payloads):\n        raise ValueError(f"Optimizer shard payloads missing: {directory}")\n\n\ndef verify_full(directory, step):\n    data = directory / \'data.pt\'\n    if not data.is_file() or data.stat().st_size == 0:\n        raise ValueError(f"Dataloader state missing: {directory}")\n    actor = directory / \'actor\'\n    manifest = json_object(actor / \'ckpt_contents.json\')\n    if manifest.get(\'global_step\') != step:\n        raise ValueError(f"Checkpoint manifest step mismatch: {directory}")\n    if not {\'model\', \'optimizer\', \'extra\'} <= set(manifest.get(\'save_contents\', [])):\n        raise ValueError(f"Full resume state not declared: {directory}")\n    model, entry = component_directory(actor, manifest, \'model\')\n    if entry.get(\'format\') == \'huggingface\':\n        inspect_weights(model)\n    else:\n        state_payload(model, \'model\')\n    for key in (\'optimizer\', \'lr_scheduler\', \'rng_state\'):\n        component, _ = component_directory(actor, manifest, key)\n        state_payload(component, key)\n\n\ndef verify_published(path, step):\n    if path.is_symlink():\n        raise ValueError(f"Snapshot directory is a symlink: {path}")\n    verify_snapshot(path)\n    metadata = dict(line.split(\'=\', 1) for line in (path / \'.snapshot_meta\').read_text().splitlines() if \'=\' in line)\n    if metadata.get(\'step_dir\') != f\'global_step_{step}\':\n        raise ValueError(f"Snapshot step metadata mismatch: {path}")\n    for base, dirs, files in os.walk(path, followlinks=False):\n        if any((Path(base) / name).is_symlink() for name in dirs + files):\n            raise ValueError(f"Snapshot must not depend on symlinks: {path}")\n\n\ndef copy_regular(source, target):\n    target.parent.mkdir(parents=True, exist_ok=True)\n    if target.is_symlink():\n        target.unlink()\n    shutil.copy2(source, target, follow_symlinks=True)\n\n\ndef harvest(step, directory, snap, full, policy, tokenizer):\n    destination = snap / f\'global_step_{step}\'\n    if destination.is_symlink():\n        raise ValueError(f\'Unexpected snapshot symlink: {destination}\')\n    if destination.exists():\n        # Preserve snapshots produced by the existing watcher or manual exporter.\n        if (destination / \'.complete\').is_file():\n            verify_published(destination, step)\n            return\n        # Repair the old window: rename succeeded but touch .complete did not.\n        inspect_weights(destination)\n        tokenizer_is_usable(destination)\n        metadata = dict(line.split(\'=\', 1) for line in (destination / \'.snapshot_meta\').read_text().splitlines() if \'=\' in line)\n        if metadata.get(\'step_dir\') != directory.name:\n            raise ValueError(f"Incomplete destination needs inspection: {destination}")\n        source = resolve_source(directory)\n        names, _ = inspect_weights(source)\n        if any((source / name).stat().st_size != (destination / name).stat().st_size for name in names):\n            raise ValueError(f"Incomplete destination size mismatch: {destination}")\n        for p in destination.rglob(\'*\'):\n            if p.is_symlink():\n                raise ValueError(f"Unexpected destination symlink: {p}")\n        atomic_text(destination / \'.complete\', \'verified by v5\\n\')\n        verify_published(destination, step)\n        log(\'PASS\', f\'repaired completion marker: {destination}\')\n        return\n\n    source = resolve_source(directory)\n    names, _ = inspect_weights(source)\n    stage = snap / f\'.global_step_{step}.incomplete\'\n    if stage.is_symlink():\n        raise ValueError(f"Unexpected staging symlink: {stage}")\n    stage.mkdir(exist_ok=True)\n    # An interrupted stage can be reused, but every required weight is recopied.\n    # This deliberately uses copies, preserving the original watcher\'s behavior.\n    for name in names:\n        log(\'INFO\', f\'harvesting selected step {step}: {name}\')\n        copy_regular(source / name, stage / name)\n    for name in (\'config.json\', \'model.safetensors.index.json\') + TOKENIZER_FILES:\n        candidate = source / name\n        if not candidate.is_file() and name in TOKENIZER_FILES and tokenizer:\n            candidate = tokenizer / name\n        if candidate.is_file():\n            copy_regular(candidate, stage / name)\n    inspect_weights(stage)\n    tokenizer_is_usable(stage)\n    if any((source / name).stat().st_size != (stage / name).stat().st_size for name in names):\n        raise ValueError(f\'Source/destination file size mismatch: step {step}\')\n    atomic_text(stage / \'.snapshot_meta\', f\'experiment_src={full}\\nstep_dir={directory.name}\\n\'\n                f\'policy_file={policy}\\nharvested_at={datetime.now().astimezone().isoformat()}\\n\')\n    atomic_text(stage / \'.complete\', \'v5\\n\')\n    verify_published(stage, step)\n    if destination.exists():\n        raise ValueError(f\'Destination appeared during export: {destination}\')\n    # The completion marker moves together with the already verified directory.\n    stage.rename(destination)\n    log(\'PASS\', f\'complete: {destination}\')\n\n\ndef prune(full, snap, selected, latest, keep):\n    records = committed_dirs(full, latest)\n    if len(records) <= keep:\n        log(\'INFO\', f\'nothing to prune; found={len(records)} keep={keep}\')\n        return\n    retained = records[-keep:]\n    if retained[-1][0] != latest:\n        raise ValueError(f\'Latest committed full checkpoint {latest} missing; no pruning\')\n    # Never count an incomplete newer directory as one of the retained resumes.\n    for step, path in retained:\n        verify_full(path, step)\n    candidates = records[:-keep]\n    for step, _ in candidates:\n        if step in selected:\n            verify_published(snap / f\'global_step_{step}\', step)\n    identities = {p: (p.stat().st_dev, p.stat().st_ino) for _, p in candidates}\n    for step, path in candidates:\n        if latest_step(full) < latest:\n            raise ValueError(\'Tracker moved backwards; pruning stopped\')\n        if path.is_symlink() or (path.stat().st_dev, path.stat().st_ino) != identities[path]:\n            raise ValueError(f\'Checkpoint directory changed: {path}\')\n        log(\'INFO\', f\'deleting committed full checkpoint step {step}\')\n        shutil.rmtree(path)\n    log(\'PASS\', f\'prune complete: latest={latest} deleted={len(candidates)} \'\n        f\'kept={",".join(str(s) for s, _ in retained)}\')\n\n\ndef sweep(args):\n    # Same order as the previously delivered manual cleanup/export scripts.\n    with ExitStack() as stack:\n        for path, label in ((args.full / \'.manual_full_cleanup.lock\', \'Manual cleanup\'),\n                            (args.snap / \'.manual_weight_export.lock\', \'Manual export\'),\n                            (args.full / \'.full_ckpt_prune.lock\', \'Pruner\')):\n            stack.enter_context(locked(path, label))\n        if not (args.full / \'latest_checkpointed_iteration.txt\').exists():\n            log(\'INFO\', \'no committed checkpoint tracker yet\')\n            return\n        latest = latest_step(args.full)\n        selected = policy_steps(args.policy)\n        if args.mode != \'prune\':\n            for step, directory in committed_dirs(args.full, latest):\n                if step in selected:\n                    harvest(step, directory, args.snap, args.full, args.policy, args.tokenizer)\n        # A failed harvest raises before reaching deletion.\n        if args.mode != \'export\':\n            prune(args.full, args.snap, selected, latest, args.keep)\n        log(\'PASS\', f\'sweep complete: committed={latest} keep={args.keep}\')\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument(\'mode\', choices=(\'watch\', \'oneshot\', \'export\', \'prune\'))\n    parser.add_argument(\'full\', type=Path)\n    parser.add_argument(\'snap\', type=Path)\n    parser.add_argument(\'policy\', type=Path)\n    parser.add_argument(\'value\', nargs=\'?\', default=None)\n    args = parser.parse_args()\n    args.keep = int(args.value if args.mode == \'prune\' and args.value else os.environ.get(\'KEEP_FULL_CKPTS\', \'2\'))\n    args.poll = float(args.value if args.mode != \'prune\' and args.value else \'30\')\n    if args.keep < 1 or args.poll <= 0:\n        raise ValueError(\'KEEP must be >= 1 and poll interval must be > 0\')\n    for path in (args.full, args.snap):\n        if path.is_symlink():\n            raise ValueError(f\'Unexpected directory symlink: {path}\')\n    args.full = args.full.resolve(strict=True)\n    args.snap.mkdir(parents=True, exist_ok=True)\n    args.snap = args.snap.resolve(strict=True)\n    root = Path(os.environ[\'EXPECTED_FULL_CKPT_ROOT\']).resolve(strict=True)\n    if args.full.parent != root:\n        raise ValueError(f\'Expected a direct experiment subdirectory of {root}\')\n    if args.full == args.snap or args.full in args.snap.parents or args.snap in args.full.parents:\n        raise ValueError(\'Full checkpoints and snapshots must be separate directories\')\n    policy_steps(args.policy)\n    args.tokenizer = Path(os.environ[\'TOKENIZER_FALLBACK_DIR\']) if os.environ.get(\'TOKENIZER_FALLBACK_DIR\') else None\n    for path in (args.full, args.snap):\n        reject_local_only_lustre_locks(path)\n    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):\n        signal.signal(number, lambda signum, _frame: sys.exit(128 + signum))\n    with locked(args.snap / \'.watcher.lock\', \'Watcher\'):\n        migrate_legacy_locks(args.full, args.snap)\n        # Do not keep a PBS training lock alive after its parent exits.\n        try:\n            training_fd = int(os.environ.get(\'RG_TRAINING_LOCK_FD\', \'9\'))\n            if os.path.samestat(os.fstat(training_fd), (args.full / \'.training.lock\').stat()):\n                os.close(training_fd)\n        except OSError:\n            pass\n        ready = os.environ.get(\'WATCHER_READY_FILE\')\n        if ready:\n            atomic_text(Path(ready), str(os.getpid()) + \'\\n\')\n        log(\'READY\', f\'pid={os.getpid()} host={socket.gethostname()} mode={args.mode}\')\n        failures = 0\n        while True:\n            try:\n                sweep(args)\n                failures = 0\n            except (OSError, ValueError, KeyError, TypeError) as error:\n                failures += 1\n                log(\'ERROR\', f\'sweep failed ({failures}/3); no further pruning: {error}\')\n                if args.mode != \'watch\' or failures >= 3:\n                    return 74\n            if args.mode != \'watch\':\n                return 0\n            time.sleep(args.poll)\n\n\nif __name__ == \'__main__\':\n    try:\n        raise SystemExit(main())\n    except (OSError, ValueError, KeyError, TypeError) as error:\n        log(\'ERROR\', str(error))\n        raise SystemExit(74)\n', 'scripts/snapshot_watcher_v5_selected.sh': '#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\nexec python3 "$SCRIPT_DIR/rg_checkpoint_maintenance_v5.py" "$@"\n', 'scripts/prune_full_checkpoints_v5_selected.sh': '#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\nexec python3 "$SCRIPT_DIR/rg_checkpoint_maintenance_v5.py" prune "$@"\n'}
WATCH_BLOCK = '# RG_SNAPSHOT_WATCHER_FIX_V5\nWATCHER_LOG=$PROJ/logs/snapshot_watcher_${EXPERIMENT_NAME}.log\nWATCHER_PID=\nTRAIN_PID=\nEXIT_REASON=normal\ncommand -v setsid >/dev/null || { echo "[FAIL] setsid is required" >&2; exit 2; }\nWATCHER_READY_FILE=$(mktemp "$PROJ/logs/.watcher_ready.XXXXXXXX")\n\nstop_watcher() {\n  [[ -n "$WATCHER_PID" ]] || return 0\n  if kill -0 "$WATCHER_PID" 2>/dev/null; then\n    kill -TERM "$WATCHER_PID" 2>/dev/null || true\n    for ((rg_wait=0; rg_wait<10; rg_wait++)); do\n      kill -0 "$WATCHER_PID" 2>/dev/null || break\n      sleep 1\n    done\n    if kill -0 "$WATCHER_PID" 2>/dev/null; then\n      kill -KILL "$WATCHER_PID" 2>/dev/null || true\n    fi\n  fi\n  wait "$WATCHER_PID" 2>/dev/null || true\n  WATCHER_PID=\n}\n\ncleanup() {\n  local status=$?\n  trap - EXIT TERM INT HUP\n  set +x\n  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then\n    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true\n    for ((rg_wait=0; rg_wait<5; rg_wait++)); do\n      kill -0 "$TRAIN_PID" 2>/dev/null || break\n      sleep 1\n    done\n    kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true\n    wait "$TRAIN_PID" 2>/dev/null || true\n  fi\n  stop_watcher\n  rm -f -- "$WATCHER_READY_FILE"\n  echo "[EXIT] job=${PBS_JOBID:-manual} reason=$EXIT_REASON status=$status time=$(date --iso-8601=seconds)"\n  exit "$status"\n}\ntrap cleanup EXIT\ntrap \'EXIT_REASON=SIGTERM; exit 143\' TERM\ntrap \'EXIT_REASON=SIGINT; exit 130\' INT\ntrap \'EXIT_REASON=SIGHUP; exit 129\' HUP\n\n# fd 9 proves ownership of the training lock during one-time legacy migration.\n# The Python watcher closes its inherited fd 9 after migration.\nWATCHER_READY_FILE="$WATCHER_READY_FILE" \\\n  bash "$WATCHER" watch "$FULL_CKPT_DIR" "$SNAP_DIR" "$SNAPSHOT_POLICY" 30 >> "$WATCHER_LOG" 2>&1 &\nWATCHER_PID=$!\nfor ((rg_start=0; rg_start<20; rg_start++)); do\n  kill -0 "$WATCHER_PID" 2>/dev/null || break\n  [[ "$(cat "$WATCHER_READY_FILE")" == "$WATCHER_PID" ]] && break\n  sleep 1\ndone\nif ! kill -0 "$WATCHER_PID" 2>/dev/null || [[ "$(cat "$WATCHER_READY_FILE")" != "$WATCHER_PID" ]]; then\n  EXIT_REASON=watcher_start_failed\n  echo "[FAIL] snapshot watcher did not become ready; see $WATCHER_LOG" >&2\n  tail -n 15 "$WATCHER_LOG" >&2 || true\n  exit 74\nfi\necho "[PASS] snapshot watcher ready: pid=$WATCHER_PID log=$WATCHER_LOG"\n\n'
WAIT_BLOCK = ' &\nTRAIN_PID=$!\nset +x\nwhile kill -0 "$TRAIN_PID" 2>/dev/null; do\n  if ! kill -0 "$WATCHER_PID" 2>/dev/null; then\n    EXIT_REASON=watcher_died\n    echo "[FAIL] snapshot watcher exited during training; see $WATCHER_LOG" >&2\n    tail -n 15 "$WATCHER_LOG" >&2 || true\n    exit 74\n  fi\n  sleep 5 &\n  wait $! || true\ndone\nif wait "$TRAIN_PID"; then\n  TRAIN_STATUS=0\nelse\n  TRAIN_STATUS=$?\nfi\nTRAIN_PID=\nif [[ "$TRAIN_STATUS" -ne 0 ]]; then\n  EXIT_REASON=training_failed\n  exit "$TRAIN_STATUS"\nfi\n\n# A final sweep is attempted only after successful training, never on a signal.\nstop_watcher\nif ! bash "$WATCHER" oneshot "$FULL_CKPT_DIR" "$SNAP_DIR" "$SNAPSHOT_POLICY" 30 >> "$WATCHER_LOG" 2>&1; then\n  EXIT_REASON=final_snapshot_failed\n  echo "[FAIL] final snapshot sweep failed; see $WATCHER_LOG" >&2\n  exit 74\nfi\n\n'
ORIGINAL_WATCH_BLOCK = 'WATCHER_LOG=$PROJ/logs/snapshot_watcher_${EXPERIMENT_NAME}.log\nbash "$WATCHER" watch "$FULL_CKPT_DIR" "$SNAP_DIR" "$SNAPSHOT_POLICY" 30 >> "$WATCHER_LOG" 2>&1 &\nWATCHER_PID=$!\nstop_watcher() {\n  local require_success=${1:-0}\n  if kill -0 "$WATCHER_PID" 2>/dev/null; then\n    kill "$WATCHER_PID" 2>/dev/null || true\n    wait "$WATCHER_PID" 2>/dev/null || true\n  fi\n  bash "$WATCHER" oneshot "$FULL_CKPT_DIR" "$SNAP_DIR" "$SNAPSHOT_POLICY" 30 >> "$WATCHER_LOG" 2>&1 || {\n    [[ "$require_success" -eq 0 ]] || return 1\n  }\n}\ncleanup() {\n  status=$?\n  trap - EXIT TERM INT HUP\n  stop_watcher 0 || true\n  exit "$status"\n}\ntrap cleanup EXIT TERM INT HUP\n\n'
JOB = 'jobs/0390_q3_rg_sixcat_r08_16k_from25_full500_k8.pbs'
EXPERIMENT = 'inst2507_rg_sixcat_r08_k8_full_v1_len16k'
MARKER = '# RG_SNAPSHOT_WATCHER_FIX_V5'


def log(level, message):
    print(f'[{level}] {message}', flush=True)


def patch_job(text):
    if MARKER in text:
        return text
    if text.count(ORIGINAL_WATCH_BLOCK) != 1:
        raise ValueError('PBS watcher/cleanup block differs from the uploaded version. '
                         'No files changed; provide the current PBS for an exact patch.')
    for old in ('scripts/snapshot_watcher_v4_selected.sh',
                'scripts/prune_full_checkpoints_v4_selected.sh'):
        if text.count(old) != 1:
            raise ValueError(f'Expected one PBS reference to {old}; no files changed.')
        text = text.replace(old, old.replace('_v4_', '_v5_'))
    text = text.replace(ORIGINAL_WATCH_BLOCK, WATCH_BLOCK + '\n')
    needle = 'bash "$LAUNCHER" \\\n'
    if text.count(needle) != 1:
        raise ValueError('Cannot locate exactly one training launch; no files changed.')
    text = text.replace(needle, 'setsid bash "$LAUNCHER" \\\n')
    old_tail = '\n\ntrap - EXIT TERM INT HUP\nstop_watcher 1\nset +x\n'
    if text.count(old_tail) != 1:
        raise ValueError('Unknown PBS finalization block; no files changed.')
    text = text.replace(old_tail, WAIT_BLOCK + '\n')
    return text


def atomic_write(path, data, mode):
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def install(project):
    job = project / JOB
    original = job.read_text()
    patched = patch_job(original)
    desired = dict(PAYLOADS)
    desired[JOB] = patched
    # Validate all payloads before writing any project file.
    for name, content in desired.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
        else:
            subprocess.run(['bash', '-n'], input=content, text=True, check=True)
    changes = []
    for name, content in desired.items():
        path = project / name
        if path.is_symlink():
            raise ValueError(f'Refusing to replace a symlink: {path}')
        old = path.read_bytes() if path.exists() else None
        new = content.encode()
        if old == new:
            continue
        if name != JOB and old is not None:
            raise ValueError(f'An existing v5 file has local changes: {path}; no files changed.')
        changes.append((path, old, new, path.stat().st_mode & 0o777 if path.exists() else 0o750))
    if not changes:
        log('PASS', 'v5 repair is already installed')
        return
    backup = project / 'backups' / ('snapshot_watcher_fix_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    for path, old, _, mode in changes:
        if old is not None:
            target = backup / path.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old)
            target.chmod(mode)
    done = []
    try:
        # Publish the PBS last, after its new dependencies are all in place.
        for path, old, new, mode in changes:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(path, new, mode)
            done.append((path, old, mode))
    except BaseException:
        for path, old, mode in reversed(done):
            if old is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, old, mode)
        raise
    log('PASS', f'patched PBS: {job}')
    log('PASS', f'backup: {backup}')
    log('PASS', 'v5 watcher + retention installed; existing training settings and policy preserved')
    log('INFO', 'No job submitted/stopped and no checkpoints changed by installation.')
    log('INFO', 'The next newly submitted PBS job starts v5 automatically. Running/queued PBS copies are unchanged.')


def recover(project, tokenizer):
    full = project / 'ckpts/verl_full' / EXPERIMENT
    snap = project / 'ckpts/model_snapshots' / EXPERIMENT
    policy = project / 'config/reasoning_gym_curve_steps_16k_from25.txt'
    with (full / '.training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('DEFERRED', 'Training still holds its lock. Recovery was not started; the next newly submitted PBS will use v5.')
            return 0
        env = dict(os.environ, EXPECTED_FULL_CKPT_ROOT=str(full.parent), KEEP_FULL_CKPTS='2',
                   TOKENIZER_FALLBACK_DIR=str(tokenizer), RG_TRAINING_LOCK_FD=str(lock.fileno()))
        env.pop('WATCHER_READY_FILE', None)
        log('INFO', 'Recovering existing selected snapshots, then retaining the newest two full checkpoints.')
        result = subprocess.run([sys.executable, str(project / 'scripts/rg_checkpoint_maintenance_v5.py'),
                                 'oneshot', str(full), str(snap), str(policy), '30'],
                                env=env, pass_fds=(lock.fileno(),))
        return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-dir', type=Path, default=Path.cwd())
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--tokenizer-dir', type=Path)
    args = parser.parse_args()
    project = args.project_dir.resolve(strict=True)
    # Serializes repeated invocations without touching any training lock.
    with (project / '.snapshot_watcher_repair.lock').open('a+') as install_lock:
        fcntl.flock(install_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install(project)
    if args.recover:
        return recover(project, args.tokenizer_dir or project.parent.parent / 'models/Qwen3-30B-A3B-Instruct-2507')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        log('FAIL', str(error))
        raise SystemExit(1)
