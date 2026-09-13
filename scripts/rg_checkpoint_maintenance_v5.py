#!/usr/bin/env python3
"""Checkpoint maintenance v5: process locks, atomic HF snapshots, guarded pruning.
Only the standard library is used. Weight validation reads headers, not tensors.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import struct
import sys
import tempfile
import time

HF_LAYOUTS = ("actor/model/huggingface", "actor/huggingface")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    "chat_template.jinja", "generation_config.json")

def log(level, message):
    print(f"[watcher {datetime.now().astimezone().isoformat(timespec='seconds')}] {level}: {message}", flush=True)

def json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check_tensor_file(path: Path) -> set[str]:
    """Check safetensors header/offsets/file length, without reading tensor data.

    This is a structural check, not a full checksum of weight values.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Missing safetensors header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100_000_000 or 8 + length > size:
            raise ValueError(f"Invalid/truncated safetensors header: {path}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"No tensors in {path}")
    intervals = []
    for name, value in tensors.items():
        offsets = value.get("data_offsets") if isinstance(value, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(item) is not int for item in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError(f"Invalid offsets for tensor {name!r}: {path}")
        intervals.append(tuple(offsets))
    end = 0
    for start, next_end in sorted(intervals):
        if start != end:
            raise ValueError(f"Non-contiguous/overlapping tensor data: {path}")
        end = next_end
    if 8 + length + end != size:
        raise ValueError(f"Tensor data length does not match header: {path}")
    return set(tensors)


def inspect_weights(directory: Path) -> tuple[list[str], dict]:
    config = json_object(directory / "config.json")
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json_object(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Missing/empty weight_map: {index_path}")
        names = set()
        for name in weight_map.values():
            if (not isinstance(name, str) or Path(name).is_absolute()
                    or ".." in Path(name).parts or not name.endswith(".safetensors")):
                raise ValueError(f"Invalid model shard name in {index_path}: {name!r}")
            names.add(name)
        names = sorted(names)
    else:
        names = ["model.safetensors"]
        weight_map = None
    headers = {name: check_tensor_file(directory / name) for name in names}
    if weight_map is not None:
        for tensor, shard in weight_map.items():
            if tensor not in headers[shard]:
                raise ValueError(f"Index tensor {tensor!r} is absent from {directory / shard}")
    return names, config


def tokenizer_is_usable(directory: Path) -> None:
    json_object(directory / "tokenizer_config.json")
    if (directory / "tokenizer.json").is_file():
        json_object(directory / "tokenizer.json")
    elif not any((directory / name).is_file() for name in ("tokenizer.model", "spiece.model")):
        if not all((directory / name).is_file() for name in ("vocab.json", "merges.txt")):
            raise ValueError(f"Tokenizer data files are missing: {directory}")


def verify_snapshot(directory: Path) -> None:
    inspect_weights(directory)
    tokenizer_is_usable(directory)
    if not (directory / ".complete").is_file() or not (directory / ".snapshot_meta").is_file():
        raise ValueError(f"Existing snapshot lacks completion metadata: {directory}")


def resolve_source(step_dir: Path) -> Path:
    for layout in HF_LAYOUTS:
        directory = step_dir / layout
        if (directory / "config.json").is_file():
            try:
                inspect_weights(directory)
            except (OSError, ValueError) as error:
                log("WARN", f"Cannot use {directory}: {error}")
                continue
            return directory
    raise ValueError(
        f"No complete HF safetensors export in {step_dir}; expected one of {HF_LAYOUTS}. "
        "This script does not convert Megatron distributed shards. Full checkpoint retained."
    )


@contextmanager
def locked(path, label):
    # Never unlink lock files: unlinking would create two independent lock inodes.
    if path.is_symlink():
        raise ValueError(f"Refusing symlink lock: {path}")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"{label} is active: {path}") from None
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "job": os.environ.get("PBS_JOBID"), "started": time.time()}, handle)
        handle.flush()
        yield


def reject_local_only_lustre_locks(path):
    # localflock cannot exclude a process on a different compute/login node.
    entries = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, extra = left.split(), right.split()
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
        if path == mount or mount in path.parents:
            entries.append((len(mount.parts), extra[0], fields[5] + ',' + extra[2]))
    if entries:
        _, fs, options = max(entries)
        if fs == 'lustre' and {'localflock', 'noflock'} & set(options.split(',')):
            raise ValueError(f"Shared cross-node locking is unavailable on {path}: {options}")


def atomic_text(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def migrate_legacy_locks(full, snap):
    """Exclude old PBS descendants using the training lock; preserve old dirs.

    The old PBS passes fd 9 to its watcher/pruner. A new PBS can acquire that
    lock only once those old processes have exited. A login-node invocation
    instead takes the training lock itself for the migration.
    """
    marker = snap / '.watcher_v5_migration.json'
    if marker.is_file():
        return
    training_lock = full / '.training.lock'
    training_fd = int(os.environ.get('RG_TRAINING_LOCK_FD', '9'))
    with ExitStack() as stack:
        inherited = False
        try:
            inherited = os.path.samestat(os.fstat(training_fd), training_lock.stat())
        except OSError:
            pass
        if inherited:
            fcntl.flock(training_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            stack.enter_context(locked(training_lock, 'Training (legacy-lock migration deferred)'))
        # These guard directories also prevent an old v4 script from starting.
        # No PID-only test is used: PIDs have no cross-node identity.
        for directory in (snap / '.watcher_lock', full / '.full_ckpt_prune_lock'):
            if directory.is_symlink():
                raise ValueError(f"Unexpected legacy lock symlink: {directory}")
            directory.mkdir(exist_ok=True)
            atomic_text(directory / 'v5_guard', 'Legacy directory retained; v5 uses flock.\n')
        atomic_text(marker, json.dumps({'host': socket.gethostname(), 'pid': os.getpid(),
                                      'migrated_at': time.time()}) + '\n')
        log('PASS', 'legacy locks migrated under the training lock; no lock directories deleted')


def policy_steps(path):
    steps = set()
    for line in path.read_text().splitlines():
        value = line.split('#', 1)[0].strip()
        if not value:
            continue
        if not re.fullmatch(r'[0-9]+', value):
            raise ValueError(f"Invalid snapshot policy line: {line!r}")
        steps.add(int(value))
    if not steps:
        raise ValueError(f"Empty snapshot policy: {path}")
    return steps


def latest_step(full):
    text = (full / 'latest_checkpointed_iteration.txt').read_text().strip()
    if not re.fullmatch(r'[0-9]+', text):
        raise ValueError('Checkpoint tracker is empty/invalid; no pruning this sweep')
    return int(text)


def committed_dirs(full, latest):
    records = []
    for path in full.iterdir():
        match = re.fullmatch(r'global_step_(0|[1-9][0-9]*)', path.name)
        if match and int(match[1]) <= latest:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Unexpected checkpoint path: {path}")
            records.append((int(match[1]), path))
    return sorted(records)


def component_directory(actor, manifest, key):
    entry = manifest.get('contents', {}).get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
        raise ValueError(f"Manifest component {key} missing: {actor}")
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f"Unsafe component path: {relative}")
    directory = (actor / relative).resolve(strict=True)
    if actor.resolve() not in directory.parents or not directory.is_dir():
        raise ValueError(f"Component outside actor directory: {directory}")
    return directory, entry


def state_payload(directory, kind):
    payloads = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f"Unexpected state symlink: {path}")
        payloads.extend(Path(base) / n for n in files if Path(n).suffix in ('.pt', '.distcp', '.bin'))
    if not payloads or any(p.stat().st_size == 0 for p in payloads):
        raise ValueError(f"Missing/empty {kind} payloads: {directory}")
    if kind == 'optimizer' and all(p.name == 'common.pt' for p in payloads):
        raise ValueError(f"Optimizer shard payloads missing: {directory}")


def verify_full(directory, step):
    data = directory / 'data.pt'
    if not data.is_file() or data.stat().st_size == 0:
        raise ValueError(f"Dataloader state missing: {directory}")
    actor = directory / 'actor'
    manifest = json_object(actor / 'ckpt_contents.json')
    if manifest.get('global_step') != step:
        raise ValueError(f"Checkpoint manifest step mismatch: {directory}")
    if not {'model', 'optimizer', 'extra'} <= set(manifest.get('save_contents', [])):
        raise ValueError(f"Full resume state not declared: {directory}")
    model, entry = component_directory(actor, manifest, 'model')
    if entry.get('format') == 'huggingface':
        inspect_weights(model)
    else:
        state_payload(model, 'model')
    for key in ('optimizer', 'lr_scheduler', 'rng_state'):
        component, _ = component_directory(actor, manifest, key)
        state_payload(component, key)


def verify_published(path, step):
    if path.is_symlink():
        raise ValueError(f"Snapshot directory is a symlink: {path}")
    verify_snapshot(path)
    metadata = dict(line.split('=', 1) for line in (path / '.snapshot_meta').read_text().splitlines() if '=' in line)
    if metadata.get('step_dir') != f'global_step_{step}':
        raise ValueError(f"Snapshot step metadata mismatch: {path}")
    for base, dirs, files in os.walk(path, followlinks=False):
        if any((Path(base) / name).is_symlink() for name in dirs + files):
            raise ValueError(f"Snapshot must not depend on symlinks: {path}")


def copy_regular(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    shutil.copy2(source, target, follow_symlinks=True)


def harvest(step, directory, snap, full, policy, tokenizer):
    destination = snap / f'global_step_{step}'
    if destination.is_symlink():
        raise ValueError(f'Unexpected snapshot symlink: {destination}')
    if destination.exists():
        # Preserve snapshots produced by the existing watcher or manual exporter.
        if (destination / '.complete').is_file():
            verify_published(destination, step)
            return
        # Repair the old window: rename succeeded but touch .complete did not.
        inspect_weights(destination)
        tokenizer_is_usable(destination)
        metadata = dict(line.split('=', 1) for line in (destination / '.snapshot_meta').read_text().splitlines() if '=' in line)
        if metadata.get('step_dir') != directory.name:
            raise ValueError(f"Incomplete destination needs inspection: {destination}")
        source = resolve_source(directory)
        names, _ = inspect_weights(source)
        if any((source / name).stat().st_size != (destination / name).stat().st_size for name in names):
            raise ValueError(f"Incomplete destination size mismatch: {destination}")
        for p in destination.rglob('*'):
            if p.is_symlink():
                raise ValueError(f"Unexpected destination symlink: {p}")
        atomic_text(destination / '.complete', 'verified by v5\n')
        verify_published(destination, step)
        log('PASS', f'repaired completion marker: {destination}')
        return

    source = resolve_source(directory)
    names, _ = inspect_weights(source)
    stage = snap / f'.global_step_{step}.incomplete'
    if stage.is_symlink():
        raise ValueError(f"Unexpected staging symlink: {stage}")
    stage.mkdir(exist_ok=True)
    # An interrupted stage can be reused, but every required weight is recopied.
    # This deliberately uses copies, preserving the original watcher's behavior.
    for name in names:
        log('INFO', f'harvesting selected step {step}: {name}')
        copy_regular(source / name, stage / name)
    for name in ('config.json', 'model.safetensors.index.json') + TOKENIZER_FILES:
        candidate = source / name
        if not candidate.is_file() and name in TOKENIZER_FILES and tokenizer:
            candidate = tokenizer / name
        if candidate.is_file():
            copy_regular(candidate, stage / name)
    inspect_weights(stage)
    tokenizer_is_usable(stage)
    if any((source / name).stat().st_size != (stage / name).stat().st_size for name in names):
        raise ValueError(f'Source/destination file size mismatch: step {step}')
    atomic_text(stage / '.snapshot_meta', f'experiment_src={full}\nstep_dir={directory.name}\n'
                f'policy_file={policy}\nharvested_at={datetime.now().astimezone().isoformat()}\n')
    atomic_text(stage / '.complete', 'v5\n')
    verify_published(stage, step)
    if destination.exists():
        raise ValueError(f'Destination appeared during export: {destination}')
    # The completion marker moves together with the already verified directory.
    stage.rename(destination)
    log('PASS', f'complete: {destination}')


def prune(full, snap, selected, latest, keep):
    records = committed_dirs(full, latest)
    if len(records) <= keep:
        log('INFO', f'nothing to prune; found={len(records)} keep={keep}')
        return
    retained = records[-keep:]
    if retained[-1][0] != latest:
        raise ValueError(f'Latest committed full checkpoint {latest} missing; no pruning')
    # Never count an incomplete newer directory as one of the retained resumes.
    for step, path in retained:
        verify_full(path, step)
    candidates = records[:-keep]
    for step, _ in candidates:
        if step in selected:
            verify_published(snap / f'global_step_{step}', step)
    identities = {p: (p.stat().st_dev, p.stat().st_ino) for _, p in candidates}
    for step, path in candidates:
        if latest_step(full) < latest:
            raise ValueError('Tracker moved backwards; pruning stopped')
        if path.is_symlink() or (path.stat().st_dev, path.stat().st_ino) != identities[path]:
            raise ValueError(f'Checkpoint directory changed: {path}')
        log('INFO', f'deleting committed full checkpoint step {step}')
        shutil.rmtree(path)
    log('PASS', f'prune complete: latest={latest} deleted={len(candidates)} '
        f'kept={",".join(str(s) for s, _ in retained)}')


def sweep(args):
    # Same order as the previously delivered manual cleanup/export scripts.
    with ExitStack() as stack:
        for path, label in ((args.full / '.manual_full_cleanup.lock', 'Manual cleanup'),
                            (args.snap / '.manual_weight_export.lock', 'Manual export'),
                            (args.full / '.full_ckpt_prune.lock', 'Pruner')):
            stack.enter_context(locked(path, label))
        if not (args.full / 'latest_checkpointed_iteration.txt').exists():
            log('INFO', 'no committed checkpoint tracker yet')
            return
        latest = latest_step(args.full)
        selected = policy_steps(args.policy)
        if args.mode != 'prune':
            for step, directory in committed_dirs(args.full, latest):
                if step in selected:
                    harvest(step, directory, args.snap, args.full, args.policy, args.tokenizer)
        # A failed harvest raises before reaching deletion.
        if args.mode != 'export':
            prune(args.full, args.snap, selected, latest, args.keep)
        log('PASS', f'sweep complete: committed={latest} keep={args.keep}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('watch', 'oneshot', 'export', 'prune'))
    parser.add_argument('full', type=Path)
    parser.add_argument('snap', type=Path)
    parser.add_argument('policy', type=Path)
    parser.add_argument('value', nargs='?', default=None)
    args = parser.parse_args()
    args.keep = int(args.value if args.mode == 'prune' and args.value else os.environ.get('KEEP_FULL_CKPTS', '2'))
    args.poll = float(args.value if args.mode != 'prune' and args.value else '30')
    if args.keep < 1 or args.poll <= 0:
        raise ValueError('KEEP must be >= 1 and poll interval must be > 0')
    for path in (args.full, args.snap):
        if path.is_symlink():
            raise ValueError(f'Unexpected directory symlink: {path}')
    args.full = args.full.resolve(strict=True)
    args.snap.mkdir(parents=True, exist_ok=True)
    args.snap = args.snap.resolve(strict=True)
    root = Path(os.environ['EXPECTED_FULL_CKPT_ROOT']).resolve(strict=True)
    if args.full.parent != root:
        raise ValueError(f'Expected a direct experiment subdirectory of {root}')
    if args.full == args.snap or args.full in args.snap.parents or args.snap in args.full.parents:
        raise ValueError('Full checkpoints and snapshots must be separate directories')
    policy_steps(args.policy)
    args.tokenizer = Path(os.environ['TOKENIZER_FALLBACK_DIR']) if os.environ.get('TOKENIZER_FALLBACK_DIR') else None
    for path in (args.full, args.snap):
        reject_local_only_lustre_locks(path)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda signum, _frame: sys.exit(128 + signum))
    with locked(args.snap / '.watcher.lock', 'Watcher'):
        migrate_legacy_locks(args.full, args.snap)
        # Do not keep a PBS training lock alive after its parent exits.
        try:
            training_fd = int(os.environ.get('RG_TRAINING_LOCK_FD', '9'))
            if os.path.samestat(os.fstat(training_fd), (args.full / '.training.lock').stat()):
                os.close(training_fd)
        except OSError:
            pass
        ready = os.environ.get('WATCHER_READY_FILE')
        if ready:
            atomic_text(Path(ready), str(os.getpid()) + '\n')
        log('READY', f'pid={os.getpid()} host={socket.gethostname()} mode={args.mode}')
        failures = 0
        while True:
            try:
                sweep(args)
                failures = 0
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures += 1
                log('ERROR', f'sweep failed ({failures}/3); no further pruning: {error}')
                if args.mode != 'watch' or failures >= 3:
                    return 74
            if args.mode != 'watch':
                return 0
            time.sleep(args.poll)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        log('ERROR', str(error))
        raise SystemExit(74)
