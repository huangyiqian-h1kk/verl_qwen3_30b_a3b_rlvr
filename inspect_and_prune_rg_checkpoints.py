#!/usr/bin/env python3
"""Inspect steps 25..50; optionally remove ONLY full steps 25/30/35/40.

Place next to export_rg_weight_only_snapshots.py. Default is inspection only.
--delete-old checks all snapshots and the retained full checkpoints first, then
removes the four explicitly requested directories. No training settings change.
Checks manifests, state-file inventory and HF structure; does not load optimizer
tensors or perform a training resume.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys

try:
    from export_rg_weight_only_snapshots import (
        DEFAULT_PROJECT, DEFAULT_EXPERIMENT, inspect_weights, verify_snapshot,
    )
except ImportError:
    raise SystemExit("Place export_rg_weight_only_snapshots.py next to this script.")


STEPS = (45, 50, 55, 60, 65, 70, 75, 80)
DELETE_STEPS = (45, 50, 55, 60, 65, 70, 75)
KEEP_STEPS = (80,)

def log(level, message):
    print(f"[{level}] {message}", flush=True)


def gib(size):
    return f"{size / 1024**3:.2f}"


def inventory(directory):
    """lstat only; no traversal into symlinked directories outside a checkpoint."""
    records = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                records.append((path, info))
    return records


def sizes(records):
    # Each directory is measured independently. Count each inode once within it.
    unique = {(info.st_dev, info.st_ino): info for _, info in records}
    return (sum(info.st_size for info in unique.values()),
            sum(info.st_blocks * 512 for info in unique.values()))


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def component_directory(actor, manifest, name):
    entry = manifest.get("contents", {}).get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"Missing {name} in {actor / 'ckpt_contents.json'}")
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unexpected {name} path: {relative}")
    directory = (actor / relative).resolve(strict=True)
    if not inside(directory, actor.resolve()) or not directory.is_dir():
        raise ValueError(f"Invalid {name} directory: {directory}")
    return directory, entry


def check_state_inventory(directory, name):
    records = inventory(directory)
    # .json or .metadata alone cannot stand in for saved state payloads.
    payloads = [(path, info) for path, info in records
                if path.suffix in (".pt", ".distcp", ".bin")]
    if not payloads or any(info.st_size <= 0 for _, info in payloads):
        raise ValueError(f"Missing/empty {name} state files: {directory}")
    if name == "optimizer" and not any(path.name != "common.pt" for path, _ in payloads):
        raise ValueError(f"Only common.pt remains, no optimizer shard payloads: {directory}")
    return records


def check_full(directory, step):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Not a regular checkpoint directory: {directory}")
    data = directory / "data.pt"
    if not data.is_file() or data.stat().st_size == 0:
        raise ValueError(f"Missing/empty dataloader state: {data}")
    actor = directory / "actor"
    manifest = json.loads((actor / "ckpt_contents.json").read_text())
    if manifest.get("global_step") != step:
        raise ValueError(f"Manifest step mismatch at {directory}")
    if not {"model", "optimizer", "extra"} <= set(manifest.get("save_contents", [])):
        raise ValueError(f"Manifest does not declare model + optimizer + extra: {directory}")
    model, model_info = component_directory(actor, manifest, "model")
    if model_info.get("format") == "huggingface":
        inspect_weights(model)
    else:
        check_state_inventory(model, "model")
    optimizer, _ = component_directory(actor, manifest, "optimizer")
    optimizer_records = check_state_inventory(optimizer, "optimizer")
    scheduler, _ = component_directory(actor, manifest, "lr_scheduler")
    check_state_inventory(scheduler, "lr_scheduler")
    extra, _ = component_directory(actor, manifest, "rng_state")
    check_state_inventory(extra, "rng_state")
    return optimizer_records


def check_snapshot(snapshot, step, delete_dirs):
    if snapshot.is_symlink():
        raise ValueError(f"Snapshot is a directory symlink, not an independent directory: {snapshot}")
    verify_snapshot(snapshot)
    metadata = {}
    for line in (snapshot / ".snapshot_meta").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    if metadata.get("step_dir") != f"global_step_{step}":
        raise ValueError(f"Snapshot metadata step mismatch: {snapshot}")
    for path in snapshot.rglob("*"):
        if path.is_symlink() and any(inside(path.resolve(), root) for root in delete_dirs):
            raise ValueError(f"Snapshot has a symlink into a checkpoint selected for removal: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--delete-old", action="store_true",
                        help="After all checks pass, delete full steps 25, 30, 35, 40 only.")
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in ("", ".", ".."):
        parser.error("--experiment must be one directory name")
    project = args.project_dir.resolve()
    full = project / "ckpts/verl_full" / args.experiment
    snapshots = project / "ckpts/model_snapshots" / args.experiment
    if not full.is_dir() or full.is_symlink() or not snapshots.is_dir() or snapshots.is_symlink():
        raise ValueError(f"Full and snapshot directories must exist under {project / 'ckpts'}")
    full = full.resolve()
    snapshots = snapshots.resolve()
    delete_dirs = [full / f"global_step_{step}" for step in DELETE_STEPS]

    # Serialize this one-off cleanup and the previously delivered manual exporter.
    # Do not touch the training lock, legacy watcher lock or retention policy.
    with (full / ".manual_full_cleanup.lock").open("a") as cleanup_lock, \
            (snapshots / ".manual_weight_export.lock").open("a") as export_lock:
        for handle in (cleanup_lock, export_lock):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another manual export/cleanup is active; nothing deleted.") from None
        tracker = full / "latest_checkpointed_iteration.txt"
        latest = int(tracker.read_text().strip())
        if latest < max(KEEP_STEPS):
            raise ValueError(f"Committed tracker is {latest}, below retained step 50; nothing deleted.")
        log("INFO", f"Full checkpoints: {full}")
        log("INFO", f"Committed tracker: {latest}; preserve full steps 45, 50 and any newer steps")
        print("step   logical_GiB   allocated_GiB   optimizer_GiB   snapshot   full_structure", flush=True)
        errors = []
        records_by_step = {}
        identities = {}
        for step in STEPS:
            directory = full / f"global_step_{step}"
            snapshot_ok = False
            try:
                check_snapshot(snapshots / f"global_step_{step}", step, delete_dirs)
                snapshot_ok = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"step {step} snapshot: {error}")
            if not directory.exists():
                if step in KEEP_STEPS:
                    errors.append(f"Required retained full checkpoint is missing: {directory}")
                print(f"{step:<6} {'-':>11} {'-':>15} {'-':>15} {'PASS' if snapshot_ok else 'FAIL':>10}   MISSING", flush=True)
                continue
            info = directory.lstat()
            identities[step] = (info.st_dev, info.st_ino)
            if directory.is_symlink():
                errors.append(f"Checkpoint directory is a symlink: {directory}")
                print(f"{step:<6} {'-':>11} {'-':>15} {'-':>15} {'PASS' if snapshot_ok else 'FAIL':>10}   FAIL", flush=True)
                continue
            records = inventory(directory)
            records_by_step[step] = records
            logical, allocated = sizes(records)
            full_ok, optimizer_size = False, "-"
            try:
                optimizer_records = check_full(directory, step)
                optimizer_size = gib(sizes(optimizer_records)[0])
                full_ok = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"step {step} full checkpoint: {error}")
            print(f"{step:<6} {gib(logical):>11} {gib(allocated):>15} {optimizer_size:>15} "
                  f"{'PASS' if snapshot_ok else 'FAIL':>10}   {'PASS' if full_ok else 'FAIL'}", flush=True)

        if errors:
            for error in errors:
                log("FAIL", error)
            raise ValueError("Checks failed. No checkpoint directories deleted.")
        pending = [step for step in DELETE_STEPS if step in records_by_step]
        link_counts = Counter()
        inodes = {}
        for step in pending:
            for _, info in records_by_step[step]:
                key = (info.st_dev, info.st_ino)
                link_counts[key] += 1
                inodes[key] = info
        # Only inodes losing every hard link can release their data blocks.
        reclaim = sum(info.st_blocks * 512 for key, info in inodes.items()
                      if link_counts[key] == info.st_nlink)
        log("INFO", f"Estimated reclaimable file blocks: {gib(reclaim)} GiB (other hard links remain allocated).")
        log("PASS", "Snapshot, completed-save manifest and state-file inventory checks passed; no optimizer tensors loaded.")
        if not args.delete_old:
            log("INFO", "Inspection only. Add --delete-old to remove full steps 25, 30, 35, 40 after these checks.")
            return 0
        if not pending:
            log("PASS", "Requested old full checkpoint directories are already absent.")
            return 0
        # Recheck the retained checkpoints and publication tracker before deleting.
        if int(tracker.read_text().strip()) < max(KEEP_STEPS):
            raise ValueError("Checkpoint tracker moved backwards; nothing deleted.")
        for step in KEEP_STEPS:
            check_full(full / f"global_step_{step}", step)
        for step in pending:
            directory = full / f"global_step_{step}"
            info = directory.lstat()
            if directory.is_symlink() or (info.st_dev, info.st_ino) != identities[step]:
                raise ValueError(f"Checkpoint directory changed since inspection: {directory}")
        for step in pending:
            directory = full / f"global_step_{step}"
            log("DELETE", str(directory))
            shutil.rmtree(directory)
        for step in STEPS:
            verify_snapshot(snapshots / f"global_step_{step}")
        log("PASS", f"Deleted full steps: {', '.join(map(str, pending))}. Full steps 45/50 and all six snapshots retained.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        log("FAIL", str(error))
        raise SystemExit(1)
