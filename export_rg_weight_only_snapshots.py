#!/usr/bin/env python3
"""One-off HF snapshot export; no training/watcher/policy edits and no pruning.

Defaults to committed steps 25, 30, 35, 40, 45, 50 of the r08 16k experiment.
Safetensors are hard-linked by default (copied if cross-filesystem); small model
and tokenizer files are copied. Use --copy for independent weight copies.
Only Python's standard library is required. No model/optimizer is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

DEFAULT_PROJECT = Path("/groups/gcg51557/experiments/0390_rlsd/RLVR/verl_qwen3_30b_a3b_rlvr")
DEFAULT_MODEL = Path("/groups/gcg51557/experiments/0390_rlsd/models/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_EXPERIMENT = "inst2507_rg_sixcat_r08_k8_full_v1_len16k"
STEPS = (45, 50, 55, 60, 65, 70, 75, 80)
HF_LAYOUTS = ("actor/model/huggingface", "actor/huggingface")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json",
    "merges.txt", "chat_template.jinja",
)


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


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


def export_one(step: int, source: Path, destination: Path, full_dir: Path,
               tokenizer_dir: Path, force_copy: bool, requested: list[int]) -> None:
    weight_names, _ = inspect_weights(source)
    stage = Path(tempfile.mkdtemp(prefix=f".global_step_{step}.manual-", dir=destination.parent))
    linked = copied = 0
    try:
        # Deliberately select HF model files: no optimizer, RNG state or data.pt.
        for name in weight_names:
            source_file = (source / name).resolve(strict=True)
            target_file = stage / name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if not force_copy:
                try:
                    os.link(source_file, target_file)
                    linked += 1
                    continue
                except OSError as error:
                    if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
                        raise
            log("INFO", f"step {step}: copying weight shard {name}")
            shutil.copy2(source_file, target_file)
            copied += 1
        for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, stage / name)
        for name in TOKENIZER_FILES:
            candidate = source / name
            if not candidate.is_file():
                candidate = tokenizer_dir / name
            if candidate.is_file():
                shutil.copy2(candidate, stage / name)
        if not (stage / "generation_config.json").exists() and (tokenizer_dir / "generation_config.json").is_file():
            shutil.copy2(tokenizer_dir / "generation_config.json", stage / "generation_config.json")

        inspect_weights(stage)
        tokenizer_is_usable(stage)
        for name in weight_names:
            if (source / name).stat().st_size != (stage / name).stat().st_size:
                raise ValueError(f"Source/destination size mismatch: step {step}, {name}")
        (stage / ".snapshot_meta").write_text(
            f"experiment_src={full_dir}\nstep_dir=global_step_{step}\n"
            f"source_hf={source}\nselection=manual_explicit_steps\n"
            f"requested_steps={','.join(map(str, requested))}\n"
            f"harvested_at={datetime.now().astimezone().isoformat()}\n"
            f"hardlinked_weight_files={linked}\ncopied_weight_files={copied}\n",
            encoding="utf-8",
        )
        (stage / ".complete").touch()
        if destination.exists():
            raise FileExistsError(f"Destination appeared during export; preserved: {destination}")
        stage.rename(destination)
        log("PASS", f"step {step}: {destination} (weight shards: {linked} linked, {copied} copied)")
    finally:
        # Only our own unpublished temporary directory may be removed.
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--steps", type=int, nargs="+", default=STEPS)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--copy", action="store_true", help="Copy weight bytes instead of hard-linking them.")
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in ("", ".", ".."):
        parser.error("--experiment must be a directory name, not a path")
    steps = sorted(set(args.steps))
    if not steps or steps[0] < 1:
        parser.error("--steps must contain positive integers")
    project = args.project_dir.resolve()
    full_dir = project / "ckpts/verl_full" / args.experiment
    snap_dir = project / "ckpts/model_snapshots" / args.experiment
    tracker = full_dir / "latest_checkpointed_iteration.txt"
    latest = int(tracker.read_text().strip())
    if max(steps) > latest:
        raise ValueError(f"Requested step {max(steps)} is newer than committed tracker {latest}; nothing exported.")
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Independent one-off exporter lock. Leave the legacy watcher lock untouched.
    with (snap_dir / ".manual_weight_export.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another manual exporter is active for this experiment.") from None
        plans = []
        # Validate all requested sources / existing snapshots before publishing any.
        for step in steps:
            destination = snap_dir / f"global_step_{step}"
            if destination.exists():
                verify_snapshot(destination)
                log("SKIP", f"step {step}: existing complete snapshot verified")
                continue
            source = resolve_source(full_dir / f"global_step_{step}")
            plans.append((step, source, destination))
            log("READY", f"step {step}: {source}")
        for step, source, destination in plans:
            export_one(step, source, destination, full_dir, args.tokenizer_dir,
                       args.copy, steps)
        for step in steps:
            verify_snapshot(snap_dir / f"global_step_{step}")
        log("PASS", f"All requested weight-only snapshots ready: {', '.join(map(str, steps))}")
        log("INFO", "No full checkpoints deleted; no training scripts, watcher locks or snapshot policies changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        log("FAIL", str(error))
        raise SystemExit(1)
