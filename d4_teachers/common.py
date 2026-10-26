"""CPU-only, durable control primitives shared by the teacher commands."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import time

from d4_training_continuation.common import (
    below,
    digest,
    files,
    need,
    now,
    read,
    safe_id,
    verify_files,
    write,
    active,
)

__all__ = [
    "below",
    "digest",
    "files",
    "need",
    "now",
    "read",
    "safe_id",
    "verify_files",
    "write",
    "active",
    "file_sha",
    "model_inventory",
    "lock",
    "is_fenced",
    "age_seconds",
    "allocated_storage",
]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_inventory(path):
    """HF assets, excluding download caches/zero-byte locks and transient logs."""
    root, result = Path(path), {}
    suffixes = {
        ".json",
        ".safetensors",
        ".bin",
        ".model",
        ".txt",
        ".tiktoken",
        ".jinja",
        ".py",
    }
    for p in sorted(root.rglob("*")):
        relative = p.relative_to(root)
        if not p.is_file() or any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            continue
        if p.suffix not in suffixes:
            continue
        stat = p.stat()
        need(stat.st_size > 0, "Empty HF asset: " + str(p))
        result[str(relative)] = dict(
            bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=file_sha(p) if stat.st_size <= 1048576 else None,
            symlink_target=str(p.resolve()) if p.is_symlink() else None,
        )
    need(
        "config.json" in result
        and any(p.endswith((".safetensors", ".bin")) for p in result),
        "HF model assets missing",
    )
    return result


@contextmanager
def lock(path, wait=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        yield


def is_fenced(campaign, domain=None):
    campaign = Path(campaign)
    return (campaign / "STOP.json").exists() or bool(
        domain and (campaign / "runs" / domain / "STOP.json").exists()
    )


def age_seconds(timestamp):
    return (
        datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)
    ).total_seconds()


def allocated_storage(root):
    """Union of allocated blocks, counting hard links once; never follows symlinks."""
    seen = set()
    blocks = count = 0
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in names:
            p = Path(parent) / name
            if p.is_symlink():
                continue
            try:
                s = p.stat()
            except FileNotFoundError:
                continue  # A concurrent retention pass can remove a file.
            inode = (s.st_dev, s.st_ino)
            if inode not in seen:
                seen.add(inode)
                blocks += s.st_blocks * 512
                count += 1
    return dict(
        allocated_bytes=blocks, allocated_TiB=blocks / 2**40, unique_files=count
    )
