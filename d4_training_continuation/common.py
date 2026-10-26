"""Durable records and shared-filesystem leases; no GPU imports."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from . import BASE


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temp.open("x") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def acquire(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BaseException:
        handle.close()
        raise


@contextmanager
def locked(path):
    handle = acquire(path)
    try:
        yield
    finally:
        handle.close()


def active(path):
    try:
        with locked(path):
            return False
    except BlockingIOError:
        return True


def below(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path != root and root in path.parents, "Path escapes owned directory: " + str(path))
    return path


def paths(root, kind):
    need(kind in ("random", "domain"), "Unknown setup")
    root = Path(root).resolve()
    base = root / "outputs/multidomain"
    suffix = "_db" if kind == "domain" else ""
    single = base / (BASE + ("_db_v1" if kind == "domain" else ""))
    old = base / (BASE + suffix + "_n2_once_v1")
    run = base / (BASE + suffix + "_continuation_v1")
    return dict(root=root, single=single, old=old, old_control=old / "handoff",
                run=run, control=run / "control", monitor=single / "monitor")


def extended_deadline(previous, hours, at=None):
    at = datetime.fromisoformat(at or now())
    old = datetime.fromisoformat(previous) if previous else at
    need(old.tzinfo is not None and at.tzinfo is not None, "Deadline must include UTC offset")
    need(0 < hours <= 168, "Invalid extension")
    return (max(old, at) + timedelta(hours=hours)).isoformat()


def expired(deadline):
    return datetime.fromisoformat(now()) >= datetime.fromisoformat(deadline)


def safe_id(value):
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value), "Invalid operation ID")
    return value


def files(root):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            need(not p.is_symlink(), "Snapshot symlink: " + str(p))
            result[str(p.relative_to(root))] = sha(p)
    return result


def verify_files(root, expected):
    for name, checksum in expected.items():
        p = below(Path(root) / name, root)
        need(not p.is_symlink() and sha(p) == checksum, "Source changed: " + str(p))
