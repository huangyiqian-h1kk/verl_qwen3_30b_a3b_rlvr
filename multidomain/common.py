"""Shared serialization and error types. No training framework imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class DataError(ValueError):
    pass

class InfrastructureError(RuntimeError):
    """A scoring/serving failure, never a negative training reward."""


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read_config(path):
    import yaml
    from multidomain.preparation.check_config import resolve
    return resolve(yaml.safe_load(Path(path).read_text()))
