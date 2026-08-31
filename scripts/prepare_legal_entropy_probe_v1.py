#!/usr/bin/env python3
"""Create one immutable legal-reasoning probe shared by every checkpoint.

The selection rule intentionally matches the existing legal entropy probe:
keep records whose ``incident`` is longer than 250 characters and whose
``argue`` is longer than 50 characters, shuffle with ``random.Random(seed)``,
then take the first N records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original legal JSONL")
    parser.add_argument("--output", required=True, help="Frozen probe JSONL")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-if-matches",
        action="store_true",
        help="Reuse an existing probe only after verifying its manifest and hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    if output.exists() or manifest_path.exists():
        if args.force:
            pass
        elif args.reuse_if_matches and output.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selection = manifest.get("selection") or {}
            checks = {
                "input_path": manifest.get("input") == str(source),
                "input_sha256": manifest.get("input_sha256") == sha256_file(source),
                "output_sha256": manifest.get("output_sha256") == sha256_file(output),
                "seed": selection.get("seed") == args.seed,
                "sample_size": selection.get("sample_size") == args.sample_size,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise SystemExit(
                    "existing frozen probe does not match requested configuration: "
                    + ", ".join(failed)
                )
            print(f"[PASS] reusing verified frozen probe: {output}")
            return
        else:
            raise SystemExit(
                f"refusing to overwrite frozen probe: {output}; use --force intentionally"
            )

    rows = load_jsonl(source)
    eligible: list[dict[str, Any]] = []
    for input_index, row in enumerate(rows):
        incident = row.get("incident")
        argue = row.get("argue")
        if not isinstance(incident, str) or not isinstance(argue, str):
            continue
        if len(incident) <= 250 or len(argue) <= 50:
            continue
        copied = dict(row)
        copied["_entropy_probe_input_index"] = input_index
        copied["_entropy_probe_case_id"] = canonical_hash(row)
        eligible.append(copied)

    if len(eligible) < args.sample_size:
        raise SystemExit(
            f"only {len(eligible)} eligible rows, fewer than requested {args.sample_size}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[: args.sample_size]
    for probe_index, row in enumerate(selected):
        row["_entropy_probe_index"] = probe_index

    atomic_write_jsonl(output, selected)
    output_sha = sha256_file(output)
    manifest = {
        "schema_version": 1,
        "input": str(source),
        "input_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": output_sha,
        "selection": {
            "seed": args.seed,
            "sample_size": args.sample_size,
            "incident_min_chars_exclusive": 250,
            "argue_min_chars_exclusive": 50,
            "eligible_rows": len(eligible),
            "total_input_rows": len(rows),
        },
        "case_ids": [row["_entropy_probe_case_id"] for row in selected],
    }
    atomic_write_json(manifest_path, manifest)

    print(f"[PASS] frozen legal probe: {output}")
    print(f"[PASS] rows={len(selected)} sha256={output_sha}")
    print(f"[PASS] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
