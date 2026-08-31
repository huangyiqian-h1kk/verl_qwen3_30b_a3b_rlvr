#!/usr/bin/env python3
"""Fail-closed preflight for a frozen Reasoning Gym training run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from packaging.version import Version

import reasoning_gym


EXPECTED_SCHEMA = "reasoning_gym_static_v2"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tagged_object_hook(value: dict[str, Any]) -> Any:
    kind = value.get("__rg_json_type__")
    raw = value.get("value")
    if kind == "datetime":
        return dt.datetime.fromisoformat(str(raw))
    if kind == "date":
        return dt.date.fromisoformat(str(raw))
    if kind == "time":
        return dt.time.fromisoformat(str(raw))
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_items(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--item must be key=value: {item}")
        key, value = item.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or empty fingerprint key: {key!r}")
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reward-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--fingerprint-file", required=True)
    parser.add_argument("--item", action="append", default=[])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    audit_path = data_dir / "audit_report.json"
    complete_path = data_dir / "BUILD_COMPLETE"
    required_files = [
        manifest_path,
        audit_path,
        complete_path,
        Path(args.reward_file),
        Path(args.launcher),
        Path(args.system_prompt_file),
        Path(args.model_path) / "config.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(f"[FAIL] missing required paths: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise SystemExit("[FAIL] dataset manifest/audit is not complete")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise SystemExit(f"[FAIL] unexpected schema: {manifest.get('schema_version')}")
    actual_version = importlib.metadata.version("reasoning-gym")
    if Version(actual_version).base_version != Version(str(manifest["reasoning_gym_version"])).base_version:
        raise SystemExit(
            f"[FAIL] Reasoning Gym mismatch: runtime={actual_version}, data={manifest['reasoning_gym_version']}"
        )

    complete_values = {}
    for line in complete_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            complete_values[key] = value
    if complete_values.get("status") != "PASS":
        raise SystemExit("[FAIL] BUILD_COMPLETE does not say status=PASS")
    if complete_values.get("manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("[FAIL] manifest hash differs from BUILD_COMPLETE")

    frames = {}
    for split, expected_rows in (("train", 64000), ("validation", 1256)):
        info = manifest["outputs"][split]
        path = Path(info["path"])
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing {split} parquet: {path}")
        if sha256_file(path) != info["sha256"]:
            raise SystemExit(f"[FAIL] {split} parquet SHA256 mismatch")
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows or len(frame) != int(info["rows"]):
            raise SystemExit(f"[FAIL] {split} rows={len(frame)}, expected={expected_rows}")
        expected_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        if set(frame.columns) != expected_columns:
            raise SystemExit(f"[FAIL] {split} columns={list(frame.columns)}")
        frames[split] = frame

    strata = {}
    categories, tasks = set(), set()
    for split, frame in frames.items():
        for row in frame.itertuples(index=False):
            extra = row.extra_info
            if extra["rg_schema_version"] != EXPECTED_SCHEMA:
                raise SystemExit("[FAIL] row schema mismatch")
            categories.add(str(extra["rg_category"]))
            tasks.add(str(extra["rg_task"]))
            key = f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"
            strata.setdefault(key, (split, row))
    if len(categories) != 6 or len(tasks) != 79 or len(strata) != 157:
        raise SystemExit(
            f"[FAIL] inventory mismatch: categories={len(categories)}, tasks={len(tasks)}, strata={len(strata)}"
        )

    # Replay one frozen oracle per stratum using its exact stored config.
    for key, (_split, row) in sorted(strata.items()):
        extra = row.extra_info
        config = json.loads(str(extra["rg_config_json"]), object_hook=tagged_object_hook)
        entry = json.loads(str(extra["rg_entry_json"]))
        dataset = reasoning_gym.create_dataset(str(extra["rg_task"]), **config)
        score = float(dataset.score_answer(str(row.reward_model["ground_truth"]), entry))
        if score < 1.0 - 1e-12:
            raise SystemExit(f"[FAIL] frozen oracle replay failed for {key}: {score}")

    fingerprint = {
        "schema_version": "rg_training_fingerprint_v1",
        "parameters": parse_items(args.item),
        "files": {
            "train": manifest["outputs"]["train"]["sha256"],
            "validation": manifest["outputs"]["validation"]["sha256"],
            "dataset_manifest": sha256_file(manifest_path),
            "reward": sha256_file(Path(args.reward_file)),
            "launcher": sha256_file(Path(args.launcher)),
            "system_prompt": sha256_file(Path(args.system_prompt_file)),
            "model_config": sha256_file(Path(args.model_path) / "config.json"),
        },
    }
    fingerprint_path = Path(args.fingerprint_file)
    if fingerprint_path.exists():
        existing = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise SystemExit(
                "[FAIL] resume fingerprint differs from the original run:\n"
                + json.dumps({"existing": existing, "current": fingerprint}, indent=2)
            )
    else:
        atomic_write_json(fingerprint_path, fingerprint)

    result = {
        "status": "PASS",
        "reasoning_gym_version": actual_version,
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "categories": len(categories),
        "tasks": len(tasks),
        "strata": len(strata),
        "fingerprint": str(fingerprint_path.resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
