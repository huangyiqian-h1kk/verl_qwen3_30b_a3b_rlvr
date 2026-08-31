#!/usr/bin/env python3
"""Audit the existing full_v1 train/own-holdout datasets before GRPO.

This script intentionally does not create or use cross-task validation sets.
It verifies same-dataset train/holdout separation and the exact unicode system
prompt, then emits a reproducibility manifest with file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATASETS = {
    "dapo": ("dapo_unicode_train.parquet", "dapo_unicode_holdout.parquet"),
    "acereason": ("acereason_unicode_train.parquet", "acereason_unicode_holdout.parquet"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"prompt is not a list: {type(value)!r}")
    messages = []
    for item in value:
        if hasattr(item, "item") and not isinstance(item, dict):
            item = item.item()
        if not isinstance(item, dict):
            raise TypeError(f"prompt message is not a dict: {type(item)!r}")
        messages.append(item)
    return messages


def extract_problem_and_system(value: Any) -> tuple[str, str]:
    messages = prompt_messages(value)
    systems = [str(item.get("content", "")) for item in messages if item.get("role") == "system"]
    users = [str(item.get("content", "")) for item in messages if item.get("role") == "user"]
    if len(systems) != 1 or len(users) != 1:
        raise ValueError(f"expected one system and one user message; got {len(systems)} / {len(users)}")
    return users[0], systems[0]


def ground_truth_from_row(row: pd.Series) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth")
    elif hasattr(reward_model, "item"):
        value = reward_model.item()
        value = value.get("ground_truth") if isinstance(value, dict) else value
    else:
        value = reward_model
    if value is None or not str(value).strip():
        raise ValueError("missing reward_model.ground_truth")
    return str(value).strip()


def audit_file(path: Path, expected_system_prompt: str) -> tuple[dict[str, Any], set[str]]:
    frame = pd.read_parquet(path)
    required = {"prompt", "reward_model", "data_source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    keys: set[str] = set()
    systems: set[str] = set()
    empty_problem = 0
    for _, row in frame.iterrows():
        problem, system = extract_problem_and_system(row["prompt"])
        systems.add(system)
        if not problem.strip():
            empty_problem += 1
        keys.add(normalized_problem(problem))
        ground_truth_from_row(row)

    duplicate_count = len(frame) - len(keys)
    if empty_problem:
        raise ValueError(f"{path}: {empty_problem} empty problems")
    if duplicate_count:
        raise ValueError(f"{path}: {duplicate_count} lexical duplicate problems")
    if systems != {expected_system_prompt}:
        raise ValueError(
            f"{path}: system prompt mismatch; found {len(systems)} distinct prompt(s)"
        )

    return {
        "path": str(path.resolve()),
        "rows": len(frame),
        "sha256": sha256_file(path),
        "data_sources": sorted(str(x) for x in frame["data_source"].dropna().unique()),
        "unique_problem_keys": len(keys),
        "duplicate_problem_keys": duplicate_count,
    }, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    system_prompt_path = Path(args.system_prompt_file).resolve()
    output_path = Path(args.output).resolve()
    expected_prompt = system_prompt_path.read_text(encoding="utf-8").rstrip("\n")

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "system_prompt_path": str(system_prompt_path),
        "system_prompt_sha256": hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
        "datasets": {},
        "policy": {
            "same_dataset_holdout_only": True,
            "cross_task_validation": False,
        },
    }

    for dataset, (train_name, holdout_name) in DATASETS.items():
        train_path = data_dir / train_name
        holdout_path = data_dir / holdout_name
        if not train_path.is_file() or not holdout_path.is_file():
            raise FileNotFoundError(f"{dataset}: missing {train_path} or {holdout_path}")
        train_report, train_keys = audit_file(train_path, expected_prompt)
        holdout_report, holdout_keys = audit_file(holdout_path, expected_prompt)
        overlap = train_keys & holdout_keys
        if overlap:
            raise ValueError(f"{dataset}: {len(overlap)} train/holdout problem overlaps")
        report["datasets"][dataset] = {
            "train": train_report,
            "holdout": holdout_report,
            "train_holdout_overlap": 0,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[PASS] dataset audit written to {output_path}")


if __name__ == "__main__":
    main()
