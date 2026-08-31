#!/usr/bin/env python3
"""Fail-fast checks shared by checkpoint rehearsal, pilot, and full runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_MODEL_BASENAMES = {
    "qwen3": "Qwen3-30B-A3B",
    "inst2507": "Qwen3-30B-A3B-Instruct-2507",
}
FORBIDDEN_MODEL_MARKERS = ("Base", "Thinking-2507")


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        fail(f"cannot read git commit from {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        fail(f"cannot decode prompt messages from {type(value)!r}")
    return value


def prompt_length_stats(
    path: Path, tokenizer: Any, max_prompt_length: int
) -> dict[str, int]:
    frame = pd.read_parquet(path, columns=["prompt"])
    lengths: list[int] = []
    for value in frame["prompt"]:
        token_ids = tokenizer.apply_chat_template(
            prompt_messages(value),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        lengths.append(len(token_ids))
    over = sum(length > max_prompt_length for length in lengths)
    return {
        "rows": len(lengths),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "over_limit": over,
        "limit": max_prompt_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--model-key", choices=sorted(ALLOWED_MODEL_BASENAMES), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dataset-tag", choices=["dapo", "acereason", "deepmath_l89"], required=True
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--reward-file", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--verl-src", required=True)
    parser.add_argument("--expected-verl-commit", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    model_path = Path(args.model_path).resolve()
    expected_basename = ALLOWED_MODEL_BASENAMES[args.model_key]
    if model_path.name != expected_basename:
        fail(
            f"MODEL_KEY={args.model_key} requires basename {expected_basename!r}, "
            f"got {model_path.name!r}"
        )
    if any(marker.lower() in str(model_path).lower() for marker in FORBIDDEN_MODEL_MARKERS):
        fail(f"forbidden model path: {model_path}")
    if not (model_path / "config.json").is_file():
        fail(f"model config not found: {model_path / 'config.json'}")

    required_files = {
        "train": Path(args.train_file),
        "validation": Path(args.val_file),
        "reward": Path(args.reward_file),
        "launcher": Path(args.launcher),
        "dataset audit": Path(args.dataset_audit),
    }
    for label, path in required_files.items():
        if not path.is_file():
            fail(f"{label} file not found: {path}")

    current_commit = git_head(Path(args.verl_src))
    if current_commit != args.expected_verl_commit and not args.allow_commit_mismatch:
        fail(
            f"verl commit mismatch: expected {args.expected_verl_commit}, got {current_commit}. "
            "Do not resume an existing run with a different commit."
        )

    audit = json.loads(Path(args.dataset_audit).read_text(encoding="utf-8"))
    if audit.get("policy", {}).get("cross_task_validation") is not False:
        fail("dataset audit does not declare cross_task_validation=false")
    dataset_report = audit.get("datasets", {}).get(args.dataset_tag)
    if not dataset_report:
        fail(f"dataset audit has no section for {args.dataset_tag}")
    audited_paths = {
        Path(dataset_report["train"]["path"]).resolve(),
        Path(dataset_report["holdout"]["path"]).resolve(),
    }
    supplied_paths = {Path(args.train_file).resolve(), Path(args.val_file).resolve()}
    if audited_paths != supplied_paths:
        fail(
            "train/validation paths differ from audited same-dataset pair: "
            f"audited={sorted(map(str, audited_paths))}, supplied={sorted(map(str, supplied_paths))}"
        )
    for split, supplied in (("train", Path(args.train_file)), ("holdout", Path(args.val_file))):
        audited_hash = dataset_report[split].get("sha256")
        current_hash = sha256_file(supplied)
        if not audited_hash or current_hash != audited_hash:
            fail(
                f"{split} data hash differs from dataset audit: "
                f"audited={audited_hash}, current={current_hash}"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    probe_messages = [
        {"role": "system", "content": "Return a response using the requested unicode tags."},
        {"role": "user", "content": "What is 1+1?"},
    ]
    rendered = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<think>" in rendered or "</think>" in rendered:
        fail(
            "enable_thinking=False still produced built-in <think> markup. "
            "Do not start unicode-format GRPO until tokenizer/template behavior is resolved."
        )

    prompt_lengths = {
        "train": prompt_length_stats(
            Path(args.train_file), tokenizer, args.max_prompt_length
        ),
        "validation": prompt_length_stats(
            Path(args.val_file), tokenizer, args.max_prompt_length
        ),
    }
    if any(stats["over_limit"] for stats in prompt_lengths.values()):
        fail(
            "one or more rendered prompts exceed data.max_prompt_length; "
            f"refusing silent left truncation: {prompt_lengths}"
        )

    summary = {
        "project": str(project),
        "model_key": args.model_key,
        "model_path": str(model_path),
        "dataset_tag": args.dataset_tag,
        "train_file": str(Path(args.train_file).resolve()),
        "val_file": str(Path(args.val_file).resolve()),
        "verl_commit": current_commit,
        "enable_thinking": False,
        "prompt_token_lengths": prompt_lengths,
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "python": sys.version.split()[0],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[PASS] model/data/commit/chat-template preflight")


if __name__ == "__main__":
    main()
