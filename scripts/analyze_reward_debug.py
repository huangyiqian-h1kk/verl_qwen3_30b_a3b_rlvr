#!/usr/bin/env python3
"""Summarize sampled/per-worker reward JSONL files from a GRPO pilot."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def prompt_id(record: dict[str, Any]) -> str:
    extra = record.get("extra_info")
    if isinstance(extra, dict):
        parts = [
            str(record.get("data_source", "")),
            str(extra.get("source", "")),
            str(extra.get("split", "")),
        ]
        for key in ("index", "id", "uid", "problem_id"):
            if key in extra:
                parts.append(f"{key}={extra[key]}")
                return "|".join(parts)
    return f"unknown:{hash(record.get('ground_truth', ''))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="glob, e.g. outputs/reward_debug/pilot_*.jsonl")
    parser.add_argument("--expected-group-size", type=int, default=8)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.pattern))]
    if not paths:
        raise SystemExit(f"[FAIL] no files matched: {args.pattern}")

    records: list[dict[str, Any]] = []
    bad_lines = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad_lines += 1

    if not records:
        raise SystemExit("[FAIL] no valid reward records")

    scores = [float(record["score"]) for record in records]
    accs = [float(record.get("acc", 0)) for record in records]
    formats = [float(record.get("format", 0)) for record in records]
    extractions = Counter(str(record.get("answer_extraction")) for record in records)

    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[prompt_id(record)].append(float(record["score"]))
    # A training prompt may be sampled again in a later step. Split each
    # prompt's record stream into rollout-sized chunks instead of discarding
    # repeated prompts whose total count is 2*n, 3*n, ...
    complete_groups = []
    for values in groups.values():
        for start in range(0, len(values), args.expected_group_size):
            chunk = values[start : start + args.expected_group_size]
            if len(chunk) == args.expected_group_size:
                complete_groups.append(chunk)
    mixed_groups = [values for values in complete_groups if len(set(values)) > 1]
    all_zero_groups = [values for values in complete_groups if max(values) == 0]
    all_same_groups = [values for values in complete_groups if len(set(values)) == 1]

    summary = {
        "files": [str(path) for path in paths],
        "records": len(records),
        "bad_json_lines": bad_lines,
        "score_mean": mean(scores),
        "score_std": pstdev(scores),
        "accuracy_mean": mean(accs),
        "strict_format_mean": mean(formats),
        "answer_extraction": dict(extractions),
        "groups_total": len(groups),
        "groups_complete": len(complete_groups),
        "groups_mixed_reward": len(mixed_groups),
        "groups_uniform_reward": len(all_same_groups),
        "groups_all_zero_reward": len(all_zero_groups),
        "mixed_group_fraction": (
            len(mixed_groups) / len(complete_groups) if complete_groups else None
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not complete_groups:
        raise SystemExit(
            "[FAIL] no complete rollout-sized prompt groups were reconstructed; "
            "check REWARD_DEBUG_RATE=1 and extra_info.index"
        )
    if not mixed_groups:
        raise SystemExit("[FAIL] no complete prompt group has reward variance; GRPO has no useful group signal")
    if bad_lines:
        raise SystemExit("[FAIL] malformed JSONL lines detected")
    print("[PASS] reward debug contains at least one mixed-reward prompt group")


if __name__ == "__main__":
    main()
