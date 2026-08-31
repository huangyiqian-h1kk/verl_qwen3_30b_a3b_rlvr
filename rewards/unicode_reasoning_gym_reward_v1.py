#!/usr/bin/env python3
"""verl custom reward: Reasoning Gym verifier + Unicode format reward."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from reasoning_gym import get_score_answer_fn


ACC_WEIGHT = float(os.environ.get("ACC_WEIGHT", "1.0"))
FORMAT_WEIGHT = float(os.environ.get("FORMAT_WEIGHT", "0.2"))
DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
DEBUG_RATE = float(os.environ.get("REWARD_DEBUG_RATE", "0.0"))

STRICT_PATTERN = re.compile(
    r"^\s*《reasoning》\s*(?P<reasoning>\S(?:.*?\S)?)\s*"
    r"《/reasoning》\s*《answer》\s*(?P<answer>\S(?:.*?\S)?)\s*"
    r"《/answer》\s*$",
    re.S,
)
ANSWER_BLOCK_PATTERN = re.compile(r"《answer》\s*(?P<answer>.*?)\s*《/answer》", re.S)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$")


@lru_cache(maxsize=None)
def score_fn_for(task: str):
    return get_score_answer_fn(task)


def last_balanced_box(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    pos, begin, depth = start + len(marker), start + len(marker), 1
    while pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:pos].strip()
        pos += 1
    return None


def extract_answer(solution: str) -> tuple[str | None, str, float]:
    strict = STRICT_PATTERN.match(solution)
    if strict:
        candidate, extraction, fmt = strict.group("answer").strip(), "strict", 1.0
    else:
        blocks = list(ANSWER_BLOCK_PATTERN.finditer(solution))
        if blocks and blocks[-1].group("answer").strip():
            candidate, extraction = blocks[-1].group("answer").strip(), "answer_block"
        else:
            boxed = last_balanced_box(solution)
            if boxed:
                candidate, extraction = boxed, "boxed"
            else:
                lines = ANSWER_LINE_PATTERN.findall(solution)
                candidate = lines[-1].strip() if lines else None
                extraction = "answer_line" if lines else "none"
        fmt = 0.0
    if candidate:
        boxed = last_balanced_box(candidate)
        if candidate.lstrip().startswith(r"\boxed{") and boxed is not None:
            candidate = boxed
        if len(candidate) >= 2 and candidate[0] == "$" and candidate[-1] == "$":
            candidate = candidate[1:-1].strip()
    return candidate, extraction, fmt


def parse_extra(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        return json.loads(extra_info)
    raise TypeError(f"extra_info must be dict or JSON string, got {type(extra_info).__name__}")


def should_debug(solution: str) -> bool:
    if not DEBUG_LOG or DEBUG_RATE <= 0:
        return False
    if DEBUG_RATE >= 1:
        return True
    digest = hashlib.sha1(solution.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64) < DEBUG_RATE


def debug_path() -> Path:
    base = Path(DEBUG_LOG)
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(base.suffix)] if base.suffix else base.name
    return base.with_name(f"{stem}.pid{os.getpid()}{suffix}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | str | None = None,
    **_: Any,
) -> dict[str, Any]:
    del ground_truth  # The task-specific verifier consumes the full RG entry.
    extra = parse_extra(extra_info)
    task = str(extra["rg_task"])
    entry = json.loads(extra["rg_entry_json"])
    candidate, extraction, fmt = extract_answer(solution_str)
    error = None
    try:
        rg_score = float(score_fn_for(task)(candidate, entry)) if candidate is not None else 0.0
    except Exception as exc:
        rg_score = 0.0
        error = f"{type(exc).__name__}: {exc}"
    acc = float(rg_score >= 1.0 - 1e-12)
    score = ACC_WEIGHT * rg_score + FORMAT_WEIGHT * fmt
    record = {
        "score": score,
        "acc": acc,
        "rg_score": rg_score,
        "format": fmt,
        "pred": candidate if candidate is not None else "[NO_ANSWER]",
        "answer_extraction": extraction,
        "rg_task": task,
        "rg_tier": extra.get("rg_tier"),
        "score_error": error,
    }
    if should_debug(solution_str):
        path = debug_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {**record, "data_source": data_source, "extra_info": extra, "solution_str": solution_str},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    return record


def self_test() -> None:
    entry = {
        "question": "How many legs do two dogs have?",
        "answer": "8",
        "metadata": {"source_dataset": "leg_counting", "animals": {"dog": 2}, "total_legs": 8},
    }
    extra = {"rg_task": "leg_counting", "rg_tier": "test", "rg_entry_json": json.dumps(entry)}
    solution = "《reasoning》Two dogs have eight legs.《/reasoning》《answer》8《/answer》"
    result = compute_score("reasoning_gym/leg_counting/test", solution, "8", extra)
    assert result["acc"] == 1.0 and result["format"] == 1.0, result
    print("[PASS] unicode_reasoning_gym_reward_v1 self-test")


if __name__ == "__main__":
    self_test()

