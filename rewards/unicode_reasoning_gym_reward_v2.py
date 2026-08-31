#!/usr/bin/env python3
"""verl reward: exact stored Reasoning Gym verifier plus Unicode format.

Reasoning Gym exposes heterogeneous partial scores.  GRPO optimization uses a
binary full-correctness reward; the native score is retained as a diagnostic.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import reasoning_gym


ACC_WEIGHT = float(os.environ.get("ACC_WEIGHT", "1.0"))
FORMAT_WEIGHT = float(os.environ.get("FORMAT_WEIGHT", "0.05"))
DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
DEBUG_RATE = float(os.environ.get("REWARD_DEBUG_RATE", "0.0"))
EXPECTED_SCHEMA = "reasoning_gym_static_v2"

STRICT_PATTERN = re.compile(
    r"^\s*《reasoning》\s*(?P<reasoning>\S(?:.*?\S)?)\s*"
    r"《/reasoning》\s*《answer》\s*(?P<answer>\S(?:.*?\S)?)\s*"
    r"《/answer》\s*$",
    re.S,
)
ANSWER_BLOCK_PATTERN = re.compile(r"《answer》\s*(?P<answer>.*?)\s*《/answer》", re.S)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$")


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


def last_balanced_box(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    position, begin, depth = start + len(marker), start + len(marker), 1
    while position < len(text):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:position].strip()
        position += 1
    return None


def extract_answer(solution: str) -> tuple[str | None, str, float]:
    strict = STRICT_PATTERN.match(solution)
    if strict:
        candidate = strict.group("answer").strip()
        extraction, format_score = "strict", 1.0
    else:
        blocks = list(ANSWER_BLOCK_PATTERN.finditer(solution))
        if blocks and blocks[-1].group("answer").strip():
            candidate = blocks[-1].group("answer").strip()
            extraction = "answer_block"
        else:
            boxed = last_balanced_box(solution)
            if boxed:
                candidate, extraction = boxed, "boxed"
            else:
                lines = ANSWER_LINE_PATTERN.findall(solution)
                candidate = lines[-1].strip() if lines else None
                extraction = "answer_line" if lines else "none"
        format_score = 0.0
    if candidate:
        boxed = last_balanced_box(candidate)
        if candidate.lstrip().startswith(r"\boxed{") and boxed is not None:
            candidate = boxed
        if len(candidate) >= 2 and candidate[0] == "$" and candidate[-1] == "$":
            candidate = candidate[1:-1].strip()
    return candidate, extraction, format_score


def parse_extra(extra_info: Any) -> dict[str, Any]:
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        parsed = json.loads(extra_info)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"extra_info must be a dict or JSON mapping, got {type(extra_info).__name__}")


@lru_cache(maxsize=512)
def configured_dataset(task: str, config_json: str):
    config = json.loads(config_json, object_hook=tagged_object_hook)
    if not isinstance(config, dict):
        raise TypeError("rg_config_json must decode to a mapping")
    return reasoning_gym.create_dataset(task, **config)


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
    del ground_truth  # The native verifier consumes the frozen full entry.
    extra = parse_extra(extra_info)
    schema = str(extra.get("rg_schema_version", ""))
    if schema != EXPECTED_SCHEMA:
        raise RuntimeError(f"unexpected rg_schema_version={schema!r}; expected={EXPECTED_SCHEMA!r}")
    task = str(extra["rg_task"])
    config_json = str(extra["rg_config_json"])
    entry = json.loads(str(extra["rg_entry_json"]))
    candidate, extraction, format_score = extract_answer(solution_str)

    error = None
    try:
        raw_score = (
            float(configured_dataset(task, config_json).score_answer(candidate, entry))
            if candidate is not None
            else 0.0
        )
        if not math.isfinite(raw_score):
            raise ValueError(f"non-finite native score: {raw_score}")
        raw_score = min(1.0, max(0.0, raw_score))
    except Exception as exc:
        raw_score = 0.0
        error = f"{type(exc).__name__}: {exc}"

    accuracy = float(raw_score >= 1.0 - 1e-12)
    total = ACC_WEIGHT * accuracy + FORMAT_WEIGHT * format_score
    record = {
        "score": total,
        "acc": accuracy,
        "rg_score": raw_score,
        "format": format_score,
        "pred": candidate if candidate is not None else "[NO_ANSWER]",
        "answer_extraction": extraction,
        "rg_category": extra.get("rg_category"),
        "rg_task": task,
        "rg_tier": extra.get("rg_tier"),
        "rg_config_sha1": hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:16],
        "score_error": error,
    }
    if should_debug(solution_str):
        path = debug_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **record,
                        "data_source": data_source,
                        "sample_id": extra.get("index"),
                        "solution_str": solution_str,
                    },
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
    config = {"seed": 1, "size": 1, "min_animals": 1, "max_animals": 4, "min_instances": 1, "max_instances": 4}
    extra = {
        "index": "self-test",
        "rg_schema_version": EXPECTED_SCHEMA,
        "rg_category": "arithmetic",
        "rg_task": "leg_counting",
        "rg_tier": "test",
        "rg_config_json": json.dumps(config),
        "rg_entry_json": json.dumps(entry),
    }
    solution = "《reasoning》Two dogs have eight legs.《/reasoning》《answer》8《/answer》"
    result = compute_score("reasoning_gym/arithmetic/leg_counting/test", solution, "8", extra)
    assert result["acc"] == 1.0 and result["format"] == 1.0, result
    assert abs(result["score"] - (ACC_WEIGHT + FORMAT_WEIGHT)) < 1e-12, result
    print("[PASS] unicode_reasoning_gym_reward_v2 self-test")


if __name__ == "__main__":
    self_test()
