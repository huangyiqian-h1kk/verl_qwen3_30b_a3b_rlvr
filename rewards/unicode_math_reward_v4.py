#!/usr/bin/env python3
"""GRPO reward for the project's unicode explicit-reasoning protocol.

Design choices:
  * correctness and format are scored independently;
  * an isolated/opening tag receives no reward;
  * correctness can still be recognized when the full format is malformed;
  * debug logging is sampled and separated by worker PID.

The public entry point is ``compute_score`` for verl custom rewards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from verl.utils.reward_score import math_dapo as _verl_math_dapo
except Exception:  # permits offline unit tests outside the pinned verl environment
    _verl_math_dapo = None


ACC_WEIGHT = float(os.environ.get("ACC_WEIGHT", "1.0"))
FORMAT_WEIGHT = float(os.environ.get("FORMAT_WEIGHT", "0.2"))
DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
DEBUG_RATE = float(os.environ.get("REWARD_DEBUG_RATE", "0.0"))
REQUIRE_VERL_VERIFIER = os.environ.get("REQUIRE_VERL_MATH_DAPO", "0") == "1"

OPEN_REASONING = "《reasoning》"
CLOSE_REASONING = "《/reasoning》"
OPEN_ANSWER = "《answer》"
CLOSE_ANSWER = "《/answer》"

STRICT_PATTERN = re.compile(
    r"^\s*《reasoning》\s*(?P<reasoning>\S(?:.*?\S)?)\s*"
    r"《/reasoning》\s*《answer》\s*(?P<answer>\S(?:.*?\S)?)\s*"
    r"《/answer》\s*$",
    re.S,
)
ANSWER_BLOCK_PATTERN = re.compile(
    r"《answer》\s*(?P<answer>.*?)\s*《/answer》", re.S
)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.+?)\s*$")


def _last_balanced_command(text: str, command: str) -> str | None:
    """Return the content of the last balanced ``\\command{...}`` occurrence."""
    marker = f"\\{command}{{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    content_start = i
    depth = 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:i].strip()
        i += 1
    return None


def strict_format_and_answer(text: str) -> tuple[float, str | None]:
    match = STRICT_PATTERN.match(text)
    if match is None:
        return 0.0, None
    return 1.0, match.group("answer").strip()


def extract_answer_candidate(text: str, strict_answer: str | None = None) -> tuple[str | None, str]:
    """Extract an answer without requiring the complete reasoning format."""
    if strict_answer:
        return strict_answer, "strict"

    blocks = list(ANSWER_BLOCK_PATTERN.finditer(text))
    if blocks:
        candidate = blocks[-1].group("answer").strip()
        if candidate:
            return candidate, "answer_block"

    boxed = _last_balanced_command(text, "boxed")
    if boxed:
        return boxed, "boxed"

    lines = ANSWER_LINE_PATTERN.findall(text)
    if lines:
        return lines[-1].strip(), "answer_line"

    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        if tail:
            candidate = tail.splitlines()[0].strip()
            if candidate:
                return candidate, "hash_answer"

    return None, "none"


def _unbox_all(text: str) -> str:
    out: list[str] = []
    cursor = 0
    marker = "\\boxed{"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        out.append(text[cursor:start])
        content_start = start + len(marker)
        i = content_start
        depth = 1
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            out.append(text[start:])
            break
        out.append(text[content_start:i])
        cursor = i + 1
    return "".join(out)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = _unbox_all(str(value).strip())
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("$", "").replace(",", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".").lower()


_LATEX_FRAC = re.compile(r"^\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}$")
_PLAIN_FRAC = re.compile(r"^([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)$")


def _to_number(normalized: str) -> Fraction | None:
    if not normalized:
        return None
    text = normalized
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        match = _LATEX_FRAC.match(text)
        if match:
            value = Fraction(match.group(1)) / Fraction(match.group(2))
        elif _PLAIN_FRAC.match(text):
            value = Fraction(text)
        else:
            value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def answers_match(prediction: Any, ground_truth: Any) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pred_num = _to_number(pred)
    gold_num = _to_number(gold)
    if pred_num is not None and gold_num is not None and pred_num == gold_num:
        return True

    # Use the verifier shipped with the pinned verl checkout as an additional
    # normalization path. We pass only the extracted final answer, so format
    # compliance remains an independent signal.
    if _verl_math_dapo is not None:
        try:
            result = _verl_math_dapo.compute_score(
                f"Answer: {prediction}", str(ground_truth)
            )
            return bool(result.get("acc", False))
        except Exception:
            return False
    return False


def tag_diagnostics(text: str) -> dict[str, float]:
    counts = {
        "open_reasoning": text.count(OPEN_REASONING),
        "close_reasoning": text.count(CLOSE_REASONING),
        "open_answer": text.count(OPEN_ANSWER),
        "close_answer": text.count(CLOSE_ANSWER),
    }
    all_once = all(value == 1 for value in counts.values())
    positions = [
        text.find(OPEN_REASONING),
        text.find(CLOSE_REASONING),
        text.find(OPEN_ANSWER),
        text.find(CLOSE_ANSWER),
    ]
    ordered = all_once and positions == sorted(positions)
    return {
        "all_tags_once": float(all_once),
        "tags_ordered": float(ordered),
        **{f"count_{key}": float(value) for key, value in counts.items()},
    }


def _should_debug(solution_str: str) -> bool:
    if not DEBUG_LOG or DEBUG_RATE <= 0:
        return False
    if DEBUG_RATE >= 1:
        return True
    digest = hashlib.sha1(solution_str.encode("utf-8", errors="replace")).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    return draw < DEBUG_RATE


def _debug_path() -> Path:
    base = Path(DEBUG_LOG)
    suffix = base.suffix or ".jsonl"
    stem = base.name[: -len(base.suffix)] if base.suffix else base.name
    return base.with_name(f"{stem}.pid{os.getpid()}{suffix}")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    fmt, strict_answer = strict_format_and_answer(solution_str)
    candidate, extraction = extract_answer_candidate(solution_str, strict_answer)
    acc = float(candidate is not None and answers_match(candidate, ground_truth))
    score = ACC_WEIGHT * acc + FORMAT_WEIGHT * fmt

    record: dict[str, Any] = {
        "score": score,
        "acc": acc,
        "format": fmt,
        "pred": candidate if candidate is not None else "[NO_ANSWER]",
        "answer_extraction": extraction,
        **tag_diagnostics(solution_str),
    }

    if _should_debug(solution_str):
        path = _debug_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        debug_record = {
            **record,
            "data_source": data_source,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "solution_str": solution_str,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(debug_record, ensure_ascii=False, default=str) + "\n")

    return record


def _self_test() -> None:
    if REQUIRE_VERL_VERIFIER and _verl_math_dapo is None:
        raise SystemExit(
            "[FAIL] REQUIRE_VERL_MATH_DAPO=1 but verl.utils.reward_score.math_dapo "
            "could not be imported"
        )

    only_open = compute_score("test", OPEN_REASONING, "34")
    assert only_open["score"] == 0.0, only_open

    unformatted_correct = compute_score("test", r"The result is \boxed{34}.", "34")
    assert unformatted_correct["acc"] == 1.0
    assert unformatted_correct["format"] == 0.0
    assert unformatted_correct["score"] == ACC_WEIGHT

    complete = (
        "《reasoning》\n17 times 2 is 34.\n《/reasoning》\n"
        "《answer》\n\\boxed{34}\n《/answer》"
    )
    complete_result = compute_score("test", complete, "34")
    assert complete_result["acc"] == 1.0
    assert complete_result["format"] == 1.0
    assert complete_result["score"] == ACC_WEIGHT + FORMAT_WEIGHT

    wrong = complete.replace("34}", "35}")
    wrong_result = compute_score("test", wrong, "34")
    assert wrong_result["acc"] == 0.0
    assert wrong_result["score"] == FORMAT_WEIGHT

    empty_hash = compute_score("test", "some reasoning\n####", "34")
    assert empty_hash["score"] == 0.0
    assert empty_hash["answer_extraction"] == "none"

    assert answers_match(r"\boxed{\frac{1}{2}}", "0.5")
    assert normalize_answer(r"\boxed{1} and \boxed{2}") == "1and2"
    assert answers_match("50%", "0.5")
    assert not answers_match("35", "34")
    verifier = "verl.math_dapo+local" if _verl_math_dapo is not None else "local fallback"
    print(f"unicode_math_reward_v4 self-test OK ({verifier})")


if __name__ == "__main__":
    _self_test()
