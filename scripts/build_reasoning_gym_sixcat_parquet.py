#!/usr/bin/env python3
"""Build and audit a frozen six-category Reasoning Gym RLVR dataset.

The output is deliberately static: training never calls a procedural generator.
Each row stores the exact Reasoning Gym task config and entry required to replay
the native verifier.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import datetime as dt
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import random
import signal
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from packaging.version import Version

import reasoning_gym
from reasoning_gym.coaching.attributes import RangeAttributeDefinition
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.factory import CURRICULA, DATASETS, create_curriculum, create_dataset


SCHEMA_VERSION = "reasoning_gym_static_v2"
TIERS = ("medium", "hard")
_TOKENIZER = None
_TOKENIZER_PATH = None


def json_default(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return {"__rg_json_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"__rg_json_type__": "date", "value": value.isoformat()}
    if isinstance(value, dt.time):
        return {"__rg_json_type__": "time", "value": value.isoformat()}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
    )


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def package_version() -> str:
    try:
        return importlib.metadata.version("reasoning-gym")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def validate_rg_version(expected: str) -> str:
    actual = package_version()
    if actual == "unknown" or Version(actual).base_version != Version(expected).base_version:
        raise RuntimeError(f"Reasoning Gym version mismatch: expected={expected}, actual={actual}")
    return actual


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"config must contain a mapping: {path}")
    return value


def discover_selected_tasks(categories: dict[str, list[str]]) -> dict[str, list[str]]:
    selected = {str(category): sorted(map(str, tasks)) for category, tasks in categories.items()}
    actual: dict[str, list[str]] = {category: [] for category in selected}
    for task, (dataset_cls, _config_cls) in DATASETS.items():
        parts = dataset_cls.__module__.split(".")
        if len(parts) >= 2 and parts[0] == "reasoning_gym" and parts[1] in actual:
            actual[parts[1]].append(task)
    actual = {category: sorted(tasks) for category, tasks in actual.items()}
    if actual != selected:
        details = {}
        for category in selected:
            details[category] = {
                "missing_from_registry": sorted(set(selected[category]) - set(actual.get(category, []))),
                "unexpected_in_registry": sorted(set(actual.get(category, [])) - set(selected[category])),
            }
        raise RuntimeError(f"Reasoning Gym registry drift detected: {json.dumps(details, indent=2)}")
    flattened = [task for tasks in selected.values() for task in tasks]
    if len(flattened) != 79 or len(set(flattened)) != 79:
        raise RuntimeError(f"expected exactly 79 unique selected tasks, got {len(set(flattened))}")
    return selected


def apportion(total: int, names: list[str]) -> dict[str, int]:
    """Equal deterministic integer apportionment in the supplied name order."""
    if total < 0 or not names:
        raise ValueError("invalid apportionment request")
    base, remainder = divmod(total, len(names))
    return {name: base + int(index < remainder) for index, name in enumerate(names)}


def weighted_apportion(total: int, names: list[str], weights: dict[str, float] | None) -> dict[str, int]:
    if not weights:
        return apportion(total, names)
    unknown = set(weights) - set(names)
    if unknown:
        raise ValueError(f"task weights contain unknown tasks: {sorted(unknown)}")
    resolved = {name: float(weights.get(name, 1.0)) for name in names}
    if any(not math.isfinite(value) or value <= 0 for value in resolved.values()):
        raise ValueError(f"all task weights must be finite and positive: {resolved}")
    denominator = sum(resolved.values())
    exact = {name: total * resolved[name] / denominator for name in names}
    result = {name: math.floor(exact[name]) for name in names}
    remainder = total - sum(result.values())
    order = sorted(names, key=lambda name: (-(exact[name] - result[name]), names.index(name)))
    for name in order[:remainder]:
        result[name] += 1
    return result


def tier_level(attribute: Any, tier: str, policy: dict[str, Any]) -> int:
    n_levels = len(attribute.levels)
    if n_levels < 2:
        return 0
    if isinstance(attribute, RangeAttributeDefinition) and attribute.ensure_interval:
        key = f"interval_range_{tier}_level"
        return min(int(policy[key]), n_levels - 2)
    medium = (n_levels - 1) // 2
    return medium if tier == "medium" else min(n_levels - 1, medium + 1)


def curriculum_profile(task: str, tier: str, policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    curriculum = create_curriculum(task)
    levels = {}
    for name, attribute in curriculum.attributes.items():
        level = tier_level(attribute, tier, policy)
        curriculum.set_attr_level(name, level)
        levels[name] = level
    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    profile = dataclasses.asdict(curriculum.generate_configuration(context=context))
    profile.pop("seed", None)
    profile.pop("size", None)
    return profile, levels


def merge_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = canonical_json([base_seed, *parts]).encode("utf-8")
    # Keep the seed inside the range accepted by Python/numpy-backed generators.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_000_000_000


def make_plan(config: dict[str, Any], config_sha256: str, system_prompt_sha256: str) -> dict[str, Any]:
    categories = discover_selected_tasks(config["categories"])
    fixed = config.get("fixed_tasks", {})
    overrides = config.get("task_overrides", {})
    task_weights = config.get("task_weights", {})
    policy = config["difficulty_policy"]
    dataset_cfg = config["dataset"]

    train_category_counts = apportion(int(dataset_cfg["train_size"]), list(categories))
    specs: list[dict[str, Any]] = []
    for category, tasks in categories.items():
        task_counts = weighted_apportion(train_category_counts[category], tasks, task_weights.get(category))
        for task in tasks:
            if task in fixed:
                tier_profiles = [(str(fixed[task]["tier"]), dict(fixed[task].get("config", {})), {})]
                if task in CURRICULA:
                    raise RuntimeError(f"fixed task unexpectedly has a curriculum: {task}")
            else:
                if task not in CURRICULA:
                    raise RuntimeError(f"selected task has no curriculum and no fixed profile: {task}")
                tier_profiles = []
                for tier in TIERS:
                    profile, levels = curriculum_profile(task, tier, policy)
                    profile = merge_dict(profile, overrides.get(task, {}).get(tier))
                    tier_profiles.append((tier, profile, levels))
                if canonical_json(tier_profiles[0][1]) == canonical_json(tier_profiles[1][1]):
                    raise RuntimeError(f"medium and hard profiles are identical: {task}")

            train_tier_counts = apportion(task_counts[task], [item[0] for item in tier_profiles])
            for tier, profile, levels in tier_profiles:
                # Instantiate once so dataclass validation fails before generation.
                create_dataset(task, seed=1, size=1, **profile)
                spec_key = f"{category}/{task}/{tier}"
                specs.append(
                    {
                        "key": spec_key,
                        "category": category,
                        "task": task,
                        "tier": tier,
                        "profile_config": profile,
                        "curriculum_levels": levels,
                        "train_count": train_tier_counts[tier],
                        "validation_count": int(dataset_cfg["validation_per_stratum"]),
                        "train_seed": stable_seed(int(dataset_cfg["train_base_seed"]), category, task, tier),
                        "validation_seed": stable_seed(
                            int(dataset_cfg["validation_base_seed"]), category, task, tier
                        ),
                    }
                )

    if len(specs) != 157:
        raise RuntimeError(f"expected 157 task/tier strata, got {len(specs)}")
    if sum(spec["train_count"] for spec in specs) != int(dataset_cfg["train_size"]):
        raise RuntimeError("train quota apportionment does not sum to train_size")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": config["version"],
        "reasoning_gym_version": package_version(),
        "config_sha256": config_sha256,
        "system_prompt_sha256": system_prompt_sha256,
        "category_train_counts": train_category_counts,
        "task_train_counts": {
            f"{spec['category']}/{spec['task']}": sum(
                item["train_count"]
                for item in specs
                if item["category"] == spec["category"] and item["task"] == spec["task"]
            )
            for spec in specs
        },
        "specs": specs,
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan).encode("utf-8"))
    return plan


def _get_tokenizer(path: str):
    global _TOKENIZER, _TOKENIZER_PATH
    if _TOKENIZER is None or _TOKENIZER_PATH != path:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        _TOKENIZER_PATH = path
    return _TOKENIZER


def chat_token_count(tokenizer: Any, system_prompt: str, question: str) -> int:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(token_ids)


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum, _frame):
        raise TimeoutError(f"operation exceeded {seconds:.3f}s")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def display_answer(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return canonical_json(value)
    return str(value)


def boxnet_oracle(entry: dict[str, Any]) -> str:
    """Construct one valid sequential plan for Boxnet's answer-less entries."""
    state = entry.get("metadata", {}).get("initial_state")
    if not isinstance(state, dict):
        raise RuntimeError("boxnet entry has no initial_state")

    boxes: dict[str, list[tuple[float, float]]] = defaultdict(list)
    targets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key, items in state.items():
        row, column = map(float, str(key).split("_", 1))
        for item in items:
            if str(item).startswith("box_"):
                boxes[str(item)[4:]].append((row, column))
            elif str(item).startswith("target_"):
                targets[str(item)[7:]].append((row, column))

    plans = []
    for colour in sorted(boxes):
        remaining_targets = list(targets[colour])
        if len(remaining_targets) != len(boxes[colour]):
            raise RuntimeError(f"boxnet count mismatch for {colour}")
        for start in boxes[colour]:
            # Pair to a nearest remaining target to keep the stored oracle short.
            target = min(
                remaining_targets,
                key=lambda point: abs(point[0] - start[0]) + abs(point[1] - start[1]),
            )
            remaining_targets.remove(target)
            current = start
            while current != target:
                row, column = current
                if row < target[0]:
                    nxt = (row + 1.0, column)
                elif row > target[0]:
                    nxt = (row - 1.0, column)
                elif column < target[1]:
                    nxt = (row, column + 1.0)
                else:
                    nxt = (row, column - 1.0)
                plans.append(
                    {
                        f"Agent[{row}, {column}]": (
                            f"move(box_{colour}, square[{nxt[0]}, {nxt[1]}])"
                        )
                    }
                )
                current = nxt
            plans.append({f"Agent[{current[0]}, {current[1]}]": f"move(box_{colour}, target_{colour})"})
    return json.dumps(plans, ensure_ascii=False, separators=(",", ":"))


def rush_hour_oracle(entry: dict[str, Any], max_states: int = 2_000_000) -> str:
    """Solve a frozen 6x6 Rush Hour board with deterministic BFS."""
    from reasoning_gym.games.rush_hour import BOARD_SIZE, H, TARGET, V, Board

    board = Board(str(entry.get("metadata", {}).get("board_config", "")))
    pieces = tuple((piece.size, piece.stride, piece.fixed) for piece in board._pieces)
    initial = tuple(piece.position for piece in board._pieces)

    def occupied(state: tuple[int, ...]) -> set[int]:
        cells = set()
        for position, (size, stride, _fixed) in zip(state, pieces):
            cells.update(position + offset * stride for offset in range(size))
        return cells

    queue = deque([initial])
    parent: dict[tuple[int, ...], tuple[tuple[int, ...], int, int] | None] = {initial: None}
    goal = None
    while queue:
        state = queue.popleft()
        if state[0] == TARGET:
            goal = state
            break
        all_cells = occupied(state)
        for index, (position, (size, stride, fixed)) in enumerate(zip(state, pieces)):
            if fixed:
                continue
            own = {position + offset * stride for offset in range(size)}
            other = all_cells - own
            candidates = []
            if stride == H:
                column = position % BOARD_SIZE
                if column > 0 and position - 1 not in other:
                    candidates.append(-1)
                if column + size < BOARD_SIZE and position + size not in other:
                    candidates.append(1)
            elif stride == V:
                if position - V >= 0 and position - V not in other:
                    candidates.append(-1)
                tail = position + (size - 1) * V
                if tail + V < BOARD_SIZE * BOARD_SIZE and tail + V not in other:
                    candidates.append(1)
            for direction in candidates:
                updated = list(state)
                updated[index] += direction * stride
                child = tuple(updated)
                if child in parent:
                    continue
                parent[child] = (state, index, direction)
                queue.append(child)
                if len(parent) > max_states:
                    raise RuntimeError(f"rush_hour BFS exceeded {max_states} states")
    if goal is None:
        raise RuntimeError("rush_hour BFS found no solution")

    unit_moves = []
    cursor = goal
    while parent[cursor] is not None:
        previous, index, direction = parent[cursor]
        unit_moves.append((index, direction))
        cursor = previous
    unit_moves.reverse()

    compressed: list[tuple[int, int]] = []
    for index, direction in unit_moves:
        if compressed and compressed[-1][0] == index and (compressed[-1][1] > 0) == (direction > 0):
            compressed[-1] = (index, compressed[-1][1] + direction)
        else:
            compressed.append((index, direction))
    return " ".join(f"{chr(ord('A') + index)}{steps:+d}" for index, steps in compressed)


def oracle_candidate(task: str, entry: dict[str, Any]) -> Any:
    candidate = entry.get("answer")
    if candidate is None and isinstance(entry.get("metadata"), dict):
        candidate = entry["metadata"].get("possible_answer")
    if candidate is None and task == "boxnet":
        candidate = boxnet_oracle(entry)
    if candidate is None and task == "rush_hour":
        candidate = rush_hour_oracle(entry)
    return candidate


def sample_id(split: str, task: str, tier: str, seed: int, source_index: int, question: str) -> str:
    value = canonical_json([split, task, tier, seed, source_index, question])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def rejection_reason(
    task: str,
    question: str,
    answer: str,
    entry: dict[str, Any],
    entry_json: str,
    prompt_tokens: int,
    oracle_tokens: int,
    audit: dict[str, Any],
) -> str | None:
    if not question.strip():
        return "empty_question"
    if bool(audit.get("reject_empty_answer", True)) and not answer.strip():
        return "empty_answer"
    if any(str(item) in answer for item in audit.get("reject_answer_substrings", [])):
        return "rejected_answer_substring"
    if prompt_tokens > int(audit["max_prompt_tokens"]):
        return "prompt_too_long"
    if oracle_tokens > int(audit["max_oracle_tokens"]):
        return "oracle_too_long"
    if len(entry_json.encode("utf-8")) > int(audit["max_entry_json_bytes"]):
        return "entry_json_too_large"
    if task == "codeio":
        metadata = entry.get("metadata", {})
        if not metadata.get("input_data") or not metadata.get("output_data"):
            return "codeio_empty_io"
    return None


def retryable_generation_exception(task: str, exc: Exception) -> bool:
    # Some native generators expose their own bounded rejection sampling as a
    # ValueError.  Treat only known, exact failure signatures as a rejected
    # candidate; every other exception remains fatal.
    message = str(exc)
    return task == "knight_swap" and isinstance(exc, ValueError) and message.startswith(
        "Failed to generate valid puzzle after trying "
    )


def codeio_record_is_safe(record: dict[str, Any]) -> bool:
    """Conservative static filter before CodeIO's native generator calls exec."""
    blocked_modules = {
        "builtins",
        "ctypes",
        "ftplib",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shelve",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "urllib",
    }
    blocked_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
    source = str(record.get("code_sample", "")) + "\n" + str(record.get("input_generator", ""))
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in blocked_modules for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "").split(".")[0] in blocked_modules:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
            return False
    return True


def generate_shard(job: dict[str, Any]) -> dict[str, Any]:
    spec = job["spec"]
    split = job["split"]
    count = int(job["count"])
    seed = int(job["seed"])
    output_path = Path(job["output_path"])
    manifest_path = Path(job["manifest_path"])
    audit = job["audit"]
    system_prompt = job["system_prompt"]
    tokenizer = _get_tokenizer(job["tokenizer_path"])
    dataset_cfg = job["dataset_cfg"]

    max_attempts = max(
        count * int(dataset_cfg["max_attempt_multiplier"]),
        int(dataset_cfg["min_attempts_per_stratum"]),
    )
    row_config = dict(spec["profile_config"])
    row_config.update({"seed": seed, "size": max_attempts})
    dataset = create_dataset(spec["task"], **row_config)
    if spec["task"] == "codeio":
        if not bool(audit.get("codeio_static_safety_filter", False)):
            raise RuntimeError("CodeIO requires audit.codeio_static_safety_filter=true")
        original_count = len(dataset._jsonl_data)
        safe_records = [record for record in dataset._jsonl_data if codeio_record_is_safe(record)]
        if not safe_records:
            raise RuntimeError(f"CodeIO difficulty {row_config.get('difficulty')} has no statically safe records")
        dataset.__class__._jsonl_data = safe_records
        codeio_filter = {"records_before": original_count, "records_after": len(safe_records)}
    else:
        codeio_filter = None

    rows = []
    rejections: Counter[str] = Counter()
    generation_times = []
    verifier_times = []
    seen_questions = set(map(str, job.get("exclude_questions", [])))
    for source_index in range(max_attempts):
        started = time.monotonic()
        try:
            with time_limit(float(audit["max_generate_seconds"])):
                entry = dataset[source_index]
        except Exception as exc:
            if retryable_generation_exception(spec["task"], exc):
                rejections["native_bounded_rejection"] += 1
                continue
            raise RuntimeError(
                f"{spec['key']} source_index={source_index} generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        generation_times.append(time.monotonic() - started)

        if not isinstance(entry, dict) or "question" not in entry:
            raise RuntimeError(f"{spec['key']} source_index={source_index}: malformed entry")
        question = str(entry["question"])
        if question in seen_questions:
            rejections["duplicate_within_or_across_split"] += 1
            continue
        try:
            with time_limit(float(audit["max_generate_seconds"])):
                answer_value = oracle_candidate(spec["task"], entry)
        except Exception as exc:
            raise RuntimeError(
                f"{spec['key']} source_index={source_index} oracle construction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if answer_value is None:
            raise RuntimeError(f"{spec['key']} source_index={source_index}: missing oracle answer")
        answer = display_answer(answer_value)
        entry_json = canonical_json(entry)
        prompt_tokens = chat_token_count(tokenizer, system_prompt, question)
        oracle_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        reason = rejection_reason(
            spec["task"], question, answer, entry, entry_json, prompt_tokens, oracle_tokens, audit
        )
        if reason:
            rejections[reason] += 1
            continue

        score_started = time.monotonic()
        try:
            with time_limit(float(audit["max_verifier_seconds"])):
                oracle_score = float(dataset.score_answer(answer, entry))
        except Exception as exc:
            raise RuntimeError(
                f"{spec['key']} source_index={source_index} verifier failed: {type(exc).__name__}: {exc}"
            ) from exc
        verifier_seconds = time.monotonic() - score_started
        verifier_times.append(verifier_seconds)
        required_score = float(audit["require_oracle_score"])
        if not math.isfinite(oracle_score) or oracle_score < required_score - 1e-12:
            raise RuntimeError(
                f"{spec['key']} source_index={source_index}: oracle self-score={oracle_score}, "
                f"required={required_score}"
            )

        uid = sample_id(split, spec["task"], spec["tier"], seed, source_index, question)
        rows.append(
            {
                "data_source": f"reasoning_gym/{spec['category']}/{spec['task']}/{spec['tier']}",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "ability": f"reasoning_gym/{spec['category']}",
                "reward_model": {"style": "reasoning_gym_native_binary_v2", "ground_truth": answer},
                "extra_info": {
                    "index": uid,
                    "split": split,
                    "rg_schema_version": SCHEMA_VERSION,
                    "rg_category": spec["category"],
                    "rg_task": spec["task"],
                    "rg_tier": spec["tier"],
                    "rg_seed": seed,
                    "rg_source_index": source_index,
                    "rg_oracle_score": oracle_score,
                    "rg_prompt_tokens": prompt_tokens,
                    "rg_oracle_tokens": oracle_tokens,
                    "rg_config_json": canonical_json(row_config),
                    "rg_entry_json": entry_json,
                },
            }
        )
        seen_questions.add(question)
        if len(rows) == count:
            break

    if len(rows) != count:
        raise RuntimeError(
            f"{spec['key']} {split}: accepted {len(rows)}/{count} after {max_attempts} attempts; "
            f"rejections={dict(rejections)}"
        )

    frame = pd.DataFrame(rows)
    atomic_write_parquet(frame, output_path)
    result = {
        "key": spec["key"],
        "split": split,
        "rows": len(rows),
        "seed": seed,
        "profile_config": spec["profile_config"],
        "row_config": row_config,
        "curriculum_levels": spec["curriculum_levels"],
        "attempts": source_index + 1,
        "rejections": dict(sorted(rejections.items())),
        "codeio_static_filter": codeio_filter,
        "prompt_tokens_min": min(row["extra_info"]["rg_prompt_tokens"] for row in rows),
        "prompt_tokens_max": max(row["extra_info"]["rg_prompt_tokens"] for row in rows),
        "oracle_tokens_max": max(row["extra_info"]["rg_oracle_tokens"] for row in rows),
        "generation_seconds_mean": statistics.fmean(generation_times),
        "generation_seconds_max": max(generation_times),
        "verifier_seconds_mean": statistics.fmean(verifier_times),
        "verifier_seconds_max": max(verifier_times),
        "parquet_sha256": sha256_file(output_path),
        "plan_sha256": job["plan_sha256"],
    }
    atomic_write_json(manifest_path, result)
    return result


def questions_from_parquet(path: Path) -> set[str]:
    frame = pd.read_parquet(path, columns=["prompt"])
    return {str(messages[-1]["content"]) for messages in frame["prompt"]}


def generate_pair(pair_job: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate train first, then validation excluding every train prompt."""
    spec = pair_job["spec"]
    output_dir = Path(pair_job["output_dir"])
    results = []

    train_path, train_manifest_path = shard_paths(output_dir, "train", spec)
    train_count = int(pair_job["train_count"])
    train_valid = bool(pair_job["resume"]) and valid_resumable_shard(
        train_path, train_manifest_path, train_count, pair_job["plan_sha256"]
    )
    if train_valid:
        train_result = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    else:
        train_job = dict(pair_job["common"])
        train_job.update(
            {
                "spec": spec,
                "split": "train",
                "count": train_count,
                "seed": int(spec["train_seed"]),
                "output_path": str(train_path),
                "manifest_path": str(train_manifest_path),
            }
        )
        train_result = generate_shard(train_job)
    results.append(train_result)
    train_questions = questions_from_parquet(train_path)

    validation_path, validation_manifest_path = shard_paths(output_dir, "validation", spec)
    validation_count = int(pair_job["validation_count"])
    validation_valid = train_valid and bool(pair_job["resume"]) and valid_resumable_shard(
        validation_path, validation_manifest_path, validation_count, pair_job["plan_sha256"]
    )
    if validation_valid:
        validation_result = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    else:
        validation_job = dict(pair_job["common"])
        validation_job.update(
            {
                "spec": spec,
                "split": "validation",
                "count": validation_count,
                "seed": int(spec["validation_seed"]),
                "output_path": str(validation_path),
                "manifest_path": str(validation_manifest_path),
                "exclude_questions": sorted(train_questions),
            }
        )
        validation_result = generate_shard(validation_job)
    results.append(validation_result)
    return results


def shard_paths(output_dir: Path, split: str, spec: dict[str, Any]) -> tuple[Path, Path]:
    stem = f"{spec['category']}__{spec['task']}__{spec['tier']}"
    parquet = output_dir / "shards" / split / f"{stem}.parquet"
    return parquet, parquet.with_suffix(".manifest.json")


def valid_resumable_shard(parquet: Path, manifest: Path, rows: int, plan_sha256: str) -> bool:
    if not parquet.is_file() or not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        return (
            int(value["rows"]) == rows
            and value["plan_sha256"] == plan_sha256
            and value["parquet_sha256"] == sha256_file(parquet)
        )
    except Exception:
        return False


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def merge_and_audit(
    output_dir: Path,
    plan: dict[str, Any],
    config: dict[str, Any],
    tokenizer_path: Path,
    system_prompt_path: Path,
    shard_manifests: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation"):
        frames = []
        for spec in plan["specs"]:
            parquet, _ = shard_paths(output_dir, split, spec)
            frames.append(pd.read_parquet(parquet))
        frame = pd.concat(frames, ignore_index=True)
        expected = sum(spec[f"{split}_count"] for spec in plan["specs"])
        if len(frame) != expected:
            raise RuntimeError(f"{split}: merged rows={len(frame)}, expected={expected}")
        random_state = int(config["dataset"]["shuffle_seed"]) + (0 if split == "train" else 1)
        split_frames[split] = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    train_ids = [str(value["index"]) for value in split_frames["train"]["extra_info"]]
    val_ids = [str(value["index"]) for value in split_frames["validation"]["extra_info"]]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise RuntimeError("duplicate sample ids after merge")

    def questions(frame: pd.DataFrame) -> list[str]:
        return [str(messages[-1]["content"]) for messages in frame["prompt"]]

    train_questions = questions(split_frames["train"])
    val_questions = questions(split_frames["validation"])
    if len(set(train_questions)) != len(train_questions):
        raise RuntimeError("duplicate train prompts after merge")
    if len(set(val_questions)) != len(val_questions):
        raise RuntimeError("duplicate validation prompts after merge")
    overlap = set(train_questions) & set(val_questions)
    if overlap:
        raise RuntimeError(f"train/validation prompt overlap: {len(overlap)}")

    outputs = {}
    for split, frame in split_frames.items():
        filename = "train_64k.parquet" if split == "train" else "validation_fixed.parquet"
        path = output_dir / filename
        atomic_write_parquet(frame, path)
        outputs[split] = {
            "path": str(path.resolve()),
            "rows": len(frame),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    category_counts = Counter()
    task_counts = Counter()
    tier_counts = Counter()
    for extra in split_frames["train"]["extra_info"]:
        category_counts[str(extra["rg_category"])] += 1
        task_counts[f"{extra['rg_category']}/{extra['rg_task']}"] += 1
        tier_counts[f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"] += 1

    all_prompt_maxima = [int(item["prompt_tokens_max"]) for item in shard_manifests]
    all_verifier_maxima = [float(item["verifier_seconds_max"]) for item in shard_manifests]
    audit_report = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "reasoning_gym_version": package_version(),
        "selected_categories": len(config["categories"]),
        "selected_tasks": sum(len(tasks) for tasks in config["categories"].values()),
        "strata": len(plan["specs"]),
        "train_rows": len(split_frames["train"]),
        "validation_rows": len(split_frames["validation"]),
        "duplicate_train_prompts": 0,
        "duplicate_validation_prompts": 0,
        "train_validation_prompt_overlap": 0,
        "generation_errors": 0,
        "verifier_errors": 0,
        "oracle_self_score_failures": 0,
        "category_train_counts": dict(category_counts),
        "task_train_counts": dict(task_counts),
        "stratum_train_counts": dict(tier_counts),
        "max_prompt_tokens_observed": max(all_prompt_maxima),
        "max_verifier_seconds_observed": max(all_verifier_maxima),
        "verifier_stratum_max_p95": percentile(all_verifier_maxima, 0.95),
    }
    manifest = {
        "status": "COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "dataset_version": config["version"],
        "reasoning_gym_version": package_version(),
        "reasoning_gym_git_tag": "v0.1.25",
        "plan_sha256": plan["plan_sha256"],
        "config_path": str(Path(config["_config_path"]).resolve()),
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "system_prompt_path": str(system_prompt_path.resolve()),
        "system_prompt_sha256": sha256_file(system_prompt_path),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "outputs": outputs,
        "audit_report": str((output_dir / "audit_report.json").resolve()),
        "shard_manifests": len(shard_manifests),
    }
    return manifest, audit_report


def build_jobs(
    output_dir: Path,
    plan: dict[str, Any],
    config: dict[str, Any],
    system_prompt: str,
    tokenizer_path: Path,
    resume: bool,
    smoke_count: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = []
    completed = []
    for spec in plan["specs"]:
        train_count = int(spec["train_count"])
        validation_count = int(spec["validation_count"])
        if smoke_count is not None:
            train_count = min(train_count, smoke_count)
            validation_count = min(validation_count, smoke_count)
        train_path, train_manifest = shard_paths(output_dir, "train", spec)
        validation_path, validation_manifest = shard_paths(output_dir, "validation", spec)
        pair_valid = (
            resume
            and valid_resumable_shard(train_path, train_manifest, train_count, plan["plan_sha256"])
            and valid_resumable_shard(
                validation_path, validation_manifest, validation_count, plan["plan_sha256"]
            )
        )
        if pair_valid:
            completed.extend(
                [
                    json.loads(train_manifest.read_text(encoding="utf-8")),
                    json.loads(validation_manifest.read_text(encoding="utf-8")),
                ]
            )
            continue
        jobs.append(
            {
                "spec": spec,
                "output_dir": str(output_dir),
                "train_count": train_count,
                "validation_count": validation_count,
                "resume": resume,
                "plan_sha256": plan["plan_sha256"],
                "common": {
                    "audit": config["audit"],
                    "dataset_cfg": config["dataset"],
                    "system_prompt": system_prompt,
                    "tokenizer_path": str(tokenizer_path),
                    "plan_sha256": plan["plan_sha256"],
                },
            }
        )
    return jobs, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke-count", type=int)
    parser.add_argument("--acknowledge-codeio-exec", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be in [1, 32]")
    if args.smoke_count is not None and args.smoke_count < 1:
        raise SystemExit("--smoke-count must be positive")

    config_path = Path(args.config).resolve()
    system_prompt_path = Path(args.system_prompt_file).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not system_prompt_path.is_file():
        raise FileNotFoundError(system_prompt_path)
    if not args.plan_only and not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise RuntimeError("system prompt is empty")

    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    actual_rg_version = validate_rg_version(str(config["reasoning_gym_version"]))
    plan = make_plan(config, sha256_file(config_path), sha256_file(system_prompt_path))
    print(
        json.dumps(
            {
                "reasoning_gym_version": actual_rg_version,
                "tasks": 79,
                "strata": 157,
                "train_rows": sum(item["train_count"] for item in plan["specs"]),
                "validation_rows": sum(item["validation_count"] for item in plan["specs"]),
                "plan_sha256": plan["plan_sha256"],
            },
            indent=2,
        )
    )
    if args.plan_only:
        print("[PASS] registry, curriculum profiles, overrides, and quotas validated")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "plan.json", plan)
    if not args.acknowledge_codeio_exec:
        raise SystemExit(
            "[FAIL] CodeIO generation executes the curated code bundled with Reasoning Gym. "
            "Run this dedicated CPU build job with --acknowledge-codeio-exec after reviewing the warning."
        )

    jobs, shard_manifests = build_jobs(
        output_dir,
        plan,
        config,
        system_prompt,
        tokenizer_path,
        args.resume,
        args.smoke_count,
    )
    if jobs:
        print(f"[INFO] generating {len(jobs)} train/validation stratum pairs with workers={args.workers}")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(generate_pair, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                results = future.result()
                shard_manifests.extend(results)
                for result in results:
                    print(
                        f"[PASS] {result['split']} {result['key']}: rows={result['rows']} "
                        f"attempts={result['attempts']} prompt_max={result['prompt_tokens_max']}"
                    )

    if args.smoke_count is not None:
        print("[PASS] smoke generation completed; final merge intentionally skipped")
        return

    # Re-read manifests in plan order so final metadata is deterministic.
    shard_manifests = []
    for split in ("train", "validation"):
        for spec in plan["specs"]:
            _parquet, manifest = shard_paths(output_dir, split, spec)
            shard_manifests.append(json.loads(manifest.read_text(encoding="utf-8")))

    manifest, audit_report = merge_and_audit(
        output_dir, plan, config, tokenizer_path, system_prompt_path, shard_manifests
    )
    atomic_write_json(output_dir / "audit_report.json", audit_report)
    atomic_write_json(output_dir / "manifest.json", manifest)
    manifest_sha = sha256_file(output_dir / "manifest.json")
    atomic_write_text(
        output_dir / "BUILD_COMPLETE",
        f"status=PASS\nmanifest_sha256={manifest_sha}\ncompleted_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
    )
    print(json.dumps(audit_report, ensure_ascii=False, indent=2))
    print(f"[PASS] frozen dataset complete: {output_dir}")


if __name__ == "__main__":
    main()
