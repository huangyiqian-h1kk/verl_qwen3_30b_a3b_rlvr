#!/usr/bin/env python3
"""Generate a deterministic Reasoning Gym calibration set in verl format."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import reasoning_gym


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def nested_get(value: Any, dotted_path: str) -> Any:
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def accepted(entry: dict[str, Any], rule: dict[str, Any] | None) -> bool:
    if not rule:
        return True
    answer = str(entry.get("answer", ""))
    if "answer_not_equal" in rule and answer == str(rule["answer_not_equal"]):
        return False
    if "answer_not_contains" in rule and str(rule["answer_not_contains"]) in answer:
        return False
    if "metadata_path" in rule:
        value = nested_get(entry.get("metadata", {}), str(rule["metadata_path"]))
        if value is None:
            return False
        if "min_value" in rule and value < rule["min_value"]:
            return False
        if "max_value" in rule and value > rule["max_value"]:
            return False
    return True


def read_system_prompt(config_path: Path, config: dict[str, Any], override: str | None) -> tuple[str, str]:
    prompt_path = Path(override) if override else Path(config["system_prompt_file"])
    if not prompt_path.is_absolute():
        # Config paths are project-relative in production; for a standalone
        # pack, also accept a file relative to the YAML directory.
        candidates = [Path.cwd() / prompt_path, config_path.parent / prompt_path]
        prompt_path = next((p for p in candidates if p.is_file()), candidates[0])
    if not prompt_path.is_file():
        raise FileNotFoundError(f"system prompt not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty system prompt: {prompt_path}")
    return text, str(prompt_path.resolve())


def make_unique_id(task: str, tier: str, seed: int, source_index: int, question: str) -> str:
    payload = canonical_json([task, tier, seed, source_index, question])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def oracle_candidate(entry: dict[str, Any]) -> Any:
    candidate = entry.get("answer")
    if candidate is None:
        candidate = entry.get("metadata", {}).get("possible_answer")
    return candidate


def display_answer(candidate: Any) -> str:
    if isinstance(candidate, (dict, list, tuple)):
        return canonical_json(candidate)
    return str(candidate)


def generate_stratum(
    spec: dict[str, Any], count: int, seed: int, system_prompt: str, split: str
) -> list[dict[str, Any]]:
    task = str(spec["task"])
    tier = str(spec["tier"])
    task_config = dict(spec.get("config", {}))
    accept_rule = spec.get("accept")
    # Filters can reject examples, so expose a larger deterministic virtual set.
    virtual_size = max(count * 50, 1000)
    dataset = reasoning_gym.create_dataset(task, size=virtual_size, seed=seed, **task_config)

    rows: list[dict[str, Any]] = []
    for source_index in range(virtual_size):
        entry = dataset[source_index]
        if not accepted(entry, accept_rule):
            continue
        question = str(entry["question"])
        candidate = oracle_candidate(entry)
        oracle_score = float(dataset.score_answer(display_answer(candidate), entry))
        if oracle_score < 1.0 - 1e-12:
            raise RuntimeError(f"{task}/{tier} source_index={source_index}: stored oracle scored {oracle_score}")
        answer = display_answer(candidate)
        uid = make_unique_id(task, tier, seed, source_index, question)
        entry_json = canonical_json(entry)
        rows.append(
            {
                "data_source": f"reasoning_gym/{task}/{tier}",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "ability": "reasoning_gym",
                "reward_model": {"style": "reasoning_gym", "ground_truth": answer},
                # Keep a fixed Arrow schema: heterogeneous RG metadata is JSON.
                "extra_info": {
                    "index": uid,
                    "split": split,
                    "rg_task": task,
                    "rg_tier": tier,
                    "rg_seed": int(seed),
                    "rg_source_index": int(source_index),
                    "rg_oracle_score": oracle_score,
                    "rg_config_json": canonical_json(task_config),
                    "rg_entry_json": entry_json,
                },
            }
        )
        if len(rows) == count:
            return rows
    raise RuntimeError(f"{task}/{tier}: accepted only {len(rows)} of requested {count} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help=".parquet or .jsonl")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--samples-per-stratum", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", default="calibration")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    system_prompt, system_prompt_path = read_system_prompt(config_path, config, args.system_prompt_file)
    count = args.samples_per_stratum or int(config["samples_per_stratum"])
    base_seed = args.seed if args.seed is not None else int(config["base_seed"])
    if count <= 0:
        raise ValueError("samples_per_stratum must be positive")

    rows: list[dict[str, Any]] = []
    strata_counts: dict[str, int] = {}
    for stratum_index, spec in enumerate(config["strata"]):
        stratum_seed = base_seed + 100_003 * stratum_index
        part = generate_stratum(spec, count, stratum_seed, system_prompt, args.split)
        key = f"{spec['task']}/{spec['tier']}"
        strata_counts[key] = len(part)
        rows.extend(part)

    random.Random(base_seed).shuffle(rows)
    ids = [row["extra_info"]["index"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate generated ids")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    elif output.suffix == ".jsonl":
        frame.to_json(output, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("output must end in .parquet or .jsonl")

    try:
        rg_version = importlib.metadata.version("reasoning-gym")
    except importlib.metadata.PackageNotFoundError:
        rg_version = "unknown"
    manifest = {
        "dataset_version": config["version"],
        "reasoning_gym_version": rg_version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "system_prompt_path": system_prompt_path,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "base_seed": base_seed,
        "split": args.split,
        "rows": len(rows),
        "strata_counts": strata_counts,
        "output": str(output.resolve()),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
