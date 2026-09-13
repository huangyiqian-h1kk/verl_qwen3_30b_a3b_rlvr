#!/usr/bin/env python3
"""Fail-closed preflight for a frozen Reasoning Gym training run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
from packaging.version import Version

EXPECTED_SCHEMA = "reasoning_gym_static_v2"
EXPECTED_SYSTEM_ROLE = "system"
EXPECTED_USER_ROLE = "user"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, dict):
            return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"expected mapping-like value, got {type(value).__name__}")


def as_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"prompt is not a message sequence: {type(value).__name__}")
    return [
        {"role": str(as_mapping(item)["role"]), "content": str(as_mapping(item)["content"])}
        for item in value
    ]


def contains_fraction(value: Any) -> bool:
    if isinstance(value, Fraction):
        return True
    if isinstance(value, dict):
        return any(contains_fraction(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_fraction(item) for item in value)
    return False


def load_reward_module(path: Path):
    spec = importlib.util.spec_from_file_location("rg_training_reward_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reward module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("compute_score", "extract_answer", "decode_tagged_mapping"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"reward module is missing callable {name}: {path}")
    return module


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

    reward_path = Path(args.reward_file).resolve()
    system_prompt_path = Path(args.system_prompt_file).resolve()
    expected_system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    reward_module = load_reward_module(reward_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise SystemExit("[FAIL] dataset manifest/audit is not complete")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise SystemExit(f"[FAIL] unexpected schema: {manifest.get('schema_version')}")
    if manifest.get("system_prompt_sha256") != sha256_file(system_prompt_path):
        raise SystemExit("[FAIL] dataset manifest and training system prompt differ")
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
    for split, expected_rows in (("train", 64000), ("validation", 1240)):
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
    fraction_tagged_payloads = 0
    for split, frame in frames.items():
        for row in frame.itertuples(index=False):
            extra = as_mapping(row.extra_info)
            reward_model = as_mapping(row.reward_model)
            messages = as_messages(row.prompt)
            if len(messages) != 2:
                raise SystemExit(f"[FAIL] {split} row does not contain exactly system+user messages")
            if messages[0]["role"] != EXPECTED_SYSTEM_ROLE or messages[0]["content"].strip() != expected_system_prompt:
                raise SystemExit(f"[FAIL] {split} row system prompt mismatch")
            if messages[1]["role"] != EXPECTED_USER_ROLE:
                raise SystemExit(f"[FAIL] {split} row user role mismatch")
            if reward_model.get("style") != "reasoning_gym_native_binary_v2":
                raise SystemExit(f"[FAIL] {split} row reward style mismatch")
            if extra["rg_schema_version"] != EXPECTED_SCHEMA:
                raise SystemExit("[FAIL] row schema mismatch")
            for field_name in ("rg_config_json", "rg_entry_json"):
                payload = str(extra[field_name])
                if "__rg_json_type__" in payload and '"fraction"' in payload:
                    decoded = reward_module.decode_tagged_mapping(payload, field_name)
                    if not contains_fraction(decoded):
                        raise SystemExit(
                            f"[FAIL] {field_name} contains a fraction tag but did not "
                            f"restore Fraction for {extra['rg_task']}"
                        )
                    fraction_tagged_payloads += 1
            ground_truth = str(reward_model["ground_truth"])
            formatted_oracle = (
                "《reasoning》Oracle format compatibility check.《/reasoning》"
                f"《answer》{ground_truth}《/answer》"
            )
            candidate, extraction, format_score = reward_module.extract_answer(formatted_oracle)
            if extraction != "strict" or format_score != 1.0 or not candidate:
                raise SystemExit(
                    f"[FAIL] formatted oracle extraction failed for {extra['rg_task']}: "
                    f"extraction={extraction!r}, format={format_score!r}, "
                    f"candidate={candidate!r}"
                )
            categories.add(str(extra["rg_category"]))
            tasks.add(str(extra["rg_task"]))
            key = f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"
            strata.setdefault(key, (split, row))
    if len(categories) != 6 or len(tasks) != 78 or len(strata) != 155:
        raise SystemExit(
            f"[FAIL] inventory mismatch: categories={len(categories)}, tasks={len(tasks)}, strata={len(strata)}"
        )

    # Replay one strictly formatted frozen oracle per stratum through the exact
    # custom reward function used by verl.  This covers answer extraction,
    # format scoring, stored config/entry decoding, and the native verifier.
    for key, (_split, row) in sorted(strata.items()):
        extra = as_mapping(row.extra_info)
        reward_model = as_mapping(row.reward_model)
        ground_truth = str(reward_model["ground_truth"])
        formatted_oracle = (
            "《reasoning》Oracle format compatibility check.《/reasoning》"
            f"《answer》{ground_truth}《/answer》"
        )
        result = reward_module.compute_score(
            data_source=str(row.data_source),
            solution_str=formatted_oracle,
            ground_truth=ground_truth,
            extra_info=extra,
        )
        if not (
            result.get("acc") == 1.0
            and float(result.get("rg_score", 0.0)) >= 1.0 - 1e-12
            and result.get("format") == 1.0
            and result.get("answer_extraction") == "strict"
            and result.get("score_error") is None
        ):
            raise SystemExit(f"[FAIL] custom reward replay failed for {key}: {result}")

    fingerprint = {
        "schema_version": "rg_training_fingerprint_v1",
        "parameters": parse_items(args.item),
        "files": {
            "train": manifest["outputs"]["train"]["sha256"],
            "validation": manifest["outputs"]["validation"]["sha256"],
            "dataset_manifest": sha256_file(manifest_path),
            "reward": sha256_file(reward_path),
            "launcher": sha256_file(Path(args.launcher)),
            "system_prompt": sha256_file(system_prompt_path),
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
        "fraction_tagged_payloads": fraction_tagged_payloads,
        "fingerprint": str(fingerprint_path.resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
