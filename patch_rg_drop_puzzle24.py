#!/usr/bin/env python3
"""Patch the CURRENT repository: drop puzzle24, retain quotas, stop SHA checks.

Run in the ABCI repository, with its usual Python environment active:
    python patch_rg_drop_puzzle24.py --data-dir data/reasoning_gym_sixcat_v1

The saved plan is authoritative for retained profiles, seeds and quotas.
Existing generator/reward fixes are preserved. No generation runs here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import json
import os
from pathlib import Path
import shutil

import yaml

MARKER = "# rg-retained-plan-v1"

FROZEN_BRANCH = '''
    # rg-retained-plan-v1: preserve existing per-stratum quotas and profiles.
    if config.get("retained_plan_file"):
        retained_path = Path(config["_config_path"]).parent / config["retained_plan_file"]
        plan = json.loads(retained_path.read_text(encoding="utf-8"))
        selected = discover_selected_tasks(config["categories"], config.get("excluded_tasks", {}))
        expected_tasks = {(category, task) for category, tasks in selected.items() for task in tasks}
        actual_tasks = {(s["category"], s["task"]) for s in plan["specs"]}
        if actual_tasks != expected_tasks:
            raise RuntimeError("retained plan tasks differ from YAML categories")
        if str(plan["reasoning_gym_version"]) != str(config["reasoning_gym_version"]):
            raise RuntimeError("retained plan Reasoning Gym version differs from YAML")
        if sum(s["train_count"] for s in plan["specs"]) != int(config["dataset"]["train_size"]):
            raise RuntimeError("retained plan train quotas differ from YAML train_size")
        plan.update(dataset_version=config["version"], config_sha256=config_sha256,
                    system_prompt_sha256=system_prompt_sha256,
                    excluded_tasks=config.get("excluded_tasks", {}))
        plan.pop("plan_sha256", None)
        plan["plan_sha256"] = sha256_bytes(canonical_json(plan).encode("utf-8"))
        return plan
'''

RESUME_FUNCTION = '''
def valid_resumable_shard(parquet: Path, manifest: Path, rows: int, plan_sha256: str,
                          *, spec: dict[str, Any], split: str, system_prompt: str) -> bool:
    """Use rows/config/prompt compatibility; never compare SHA values."""
    if not parquet.is_file() or not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if (int(value["rows"]) != rows or value["key"] != spec["key"]
                or value["split"] != split
                or int(value["seed"]) != int(spec[f"{split}_seed"])
                or value["profile_config"] != spec["profile_config"]):
            raise ValueError("manifest row count, stratum, seed or profile differs")
        frame = pd.read_parquet(parquet, columns=["extra_info", "prompt"])
        if len(frame) != rows:
            raise ValueError("actual parquet row count differs")
        for extra, messages in zip(frame["extra_info"], frame["prompt"]):
            if (extra["rg_schema_version"] != SCHEMA_VERSION
                    or str(extra["rg_category"]) != spec["category"]
                    or str(extra["rg_task"]) != spec["task"]
                    or str(extra["rg_tier"]) != spec["tier"]
                    or int(extra["rg_seed"]) != int(spec[f"{split}_seed"])
                    or json.loads(str(extra["rg_config_json"])) != value["row_config"]):
                raise ValueError("parquet row schema, stratum or config differs")
            if (len(messages) != 2 or messages[0]["role"] != "system"
                    or str(messages[0]["content"]).strip() != system_prompt.strip()
                    or messages[1]["role"] != "user"):
                raise ValueError("parquet prompt differs from current system prompt")
        return True
    except Exception as exc:
        print(f"[REBUILD] {parquet.name}: {type(exc).__name__}: {exc}", flush=True)
        return False
'''

PREFLIGHT_PLAN = '''
    retained_plan = json.loads((data_dir / "plan.json").read_text(encoding="utf-8"))
    expected_specs = {s["key"]: s for s in retained_plan["specs"]}
    if not expected_specs or len(expected_specs) != len(retained_plan["specs"]):
        raise SystemExit("[FAIL] empty plan or duplicate stratum keys")
    expected_categories = {s["category"] for s in expected_specs.values()}
    expected_tasks = {s["task"] for s in expected_specs.values()}
    for split in ("train", "validation"):
        if any(int(s[f"{split}_count"]) <= 0 for s in expected_specs.values()):
            raise SystemExit(f"[FAIL] {split} plan contains a non-positive quota")
'''

INVENTORY_CHECK = '''
    if categories != expected_categories or tasks != expected_tasks or set(strata) != set(expected_specs):
        raise SystemExit("[FAIL] actual task/stratum inventory differs from retained plan")
    from collections import Counter
    for split, frame in frames.items():
        observed = Counter()
        for extra_value in frame["extra_info"]:
            extra = as_mapping(extra_value)
            observed[f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"] += 1
        expected = {key: int(s[f"{split}_count"]) for key, s in expected_specs.items()}
        if dict(observed) != expected:
            raise SystemExit(f"[FAIL] {split} per-stratum row counts differ from retained plan")
'''

FINGERPRINT_BLOCK = '''
    # Keep parameter compatibility; file SHA changes do not block a run.
    fingerprint = {
        "schema_version": "rg_training_run_record_v2",
        "parameters": parse_items(args.item),
        "sha_verification": False,
        "files": {
            "train": str(manifest["outputs"]["train"]["path"]),
            "validation": str(manifest["outputs"]["validation"]["path"]),
            "dataset_manifest": str(manifest_path),
            "reward": str(reward_path),
            "launcher": str(Path(args.launcher).resolve()),
            "system_prompt": str(system_prompt_path),
            "model_config": str((Path(args.model_path) / "config.json").resolve()),
        },
    }
    fingerprint_path = Path(args.fingerprint_file)
    if fingerprint_path.exists():
        existing = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if existing.get("parameters") != fingerprint["parameters"]:
            raise SystemExit("[FAIL] training parameters differ from the original run")
    atomic_write_json(fingerprint_path, fingerprint)
'''


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Cannot match this source version exactly: {old[:100]!r}")
    return source.replace(old, new, 1)


def function_node(source: str, name: str) -> ast.FunctionDef:
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one function named {name}")
    return nodes[0]


def replace_lines(source: str, start: int, end: int, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[:start - 1]) + replacement.rstrip("\n") + "\n" + "".join(lines[end:])


def patch_builder(source: str) -> str:
    if MARKER in source:
        return source
    # Modify only the planning/resume guards and display counts, preserving
    # the user's latest generator, timeout, serialization and oracle fixes.
    source = replace_once(source, "if len(flattened) != 78 or len(set(flattened)) != 78:",
                          "if not flattened or len(flattened) != len(set(flattened)):")
    source = replace_once(source, "expected exactly 78 unique selected tasks, got",
                          "expected nonempty, nonduplicated selected tasks, got")
    source = replace_once(source, "if len(specs) != 155:",
                          "if len(specs) != sum(1 if task in fixed else len(TIERS) for tasks in categories.values() for task in tasks):")
    source = replace_once(source, "expected 155 task/tier strata, got", "unexpected task/tier stratum count:")
    source = replace_once(source, '"tasks": 78,', '"tasks": len({s["task"] for s in plan["specs"]}),')
    source = replace_once(source, '"strata": 155,', '"strata": len(plan["specs"]),')
    node = function_node(source, "make_plan")
    lines = source.splitlines(keepends=True)
    lines.insert(node.body[0].lineno - 1, FROZEN_BRANCH.lstrip("\n"))
    source = "".join(lines)
    node = function_node(source, "valid_resumable_shard")
    source = replace_lines(source, node.lineno, node.end_lineno, RESUME_FUNCTION.lstrip("\n"))
    # Add semantic expectations to all four existing resume call sites.
    tree = ast.parse(source)
    edits = []
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    for function in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "valid_resumable_shard":
                continue
            first_arg = ast.unparse(node.args[0])
            split = "validation" if first_arg.startswith("validation_") else "train" if first_arg.startswith("train_") else None
            if split is None or function.name not in ("generate_pair", "build_jobs"):
                raise RuntimeError("Unknown resume call site; no files have been changed")
            prompt = 'pair_job["common"]["system_prompt"]' if function.name == "generate_pair" else "system_prompt"
            node.keywords.extend([
                ast.keyword(arg="spec", value=ast.Name(id="spec", ctx=ast.Load())),
                ast.keyword(arg="split", value=ast.Constant(value=split)),
                ast.keyword(arg="system_prompt", value=ast.parse(prompt, mode="eval").body),
            ])
            # These call-site lines are ASCII; AST columns are byte offsets.
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            edits.append((start, end, ast.unparse(node)))
    if len(edits) != 4:
        raise RuntimeError(f"Expected 4 resume calls, found {len(edits)}")
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    compile(source, "patched_builder.py", "exec")
    return source


def patch_preflight(source: str) -> str:
    if MARKER in source:
        return source
    # The three file-integrity comparisons are independent of schema, prompt
    # content, completion status and reward checks; retain those latter checks.
    guards = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "sha256_file"
            for call in ast.walk(node.test)
        ):
            guards.append(node)
    if len(guards) != 3:
        raise RuntimeError(f"Expected 3 preflight SHA guards, found {len(guards)}")
    for node in sorted(guards, key=lambda n: n.lineno, reverse=True):
        source = replace_lines(source, node.lineno, node.end_lineno,
                               " " * node.col_offset + "# File SHA verification disabled.")
    source = replace_once(source, "    frames = {}\n", PREFLIGHT_PLAN.lstrip("\n") + "\n    frames = {}\n")
    source = replace_once(source, '    for split, expected_rows in (("train", 64000), ("validation", 1240)):',
                          '    for split in ("train", "validation"):\n        expected_rows = sum(int(s[f"{split}_count"]) for s in expected_specs.values())')
    candidates = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If)
                  and "len(categories)" in ast.unparse(n.test) and "len(strata)" in ast.unparse(n.test)]
    if len(candidates) != 1:
        raise RuntimeError("Cannot find the preflight inventory guard")
    node = candidates[0]
    source = replace_lines(source, node.lineno, node.end_lineno, INVENTORY_CHECK.lstrip("\n"))
    start = source.index("    fingerprint = {\n")
    end = source.index("    result = {\n", start)
    source = source[:start] + FINGERPRINT_BLOCK.lstrip("\n") + "\n" + source[end:]
    source += "\n" + MARKER + "\n"
    compile(source, "patched_preflight.py", "exec")
    return source


def retained_config(config: dict, plan: dict) -> tuple[dict, dict]:
    config, plan = copy.deepcopy(config), copy.deepcopy(plan)
    games = config["categories"]["games"]
    config["categories"]["games"] = [task for task in games if task != "puzzle24"]
    excluded = config.setdefault("excluded_tasks", {}).setdefault("games", [])
    if "puzzle24" not in excluded:
        excluded.append("puzzle24")
    config.get("task_overrides", {}).pop("puzzle24", None)
    config.get("task_weights", {}).get("games", {}).pop("puzzle24", None)
    plan["specs"] = [s for s in plan["specs"] if s["task"] != "puzzle24"]
    expected = {(c, task) for c, tasks in config["categories"].items() for task in tasks}
    actual = {(s["category"], s["task"]) for s in plan["specs"]}
    if expected != actual:
        raise RuntimeError(f"Current YAML and saved plan tasks disagree: {expected ^ actual}")
    if len({s["key"] for s in plan["specs"]}) != len(plan["specs"]):
        raise RuntimeError("Saved plan contains duplicate stratum keys")
    if not plan["specs"]:
        raise RuntimeError("No retained strata")
    for spec in plan["specs"]:
        if spec["key"] != f"{spec['category']}/{spec['task']}/{spec['tier']}":
            raise RuntimeError("Saved plan has an inconsistent stratum key")
        for split in ("train", "validation"):
            if int(spec[f"{split}_count"]) <= 0:
                raise RuntimeError("Saved plan has a non-positive quota")
            int(spec[f"{split}_seed"])
    if str(plan["reasoning_gym_version"]) != str(config["reasoning_gym_version"]):
        raise RuntimeError("Saved plan and YAML Reasoning Gym versions differ")
    config["dataset"]["train_size"] = sum(s["train_count"] for s in plan["specs"])
    suffix = "_nopuzzle24_retained_v1"
    if not config["version"].endswith(suffix):
        config["version"] += suffix
    config["retained_plan_file"] = "reasoning_gym_retained_plan.json"
    plan["dataset_version"] = config["version"]
    plan["excluded_tasks"] = config["excluded_tasks"]
    plan["category_train_counts"] = {
        c: sum(s["train_count"] for s in plan["specs"] if s["category"] == c)
        for c in config["categories"]
    }
    plan["task_train_counts"] = {
        f"{c}/{task}": sum(s["train_count"] for s in plan["specs"] if s["category"] == c and s["task"] == task)
        for c, tasks in config["categories"].items() for task in tasks
    }
    plan["reuse_policy"] = "retained profiles, seeds and quotas; no SHA verification"
    return config, plan


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=Path("data/reasoning_gym_sixcat_v1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    data_dir = (root / args.data_dir).resolve()
    config_path = root / "config/reasoning_gym_sixcat_profiles_v1.yaml"
    builder = root / "scripts/build_reasoning_gym_sixcat_parquet.py"
    preflight = root / "scripts/preflight_reasoning_gym_training.py"
    retained_path = config_path.with_name("reasoning_gym_retained_plan.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Re-running the installer uses the same immutable baseline.
    plan_path = retained_path if config.get("retained_plan_file") == retained_path.name else data_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config, plan = retained_config(config, plan)
    contents = {
        builder: patch_builder(builder.read_text(encoding="utf-8")),
        preflight: patch_preflight(preflight.read_text(encoding="utf-8")),
        config_path: ("# Retained profiles/seeds/quotas come from retained_plan_file.\n"
                      "# Changing difficulty YAML alone will not override this saved plan.\n"
                      + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)),
        retained_path: json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    }
    print(json.dumps({
        "drop": ["games/puzzle24/medium", "games/puzzle24/hard"],
        "tasks": len({s["task"] for s in plan["specs"]}), "strata": len(plan["specs"]),
        "train_rows": sum(s["train_count"] for s in plan["specs"]),
        "validation_rows": sum(s["validation_count"] for s in plan["specs"]),
        "sha_verification": False, "data_dir": str(data_dir),
    }, indent=2))
    # This is just an inventory; the builder checks actual rows/profiles next.
    missing = []
    for spec in plan["specs"]:
        for split in ("train", "validation"):
            path = data_dir / "shards" / split / (spec["key"].replace("/", "__") + ".parquet")
            if not path.is_file() or not path.with_suffix(".manifest.json").is_file():
                missing.append(f"{split} {spec['key']}")
    print("[INFO] Retained shards missing parquet or manifest:", json.dumps(missing))
    if args.dry_run:
        print("[PASS] Patch matches current source; dry run made no changes.")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = root / "rg_patch_backups" / stamp
    originals = {path: path.read_bytes() if path.exists() else None for path in contents}
    for path, previous in originals.items():
        if previous is not None:
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    try:
        for path, text in contents.items():
            atomic_text(path, text)
    except Exception:
        for path, previous in originals.items():
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                shutil.copy2(backup / path.relative_to(root), path)
        raise
    print(f"[PASS] Patched current files; originals saved under {backup}")
    print("[INFO] Existing shards were not deleted or modified.")
    print("[NEXT] qsub jobs/0390_q3_rg_sixcat_build_v1.pbs")


if __name__ == "__main__":
    main()
