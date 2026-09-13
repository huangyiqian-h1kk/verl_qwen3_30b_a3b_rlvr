#!/usr/bin/env python3
"""Remove only algebra/intermediate_integration/medium from the retained plan.

Run after patch_rg_drop_puzzle24.py, from the repository root:
    python patch_rg_skip_integration_medium.py
    qsub jobs/0390_q3_rg_sixcat_build_v1.pbs

Preserves all other profiles, seeds, quotas, code and existing shards.
The builder writes the updated output plan when the PBS job runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import yaml

TARGET = "algebra/intermediate_integration/medium"
KEEP = "algebra/intermediate_integration/hard"


def update_plan(config: dict, plan: dict) -> tuple[dict, dict]:
    config, plan = copy.deepcopy(config), copy.deepcopy(plan)
    keys = [s["key"] for s in plan["specs"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate stratum keys in retained plan")
    if KEEP not in keys:
        raise ValueError(f"Expected the retained hard stratum: {KEEP}")
    if TARGET not in keys and TARGET not in plan.get("excluded_strata", []):
        raise ValueError(f"Cannot find {TARGET} in this plan")
    plan["specs"] = [s for s in plan["specs"] if s["key"] != TARGET]
    excluded = plan.setdefault("excluded_strata", [])
    if TARGET not in excluded:
        excluded.append(TARGET)
    categories, tasks = Counter(), Counter()
    for spec in plan["specs"]:
        count = int(spec["train_count"])
        categories[spec["category"]] += count
        tasks[f"{spec['category']}/{spec['task']}"] += count
    selected = {(c, t) for c, names in config["categories"].items() for t in names}
    retained = {(s["category"], s["task"]) for s in plan["specs"]}
    if selected != retained:
        raise ValueError("YAML task selection and retained plan disagree")
    plan["category_train_counts"] = dict(categories)
    plan["task_train_counts"] = dict(tasks)
    config["dataset"]["train_size"] = sum(categories.values())
    suffix = "_skip_integration_medium_v1"
    if not config["version"].endswith(suffix):
        config["version"] += suffix
    plan["dataset_version"] = config["version"]
    # The builder will calculate new descriptive metadata, without comparing
    # it against the hashes stored in existing shards.
    plan.pop("plan_sha256", None)
    return config, plan


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("retained_plan_file"):
        raise SystemExit("[FAIL] This follow-up requires the previous puzzle24 retained-plan patch.")
    plan_path = (config_path.parent / config["retained_plan_file"]).resolve()
    plan_path.relative_to(root)
    for name in ("build_reasoning_gym_sixcat_parquet.py", "preflight_reasoning_gym_training.py"):
        if "# rg-retained-plan-v1" not in (root / "scripts" / name).read_text(encoding="utf-8"):
            raise SystemExit(f"[FAIL] The previous retained-plan patch is missing from {name}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config, plan = update_plan(config, plan)
    print(json.dumps({
        "excluded_stratum": TARGET,
        "tasks": len({s["task"] for s in plan["specs"]}),
        "strata": len(plan["specs"]),
        "train_rows": sum(s["train_count"] for s in plan["specs"]),
        "validation_rows": sum(s["validation_count"] for s in plan["specs"]),
    }, indent=2))
    missing = []
    for spec in plan["specs"]:
        for split in ("train", "validation"):
            parquet = data_dir / "shards" / split / (spec["key"].replace("/", "__") + ".parquet")
            if not parquet.is_file() or not parquet.with_suffix(".manifest.json").is_file():
                missing.append(f"{split} {spec['key']}")
    print("[INFO] Retained shards missing parquet or manifest:", json.dumps(missing))
    if args.dry_run:
        print("[PASS] Dry run; no files changed.")
        return
    contents = {
        config_path: (
            "# Profiles, seeds and quotas come from retained_plan_file.\n"
            "# intermediate_integration/medium is omitted; hard is retained.\n"
            + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        ),
        plan_path: json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    }
    # Repeat execution is harmless and does not create unnecessary backups.
    if all(path.read_text(encoding="utf-8") == text for path, text in contents.items()):
        print("[PASS] Already applied.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = root / "rg_patch_backups" / stamp
    for path in contents:
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    try:
        for path, text in contents.items():
            atomic_text(path, text)
    except Exception:
        for path in contents:
            shutil.copy2(backup / path.relative_to(root), path)
        raise
    print(f"[PASS] Updated YAML and retained plan. Backup: {backup}")
    print("[NEXT] qsub jobs/0390_q3_rg_sixcat_build_v1.pbs")


if __name__ == "__main__":
    main()
