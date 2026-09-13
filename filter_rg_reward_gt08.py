#!/usr/bin/env python3
"""Filter whole tasks using the initial validation of job 2238156.

Run from the repository root with the usual training Python environment:
    python filter_rg_reward_gt08.py

To repair metadata in outputs made by the previous script version:
    python filter_rg_reward_gt08.py --repair-metadata

The 21 exclusions below have task-mean initial reward strictly above 0.8.
Both train and validation are filtered. Original files are kept intact.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import re
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path


BASELINE_LOG = "0390_q3_rg6_k8_v2_2238156.pbs1_2026-09-06_235411.log"
# Each observed stratum has 8 validation examples. Average across a task's tiers.
# These are the logged reward/mean@1 values, including format reward.
EXCLUDED_TASK_REWARDS = {
    "algorithmic/base_conversion": 0.9781249612569809,
    "algorithmic/game_of_life": 0.856249961303547,
    "algorithmic/game_of_life_halting": 0.9781249612569809,
    "algorithmic/graph_color": 0.856249961303547,
    "algorithmic/letter_counting": 1.0499999523162842,
    "algorithmic/number_filtering": 0.9874999553430825,
    "algorithmic/string_splitting": 0.9718749672174454,
    "algorithmic/string_synthesis": 1.0499999523162842,
    "algorithmic/word_sequence_reversal": 1.0499999523162842,
    "arithmetic/basic_arithmetic": 1.0499999523162842,
    "arithmetic/chain_sum": 1.0499999523162842,
    "arithmetic/count_bits": 1.0499999523162842,
    "arithmetic/decimal_chain_sum": 0.8624999613966793,
    "arithmetic/fraction_simplification": 1.0499999523162842,
    "arithmetic/gcd": 1.0499999523162842,
    "arithmetic/gsm_symbolic": 0.9249999583698809,
    "arithmetic/lcm": 0.9874999553430825,
    "arithmetic/number_format": 1.0499999523162842,
    "arithmetic/prime_factorization": 1.0499999523162842,
    "games/mini_sudoku": 1.0468749552965164,
    "geometry/simple_geometry": 0.9249999583698809,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_pbs(source: str, output_dir: Path, train_rows: int) -> str:
    # Keep enough epochs available for the existing 500-step budget.
    epochs = max(2, math.ceil(500 / max(1, train_rows // 256)))
    source, count = re.subn(r"(?m)^DATA_DIR=.*$", "DATA_DIR=" + shlex.quote(str(output_dir)), source)
    if count != 1:
        raise ValueError("Expected one DATA_DIR assignment in the current PBS")
    prefix = "inst2507_rg_sixcat_k8_full_"
    if prefix not in source:
        raise ValueError("Cannot locate the existing experiment name in the PBS")
    source = source.replace(prefix, "inst2507_rg_sixcat_r08_k8_full_")
    source = re.sub(r"(?m)^TOTAL_EPOCHS=\d+", f"TOTAL_EPOCHS={epochs}", source)
    source, count = re.subn(r"trainer\.total_epochs=\d+", f"trainer.total_epochs={epochs}", source)
    if count != 1 or "trainer.total_training_steps=500" not in source:
        raise ValueError("Cannot locate the existing 500-step training budget")
    return source


def repair_metadata(root: Path) -> None:
    """Restore per-stratum counts from existing Parquets; never rewrite their rows."""
    import pyarrow.parquet as pq

    data = root / "data/reasoning_gym_sixcat_r08"
    manifest_path = data / "manifest.json"
    manifest = read_json(manifest_path)
    specs = {spec["key"]: spec for spec in read_json(data / "plan.json")["specs"]}
    for split in ("train", "validation"):
        info = manifest["outputs"][split]
        table = pq.read_table(Path(info["path"]), columns=["extra_info"])
        observed = Counter(
            f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"
            for extra in table["extra_info"].to_pylist()
        )
        if set(observed) - set(specs):
            raise SystemExit(f"[FAIL] {split} has strata absent from plan; metadata was not changed")
        if len(table) != int(info["rows"]):
            raise SystemExit(f"[FAIL] {split} actual total differs from manifest; metadata was not changed")
        values = {key: int(observed[key]) for key in specs}
        if any(value > int(specs[key][f"{split}_count"]) for key, value in values.items()):
            raise SystemExit(f"[FAIL] {split} actual count exceeds plan; metadata was not changed")
        info["stratum_rows"] = values
        print(f"[PASS] {split}: {len(table)} rows, {len(values)} stratum counts measured")

    # Run the installed preflight's exact metadata check before writing.
    path = root / "scripts/preflight_reasoning_gym_training.py"
    spec = importlib.util.spec_from_file_location("rg_metadata_repair_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.rg_actual_output_counts(manifest, specs)
    backup = manifest_path.with_name("manifest.before_stratum_rows_repair.json")
    if not backup.exists():
        shutil.copy2(manifest_path, backup)
    temporary = manifest_path.with_name("manifest.stratum_rows_repair.tmp")
    write_json(temporary, manifest)
    shutil.copymode(manifest_path, temporary)
    temporary.replace(manifest_path)
    print("[PASS] installed preflight actual-count check passed; manifest repaired")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repair-metadata", action="store_true",
                        help="repair existing r08 manifest counts without filtering again")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.repair_metadata:
        repair_metadata(root)
        return
    data = root / "data/reasoning_gym_sixcat_v1"
    out = root / "data/reasoning_gym_sixcat_r08"
    pbs = root / "jobs/0390_q3_rg_sixcat_full500_k8.pbs"
    new_pbs = root / "jobs/0390_q3_rg_sixcat_r08_full500_k8.pbs"
    if out.exists() or new_pbs.exists():
        raise SystemExit(f"[FAIL] output already exists: {out} or {new_pbs}; original files are untouched")
    preflight = (root / "scripts/preflight_reasoning_gym_training.py").read_text(encoding="utf-8")
    if "expected_specs" not in preflight:
        raise SystemExit("[FAIL] This script expects your current preflight with dynamic retained-plan counts")
    manifest = read_json(data / "manifest.json")
    parent_audit = read_json(data / "audit_report.json")
    plan = read_json(data / "plan.json")
    if manifest.get("status") != "COMPLETE" or parent_audit.get("status") != "PASS":
        raise SystemExit("[FAIL] source dataset is not complete")
    if not (data / "BUILD_COMPLETE").is_file():
        raise SystemExit("[FAIL] source BUILD_COMPLETE is missing")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("[FAIL] Activate your training conda environment (pyarrow is required)") from exc

    tables, counts, original_counts = {}, {}, {}
    for split, filename in (("train", "train_64k.parquet"), ("validation", "validation_fixed.parquet")):
        table = pq.read_table(data / filename)
        sources = table["data_source"].to_pylist()
        if any(not isinstance(s, str) or len(s.split("/")) != 4 or not s.startswith("reasoning_gym/") for s in sources):
            raise SystemExit("[FAIL] unexpected data_source format")
        keep = ["/".join(s.split("/")[1:3]) not in EXCLUDED_TASK_REWARDS for s in sources]
        tables[split] = table.filter(pa.array(keep))
        original_counts[split] = len(table)
        counts[split] = Counter(s.removeprefix("reasoning_gym/") for s, yes in zip(sources, keep) if yes)
        if not counts[split]:
            raise SystemExit(f"[FAIL] filtering would leave {split} empty")
    if set(counts["train"]) != set(counts["validation"]):
        raise SystemExit("[FAIL] retained train/validation stratum inventories differ")

    original_specs = {s["key"]: s for s in plan["specs"]}
    if not set(counts["train"]) <= original_specs.keys():
        raise SystemExit("[FAIL] source plan does not describe the Parquet strata")
    specs = []
    for key, old in original_specs.items():
        if key in counts["train"]:
            spec = copy.deepcopy(old)
            for split in tables:
                spec[f"{split}_count"] = counts[split][key]
            specs.append(spec)
    task_counts, category_counts = Counter(), Counter()
    for key, count in counts["train"].items():
        category, task, _tier = key.split("/")
        category_counts[category] += count
        task_counts[f"{category}/{task}"] += count

    new_pbs_source = make_pbs(pbs.read_text(encoding="utf-8"), out, len(tables["train"]))
    subprocess.run(["bash", "-n"], input=new_pbs_source, text=True, check=True)
    out.mkdir(parents=True)
    outputs = {}
    for split, table in tables.items():
        path = out / ("train_64k.parquet" if split == "train" else "validation_fixed.parquet")
        pq.write_table(table, path, compression="zstd")
        outputs[split] = {
            "path": str(path), "rows": len(table), "bytes": path.stat().st_size,
            "stratum_rows": {spec["key"]: int(counts[split][spec["key"]]) for spec in specs},
        }
        print(f"[PASS] {split}: {original_counts[split]} -> {len(table)} rows", flush=True)

    report = {
        "source_data_dir": str(data), "baseline_log": BASELINE_LOG,
        "selection": "whole task, initial reward/mean@1 > 0.8",
        "reward_includes_format": True, "excluded_task_rewards": EXCLUDED_TASK_REWARDS,
        "train_rows": len(tables["train"]), "validation_rows": len(tables["validation"]),
        "tasks": len(task_counts), "strata": len(specs), "categories": len(category_counts),
    }
    # All retained rows are unchanged subsets of the audited parent dataset.
    # No generators or verifiers are re-executed and no SHA checks are introduced.
    plan.update(specs=specs, dataset_version="reasoning_gym_sixcat_r08",
                category_train_counts=dict(category_counts), task_train_counts=dict(task_counts),
                reuse_policy="fixed subset selected by initial task reward; no generation or SHA checks",
                derivation=report)
    excluded = plan.setdefault("excluded_tasks", {})
    for key in EXCLUDED_TASK_REWARDS:
        category, task = key.split("/")
        names = excluded.setdefault(category, [])
        if task not in names:
            names.append(task)
    plan.pop("plan_sha256", None)
    write_json(out / "plan.json", plan)
    audit = {
        "status": "PASS", "schema_version": manifest["schema_version"],
        "reasoning_gym_version": manifest["reasoning_gym_version"],
        "audit_method": "unchanged subset of completed parent; counts measured after filtering",
        "parent_audit_report": str(data / "audit_report.json"),
        "native_verifier_checks": "inherited from parent dataset; not rerun by filter",
        "selected_categories": len(category_counts), "selected_tasks": len(task_counts),
        "strata": len(specs), "train_rows": len(tables["train"]),
        "validation_rows": len(tables["validation"]),
        "category_train_counts": dict(category_counts), "task_train_counts": dict(task_counts),
        "stratum_train_counts": dict(counts["train"]),
        "duplicate_train_prompts": 0, "duplicate_validation_prompts": 0,
        "train_validation_prompt_overlap": 0,
    }
    write_json(out / "audit_report.json", audit)
    manifest.update(dataset_version="reasoning_gym_sixcat_r08", outputs=outputs,
                    audit_report=str(out / "audit_report.json"), shard_manifests=0, derivation=report)
    manifest.pop("plan_sha256", None)
    write_json(out / "manifest.json", manifest)
    write_json(out / "filter_report.json", report)
    (out / "BUILD_COMPLETE").write_text("status=PASS\nmethod=filtered_parent_dataset\n", encoding="utf-8")
    new_pbs.write_text(new_pbs_source, encoding="utf-8")
    print(f"[PASS] retained {len(task_counts)} tasks / {len(specs)} strata")
    print(f"[PASS] PBS: {new_pbs}")
    print("Submit the filtered experiment with:")
    print("qsub -v RTYPE=rt_HF,ACTOR_LR=1e-6,RUN_VERSION=v1 " + str(new_pbs))


if __name__ == "__main__":
    main()
