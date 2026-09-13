#!/usr/bin/env python3
"""Deduplicate final RG outputs and check their actual counts in preflight.

Run from the repository after patch_rg_drop_puzzle24.py and
patch_rg_skip_integration_medium.py:
    python patch_rg_merge_unique.py
    qsub jobs/0390_q3_rg_sixcat_build_v1.pbs

Existing source files are backed up. Shards, generation quotas, YAML and saved
plans are unchanged. No regeneration or SHA verification is introduced.
Exact user-message text is the key, matching the builder's existing audit.
Validation has priority. Within each split the first row in plan/shard order
is retained, before the existing deterministic shuffle.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import os
from pathlib import Path
import shutil

MARKER = '# rg-global-unique-merge-v1'

MERGE_HELPER = r'''
def rg_unique_merge(split_frames, plan):
    """Preserve validation first, then retain disjoint unique training rows."""
    from collections import Counter
    report = {
        "policy": "exact_user_content; validation_first; first_in_plan_order",
        "splits": {},
    }
    validation_questions = set()
    for split in ("validation", "train"):
        frame = split_frames[split]
        kept = []
        seen = set()
        before, after = Counter(), Counter()
        duplicate, overlap = Counter(), Counter()
        for position, (messages, extra) in enumerate(zip(frame["prompt"], frame["extra_info"])):
            key = f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"
            question = str(messages[-1]["content"])
            before[key] += 1
            if split == "train" and question in validation_questions:
                overlap[key] += 1
                continue
            if question in seen:
                duplicate[key] += 1
                continue
            seen.add(question)
            kept.append(position)
            after[key] += 1
        if split == "validation":
            validation_questions = seen
        result = frame.iloc[kept].reset_index(drop=True)
        split_frames[split] = result
        per_stratum = {
            s["key"]: {
                "input_rows": before[s["key"]],
                "kept_rows": after[s["key"]],
                "removed_duplicate": duplicate[s["key"]],
                "removed_validation_overlap": overlap[s["key"]],
            }
            for s in plan["specs"]
        }
        report["splits"][split] = {
            "input_rows": len(frame), "kept_rows": len(result),
            "removed_duplicate": sum(duplicate.values()),
            "removed_validation_overlap": sum(overlap.values()),
            "per_stratum": per_stratum,
        }
        print(f"[DEDUP] {split}: {len(frame)} -> {len(result)}; "
              f"duplicates={sum(duplicate.values())}; "
              f"validation_overlap={sum(overlap.values())}", flush=True)
        for key, counts in per_stratum.items():
            if counts["kept_rows"] != counts["input_rows"]:
                print(f"[DEDUP] {split} {key}: {counts}", flush=True)
        if result.empty:
            raise RuntimeError(f"{split}: no rows remain after global deduplication")
    return report
'''

PREFLIGHT_HELPERS = r'''
def rg_actual_output_counts(manifest, expected_specs):
    """Check actual-count metadata against original generation upper bounds."""
    counts = {}
    if manifest.get("merge_policy") != "exact_user_content; validation_first; first_in_plan_order":
        raise SystemExit("[FAIL] rebuild final outputs with the global deduplication patch")
    for split in ("train", "validation"):
        info = manifest["outputs"][split]
        values = info.get("stratum_rows")
        if not isinstance(values, dict) or set(values) != set(expected_specs):
            raise SystemExit(f"[FAIL] {split} actual-count inventory differs from plan")
        for key, value in values.items():
            if type(value) is not int or not 0 <= value <= int(expected_specs[key][f"{split}_count"]):
                raise SystemExit(f"[FAIL] {split} {key}: invalid retained count {value!r}")
        if sum(values.values()) != int(info["rows"]) or int(info["rows"]) <= 0:
            raise SystemExit(f"[FAIL] {split} manifest total differs from stratum counts")
        counts[split] = values
    return counts


def rg_check_unique_outputs(frames, actual_counts, as_mapping, as_messages):
    from collections import Counter
    questions = {}
    ids = {}
    for split, frame in frames.items():
        observed = Counter()
        qs, sample_ids = [], []
        for extra_value, prompt in zip(frame["extra_info"], frame["prompt"]):
            extra = as_mapping(extra_value)
            key = f"{extra['rg_category']}/{extra['rg_task']}/{extra['rg_tier']}"
            observed[key] += 1
            qs.append(as_messages(prompt)[-1]["content"])
            sample_ids.append(str(extra["index"]))
        expected = {k: v for k, v in actual_counts[split].items() if v > 0}
        if dict(observed) != expected:
            raise SystemExit(f"[FAIL] {split} actual per-stratum counts differ from manifest")
        if len(set(qs)) != len(qs) or len(set(sample_ids)) != len(sample_ids):
            raise SystemExit(f"[FAIL] {split} contains duplicate prompts or sample IDs")
        questions[split], ids[split] = set(qs), set(sample_ids)
    if questions["train"] & questions["validation"]:
        raise SystemExit("[FAIL] train/validation prompt overlap")
    if ids["train"] & ids["validation"]:
        raise SystemExit("[FAIL] train/validation sample ID overlap")
'''


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Cannot locate expected code once: {old[:120]!r}; no files changed")
    return source.replace(old, new, 1)


def patch_builder(source):
    if MARKER in source:
        return source
    source = once(source, 'def merge_and_audit(', MERGE_HELPER.lstrip() + '\n\ndef merge_and_audit(')
    old = '''        random_state = int(config["dataset"]["shuffle_seed"]) + (0 if split == "train" else 1)
        split_frames[split] = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
'''
    new = '''        split_frames[split] = frame

    dedup_report = rg_unique_merge(split_frames, plan)
    for split, frame in split_frames.items():
        random_state = int(config["dataset"]["shuffle_seed"]) + (0 if split == "train" else 1)
        split_frames[split] = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
'''
    source = once(source, old, new)
    source = once(source, '            "rows": len(frame),\n', '''            "rows": len(frame),
            "stratum_rows": {
                key: counts["kept_rows"]
                for key, counts in dedup_report["splits"][split]["per_stratum"].items()
            },
''')
    source = once(source, '        "duplicate_train_prompts": 0,', '''        "deduplication": dedup_report,
        "actual_train_tasks": len(task_counts),
        "actual_train_strata": len(tier_counts),
        "duplicate_train_prompts": 0,''')
    source = once(source, '        "outputs": outputs,', '''        "merge_policy": dedup_report["policy"],
        "outputs": outputs,''')
    # Original assertions still check uniqueness and train/validation separation.
    source += '\n' + MARKER + '\n'
    compile(source, 'patched_builder.py', 'exec')
    return source


def patch_preflight(source):
    if MARKER in source:
        return source
    if '# rg-retained-plan-v1' not in source:
        raise RuntimeError('Expected the installed drop_puzzle24 preflight patch; no files changed')
    source = once(source, 'def main() -> None:', PREFLIGHT_HELPERS.lstrip() + '\n\ndef main() -> None:')
    source = once(source, '    frames = {}\n', '''    actual_counts = rg_actual_output_counts(manifest, expected_specs)
    frames = {}
''')
    source = once(source,
        '        expected_rows = sum(int(s[f"{split}_count"]) for s in expected_specs.values())',
        '        expected_rows = sum(actual_counts[split].values())')
    start = source.index('    if categories != expected_categories or tasks != expected_tasks or set(strata) != set(expected_specs):')
    end = source.index('    # Replay one strictly formatted frozen oracle per stratum', start)
    source = source[:start] + '''    rg_check_unique_outputs(frames, actual_counts, as_mapping, as_messages)

''' + source[end:]
    source += '\n' + MARKER + '\n'
    compile(source, 'patched_preflight.py', 'exec')
    return source


def atomic_text(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(value, encoding='utf-8')
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.repo_root.resolve()
    changes = {}
    for relative, patch in (
        ('scripts/build_reasoning_gym_sixcat_parquet.py', patch_builder),
        ('scripts/preflight_reasoning_gym_training.py', patch_preflight),
    ):
        path = root / relative
        source = path.read_text(encoding='utf-8')
        result = patch(source)
        if result != source:
            changes[path] = result
    if not changes:
        print('[PASS] Global deduplication patch already installed.')
        return
    if args.dry_run:
        print('[PASS] Both source patches match; dry run made no changes.')
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = root / 'rg_patch_backups' / f'merge_unique_{stamp}'
    for path in changes:
        saved = backup / path.relative_to(root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    try:
        for path, result in changes.items():
            atomic_text(path, result)
    except Exception:
        for path in changes:
            shutil.copy2(backup / path.relative_to(root), path)
        raise
    print(f'[PASS] Patched builder and preflight; backup: {backup}')
    print('[INFO] Shards, YAML and retained generation plan are unchanged.')
    print('[NEXT] qsub jobs/0390_q3_rg_sixcat_build_v1.pbs')


if __name__ == '__main__':
    main()
