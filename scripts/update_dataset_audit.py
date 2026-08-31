#!/usr/bin/env python3
"""
update_dataset_audit.py

Add or refresh one dataset section in audit_v2.json (schema_version 1),
computing everything preflight_model_and_data.py verifies:
path / rows / sha256 / data_sources / unique_problem_keys /
duplicate_problem_keys / train_holdout_overlap.

Also cross-checks that every row's embedded system prompt is byte-equal
to the canonical config file (the audit's system_prompt_sha256 source),
so a drifted build cannot slip in.

The previous audit file is backed up as audit_v2.json.bak.<timestamp>.

Usage:
  python update_dataset_audit.py \
      --audit   $PROJ/data/full_v1/audit_v2.json \
      --tag     deepmath_l89 \
      --train   $PROJ/data/full_v1/deepmath_l89_unicode_train.parquet \
      --holdout $PROJ/data/full_v1/deepmath_l89_unicode_holdout.parquet
"""

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dedup_key(problem: str) -> str:
    s = unicodedata.normalize("NFKC", problem).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]*\$[ \t]*", "$", s)
    return s.strip()


def split_report(path: Path, canonical_prompt: str) -> tuple[dict, set]:
    df = pd.read_parquet(path)
    keys, sources = [], set()
    prompt_mismatch = 0
    for _, r in df.iterrows():
        msgs = list(r["prompt"])
        sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
        usr_msg = next((m for m in msgs if m.get("role") == "user"), None)
        if sys_msg is None or usr_msg is None:
            raise SystemExit(f"[FAIL] row without system/user message in {path}")
        if sys_msg["content"] != canonical_prompt:
            prompt_mismatch += 1
        keys.append(dedup_key(usr_msg["content"]))
        sources.add(str(r["data_source"]))
    if prompt_mismatch:
        raise SystemExit(
            f"[FAIL] {prompt_mismatch}/{len(df)} rows in {path} embed a system "
            "prompt that differs from the canonical config file. Rebuild the "
            "dataset from the canonical prompt before auditing."
        )
    key_set = set(keys)
    report = {
        "path": str(path.resolve()),
        "rows": int(len(df)),
        "sha256": sha256_file(path),
        "data_sources": sorted(sources),
        "unique_problem_keys": len(key_set),
        "duplicate_problem_keys": len(keys) - len(key_set),
    }
    return report, key_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    args = ap.parse_args()

    audit_path = Path(args.audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != 1:
        raise SystemExit(f"[FAIL] unexpected schema_version: {audit.get('schema_version')}")

    prompt_file = Path(audit["system_prompt_path"])
    canonical_prompt = prompt_file.read_text(encoding="utf-8").rstrip("\n")
    current_sha = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    if current_sha != audit.get("system_prompt_sha256"):
        raise SystemExit(
            "[FAIL] system prompt file hash changed since the audit was created: "
            f"audit={audit.get('system_prompt_sha256')}, current={current_sha}. "
            "Resolve the prompt drift first."
        )

    train_report, train_keys = split_report(Path(args.train), canonical_prompt)
    hold_report, hold_keys = split_report(Path(args.holdout), canonical_prompt)
    overlap = len(train_keys & hold_keys)

    section = {
        "train": train_report,
        "holdout": hold_report,
        "train_holdout_overlap": overlap,
    }
    if overlap:
        raise SystemExit(f"[FAIL] train/holdout overlap = {overlap}; refusing to audit a leaking split")
    if train_report["duplicate_problem_keys"] or hold_report["duplicate_problem_keys"]:
        raise SystemExit("[FAIL] duplicate problem keys inside a split; rebuild with dedup")

    backup = audit_path.with_suffix(
        audit_path.suffix + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(audit_path, backup)

    existed = args.tag in audit.get("datasets", {})
    audit.setdefault("datasets", {})[args.tag] = section
    audit["updated_at"] = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[PASS] audit {'updated' if existed else 'extended'}: tag={args.tag}")
    print(f"       train:   rows={train_report['rows']} sha256={train_report['sha256'][:16]}...")
    print(f"       holdout: rows={hold_report['rows']} sha256={hold_report['sha256'][:16]}...")
    print(f"       overlap={overlap}  backup={backup.name}")


if __name__ == "__main__":
    main()
