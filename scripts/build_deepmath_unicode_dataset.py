#!/usr/bin/env python3
"""
build_deepmath_unicode_dataset.py

Build the DeepMath level-filtered training + holdout parquets for the
unicode RLVR arm, following the same conventions as full_v1
(dedup -> unicode prompt -> seeded split -> manifest).

Differences from build_unicode_verl_datasets.py:
  * reads the CANONICAL system prompt from
    $PROJ/config/unicode_system_prompt.txt (the file your run
    fingerprint hashes) instead of embedding a copy -- one source of
    truth for training data, val data, and fingerprint.
  * difficulty-level filter (--levels 8,9).

Usage (login node or anywhere with the raw download):
  python build_deepmath_unicode_dataset.py \
      --input  $PROJ/data/raw_deepmath \
      --out-dir $PROJ/data/full_v1 \
      --system-prompt-file $PROJ/config/unicode_system_prompt.txt \
      --levels 8,9 --holdout-size 500 --seed 42
"""

import argparse
import glob
import json
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


def dedup_key(problem: str) -> str:
    s = unicodedata.normalize("NFKC", problem).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]*\$[ \t]*", "$", s)
    return s.strip()


def load_raw(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(glob.glob(str(p / "**" / "*.parquet"), recursive=True)) if p.is_dir() else [str(p)]
    if not files:
        raise SystemExit(f"no parquet files under {input_path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(files)} file(s); columns: {df.columns.tolist()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--system-prompt-file", required=True,
                    help="canonical unicode system prompt (fingerprinted file)")
    ap.add_argument("--levels", default="8,9", help="comma-separated difficulty levels to keep")
    ap.add_argument("--holdout-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-source", default="deepmath_l89_unicode")
    ap.add_argument("--prefix", default="deepmath_l89_unicode")
    args = ap.parse_args()

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").rstrip("\n")
    if "《reasoning》" not in system_prompt or "《answer》" not in system_prompt:
        raise SystemExit(f"[FAIL] {args.system_prompt_file} does not look like the unicode prompt")

    levels = {int(x) for x in args.levels.split(",") if x.strip()}
    df = load_raw(args.input)

    seen, rows = set(), []
    n_raw = n_bad = n_dup = n_offlevel = 0
    for _, r in df.iterrows():
        n_raw += 1
        q = str(r.get("question") or "").strip()
        a = str(r.get("final_answer") if r.get("final_answer") is not None else "").strip()
        if not q or not a:
            n_bad += 1
            continue
        try:
            level = int(float(r.get("difficulty")))
        except (TypeError, ValueError):
            n_bad += 1
            continue
        if level not in levels:
            n_offlevel += 1
            continue
        key = dedup_key(q)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        rows.append({"question": q, "answer": a, "level": level,
                     "topic": str(r.get("topic") or "")})

    stats = {"raw_rows": n_raw, "unusable": n_bad, "off_level": n_offlevel,
             "duplicates_removed": n_dup, "kept_unique": len(rows),
             "levels": sorted(levels)}
    print("[deepmath]", stats)
    if len(rows) <= args.holdout_size:
        raise SystemExit(f"[FAIL] only {len(rows)} rows after filtering; holdout={args.holdout_size}")

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    hold = [rows[i] for i in idx[:args.holdout_size]]
    train = [rows[i] for i in idx[args.holdout_size:]]

    def to_verl(items, split):
        return [{
            "data_source": args.data_source,
            "prompt": [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": r["question"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": r["answer"]},
            "extra_info": {"index": i, "source": args.data_source, "split": split,
                           "prompt_style": "unicode_openr1_style",
                           "difficulty_level": r["level"], "topic": r["topic"]},
        } for i, r in enumerate(items)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{args.prefix}_train.parquet"
    hold_path = out_dir / f"{args.prefix}_holdout.parquet"
    pd.DataFrame(to_verl(train, "train")).to_parquet(train_path, index=False)
    pd.DataFrame(to_verl(hold, "holdout")).to_parquet(hold_path, index=False)

    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args), "stats": stats,
        "system_prompt_file": args.system_prompt_file,
        "level_histogram_kept": {
            str(l): sum(1 for r in rows if r["level"] == l) for l in sorted(levels)},
        "outputs": {"train": {"path": str(train_path), "rows": len(train)},
                    "holdout": {"path": str(hold_path), "rows": len(hold)}},
    }
    with (out_dir / f"{args.prefix}_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"train:   {len(train):>6d} -> {train_path}")
    print(f"holdout: {len(hold):>6d} -> {hold_path}")
    print(f"manifest -> {out_dir / (args.prefix + '_manifest.json')}")


if __name__ == "__main__":
    main()
