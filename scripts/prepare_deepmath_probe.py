#!/usr/bin/env python3
"""
prepare_deepmath_probe.py

Build a difficulty-stratified probe set from zwhe99/DeepMath-103K for
baseline pass@k evaluation of the untrained model, using the SAME
unicode system prompt as training.

First download the dataset on a login node (internet access):

  export HF_HOME=$PROJ/hf_home
  huggingface-cli download zwhe99/DeepMath-103K --repo-type dataset \
      --local-dir $PROJ/data/raw_deepmath

Then:

  python prepare_deepmath_probe.py \
      --input  $PROJ/data/raw_deepmath \
      --out    $PROJ/data/deepmath_probe/deepmath_probe.parquet \
      --per-level 120 --seed 42

Outputs a verl-style parquet (prompt/reward_model/extra_info) whose
extra_info carries difficulty and topic, so the eval script can break
results down by difficulty level.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import pandas as pd

UNICODE_SYSTEM_PROMPT = """You are a reasoning assistant.

You must strictly respond in the following format:

《reasoning》
Write your reasoning process here.
《/reasoning》
《answer》
Write only the final answer here.
《/answer》"""

Q_KEYS = ["question", "problem", "prompt"]
A_KEYS = ["final_answer", "answer", "expected_answer"]
D_KEYS = ["difficulty", "level"]
T_KEYS = ["topic", "domain", "category"]


def pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip():
            return row[k]
    return None


def load_all(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    files = sorted(glob.glob(str(p / "**" / "*.parquet"), recursive=True)) if p.is_dir() else [str(p)]
    if not files:
        raise SystemExit(f"no parquet files found under {input_path}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"loaded {len(df)} rows from {len(files)} parquet file(s); columns: {df.columns.tolist()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="downloaded dataset dir (or one parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=120, help="problems sampled per difficulty level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-source", default="deepmath_unicode")
    args = ap.parse_args()

    df = load_all(args.input)
    rng = random.Random(args.seed)

    rows_by_level = {}
    n_bad = 0
    for _, r in df.iterrows():
        row = r.to_dict()
        q, a = pick(row, Q_KEYS), pick(row, A_KEYS)
        if not q or a is None or not str(a).strip():
            n_bad += 1
            continue
        d = pick(row, D_KEYS)
        try:
            level = int(float(d)) if d is not None else -1
        except (TypeError, ValueError):
            level = -1
        rows_by_level.setdefault(level, []).append(
            {"question": str(q).strip(), "answer": str(a).strip(),
             "difficulty": d, "topic": pick(row, T_KEYS)}
        )

    print(f"unusable rows: {n_bad}")
    print("difficulty histogram:",
          {k: len(v) for k, v in sorted(rows_by_level.items())})

    out_rows = []
    for level in sorted(rows_by_level):
        pool = rows_by_level[level]
        take = pool if len(pool) <= args.per_level else rng.sample(pool, args.per_level)
        for i, r in enumerate(take):
            out_rows.append({
                "data_source": args.data_source,
                "prompt": [
                    {"role": "system", "content": UNICODE_SYSTEM_PROMPT},
                    {"role": "user", "content": r["question"]},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": r["answer"]},
                "extra_info": {
                    "index": i,
                    "source": args.data_source,
                    "split": "probe",
                    "prompt_style": "unicode_openr1_style",
                    "difficulty": str(r["difficulty"]),
                    "difficulty_level": level,
                    "topic": str(r["topic"]),
                },
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(out, index=False)
    manifest = {
        "input": args.input, "out": str(out), "seed": args.seed,
        "per_level": args.per_level, "total_probe_rows": len(out_rows),
        "levels": {str(k): min(len(v), args.per_level) for k, v in sorted(rows_by_level.items())},
    }
    with (out.parent / "probe_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(out_rows)} probe rows -> {out}")
    print(json.dumps(manifest["levels"], indent=2))


if __name__ == "__main__":
    main()
