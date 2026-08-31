#!/usr/bin/env python3
"""
build_unicode_verl_datasets.py

Build the FULL training + validation parquets for the unicode
(《reasoning》/《answer》) RLVR experiments, replacing the tiny smoke splits.

What it does, per source dataset (DAPO / AceReason):
  1. load raw rows (parquet or jsonl)
  2. extract clean problem text (strip known instruction boilerplate)
     and normalized ground truth
  3. **deduplicate by normalized problem text** -- essential: the HF
     DAPO-Math-17k parquet contains ~1.79M rows = ~17k unique problems
     repeated ~100x; without dedup a random split leaks holdout problems
     into train
  4. shuffle with a fixed seed and split: N holdout + rest train
  5. write verl-format parquets with the SAME schema / system prompt as
     scripts/convert_to_unicode_verl_parquet.py (smoke converter)

Also builds (optional, on by default):
  - gsm8k easy-anchor val set (from gsm8k test split)
  - aime24 benchmark val set (all 30 problems)
  - a cross-overlap report between DAPO and AceReason problem texts

Outputs (under --out-dir):
  dapo_unicode_train.parquet          dapo_unicode_holdout.parquet
  acereason_unicode_train.parquet     acereason_unicode_holdout.parquet
  gsm8k_unicode_anchor.parquet        aime24_unicode_val.parquet
  build_manifest.json                 (counts, seed, args -- provenance)

Example:
  python build_unicode_verl_datasets.py \
      --data-dir  $PROJ/data \
      --out-dir   $PROJ/data/full_v1 \
      --holdout-size 500 --seed 42
"""

import argparse
import json
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Must stay IDENTICAL to scripts/convert_to_unicode_verl_parquet.py
# (training and validation prompts must share one convention).
# ---------------------------------------------------------------------------
UNICODE_SYSTEM_PROMPT = """You are a reasoning assistant.

You must strictly respond in the following format:

《reasoning》
Write your reasoning process here.
《/reasoning》
《answer》
Write only the final answer here.
《/answer》"""


# ---------------------------------------------------------------------------
# problem-text extraction
# ---------------------------------------------------------------------------

# Instruction boilerplate seen in the wild; stripped from the problem text so
# the only instruction the model sees is our unicode system prompt.
PREFIX_PATTERNS = [
    r"(?is)^Solve the following math problem step by step\..*?the answer to the problem\.\s*",
    r"(?is)^Solve the following math problem step by step\.\s*",
    r"(?is)^Solve the problem step by step\.\s*",
]
SUFFIX_PATTERNS = [
    r"(?is)\s*Remember to put your answer on its own line.*$",
    r"(?is)\s*Let'?s think step by step and output the final answer within \\boxed\{\}\.?\s*$",
    r"(?is)\s*Please reason step by step,? and put your final answer within \\boxed\{\}\.?\s*$",
    r"(?is)\s*The last line of your response should be.*$",
]


def messages_to_text(prompt) -> str | None:
    """Chat-style prompt (list/ndarray of {role, content}) or str -> joined text."""
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    try:
        items = list(prompt)
    except TypeError:
        return str(prompt)
    contents = []
    for m in items:
        if isinstance(m, dict):
            contents.append(str(m.get("content", "")))
        else:
            contents.append(str(m))
    return "\n".join(contents)


def extract_problem(text: str | None) -> str | None:
    if text is None:
        return None
    text = str(text).strip()
    for pat in PREFIX_PATTERNS:
        text = re.sub(pat, "", text).strip()
    for pat in SUFFIX_PATTERNS:
        text = re.sub(pat, "", text).strip()
    m = re.search(r"(?is)\bProblem:\s*(.*)$", text)
    if m:
        text = m.group(1).strip()
    return text or None


def normalize_ground_truth(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if "####" in s:  # gsm8k
        s = s.split("####")[-1].strip()
    m = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", s)
    if m:
        s = m[-1].strip()
    m = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if m:
        s = m[-1].strip()
    return s.strip() or None


def dedup_key(problem: str) -> str:
    """Normalization for duplicate detection (NOT for output)."""
    s = unicodedata.normalize("NFKC", problem)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]*\$[ \t]*", "$", s)  # spacing around $ varies between dumps
    return s.strip()


# ---------------------------------------------------------------------------
# loaders: each yields (problem_text, ground_truth) raw pairs
# ---------------------------------------------------------------------------

def iter_dapo(path: Path):
    df = pd.read_parquet(path, columns=["prompt", "reward_model"])
    for prompt, rm in zip(df["prompt"], df["reward_model"]):
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        yield extract_problem(messages_to_text(prompt)), normalize_ground_truth(gt)


def iter_acereason(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield extract_problem(obj.get("problem")), normalize_ground_truth(obj.get("answer"))


def iter_gsm8k(path: Path):
    df = pd.read_parquet(path)
    for q, a in zip(df["question"], df["answer"]):
        yield extract_problem(q), normalize_ground_truth(a)


def iter_aime24(path: Path):
    df = pd.read_parquet(path)
    for prompt, rm in zip(df["prompt"], df["reward_model"]):
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        yield extract_problem(messages_to_text(prompt)), normalize_ground_truth(gt)


# ---------------------------------------------------------------------------
# core pipeline
# ---------------------------------------------------------------------------

def collect_unique(pairs, source_name: str):
    """Extract + validate + dedupe. Returns (unique_rows, stats)."""
    seen: dict[str, int] = {}
    rows = []
    n_raw = n_bad = n_dup = 0
    for problem, answer in pairs:
        n_raw += 1
        if not problem or not answer:
            n_bad += 1
            continue
        key = dedup_key(problem)
        if key in seen:
            n_dup += 1
            continue
        seen[key] = len(rows)
        rows.append({"problem": problem, "answer": answer, "key": key})
    stats = {
        "source": source_name,
        "raw_rows": n_raw,
        "unusable": n_bad,
        "duplicates_removed": n_dup,
        "unique_problems": len(rows),
    }
    return rows, stats


def to_verl_rows(rows, data_source: str, split: str):
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": UNICODE_SYSTEM_PROMPT},
                    {"role": "user", "content": r["problem"]},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": r["answer"]},
                "extra_info": {
                    "index": i,
                    "source": data_source,
                    "split": split,
                    "prompt_style": "unicode_openr1_style",
                },
            }
        )
    return out


def write_parquet(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return {"path": str(path), "rows": len(rows)}


def split_train_holdout(rows, holdout_size: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    if holdout_size >= len(rows):
        raise ValueError(f"holdout_size={holdout_size} >= unique rows={len(rows)}")
    hold = [rows[i] for i in idx[:holdout_size]]
    train = [rows[i] for i in idx[holdout_size:]]
    return train, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir containing the raw files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-size", type=int, default=500)
    ap.add_argument("--gsm8k-anchor-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dapo-file", default="dapo-math-17k.parquet")
    ap.add_argument("--ace-file", default="nvidia_AceReason-Math.jsonl")
    ap.add_argument("--gsm8k-test-file", default="gsm8k_test-00000-of-00001.parquet")
    ap.add_argument("--aime-file", default="aime24.parquet")
    ap.add_argument("--skip-gsm8k", action="store_true")
    ap.add_argument("--skip-aime", action="store_true")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "system_prompt_sha_hint": UNICODE_SYSTEM_PROMPT[:40],
        "sources": {},
        "outputs": [],
    }

    # ---- DAPO --------------------------------------------------------------
    dapo_rows, st = collect_unique(iter_dapo(data_dir / args.dapo_file), "dapo")
    print("[dapo]", st)
    manifest["sources"]["dapo"] = st
    dapo_train, dapo_hold = split_train_holdout(dapo_rows, args.holdout_size, args.seed)
    manifest["outputs"].append(write_parquet(
        to_verl_rows(dapo_train, "dapo_math_unicode", "train"),
        out_dir / "dapo_unicode_train.parquet"))
    manifest["outputs"].append(write_parquet(
        to_verl_rows(dapo_hold, "dapo_math_unicode", "holdout"),
        out_dir / "dapo_unicode_holdout.parquet"))

    # ---- AceReason ---------------------------------------------------------
    ace_rows, st = collect_unique(iter_acereason(data_dir / args.ace_file), "acereason")
    print("[acereason]", st)
    manifest["sources"]["acereason"] = st
    ace_train, ace_hold = split_train_holdout(ace_rows, args.holdout_size, args.seed)
    manifest["outputs"].append(write_parquet(
        to_verl_rows(ace_train, "acereason_unicode", "train"),
        out_dir / "acereason_unicode_train.parquet"))
    manifest["outputs"].append(write_parquet(
        to_verl_rows(ace_hold, "acereason_unicode", "holdout"),
        out_dir / "acereason_unicode_holdout.parquet"))

    # ---- cross-overlap report (info only; nothing is removed) --------------
    dapo_keys = {r["key"] for r in dapo_rows}
    ace_keys = {r["key"] for r in ace_rows}
    overlap = dapo_keys & ace_keys
    dapo_hold_leak = sum(1 for r in dapo_hold if r["key"] in ace_keys)
    ace_hold_leak = sum(1 for r in ace_hold if r["key"] in dapo_keys)
    cross = {
        "dapo_ace_shared_problems": len(overlap),
        "dapo_holdout_also_in_ace_train_pool": dapo_hold_leak,
        "ace_holdout_also_in_dapo_train_pool": ace_hold_leak,
        "note": "relevant only when using one dataset's holdout as the OTHER "
                "dataset's cross-val; same-dataset holdout purity is guaranteed "
                "by dedup+split.",
    }
    print("[cross-overlap]", cross)
    manifest["cross_overlap"] = cross

    # ---- gsm8k easy anchor -------------------------------------------------
    if not args.skip_gsm8k:
        g_rows, st = collect_unique(iter_gsm8k(data_dir / args.gsm8k_test_file), "gsm8k")
        print("[gsm8k]", st)
        manifest["sources"]["gsm8k"] = st
        rng = random.Random(args.seed)
        anchor = rng.sample(g_rows, min(args.gsm8k_anchor_size, len(g_rows)))
        manifest["outputs"].append(write_parquet(
            to_verl_rows(anchor, "gsm8k_unicode", "anchor"),
            out_dir / "gsm8k_unicode_anchor.parquet"))

    # ---- aime24 benchmark val ----------------------------------------------
    if not args.skip_aime:
        a_rows, st = collect_unique(iter_aime24(data_dir / args.aime_file), "aime24")
        print("[aime24]", st)
        manifest["sources"]["aime24"] = st
        manifest["outputs"].append(write_parquet(
            to_verl_rows(a_rows, "aime24_unicode", "benchmark"),
            out_dir / "aime24_unicode_val.parquet"))

    with (out_dir / "build_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for o in manifest["outputs"]:
        print(f"  {o['rows']:>7d}  {o['path']}")
    print(f"  manifest: {out_dir / 'build_manifest.json'}")


if __name__ == "__main__":
    main()
