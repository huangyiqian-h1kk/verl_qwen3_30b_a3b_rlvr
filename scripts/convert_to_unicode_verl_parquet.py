#!/usr/bin/env python
import argparse
import json
import re
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


def unwrap(x):
    """Handle numpy/pandas object wrappers."""
    if isinstance(x, dict):
        return x
    return x


def get_from_obj(obj, key, default=None):
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return obj[key]
    except Exception:
        return default


def extract_text_from_prompt(prompt):
    """
    Accepts HF chat-style prompt:
      [{"role": "user", "content": "..."}]
    or plain string.
    Tries to remove common DAPO-style instruction.
    """
    if prompt is None:
        return None

    if isinstance(prompt, list):
        contents = []
        for m in prompt:
            if isinstance(m, dict):
                contents.append(str(m.get("content", "")))
            else:
                contents.append(str(m))
        text = "\n".join(contents)
    else:
        text = str(prompt)

    text = text.strip()

    # Remove common DAPO prompt boilerplate if present.
    patterns = [
        r"(?is)^Solve the following math problem step by step\.\s*",
        r"(?is)^Solve the problem step by step\.\s*",
        r"(?is)The last line of your response should be of the form\s+Answer:\s*\$Answer\.?\s*",
        r"(?is)The last line of your response should be:\s*Answer:\s*\$Answer\.?\s*",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text).strip()

    # If there is a "Problem:" marker, keep after it.
    m = re.search(r"(?is)\bProblem:\s*(.*)$", text)
    if m:
        text = m.group(1).strip()

    return text if text else None


def normalize_ground_truth(x):
    if x is None:
        return None

    s = str(x).strip()

    # GSM8K-like
    if "####" in s:
        s = s.split("####")[-1].strip()

    # DAPO-like
    m = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", s)
    if m:
        s = m[-1].strip()

    # Simple boxed answer; do not overdo nested parsing here.
    m = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if m:
        s = m[-1].strip()

    return s.strip() if s.strip() else None


def load_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")

    raise ValueError(f"Unsupported input format: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--val-size", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=16)
    args = ap.parse_args()

    raw_rows = load_rows(args.input)

    out_rows = []
    skipped = 0

    for i, obj in enumerate(raw_rows):
        if args.max_samples > 0 and len(out_rows) >= args.max_samples:
            break

        # Case 1: raw AceReason-like
        problem = get_from_obj(obj, "problem")
        answer = get_from_obj(obj, "answer")

        # Case 2: verl/DAPO-like
        if problem is None:
            problem = extract_text_from_prompt(get_from_obj(obj, "prompt"))

        if answer is None:
            rm = get_from_obj(obj, "reward_model")
            if isinstance(rm, dict):
                answer = rm.get("ground_truth")
            elif rm is not None:
                try:
                    answer = rm["ground_truth"]
                except Exception:
                    answer = None

        problem = str(problem).strip() if problem is not None else None
        answer = normalize_ground_truth(answer)

        if not problem or not answer:
            skipped += 1
            continue

        out_rows.append(
            {
                "data_source": args.source_name,
                "prompt": [
                    {"role": "system", "content": UNICODE_SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "index": i,
                    "source": args.source_name,
                    "prompt_style": "unicode_openr1_style",
                },
            }
        )

    if not out_rows:
        raise RuntimeError(f"No usable rows. skipped={skipped}")

    val_size = min(args.val_size, max(1, len(out_rows) // 4))
    val_rows = out_rows[:val_size]
    train_rows = out_rows[val_size:]

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(args.out_train, index=False)
    pd.DataFrame(val_rows).to_parquet(args.out_val, index=False)

    print(
        {
            "input": args.input,
            "source_name": args.source_name,
            "raw_rows_seen": len(raw_rows),
            "usable_rows": len(out_rows),
            "skipped": skipped,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "out_train": args.out_train,
            "out_val": args.out_val,
        }
    )


if __name__ == "__main__":
    main()
