#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

PROMPTS = {
    "native": """You are a reasoning assistant. You must strictly respond in the following format:\n<think>\nWrite your reasoning process here.\n</think>\n<answer>\nWrite only the final answer here.\n</answer>""",
    "unicode": """You are a reasoning assistant. You must strictly respond in the following format:\n《reasoning》\nWrite your reasoning process here.\n《/reasoning》\n《answer》\nWrite only the final answer here.\n《/answer》""",
    "chinese": """你是一个推理助手。必须严格按照以下格式回答：\n推理过程：\n这里写必要推理。\n最终答案：\n这里只写最终答案。""",
    "angle": """You are a reasoning assistant. You must strictly respond in the following format:\n<reasoning>\nWrite your reasoning process here.\n</reasoning>\n<final_answer>\nWrite only the final answer here.\n</final_answer>""",
}

PROBLEMS = [
    ("What is 2+2?", "4"),
    ("If a box has 3 red balls and 5 blue balls, how many balls are there?", "8"),
    ("Compute 7 minus 4.", "3"),
    ("What is 6 times 6?", "36"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", choices=list(PROMPTS), default="unicode")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (problem, answer) in enumerate(PROBLEMS):
        rows.append({
            "data_source": f"tiny_format_{args.format}",
            "prompt": [
                {"role": "system", "content": PROMPTS[args.format]},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": i, "format": args.format, "problem": problem},
        })
    df = pd.DataFrame(rows)
    train = out / "train.parquet"
    val = out / "val.parquet"
    df.iloc[:2].to_parquet(train, index=False)
    df.iloc[2:].to_parquet(val, index=False)
    print(f"Wrote {train} rows=2")
    print(f"Wrote {val} rows=2")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
