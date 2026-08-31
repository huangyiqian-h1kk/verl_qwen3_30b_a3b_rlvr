#!/usr/bin/env python3
"""
deepmath_baseline_passk.py

Baseline pass@k evaluation of an (untrained) model on the DeepMath
probe set, judged with the shared unicode_math_verifier
(math-verify primary + v3 fallback).

Defaults reproduce the TRAINING rollout distribution (temperature=1.0,
top_p=1.0) so the measured pass rates are exactly what GRPO will see;
override --temperature/--top-p for other measurement conventions.

Outputs:
  <out>.jsonl        one record per (problem, sample): verdicts, lengths
  <out>.summary.json aggregate: mean acc, unbiased pass@k, per-difficulty
                     table, format compliance, truncation rate

Example (single node, 8 GPUs):
  python deepmath_baseline_passk.py \
      --model $WORK/models/Qwen3-30B-A3B-Instruct-2507 \
      --probe $PROJ/data/deepmath_probe/deepmath_probe.parquet \
      --out   $PROJ/outputs/deepmath_baseline/inst2507 \
      --n 8 --max-tokens 8192 --tp 8
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unicode_math_verifier import judge, HAS_MATH_VERIFY  # noqa: E402


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Codex paper): 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N problems")
    args = ap.parse_args()

    assert HAS_MATH_VERIFY, "pip install math-verify --break-system-packages first"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.probe)
    if args.limit:
        df = df.head(args.limit)
    print(f"probe: {len(df)} problems from {args.probe}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, meta = [], []
    for _, r in df.iterrows():
        msgs = [dict(m) for m in r["prompt"]]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        ei = dict(r["extra_info"]) if r["extra_info"] is not None else {}
        meta.append({
            "ground_truth": r["reward_model"]["ground_truth"],
            "difficulty_level": int(ei.get("difficulty_level", -1)),
            "topic": ei.get("topic", ""),
        })

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              trust_remote_code=True, gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)
    print(f"generating: n={args.n} T={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}")
    outs = llm.generate(prompts, sp)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = open(f"{out_prefix}.jsonl", "w", encoding="utf-8")

    per_problem = []
    for i, o in enumerate(outs):
        gt = meta[i]["ground_truth"]
        c = fmt_ok = trunc = 0
        for s_idx, comp in enumerate(o.outputs):
            v = judge(comp.text, gt)
            c += int(v["correct"])
            fmt_ok += int(v["answer_block_found"])
            truncated = comp.finish_reason == "length"
            trunc += int(truncated)
            jsonl.write(json.dumps({
                "problem_idx": i, "sample_idx": s_idx,
                "correct": v["correct"], "via": v["via"],
                "answer_block_found": v["answer_block_found"],
                "truncated": truncated,
                "gen_len": len(comp.token_ids),
                "difficulty_level": meta[i]["difficulty_level"],
                "topic": meta[i]["topic"],
                "ground_truth": gt,
                "response_tail": comp.text[-600:],
            }, ensure_ascii=False) + "\n")
        per_problem.append({
            "n": len(o.outputs), "c": c,
            "level": meta[i]["difficulty_level"],
            "fmt": fmt_ok, "trunc": trunc,
        })
    jsonl.close()

    # ---- aggregate ----
    N = len(per_problem)
    n_samp = args.n
    ks = [k for k in (1, 2, 4, 8, 16) if k <= n_samp]
    summary = {
        "model": args.model, "probe": args.probe, "n": n_samp,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "num_problems": N,
        "mean_acc": sum(p["c"] / p["n"] for p in per_problem) / N,
        "format_compliance": sum(p["fmt"] for p in per_problem) / (N * n_samp),
        "truncation_rate": sum(p["trunc"] for p in per_problem) / (N * n_samp),
    }
    for k in ks:
        summary[f"pass@{k}"] = sum(pass_at_k(p["n"], p["c"], k) for p in per_problem) / N

    by_level = defaultdict(list)
    for p in per_problem:
        by_level[p["level"]].append(p)
    summary["by_difficulty"] = {}
    for lvl in sorted(by_level):
        ps = by_level[lvl]
        entry = {
            "num_problems": len(ps),
            "mean_acc": sum(p["c"] / p["n"] for p in ps) / len(ps),
            "all_correct_frac": sum(1 for p in ps if p["c"] == p["n"]) / len(ps),
            "all_wrong_frac": sum(1 for p in ps if p["c"] == 0) / len(ps),
            "mixed_group_frac": sum(1 for p in ps if 0 < p["c"] < p["n"]) / len(ps),
        }
        for k in ks:
            entry[f"pass@{k}"] = sum(pass_at_k(p["n"], p["c"], k) for p in ps) / len(ps)
        summary["by_difficulty"][str(lvl)] = entry

    with open(f"{out_prefix}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY ================")
    print(f"problems={N}  mean_acc={summary['mean_acc']:.3f}  "
          f"format={summary['format_compliance']:.3f}  trunc={summary['truncation_rate']:.3f}")
    for k in ks:
        print(f"pass@{k}: {summary[f'pass@{k}']:.3f}")
    print(f"{'lvl':>4} {'#prob':>6} {'acc':>6} {'pass@8':>7} {'allC':>6} {'allW':>6} {'mixed':>6}")
    for lvl, e in summary["by_difficulty"].items():
        print(f"{lvl:>4} {e['num_problems']:>6} {e['mean_acc']:>6.3f} "
              f"{e.get('pass@8', float('nan')):>7.3f} {e['all_correct_frac']:>6.2f} "
              f"{e['all_wrong_frac']:>6.2f} {e['mixed_group_frac']:>6.2f}")
    print(f"\njsonl:   {out_prefix}.jsonl")
    print(f"summary: {out_prefix}.summary.json")


if __name__ == "__main__":
    main()
