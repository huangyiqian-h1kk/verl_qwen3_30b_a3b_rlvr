#!/usr/bin/env python3
"""K-sample vLLM baseline using Reasoning Gym's task-specific verifiers."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reasoning_gym import get_score_answer_fn


STRICT_PATTERN = re.compile(
    r"^\s*《reasoning》\s*(?P<reasoning>\S(?:.*?\S)?)\s*"
    r"《/reasoning》\s*《answer》\s*(?P<answer>\S(?:.*?\S)?)\s*"
    r"《/answer》\s*$",
    re.S,
)
ANSWER_BLOCK_PATTERN = re.compile(r"《answer》\s*(?P<answer>.*?)\s*《/answer》", re.S)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*(.+?)\s*$")


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, dict):
            return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"expected dict-like value, got {type(value).__name__}")


def as_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"prompt is not a message list: {type(value).__name__}")
    return [{"role": str(as_dict(x)["role"]), "content": str(as_dict(x)["content"])} for x in value]


def last_balanced_box(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    pos = start + len(marker)
    begin = pos
    depth = 1
    while pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:pos].strip()
        pos += 1
    return None


def extract_answer(text: str) -> tuple[str | None, str, float]:
    strict = STRICT_PATTERN.match(text)
    if strict:
        candidate = strict.group("answer").strip()
        source = "strict"
        fmt = 1.0
    else:
        blocks = list(ANSWER_BLOCK_PATTERN.finditer(text))
        if blocks and blocks[-1].group("answer").strip():
            candidate = blocks[-1].group("answer").strip()
            source = "answer_block"
        else:
            boxed = last_balanced_box(text)
            if boxed:
                candidate, source = boxed, "boxed"
            else:
                lines = ANSWER_LINE_PATTERN.findall(text)
                candidate = lines[-1].strip() if lines else None
                source = "answer_line" if lines else "none"
        fmt = 0.0
    if candidate:
        boxed = last_balanced_box(candidate)
        if candidate.lstrip().startswith(r"\boxed{") and boxed is not None:
            candidate = boxed
        if len(candidate) >= 2 and candidate[0] == "$" and candidate[-1] == "$":
            candidate = candidate[1:-1].strip()
    return candidate, source, fmt


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_llm(args: argparse.Namespace) -> LLM:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.enable_expert_parallel:
        kwargs["enable_expert_parallel"] = True
    return LLM(**kwargs)


def summarize(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    by_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_index[rec["index"]].append(rec)
    groups = []
    for index, values in by_index.items():
        values = sorted(values, key=lambda x: x["sample"])[:k]
        if len(values) != k:
            raise RuntimeError(f"{index}: expected {k} samples, found {len(values)}")
        binary = [int(v["correct"]) for v in values]
        rg_scores = [float(v["rg_score"]) for v in values]
        total_scores = [float(v["total_reward"]) for v in values]
        formats = [float(v["format"]) for v in values]
        groups.append(
            {
                "index": index,
                "task": values[0]["task"],
                "tier": values[0]["tier"],
                "successes": sum(binary),
                "mean_rg_score": mean(rg_scores),
                "binary_mixed": 0 < sum(binary) < k,
                "rg_nonzero_variance": max(rg_scores) - min(rg_scores) > 1e-12,
                "total_nonzero_variance": max(total_scores) - min(total_scores) > 1e-12,
                "format_rate": mean(formats),
                "truncated_rate": mean(float(v["truncated"]) for v in values),
            }
        )

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        hist = {str(i): sum(g["successes"] == i for g in subset) for i in range(k + 1)}
        return {
            "N": n,
            "K": k,
            "rollout_accuracy": sum(g["successes"] for g in subset) / max(n * k, 1),
            "mean_rg_score": mean(g["mean_rg_score"] for g in subset) if subset else 0.0,
            "pass_at_K": sum(g["successes"] > 0 for g in subset) / max(n, 1),
            "all_correct_ratio": sum(g["successes"] == k for g in subset) / max(n, 1),
            "all_wrong_ratio": sum(g["successes"] == 0 for g in subset) / max(n, 1),
            "binary_mixed_group_ratio": sum(g["binary_mixed"] for g in subset) / max(n, 1),
            "rg_nonzero_variance_group_ratio": sum(g["rg_nonzero_variance"] for g in subset) / max(n, 1),
            "total_nonzero_variance_group_ratio": sum(g["total_nonzero_variance"] for g in subset) / max(n, 1),
            "format_rate": mean(g["format_rate"] for g in subset) if subset else 0.0,
            "truncated_rate": mean(g["truncated_rate"] for g in subset) if subset else 0.0,
            "success_hist": hist,
        }

    by_stratum = {}
    for task, tier in sorted({(g["task"], g["tier"]) for g in groups}):
        by_stratum[f"{task}/{tier}"] = aggregate([g for g in groups if g["task"] == task and g["tier"] == tier])
    return {"overall": aggregate(groups), "by_stratum": by_stratum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="records JSONL path")
    parser.add_argument("--max-samples", type=int, default=384)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--compare-k", type=int, default=8)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=11264)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--format-weight", type=float, default=0.2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.compare_k <= args.n):
        raise ValueError("compare-k must be between 1 and n")

    path = Path(args.data)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError("data must be .parquet or .jsonl")
    frame = frame.head(args.max_samples).copy()
    rows = frame.to_dict(orient="records")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    prompts = [render_prompt(tokenizer, as_messages(row["prompt"])) for row in rows]
    llm = build_llm(args)
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)
    if len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(rows)} prompts")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    score_fns: dict[str, Any] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for row, request_output in zip(rows, outputs):
            extra = as_dict(row["extra_info"])
            task = str(extra["rg_task"])
            tier = str(extra["rg_tier"])
            entry = json.loads(extra["rg_entry_json"])
            score_fn = score_fns.setdefault(task, get_score_answer_fn(task))
            if len(request_output.outputs) != args.n:
                raise RuntimeError(f"{extra['index']}: vLLM returned {len(request_output.outputs)} samples")
            for sample, candidate_output in enumerate(request_output.outputs):
                text = candidate_output.text
                prediction, extraction, fmt = extract_answer(text)
                score_error = None
                try:
                    rg_score = float(score_fn(prediction, entry)) if prediction is not None else 0.0
                    if not math.isfinite(rg_score):
                        raise ValueError(f"non-finite score {rg_score}")
                except Exception as exc:  # preserve the failed example for audit
                    rg_score = 0.0
                    score_error = f"{type(exc).__name__}: {exc}"
                correct = rg_score >= 1.0 - 1e-12
                token_count = len(candidate_output.token_ids)
                finish_reason = str(candidate_output.finish_reason or "")
                rec = {
                    "index": str(extra["index"]),
                    "sample": sample,
                    "task": task,
                    "tier": tier,
                    "question": entry["question"],
                    "ground_truth": as_dict(row["reward_model"])["ground_truth"],
                    "prediction": prediction,
                    "answer_extraction": extraction,
                    "rg_score": rg_score,
                    "correct": correct,
                    "format": fmt,
                    "total_reward": rg_score + args.format_weight * fmt,
                    "finish_reason": finish_reason,
                    "response_tokens": token_count,
                    "truncated": finish_reason == "length" or token_count >= args.max_tokens,
                    "score_error": score_error,
                    "text": text,
                }
                records.append(rec)
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summaries = {f"K={args.compare_k}": summarize(records, args.compare_k)}
    if args.n != args.compare_k:
        summaries[f"K={args.n}"] = summarize(records, args.n)
    overall_small = summaries[f"K={args.compare_k}"]["overall"]
    overall_large = summaries[f"K={args.n}"]["overall"]
    summary = {
        "model": args.model,
        "data": str(path.resolve()),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
            "seed": args.seed,
        },
        "summaries": summaries,
        "K_gain": {
            "from": args.compare_k,
            "to": args.n,
            "binary_mixed_group_ratio_delta": overall_large["binary_mixed_group_ratio"]
            - overall_small["binary_mixed_group_ratio"],
            "rg_nonzero_variance_group_ratio_delta": overall_large["rg_nonzero_variance_group_ratio"]
            - overall_small["rg_nonzero_variance_group_ratio"],
            "all_wrong_ratio_delta": overall_large["all_wrong_ratio"] - overall_small["all_wrong_ratio"],
            "all_correct_ratio_delta": overall_large["all_correct_ratio"] - overall_small["all_correct_ratio"],
        },
        "score_errors": sum(rec["score_error"] is not None for rec in records),
        "records": str(output_path.resolve()),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table_rows = []
    for label, content in summaries.items():
        for stratum, metrics in content["by_stratum"].items():
            table_rows.append({"K": int(label.split("=")[1]), "stratum": stratum, **metrics})
    table_path = output_path.with_suffix(".by_stratum.csv")
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
