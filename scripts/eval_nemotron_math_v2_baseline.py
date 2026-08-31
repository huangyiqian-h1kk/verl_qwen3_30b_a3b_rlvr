#!/usr/bin/env python3
"""K-sample vLLM baseline using the same custom verifier as GRPO training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reward_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("baseline_reward_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "compute_score", None)):
        raise RuntimeError(f"compute_score not found in {path}")
    if callable(getattr(module, "_self_test", None)):
        module._self_test()
    return module


def render_prompt(tokenizer: Any, system_prompt: str, problem: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def classify_group(successes: int, k: int) -> str:
    if successes == 0:
        return "all_wrong"
    if successes == k:
        return "all_correct"
    if 1 <= successes <= min(6, k - 1):
        return "primary_eligible"
    return "borderline_easy"


def aggregate(groups: list[dict[str, Any]], k: int) -> dict[str, Any]:
    if not groups:
        return {"num_prompts": 0, "K": k}
    successes = [int(group["success_count"]) for group in groups]
    formats = [int(group["format_count"]) for group in groups]
    reward_values = [
        float(value)
        for group in groups
        for value in group["sample_rewards"]
    ]
    response_tokens = [
        int(value)
        for group in groups
        for value in group["response_tokens"]
    ]
    truncated = [
        bool(value)
        for group in groups
        for value in group["truncated"]
    ]
    total = len(groups) * k
    accuracy = sum(successes) / total
    mixed = sum(0 < value < k for value in successes) / len(groups)
    summary = {
        "num_prompts": len(groups),
        "K": k,
        "num_completions": total,
        "accuracy_mean_over_NK": accuracy,
        "first_sample_accuracy": (
            sum(bool(group["first_sample_correct"]) for group in groups)
            / len(groups)
        ),
        "pass_at_K": sum(value > 0 for value in successes) / len(groups),
        "all_correct_ratio": sum(value == k for value in successes) / len(groups),
        "all_wrong_ratio": sum(value == 0 for value in successes) / len(groups),
        "mixed_group_ratio": mixed,
        "format_rate": sum(formats) / total,
        "mean_reward": statistics.fmean(reward_values),
        "response_tokens_mean": statistics.fmean(response_tokens),
        "response_tokens_p50": percentile(response_tokens, 0.50),
        "response_tokens_p90": percentile(response_tokens, 0.90),
        "response_tokens_p95": percentile(response_tokens, 0.95),
        "truncation_rate": sum(truncated) / total,
        "success_hist": {
            str(value): successes.count(value) for value in range(k + 1)
        },
    }
    return summary


def verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    accuracy = float(metrics["accuracy_mean_over_NK"])
    mixed = float(metrics["mixed_group_ratio"])
    all_correct = float(metrics["all_correct_ratio"])
    all_wrong = float(metrics["all_wrong_ratio"])
    if (
        0.10 <= accuracy <= 0.65
        and mixed >= 0.60
        and all_correct <= 0.25
        and all_wrong <= 0.50
    ):
        label = "SUITABLE"
        explanation = (
            "Difficulty is in the intended GRPO range and most prompt groups "
            "contain both positive and negative rollouts."
        )
    elif accuracy > 0.65 or all_correct > 0.30:
        label = "TOO_EASY_UNFILTERED"
        explanation = (
            "The unfiltered sample is near the policy ceiling; do not use it "
            "directly for a long GRPO run."
        )
    elif accuracy < 0.10 or all_wrong > 0.60:
        label = "TOO_HARD_UNFILTERED"
        explanation = (
            "Too many groups lack a positive rollout; use a less difficult "
            "stratum or retain only model-solvable groups."
        )
    elif mixed >= 0.40:
        label = "USABLE_AFTER_DIFFICULTY_FILTERING"
        explanation = (
            "The pool has useful signal but requires offline filtering before "
            "training."
        )
    else:
        label = "BORDERLINE"
        explanation = "The observed group composition is not yet reliable for GRPO."
    return {
        "label": label,
        "explanation": explanation,
        "primary_filter": "retain success_count in [1, 6] out of K=8",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reward-file", type=Path, required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=11264)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n <= 1:
        raise SystemExit("--n must be greater than one for group analysis")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")
    for required in [args.data, args.reward_file, args.system_prompt_file]:
        if not required.is_file():
            raise SystemExit(f"required file not found: {required}")

    frame = pd.read_parquet(args.data)
    required_columns = {"uid", "problem", "expected_answer", "data_source"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise SystemExit(f"dataset is missing columns: {sorted(missing)}")
    if frame["uid"].duplicated().any():
        raise SystemExit("dataset contains duplicate uid values")
    if args.max_samples > 0:
        if len(frame) < args.max_samples:
            raise SystemExit(
                f"dataset has {len(frame)} rows, fewer than "
                f"--max-samples={args.max_samples}"
            )
        if len(frame) > args.max_samples:
            frame = frame.sample(
                n=args.max_samples, random_state=args.seed
            ).reset_index(drop=True)
        else:
            frame = frame.copy()
    if frame.empty:
        raise SystemExit("dataset contains no rows")

    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise SystemExit("system prompt is empty")
    reward_module = load_reward_module(args.reward_file)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    prompts: list[str] = []
    prompt_lengths: list[int] = []
    records = frame.to_dict(orient="records")
    too_long: list[tuple[str, int]] = []
    verifier_failures: list[str] = []
    for row in records:
        prompt = render_prompt(tokenizer, system_prompt, str(row["problem"]))
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if length > args.max_prompt_tokens:
            too_long.append((str(row["uid"]), length))
            continue
        gold_probe = (
            "《reasoning》\nGold-answer verifier self-check.\n《/reasoning》\n"
            f"《answer》\n{row['expected_answer']}\n《/answer》"
        )
        gold_score = reward_module.compute_score(
            data_source=str(row["data_source"]),
            solution_str=gold_probe,
            ground_truth=row["expected_answer"],
            extra_info={"uid": row["uid"], "baseline_self_check": True},
        )
        if not bool(gold_score.get("acc", False)):
            verifier_failures.append(str(row["uid"]))
            continue
        prompts.append(prompt)
        prompt_lengths.append(length)

    if too_long or verifier_failures:
        detail = {
            "prompt_too_long": too_long[:20],
            "verifier_self_check_failed": verifier_failures[:20],
        }
        raise SystemExit(
            "dataset preflight failed; regenerate the sample or inspect reward "
            f"compatibility:\n{json.dumps(detail, ensure_ascii=False, indent=2)}"
        )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling)
    if len(outputs) != len(records):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} prompt outputs for {len(records)} prompts"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, Any]] = []
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.out.open("w", encoding="utf-8") as handle:
        for row, prompt_len, request_output in zip(
            records, prompt_lengths, outputs
        ):
            if len(request_output.outputs) != args.n:
                raise RuntimeError(
                    f"uid={row['uid']} returned {len(request_output.outputs)} "
                    f"samples, expected {args.n}"
                )
            sample_correct: list[bool] = []
            sample_formats: list[bool] = []
            sample_rewards: list[float] = []
            response_tokens: list[int] = []
            truncated: list[bool] = []
            for sample_index, candidate in enumerate(request_output.outputs):
                text = candidate.text
                score = reward_module.compute_score(
                    data_source=str(row["data_source"]),
                    solution_str=text,
                    ground_truth=row["expected_answer"],
                    extra_info={
                        "uid": row["uid"],
                        "baseline_index": row.get("baseline_index"),
                        "sample_index": sample_index,
                    },
                )
                correct = bool(score.get("acc", False))
                format_ok = bool(score.get("format", False))
                token_count = len(candidate.token_ids)
                finish_reason = str(candidate.finish_reason or "")
                is_truncated = (
                    finish_reason == "length" or token_count >= args.max_tokens
                )
                sample_correct.append(correct)
                sample_formats.append(format_ok)
                sample_rewards.append(float(score.get("score", 0.0)))
                response_tokens.append(token_count)
                truncated.append(is_truncated)
                output_record = {
                    "uid": row["uid"],
                    "baseline_index": row.get("baseline_index"),
                    "sample_index": sample_index,
                    "data_source": row["data_source"],
                    "problem": row["problem"],
                    "ground_truth": row["expected_answer"],
                    "prompt_tokens": prompt_len,
                    "response_tokens": token_count,
                    "finish_reason": finish_reason,
                    "truncated": is_truncated,
                    "correct": correct,
                    "format_ok": format_ok,
                    "reward": sample_rewards[-1],
                    "pred": score.get("pred"),
                    "answer_extraction": score.get("answer_extraction"),
                    "text": text,
                }
                handle.write(
                    json.dumps(output_record, ensure_ascii=False, default=str)
                    + "\n"
                )
            success_count = sum(sample_correct)
            group = {
                "uid": row["uid"],
                "data_source": str(row["data_source"]),
                "success_count": success_count,
                "format_count": sum(sample_formats),
                "first_sample_correct": sample_correct[0],
                "sample_rewards": sample_rewards,
                "response_tokens": response_tokens,
                "truncated": truncated,
                "difficulty_class": classify_group(success_count, args.n),
            }
            groups.append(group)
            by_source[group["data_source"]].append(group)

    overall = aggregate(groups, args.n)
    source_metrics = {
        source: aggregate(source_groups, args.n)
        for source, source_groups in sorted(by_source.items())
    }
    decision = verdict(overall)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "reward_file": str(args.reward_file),
        "reward_sha256": sha256_file(args.reward_file),
        "system_prompt_file": str(args.system_prompt_file),
        "system_prompt_sha256": sha256_file(args.system_prompt_file),
        "sampling": {
            "N": len(records),
            "K": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "seed": args.seed,
        },
        "overall": overall,
        "by_data_source": source_metrics,
        "decision": decision,
        "outputs": {
            "rollouts_jsonl": str(args.out),
            "groups_parquet": str(args.out.with_suffix(".groups.parquet")),
        },
    }

    group_frame = frame.copy()
    group_map = {str(group["uid"]): group for group in groups}
    group_frame["success_count"] = group_frame["uid"].map(
        lambda uid: group_map[str(uid)]["success_count"]
    )
    group_frame["correct_rate"] = group_frame["success_count"] / args.n
    group_frame["format_count"] = group_frame["uid"].map(
        lambda uid: group_map[str(uid)]["format_count"]
    )
    group_frame["difficulty_class"] = group_frame["uid"].map(
        lambda uid: group_map[str(uid)]["difficulty_class"]
    )
    group_path = args.out.with_suffix(".groups.parquet")
    group_frame.to_parquet(group_path, index=False)
    summary["outputs"]["groups_sha256"] = sha256_file(group_path)

    summary_path = args.out.with_suffix(".summary.json")
    summary["outputs"]["summary_json"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
