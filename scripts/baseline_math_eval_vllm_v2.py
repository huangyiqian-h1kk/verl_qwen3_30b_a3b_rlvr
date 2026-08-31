#!/usr/bin/env python3
"""
Baseline pass@K / mixed-group probe for RLVR math data with vLLM.

Key improvements over the first draft:
  - Supports model aliases: inst, a3b, thinking, qwen3_4b, etc.
  - Resolves local model paths from $WORK/models or explicit --model-path.
  - Uses local files by default for tokenizer/model paths.
  - Robustly extracts GSM8K-style ground truth after ####.
  - Supports verl-style parquet with reward_model.ground_truth.
  - Writes an exact config JSON next to the outputs for experiment records.
"""

import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPTS = {
    "plain": "You are a helpful assistant. Solve the problem. Put the final answer at the end.",
    "dapo": "Solve the problem step by step. The last line of your response should be of the form Answer: $Answer.",
    "native": "You are a reasoning assistant. Respond exactly in this format:\n<think>\nreasoning\n</think>\n<answer>\nfinal answer only\n</answer>",
    "unicode": "You are a reasoning assistant. Respond exactly in this format:\n《reasoning》\nreasoning\n《/reasoning》\n《answer》\nfinal answer only\n《/answer》",
    "chinese": "你是一个推理助手。必须严格按照以下格式回答：\n推理过程：\n这里写必要推理。\n最终答案：\n这里只写最终答案。",
    "angle": "You are a reasoning assistant. Respond exactly in this format:\n<reasoning>\nreasoning\n</reasoning>\n<final_answer>\nfinal answer only\n</final_answer>",
}

MODEL_ALIASES = {
    "inst": "Qwen3-30B-A3B-Instruct-2507",
    "instruct": "Qwen3-30B-A3B-Instruct-2507",
    "a3b": "Qwen3-30B-A3B",
    "base": "Qwen3-30B-A3B",
    "thinking": "Qwen3-30B-A3B-Thinking-2507",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_4b_inst": "Qwen3-4B-Instruct-2507",
}


def resolve_model_path(model: str | None, model_path: str | None) -> str:
    if model_path:
        return str(Path(model_path).expanduser())
    if model is None:
        raise ValueError("Either --model or --model-path is required.")
    if model in MODEL_ALIASES:
        work = os.environ.get("WORK", "/groups/gcg51557/experiments/0390_rlsd")
        return str(Path(work) / "models" / MODEL_ALIASES[model])
    return model


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file: {path}")


def parse_maybe_dict(x: Any) -> Any:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, dict):
                return v
        except Exception:
            try:
                v = json.loads(x)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return x


def extract_final_from_text(x: Any) -> str:
    """Extract a short final answer from common math dataset answer/solution fields."""
    if x is None:
        return ""
    s = str(x).strip()

    # GSM8K canonical answer column: rationale ... #### 42
    if "####" in s:
        return s.split("####")[-1].strip()

    # DAPO/OpenR1-like final line
    ms = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", s)
    if ms:
        return ms[-1].strip()

    # Boxed final answer
    ms = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if ms:
        return ms[-1].strip()

    # If it is already short, keep it. If it is long, use the last number as a fallback.
    if len(s) <= 80:
        return s
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s


def extract_ground_truth(row: dict[str, Any]) -> str:
    # verl-style reward_model.ground_truth has highest priority.
    if "reward_model" in row and row["reward_model"] is not None:
        rm = parse_maybe_dict(row["reward_model"])
        if isinstance(rm, dict) and rm.get("ground_truth") not in (None, ""):
            return extract_final_from_text(rm.get("ground_truth"))

    # Common dataset columns.
    for k in ["ground_truth", "final_answer", "target", "label", "answer", "solution"]:
        if k in row and pd.notna(row[k]):
            return extract_final_from_text(row[k])
    return ""


def extract_problem(row: dict[str, Any]) -> str:
    for k in ["problem", "question", "query", "input", "prompt"]:
        if k in row and pd.notna(row[k]):
            v = row[k]
            if isinstance(v, list):
                parts = []
                for m in v:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content", "")))
                    else:
                        parts.append(str(m))
                return "\n".join(parts)
            return str(v)
    return ""


def normalize_answer(x: Any) -> str:
    if x is None:
        return ""
    s = extract_final_from_text(x)
    s = str(s).strip()
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("％", "%")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    # Strip obvious punctuation wrappers.
    s = s.strip(".。,:：;； ")
    return s.lower()


def extract_pred(text: str, style: str) -> str | None:
    patterns: list[str] = []
    if style == "native":
        patterns.append(r"<answer>\s*(.*?)\s*</answer>")
    elif style == "unicode":
        patterns.append(r"《answer》\s*(.*?)\s*《/answer》")
    elif style == "chinese":
        patterns.append(r"最终答案：\s*([^\n]+)")
    elif style == "angle":
        patterns.append(r"<final_answer>\s*(.*?)\s*</final_answer>")

    patterns.extend([
        r"(?i)Answer\s*:\s*([^\n]+)",
        r"\\boxed\{([^{}]+)\}",
        r"(?i)final answer\s*[:：]\s*([^\n]+)",
        r"####\s*([^\n]+)",
    ])

    for pat in patterns:
        ms = re.findall(pat, text, flags=re.S)
        if ms:
            return str(ms[-1]).strip()

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def check_format(text: str, style: str) -> bool:
    if style == "native":
        return all(x in text for x in ["<think>", "</think>", "<answer>", "</answer>"])
    if style == "unicode":
        return all(x in text for x in ["《reasoning》", "《/reasoning》", "《answer》", "《/answer》"])
    if style == "chinese":
        return "推理过程：" in text and "最终答案：" in text
    if style == "angle":
        return all(x in text for x in ["<reasoning>", "</reasoning>", "<final_answer>", "</final_answer>"])
    return True


def apply_template(tok, system_prompt: str, user: str) -> str:
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return system_prompt + "\n\n" + user + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Model alias or HF/local path. Aliases: inst, a3b, thinking, ...")
    ap.add_argument("--model-path", default=None, help="Explicit local model path. Overrides --model.")
    ap.add_argument("--data", required=True, help="Local parquet/jsonl/json/csv file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-style", choices=list(SYSTEM_PROMPTS), default="unicode")
    ap.add_argument("--system-prompt", default=None, help="Optional custom system prompt string.")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_path = resolve_model_path(args.model, args.model_path)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if "/groups/" in model_path and not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    df = load_table(data_path).head(args.max_samples).copy()
    rows = df.to_dict(orient="records")
    if not rows:
        raise ValueError(f"No rows loaded from {data_path}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS[args.prompt_style]

    prompts = []
    meta = []
    missing_problem = 0
    missing_gt = 0
    for i, row in enumerate(rows):
        problem = extract_problem(row)
        gt = extract_ground_truth(row)
        if not problem:
            missing_problem += 1
        if not gt:
            missing_gt += 1
        rendered = apply_template(tok, system_prompt, problem)
        for j in range(args.n):
            prompts.append(rendered)
            meta.append((i, j, problem, gt))

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        trust_remote_code=True,
        seed=args.seed,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, params)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_prompt: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(rows))}

    with out_path.open("w", encoding="utf-8") as f:
        for (i, j, problem, gt), o in zip(meta, outputs):
            text = o.outputs[0].text
            pred = extract_pred(text, args.prompt_style)
            ok = normalize_answer(pred) == normalize_answer(gt)
            fmt = check_format(text, args.prompt_style)
            rec = {
                "index": i,
                "sample": j,
                "ground_truth": gt,
                "pred": pred,
                "correct": bool(ok),
                "format_ok": bool(fmt),
                "problem": problem,
                "text": text,
            }
            per_prompt[i].append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    success_counts = [sum(r["correct"] for r in per_prompt[i]) for i in per_prompt]
    fmt_counts = [sum(r["format_ok"] for r in per_prompt[i]) for i in per_prompt]
    K = args.n
    N = len(success_counts)

    summary = {
        "model_arg": args.model,
        "model_path": model_path,
        "data": str(data_path),
        "prompt_style": args.prompt_style,
        "N": N,
        "K": K,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "missing_problem_rows": missing_problem,
        "missing_ground_truth_rows": missing_gt,
        "pass_at_1": sum(per_prompt[i][0]["correct"] for i in per_prompt) / max(N, 1),
        "pass_at_K": sum(c > 0 for c in success_counts) / max(N, 1),
        "all_correct_ratio": sum(c == K for c in success_counts) / max(N, 1),
        "all_wrong_ratio": sum(c == 0 for c in success_counts) / max(N, 1),
        "mixed_group_ratio": sum(0 < c < K for c in success_counts) / max(N, 1),
        "format_at_1": sum(per_prompt[i][0]["format_ok"] for i in per_prompt) / max(N, 1),
        "format_any_K": sum(c > 0 for c in fmt_counts) / max(N, 1),
        "success_hist": dict(sorted(Counter(success_counts).items())),
        "format_hist": dict(sorted(Counter(fmt_counts).items())),
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    config_path = out_path.with_suffix(".config.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(vars(args) | {"resolved_model_path": model_path}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved JSONL: {out_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
