#!/usr/bin/env python3
import argparse, json, math, os, re
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPTS = {
    "plain": "You are a helpful assistant. Solve the problem. Put the final answer at the end.",
    "dapo": "Solve the problem step by step. The last line of your response should be of the form Answer: $Answer.",
    "native": "You are a reasoning assistant. Respond exactly in this format:\n<think>\nreasoning\n</think>\n<answer>\nfinal answer only\n</answer>",
    "unicode": "You are a reasoning assistant. Respond exactly in this format:\n《reasoning》\nreasoning\n《/reasoning》\n《answer》\nfinal answer only\n《/answer》",
    "chinese": "你是一个推理助手。必须严格按照以下格式回答：\n推理过程：\n这里写必要推理。\n最终答案：\n这里只写最终答案。",
}

def normalize(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()

def extract_ground_truth(row):
    for k in ["answer", "solution", "target", "ground_truth", "final_answer"]:
        if k in row and pd.notna(row[k]):
            return str(row[k])
    if "reward_model" in row and isinstance(row["reward_model"], dict):
        return str(row["reward_model"].get("ground_truth", ""))
    if "reward_model" in row and isinstance(row["reward_model"], str):
        try:
            import ast
            return str(ast.literal_eval(row["reward_model"]).get("ground_truth", ""))
        except Exception:
            pass
    return ""

def extract_problem(row):
    for k in ["problem", "question", "query", "input", "prompt"]:
        if k in row and pd.notna(row[k]):
            v = row[k]
            if isinstance(v, list):
                # verl chat prompt
                return "\n".join([m.get("content", str(m)) for m in v if isinstance(m, dict)])
            return str(v)
    return ""

def extract_pred(text, style):
    patterns = []
    if style == "native":
        patterns.append(r"<answer>\s*(.*?)\s*</answer>")
    if style == "unicode":
        patterns.append(r"《answer》\s*(.*?)\s*《/answer》")
    if style == "chinese":
        patterns.append(r"最终答案：\s*([^\n]+)")
    patterns += [
        r"(?i)Answer\s*:\s*([^\n]+)",
        r"\\boxed\{([^{}]+)\}",
        r"(?i)final answer\s*[:：]\s*([^\n]+)",
    ]
    for pat in patterns:
        ms = re.findall(pat, text, flags=re.S)
        if ms:
            return str(ms[-1]).strip()
    # last number fallback
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

def check_format(text, style):
    if style == "native": return all(x in text for x in ["<think>", "</think>", "<answer>", "</answer>"])
    if style == "unicode": return all(x in text for x in ["《reasoning》", "《/reasoning》", "《answer》", "《/answer》"])
    if style == "chinese": return "推理过程：" in text and "最终答案：" in text
    return True

def apply_template(tok, sys, user):
    msgs = [{"role":"system","content":sys}, {"role":"user","content":user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return sys + "\n\n" + user + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="Local parquet/jsonl/csv file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-style", choices=list(SYSTEM_PROMPTS), default="unicode")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    data_path = Path(args.data)
    if data_path.suffix == ".parquet": df = pd.read_parquet(data_path)
    elif data_path.suffix == ".jsonl": df = pd.read_json(data_path, lines=True)
    elif data_path.suffix == ".csv": df = pd.read_csv(data_path)
    else: raise ValueError(f"Unsupported data file: {data_path}")
    df = df.head(args.max_samples).copy()
    rows = df.to_dict(orient="records")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    meta = []
    for i, row in enumerate(rows):
        problem = extract_problem(row)
        gt = extract_ground_truth(row)
        rendered = apply_template(tok, SYSTEM_PROMPTS[args.prompt_style], problem)
        for j in range(args.n):
            prompts.append(rendered)
            meta.append((i, j, problem, gt))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype, trust_remote_code=True)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=1)
    outputs = llm.generate(prompts, params)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    per_prompt = {i: [] for i in range(len(rows))}
    with out_path.open("w", encoding="utf-8") as f:
        for (i, j, problem, gt), o in zip(meta, outputs):
            text = o.outputs[0].text
            pred = extract_pred(text, args.prompt_style)
            ok = normalize(pred) == normalize(gt)
            fmt = check_format(text, args.prompt_style)
            rec = {"index": i, "sample": j, "problem": problem, "ground_truth": gt, "pred": pred, "correct": ok, "format_ok": fmt, "text": text}
            per_prompt[i].append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    success_counts = [sum(r["correct"] for r in per_prompt[i]) for i in per_prompt]
    fmt_counts = [sum(r["format_ok"] for r in per_prompt[i]) for i in per_prompt]
    K = args.n; N = len(success_counts)
    summary = {
        "model": args.model,
        "data": args.data,
        "prompt_style": args.prompt_style,
        "N": N,
        "K": K,
        "pass_at_1": sum(per_prompt[i][0]["correct"] for i in per_prompt) / max(N,1),
        "pass_at_K": sum(c > 0 for c in success_counts) / max(N,1),
        "all_correct_ratio": sum(c == K for c in success_counts) / max(N,1),
        "all_wrong_ratio": sum(c == 0 for c in success_counts) / max(N,1),
        "mixed_group_ratio": sum(0 < c < K for c in success_counts) / max(N,1),
        "format_at_1": sum(per_prompt[i][0]["format_ok"] for i in per_prompt) / max(N,1),
        "format_any_K": sum(c > 0 for c in fmt_counts) / max(N,1),
        "success_hist": {str(k): success_counts.count(k) for k in range(K+1)},
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
