#!/usr/bin/env python3
import argparse, json, os, re, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPTS = {
    "native_repeat": """请严格复读下面这段内容，不要解释、不要改写、不要省略任何字符：\n<think>\ntest\n</think>\n<answer>\n4\n</answer>""",
    "native_reasoning": """Solve the problem. You must strictly respond in this exact format:\n<think>\nWrite your reasoning here.\n</think>\n<answer>\nWrite only the final answer here.\n</answer>\nProblem: What is 2+2?""",
    "unicode_reasoning": """Solve the problem. You must strictly respond in this exact format:\n《reasoning》\nWrite your reasoning here.\n《/reasoning》\n《answer》\nWrite only the final answer here.\n《/answer》\nProblem: What is 2+2?""",
    "chinese_reasoning": """请解答问题，并严格按照以下格式回答：\n推理过程：\n这里写必要推理。\n最终答案：\n这里只写最终答案。\n问题：2+2 等于多少？""",
    "angle_reasoning": """Solve the problem. You must strictly respond in this exact format:\n<reasoning>\nWrite your reasoning here.\n</reasoning>\n<final_answer>\nWrite only the final answer here.\n</final_answer>\nProblem: What is 2+2?""",
    "native_one_line": """请直接输出下面这两个字符串，中间用空格隔开，不要解释：<think> </think>""",
}

STRICT_PATTERNS = {
    "native": re.compile(r"^\s*<think>\s+.*?\s+</think>\s*<answer>\s+.*?\s+</answer>\s*$", re.S),
    "unicode": re.compile(r"^\s*《reasoning》\s+.*?\s+《/reasoning》\s*《answer》\s+.*?\s+《/answer》\s*$", re.S),
    "chinese": re.compile(r"推理过程：.*最终答案：", re.S),
    "angle": re.compile(r"^\s*<reasoning>\s+.*?\s+</reasoning>\s*<final_answer>\s+.*?\s+</final_answer>\s*$", re.S),
}

def apply_template(tok, system_prompt, user_prompt, enable_thinking=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        try:
            return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tok.apply_chat_template(messages, **kwargs)

def analyze(text):
    return {
        "contains_think": "<think>" in text,
        "contains_end_think": "</think>" in text,
        "contains_answer": "<answer>" in text,
        "contains_end_answer": "</answer>" in text,
        "contains_tool_call": "<tool_call>" in text,
        "contains_unicode_reasoning": "《reasoning》" in text,
        "contains_unicode_answer": "《answer》" in text,
        "contains_chinese_reasoning": "推理过程：" in text,
        "contains_chinese_final": "最终答案：" in text,
        "contains_angle_reasoning": "<reasoning>" in text,
        "contains_final_answer": "<final_answer>" in text,
        "strict_native": bool(STRICT_PATTERNS["native"].match(text)),
        "strict_unicode": bool(STRICT_PATTERNS["unicode"].match(text)),
        "strict_chinese": bool(STRICT_PATTERNS["chinese"].search(text)),
        "strict_angle": bool(STRICT_PATTERNS["angle"].match(text)),
        "num_chars": len(text),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--enable-thinking", choices=["true", "false", "none"], default="none")
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    enable = None if args.enable_thinking == "none" else (args.enable_thinking == "true")

    rendered = []
    for variant, user_prompt in PROMPTS.items():
        prompt_text = apply_template(tok, "", user_prompt, enable_thinking=enable)
        for i in range(args.n):
            rendered.append((variant, i, prompt_text))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype, trust_remote_code=True)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, n=1)
    outputs = llm.generate([x[2] for x in rendered], params)

    counts = {}
    with out.open("w", encoding="utf-8") as f:
        for meta, output in zip(rendered, outputs):
            variant, idx, prompt_text = meta
            text = output.outputs[0].text
            rec = {
                "model": args.model,
                "variant": variant,
                "sample_id": idx,
                "prompt_tail": prompt_text[-1000:],
                "text": text,
                "analysis": analyze(text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts.setdefault(variant, {k: 0 for k in analyze("").keys()})
            for k, v in rec["analysis"].items():
                if isinstance(v, bool) and v:
                    counts[variant][k] += 1

    summary = {"model": args.model, "n_per_variant": args.n, "counts": counts, "out": str(out)}
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved JSONL: {out}")
    print(f"Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
