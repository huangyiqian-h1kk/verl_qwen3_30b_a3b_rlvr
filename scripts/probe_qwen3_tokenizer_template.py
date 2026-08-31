#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from transformers import AutoTokenizer

DEFAULT_STRINGS = [
    "<think>", "</think>", "<answer>", "</answer>",
    "<reasoning>", "</reasoning>", "<final_answer>", "</final_answer>",
    "《think》", "《/think》", "《reasoning》", "《/reasoning》", "《answer》", "《/answer》",
    "推理过程：", "最终答案：", "<tool_call>", "</tool_call>", "<|im_end|>",
]

PROMPT_VARIANTS = {
    "native_think": """You must answer exactly in this format:\n<think>\nreasoning\n</think>\n<answer>\nanswer\n</answer>""",
    "unicode": """You must answer exactly in this format:\n《reasoning》\nreasoning\n《/reasoning》\n《answer》\nanswer\n《/answer》""",
    "chinese": """请严格按照以下格式回答：\n推理过程：\n这里写推理过程。\n最终答案：\n这里只写最终答案。""",
    "angle_reasoning": """You must answer exactly in this format:\n<reasoning>\nreasoning\n</reasoning>\n<final_answer>\nanswer\n</final_answer>""",
}

def apply_template(tok, messages, enable_thinking=None):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        try:
            return tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError as e:
            return f"[TypeError enable_thinking={enable_thinking}: {e}]\n" + tok.apply_chat_template(messages, **kwargs)
    return tok.apply_chat_template(messages, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        safe = model.rstrip('/').split('/')[-1]
        print(f"\n===== Loading tokenizer: {model} =====")
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=args.local_files_only)
        report = {
            "model": model,
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
            "special_tokens_map": tok.special_tokens_map,
            "added_vocab_filtered": {k: v for k, v in tok.get_added_vocab().items()
                                     if any(x in k.lower() for x in ["think", "answer", "tool", "im_"])},
            "strings": [],
            "chat_templates": {},
        }
        for s in DEFAULT_STRINGS:
            ids = tok.encode(s, add_special_tokens=False)
            item = {
                "text": s,
                "ids": ids,
                "tokens": tok.convert_ids_to_tokens(ids),
                "decode_skip_true": tok.decode(ids, skip_special_tokens=True),
                "decode_skip_false": tok.decode(ids, skip_special_tokens=False),
            }
            report["strings"].append(item)
            print(json.dumps(item, ensure_ascii=False))

        for name, sys_prompt in PROMPT_VARIANTS.items():
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is 2+2?"},
            ]
            rendered = {"default": apply_template(tok, messages)}
            rendered["enable_thinking_true"] = apply_template(tok, messages, enable_thinking=True)
            rendered["enable_thinking_false"] = apply_template(tok, messages, enable_thinking=False)
            report["chat_templates"][name] = rendered
            print(f"\n--- TEMPLATE {safe} / {name} / default tail ---")
            print(rendered["default"][-1200:])

        path = out_dir / f"tokenizer_template_{safe}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
