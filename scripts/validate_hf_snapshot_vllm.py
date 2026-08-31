#!/usr/bin/env python3
"""Load/generation check for a harvested HF snapshot.

This is not an entropy-probing script. It only verifies that the snapshot is
complete and locally loadable. Format compliance is reported as a diagnostic;
it becomes a hard gate only when ``--require-format`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


FORMAT = re.compile(
    r"^\s*《reasoning》\s*\S(?:.*?\S)?\s*《/reasoning》\s*"
    r"《answer》\s*\S(?:.*?\S)?\s*《/answer》\s*$",
    re.S,
)


def verify_shards(snapshot: Path) -> None:
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"[FAIL] missing config.json: {snapshot}")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (snapshot / name).is_file()]
        if not shards or missing:
            raise SystemExit(f"[FAIL] invalid/missing safetensor shards: {missing[:5]}")
    elif not list(snapshot.glob("*.safetensors")):
        raise SystemExit(f"[FAIL] no safetensor weights: {snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--require-format", action="store_true")
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    if not snapshot.is_dir():
        raise SystemExit(f"[FAIL] snapshot directory not found: {snapshot}")
    if not (snapshot / ".complete").is_file():
        raise SystemExit(f"[FAIL] .complete marker missing: {snapshot}")
    if not (snapshot / "tokenizer_config.json").is_file():
        raise SystemExit(f"[FAIL] tokenizer_config.json missing: {snapshot}")
    verify_shards(snapshot)

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").rstrip("\n")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    questions = [
        "What is 17 multiplied by 23?",
        "A rectangle has perimeter 36 and one side of length 4. What is its area?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in questions
    ]
    if any("<think>" in prompt or "</think>" in prompt for prompt in prompts):
        raise SystemExit("[FAIL] tokenizer inserted built-in thinking tags")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    llm = LLM(
        model=str(snapshot),
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enable_expert_parallel=False,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
        seed=42,
    )
    outputs = llm.generate(prompts, sampling)

    failures = 0
    for question, request_output in zip(questions, outputs):
        text = request_output.outputs[0].text.strip()
        passed = FORMAT.match(text) is not None
        failures += int(not passed)
        print("=" * 72)
        print(f"Q: {question}")
        print(f"format_ok={passed}")
        print(text)
    if failures and args.require_format:
        raise SystemExit(f"[FAIL] {failures}/{len(questions)} generations violated unicode format")
    print(
        f"[PASS] snapshot is complete and loadable; "
        f"unicode_format_rate={(len(questions) - failures) / len(questions):.3f}"
    )


if __name__ == "__main__":
    main()
