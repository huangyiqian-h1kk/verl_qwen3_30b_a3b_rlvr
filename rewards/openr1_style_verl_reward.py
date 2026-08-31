import json, os, re
from pathlib import Path

FORMAT_MODE = os.environ.get("REASONING_FORMAT_MODE", "unicode")
FORMAT_WEIGHT = float(os.environ.get("FORMAT_WEIGHT", "0.2"))
TAG_WEIGHT = float(os.environ.get("TAG_WEIGHT", "0.2"))
ACC_WEIGHT = float(os.environ.get("ACC_WEIGHT", "1.0"))
DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")

PATTERNS = {
    "native": re.compile(r"^\s*<think>\s+.*?\s+</think>\s*<answer>\s+(.*?)\s+</answer>\s*$", re.S),
    "unicode": re.compile(r"^\s*《reasoning》\s+.*?\s+《/reasoning》\s*《answer》\s+(.*?)\s+《/answer》\s*$", re.S),
    "angle": re.compile(r"^\s*<reasoning>\s+.*?\s+</reasoning>\s*<final_answer>\s+(.*?)\s+</final_answer>\s*$", re.S),
}

def normalize(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()

def strict_format_and_answer(text):
    mode = FORMAT_MODE
    if mode in PATTERNS:
        m = PATTERNS[mode].match(text)
        return (1.0 if m else 0.0), (m.group(1).strip() if m else None)
    if mode == "chinese":
        ok = "推理过程：" in text and "最终答案：" in text and text.find("推理过程：") < text.find("最终答案：")
        m = re.findall(r"最终答案：\s*([^\n]+)", text)
        return (1.0 if ok else 0.0), (m[-1].strip() if m else None)
    raise ValueError(f"Unknown REASONING_FORMAT_MODE={mode}")

def tag_count(text):
    mode = FORMAT_MODE
    if mode == "native":
        tags = ["<think>", "</think>", "<answer>", "</answer>"]
    elif mode == "unicode":
        tags = ["《reasoning》", "《/reasoning》", "《answer》", "《/answer》"]
    elif mode == "angle":
        tags = ["<reasoning>", "</reasoning>", "<final_answer>", "</final_answer>"]
    elif mode == "chinese":
        tags = ["推理过程：", "最终答案："]
        return sum(0.5 for t in tags if t in text)
    return sum(0.25 for t in tags if text.count(t) == 1)

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    fmt, pred = strict_format_and_answer(solution_str)
    tag = tag_count(solution_str)
    acc = 1.0 if pred is not None and normalize(pred) == normalize(ground_truth) else 0.0
    score = ACC_WEIGHT * acc + FORMAT_WEIGHT * fmt + TAG_WEIGHT * tag
    rec = {"score": score, "acc": acc, "format": fmt, "tag_count": tag, "pred": pred or "[NO_ANSWER]", "mode": FORMAT_MODE}
    if DEBUG_LOG:
        p = Path(DEBUG_LOG); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**rec, "data_source": data_source, "ground_truth": ground_truth, "solution_str_head": solution_str[:2000]}, ensure_ascii=False) + "\n")
    return rec
