import json, os
from pathlib import Path
LOG = Path(os.environ.get("DEBUG_REWARD_LOG", "./debug_reward_samples.jsonl"))

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    rec = {
        "data_source": data_source,
        "solution_str_head": solution_str[:4000],
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "contains_think": "<think>" in solution_str,
        "contains_end_think": "</think>" in solution_str,
        "contains_answer": "<answer>" in solution_str,
        "contains_end_answer": "</answer>" in solution_str,
        "contains_unicode_reasoning": "《reasoning》" in solution_str,
        "contains_unicode_answer": "《answer》" in solution_str,
        "contains_chinese_reasoning": "推理过程：" in solution_str,
        "contains_chinese_final": "最终答案：" in solution_str,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"score": 0.0, "acc": 0.0, "format": 0.0}
