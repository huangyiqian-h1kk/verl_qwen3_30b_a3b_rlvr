import argparse
import json
import logging
import math
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ============================================================
# Basic utilities
# ============================================================
def _chunked(seq, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _sanitize_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(s))


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_THINK_OPEN_RE = re.compile(r"<think>", flags=re.I)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.I)


def _find_think_open_close(text: str) -> Tuple[Optional[re.Match], Optional[re.Match]]:
    """
    返回第一个 <think> 和第一个 </think>。
    注意：Qwen3 / reasoning chat template 常见情况是：
      - <think> 已经在 prompt 里
      - generated_text 只包含 COT ... </think> ANSWER
    因此 generated_text 里可能没有 <think>，但有 </think>。
    """
    if not text:
        return None, None
    open_m = _THINK_OPEN_RE.search(text)
    close_m = _THINK_CLOSE_RE.search(text)
    return open_m, close_m


def _extract_final_answer(text: str) -> str:
    """
    清洗最终答案（用于 eval_one）。

    支持以下非-Harmony格式：

    A) <think> COT </think> ANSWER
       -> final = ANSWER

    B) COT </think> ANSWER
       -> final = ANSWER
       这是 Qwen3 enable_thinking=True 时非常常见的情况：
       <think> 可能在 prompt prefill 中，不在 generated_text 中。

    C) <think> COT
       -> final = ""
       说明生成被截断，尚未关闭 thinking。

    D) 无 think 标签
       -> final = 原文
       常见于 instruct / non-thinking 模型。
    """
    if text is None:
        return ""

    text = text.strip()
    if not text:
        return ""

    open_m, close_m = _find_think_open_close(text)

    # Case A / B: 只要出现 </think>，final 就是其后的内容
    if close_m is not None:
        return text[close_m.end():].strip()

    # Case C: 有 <think> 但没有 </think>，说明只有未闭合 CoT，没有 final
    if open_m is not None:
        return ""

    # Case D: 没有 think 标签，整段就是 answer
    return text


def _extract_cot(text: str) -> str:
    """
    抽取非-Harmony CoT。

    支持以下格式：

    A) <think> COT </think> ANSWER
       -> cot = COT

    B) COT </think> ANSWER
       -> cot = COT
       这是 Qwen3 chat template 把 <think> 放进 prompt 后的常见输出。

    C) <think> COT
       -> cot = COT
       生成被截断但仍可保留已有 CoT。

    D) 无 think 标签
       -> cot = ""
       不强行把全文当 CoT，避免把 final answer 混入 CoT。
    """
    if text is None:
        return ""

    text = text.strip()
    if not text:
        return ""

    open_m, close_m = _find_think_open_close(text)

    # Case A: <think> ... </think>
    if open_m is not None and close_m is not None and open_m.end() <= close_m.start():
        return text[open_m.end():close_m.start()].strip()

    # Case B: ... </think>，没有 <think>
    # 说明 <think> 很可能在 prompt prefill 里，generated_text 从 CoT 内容开始。
    if open_m is None and close_m is not None:
        return text[:close_m.start()].strip()

    # Case C: 有 <think> 但没有 </think>，把 <think> 后面都作为 CoT
    if open_m is not None and close_m is None:
        return text[open_m.end():].strip()

    # Case D: 无标签，不认为有可解析 CoT
    return ""


def _find_cot_content_span(response_text: str) -> Optional[Tuple[int, int]]:
    if not response_text:
        return None

    text = response_text
    open_m, close_m = _find_think_open_close(text)

    # A) <think> COT </think>
    if open_m is not None and close_m is not None and open_m.end() <= close_m.start():
        start = open_m.end()
        end = close_m.start()
        return (start, end) if end > start else None

    # B) COT </think> ANSWER
    if open_m is None and close_m is not None:
        start = 0
        end = close_m.start()
        return (start, end) if end > start else None

    # C) <think> COT
    if open_m is not None and close_m is None:
        start = open_m.end()
        end = len(text)
        return (start, end) if end > start else None

    # D) 无 think 标签
    return None


def apply_chat_template_robust(tokenizer, messages, add_generation_prompt: bool, chat_template_kwargs: Optional[dict]):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_case_text(rec: Dict[str, Any]) -> str:
    incident = rec["incident"].strip("经审理查明：").replace("本院", "法院")
    argue = rec["argue"].replace("本院", "法院")
    return incident + argue


def build_law_prompt(incident: str, argue: str) -> str:
    return (
        "如果遇到了如下法律案情，应该适用什么法律条文?"
        "请以'应适用《XXX》第X，X，X条;《YYY》第Y，Y，Y条和应适用《OOO》第O，O，O条'格式回复，"
        "部门法名称用全称写出，如中华人民共和国XXX法。案情如下：" + incident + argue
    )


def infer_chat_template_kwargs_for_generation(model_name: str) -> Optional[dict]:
    model_lower = model_name.lower()
    if "instruct" in model_lower or "chat" in model_lower:
        return {"enable_thinking": False}
    return {"enable_thinking": True}


# ============================================================
# Evaluation code (kept self-contained)
# ============================================================
_CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}
NUM_TOKEN = r"(?:\d+|[零〇一二两三四五六七八九十百千万]+)"
NUM_PAT = re.compile(NUM_TOKEN)
SEP = r"(?:\s*[，,、]\s*|\s*(?:和|及|以及)\s*)"
ART_LIST_BLOCK_PAT = re.compile(rf"第\s*(({NUM_TOKEN})(?:{SEP}{NUM_TOKEN})*)\s*条")
ART_RANGE_PAT = re.compile(rf"第\s*({NUM_TOKEN})\s*条\s*(?:至|到|-|—|～)\s*第?\s*({NUM_TOKEN})\s*条")
_FIRST_ART_START_PAT = re.compile(rf"第\s*{NUM_TOKEN}\s*条")


def norm_law_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = s.replace("《", "").replace("》", "")
    s = s.replace("〈", "").replace("〉", "")
    s = s.replace("<", "").replace(">", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("最高人民法关于", "最高人民法院关于")
    return s


def _cmp_key_for_folder(s: str) -> str:
    s = norm_law_name(s)
    s = re.sub(r"^中华人民共和国", "", s)
    return s


def resolve_source_full_law(source_folder: str, gold_laws: Set[str]) -> Optional[str]:
    sf = _cmp_key_for_folder(source_folder)
    if not sf:
        return None

    def score(gl: str) -> Tuple[int, int]:
        gl_norm = norm_law_name(gl)
        gl_cmp = _cmp_key_for_folder(gl_norm)
        sc = 0
        if gl_cmp == sf:
            sc += 100
        if gl_cmp.endswith(sf):
            sc += 30
        if sf in gl_cmp:
            sc += 10
        if gl_cmp in sf:
            sc += 5
        folder_is_law = sf.endswith("法")
        if folder_is_law:
            if gl_norm.endswith("法"):
                sc += 25
            if ("最高人民法院关于" in gl_norm) or ("解释" in gl_norm) or ("若干问题" in gl_norm) or ("若干规定" in gl_norm) or gl_norm.endswith("规定"):
                sc -= 30
        return (sc, -len(gl_norm))

    cands = [gl for gl in gold_laws if (sf in _cmp_key_for_folder(gl) or _cmp_key_for_folder(gl) in sf)]
    if not cands:
        return None
    return max(cands, key=score)


def chinese_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    total, section, number = 0, 0, 0
    has_any = False
    for ch in s:
        if ch in _CN_NUM:
            number = _CN_NUM[ch]
            has_any = True
        elif ch in _CN_UNIT:
            has_any = True
            unit = _CN_UNIT[ch]
            if unit == 10000:
                section = (section + (number if number != 0 else 0)) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        else:
            continue
    res = total + section + number
    return res if has_any else None


def extract_articles_from_seg(seg: str) -> Set[int]:
    arts: Set[int] = set()
    if not seg:
        return arts
    for m in ART_RANGE_PAT.finditer(seg):
        a = chinese_to_int(m.group(1))
        b = chinese_to_int(m.group(2))
        if isinstance(a, int) and isinstance(b, int) and 0 < abs(b - a) <= 300:
            lo, hi = (a, b) if a <= b else (b, a)
            arts.update(range(lo, hi + 1))
    for m in ART_LIST_BLOCK_PAT.finditer(seg):
        block = m.group(1)
        for nm in NUM_PAT.finditer(block):
            val = chinese_to_int(nm.group(0))
            if isinstance(val, int):
                arts.add(val)
    return arts


def outer_booktitle_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    if not isinstance(text, str) or not text:
        return spans
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "《":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "》" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i + 1, text[start + 1:i]))
                start = None
    return spans


def extract_law2arts_from_text(text: str, extra_law_aliases: Optional[List[str]] = None) -> Dict[str, Set[int]]:
    if not isinstance(text, str):
        return {}
    t = text
    law2arts: Dict[str, Set[int]] = {}
    spans = outer_booktitle_spans(t)
    for idx, (s, e, inside) in enumerate(spans):
        law_name = norm_law_name(inside)
        if not law_name:
            continue
        seg_start = e
        seg_end = spans[idx + 1][0] if idx + 1 < len(spans) else len(t)
        seg = t[seg_start:seg_end]
        arts = extract_articles_from_seg(seg)
        if arts:
            law2arts.setdefault(law_name, set()).update(arts)
    if extra_law_aliases:
        for alias in extra_law_aliases:
            alias_norm = norm_law_name(alias)
            if not alias_norm:
                continue
            for mm in re.finditer(re.escape(alias), t):
                window = t[mm.end(): mm.end() + 200]
                arts = extract_articles_from_seg(window)
                if arts:
                    law2arts.setdefault(alias_norm, set()).update(arts)
    return law2arts


def extract_gold_law2arts(law_apply_items: List[str]) -> Dict[str, Set[int]]:
    gold: Dict[str, Set[int]] = {}
    if not law_apply_items:
        return gold
    for item in law_apply_items:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        m = _FIRST_ART_START_PAT.search(s)
        if not m:
            continue
        law_part = s[:m.start()].strip()
        seg = s[m.start():]
        law_part = re.sub(r"^(依据|依照|根据|参照|按照|依)\s*", "", law_part).strip()
        law_part = law_part.strip("，,;；:：。.\n\t ")
        law_name = norm_law_name(law_part)
        if not law_name:
            continue
        arts = extract_articles_from_seg(seg)
        if arts:
            gold.setdefault(law_name, set()).update(arts)
    return gold


def eval_one(gold_rec: Dict[str, Any], llm_text: str) -> Dict[str, Any]:
    gold_items = (gold_rec.get("law_apply_dict") or {}).get("law_apply_items") or []
    gold_law2arts = extract_gold_law2arts(gold_items)
    gold_laws = set(gold_law2arts.keys())
    pred_law2arts = extract_law2arts_from_text(llm_text, extra_law_aliases=None)
    pred_laws = set(pred_law2arts.keys())

    source_folder = gold_rec.get("source_folder", "") or ""
    source_full = norm_law_name(gold_rec.get("source_folder", "") or "")

    if source_full not in gold_laws:
        source_full = None

    acc_source_law = 1 if (source_full and source_full in pred_laws) else 0
    gold_src_arts = gold_law2arts.get(source_full, set()) if source_full else set()
    pred_src_arts = pred_law2arts.get(source_full, set()) if source_full else set()
    acc_source_article = (len(gold_src_arts & pred_src_arts) / len(gold_src_arts)) if gold_src_arts else 0.0

    acc_all_law = (len(gold_laws & pred_laws) / len(gold_laws)) if gold_laws else 0.0

    total_gold_arts = 0
    matched_arts = 0
    for gl, garts in gold_law2arts.items():
        total_gold_arts += len(garts)
        matched_arts += len(garts & pred_law2arts.get(gl, set()))
    acc_all_article = (matched_arts / total_gold_arts) if total_gold_arts else 0.0
    
    return {
        "acc_source_law": acc_source_law,
        "acc_source_article": acc_source_article,
        "acc_all_law": acc_all_law,
        "acc_all_article": acc_all_article,
        "source_full_law": source_full,
        "gold_laws": sorted(gold_laws),
        "pred_laws": sorted(pred_laws),
        "gold_source_articles": sorted(gold_src_arts),
        "pred_source_articles": sorted(pred_src_arts),
        "total_gold_articles": total_gold_arts,
        "matched_articles": matched_arts,
    }


# ============================================================
# Entropy utilities (SVS / CalibRL style token-level local entropy)
# ============================================================
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)


def masked_mean(vals: List[float], mask: List[int]) -> Optional[float]:
    denom = sum(mask)
    if denom == 0:
        return None
    return sum(v for v, m in zip(vals, mask) if m) / denom


def masked_sum(vals: List[float], mask: List[int]) -> float:
    return sum(v for v, m in zip(vals, mask) if m)


def _decode_token_spans(tokenizer, token_ids: List[int]) -> Tuple[str, List[Tuple[int, int]], List[str], List[str]]:
    """
    用增量 decode 从“真实生成 token ids”恢复每个 token 在 response 文本里的字符跨度。
    这样可以避免 prompt/response 拼接时的边界重分词问题。
    """
    prefix_ids: List[int] = []
    prev_text = ""
    spans: List[Tuple[int, int]] = []
    pieces: List[str] = []
    token_strs: List[str] = tokenizer.convert_ids_to_tokens(token_ids)
    for tid in token_ids:
        prefix_ids.append(tid)
        new_text = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        piece = new_text[len(prev_text):]
        start = len(prev_text)
        end = len(new_text)
        spans.append((start, end))
        pieces.append(piece)
        prev_text = new_text
    return prev_text, spans, pieces, token_strs

def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _build_region_masks_from_spans(
    response_text: str,
    token_spans: List[Tuple[int, int]],
) -> Tuple[List[int], List[int], List[str]]:
    cot_span = _find_cot_content_span(response_text)
    full_mask = [1] * len(token_spans)  # 完整轨迹 = 整个 response 全部 token
    cot_mask: List[int] = []
    regions: List[str] = []
    for span in token_spans:
        in_cot = 1 if (cot_span is not None and _overlap(span, cot_span) > 0) else 0
        cot_mask.append(in_cot)
        regions.append("cot" if in_cot else "non_cot")
    return cot_mask, full_mask, regions


class HFEntropyScorer:
    def __init__(self, model_name: str, trust_remote_code: bool = False, dtype: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
        model_kwargs = {"trust_remote_code": trust_remote_code, "device_map": "auto"}
        if dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        elif dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def score_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = []
        for row in rows:
            prompt_ids = row["prompt_token_ids"]
            response_ids = row["generated_token_ids"]
            full_ids = prompt_ids + response_ids
            prepared.append({
                "case_index": row["index"],
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "full_ids": full_ids,
                "prompt_len": len(prompt_ids),
                "response_len": len(response_ids),
                "source_folder": row.get("source_folder"),
                "raw_text": row.get("raw_text", ""),
            })

        max_len = max(len(x["full_ids"]) for x in prepared)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = []
        attention_mask = []
        for item in prepared:
            ids = item["full_ids"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()  # [B, T, V]

        results: List[Dict[str, Any]] = []
        for bi, item in enumerate(prepared):
            prompt_len = item["prompt_len"]
            response_len = item["response_len"]
            if response_len == 0:
                results.append({
                    "case_index": item["case_index"],
                    "cot_entropy_mean": None,
                    "full_entropy_mean": None,
                    "cot_entropy_sum": 0.0,
                    "full_entropy_sum": 0.0,
                    "cot_token_count": 0,
                    "full_token_count": 0,
                    "token_records": [],
                    "decoded_response_text": "",
                })
                continue

            # 第一个 response token 由 logits[prompt_len - 1] 预测
            response_logits = logits[bi, prompt_len - 1: prompt_len - 1 + response_len, :]
            token_entropy = entropy_from_logits(response_logits.unsqueeze(0))[0].detach().cpu().tolist()

            decoded_response_text, token_spans, pieces, token_strs = _decode_token_spans(self.tokenizer, item["response_ids"])
            cot_mask, full_mask, regions = _build_region_masks_from_spans(decoded_response_text, token_spans)

            token_records = []
            for idx, (tid, tok, piece, ent, span, reg) in enumerate(
                zip(item["response_ids"], token_strs, pieces, token_entropy, token_spans, regions)
            ):
                token_records.append({
                    "case_index": item["case_index"],
                    "source_folder": item["source_folder"],
                    "token_index": idx,
                    "token_id": int(tid),
                    "token": tok,
                    "text_piece": piece,
                    "entropy": float(ent),
                    "char_start": int(span[0]),
                    "char_end": int(span[1]),
                    "region": reg,
                })

            results.append({
                "case_index": item["case_index"],
                "source_folder": item["source_folder"],
                "decoded_response_text": decoded_response_text,
                "cot_entropy_mean": masked_mean(token_entropy, cot_mask),
                "full_entropy_mean": masked_mean(token_entropy, full_mask),
                "cot_entropy_sum": masked_sum(token_entropy, cot_mask),
                "full_entropy_sum": masked_sum(token_entropy, full_mask),
                "cot_token_count": int(sum(cot_mask)),
                "full_token_count": int(sum(full_mask)),
                "token_records": token_records,
            })
        return results


# ============================================================
# Two-stage pipeline
# ============================================================
def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def make_run_prefix(args) -> str:
    flag = "remain" if "remain" in getattr(args, "data", "NA").lower() else "applied"
    parts = [
        f"model_{_sanitize_filename(args.modelName)}",
        f"sampleTime_{_sanitize_filename(args.sampleTime)}",
        f"hint_level_{_sanitize_filename(args.hint_level)}",
        f"split_{flag}",
        f"seed_{_sanitize_filename(args.seed)}",
    ]
    return "__".join(parts)


def load_filtered_data(data_path: str, sample_time: int, seed: int) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    data = [x for x in data if (len(x["incident"]) > 250 and len(x["argue"]) > 50)]
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[: min(sample_time, len(data))]


def generate_stage(args, run_dir: Path, logger: logging.Logger):
    data = load_filtered_data(args.data, args.sampleTime, args.seed)
    logger.info("[generate] n_samples=%d", len(data))

    llm_kwargs = {}

    if "qwen3-next" in args.modelName.lower():
        llm_kwargs["gdn_prefill_backend"] = "triton"

    llm = LLM(
        model=args.modelName,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=True,
        **llm_kwargs,
    )

    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.modelName, trust_remote_code=args.trust_remote_code)

    chat_template_kwargs_gen = infer_chat_template_kwargs_for_generation(args.modelName)
    logger.info("[generate] chat_template_kwargs_gen=%s", str(chat_template_kwargs_gen))

    system_dict = {
        "none": "You are a helpful assistant.",
        "weak": "You are a helpful assistant. 用户可能会向你提问一些法律咨询问题，请试着实事求是地回答。注意案件中各个时间发生的时间点和法律的时间效力问题，注意法律的溯及力问题。",
        "strong": "You are a helpful assistant. 用户可能会向你提问一些法律咨询问题，请试着实事求是地回答。注意案件中各个时间发生的时间点和法律的时间效力问题，注意法律的溯及力问题。民事法律时间效力上的基本原则是法不溯及既往，所以应该特别注意案件事实对应的时间点是否应该适用旧法；但是存在一些例外情况，比如有利溯及（有利于保护民事主体合法权益等），新增溯及（即旧法没有规定新法新增规定的），还有新增具体规定可用作裁判说理的情况等。",
    }
    system_prompt = system_dict[args.hint_level]

    requests: List[Dict[str, Any]] = []
    prompt_texts: List[str] = []
    for idx, rec in enumerate(data):
        incident = rec["incident"].strip("经审理查明：").replace("本院", "法院")
        argue = rec["argue"].replace("本院", "法院")
        user_prompt = build_law_prompt(incident, argue)
        messages = build_messages(system_prompt, user_prompt)
        prompt_text = apply_chat_template_robust(
            tokenizer,
            messages,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs_gen,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        requests.append({
            "index": idx,
            "gold_record": rec,
            "messages": messages,
            "prompt_text": prompt_text,
            "prompt_token_ids": prompt_ids,
            "source_folder": rec.get("source_folder"),
        })
        prompt_texts.append(prompt_text)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    generated_records = []
    eval_records = []
    case_tests_jsonl = run_dir / "case_tests.jsonl"
    case_evals_jsonl = run_dir / "case_evals.jsonl"
    case_tests_pkl = run_dir / "case_tests.pkl"
    case_evals_pkl = run_dir / "case_evals.pkl"

    with case_tests_jsonl.open("w", encoding="utf-8") as f_tests, case_evals_jsonl.open("w", encoding="utf-8") as f_evals:
        for batch_reqs, batch_prompts in zip(_chunked(requests, args.batch_size), _chunked(prompt_texts, args.batch_size)):
            outs = llm.generate(batch_prompts, sampling_params=sampling, use_tqdm=False)
            for req, out in zip(batch_reqs, outs):
                first = out.outputs[0]
                raw_text = first.text
                clean_answer = _extract_final_answer(raw_text)
                cot_text = _extract_cot(raw_text)
                gen_token_ids = list(getattr(first, "token_ids", []) or [])

                if _THINK_CLOSE in raw_text and not cot_text:
                    logger.warning(
                        "[generate][%s] detected </think> but cot_text is empty. raw_text_prefix=%s",
                        req["index"],
                        raw_text[:300].replace("\n", "\\n"),
                    )

                if _THINK_OPEN in raw_text and _THINK_CLOSE not in raw_text:
                    logger.warning(
                        "[generate][%s] detected <think> without </think>; final answer may be empty. raw_text_prefix=%s",
                        req["index"],
                        raw_text[:300].replace("\n", "\\n"),
                        )
                
                row = {
                    "index": req["index"],
                    "source_folder": req.get("source_folder"),
                    "messages": req["messages"],
                    "prompt_text": req["prompt_text"],
                    "prompt_token_ids": req["prompt_token_ids"],
                    "generated_token_ids": gen_token_ids,
                    "raw_text": raw_text,
                    "clean_answer": clean_answer,
                    "cot_text": cot_text,
                    "finish_reason": getattr(first, "finish_reason", None),
                    "stop_reason": getattr(first, "stop_reason", None),
                    "chat_template_kwargs_gen": chat_template_kwargs_gen,
                    "model_name": args.modelName,
                }
                generated_records.append(row)
                f_tests.write(json.dumps(row, ensure_ascii=False) + "\n")

                ev = eval_one(req["gold_record"], clean_answer)
                ev["index"] = req["index"]
                ev["source_folder"] = req.get("source_folder")
                eval_records.append(ev)
                f_evals.write(json.dumps(ev, ensure_ascii=False) + "\n")

                logger.info("[generate][%s] source=%s", req["index"], req.get("source_folder"))
                logger.info("[generate][%s] answer=%s", req["index"], clean_answer)
                logger.info("[generate][%s] eval=%s", req["index"], ev)

    with case_tests_pkl.open("wb") as f:
        pickle.dump(generated_records, f)
    with case_evals_pkl.open("wb") as f:
        pickle.dump(eval_records, f)

    if eval_records:
        metric_keys = ["acc_source_law", "acc_source_article", "acc_all_law", "acc_all_article"]
        avg = {k: sum(x[k] for x in eval_records) / len(eval_records) for k in metric_keys}
        logger.info("[generate] average_eval=%s", avg)


def score_stage(args, run_dir: Path, logger: logging.Logger):
    case_tests_jsonl = run_dir / "case_tests.jsonl"
    if not case_tests_jsonl.exists():
        raise FileNotFoundError(f"Missing generated file: {case_tests_jsonl}")

    rows = []
    with case_tests_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    logger.info("[score] loaded_generated_rows=%d", len(rows))

    scorer = HFEntropyScorer(
        model_name=args.modelName,
        trust_remote_code=args.trust_remote_code,
        dtype=args.scorer_dtype,
    )

    traj_path = run_dir / "trajectory_entropy.jsonl"
    tok_path = run_dir / "token_entropy.jsonl"
    traj_summary_pkl = run_dir / "trajectory_entropy.pkl"

    all_traj = []
    with traj_path.open("w", encoding="utf-8") as f_traj, tok_path.open("w", encoding="utf-8") as f_tok:
        for batch in _chunked(rows, args.score_batch_size):
            batch_results = scorer.score_batch(batch)
            for base_row, score_row in zip(batch, batch_results):
                traj_row = {
                    "case_index": base_row["index"],
                    "source_folder": base_row.get("source_folder"),
                    "raw_text": base_row.get("raw_text", ""),
                    "clean_answer": base_row.get("clean_answer", ""),
                    "cot_text": base_row.get("cot_text", ""),
                    "decoded_response_text": score_row["decoded_response_text"],
                    "cot_entropy_mean": score_row["cot_entropy_mean"],
                    "full_entropy_mean": score_row["full_entropy_mean"],
                    "cot_entropy_sum": score_row["cot_entropy_sum"],
                    "full_entropy_sum": score_row["full_entropy_sum"],
                    "cot_token_count": score_row["cot_token_count"],
                    "full_token_count": score_row["full_token_count"],
                }
                all_traj.append(traj_row)
                f_traj.write(json.dumps(traj_row, ensure_ascii=False) + "\n")
                for rec in score_row["token_records"]:
                    f_tok.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.info(
                    "[score][%s] cot_entropy_mean=%s full_entropy_mean=%s cot_tokens=%s full_tokens=%s",
                    base_row["index"],
                    traj_row["cot_entropy_mean"],
                    traj_row["full_entropy_mean"],
                    traj_row["cot_token_count"],
                    traj_row["full_token_count"],
                )

    with traj_summary_pkl.open("wb") as f:
        pickle.dump(all_traj, f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="all", choices=["generate", "score", "all"])
    p.add_argument("--modelName", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--sampleTime", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hint_level", type=str, default="none", choices=["none", "weak", "strong"])
    p.add_argument("--out_dir", type=str, default=".")

    # generation
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=32000)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dtype", type=str, default="auto")

    # scorer
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument("--scorer_dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / make_run_prefix(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir / "run.log")
    logger.info("run_dir=%s", str(run_dir))
    logger.info("stage=%s", args.stage)

    if args.stage in {"generate", "all"}:
        generate_stage(args, run_dir, logger)
    if args.stage in {"score", "all"}:
        score_stage(args, run_dir, logger)

    logger.info("DONE")


if __name__ == "__main__":
    main()
