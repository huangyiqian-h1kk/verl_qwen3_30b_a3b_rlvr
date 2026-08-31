# keyword_phrase_entropy.py
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_keywords(path_or_text):
    """
    支持两种形式：
    1. 关键词文件，每行一个关键词
    2. 直接传逗号分隔字符串，例如：北京大学,清华大学,合同法
    """
    p = Path(path_or_text)
    if p.exists():
        keywords = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                kw = line.strip()
                if kw:
                    keywords.append(kw)
        return keywords

    return [x.strip() for x in path_or_text.split(",") if x.strip()]


def group_by_case(rows):
    groups = defaultdict(list)
    for r in rows:
        if "case_index" not in r:
            continue
        groups[int(r["case_index"])].append(r)

    for case_index in groups:
        groups[case_index] = sorted(
            groups[case_index],
            key=lambda x: int(x.get("token_index", 0))
        )

    return groups


def build_answer_and_offsets(tokens):
    """
    返回：
    - answer_text: 拼接后的完整答案
    - token_infos: 每个 token 的 text_piece / entropy / char_start / char_end 等

    优先使用原始 char_start / char_end。
    如果没有，就根据 text_piece 累计字符 offset。
    """
    pieces = []
    token_infos = []

    offset = 0
    has_offsets = all(
        ("char_start" in t and "char_end" in t)
        for t in tokens
    )

    for t in tokens:
        piece = t.get("text_piece", "")
        pieces.append(piece)

        if has_offsets:
            char_start = int(t["char_start"])
            char_end = int(t["char_end"])
        else:
            char_start = offset
            char_end = offset + len(piece)

        entropy = t.get("entropy", None)

        token_infos.append(
            {
                "case_index": int(t.get("case_index")),
                "token_index": int(t.get("token_index", -1)),
                "token_id": t.get("token_id"),
                "token": t.get("token"),
                "text_piece": piece,
                "entropy": None if entropy is None else float(entropy),
                "char_start": char_start,
                "char_end": char_end,
                "region": t.get("region"),
            }
        )

        offset += len(piece)

    answer_text = "".join(pieces)
    return answer_text, token_infos


def find_keyword_occurrences(answer_text, keyword, allow_overlap=False):
    """
    查找关键词出现位置。
    默认不允许同一个关键词的重叠匹配。
    """
    if not keyword:
        return []

    if allow_overlap:
        pattern = re.compile(f"(?={re.escape(keyword)})")
        return [
            (m.start(), m.start() + len(keyword))
            for m in pattern.finditer(answer_text)
        ]

    return [
        (m.start(), m.end())
        for m in re.finditer(re.escape(keyword), answer_text)
    ]


def tokens_overlapping_span(token_infos, span_start, span_end):
    """
    找出和关键词字符区间有重叠的 token。

    token 区间: [char_start, char_end)
    phrase 区间: [span_start, span_end)
    重叠条件:
      token_end > span_start and token_start < span_end
    """
    matched = []
    for t in token_infos:
        ts = t["char_start"]
        te = t["char_end"]

        if te > span_start and ts < span_end:
            if t["entropy"] is not None:
                matched.append(t)

    return matched


def compute_occurrence_metrics(case_index, keyword, occ_id, span_start, span_end, answer_text, matched_tokens):
    entropies = [t["entropy"] for t in matched_tokens]

    if not entropies:
        return None

    phrase_entropy = sum(entropies) / len(entropies)

    peak_token = max(matched_tokens, key=lambda x: x["entropy"])
    phrase_peak_entropy = peak_token["entropy"]

    return {
        "case_index": case_index,
        "keyword": keyword,
        "occurrence_id": occ_id,
        "char_start": span_start,
        "char_end": span_end,
        "matched_text": answer_text[span_start:span_end],

        "n_tokens_in_phrase": len(matched_tokens),
        "token_indices": " ".join(str(t["token_index"]) for t in matched_tokens),
        "token_text_pieces": " | ".join(t["text_piece"] for t in matched_tokens),
        "token_entropies": " ".join(f"{t['entropy']:.10f}" for t in matched_tokens),

        "phrase_entropy": phrase_entropy,
        "phrase_peak_entropy": phrase_peak_entropy,

        "peak_entropy_token_index": peak_token["token_index"],
        "peak_entropy_token_id": peak_token["token_id"],
        "peak_entropy_token": peak_token["text_piece"],
        "peak_entropy_token_raw": peak_token["token"],
        "peak_entropy_token_entropy": peak_token["entropy"],
        "peak_entropy_token_region": peak_token["region"],
    }


def build_case_summaries(occ_df):
    """
    每个 case 内，所有关键词、所有出现位置的平均。
    """
    if occ_df.empty:
        return pd.DataFrame()

    return (
        occ_df
        .groupby("case_index", as_index=False)
        .agg(
            n_occurrences=("phrase_entropy", "size"),
            phrase_entropy_mean=("phrase_entropy", "mean"),
            phrase_peak_entropy_mean=("phrase_peak_entropy", "mean"),
            peak_entropy_token_entropy_mean=("peak_entropy_token_entropy", "mean"),
        )
    )


def build_case_keyword_summaries(occ_df):
    """
    每个 case + 每个 keyword 的平均。
    """
    if occ_df.empty:
        return pd.DataFrame()

    return (
        occ_df
        .groupby(["case_index", "keyword"], as_index=False)
        .agg(
            n_occurrences=("phrase_entropy", "size"),
            phrase_entropy_mean=("phrase_entropy", "mean"),
            phrase_peak_entropy_mean=("phrase_peak_entropy", "mean"),
            peak_entropy_token_entropy_mean=("peak_entropy_token_entropy", "mean"),
        )
    )


def build_keyword_summaries(occ_df):
    """
    全局按 keyword 聚合。
    """
    if occ_df.empty:
        return pd.DataFrame()

    return (
        occ_df
        .groupby("keyword", as_index=False)
        .agg(
            n_occurrences=("phrase_entropy", "size"),
            phrase_entropy_mean=("phrase_entropy", "mean"),
            phrase_peak_entropy_mean=("phrase_peak_entropy", "mean"),
            peak_entropy_token_entropy_mean=("peak_entropy_token_entropy", "mean"),
        )
    )


def build_global_summary(occ_df, case_df, case_keyword_df):
    """
    micro:
      直接对所有关键词 occurrence 求平均。

    macro_case:
      先在每个 case 内平均，再对 case 求平均。
      只包含至少出现过关键词的 case。

    macro_case_keyword:
      先在每个 case + keyword 内平均，再对 case-keyword 单元求平均。
      只包含至少出现过该 keyword 的 case-keyword 单元。
    """
    rows = []

    if not occ_df.empty:
        rows.append({
            "level": "micro_occurrence",
            "n_units": len(occ_df),
            "phrase_entropy_mean": occ_df["phrase_entropy"].mean(),
            "phrase_peak_entropy_mean": occ_df["phrase_peak_entropy"].mean(),
            "peak_entropy_token_entropy_mean": occ_df["peak_entropy_token_entropy"].mean(),
        })

    if not case_df.empty:
        rows.append({
            "level": "macro_case",
            "n_units": len(case_df),
            "phrase_entropy_mean": case_df["phrase_entropy_mean"].mean(),
            "phrase_peak_entropy_mean": case_df["phrase_peak_entropy_mean"].mean(),
            "peak_entropy_token_entropy_mean": case_df["peak_entropy_token_entropy_mean"].mean(),
        })

    if not case_keyword_df.empty:
        rows.append({
            "level": "macro_case_keyword",
            "n_units": len(case_keyword_df),
            "phrase_entropy_mean": case_keyword_df["phrase_entropy_mean"].mean(),
            "phrase_peak_entropy_mean": case_keyword_df["phrase_peak_entropy_mean"].mean(),
            "peak_entropy_token_entropy_mean": case_keyword_df["peak_entropy_token_entropy_mean"].mean(),
        })

    return pd.DataFrame(rows)


def build_peak_token_counts(occ_df, top_k=100):
    """
    peak-entropy token 是字符串，不能真正求“平均值”。
    这里输出它的频率分布。
    peak token 的 entropy 平均值已经在 global_summary 中用
    peak_entropy_token_entropy_mean 表示。
    """
    if occ_df.empty:
        return pd.DataFrame()

    counter = Counter(occ_df["peak_entropy_token"].tolist())
    rows = []
    total = len(occ_df)

    for token, count in counter.most_common(top_k):
        rows.append({
            "peak_entropy_token": token,
            "count": count,
            "ratio": count / total,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="token-level entropy jsonl")
    parser.add_argument(
        "--keywords",
        required=True,
        help="关键词文件路径，或逗号分隔关键词，例如：北京大学,清华大学,合同法"
    )
    parser.add_argument("--out_dir", default="keyword_entropy_results")
    parser.add_argument(
        "--allow_overlap",
        action="store_true",
        help="是否允许同一个关键词重叠匹配"
    )
    parser.add_argument(
        "--top_k_peak_tokens",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    keywords = load_keywords(args.keywords)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_case(rows)

    occurrence_rows = []

    for case_index, tokens in sorted(groups.items()):
        answer_text, token_infos = build_answer_and_offsets(tokens)

        for keyword in keywords:
            spans = find_keyword_occurrences(
                answer_text,
                keyword,
                allow_overlap=args.allow_overlap,
            )

            for occ_id, (span_start, span_end) in enumerate(spans):
                matched_tokens = tokens_overlapping_span(
                    token_infos,
                    span_start,
                    span_end,
                )

                metric = compute_occurrence_metrics(
                    case_index=case_index,
                    keyword=keyword,
                    occ_id=occ_id,
                    span_start=span_start,
                    span_end=span_end,
                    answer_text=answer_text,
                    matched_tokens=matched_tokens,
                )

                if metric is not None:
                    occurrence_rows.append(metric)

    occ_df = pd.DataFrame(occurrence_rows)

    case_df = build_case_summaries(occ_df)
    case_keyword_df = build_case_keyword_summaries(occ_df)
    keyword_df = build_keyword_summaries(occ_df)
    global_df = build_global_summary(occ_df, case_df, case_keyword_df)
    peak_token_count_df = build_peak_token_counts(
        occ_df,
        top_k=args.top_k_peak_tokens,
    )

    occ_path = out_dir / "occurrences.csv"
    case_path = out_dir / "case_summary.csv"
    case_keyword_path = out_dir / "case_keyword_summary.csv"
    keyword_path = out_dir / "keyword_summary.csv"
    global_path = out_dir / "global_summary.csv"
    peak_token_path = out_dir / "peak_token_counts.csv"

    occ_df.to_csv(occ_path, index=False, encoding="utf-8-sig")
    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")
    case_keyword_df.to_csv(case_keyword_path, index=False, encoding="utf-8-sig")
    keyword_df.to_csv(keyword_path, index=False, encoding="utf-8-sig")
    global_df.to_csv(global_path, index=False, encoding="utf-8-sig")
    peak_token_count_df.to_csv(peak_token_path, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(f"  occurrence-level results     -> {occ_path}")
    print(f"  case-level summary           -> {case_path}")
    print(f"  case-keyword-level summary   -> {case_keyword_path}")
    print(f"  keyword-level summary        -> {keyword_path}")
    print(f"  global summary               -> {global_path}")
    print(f"  peak token counts            -> {peak_token_path}")

    print("\nGlobal summary:")
    if global_df.empty:
        print("No keyword occurrences found.")
    else:
        print(global_df.to_string(index=False))


if __name__ == "__main__":
    main()
