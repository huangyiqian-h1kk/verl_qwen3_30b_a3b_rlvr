#!/usr/bin/env python3
"""Compute exact-phrase entropy and coverage curves across RLVR checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HINT_ORDER = ["none", "weak", "strong"]
HINT_COLORS = {"none": "#4C78A8", "weak": "#F58518", "strong": "#54A24B"}


def load_keywords(value: str) -> list[str]:
    path = Path(value)
    if path.is_file():
        keywords = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        keywords = [item.strip() for item in value.split(",") if item.strip()]
    if not keywords:
        raise ValueError("keyword list is empty")
    if len(keywords) != len(set(keywords)):
        raise ValueError("duplicate keywords are not allowed")
    return keywords


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_runs(root: Path, include_incomplete: bool) -> list[Path]:
    runs: list[Path] = []
    for token_path in sorted(root.glob("step_*/seed_*/token_entropy.jsonl")):
        run_dir = token_path.parent
        if not include_incomplete and not (run_dir / ".complete").is_file():
            continue
        if (run_dir / "case_tests.jsonl").is_file():
            runs.append(run_dir)
    if not runs:
        raise SystemExit(f"no completed token-entropy runs found under {root}")
    return runs


def token_overlaps(row: dict[str, Any], start: int, end: int) -> bool:
    return int(row["char_end"]) > start and int(row["char_start"]) < end


def collect_occurrences(
    root: Path, keywords: list[str], include_incomplete: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    occurrence_rows: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []

    for run_dir in discover_runs(root, include_incomplete):
        case_tests = load_jsonl(run_dir / "case_tests.jsonl")
        for row in case_tests:
            denominator_rows.append(
                {
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "seed": int(row["seed"]),
                    "hint_level": row["hint_level"],
                    "case_id": row["case_id"],
                }
            )

        grouped_tokens: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        with (run_dir / "token_entropy.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    int(row["checkpoint_step"]),
                    int(row["seed"]),
                    row["hint_level"],
                    int(row["case_index"]),
                    row["case_id"],
                )
                grouped_tokens[key].append(row)

        for key, tokens in grouped_tokens.items():
            step, seed, hint, case_index, case_id = key
            tokens.sort(key=lambda item: int(item["token_index"]))
            text = "".join(str(item.get("text_piece", "")) for item in tokens)
            for keyword in keywords:
                for occurrence_id, match in enumerate(
                    re.finditer(re.escape(keyword), text)
                ):
                    matched = [
                        item
                        for item in tokens
                        if token_overlaps(item, match.start(), match.end())
                        and item.get("entropy") is not None
                    ]
                    if not matched:
                        continue
                    entropies = [float(item["entropy"]) for item in matched]
                    peak = max(matched, key=lambda item: float(item["entropy"]))
                    occurrence_rows.append(
                        {
                            "checkpoint_step": step,
                            "seed": seed,
                            "hint_level": hint,
                            "case_index": case_index,
                            "case_id": case_id,
                            "keyword": keyword,
                            "occurrence_id": occurrence_id,
                            "char_start": match.start(),
                            "char_end": match.end(),
                            "matched_text": match.group(0),
                            "n_tokens_in_phrase": len(matched),
                            "phrase_entropy": float(np.mean(entropies)),
                            "phrase_peak_entropy": float(peak["entropy"]),
                            "peak_token": peak.get("text_piece", ""),
                            "peak_token_index": int(peak["token_index"]),
                            "peak_token_region": peak.get("region"),
                        }
                    )

    return pd.DataFrame(occurrence_rows), pd.DataFrame(denominator_rows)


def summarize(
    occurrences: pd.DataFrame, denominators: pd.DataFrame, keywords: list[str]
) -> pd.DataFrame:
    denominator = (
        denominators.drop_duplicates(
            ["checkpoint_step", "seed", "hint_level", "case_id"]
        )
        .groupby(["checkpoint_step", "hint_level"], as_index=False)
        .agg(
            n_trajectories_total=("case_id", "size"),
            n_cases_total=("case_id", "nunique"),
            n_seeds=("seed", "nunique"),
        )
    )

    grid = denominator.assign(_join_key=1).merge(
        pd.DataFrame({"keyword": keywords, "_join_key": 1}), on="_join_key"
    ).drop(columns="_join_key")

    if occurrences.empty:
        for column in [
            "n_occurrences",
            "n_trajectories_with_keyword",
            "keyword_trajectory_coverage",
            "phrase_entropy_micro_occurrence",
            "phrase_entropy_macro_trajectory",
            "phrase_peak_entropy_macro_trajectory",
        ]:
            grid[column] = 0 if column.startswith("n_") else math.nan
        return grid

    trajectory = (
        occurrences.groupby(
            ["checkpoint_step", "seed", "hint_level", "case_id", "keyword"],
            as_index=False,
        )
        .agg(
            n_occurrences=("phrase_entropy", "size"),
            phrase_entropy_trajectory=("phrase_entropy", "mean"),
            phrase_peak_entropy_trajectory=("phrase_peak_entropy", "mean"),
        )
    )
    aggregate = (
        trajectory.groupby(
            ["checkpoint_step", "hint_level", "keyword"], as_index=False
        )
        .agg(
            n_occurrences=("n_occurrences", "sum"),
            n_trajectories_with_keyword=("case_id", "size"),
            phrase_entropy_macro_trajectory=(
                "phrase_entropy_trajectory",
                "mean",
            ),
            phrase_peak_entropy_macro_trajectory=(
                "phrase_peak_entropy_trajectory",
                "mean",
            ),
        )
    )
    occurrence_mean = (
        occurrences.groupby(
            ["checkpoint_step", "hint_level", "keyword"], as_index=False
        )
        .agg(phrase_entropy_micro_occurrence=("phrase_entropy", "mean"))
    )
    aggregate = aggregate.merge(
        occurrence_mean,
        on=["checkpoint_step", "hint_level", "keyword"],
        how="left",
    )
    result = grid.merge(
        aggregate,
        on=["checkpoint_step", "hint_level", "keyword"],
        how="left",
    )
    result["n_occurrences"] = result["n_occurrences"].fillna(0).astype(int)
    result["n_trajectories_with_keyword"] = (
        result["n_trajectories_with_keyword"].fillna(0).astype(int)
    )
    result["keyword_trajectory_coverage"] = (
        result["n_trajectories_with_keyword"] / result["n_trajectories_total"]
    )
    return result.sort_values(["keyword", "hint_level", "checkpoint_step"])


def ordered_hints(summary: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(summary["hint_level"].tolist()))
    return [hint for hint in HINT_ORDER if hint in present] + [
        hint for hint in present if hint not in HINT_ORDER
    ]


def plot_grid(
    summary: pd.DataFrame,
    keywords: list[str],
    y_column: str,
    y_label: str,
    output: Path,
) -> None:
    columns = 3
    rows = math.ceil(len(keywords) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 3.2 * rows), squeeze=False
    )
    hints = ordered_hints(summary)
    for keyword_index, (axis, keyword) in enumerate(zip(axes.flat, keywords), start=1):
        keyword_frame = summary[summary["keyword"] == keyword]
        for hint in hints:
            frame = keyword_frame[
                keyword_frame["hint_level"] == hint
            ].sort_values("checkpoint_step")
            axis.plot(
                frame["checkpoint_step"],
                frame[y_column],
                marker="o",
                label=hint,
                color=HINT_COLORS.get(hint, "#777777"),
            )
        # Use an ASCII key so plots render correctly on clusters without CJK fonts.
        axis.set_title(f"K{keyword_index:02d}")
        axis.set_xlabel("RLVR checkpoint step")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(keywords) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--keywords", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keywords = load_keywords(args.keywords)

    occurrences, denominators = collect_occurrences(
        root, keywords, args.include_incomplete
    )
    summary = summarize(occurrences, denominators, keywords)
    occurrences.to_csv(out_dir / "keyword_occurrences.csv", index=False)
    summary.to_csv(out_dir / "keyword_entropy_curve_summary.csv", index=False)
    pd.DataFrame(
        {
            "plot_key": [f"K{index:02d}" for index in range(1, len(keywords) + 1)],
            "keyword": keywords,
        }
    ).to_csv(out_dir / "keyword_plot_key.csv", index=False, encoding="utf-8-sig")
    plot_grid(
        summary,
        keywords,
        "phrase_entropy_macro_trajectory",
        "Phrase entropy (nats)",
        out_dir / "legal_keyword_entropy_curve",
    )
    plot_grid(
        summary,
        keywords,
        "keyword_trajectory_coverage",
        "Trajectory coverage",
        out_dir / "legal_keyword_coverage_curve",
    )
    print(summary.to_string(index=False))
    print(f"[PASS] keyword curves -> {out_dir}")


if __name__ == "__main__":
    main()
