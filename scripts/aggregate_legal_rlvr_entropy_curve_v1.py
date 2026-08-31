#!/usr/bin/env python3
"""Aggregate legal performance/entropy outputs and draw checkpoint curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = [
    "checkpoint_step",
    "seed",
    "hint_level",
    "case_index",
    "probe_case_index",
    "case_id",
]
PERFORMANCE_COLUMNS = [
    "acc_source_law",
    "acc_source_article",
    "acc_all_law",
    "acc_all_article",
]
HINT_ORDER = ["none", "weak", "strong"]
HINT_COLORS = {"none": "#4C78A8", "weak": "#F58518", "strong": "#54A24B"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_runs(root: Path, include_incomplete: bool) -> list[Path]:
    runs: list[Path] = []
    for trajectory_path in sorted(root.glob("step_*/seed_*/trajectory_entropy.jsonl")):
        run_dir = trajectory_path.parent
        if not include_incomplete and not (run_dir / ".complete").is_file():
            continue
        if not (run_dir / "case_evals.jsonl").is_file():
            continue
        runs.append(run_dir)
    if not runs:
        raise SystemExit(f"no completed entropy runs found under {root}")
    return runs


def load_all(root: Path, include_incomplete: bool) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    for run_dir in discover_runs(root, include_incomplete):
        trajectories = pd.DataFrame(load_jsonl(run_dir / "trajectory_entropy.jsonl"))
        evaluations = pd.DataFrame(load_jsonl(run_dir / "case_evals.jsonl"))
        missing_trajectory = set(KEYS) - set(trajectories.columns)
        missing_evaluation = set(KEYS) - set(evaluations.columns)
        if missing_trajectory or missing_evaluation:
            raise ValueError(
                f"{run_dir}: missing keys trajectory={sorted(missing_trajectory)} "
                f"evaluation={sorted(missing_evaluation)}"
            )
        frame = trajectories.merge(
            evaluations,
            on=KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("", "_eval"),
        )
        if len(frame) != len(trajectories) or len(frame) != len(evaluations):
            raise ValueError(
                f"{run_dir}: trajectory/evaluation row mismatch "
                f"{len(trajectories)} vs {len(evaluations)}"
            )
        frame["run_dir"] = str(run_dir)
        merged_frames.append(frame)

    data = pd.concat(merged_frames, ignore_index=True)
    duplicated = data.duplicated(KEYS, keep=False)
    if duplicated.any():
        example = data.loc[duplicated, KEYS].head().to_dict("records")
        raise ValueError(f"duplicate trajectory keys: {example}")
    return data


def finite(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def cluster_values(group: pd.DataFrame, column: str) -> np.ndarray:
    per_case = (
        group[["case_id", column]]
        .assign(**{column: pd.to_numeric(group[column], errors="coerce")})
        .dropna()
        .groupby("case_id", sort=False)[column]
        .mean()
    )
    return per_case.to_numpy(dtype=float)


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, repetitions: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    point = float(values.mean())
    if len(values) == 1 or repetitions <= 0:
        return point, math.nan, math.nan
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def bootstrap_ratio(
    group: pd.DataFrame,
    numerator: str,
    denominator: str,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float, float]:
    per_case = (
        group[["case_id", numerator, denominator]]
        .assign(
            **{
                numerator: pd.to_numeric(group[numerator], errors="coerce"),
                denominator: pd.to_numeric(group[denominator], errors="coerce"),
            }
        )
        .dropna()
        .groupby("case_id", sort=False)[[numerator, denominator]]
        .sum()
    )
    per_case = per_case[per_case[denominator] > 0]
    if per_case.empty:
        return math.nan, math.nan, math.nan
    numerator_values = per_case[numerator].to_numpy(dtype=float)
    denominator_values = per_case[denominator].to_numpy(dtype=float)
    point = float(numerator_values.sum() / denominator_values.sum())
    if len(per_case) == 1 or repetitions <= 0:
        return point, math.nan, math.nan
    indices = rng.integers(0, len(per_case), size=(repetitions, len(per_case)))
    numerator_draws = numerator_values[indices].sum(axis=1)
    denominator_draws = denominator_values[indices].sum(axis=1)
    ratios = numerator_draws / denominator_draws
    low, high = np.quantile(ratios, [0.025, 0.975])
    return point, float(low), float(high)


def summarize(
    data: pd.DataFrame, bootstrap_repetitions: int, bootstrap_seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for (step, hint), group in data.groupby(
        ["checkpoint_step", "hint_level"], sort=True
    ):
        row: dict[str, Any] = {
            "checkpoint_step": int(step),
            "hint_level": hint,
            "n_cases": int(group["case_id"].nunique()),
            "n_trajectories": int(len(group)),
            "n_seeds": int(group["seed"].nunique()),
            "effective_answer_rate": float(group["effective_answer"].astype(float).mean()),
            "truncation_rate": float(group["truncated"].astype(float).mean()),
            "response_tokens_mean": float(
                pd.to_numeric(group["full_token_count"], errors="coerce").mean()
            ),
        }

        macro_columns = [
            *PERFORMANCE_COLUMNS,
            "full_entropy_mean",
            "reasoning_entropy_mean",
            "answer_entropy_mean",
        ]
        for column in macro_columns:
            point, low, high = bootstrap_mean(
                cluster_values(group, column), rng, bootstrap_repetitions
            )
            row[column] = point
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high

        for region in ("full", "reasoning", "answer"):
            point, low, high = bootstrap_ratio(
                group,
                f"{region}_entropy_sum",
                f"{region}_token_count",
                rng,
                bootstrap_repetitions,
            )
            row[f"{region}_entropy_micro"] = point
            row[f"{region}_entropy_micro_ci_low"] = low
            row[f"{region}_entropy_micro_ci_high"] = high

        effective = group[group["effective_answer"].astype(bool)]
        for column in PERFORMANCE_COLUMNS:
            values = cluster_values(effective, column)
            row[f"{column}_legacy_effective_only"] = (
                float(values.mean()) if len(values) else math.nan
            )

        correct = group[pd.to_numeric(group["acc_source_law"], errors="coerce") == 1]
        incorrect = group[pd.to_numeric(group["acc_source_law"], errors="coerce") != 1]
        row["full_entropy_correct"] = (
            float(finite(correct["full_entropy_mean"]).mean())
            if len(finite(correct["full_entropy_mean"]))
            else math.nan
        )
        row["full_entropy_incorrect"] = (
            float(finite(incorrect["full_entropy_mean"]).mean())
            if len(finite(incorrect["full_entropy_mean"]))
            else math.nan
        )
        row["reasoning_coverage"] = float(
            (pd.to_numeric(group["reasoning_token_count"], errors="coerce") > 0).mean()
        )
        row["answer_region_coverage"] = float(
            (pd.to_numeric(group["answer_token_count"], errors="coerce") > 0).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["hint_level", "checkpoint_step"])


def ordered_hints(summary: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(summary["hint_level"].tolist()))
    return [hint for hint in HINT_ORDER if hint in present] + [
        hint for hint in present if hint not in HINT_ORDER
    ]


def line_with_ci(
    axis: plt.Axes,
    frame: pd.DataFrame,
    y: str,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    frame = frame.sort_values("checkpoint_step")
    x = frame["checkpoint_step"].to_numpy(dtype=float)
    center = frame[y].to_numpy(dtype=float)
    axis.plot(x, center, marker="o", label=label, color=color, linestyle=linestyle)
    low_name = f"{y}_ci_low"
    high_name = f"{y}_ci_high"
    if low_name in frame and high_name in frame:
        low = frame[low_name].to_numpy(dtype=float)
        high = frame[high_name].to_numpy(dtype=float)
        valid = np.isfinite(low) & np.isfinite(high)
        if valid.any():
            axis.fill_between(x[valid], low[valid], high[valid], color=color, alpha=0.14)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_entropy(summary: pd.DataFrame, out_dir: Path) -> None:
    hints = ordered_hints(summary)
    fig, axes = plt.subplots(len(hints), 1, figsize=(9, 3.4 * len(hints)), sharex=True)
    if len(hints) == 1:
        axes = [axes]
    for axis, hint in zip(axes, hints):
        frame = summary[summary["hint_level"] == hint]
        line_with_ci(axis, frame, "full_entropy_mean", "Full (macro)", "#4C78A8")
        line_with_ci(
            axis, frame, "reasoning_entropy_mean", "Reasoning (macro)", "#E45756"
        )
        line_with_ci(axis, frame, "answer_entropy_mean", "Answer (macro)", "#54A24B")
        axis.set_title(f"Hint: {hint}")
        axis.set_ylabel("Entropy (nats)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("RLVR checkpoint step")
    save_figure(fig, out_dir / "legal_entropy_curve")


def plot_performance(summary: pd.DataFrame, out_dir: Path) -> None:
    hints = ordered_hints(summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    titles = {
        "acc_source_law": "Source law",
        "acc_source_article": "Source article",
        "acc_all_law": "All laws",
        "acc_all_article": "All articles",
    }
    for axis, metric in zip(axes.flat, PERFORMANCE_COLUMNS):
        for hint in hints:
            frame = summary[summary["hint_level"] == hint]
            line_with_ci(
                axis,
                frame,
                metric,
                hint,
                HINT_COLORS.get(hint, None) or "#777777",
            )
        axis.set_title(titles[metric])
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("RLVR checkpoint step")
        axis.set_ylabel("Accuracy")
        axis.grid(alpha=0.25)
        axis.legend()
    save_figure(fig, out_dir / "legal_performance_curve")


def plot_diagnostics(summary: pd.DataFrame, out_dir: Path) -> None:
    hints = ordered_hints(summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for hint in hints:
        frame = summary[summary["hint_level"] == hint].sort_values("checkpoint_step")
        color = HINT_COLORS.get(hint, "#777777")
        axes[0, 0].plot(
            frame["checkpoint_step"],
            frame["full_entropy_correct"],
            marker="o",
            color=color,
            label=hint,
        )
        axes[0, 1].plot(
            frame["checkpoint_step"],
            frame["full_entropy_incorrect"],
            marker="o",
            color=color,
            label=hint,
        )
        axes[1, 0].plot(
            frame["checkpoint_step"],
            frame["response_tokens_mean"],
            marker="o",
            color=color,
            label=hint,
        )
        axes[1, 1].plot(
            frame["checkpoint_step"],
            frame["truncation_rate"],
            marker="o",
            color=color,
            label=hint,
        )
    titles = [
        "Full entropy: correct source law",
        "Full entropy: incorrect source law",
        "Mean response tokens",
        "Truncation rate",
    ]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("RLVR checkpoint step")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0, 0].set_ylabel("Entropy (nats)")
    axes[0, 1].set_ylabel("Entropy (nats)")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 1].set_ylabel("Ratio")
    save_figure(fig, out_dir / "legal_entropy_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--include-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(root, args.include_incomplete)
    summary = summarize(data, args.bootstrap_repetitions, args.bootstrap_seed)

    data.to_csv(out_dir / "legal_entropy_trajectory_merged.csv", index=False)
    summary.to_csv(out_dir / "legal_entropy_curve_summary.csv", index=False)
    plot_entropy(summary, out_dir)
    plot_performance(summary, out_dir)
    plot_diagnostics(summary, out_dir)

    audit = {
        "input_root": str(root),
        "n_trajectories": int(len(data)),
        "n_cases": int(data["case_id"].nunique()),
        "steps": sorted(int(value) for value in data["checkpoint_step"].unique()),
        "seeds": sorted(int(value) for value in data["seed"].unique()),
        "hint_levels": ordered_hints(summary),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "recommended_performance_scope": (
            "all cases; legacy effective-only columns are included only for "
            "comparison with the original notebook"
        ),
    }
    with (out_dir / "aggregate_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(summary.to_string(index=False))
    print(f"[PASS] curve tables and figures -> {out_dir}")


if __name__ == "__main__":
    main()
