"""
Compare TabPFN baseline results with fine-tuned results.

Default inputs:
- output/TabPFN_results/summary_results.csv
- output/TabPFN_finetune_results/summary_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS: List[str] = [
    "Train_MAE",
    "Train_RMSE",
    "Train_R2",
    "Train_MAPE",
    "Test_MAE",
    "Test_RMSE",
    "Test_R2",
    "Test_MAPE",
]

# 1 means higher is better; -1 means lower is better.
METRIC_DIRECTIONS: Dict[str, int] = {
    "Train_MAE": -1,
    "Train_RMSE": -1,
    "Train_R2": 1,
    "Train_MAPE": -1,
    "Test_MAE": -1,
    "Test_RMSE": -1,
    "Test_R2": 1,
    "Test_MAPE": -1,
}


def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0.0, np.nan).abs()
    return numerator.astype(float) / denom * 100.0


def compare_results(
    before_csv: Path,
    after_csv: Path,
    output_csv: Path,
    sort_by: str,
) -> pd.DataFrame:
    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)

    key_cols = ["Alloy", "Target"]
    missing_before = [c for c in key_cols + METRICS if c not in before_df.columns]
    missing_after = [c for c in key_cols + METRICS if c not in after_df.columns]
    if missing_before:
        raise ValueError(f"Missing columns in baseline CSV: {missing_before}")
    if missing_after:
        raise ValueError(f"Missing columns in finetune CSV: {missing_after}")

    merged = pd.merge(
        before_df[key_cols + METRICS],
        after_df[key_cols + METRICS],
        on=key_cols,
        how="outer",
        suffixes=("_Before", "_After"),
        indicator=True,
    )

    for metric in METRICS:
        before_col = f"{metric}_Before"
        after_col = f"{metric}_After"
        delta_col = f"{metric}_Delta"
        delta_pct_col = f"{metric}_DeltaPct"
        improve_col = f"{metric}_Improvement"
        improve_pct_col = f"{metric}_ImprovementPct"

        merged[delta_col] = merged[after_col] - merged[before_col]
        merged[delta_pct_col] = _safe_percent(merged[delta_col], merged[before_col])

        direction = METRIC_DIRECTIONS[metric]
        # Positive value means "improved", regardless of metric direction.
        merged[improve_col] = merged[delta_col] * direction
        merged[improve_pct_col] = _safe_percent(merged[improve_col], merged[before_col])

    if sort_by not in METRICS:
        raise ValueError(f"sort_by must be one of: {METRICS}")

    sort_col = f"{sort_by}_Improvement"
    merged = merged.sort_values(by=sort_col, ascending=False, kind="stable").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def print_brief_summary(df: pd.DataFrame):
    common_rows = df["_merge"].eq("both").sum()
    baseline_only = df["_merge"].eq("left_only").sum()
    finetune_only = df["_merge"].eq("right_only").sum()

    print("\nComparison overview")
    print("=" * 60)
    print(f"Matched tasks:       {common_rows}")
    print(f"Baseline only tasks: {baseline_only}")
    print(f"Finetune only tasks: {finetune_only}")

    if common_rows == 0:
        return

    common_df = df[df["_merge"] == "both"].copy()
    test_metrics = ["Test_MAE", "Test_RMSE", "Test_R2", "Test_MAPE"]
    print("\nImprovement stats on test metrics (matched tasks)")
    for metric in test_metrics:
        improve_col = f"{metric}_Improvement"
        improved = int((common_df[improve_col] > 0).sum())
        worsened = int((common_df[improve_col] < 0).sum())
        unchanged = int((common_df[improve_col] == 0).sum())
        mean_improve = float(common_df[improve_col].mean())
        print(
            f"- {metric}: improved={improved}, worsened={worsened}, "
            f"unchanged={unchanged}, mean_improvement={mean_improve:.6f}"
        )

    show_cols = [
        "Alloy",
        "Target",
        "Test_R2_Before",
        "Test_R2_After",
        "Test_R2_Improvement",
        "Test_MAE_Before",
        "Test_MAE_After",
        "Test_MAE_Improvement",
    ]
    print("\nTop 5 by Test_R2 improvement")
    print(common_df.sort_values("Test_R2_Improvement", ascending=False)[show_cols].head(5).to_string(index=False))


def save_visualizations(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    common_df = df[df["_merge"] == "both"].copy()
    if common_df.empty:
        print("No matched tasks for plotting.")
        return

    common_df["Task"] = common_df["Alloy"].astype(str) + " | " + common_df["Target"].astype(str)

    # 1) Per-task improvement bars for key test metrics.
    metrics = ["Test_R2_Improvement", "Test_MAE_Improvement", "Test_RMSE_Improvement", "Test_MAPE_Improvement"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        sorted_df = common_df.sort_values(metric, ascending=True)
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in sorted_df[metric]]
        ax.barh(sorted_df["Task"], sorted_df[metric], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Improvement (>0 is better)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Fine-tuning Improvement by Task (Test Metrics)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "test_metric_improvements_by_task.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Before vs after scatter for selected test metrics.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter_specs = [("Test_R2", True), ("Test_MAE", False)]
    for ax, (metric, higher_is_better) in zip(axes, scatter_specs):
        before = common_df[f"{metric}_Before"]
        after = common_df[f"{metric}_After"]
        ax.scatter(before, after, alpha=0.8, color="#1f77b4", edgecolors="k", linewidths=0.4)
        min_val = float(min(before.min(), after.min()))
        max_val = float(max(before.max(), after.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5)
        ax.set_xlabel(f"{metric} Before")
        ax.set_ylabel(f"{metric} After")
        direction_hint = "upper" if higher_is_better else "lower"
        ax.set_title(f"{metric}: better points in {direction_hint} triangle")
        ax.grid(alpha=0.25)
    fig.suptitle("Fine-tuning: Before vs After", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "before_vs_after_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Count of improved/worsened tasks on test metrics.
    test_metrics = ["Test_MAE", "Test_RMSE", "Test_R2", "Test_MAPE"]
    improved_counts = [(common_df[f"{m}_Improvement"] > 0).sum() for m in test_metrics]
    worsened_counts = [(common_df[f"{m}_Improvement"] < 0).sum() for m in test_metrics]
    unchanged_counts = [(common_df[f"{m}_Improvement"] == 0).sum() for m in test_metrics]

    x = np.arange(len(test_metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, improved_counts, width=width, label="Improved", color="#2ca02c")
    ax.bar(x, worsened_counts, width=width, label="Worsened", color="#d62728")
    ax.bar(x + width, unchanged_counts, width=width, label="Unchanged", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(test_metrics)
    ax.set_ylabel("Task Count")
    ax.set_title("How Many Tasks Improved After Fine-tuning")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "improved_worsened_counts.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline TabPFN summary results with fine-tuned summary results."
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=Path("output/TabPFN_results/summary_results.csv"),
        help="Path to baseline summary_results.csv",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=Path("output/TabPFN_finetune_results/summary_results.csv"),
        help="Path to fine-tuned summary_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/TabPFN_compare/finetune_vs_baseline_summary.csv"),
        help="Path to save comparison CSV",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="Test_R2",
        choices=METRICS,
        help="Metric used to sort rows by improvement.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("output/TabPFN_compare"),
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable visualization output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = compare_results(
        before_csv=args.before,
        after_csv=args.after,
        output_csv=args.output,
        sort_by=args.sort_by,
    )
    if not args.no_plots:
        save_visualizations(df, args.plot_dir)
    print_brief_summary(df)
    print(f"\nDetailed comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
