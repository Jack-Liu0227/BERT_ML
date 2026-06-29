from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SCORE_ROOT = Path("output") / "ood_xspace_scores"
METHOD_ORDER = [
    "random_cv_baseline",
    "target_extrapolation",
    "sparse_y_single",
    "sparse_y_cluster",
    "sparse_x_single",
    "sparse_x_cluster",
    "loco",
]
METHOD_LABELS = {
    "random_cv_baseline": "Random-CV",
    "target_extrapolation": "Target extrap.",
    "sparse_y_single": "Sparse-y single",
    "sparse_y_cluster": "Sparse-y cluster",
    "sparse_x_single": "Sparse-x single",
    "sparse_x_cluster": "Sparse-x cluster",
    "loco": "LOCO",
}
METHOD_COLORS = {
    "random_cv_baseline": "#8c8c8c",
    "target_extrapolation": "#4c78a8",
    "sparse_y_single": "#72b7b2",
    "sparse_y_cluster": "#54a24b",
    "sparse_x_single": "#f58518",
    "sparse_x_cluster": "#e45756",
    "loco": "#b279a2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clearer presentation/论文用 figures from X-space OOD scoring outputs."
    )
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT), help="Directory containing OOD score CSV outputs.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def ordered_methods(methods: pd.Series | list[str]) -> list[str]:
    available = set(str(method) for method in methods)
    known = [method for method in METHOD_ORDER if method in available]
    unknown = sorted(available - set(known))
    return known + unknown


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str], dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def load_outputs(score_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(score_root / "ood_split_summary.csv")
    samples = pd.read_csv(score_root / "all_ood_sample_scores.csv")
    methods = pd.read_csv(score_root / "ood_method_comparison.csv")
    return summary, samples, methods


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle], values: list[float], fmt: str = "{:.1f}") -> None:
    for bar, value in zip(bars, values):
        if not math.isfinite(float(value)):
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_dashboard(summary: pd.DataFrame, methods: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    methods_order = ordered_methods(methods["split_strategy"])
    method_frame = methods.set_index("split_strategy").loc[methods_order].reset_index()
    labels = [method_label(method) for method in method_frame["split_strategy"]]
    colors = [METHOD_COLORS.get(method, "#4c78a8") for method in method_frame["split_strategy"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.suptitle("X-space OOD evidence summary", fontsize=17, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    values = method_frame["test_ood_score_median_median"].astype(float).tolist()
    bars = ax.bar(labels, values, color=colors)
    baseline = float(method_frame.loc[method_frame["split_strategy"] == "random_cv_baseline", "test_ood_score_median_median"].iloc[0])
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.1, label="Random-CV baseline")
    ratio_col = "test_ood_score_median_median_vs_random_cv_ratio"
    ratios = method_frame[ratio_col].astype(float).tolist() if ratio_col in method_frame.columns else [np.nan] * len(values)
    annotate_bars(ax, bars, ratios, fmt="{:.1f}x")
    ax.set_ylabel("Median test OOD score")
    ax.set_title("A. Per-sample OOD score is higher for designed OOD splits", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)

    ax = axes[0, 1]
    box_data = [
        pd.to_numeric(summary.loc[summary["split_strategy"] == method, "test_ood_score_median"], errors="coerce").dropna()
        for method in methods_order
    ]
    box = ax.boxplot(box_data, patch_artist=True, tick_labels=labels, showfliers=False, medianprops={"color": "black"})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Split-level median OOD score")
    ax.set_title("B. Split distribution of OOD scores", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    style_axes(ax)

    ax = axes[1, 0]
    distance_metrics = [
        ("sliced_wasserstein_median_vs_random_cv_ratio", "Sliced W"),
        ("mmd_rbf_median_vs_random_cv_ratio", "MMD"),
        ("energy_distance_median_vs_random_cv_ratio", "Energy"),
    ]
    width = 0.24
    x = np.arange(len(method_frame))
    for offset_index, (metric_col, metric_label) in enumerate(distance_metrics):
        values = method_frame[metric_col].astype(float).to_numpy()
        ax.bar(x + (offset_index - 1) * width, values, width=width, label=metric_label)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Ratio vs Random-CV, log scale")
    ax.set_title("C. Whole-set distribution distances are larger", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    style_axes(ax)

    ax = axes[1, 1]
    heatmap_metrics = [
        ("test_ood_score_median_median_vs_random_cv_ratio", "OOD score"),
        ("sliced_wasserstein_median_vs_random_cv_ratio", "Sliced W"),
        ("mmd_rbf_median_vs_random_cv_ratio", "MMD"),
        ("energy_distance_median_vs_random_cv_ratio", "Energy"),
        ("test_ood_percentile_median_median_vs_random_cv_ratio", "OOD percentile"),
    ]
    heat_values = method_frame[[metric for metric, _ in heatmap_metrics]].astype(float).replace([np.inf, -np.inf], np.nan)
    log_values = np.log2(heat_values.to_numpy())
    image = ax.imshow(log_values.T, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heatmap_metrics)))
    ax.set_yticklabels([label for _, label in heatmap_metrics])
    ax.set_title("D. Evidence ratio heatmap, log2 scale", loc="left", fontweight="bold")
    for row in range(log_values.shape[1]):
        for col in range(log_values.shape[0]):
            value = heat_values.iloc[col, row]
            if math.isfinite(float(value)):
                ax.text(col, row, f"{value:.1f}x", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("log2 ratio vs Random-CV")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, figure_dir / "ood_evidence_dashboard", formats, dpi)


def plot_sample_distribution(samples: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    methods_order = ordered_methods(samples["split_strategy"])
    labels = [method_label(method) for method in methods_order]
    colors = [METHOD_COLORS.get(method, "#4c78a8") for method in methods_order]
    data = [
        np.log1p(pd.to_numeric(samples.loc[samples["split_strategy"] == method, "ood_score"], errors="coerce").dropna())
        for method in methods_order
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    parts = ax.violinplot(data, positions=np.arange(len(methods_order)), showmeans=False, showmedians=True, widths=0.78)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.6)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.0)

    medians = [float(np.median(values)) if len(values) else np.nan for values in data]
    ax.scatter(np.arange(len(methods_order)), medians, color="white", edgecolor="#222222", zorder=3, s=38)
    ax.set_xticks(np.arange(len(methods_order)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("log(1 + per-sample OOD score)")
    ax.set_title("Per-test-sample OOD score distribution", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ood_sample_score_distribution", formats, dpi)


def plot_wasserstein_scatter(summary: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for method in ordered_methods(summary["split_strategy"]):
        subset = summary[summary["split_strategy"] == method]
        ax.scatter(
            subset["sliced_wasserstein"],
            subset["test_ood_score_median"],
            s=48,
            alpha=0.78,
            color=METHOD_COLORS.get(method, "#4c78a8"),
            label=method_label(method),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Sliced Wasserstein distance, log scale")
    ax.set_ylabel("Median test OOD score, symlog scale")
    ax.set_title("Set-level W distance agrees with per-sample OOD score", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, figure_dir / "ood_wasserstein_vs_score", formats, dpi)


def plot_case_method_heatmap(summary: pd.DataFrame, figure_dir: Path, formats: list[str], dpi: int) -> None:
    work = summary.copy()
    work["case"] = work["alloy_family"].astype(str) + " / " + work["property"].astype(str)
    case_order = (
        work[["case", "alloy_family", "property"]]
        .drop_duplicates()
        .sort_values(["alloy_family", "property"], kind="stable")["case"]
        .tolist()
    )
    method_order = ordered_methods(work["split_strategy"])
    pivot = (
        work.pivot_table(index="case", columns="split_strategy", values="test_ood_score_median", aggfunc="median")
        .reindex(index=case_order, columns=method_order)
    )
    values = np.log1p(pivot.to_numpy(dtype=float))
    labels_x = [method_label(method) for method in method_order]
    labels_y = ["\n".join(wrap(case, width=22)) for case in pivot.index]

    fig_height = max(5.2, 0.42 * len(labels_y) + 2.0)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title("Case-by-method median OOD score heatmap", fontweight="bold")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if math.isfinite(float(value)):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.set_label("log(1 + median OOD score)")
    fig.tight_layout()
    save_figure(fig, figure_dir / "ood_case_method_heatmap", formats, dpi)


def main() -> None:
    args = parse_args()
    score_root = Path(args.score_root)
    figure_dir = score_root / "figures"
    summary, samples, methods = load_outputs(score_root)

    plot_dashboard(summary, methods, figure_dir, args.formats, args.dpi)
    plot_sample_distribution(samples, figure_dir, args.formats, args.dpi)
    plot_wasserstein_scatter(summary, figure_dir, args.formats, args.dpi)
    plot_case_method_heatmap(summary, figure_dir, args.formats, args.dpi)

    print(f"Wrote clearer OOD figures to {figure_dir}")


if __name__ == "__main__":
    main()
