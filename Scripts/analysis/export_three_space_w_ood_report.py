from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial import distance_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_X_REPORT_ROOT = Path("output") / "ood_xspace_scores" / "sample_w_reports"
DEFAULT_YZ_SUMMARY = (
    Path("D:/XJTU")
    / "\u5df2\u5b8c\u6210\u8bba\u6587\u6570\u636e\u6c47\u603b"
    / "Fewshot"
    / "\u9884\u5904\u7406\u6c47\u603b\u6570\u636e"
    / "OOD"
    / "comparison_outputs"
    / "wasserstein_distance_standardized"
    / "ood_wasserstein_summary.csv"
)
DEFAULT_EMBEDDING_DATA_DIR = (
    Path("D:/XJTU")
    / "\u5df2\u5b8c\u6210\u8bba\u6587\u6570\u636e\u6c47\u603b"
    / "Fewshot"
    / "\u9884\u5904\u7406\u6c47\u603b\u6570\u636e"
    / "OOD"
    / "embedding_data"
)

CASE_COLUMNS = ["alloy_family", "dataset_name", "property"]
METHOD_ORDER = [
    "random_cv_baseline",
    "loco",
    "target_extrapolation",
    "sparse_x_single",
    "sparse_y_single",
    "sparse_x_cluster",
    "sparse_y_cluster",
]
METHOD_LABELS = {
    "random_cv_baseline": "Random-CV",
    "loco": "LOCO",
    "target_extrapolation": "Target extrap.",
    "sparse_x_single": "Sparse-x single",
    "sparse_y_single": "Sparse-y single",
    "sparse_x_cluster": "Sparse-x cluster",
    "sparse_y_cluster": "Sparse-y cluster",
}
METHOD_COLORS = {
    "random_cv_baseline": "#8c8c8c",
    "loco": "#b279a2",
    "target_extrapolation": "#4c78a8",
    "sparse_x_single": "#f58518",
    "sparse_y_single": "#72b7b2",
    "sparse_x_cluster": "#e45756",
    "sparse_y_cluster": "#54a24b",
}
METHOD_CURVE_COLORS = {
    "random_cv_baseline": "#111111",
    "loco": "#0072B2",
    "target_extrapolation": "#D55E00",
    "sparse_x_single": "#E69F00",
    "sparse_y_single": "#009E73",
    "sparse_x_cluster": "#CC79A7",
    "sparse_y_cluster": "#56B4E9",
}
SPACE_COLORS = {
    "Y-space": "#6b9ac4",
    "Z-space": "#72b7b2",
    "X-space": "#d95f02",
}
FOLD_COLORS = {
    "fold_0": "#1f77b4",
    "fold_1": "#ff7f0e",
    "fold_2": "#2ca02c",
    "fold_3": "#d62728",
    "fold_4": "#9467bd",
}
GROUPED_AGGREGATION_METHODS = {"random_cv_baseline", "loco"}
FOLD_AWARE_METHODS = {"random_cv_baseline", "loco"}

YZ_DATASET_TO_CASE = {
    "Ti__titanium__UTSMPa": ("Ti", "titanium", "UTS(MPa)"),
    "Ti__titanium__Elpct": ("Ti", "titanium", "El(%)"),
    "Al__aluminum__UTSMPa": ("Al", "aluminum", "UTS(MPa)"),
    "HEA__hea__UTSMPa": ("HEA", "hea", "UTS(MPa)"),
    "HEA__hea__YSMPa": ("HEA", "hea", "YS(MPa)"),
    "HEA__hea__Elpct": ("HEA", "hea", "El(%)"),
    "Steel__steel__UTSMPa": ("Steel", "steel", "UTS(MPa)"),
    "Steel__steel__Elpct": ("Steel", "steel", "El(%)"),
    "Matbench_Steel__matbench_steels_ood__yieldstrength": (
        "MatbenchSteels",
        "matbench_steels_ood",
        "yield strength",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Chinese three-space W-OOD report from existing X-space sample "
            "W scores and old standardized Y/Z-space Wasserstein summaries."
        )
    )
    parser.add_argument(
        "--x-report-root",
        default=str(DEFAULT_X_REPORT_ROOT),
        help="Directory containing all_sample_w_values.csv and case_method_w_summary.csv.",
    )
    parser.add_argument(
        "--yz-summary",
        default=str(DEFAULT_YZ_SUMMARY),
        help="Old standardized Y/Z-space ood_wasserstein_summary.csv.",
    )
    parser.add_argument(
        "--split-summary",
        default=None,
        help="X-space ood_split_summary.csv. Defaults to <x-report-root>/../ood_split_summary.csv.",
    )
    parser.add_argument(
        "--embedding-data-dir",
        default=str(DEFAULT_EMBEDDING_DATA_DIR),
        help="Directory containing old *_all_panel_test_points.csv embedding files.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory. Defaults to --x-report-root.",
    )
    parser.add_argument("--top-n", type=int, default=25, help="Top X-space W samples per case x method.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def remove_stale_task_dashboard_figures(case_dir: Path) -> None:
    for fmt in ("png", "pdf", "svg"):
        stale_path = case_dir / f"task_w_dashboard.{fmt}"
        if stale_path.exists():
            stale_path.unlink()


def slugify(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text or "unknown"


def case_key_from_values(alloy_family: object, dataset_name: object, property_name: object) -> str:
    return "__".join(slugify(value) for value in (alloy_family, dataset_name, property_name))


def case_label_from_values(alloy_family: object, dataset_name: object, property_name: object) -> str:
    return f"{alloy_family} / {dataset_name} / {property_name}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4c78a8")


def method_curve_color(method: str) -> str:
    return METHOD_CURVE_COLORS.get(method, method_color(method))


def required_x_columns() -> tuple[list[str], list[str]]:
    sample_columns = [
        *CASE_COLUMNS,
        "method",
        "fold_id",
        "target_value",
        "ood_score",
        "sample_w_contribution",
        "sample_w_rank_desc",
    ]
    summary_columns = [
        *CASE_COLUMNS,
        "method",
        "sample_count",
        "sample_w_mean",
        "sample_w_median",
        "sample_w_q25",
        "sample_w_q75",
        "sample_w_p90",
        "sample_w_max",
        "sample_w_top10pct_mean",
    ]
    return sample_columns, summary_columns


def load_inputs(x_report_root: Path, yz_summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_path = x_report_root / "all_sample_w_values.csv"
    x_summary_path = x_report_root / "case_method_w_summary.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing X-space sample W table: {sample_path}")
    if not x_summary_path.exists():
        raise FileNotFoundError(f"Missing X-space W summary table: {x_summary_path}")
    if not yz_summary_path.exists():
        raise FileNotFoundError(f"Missing Y/Z-space Wasserstein summary: {yz_summary_path}")

    sample_values = read_csv(sample_path)
    x_summary = read_csv(x_summary_path)
    yz_summary = read_csv(yz_summary_path)

    sample_required, x_summary_required = required_x_columns()
    missing_sample = [column for column in sample_required if column not in sample_values.columns]
    missing_x_summary = [column for column in x_summary_required if column not in x_summary.columns]
    missing_yz = [
        column
        for column in ["dataset_key", "method", "aggregation", "test_n", "target_w1_std", "xy_w1_nd_std"]
        if column not in yz_summary.columns
    ]
    if missing_sample:
        raise ValueError(f"{sample_path} is missing required columns: {missing_sample}")
    if missing_x_summary:
        raise ValueError(f"{x_summary_path} is missing required columns: {missing_x_summary}")
    if missing_yz:
        raise ValueError(f"{yz_summary_path} is missing required columns: {missing_yz}")
    return sample_values, x_summary, yz_summary


def add_case_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["case_key"] = [
        case_key_from_values(row["alloy_family"], row["dataset_name"], row["property"])
        for _, row in result.iterrows()
    ]
    result["case_label"] = [
        case_label_from_values(row["alloy_family"], row["dataset_name"], row["property"])
        for _, row in result.iterrows()
    ]
    return result


def ordered_case_frame(x_summary: pd.DataFrame) -> pd.DataFrame:
    cases = add_case_fields(x_summary[CASE_COLUMNS].drop_duplicates()).sort_values(CASE_COLUMNS, kind="stable")
    return cases.reset_index(drop=True)


def build_x_metric_table(x_summary: pd.DataFrame) -> pd.DataFrame:
    x = add_case_fields(x_summary)
    keep_cols = [
        "case_key",
        "method",
        "sample_count",
        "sample_w_mean",
        "sample_w_median",
        "sample_w_q25",
        "sample_w_q75",
        "sample_w_p90",
        "sample_w_max",
        "sample_w_top10pct_mean",
    ]
    x = x[keep_cols].copy()
    x = x.rename(
        columns={
            "sample_count": "x_sample_count",
            "sample_w_mean": "X_mean_W",
            "sample_w_median": "X_median_W",
            "sample_w_q25": "X_q25_W",
            "sample_w_q75": "X_q75_W",
            "sample_w_p90": "X_p90_W",
            "sample_w_max": "X_max_W",
            "sample_w_top10pct_mean": "X_top10pct_mean_W",
        }
    )
    for column in x.columns:
        if column not in {"case_key", "method"}:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x


def select_yz_row(method_frame: pd.DataFrame, method: str) -> pd.Series | None:
    if method_frame.empty:
        return None
    preferred = "weighted_by_test_n" if method in GROUPED_AGGREGATION_METHODS else "single_split"
    selected = method_frame[method_frame["aggregation"] == preferred]
    if selected.empty and method in GROUPED_AGGREGATION_METHODS:
        selected = method_frame[method_frame["aggregation"] == "mean_over_groups"]
    if selected.empty:
        selected = method_frame
    return selected.iloc[0]


def build_yz_metric_table(yz_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary = yz_summary.copy()
    summary["method"] = summary["method"].astype(str).str.strip()
    summary["dataset_key"] = summary["dataset_key"].astype(str).str.strip()
    summary["aggregation"] = summary["aggregation"].astype(str).str.strip()
    for dataset_key, case_values in YZ_DATASET_TO_CASE.items():
        dataset_rows = summary[summary["dataset_key"] == dataset_key]
        if dataset_rows.empty:
            continue
        case_key = case_key_from_values(*case_values)
        for method in METHOD_ORDER:
            record = select_yz_row(dataset_rows[dataset_rows["method"] == method], method)
            if record is None:
                continue
            rows.append(
                {
                    "case_key": case_key,
                    "method": method,
                    "yz_dataset_key": dataset_key,
                    "yz_aggregation": record["aggregation"],
                    "yz_n_test": pd.to_numeric(record["test_n"], errors="coerce"),
                    "Y_W": pd.to_numeric(record["target_w1_std"], errors="coerce"),
                    "Z_W": pd.to_numeric(record["xy_w1_nd_std"], errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def build_three_space_summary(x_summary: pd.DataFrame, yz_summary: pd.DataFrame) -> pd.DataFrame:
    cases = ordered_case_frame(x_summary)
    base_rows: list[dict[str, object]] = []
    for _, case in cases.iterrows():
        for method in METHOD_ORDER:
            base_rows.append(
                {
                    "case_key": case["case_key"],
                    "case_label": case["case_label"],
                    "alloy_family": case["alloy_family"],
                    "dataset_name": case["dataset_name"],
                    "property": case["property"],
                    "method": method,
                    "method_label": method_label(method),
                }
            )
    base = pd.DataFrame(base_rows)
    merged = base.merge(build_x_metric_table(x_summary), on=["case_key", "method"], how="left")
    merged = merged.merge(build_yz_metric_table(yz_summary), on=["case_key", "method"], how="left")
    merged["x_sample_count"] = pd.to_numeric(merged["x_sample_count"], errors="coerce")
    merged["yz_n_test"] = pd.to_numeric(merged["yz_n_test"], errors="coerce")
    merged["n_test"] = merged["x_sample_count"].where(merged["x_sample_count"].notna(), merged["yz_n_test"])

    for ratio_column, metric_column in [
        ("Y_ratio", "Y_W"),
        ("Z_ratio", "Z_W"),
        ("X_ratio", "X_median_W"),
        ("X_top10pct_mean_ratio", "X_top10pct_mean_W"),
    ]:
        merged[ratio_column] = np.nan
        for case_key, indices in merged.groupby("case_key").groups.items():
            group = merged.loc[indices]
            baseline = group.loc[group["method"] == "random_cv_baseline", metric_column]
            baseline_value = float(baseline.iloc[0]) if len(baseline) and pd.notna(baseline.iloc[0]) else float("nan")
            if not math.isfinite(baseline_value) or baseline_value <= 0:
                continue
            merged.loc[indices, ratio_column] = pd.to_numeric(group[metric_column], errors="coerce") / baseline_value

    merged["three_space_mean_ratio"] = merged[["Y_ratio", "Z_ratio", "X_ratio"]].mean(axis=1, skipna=True)
    column_order = [
        "case_key",
        "case_label",
        *CASE_COLUMNS,
        "method",
        "method_label",
        "n_test",
        "Y_W",
        "Z_W",
        "X_median_W",
        "Y_ratio",
        "Z_ratio",
        "X_ratio",
        "three_space_mean_ratio",
        "X_top10pct_mean_W",
        "X_top10pct_mean_ratio",
        "X_mean_W",
        "X_q25_W",
        "X_q75_W",
        "X_p90_W",
        "X_max_W",
        "x_sample_count",
        "yz_n_test",
        "yz_aggregation",
        "yz_dataset_key",
    ]
    return merged[column_order]


def sort_case_summary_by_ood(case_summary: pd.DataFrame) -> pd.DataFrame:
    """Sort methods from strongest to weakest OOD evidence within one task."""
    work = case_summary.copy()
    work["_is_random_baseline"] = work["method"].eq("random_cv_baseline")
    work["_ood_sort"] = pd.to_numeric(work["X_ratio"], errors="coerce")
    work["_fallback_sort"] = pd.to_numeric(work["three_space_mean_ratio"], errors="coerce")
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    work["_method_order"] = work["method"].map(method_order).fillna(999)
    work = work.sort_values(
        ["_is_random_baseline", "_ood_sort", "_fallback_sort", "_method_order", "method"],
        ascending=[True, False, False, True, True],
        na_position="last",
        kind="stable",
    )
    return work.drop(columns=["_is_random_baseline", "_ood_sort", "_fallback_sort", "_method_order"]).reset_index(drop=True)


def sort_summary_by_case_and_ood(summary: pd.DataFrame) -> pd.DataFrame:
    parts = [sort_case_summary_by_ood(group) for _, group in summary.groupby("case_key", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else summary.copy()


def ordered_methods_by_ood(case_summary: pd.DataFrame) -> list[str]:
    return sort_case_summary_by_ood(case_summary)["method"].astype(str).tolist()


def global_methods_by_ood(summary: pd.DataFrame) -> list[str]:
    work = summary.copy()
    work["_is_random_baseline"] = work["method"].eq("random_cv_baseline")
    grouped = (
        work.groupby(["method", "_is_random_baseline"], as_index=False)
        .agg(global_x_ratio=("X_ratio", "median"), global_mean_ratio=("three_space_mean_ratio", "median"))
    )
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    grouped["_method_order"] = grouped["method"].map(method_order).fillna(999)
    grouped = grouped.sort_values(
        ["_is_random_baseline", "global_x_ratio", "global_mean_ratio", "_method_order", "method"],
        ascending=[True, False, False, True, True],
        na_position="last",
        kind="stable",
    )
    return grouped["method"].astype(str).tolist()


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def normalized_rank_x(count: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=float)
    if count == 1:
        return np.asarray([0.0], dtype=float)
    return np.linspace(0.0, 1.0, count)


def plain_number_tick(value: float, _pos: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def set_nonnegative_adaptive_y(ax: plt.Axes, values: Iterable[float], label: str) -> None:
    finite = np.asarray([float(value) for value in values if pd.notna(value) and math.isfinite(float(value))])
    max_value = float(np.nanmax(finite)) if finite.size else 1.0
    if finite.size and max_value > 15:
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("linear")
    ax.set_ylabel(label)
    _, upper = ax.get_ylim()
    padded_upper = max(upper * 1.05, max_value * 1.25, 1.0)
    ax.set_ylim(bottom=0.0, top=padded_upper)


def plot_task_ratio(case_summary: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    methods = ordered_methods_by_ood(case_summary)
    work = case_summary.set_index("method").reindex(methods)
    x = np.arange(len(methods), dtype=float)
    width = 0.24
    offsets = {"Y-space": -width, "Z-space": 0.0, "X-space": width}
    columns = {"Y-space": "Y_ratio", "Z-space": "Z_ratio", "X-space": "X_ratio"}

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    all_values: list[float] = []
    for space_name, column in columns.items():
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
        all_values.extend([value for value in values if math.isfinite(value)])
        ax.bar(
            x + offsets[space_name],
            values,
            width=width,
            label=space_name,
            color=SPACE_COLORS[space_name],
            edgecolor="#333333",
            linewidth=0.5,
        )

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="Random-CV baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(method) for method in methods], rotation=25, ha="right")
    set_nonnegative_adaptive_y(ax, all_values, "W ratio vs Random-CV")
    ax.set_title(f"A. Method-level W ratio vs Random-CV\n{work['case_label'].dropna().iloc[0]}")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_base, formats, dpi)


def plot_xspace_boxplot(case_samples: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    methods = [method for method in METHOD_ORDER if method in set(case_samples["method"])]
    values_by_method = [
        pd.to_numeric(case_samples.loc[case_samples["method"] == method, "sample_w_contribution"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for method in methods
    ]
    positions = np.arange(1, len(methods) + 1, dtype=float)
    rng = np.random.default_rng(20260601)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    box = ax.boxplot(
        values_by_method,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111111",
            "markeredgecolor": "#111111",
            "markersize": 4.5,
        },
        medianprops={"color": "#ffffff", "linewidth": 1.5},
        boxprops={"linewidth": 0.9, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 0.9, "color": "#333333"},
        capprops={"linewidth": 0.9, "color": "#333333"},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(method_color(method))
        patch.set_alpha(0.78)

    all_values: list[float] = []
    for pos, method, values in zip(positions, methods, values_by_method):
        if values.size == 0:
            continue
        all_values.extend([float(value) for value in values if math.isfinite(float(value))])
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=10,
            alpha=0.22,
            color="#222222",
            linewidth=0,
            rasterized=True,
        )
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        ax.scatter([pos], [mean_value], marker="D", color="#111111", s=22, zorder=4)
        ax.scatter([pos], [median_value], marker="_", color="#ffffff", s=180, linewidths=2.0, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([method_label(method) for method in methods], rotation=25, ha="right")
    set_nonnegative_adaptive_y(ax, all_values, "X-space sample W contribution")
    ax.set_title(f"X-space sample-level W contribution\n{case_samples['case_label'].dropna().iloc[0]}")
    ax.scatter([], [], marker="D", color="#111111", label="Mean")
    ax.plot([], [], marker="_", color="#333333", linestyle="None", markersize=12, markeredgewidth=2.0, label="Median")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_base, formats, dpi)


def fold_color(fold_id: object) -> str:
    fold = clean_text(fold_id)
    return FOLD_COLORS.get(fold, "#555555")


def fold_sort_key(fold_id: object) -> tuple[int, str]:
    fold = clean_text(fold_id)
    match = re.search(r"(\d+)$", fold)
    return (int(match.group(1)) if match else 999, fold)


def method_linestyle(method: str) -> str:
    if method == "random_cv_baseline":
        return "--"
    if method == "loco":
        return "-"
    return "-"


def finite_sample_values(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["sample_w_contribution"], errors="coerce").dropna()


def plot_sample_w_boxplot(
    ax: plt.Axes,
    case_samples: pd.DataFrame,
    methods: list[str],
    space: str,
    panel_label: str,
    ylabel: str,
) -> None:
    work = case_samples[case_samples["space"] == space].copy()
    positions = np.arange(1, len(methods) + 1, dtype=float)
    rng = np.random.default_rng(20260601)
    plotted_values: list[float] = []

    nonempty_values: list[np.ndarray] = []
    nonempty_positions: list[float] = []
    nonempty_methods: list[str] = []
    for pos, method in zip(positions, methods):
        values = finite_sample_values(work[work["method"] == method]).to_numpy(dtype=float)
        if values.size:
            nonempty_values.append(values)
            nonempty_positions.append(pos)
            nonempty_methods.append(method)
            plotted_values.extend([float(value) for value in values if math.isfinite(float(value))])

    if nonempty_values:
        box = ax.boxplot(
            nonempty_values,
            positions=nonempty_positions,
            patch_artist=True,
            widths=0.55,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111111",
                "markeredgecolor": "#111111",
                "markersize": 4.0,
            },
            medianprops={"color": "#ffffff", "linewidth": 1.35},
            boxprops={"linewidth": 0.85, "edgecolor": "#333333"},
            whiskerprops={"linewidth": 0.85, "color": "#333333"},
            capprops={"linewidth": 0.85, "color": "#333333"},
            flierprops={"marker": "", "markersize": 0},
        )
        for patch, method in zip(box["boxes"], nonempty_methods):
            patch.set_facecolor(method_color(method))
            patch.set_alpha(0.74)

    for pos, method in zip(positions, methods):
        method_rows = work[work["method"] == method].copy()
        if method_rows.empty:
            continue
        if method in FOLD_AWARE_METHODS:
            fold_ids = sorted(method_rows["fold_id"].dropna().astype(str).unique(), key=fold_sort_key)
            for fold_id in fold_ids:
                fold_rows = method_rows[method_rows["fold_id"].astype(str) == fold_id]
                values = finite_sample_values(fold_rows).to_numpy(dtype=float)
                if values.size == 0:
                    continue
                draw_values = values if len(values) <= 180 else rng.choice(values, size=180, replace=False)
                jitter = rng.uniform(-0.18, 0.18, size=len(draw_values))
                ax.scatter(
                    np.full(len(draw_values), pos) + jitter,
                    draw_values,
                    s=8,
                    alpha=0.34,
                    color=fold_color(fold_id),
                    linewidth=0,
                    rasterized=True,
                )
        else:
            values = finite_sample_values(method_rows).to_numpy(dtype=float)
            if values.size == 0:
                continue
            draw_values = values if len(values) <= 220 else rng.choice(values, size=220, replace=False)
            jitter = rng.uniform(-0.16, 0.16, size=len(draw_values))
            ax.scatter(
                np.full(len(draw_values), pos) + jitter,
                draw_values,
                s=8,
                alpha=0.24,
                color=method_color(method),
                linewidth=0,
                rasterized=True,
            )

    if not plotted_values:
        statuses = sorted(set(work.get("status", pd.Series(dtype=str)).dropna().astype(str)))
        message = "Missing sample-level values"
        if statuses:
            message += "\n" + " | ".join(statuses[:3])
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#555555")

    ax.set_xticks(positions)
    ax.set_xticklabels([method_label(method) for method in methods], rotation=28, ha="right")
    set_nonnegative_adaptive_y(ax, plotted_values, ylabel)
    ax.set_title(f"{panel_label}. {space} sample W distribution", loc="left", fontweight="bold")
    style_axes(ax)


def plot_sample_w_ranked_curve(
    ax: plt.Axes,
    case_samples: pd.DataFrame,
    methods: list[str],
    space: str,
    panel_label: str,
    ylabel: str,
) -> None:
    work = case_samples[case_samples["space"] == space].copy()
    plotted_values: list[float] = []
    legend_count = 0

    for method in methods:
        method_rows = work[work["method"] == method].copy()
        if method_rows.empty:
            continue
        if method in FOLD_AWARE_METHODS:
            fold_ids = sorted(method_rows["fold_id"].dropna().astype(str).unique(), key=fold_sort_key)
            for fold_id in fold_ids:
                values = finite_sample_values(method_rows[method_rows["fold_id"].astype(str) == fold_id])
                if values.empty:
                    continue
                values = values.sort_values(ascending=False).reset_index(drop=True)
                plotted_values.extend([float(value) for value in values if math.isfinite(float(value))])
                rank_x = normalized_rank_x(len(values))
                ax.plot(
                    rank_x,
                    values.to_numpy(dtype=float),
                    color=fold_color(fold_id),
                    linestyle=method_linestyle(method),
                    linewidth=1.35,
                    alpha=0.9,
                    label=f"{method_label(method)} {fold_id}",
                )
                legend_count += 1
        else:
            values = finite_sample_values(method_rows)
            if values.empty:
                continue
            values = values.sort_values(ascending=False).reset_index(drop=True)
            plotted_values.extend([float(value) for value in values if math.isfinite(float(value))])
            rank_x = normalized_rank_x(len(values))
            ax.plot(
                rank_x,
                values.to_numpy(dtype=float),
                color=method_color(method),
                linestyle="-",
                linewidth=1.8,
                alpha=0.94,
                label=method_label(method),
            )
            legend_count += 1

    if not plotted_values:
        statuses = sorted(set(work.get("status", pd.Series(dtype=str)).dropna().astype(str)))
        message = "Missing sample-level values"
        if statuses:
            message += "\n" + " | ".join(statuses[:3])
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#555555")

    ax.set_xlabel("Normalized rank within method/fold")
    set_nonnegative_adaptive_y(ax, plotted_values, ylabel)
    ax.set_title(f"{panel_label}. {space} ranked sample W curve", loc="left", fontweight="bold")
    if legend_count:
        ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper right")
    style_axes(ax)


def plot_aggregate_ranked_curve(
    ax: plt.Axes,
    case_samples: pd.DataFrame,
    methods: list[str],
    space: str,
    panel_label: str,
    ylabel: str,
    show_legend: bool = False,
) -> None:
    work = case_samples[case_samples["space"] == space].copy()
    plotted_values: list[float] = []
    plotted_methods: list[str] = []

    for method in methods:
        method_rows = work[work["method"] == method].copy()
        if method_rows.empty:
            continue
        values = finite_sample_values(method_rows)
        if values.empty:
            continue
        values = values.sort_values(ascending=False).reset_index(drop=True)
        plotted_values.extend([float(value) for value in values if math.isfinite(float(value))])
        plotted_methods.append(method)
        rank_x = normalized_rank_x(len(values))
        ax.plot(
            rank_x,
            values.to_numpy(dtype=float),
            color=method_curve_color(method),
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            label=method_label(method),
        )

    if not plotted_values:
        statuses = sorted(set(work.get("status", pd.Series(dtype=str)).dropna().astype(str)))
        message = "Missing sample-level values"
        if statuses:
            message += "\n" + " | ".join(statuses[:3])
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#555555")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized rank within method")
    set_nonnegative_adaptive_y(ax, plotted_values, ylabel)
    ax.set_title(f"{panel_label}. {space} aggregate ranked sample W curve", loc="left", fontweight="bold")
    if show_legend and plotted_methods:
        ax.legend(frameon=False, fontsize=7.1, ncol=2, loc="upper right")
    style_axes(ax)


def add_fold_curve_legend(ax: plt.Axes, fold_ids: list[str]) -> None:
    fig = ax.figure
    method_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.8, label="LOCO"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.8, label="Random-CV"),
    ]
    fig.legend(
        handles=method_handles,
        title="Method",
        frameon=False,
        fontsize=7.0,
        title_fontsize=7.2,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.875),
    )

    fold_handles = [
        Line2D([0], [0], color=fold_color(fold_id), linestyle="-", linewidth=2.0, label=fold_id)
        for fold_id in fold_ids
    ]
    if fold_handles:
        fig.legend(
            handles=fold_handles,
            title="Fold",
            frameon=False,
            fontsize=7.0,
            title_fontsize=7.2,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.79),
        )


def plot_fold_ranked_curve(
    ax: plt.Axes,
    case_samples: pd.DataFrame,
    space: str,
    panel_label: str,
    ylabel: str,
    show_legend: bool = False,
) -> None:
    work = case_samples[case_samples["space"] == space].copy()
    fold_methods = [method for method in ["loco", "random_cv_baseline"] if method in set(work["method"].astype(str))]
    fold_ids = sorted(
        work.loc[work["method"].isin(fold_methods), "fold_id"].dropna().astype(str).unique(),
        key=fold_sort_key,
    )
    plotted_values: list[float] = []
    plotted_line_count = 0

    for fold_id in fold_ids:
        for method in fold_methods:
            method_rows = work[(work["method"] == method) & (work["fold_id"].astype(str) == fold_id)]
            values = finite_sample_values(method_rows)
            if values.empty:
                continue
            values = values.sort_values(ascending=False).reset_index(drop=True)
            plotted_values.extend([float(value) for value in values if math.isfinite(float(value))])
            rank_x = normalized_rank_x(len(values))
            ax.plot(
                rank_x,
                values.to_numpy(dtype=float),
                color=fold_color(fold_id),
                linestyle=method_linestyle(method),
                linewidth=1.45 if method == "random_cv_baseline" else 1.7,
                alpha=0.9,
            )
            plotted_line_count += 1

    if not plotted_values:
        statuses = sorted(set(work.get("status", pd.Series(dtype=str)).dropna().astype(str)))
        message = "Missing LOCO/Random-CV fold-level values"
        if statuses:
            message += "\n" + " | ".join(statuses[:3])
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#555555")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized rank within method/fold")
    set_nonnegative_adaptive_y(ax, plotted_values, ylabel)
    ax.set_title(f"{panel_label}. {space} LOCO/Random-CV fold ranked W curve", loc="left", fontweight="bold")
    if show_legend and plotted_line_count:
        add_fold_curve_legend(ax, fold_ids)
    style_axes(ax)


def plot_task_curves(
    case_summary: pd.DataFrame,
    case_samples: pd.DataFrame,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    methods = ordered_methods_by_ood(case_summary)
    case_label = case_summary["case_label"].dropna().iloc[0]

    fig, axes = plt.subplots(3, 3, figsize=(21.5, 12.2))
    fig.suptitle(
        f"Sample-level W contribution distributions and ranked curves\n{case_label}",
        fontsize=15,
        fontweight="bold",
        y=0.982,
    )

    panel_specs = [
        ("X-space", "B", "C", "D", "X-space sample W"),
        ("Y-space", "E", "F", "G", "Y-space sample W"),
        ("Z-space", "H", "I", "J", "Z-space sample W"),
    ]
    for row, (space, box_label, aggregate_label, fold_label, ylabel) in enumerate(panel_specs):
        plot_sample_w_boxplot(
            axes[row, 0],
            case_samples,
            methods,
            space,
            box_label,
            ylabel,
        )
        plot_aggregate_ranked_curve(
            axes[row, 1],
            case_samples,
            methods,
            space,
            aggregate_label,
            ylabel,
            show_legend=row == 0,
        )
        plot_fold_ranked_curve(
            axes[row, 2],
            case_samples,
            space,
            fold_label,
            ylabel,
            show_legend=row == 0,
        )

    fig.text(
        0.01,
        0.01,
        "Left panels show full sample W distributions. Middle panels collapse all finite sample contributions by method. Right panels use fold color plus LOCO solid / Random-CV dashed line style.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.025, 0.9, 0.965))

    save_figure(fig, output_base, formats, dpi)


def plot_x_ratio_heatmap(summary: pd.DataFrame, output_base: Path, formats: Iterable[str], dpi: int) -> None:
    pivot = summary.pivot_table(index="case_label", columns="method", values="X_ratio", aggfunc="first")
    pivot = pivot.reindex(columns=global_methods_by_ood(summary))
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1.0)

    fig_height = max(5.2, 0.46 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#f2f2f2")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([method_label(method) for method in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("X-space median sample W ratio vs Random-CV")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "-" if not math.isfinite(value) else f"{value:.2f}"
            text_color = "#ffffff" if math.isfinite(value) and value >= 0.62 * vmax else "#111111"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("X ratio")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, output_base, formats, dpi)


def prepare_sample_values(sample_values: pd.DataFrame) -> pd.DataFrame:
    samples = add_case_fields(sample_values)
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    samples["_method_order"] = samples["method"].map(method_order).fillna(999)
    sort_cols = ["case_key", "_method_order", "method", "sample_w_rank_desc"]
    return samples.sort_values(sort_cols, kind="stable").drop(columns="_method_order").reset_index(drop=True)


def build_top_samples(sample_values: pd.DataFrame, top_n: int) -> pd.DataFrame:
    samples = prepare_sample_values(sample_values)
    rows: list[pd.DataFrame] = []
    for (_, method), group in samples.groupby(["case_key", "method"], sort=False):
        rows.append(group.sort_values("sample_w_contribution", ascending=False, kind="stable").head(top_n))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "case_key",
        "case_label",
        *CASE_COLUMNS,
        "method",
        "fold_id",
        "ID",
        "target_value",
        "sample_w_contribution",
        "sample_w_rank_desc",
        "ood_score",
        "ood_percentile_vs_train",
    ]
    return result[[column for column in keep_cols if column in result.columns]]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def top10_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if finite.empty:
        return float("nan")
    top_n = max(1, int(math.ceil(len(finite) * 0.1)))
    return float(finite.tail(top_n).mean())


def one_dimensional_w1_sample_contributions(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = np.asarray(train_values, dtype=float)
    test = np.asarray(test_values, dtype=float)
    if len(train) == 0 or len(test) == 0 or not np.isfinite(train).all() or not np.isfinite(test).all():
        return float("nan"), np.full(len(test), np.nan), np.full(len(test), np.nan)

    train_order = np.argsort(train, kind="mergesort")
    test_order = np.argsort(test, kind="mergesort")
    train_sorted = train[train_order]
    test_sorted = test[test_order]
    train_mass = 1.0 / len(train_sorted)
    test_mass = 1.0 / len(test_sorted)
    train_remaining = train_mass
    test_remaining = test_mass
    i = 0
    j = 0
    test_mass_contrib = np.zeros(len(test_sorted), dtype=float)
    total = 0.0

    while i < len(train_sorted) and j < len(test_sorted):
        moved = min(train_remaining, test_remaining)
        cost = moved * abs(float(train_sorted[i]) - float(test_sorted[j]))
        test_mass_contrib[test_order[j]] += cost
        total += cost
        train_remaining -= moved
        test_remaining -= moved
        if train_remaining <= 1e-15:
            i += 1
            train_remaining = train_mass
        if test_remaining <= 1e-15:
            j += 1
            test_remaining = test_mass

    sample_scores = test_mass_contrib / test_mass
    return float(total), sample_scores, test_mass_contrib


def nd_w1_sample_contributions(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    train = np.asarray(train_values, dtype=float)
    test = np.asarray(test_values, dtype=float)
    if (
        train.ndim != 2
        or test.ndim != 2
        or len(train) == 0
        or len(test) == 0
        or train.shape[1] != test.shape[1]
        or not np.isfinite(train).all()
        or not np.isfinite(test).all()
    ):
        return float("nan"), np.full(len(test), np.nan), np.full(len(test), np.nan)

    train_count, test_count = train.shape[0], test.shape[0]
    costs = distance_matrix(train, test, p=2).ravel()
    row_constraints = sparse.block_diag((np.ones((1, test_count)),) * train_count, format="csr")
    col_constraints = sparse.hstack((sparse.eye(test_count, format="csr"),) * train_count, format="csr")
    constraints = sparse.vstack((row_constraints, col_constraints), format="csr")
    masses = np.concatenate(
        [np.full(train_count, 1.0 / train_count), np.full(test_count, 1.0 / test_count)]
    )
    result = linprog(costs, A_eq=constraints, b_eq=masses, bounds=(0, None), method="highs")
    if not result.success:
        return float("nan"), np.full(test_count, np.nan), np.full(test_count, np.nan)

    plan = np.asarray(result.x, dtype=float).reshape(train_count, test_count)
    cost_matrix = costs.reshape(train_count, test_count)
    test_mass_contrib = (plan * cost_matrix).sum(axis=0)
    test_mass_contrib[np.abs(test_mass_contrib) < 1e-15] = 0.0
    sample_scores = test_mass_contrib / (1.0 / test_count)
    return float(result.fun), sample_scores, test_mass_contrib


def standardized_target_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    train = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    test = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float)
    train_mean = float(np.nanmean(train))
    train_std = float(pd.Series(train).std(ddof=1))
    if not math.isfinite(train_std) or train_std <= 0:
        return np.full(len(train), np.nan), np.full(len(test), np.nan)
    return (train - train_mean) / train_std, (test - train_mean) / train_std


def standardized_z_arrays(train_z: pd.DataFrame, test_z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = train_z[["x", "y"]].astype(float).to_numpy()
    test = test_z[["x", "y"]].astype(float).to_numpy()
    means = np.nanmean(train, axis=0)
    stds = np.asarray([pd.Series(train[:, idx]).std(ddof=1) for idx in range(train.shape[1])], dtype=float)
    if not np.isfinite(stds).all() or np.any(stds <= 0):
        return np.full_like(train, np.nan, dtype=float), np.full_like(test, np.nan, dtype=float)
    return (train - means) / stds, (test - means) / stds


def yz_dataset_key_for_case(case_key: str) -> str | None:
    for dataset_key, case_values in YZ_DATASET_TO_CASE.items():
        if case_key_from_values(*case_values) == case_key:
            return dataset_key
    return None


def load_embedding_backgrounds(embedding_data_dir: Path) -> dict[str, pd.DataFrame]:
    backgrounds: dict[str, pd.DataFrame] = {}
    for dataset_key, case_values in YZ_DATASET_TO_CASE.items():
        path = embedding_data_dir / f"{dataset_key}__all_panel_test_points.csv"
        if not path.exists():
            continue
        frame = read_csv(path)
        if not {"x", "y"}.issubset(frame.columns) or not ({"ID", "__source_index__"} & set(frame.columns)):
            continue
        columns = [column for column in ["ID", "__source_index__", "x", "y"] if column in frame.columns]
        background = frame[columns].copy()
        if "ID" in background.columns:
            background["__id_key__"] = pd.to_numeric(background["ID"], errors="coerce")
        if "__source_index__" in background.columns:
            background["__source_index_key__"] = pd.to_numeric(background["__source_index__"], errors="coerce")
        background["x"] = pd.to_numeric(background["x"], errors="coerce")
        background["y"] = pd.to_numeric(background["y"], errors="coerce")
        if "__id_key__" in background.columns and background["__id_key__"].notna().any():
            background = (
                background.dropna(subset=["__id_key__", "x", "y"])
                .sort_values("__id_key__")
                .drop_duplicates("__id_key__", keep="first")
            )
        else:
            background = (
                background.dropna(subset=["__source_index_key__", "x", "y"])
                .sort_values("__source_index_key__")
                .drop_duplicates("__source_index_key__", keep="first")
            )
        keep_columns = [column for column in ["__id_key__", "__source_index_key__", "x", "y"] if column in background.columns]
        backgrounds[case_key_from_values(*case_values)] = background[keep_columns].copy()
    return backgrounds


def base_sample_rows_from_test(split_row: pd.Series, test_df: pd.DataFrame, space: str) -> pd.DataFrame:
    row_count = len(test_df)
    result = pd.DataFrame(
        {
            "case_key": case_key_from_values(split_row["alloy_family"], split_row["dataset_name"], split_row["property"]),
            "case_label": case_label_from_values(split_row["alloy_family"], split_row["dataset_name"], split_row["property"]),
            "alloy_family": split_row["alloy_family"],
            "dataset_name": split_row["dataset_name"],
            "property": split_row["property"],
            "method": split_row["split_strategy"],
            "split_id": split_row["split_id"],
            "fold_id": clean_text(split_row.get("fold_id", "")),
            "source_split_dir": split_row["source_split_dir"],
            "__row_id__": test_df["__row_id__"].to_numpy() if "__row_id__" in test_df.columns else np.arange(row_count),
            "__source_index__": test_df["__source_index__"].to_numpy()
            if "__source_index__" in test_df.columns
            else np.full(row_count, np.nan),
            "target_col": split_row["target_col"],
            "target_value": pd.to_numeric(test_df[split_row["target_col"]], errors="coerce").to_numpy()
            if split_row["target_col"] in test_df.columns
            else np.nan,
            "space": space,
        }
    )
    if "ID" in test_df.columns:
        result["ID"] = test_df["ID"].to_numpy()
    else:
        result["ID"] = np.nan
    return result


def x_space_long_samples(sample_values: pd.DataFrame, split_summary: pd.DataFrame) -> pd.DataFrame:
    x = add_case_fields(sample_values).copy()
    split_w = split_summary[
        ["source_split_dir", "sliced_wasserstein"]
    ].drop_duplicates("source_split_dir", keep="first")
    x = x.merge(split_w, on="source_split_dir", how="left")
    x["space"] = "X-space"
    x["space_label"] = "X-space"
    x["split_w"] = pd.to_numeric(x["sliced_wasserstein"], errors="coerce")
    x["sample_w_mass_contribution"] = pd.to_numeric(x.get("sample_w_mass_contribution"), errors="coerce")
    x["status"] = "ok"
    keep_cols = [
        "case_key",
        "case_label",
        *CASE_COLUMNS,
        "method",
        "split_id",
        "fold_id",
        "source_split_dir",
        "__row_id__",
        "__source_index__",
        "ID",
        "target_col",
        "target_value",
        "space",
        "space_label",
        "sample_w_contribution",
        "sample_w_mass_contribution",
        "split_w",
        "status",
        "ood_score",
        "ood_percentile_vs_train",
    ]
    return x[[column for column in keep_cols if column in x.columns]].copy()


def yz_space_long_samples(split_summary: pd.DataFrame, embedding_data_dir: Path) -> pd.DataFrame:
    backgrounds = load_embedding_backgrounds(embedding_data_dir)
    rows: list[pd.DataFrame] = []
    for _, split_row in split_summary.iterrows():
        train_path = Path(str(split_row["train_file"]))
        test_path = Path(str(split_row["test_file"]))
        train_df = read_csv(train_path)
        test_df = read_csv(test_path)

        y_base = base_sample_rows_from_test(split_row, test_df, "Y-space")
        y_train, y_test = standardized_target_arrays(train_df, test_df, str(split_row["target_col"]))
        y_split_w, y_scores, y_mass = one_dimensional_w1_sample_contributions(y_train, y_test)
        y_base["space_label"] = "Y-space"
        y_base["sample_w_contribution"] = y_scores
        y_base["sample_w_mass_contribution"] = y_mass
        y_base["split_w"] = y_split_w
        y_base["status"] = "ok" if math.isfinite(y_split_w) else "invalid_target_distribution"
        rows.append(y_base)

        z_base = base_sample_rows_from_test(split_row, test_df, "Z-space")
        z_base["space_label"] = "Z-space"
        case_key = str(z_base["case_key"].iloc[0]) if not z_base.empty else ""
        background = backgrounds.get(case_key)
        if background is None:
            z_base["sample_w_contribution"] = np.nan
            z_base["sample_w_mass_contribution"] = np.nan
            z_base["split_w"] = np.nan
            z_base["status"] = "missing_embedding_file"
            rows.append(z_base)
            continue

        if "__id_key__" in background.columns and "ID" in train_df.columns and "ID" in test_df.columns:
            train_keys = pd.DataFrame({"__id_key__": pd.to_numeric(train_df["ID"], errors="coerce")})
            test_keys = pd.DataFrame({"__id_key__": pd.to_numeric(test_df["ID"], errors="coerce")})
            train_z = train_keys.merge(background, on="__id_key__", how="left", sort=False)
            test_z = test_keys.merge(background, on="__id_key__", how="left", sort=False)
        elif (
            "__source_index_key__" in background.columns
            and "__source_index__" in train_df.columns
            and "__source_index__" in test_df.columns
        ):
            train_keys = pd.DataFrame(
                {"__source_index_key__": pd.to_numeric(train_df["__source_index__"], errors="coerce")}
            )
            test_keys = pd.DataFrame(
                {"__source_index_key__": pd.to_numeric(test_df["__source_index__"], errors="coerce")}
            )
            train_z = train_keys.merge(background, on="__source_index_key__", how="left", sort=False)
            test_z = test_keys.merge(background, on="__source_index_key__", how="left", sort=False)
        else:
            z_base["sample_w_contribution"] = np.nan
            z_base["sample_w_mass_contribution"] = np.nan
            z_base["split_w"] = np.nan
            z_base["status"] = "missing_embedding_join_key"
            rows.append(z_base)
            continue
        if train_z[["x", "y"]].isna().any().any() or test_z[["x", "y"]].isna().any().any():
            z_base["sample_w_contribution"] = np.nan
            z_base["sample_w_mass_contribution"] = np.nan
            z_base["split_w"] = np.nan
            z_base["status"] = "missing_embedding_rows"
            rows.append(z_base)
            continue

        z_train, z_test = standardized_z_arrays(train_z, test_z)
        z_split_w, z_scores, z_mass = nd_w1_sample_contributions(z_train, z_test)
        z_base["sample_w_contribution"] = z_scores
        z_base["sample_w_mass_contribution"] = z_mass
        z_base["split_w"] = z_split_w
        z_base["status"] = "ok" if math.isfinite(z_split_w) else "ot_failed"
        rows.append(z_base)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_three_space_sample_values(
    sample_values: pd.DataFrame,
    split_summary_path: Path,
    embedding_data_dir: Path,
) -> pd.DataFrame:
    if not split_summary_path.exists():
        raise FileNotFoundError(f"Missing split summary for Y/Z sample contributions: {split_summary_path}")
    split_summary = read_csv(split_summary_path)
    required = [
        *CASE_COLUMNS,
        "target_col",
        "split_strategy",
        "split_id",
        "fold_id",
        "source_split_dir",
        "train_file",
        "test_file",
        "sliced_wasserstein",
    ]
    missing = [column for column in required if column not in split_summary.columns]
    if missing:
        raise ValueError(f"{split_summary_path} is missing required columns: {missing}")

    x_rows = x_space_long_samples(sample_values, split_summary)
    yz_rows = yz_space_long_samples(split_summary, embedding_data_dir)
    combined = pd.concat([x_rows, yz_rows], ignore_index=True, sort=False)
    combined["sample_w_contribution"] = pd.to_numeric(combined["sample_w_contribution"], errors="coerce")
    combined["sample_w_mass_contribution"] = pd.to_numeric(combined["sample_w_mass_contribution"], errors="coerce")
    combined["split_w"] = pd.to_numeric(combined["split_w"], errors="coerce")
    combined["fold_id"] = combined["fold_id"].fillna("").astype(str)
    combined["space_w_rank_desc"] = (
        combined.groupby(["case_key", "method", "space"], dropna=False)["sample_w_contribution"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    space_order = {"X-space": 0, "Y-space": 1, "Z-space": 2}
    combined["_method_order"] = combined["method"].map(method_order).fillna(999)
    combined["_space_order"] = combined["space"].map(space_order).fillna(999)
    combined = combined.sort_values(
        ["case_key", "_space_order", "_method_order", "method", "fold_id", "space_w_rank_desc"],
        kind="stable",
    ).drop(columns=["_method_order", "_space_order"])
    return combined.reset_index(drop=True)


def summarize_three_space_sample_values(sample_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["case_key", "case_label", *CASE_COLUMNS, "method", "space"]
    for keys, group in sample_values.groupby(group_cols, dropna=False, sort=True):
        values = pd.to_numeric(group["sample_w_contribution"], errors="coerce")
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "method_label": method_label(str(row["method"])),
                "space_label": str(row["space"]),
                "row_count": int(len(group)),
                "sample_count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "fold_count": int(group.loc[group["fold_id"].astype(str).ne(""), "fold_id"].nunique()),
                "sample_w_mean": float(values.mean()),
                "sample_w_median": float(values.median()),
                "sample_w_q25": float(values.quantile(0.25)),
                "sample_w_q75": float(values.quantile(0.75)),
                "sample_w_p90": float(values.quantile(0.90)),
                "sample_w_max": float(values.max()),
                "sample_w_top10pct_mean": top10_mean(values),
                "status": " | ".join(sorted(set(group["status"].dropna().astype(str)))) if "status" in group else "",
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    method_order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    space_order = {"X-space": 0, "Y-space": 1, "Z-space": 2}
    result["_method_order"] = result["method"].map(method_order).fillna(999)
    result["_space_order"] = result["space"].map(space_order).fillna(999)
    return (
        result.sort_values(["case_key", "_space_order", "_method_order", "method"], kind="stable")
        .drop(columns=["_method_order", "_space_order"])
        .reset_index(drop=True)
    )


def format_number(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except Exception:
        return ""
    if not math.isfinite(numeric):
        return ""
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    return f"{numeric:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    work = frame[columns].copy()
    if max_rows is not None:
        work = work.head(max_rows)
    headers = list(work.columns)
    rows = []
    for _, row in work.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                cells.append(format_number(value))
            else:
                cells.append("" if pd.isna(value) else str(value))
        rows.append(cells)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_report_legacy_mojibake(
    summary: pd.DataFrame,
    top_samples: pd.DataFrame,
    output_root: Path,
    x_report_root: Path,
    yz_summary_path: Path,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing_yz = summary[summary[["Y_W", "Z_W"]].isna().all(axis=1)]
    missing_cases = sorted(missing_yz["case_label"].dropna().unique())
    top_x = (
        summary[summary["method"] != "random_cv_baseline"]
        .sort_values("X_ratio", ascending=False, na_position="last", kind="stable")
        .head(15)
    )
    top_sample_rows = (
        top_samples.sort_values("sample_w_contribution", ascending=False, na_position="last", kind="stable").head(20)
        if not top_samples.empty
        else pd.DataFrame()
    )

    lines: list[str] = [
        "# 三空间 W-OOD 报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 输入数据",
        "",
        f"- X-space 样本级 W：`{x_report_root / 'all_sample_w_values.csv'}`",
        f"- X-space method 汇总：`{x_report_root / 'case_method_w_summary.csv'}`",
        f"- Y/Z-space 旧 Wasserstein 汇总：`{yz_summary_path}`",
        "",
        "## 指标定义",
        "",
        "- `Y-space`：目标值空间，使用 `target_w1_std`，对应标准化后的 `W1(y*)`。",
        "- `Z-space`：二维 embedding 空间，使用 `xy_w1_nd_std`，对应标准化后的 `W1(z*)`。",
        "- `X-space`：成分 + 工艺输入空间，使用 `sample_w_contribution` 的 method 中位数作为 method 级 W。",
        "",
        "## 计算公式",
        "",
        "### 1. 训练集标准化",
        "",
        "所有空间都只使用训练集统计量做标准化。对任意特征或坐标 `v`：",
        "",
        "```text",
        "v* = (v - mean_train(v)) / std_train(v)",
        "```",
        "",
        "其中 `mean_train` 和 `std_train` 只由当前 split/fold 的训练集计算，测试集只做 transform。",
        "",
        "### 2. Y-space: W1(y*)",
        "",
        "Y-space 使用目标值，例如 UTS、YS 或 El。训练集和测试集目标值标准化后分别记为：",
        "",
        "```text",
        "A_y = {y_i* | i in train}",
        "B_y = {y_j* | j in test}",
        "```",
        "",
        "一维 Wasserstein-1 距离为：",
        "",
        "```text",
        "Y_W = W1(A_y, B_y)",
        "```",
        "",
        "它解释的是目标值分布偏移，也就是 label/target distribution shift。",
        "",
        "### 3. Z-space: W1(z*)",
        "",
        "Z-space 使用旧结果中的二维 embedding 坐标 `(x, y)`。标准化后：",
        "",
        "```text",
        "z_i* = [(x_i - mean_train(x)) / std_train(x),",
        "        (y_i - mean_train(y)) / std_train(y)]",
        "```",
        "",
        "二维 Wasserstein 距离为：",
        "",
        "```text",
        "Z_W = W1_nd({z_i* | i in train}, {z_j* | j in test})",
        "```",
        "",
        "它解释的是二维表征空间分布偏移。",
        "",
        "### 4. X-space: sample-level sliced W contribution",
        "",
        "X-space uses composition + processing input features. Zeros represent physical absence. Rare NaNs/blanks are filled with 0, then standard scaler is fitted on the training split only:",
        "",
        "```text",
        "z_train_i = standardized X vector of train sample i",
        "z_test_j  = standardized X vector of test sample j",
        "```",
        "",
        "对每个随机单位方向 `theta_r`，把多维 X 向量投影到一维：",
        "",
        "```text",
        "a_i^(r) = theta_r^T z_train_i",
        "b_j^(r) = theta_r^T z_test_j",
        "```",
        "",
        "在该一维方向上，把训练集和测试集视为经验分布：",
        "",
        "```text",
        "train sample mass = 1 / N_train",
        "test sample mass  = 1 / N_test",
        "```",
        "",
        "排序后做一维 Wasserstein-1 最优匹配。测试样本 `j` 在第 `r` 个方向上的质量加权贡献为：",
        "",
        "```text",
        "c_j^(r) = sum_over_matches moved_mass * |a_i^(r) - b_j^(r)|",
        "```",
        "",
        "把质量贡献换算成与 W1 同单位的样本级贡献：",
        "",
        "```text",
        "w_j^(r) = c_j^(r) / (1 / N_test)",
        "```",
        "",
        "对所有投影方向取平均，得到最终样本级 X-space W：",
        "",
        "```text",
        "sample_w_contribution_j = mean_r w_j^(r)",
        "```",
        "",
        "该值不是严格意义的“单样本 Wasserstein distance”，而是测试样本对 sliced Wasserstein 分布偏移的样本级贡献。它满足一致性关系：",
        "",
        "```text",
        "mean_j(sample_w_contribution_j) ≈ split-level sliced Wasserstein",
        "```",
        "",
        "### 5. Method 级 X-space W 和 ratio",
        "",
        "每个 task × method 的 X-space 主 W 值使用样本级 W 的中位数：",
        "",
        "```text",
        "X_median_W_method = median_j(sample_w_contribution_j)",
        "```",
        "",
        "高尾部辅助证据使用 top 10% 样本的均值：",
        "",
        "```text",
        "X_top10pct_mean_W_method = mean(top 10% sample_w_contribution_j)",
        "```",
        "",
        "三空间 ratio 都以 `random_cv_baseline` 为基准：",
        "",
        "```text",
        "Y_ratio = W1(y*)_method / W1(y*)_random_cv",
        "Z_ratio = W1(z*)_method / W1(z*)_random_cv",
        "X_ratio = median(sample_w_contribution)_method / median(sample_w_contribution)_random_cv",
        "```",
        "",
        "注意：`sample_w_contribution` 是测试样本对 sliced Wasserstein 分布偏移的样本级贡献，不写作单样本 Wasserstein distance。",
        "",
        "## 主要结果",
        "",
        "按 X-space ratio 排序的高偏移 method：",
        "",
        markdown_table(
            top_x,
            ["case_label", "method_label", "Y_ratio", "Z_ratio", "X_ratio", "X_median_W", "X_top10pct_mean_W"],
            max_rows=15,
        ),
        "",
        "X-space top-W 样本（全局前 20）：",
        "",
    ]
    if top_sample_rows.empty:
        lines.append("无可用 top 样本。")
    else:
        sample_cols = [
            column
            for column in ["case_label", "method", "ID", "target_value", "sample_w_contribution", "sample_w_rank_desc"]
            if column in top_sample_rows.columns
        ]
        lines.append(markdown_table(top_sample_rows, sample_cols, max_rows=20))
    lines.extend(
        [
            "",
            "## 缺失说明",
            "",
        ]
    )
    if missing_cases:
        lines.append("以下 case 在旧 Y/Z-space 汇总中缺失，因此对应 `Y_W`、`Z_W` 和 ratio 标记为空：")
        lines.extend(f"- {case}" for case in missing_cases)
    else:
        lines.append("所有 X-space case 均匹配到旧 Y/Z-space 汇总。")

    lines.extend(
        [
            "",
            "## 分 case 图表",
            "",
        ]
    )
    for case_key, case_summary in summary.groupby("case_key", sort=False):
        case_label = case_summary["case_label"].iloc[0]
        case_dir = output_root / "cases" / case_key
        ratio_png = case_dir / "task_w_ratio.png"
        curves_png = case_dir / "task_w_curves.png"
        ratio_rel = ratio_png.relative_to(output_root).as_posix()
        curves_rel = curves_png.relative_to(output_root).as_posix()
        lines.extend(
            [
                f"### {case_label}",
                "",
                f"![{case_label} task W ratio]({ratio_rel})",
                "",
                f"![{case_label} task W curves]({curves_rel})",
                "",
                markdown_table(
                    case_summary,
                    ["method_label", "Y_ratio", "Z_ratio", "X_ratio", "X_median_W", "X_top10pct_mean_W"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 解释建议",
            "",
            "- 主结论优先看 `X_ratio` 和 X-space 样本级 W 箱线图，因为它们直接对应成分 + 工艺输入分布偏移。",
            "- `Y_ratio` 说明目标值分布是否偏移，适合作为 label shift 辅助证据。",
            "- `Z_ratio` 说明二维表征空间是否偏移，适合作为可视化和表征空间辅助证据。",
            "- 如果某个 method 的 `X_ratio` 明显大于 1，且箱线图整体上移或高尾部更长，可以支持该测试集比 random-CV 更 OOD。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    summary: pd.DataFrame,
    top_samples: pd.DataFrame,
    output_root: Path,
    x_report_root: Path,
    yz_summary_path: Path,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing_yz = summary[summary[["Y_W", "Z_W"]].isna().all(axis=1)]
    missing_cases = sorted(missing_yz["case_label"].dropna().unique())
    top_x = (
        summary[summary["method"] != "random_cv_baseline"]
        .sort_values("X_ratio", ascending=False, na_position="last", kind="stable")
        .head(15)
    )
    top_sample_rows = (
        top_samples.sort_values("sample_w_contribution", ascending=False, na_position="last", kind="stable").head(20)
        if not top_samples.empty
        else pd.DataFrame()
    )

    lines: list[str] = [
        "# 三空间 W-OOD 报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 输入与输出",
        "",
        f"- X-space 样本级贡献输入：`{x_report_root / 'all_sample_w_values.csv'}`",
        f"- X-space method 汇总输入：`{x_report_root / 'case_method_w_summary.csv'}`",
        f"- 旧 Y/Z-space method 汇总输入：`{yz_summary_path}`",
        f"- 三空间样本级 long table：`{output_root / 'three_space_sample_w_values.csv'}`",
        f"- 三空间样本级 summary：`{output_root / 'three_space_sample_w_summary.csv'}`",
        f"- 三空间 ratio summary：`{output_root / 'three_space_w_ratio_summary.csv'}`",
        "",
        "## 指标含义",
        "",
        "- `Y-space`：目标值空间，例如 UTS、YS、El；用于说明目标分布偏移。",
        "- `Z-space`：二维 embedding 空间，使用旧 embedding 文件中的 `x, y`；用于说明表征空间分布偏移。",
        "- `X-space`：成分 + 工艺输入空间；这是输入 OOD 的主证据。",
        "- `sample_w_contribution` 表示样本对 Wasserstein / sliced Wasserstein 分布距离的贡献，不写作单个样本自身的 Wasserstein distance。",
        "",
        "## 计算公式",
        "",
        "### 1. 训练集标准化",
        "",
        "所有 scaler 只在当前 split/fold 的训练集上拟合。对任意变量 `v`：",
        "",
        "```text",
        "v* = (v - mean_train(v)) / std_train(v)",
        "```",
        "",
        "其中 `mean_train` 和 `std_train` 只由训练集计算，测试集只做 transform。",
        "",
        "### 2. Y-space: sample-level W1 contribution",
        "",
        "Y-space 使用目标值。先用训练集目标值标准化：",
        "",
        "```text",
        "y_train_i* = (y_train_i - mean_train(y)) / std_train(y)",
        "y_test_j*  = (y_test_j  - mean_train(y)) / std_train(y)",
        "```",
        "",
        "排序后做一维 Wasserstein-1 匹配。若测试样本 `j` 的质量加权贡献为 `c_y,j`，则：",
        "",
        "```text",
        "W_y,j = c_y,j / (1 / N_test)",
        "mean_j(W_y,j) = W1(y_train*, y_test*)",
        "```",
        "",
        "它解释的是目标值分布偏移，属于辅助证据，不单独证明输入 OOD。",
        "",
        "### 3. Z-space: sample-level W1 contribution",
        "",
        "Z-space 使用旧 embedding 文件中的二维坐标 `(x, y)`。先用训练集坐标标准化：",
        "",
        "```text",
        "z_train_i* = [(x_train_i - mean_train(x)) / std_train(x),",
        "              (y_train_i - mean_train(y)) / std_train(y)]",
        "z_test_j*  = [(x_test_j  - mean_train(x)) / std_train(x),",
        "              (y_test_j  - mean_train(y)) / std_train(y)]",
        "```",
        "",
        "用二维欧氏距离构造 cost matrix，并解精确 OT transport plan。若测试样本 `j` 的质量加权贡献为 `c_z,j`，则：",
        "",
        "```text",
        "W_z,j = c_z,j / (1 / N_test)",
        "mean_j(W_z,j) = W1(z_train*, z_test*)",
        "```",
        "",
        "它解释的是二维表征空间的分布偏移，属于辅助证据。",
        "",
        "### 4. X-space: sample-level sliced W contribution",
        "",
        "X-space 使用成分 + 工艺输入特征。数据中的 0 保留其物理含义；少量真正的 NaN/空值填 0。随后只在训练集拟合 standard scaler：",
        "",
        "```text",
        "z_train_i = standardized X vector of train sample i",
        "z_test_j  = standardized X vector of test sample j",
        "```",
        "",
        "对每个随机单位方向 `theta_r`，把多维 X 向量投影到一维：",
        "",
        "```text",
        "a_i^(r) = theta_r^T z_train_i",
        "b_j^(r) = theta_r^T z_test_j",
        "```",
        "",
        "在该方向上，把训练集和测试集视作经验分布：",
        "",
        "```text",
        "train sample mass = 1 / N_train",
        "test sample mass  = 1 / N_test",
        "```",
        "",
        "排序后做一维 Wasserstein-1 匹配。测试样本 `j` 在第 `r` 个方向上的质量加权贡献为：",
        "",
        "```text",
        "c_j^(r) = sum_over_matches moved_mass * |a_i^(r) - b_j^(r)|",
        "w_j^(r) = c_j^(r) / (1 / N_test)",
        "sample_w_contribution_j = mean_r w_j^(r)",
        "```",
        "",
        "一致性关系为：",
        "",
        "```text",
        "mean_j(sample_w_contribution_j) ~= split-level sliced Wasserstein",
        "```",
        "",
        "### 5. Method-level ratio",
        "",
        "X-space 的 method-level 主指标使用样本级贡献的中位数，以降低极端样本主导结论的风险：",
        "",
        "```text",
        "X_median_W_method = median_j(sample_w_contribution_j)",
        "X_top10pct_mean_W_method = mean(top 10% sample_w_contribution_j)",
        "```",
        "",
        "三空间 ratio 都以 `random_cv_baseline` 为基准：",
        "",
        "```text",
        "Y_ratio = W1(y*)_method / W1(y*)_random_cv",
        "Z_ratio = W1(z*)_method / W1(z*)_random_cv",
        "X_ratio = median(sample_w_contribution)_method / median(sample_w_contribution)_random_cv",
        "```",
        "",
        "如果某个 OOD method 的 ratio 明显大于 1，说明该 method 的测试集相对训练集比随机划分更偏离。",
        "",
        "## 主要结果",
        "",
        "按 X-space ratio 排序的高偏移 method：",
        "",
        markdown_table(
            top_x,
            ["case_label", "method_label", "Y_ratio", "Z_ratio", "X_ratio", "X_median_W", "X_top10pct_mean_W"],
            max_rows=15,
        ),
        "",
        "X-space top-W 样本（全局前 20）：",
        "",
    ]

    if top_sample_rows.empty:
        lines.append("无可用 top 样本。")
    else:
        sample_cols = [
            column
            for column in ["case_label", "method", "ID", "target_value", "sample_w_contribution", "sample_w_rank_desc"]
            if column in top_sample_rows.columns
        ]
        lines.append(markdown_table(top_sample_rows, sample_cols, max_rows=20))

    lines.extend(["", "## 缺失说明", ""])
    if missing_cases:
        lines.append("以下 case 在旧 Y/Z-space method 汇总或 embedding 文件中缺失，因此对应 `Y_W`、`Z_W` 或 ratio 标记为空：")
        lines.extend(f"- {case}" for case in missing_cases)
    else:
        lines.append("所有 X-space case 均匹配到旧 Y/Z-space 汇总。")

    lines.extend(
        [
            "",
            "## 分 task 图表",
            "",
            "每个 task 输出两个单图：`task_w_ratio` 为三空间 method-level ratio；`task_w_curves` 为 3×3 图，左列保留完整样本 W 箱线图，中列为各 method 聚合后的样本贡献排序曲线，右列为 LOCO/Random-CV 分 fold 排序曲线。右列使用 fold 颜色区分 fold，LOCO 为实线，Random-CV 为虚线。",
            "",
        ]
    )
    for case_key, case_summary in summary.groupby("case_key", sort=False):
        case_label = case_summary["case_label"].iloc[0]
        case_dir = output_root / "cases" / case_key
        ratio_png = case_dir / "task_w_ratio.png"
        curves_png = case_dir / "task_w_curves.png"
        ratio_rel = ratio_png.relative_to(output_root).as_posix()
        curves_rel = curves_png.relative_to(output_root).as_posix()
        lines.extend(
            [
                f"### {case_label}",
                "",
                f"![{case_label} task W ratio]({ratio_rel})",
                "",
                f"![{case_label} task W curves]({curves_rel})",
                "",
                markdown_table(
                    case_summary,
                    ["method_label", "Y_ratio", "Z_ratio", "X_ratio", "X_median_W", "X_top10pct_mean_W"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 解释建议",
            "",
            "- 主结论优先看 `X_ratio`、X-space 箱线图和 X-space 排序曲线，因为它们直接对应成分 + 工艺输入分布偏移。",
            "- `Y_ratio` 用于说明目标值是否也发生偏移，适合作为 label shift 辅助证据。",
            "- `Z_ratio` 用于说明二维表征空间是否偏移，适合作为可视化和表征空间辅助证据。",
            "- 若某个 method 的 `X_ratio` 明显大于 1，并且箱线图整体上移或排序曲线高尾更长，可以支持该测试集比 random-CV 更 OOD。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    x_report_root = Path(args.x_report_root)
    yz_summary_path = Path(args.yz_summary)
    split_summary_path = Path(args.split_summary) if args.split_summary else x_report_root.parent / "ood_split_summary.csv"
    embedding_data_dir = Path(args.embedding_data_dir)
    output_root = Path(args.output_root) if args.output_root else x_report_root
    formats = list(args.formats)

    reused_cached_tables = False
    try:
        sample_values, x_summary, yz_summary = load_inputs(x_report_root, yz_summary_path)
    except FileNotFoundError as error:
        cached_summary_path = output_root / "three_space_w_ratio_summary.csv"
        cached_samples_path = output_root / "three_space_sample_w_values.csv"
        cached_sample_summary_path = output_root / "three_space_sample_w_summary.csv"
        cached_top_samples_path = output_root / "xspace_top_w_samples.csv"
        can_reuse_cached_tables = (
            "Missing Y/Z-space Wasserstein summary" in str(error)
            and cached_summary_path.exists()
            and cached_samples_path.exists()
        )
        if not can_reuse_cached_tables:
            raise
        reused_cached_tables = True
        print(f"Missing Y/Z-space source; reusing cached three-space tables under: {output_root}")
        summary = sort_summary_by_case_and_ood(read_csv(cached_summary_path))
        three_space_samples = read_csv(cached_samples_path)
        three_space_sample_summary = (
            read_csv(cached_sample_summary_path)
            if cached_sample_summary_path.exists()
            else summarize_three_space_sample_values(three_space_samples)
        )
        top_samples = read_csv(cached_top_samples_path) if cached_top_samples_path.exists() else pd.DataFrame()
    else:
        summary = sort_summary_by_case_and_ood(build_three_space_summary(x_summary, yz_summary))
        sample_values = prepare_sample_values(sample_values)
        top_samples = build_top_samples(sample_values, args.top_n)
        three_space_samples = build_three_space_sample_values(sample_values, split_summary_path, embedding_data_dir)
        three_space_sample_summary = summarize_three_space_sample_values(three_space_samples)

        write_csv(summary, output_root / "three_space_w_ratio_summary.csv")
        write_csv(top_samples, output_root / "xspace_top_w_samples.csv")
        write_csv(three_space_samples, output_root / "three_space_sample_w_values.csv")
        write_csv(three_space_sample_summary, output_root / "three_space_sample_w_summary.csv")

    for case_key, case_summary in summary.groupby("case_key", sort=False):
        case_dir = output_root / "cases" / case_key
        case_samples = three_space_samples[three_space_samples["case_key"] == case_key].copy()
        write_csv(case_summary, case_dir / "task_w_summary.csv")
        plot_task_ratio(case_summary, case_dir / "task_w_ratio", formats, args.dpi)
        plot_task_curves(case_summary, case_samples, case_dir / "task_w_curves", formats, args.dpi)
        remove_stale_task_dashboard_figures(case_dir)

    report = build_report(summary, top_samples, output_root, x_report_root, yz_summary_path)
    report_path = output_root / "three_space_w_ood_report.md"
    report_path.write_text(report, encoding="utf-8")

    table_action = "Reused cached" if reused_cached_tables else "Saved"
    print(f"{table_action} three-space W summary: {output_root / 'three_space_w_ratio_summary.csv'}")
    print(f"{table_action} X-space top-W samples: {output_root / 'xspace_top_w_samples.csv'}")
    print(f"{table_action} three-space sample W values: {output_root / 'three_space_sample_w_values.csv'}")
    print(f"{table_action} three-space sample W summary: {output_root / 'three_space_sample_w_summary.csv'}")
    print(f"Saved Markdown report: {report_path}")
    print(f"Saved case figures under: {output_root / 'cases'}")
    missing_yz_cases = sorted(summary[summary[["Y_W", "Z_W"]].isna().all(axis=1)]["case_label"].dropna().unique())
    if missing_yz_cases:
        print("Y/Z-space missing cases:")
        for case_label in missing_yz_cases:
            print(f" - {case_label}")


if __name__ == "__main__":
    main()
