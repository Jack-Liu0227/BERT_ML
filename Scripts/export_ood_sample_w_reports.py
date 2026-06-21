from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from textwrap import wrap
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SCORE_ROOT = Path("output") / "ood_xspace_scores"
DEFAULT_OUTPUT_SUBDIR = "sample_w_reports"
W_SCORE_COL = "sliced_wasserstein_sample_score"
W_MASS_COL = "sliced_wasserstein_mass_contribution"

CASE_COLUMNS = ["alloy_family", "dataset_name", "property"]
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
        description=(
            "Export core per-test-sample contributions to sliced Wasserstein distance. "
            "By default this writes only the two tables needed by the three-space report; "
            "use --detailed for the legacy per-case dashboards and extra audit files."
        )
    )
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT), help="Directory containing OOD score CSV outputs.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <score-root>/sample_w_reports.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--top-n", type=int, default=25, help="Rows per method in each case top-W table when --detailed is used.")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also write legacy per-case CSVs, dashboards, heatmap, consistency check, and manifest.",
    )
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


def ordered_methods(methods: Iterable[str]) -> list[str]:
    available = {str(method) for method in methods}
    known = [method for method in METHOD_ORDER if method in available]
    unknown = sorted(available - set(known))
    return known + unknown


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4c78a8")


def slugify(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text or "unknown"


def case_key(row: pd.Series) -> str:
    return "__".join(slugify(row[col]) for col in CASE_COLUMNS)


def case_label_from_frame(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    return f"{first['alloy_family']} / {first['dataset_name']} / {first['property']}"


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


def set_nonnegative_adaptive_axis(ax: plt.Axes, values: Iterable[float] | None = None) -> None:
    finite = np.asarray([], dtype=float)
    if values is not None:
        finite = np.asarray([float(value) for value in values if pd.notna(value) and math.isfinite(float(value))])
    max_value = float(np.nanmax(finite)) if finite.size else 1.0
    if finite.size and max_value > 15:
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("linear")
    _, upper = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=upper)


def required_columns() -> list[str]:
    return [
        *CASE_COLUMNS,
        "split_strategy",
        "split_id",
        "fold_id",
        "source_split_dir",
        "__row_id__",
        "__source_index__",
        "target_col",
        "target_value",
        "ood_score",
        "ood_percentile_vs_train",
        W_SCORE_COL,
        W_MASS_COL,
        "sliced_wasserstein_rank_desc",
    ]


def load_inputs(score_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples_path = score_root / "all_ood_sample_scores.csv"
    summary_path = score_root / "ood_split_summary.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing sample score file: {samples_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing split summary file: {summary_path}")

    samples = read_csv(samples_path)
    summary = read_csv(summary_path)
    missing = [column for column in required_columns() if column not in samples.columns]
    if missing:
        raise ValueError(f"Sample score file is missing required columns: {missing}")
    if "sliced_wasserstein" not in summary.columns:
        raise ValueError("Split summary file is missing required column: sliced_wasserstein")
    return samples, summary


def build_sample_w_table(samples: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *CASE_COLUMNS,
        "split_strategy",
        "split_id",
        "fold_id",
        "source_split_dir",
        "__row_id__",
        "__source_index__",
        "target_col",
        "target_value",
        "ood_score",
        "ood_percentile_vs_train",
        W_SCORE_COL,
        W_MASS_COL,
        "sliced_wasserstein_rank_desc",
    ]
    if "ID" in samples.columns:
        columns.insert(columns.index("target_col"), "ID")
    table = samples[columns].copy()
    table = table.rename(
        columns={
            "split_strategy": "method",
            W_SCORE_COL: "sample_w_contribution",
            W_MASS_COL: "sample_w_mass_contribution",
            "sliced_wasserstein_rank_desc": "sample_w_rank_desc",
        }
    )
    table["case_label"] = (
        table["alloy_family"].astype(str)
        + " / "
        + table["dataset_name"].astype(str)
        + " / "
        + table["property"].astype(str)
    )
    sort_cols = ["alloy_family", "dataset_name", "property", "method", "fold_id", "sample_w_rank_desc"]
    return table.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def top10_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if finite.empty:
        return float("nan")
    top_n = max(1, int(math.ceil(len(finite) * 0.1)))
    return float(finite.tail(top_n).mean())


def summarize_w(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [*CASE_COLUMNS, "method"]
    for keys, group in table.groupby(group_cols, sort=True, dropna=False):
        key_values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        values = pd.to_numeric(group["sample_w_contribution"], errors="coerce")
        mass_values = pd.to_numeric(group["sample_w_mass_contribution"], errors="coerce")
        row = {
            **key_values,
            "case_label": f"{key_values['alloy_family']} / {key_values['dataset_name']} / {key_values['property']}",
            "sample_count": int(values.notna().sum()),
            "fold_count": int(group["fold_id"].dropna().nunique()),
            "sample_w_mean": float(values.mean()),
            "sample_w_median": float(values.median()),
            "sample_w_q25": float(values.quantile(0.25)),
            "sample_w_q75": float(values.quantile(0.75)),
            "sample_w_p90": float(values.quantile(0.90)),
            "sample_w_max": float(values.max()),
            "sample_w_top10pct_mean": top10_mean(values),
            "sample_w_mass_sum": float(mass_values.sum()),
            "ood_score_median": float(pd.to_numeric(group["ood_score"], errors="coerce").median()),
            "ood_percentile_median": float(pd.to_numeric(group["ood_percentile_vs_train"], errors="coerce").median()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    summary["_method_order"] = summary["method"].map(method_order).fillna(999)
    summary = summary.sort_values([*CASE_COLUMNS, "_method_order", "method"], kind="stable").drop(columns="_method_order")
    return summary.reset_index(drop=True)


def summarize_case_top_samples(case_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    parts = []
    for method in ordered_methods(case_table["method"]):
        method_frame = case_table[case_table["method"] == method].copy()
        method_frame = method_frame.sort_values("sample_w_contribution", ascending=False, kind="stable").head(top_n)
        parts.append(method_frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=case_table.columns)


def plot_case_dashboard(case_table: pd.DataFrame, case_summary: pd.DataFrame, output_base: Path, formats: list[str], dpi: int) -> None:
    methods = ordered_methods(case_table["method"])
    labels = [method_label(method) for method in methods]
    colors = [method_color(method) for method in methods]
    case_label = case_label_from_frame(case_table)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    fig.suptitle(
        f"Sample-level contribution to sliced Wasserstein distance\n{case_label}",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    ax = axes[0]
    data = [
        pd.to_numeric(case_table.loc[case_table["method"] == method, "sample_w_contribution"], errors="coerce").dropna()
        for method in methods
    ]
    parts = ax.violinplot(data, positions=np.arange(len(methods)), showmeans=False, showmedians=True, widths=0.82)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.55)
        body.set_linewidth(0.6)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.0)
    for index, values in enumerate(data):
        if values.empty:
            continue
        if len(values) > 220:
            values = values.sample(220, random_state=42)
        jitter = np.random.default_rng(index).normal(0.0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=9,
            alpha=0.35,
            color=colors[index],
            edgecolor="none",
        )
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Sample W contribution")
    ax.set_title("A. Per-sample W distribution", loc="left", fontweight="bold")
    all_distribution_values = pd.to_numeric(case_table["sample_w_contribution"], errors="coerce").dropna()
    set_nonnegative_adaptive_axis(ax, all_distribution_values)
    style_axes(ax)

    ax = axes[1]
    for method, color, label in zip(methods, colors, labels):
        values = (
            pd.to_numeric(case_table.loc[case_table["method"] == method, "sample_w_contribution"], errors="coerce")
            .dropna()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        if values.empty:
            continue
        x = normalized_rank_x(len(values))
        ax.plot(x, values, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Normalized rank within method")
    ax.set_ylabel("Sample W contribution")
    ax.set_title("B. Ranked W contribution curve", loc="left", fontweight="bold")
    curve_values = pd.to_numeric(case_table["sample_w_contribution"], errors="coerce").dropna()
    set_nonnegative_adaptive_axis(ax, curve_values)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[2]
    summary_indexed = case_summary.set_index("method").reindex(methods)
    x = np.arange(len(methods))
    width = 0.38
    medians = summary_indexed["sample_w_median"].astype(float).to_numpy()
    top_means = summary_indexed["sample_w_top10pct_mean"].astype(float).to_numpy()
    ax.bar(x - width / 2, medians, width=width, color=colors, alpha=0.72, label="Median W")
    ax.bar(x + width / 2, top_means, width=width, color=colors, alpha=0.98, hatch="//", label="Top 10% mean W")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Sample W contribution")
    ax.set_title("C. Method-level W summary", loc="left", fontweight="bold")
    summary_values = np.concatenate([medians[np.isfinite(medians)], top_means[np.isfinite(top_means)]])
    set_nonnegative_adaptive_axis(ax, summary_values)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    fig.tight_layout()
    save_figure(fig, output_base, formats, dpi)


def plot_global_heatmap(case_summary: pd.DataFrame, output_base: Path, formats: list[str], dpi: int) -> None:
    work = case_summary.copy()
    work["case_label"] = (
        work["alloy_family"].astype(str)
        + " / "
        + work["dataset_name"].astype(str)
        + " / "
        + work["property"].astype(str)
    )
    case_order = (
        work[[*CASE_COLUMNS, "case_label"]]
        .drop_duplicates()
        .sort_values(CASE_COLUMNS, kind="stable")["case_label"]
        .tolist()
    )
    methods = ordered_methods(work["method"])
    pivot = (
        work.pivot_table(index="case_label", columns="method", values="sample_w_median", aggfunc="median")
        .reindex(index=case_order, columns=methods)
    )
    values = np.log1p(pivot.to_numpy(dtype=float))
    fig_height = max(5.0, 0.42 * len(case_order) + 2.0)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([method_label(method) for method in methods], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(case_order)))
    ax.set_yticklabels(["\n".join(wrap(label, 28)) for label in case_order], fontsize=8)
    ax.set_title("Median sample-level W contribution by case and method", fontweight="bold")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if math.isfinite(float(value)):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("log(1 + median sample W contribution)")
    fig.tight_layout()
    save_figure(fig, output_base, formats, dpi)


def export_case_outputs(sample_table: pd.DataFrame, case_summary: pd.DataFrame, output_dir: Path, formats: list[str], dpi: int, top_n: int) -> int:
    case_count = 0
    for _, case_frame in sample_table.groupby(CASE_COLUMNS, sort=True, dropna=False):
        case_count += 1
        case_dir = output_dir / "cases" / case_key(case_frame.iloc[0])
        case_dir.mkdir(parents=True, exist_ok=True)

        current_case_summary = case_summary.merge(
            case_frame[CASE_COLUMNS].drop_duplicates(),
            on=CASE_COLUMNS,
            how="inner",
        )
        write_csv(case_frame, case_dir / "sample_w_values.csv")
        write_csv(current_case_summary, case_dir / "method_w_summary.csv")
        write_csv(summarize_case_top_samples(case_frame, top_n=top_n), case_dir / f"top_{top_n}_sample_w_values_by_method.csv")
        plot_case_dashboard(case_frame, current_case_summary, case_dir / "sample_w_dashboard", formats, dpi)
    return case_count


def validate_consistency(sample_table: pd.DataFrame, split_summary: pd.DataFrame) -> pd.DataFrame:
    sample_check = (
        sample_table.groupby("source_split_dir", as_index=False)
        .agg(
            sample_w_mean=("sample_w_contribution", "mean"),
            sample_w_mass_sum=("sample_w_mass_contribution", "sum"),
            sample_count=("sample_w_contribution", "size"),
        )
    )
    summary_cols = ["source_split_dir", "sliced_wasserstein", *CASE_COLUMNS, "split_strategy", "fold_id"]
    merged = sample_check.merge(split_summary[summary_cols], on="source_split_dir", how="left")
    merged["mean_minus_split_w"] = merged["sample_w_mean"] - merged["sliced_wasserstein"]
    merged["mass_sum_minus_split_w"] = merged["sample_w_mass_sum"] - merged["sliced_wasserstein"]
    merged["abs_mean_error"] = merged["mean_minus_split_w"].abs()
    merged["abs_mass_error"] = merged["mass_sum_minus_split_w"].abs()
    return merged.sort_values("abs_mean_error", ascending=False, kind="stable").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    score_root = Path(args.score_root)
    output_dir = Path(args.output_dir) if args.output_dir else score_root / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, split_summary = load_inputs(score_root)
    sample_table = build_sample_w_table(samples)
    case_summary = summarize_w(sample_table)

    write_csv(sample_table, output_dir / "all_sample_w_values.csv")
    write_csv(case_summary, output_dir / "case_method_w_summary.csv")
    case_count = int(sample_table[CASE_COLUMNS].drop_duplicates().shape[0])

    consistency = None
    if args.detailed:
        consistency = validate_consistency(sample_table, split_summary)
        write_csv(consistency, output_dir / "sample_w_consistency_check.csv")
        plot_global_heatmap(case_summary, output_dir / "global_case_method_w_heatmap", args.formats, args.dpi)
        case_count = export_case_outputs(sample_table, case_summary, output_dir, args.formats, args.dpi, args.top_n)
        metadata = {
            "score_root": str(score_root),
            "output_dir": str(output_dir),
            "sample_rows": int(len(sample_table)),
            "case_count": int(case_count),
            "method_count": int(sample_table["method"].nunique()),
            "w_column": W_SCORE_COL,
            "mass_column": W_MASS_COL,
            "max_abs_mean_error": float(consistency["abs_mean_error"].max()),
            "max_abs_mass_error": float(consistency["abs_mass_error"].max()),
            "description": "sample-level contribution to sliced Wasserstein distance",
        }
        (output_dir / "sample_w_report_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Wrote core sample-level W tables to {output_dir}")
    print(f"Rows: {len(sample_table)}")
    print(f"Cases: {case_count}")
    print(f"Methods: {sample_table['method'].nunique()}")
    if consistency is not None:
        print(f"Max |mean(sample W) - split W|: {float(consistency['abs_mean_error'].max()):.6g}")
        print(f"Max |sum(sample W mass) - split W|: {float(consistency['abs_mass_error'].max()):.6g}")
    else:
        print("Detailed per-case files skipped. Use --detailed to regenerate legacy dashboards and audit files.")


if __name__ == "__main__":
    main()
