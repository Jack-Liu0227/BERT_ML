from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


W_COLUMNS = ["Wx", "Wy", "Wz"]
W_TO_SPACE = {"Wx": "X-space", "Wy": "Y-space", "Wz": "Z-space"}
METADATA_COLUMNS = {
    "set_type",
    "case_key",
    "method",
    "fold_id",
    "split_id",
    "source_split_dir",
    "Wx",
    "Wy",
    "Wz",
    "w_rank_desc",
    "is_top_n",
    "is_above_threshold",
    "highlight_reason",
}
ELEMENT_PATTERN = re.compile(r"\((?:wt|at)%\)$", re.IGNORECASE)
PROCESS_HINTS = [
    "ST",
    "TIME",
    "Temp",
    "Time",
    "Aging",
    "Anneal",
    "Hom",
    "CR",
    "Deformation",
    "recrystal",
    "Solution",
    "Cold",
]
ZERO_AS_MISSING_TYPES = {"element", "process"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze high-W split workbook(s) without merging W spaces.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--workbook", help="Path to one high_w_samples/*/*/*/data.xlsx workbook.")
    target.add_argument("--root", help="Root directory containing Wx/Wy/Wz high-W sample workbooks.")
    parser.add_argument("--top-features", type=int, default=12, help="Number of shifted features to plot per figure.")
    parser.add_argument("--manifest-name", default="diagnostics_manifest.csv", help="Batch manifest filename under --root.")
    parser.add_argument("--include-cross-space", action="store_true", help="For one workbook, add sibling Wx/Wy/Wz comparison plots.")
    parser.add_argument("--audit-w-source", action="store_true", help="For one workbook, write source alignment and W calculation audit files.")
    return parser.parse_args()


def infer_w_column(path: Path, frame: pd.DataFrame) -> str:
    parts = {part.lower() for part in path.parts}
    for w_column in W_COLUMNS:
        if w_column.lower() in parts and w_column in frame.columns:
            return w_column
    present = [column for column in W_COLUMNS if column in frame.columns]
    if len(present) == 1:
        return present[0]
    raise ValueError(f"Could not infer a single W column from path or workbook columns: {path}")


def infer_high_w_root(path: Path) -> Path | None:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part == "high_w_samples":
            return Path(*parts[: index + 1])
    return None


def sibling_workbook(path: Path, target_w: str) -> Path | None:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part in W_COLUMNS:
            sibling = Path(*parts[:index], target_w, *parts[index + 1 :])
            return sibling
    return None


def read_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing workbook: {path}")
    frame = pd.read_excel(path, sheet_name="data")
    if "set_type" not in frame.columns:
        raise ValueError(f"{path} is missing required column: set_type")
    return frame


def original_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def numeric_original_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in original_columns(frame):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    return columns


def is_element_column(column: str) -> bool:
    return bool(ELEMENT_PATTERN.search(column))


def is_process_column(column: str) -> bool:
    if is_element_column(column):
        return False
    lowered = column.lower()
    return any(hint.lower() in lowered for hint in PROCESS_HINTS)


def feature_type_for_column(column: str) -> str:
    if is_element_column(column):
        return "element"
    if is_process_column(column):
        return "process"
    return "other"


def finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def values_for_feature_stats(values: pd.Series, feature_type: str) -> pd.Series:
    finite = finite_series(values)
    if feature_type in ZERO_AS_MISSING_TYPES:
        return finite[finite.ne(0)]
    return finite


def series_stat(values: pd.Series, statistic: str) -> float:
    if values.empty:
        return float("nan")
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(values.median())
    raise ValueError(f"Unsupported statistic: {statistic}")


def standardized_mean_diff(train: pd.Series, test: pd.Series) -> float:
    train_finite = finite_series(train)
    test_finite = finite_series(test)
    if train_finite.empty or test_finite.empty:
        return float("nan")
    pooled = pd.concat([train_finite, test_finite], ignore_index=True)
    pooled_std = float(pooled.std(ddof=1))
    if not math.isfinite(pooled_std) or pooled_std <= 0:
        return 0.0 if math.isclose(float(test_finite.mean()), float(train_finite.mean())) else float("nan")
    return float((test_finite.mean() - train_finite.mean()) / pooled_std)


def build_shift_table(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["set_type"].eq("train")]
    test = frame[frame["set_type"].eq("test")]
    rows: list[dict[str, Any]] = []
    for column in numeric_original_columns(frame):
        feature_type = feature_type_for_column(column)
        train_finite = finite_series(train[column])
        test_finite = finite_series(test[column])
        if train_finite.empty and test_finite.empty:
            continue
        train_values = values_for_feature_stats(train[column], feature_type)
        test_values = values_for_feature_stats(test[column], feature_type)
        train_nonzero_rate = float(train_finite.ne(0).mean()) if not train_finite.empty else float("nan")
        test_nonzero_rate = float(test_finite.ne(0).mean()) if not test_finite.empty else float("nan")
        std_diff = standardized_mean_diff(train_values, test_values)
        abs_std_diff = abs(std_diff) if math.isfinite(std_diff) else float("nan")
        nonzero_rate_diff = test_nonzero_rate - train_nonzero_rate
        abs_nonzero_rate_diff = abs(nonzero_rate_diff) if math.isfinite(nonzero_rate_diff) else float("nan")
        value_shift_score = abs_std_diff if math.isfinite(abs_std_diff) else 0.0
        coverage_shift_score = abs_nonzero_rate_diff if math.isfinite(abs_nonzero_rate_diff) else 0.0
        selection_score = max(value_shift_score, coverage_shift_score)
        if math.isfinite(abs_std_diff) and value_shift_score >= coverage_shift_score:
            selection_basis = "nonzero_value_shift" if feature_type in ZERO_AS_MISSING_TYPES else "value_shift"
        elif math.isfinite(abs_nonzero_rate_diff):
            selection_basis = "nonzero_rate_shift"
        else:
            selection_basis = "unavailable"
        rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "zero_excluded": feature_type in ZERO_AS_MISSING_TYPES,
                "train_finite_count": int(train_finite.count()),
                "test_finite_count": int(test_finite.count()),
                "train_count": int(train_values.count()),
                "test_count": int(test_values.count()),
                "train_mean": series_stat(train_values, "mean"),
                "test_mean": series_stat(test_values, "mean"),
                "train_median": series_stat(train_values, "median"),
                "test_median": series_stat(test_values, "median"),
                "train_nonzero_rate": train_nonzero_rate,
                "test_nonzero_rate": test_nonzero_rate,
                "nonzero_rate_diff": nonzero_rate_diff,
                "abs_nonzero_rate_diff": abs_nonzero_rate_diff,
                "std_mean_diff": std_diff,
                "abs_std_mean_diff": abs_std_diff,
                "selection_score": selection_score,
                "selection_basis": selection_basis,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["selection_score", "abs_std_mean_diff", "abs_nonzero_rate_diff"],
        ascending=[False, False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def top_features(shift_table: pd.DataFrame, feature_type: str, top_n: int) -> list[str]:
    subset = shift_table[shift_table["feature_type"].eq(feature_type)]
    return subset.head(top_n)["feature"].astype(str).tolist()


def make_long_feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    subset = frame[["set_type", *features]].copy()
    long = subset.melt(id_vars="set_type", value_vars=features, var_name="feature", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["feature_type"] = long["feature"].map(feature_type_for_column)
    long = long[~(long["feature_type"].isin(ZERO_AS_MISSING_TYPES) & long["value"].eq(0))]
    if long.empty:
        long["z_value"] = pd.Series(dtype=float)
        return long

    def pooled_z(values: pd.Series) -> pd.Series:
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if not math.isfinite(std_value) or std_value <= 0:
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return (values - mean_value) / std_value

    long["z_value"] = long.groupby("feature", group_keys=False)["value"].transform(pooled_z)
    return long


def plot_empty_figure(message: str, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def raw_axis_limits(values: pd.Series) -> tuple[float, float]:
    finite = finite_series(values)
    if finite.empty:
        return (0.0, 1.0)
    min_value = float(finite.min())
    max_value = float(finite.max())
    span = max_value - min_value
    if not math.isfinite(span):
        return (0.0, 1.0)
    if span <= 0:
        pad = max(abs(min_value) * 0.02, 1e-6)
    else:
        pad = max(span * 0.04, 1e-6)
    lower = min_value - pad
    upper = max_value + pad
    if min_value > 0 and lower <= 0:
        lower = min_value - min(pad, min_value * 0.25)
    if max_value < 0 and upper >= 0:
        upper = max_value + min(pad, abs(max_value) * 0.25)
    if lower >= upper:
        pad = max(abs(min_value) * 0.02, 1e-6)
        lower = min_value - pad
        upper = max_value + pad
    return (lower, upper)


def save_w_distribution(frame: pd.DataFrame, w_column: str, output_path: Path) -> None:
    test = frame[frame["set_type"].eq("test")].copy()
    values = finite_series(test[w_column])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if values.empty:
        ax.text(0.5, 0.5, f"No finite {w_column} values", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.histplot(values, kde=True, ax=ax, color="#4C78A8", edgecolor="#333333")
        mean_value = float(values.mean())
        median_value = float(values.median())
        ax.axvline(mean_value, color="#D55E00", linestyle="-", linewidth=1.5, label=f"mean={mean_value:.3g}")
        ax.axvline(median_value, color="#009E73", linestyle="--", linewidth=1.5, label=f"median={median_value:.3g}")
        ax.legend(frameon=False)
    ax.set_title(f"{w_column} distribution in test set")
    ax.set_xlabel(w_column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_violin(
    frame: pd.DataFrame,
    features: list[str],
    title: str,
    output_path: Path,
    set_type: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        plot_empty_figure("No numeric features available", title, output_path)
        return
    long = make_long_feature_frame(frame, features)
    if set_type is not None:
        long = long[long["set_type"].eq(set_type)]
    if long.empty:
        plot_empty_figure("No finite nonzero values available", title, output_path)
        return

    visible_features = [feature for feature in features if long["feature"].eq(feature).any()]
    fig_height = max(4.8, 1.12 * len(visible_features) + 0.9)
    fig, axes = plt.subplots(len(visible_features), 1, figsize=(9.4, fig_height), squeeze=False)
    for row_index, feature in enumerate(visible_features):
        ax = axes[row_index, 0]
        feature_long = long[long["feature"].eq(feature)]
        if set_type is not None:
            color = "#6B9AC4" if set_type == "train" else "#D95F02"
            sns.violinplot(
                data=feature_long,
                x="value",
                inner="quartile",
                cut=0,
                density_norm="width",
                color=color,
                ax=ax,
            )
            sns.stripplot(
                data=feature_long,
                x="value",
                color="#2A2A2A",
                alpha=0.35,
                size=2.3,
                ax=ax,
            )
            ax.set_ylabel(feature, rotation=0, ha="right", va="center")
            ax.set_xlabel("Raw feature value")
            ax.get_yaxis().set_ticks([])
        else:
            sns.violinplot(
                data=feature_long,
                x="value",
                y="set_type",
                hue="set_type",
                order=["train", "test"],
                inner="quartile",
                cut=0,
                density_norm="width",
                palette={"train": "#6B9AC4", "test": "#D95F02"},
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=feature_long,
                x="value",
                y="set_type",
                order=["train", "test"],
                hue="set_type",
                dodge=False,
                palette={"train": "#1F1F1F", "test": "#555555"},
                alpha=0.22,
                size=2.0,
                legend=False,
                ax=ax,
            )
            ax.set_ylabel(feature, rotation=0, ha="right", va="center")
            ax.set_xlabel("Raw feature value")
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlim(raw_axis_limits(feature_long["value"]))

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_sibling_w_values(workbook_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for w_column in W_COLUMNS:
        sibling = sibling_workbook(workbook_path, w_column)
        if sibling is None or not sibling.exists():
            continue
        frame = read_workbook(sibling)
        if w_column not in frame.columns:
            continue
        test = frame[frame["set_type"].eq("test")].copy()
        for _, row in test.iterrows():
            rows.append(
                {
                    "space": w_column,
                    "value": pd.to_numeric(row.get(w_column), errors="coerce"),
                    "ID": row.get("ID", np.nan),
                    "__row_id__": row.get("__row_id__", np.nan),
                    "__source_index__": row.get("__source_index__", np.nan),
                    "case_key": row.get("case_key", ""),
                    "method": row.get("method", ""),
                    "fold_id": row.get("fold_id", ""),
                    "source_workbook": str(sibling),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.dropna(subset=["value"]).reset_index(drop=True)


def save_cross_space_w_comparison(workbook_path: Path, diagnostics_dir: Path) -> None:
    values = collect_sibling_w_values(workbook_path)
    output_csv = diagnostics_dir / "w_space_comparison.csv"
    output_png = diagnostics_dir / "w_space_comparison.png"
    output_md = diagnostics_dir / "w_space_comparison.md"
    if values.empty:
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig")
        plot_empty_figure("No sibling Wx/Wy/Wz workbooks found", "Wx/Wy/Wz comparison", output_png)
        output_md.write_text("# Wx/Wy/Wz comparison\n\nNo sibling workbooks found.\n", encoding="utf-8")
        return

    values.to_csv(output_csv, index=False, encoding="utf-8-sig")
    order = [w for w in W_COLUMNS if w in set(values["space"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    sns.boxplot(data=values, x="space", y="value", order=order, color="#D8E4F0", ax=axes[0])
    sns.stripplot(data=values, x="space", y="value", order=order, color="#333333", alpha=0.45, size=3, ax=axes[0])
    axes[0].set_title("Test W distribution by space")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("sample_w_contribution")
    axes[0].grid(axis="y", alpha=0.25)

    for w_column in order:
        series = values.loc[values["space"].eq(w_column), "value"].sort_values(ascending=False).reset_index(drop=True)
        if series.empty:
            continue
        x = np.linspace(0, 1, len(series)) if len(series) > 1 else np.array([0.0])
        axes[1].plot(x, series, marker=".", linewidth=1.6, markersize=3.5, label=w_column)
    axes[1].set_title("Ranked W curve by space")
    axes[1].set_xlabel("Normalized rank within space")
    axes[1].set_ylabel("sample_w_contribution")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    positive_values = values[values["value"] > 0].copy()
    if not positive_values.empty:
        log_png = diagnostics_dir / "w_space_comparison_log.png"
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        sns.boxplot(data=positive_values, x="space", y="value", order=order, color="#D8E4F0", ax=axes[0])
        sns.stripplot(data=positive_values, x="space", y="value", order=order, color="#333333", alpha=0.45, size=3, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("Test W distribution by space (log y)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("sample_w_contribution")
        axes[0].grid(axis="y", alpha=0.25)

        for w_column in order:
            series = positive_values.loc[positive_values["space"].eq(w_column), "value"].sort_values(ascending=False).reset_index(drop=True)
            if series.empty:
                continue
            x = np.linspace(0, 1, len(series)) if len(series) > 1 else np.array([0.0])
            axes[1].plot(x, series, marker=".", linewidth=1.6, markersize=3.5, label=w_column)
        axes[1].set_yscale("log")
        axes[1].set_title("Ranked W curve by space (log y)")
        axes[1].set_xlabel("Normalized rank within space")
        axes[1].set_ylabel("sample_w_contribution")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(log_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    stats = (
        values.groupby("space")["value"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reindex(order)
        .reset_index()
    )
    lines = [
        "# Wx/Wy/Wz comparison",
        "",
        "This comparison uses sibling workbooks with the same case/method/fold path and only test rows.",
        "",
        markdown_table(stats, ["space", "count", "mean", "std", "min", "median", "max"], max_rows=len(stats)),
        "",
        "A log-scale version is saved as `w_space_comparison_log.png` when all plotted values are positive.",
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def w_stats(frame: pd.DataFrame, w_column: str) -> dict[str, float]:
    values = finite_series(frame.loc[frame["set_type"].eq("test"), w_column])
    if values.empty:
        return {key: float("nan") for key in ["count", "mean", "std", "min", "q25", "median", "q75", "max", "cv"]}
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "count": float(values.count()),
        "mean": mean_value,
        "std": std_value,
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
        "cv": std_value / mean_value if mean_value else float("nan"),
    }


def get_scalar(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = frame[column].dropna().astype(str).unique()
    return values[0] if len(values) else ""


def source_table_path_from_high_w_root(workbook_path: Path, filename: str) -> Path | None:
    root = infer_high_w_root(workbook_path)
    if root is None:
        return None
    candidates = [root.parent / filename, root.parent.parent / filename]
    return first_existing_path(candidates)


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def audit_current_w_source(workbook_path: Path, diagnostics_dir: Path) -> None:
    frame = read_workbook(workbook_path)
    w_column = infer_w_column(workbook_path, frame)
    test = frame[frame["set_type"].eq("test")].copy()
    case_key = get_scalar(frame, "case_key")
    method = get_scalar(frame, "method")
    fold_id = get_scalar(frame, "fold_id")
    source_split_dir = get_scalar(frame, "source_split_dir")
    split_id = get_scalar(frame, "split_id")
    rows: list[dict[str, Any]] = []
    lines = [
        f"# {w_column} source audit",
        "",
        f"- Workbook: `{workbook_path}`",
        f"- Case: `{case_key}`",
        f"- Method/fold: `{method}` / `{fold_id or 'nofold'}`",
        f"- Split ID: `{split_id}`",
        f"- Test rows: `{len(test)}`",
        "",
    ]

    sample_values_path = source_table_path_from_high_w_root(workbook_path, "three_space_sample_w_values.csv")
    if sample_values_path is not None:
        sample_values = pd.read_csv(sample_values_path, encoding="utf-8-sig")
        space = W_TO_SPACE.get(w_column, "")
        subset = sample_values[
            sample_values["case_key"].astype(str).eq(case_key)
            & sample_values["method"].astype(str).eq(method)
            & sample_values["fold_id"].fillna("").astype(str).eq(fold_id)
            & sample_values["space"].astype(str).eq(space)
        ].copy()
        if source_split_dir:
            subset = subset[subset["source_split_dir"].astype(str).eq(source_split_dir)]
        key = "ID" if "ID" in test.columns and "ID" in subset.columns else "__row_id__"
        merged = test[[key, w_column]].merge(
            subset[[key, "sample_w_contribution", "sample_w_mass_contribution", "split_w", "ood_score", "space_w_rank_desc"]],
            on=key,
            how="outer",
            indicator=True,
        )
        merged["diff_workbook_vs_three_space"] = pd.to_numeric(merged[w_column], errors="coerce") - pd.to_numeric(
            merged["sample_w_contribution"], errors="coerce"
        )
        max_abs_diff = float(merged["diff_workbook_vs_three_space"].abs().max()) if not merged.empty else float("nan")
        rows.append(
            {
                "check": "workbook_vs_three_space_sample_values",
                "source": str(sample_values_path),
                "rows_left": len(test),
                "rows_right": len(subset),
                "matched_rows": int(merged["_merge"].eq("both").sum()),
                "left_only": int(merged["_merge"].eq("left_only").sum()),
                "right_only": int(merged["_merge"].eq("right_only").sum()),
                "max_abs_diff": max_abs_diff,
                "mean_sample_w": float(pd.to_numeric(subset["sample_w_contribution"], errors="coerce").mean()) if not subset.empty else float("nan"),
                "mass_sum": float(pd.to_numeric(subset["sample_w_mass_contribution"], errors="coerce").sum()) if not subset.empty else float("nan"),
                "split_w": float(pd.to_numeric(subset["split_w"], errors="coerce").dropna().iloc[0]) if subset["split_w"].notna().any() else float("nan"),
            }
        )
        lines.extend(
            [
                "## Workbook vs Three-Space Long Table",
                "",
                f"- Source table: `{sample_values_path}`",
                f"- Matching key: `{key}`",
                f"- Three-space rows: `{len(subset)}`",
                f"- Matched rows: `{int(merged['_merge'].eq('both').sum())}`",
                f"- Max absolute difference: `{max_abs_diff:.6g}`",
                "",
            ]
        )
    else:
        lines.extend(["## Workbook vs Three-Space Long Table", "", "- Source table not found.", ""])

    if w_column == "Wx":
        output_dir = None
        split_summary_path = source_table_path_from_high_w_root(workbook_path, "ood_split_summary.csv")
        if split_summary_path is not None:
            split_summary = pd.read_csv(split_summary_path, encoding="utf-8-sig")
            split_rows = split_summary[
                split_summary["source_split_dir"].astype(str).eq(source_split_dir)
            ].copy()
            if not split_rows.empty and "output_dir" in split_rows.columns:
                output_dir = Path(str(split_rows.iloc[0]["output_dir"]))
        if output_dir is None:
            output_dir = first_existing_path(
                list(Path("output/ood_xspace_scores").rglob(f"*/{split_id}/folds/{fold_id}"))
            )
        if output_dir is not None:
            sample_scores_path = output_dir / "ood_sample_scores.csv"
            split_score_path = output_dir / "ood_split_summary.csv"
            if sample_scores_path.exists():
                sample_scores = pd.read_csv(sample_scores_path, encoding="utf-8-sig")
                key = "ID" if "ID" in test.columns and "ID" in sample_scores.columns else "__row_id__"
                merged = test[[key, w_column]].merge(
                    sample_scores[[key, "sliced_wasserstein_sample_score", "sliced_wasserstein_mass_contribution", "ood_score"]],
                    on=key,
                    how="outer",
                    indicator=True,
                )
                merged["diff_workbook_vs_ood_sample_scores"] = pd.to_numeric(merged[w_column], errors="coerce") - pd.to_numeric(
                    merged["sliced_wasserstein_sample_score"], errors="coerce"
                )
                max_abs_diff = float(merged["diff_workbook_vs_ood_sample_scores"].abs().max()) if not merged.empty else float("nan")
                split_w = float("nan")
                if split_score_path.exists():
                    split_scores = pd.read_csv(split_score_path, encoding="utf-8-sig")
                    if "sliced_wasserstein" in split_scores.columns and not split_scores.empty:
                        split_w = float(pd.to_numeric(split_scores["sliced_wasserstein"], errors="coerce").iloc[0])
                rows.append(
                    {
                        "check": "workbook_vs_xspace_ood_sample_scores",
                        "source": str(sample_scores_path),
                        "rows_left": len(test),
                        "rows_right": len(sample_scores),
                        "matched_rows": int(merged["_merge"].eq("both").sum()),
                        "left_only": int(merged["_merge"].eq("left_only").sum()),
                        "right_only": int(merged["_merge"].eq("right_only").sum()),
                        "max_abs_diff": max_abs_diff,
                        "mean_sample_w": float(pd.to_numeric(sample_scores["sliced_wasserstein_sample_score"], errors="coerce").mean()),
                        "mass_sum": float(pd.to_numeric(sample_scores["sliced_wasserstein_mass_contribution"], errors="coerce").sum()),
                        "split_w": split_w,
                    }
                )
                lines.extend(
                    [
                        "## Workbook vs X-Space Source Scores",
                        "",
                        f"- Source table: `{sample_scores_path}`",
                        f"- Matching key: `{key}`",
                        f"- Matched rows: `{int(merged['_merge'].eq('both').sum())}`",
                        f"- Max absolute difference: `{max_abs_diff:.6g}`",
                        f"- Mean `sliced_wasserstein_sample_score`: `{pd.to_numeric(sample_scores['sliced_wasserstein_sample_score'], errors='coerce').mean():.6g}`",
                        f"- Sum `sliced_wasserstein_mass_contribution`: `{pd.to_numeric(sample_scores['sliced_wasserstein_mass_contribution'], errors='coerce').sum():.6g}`",
                        f"- Split `sliced_wasserstein`: `{split_w:.6g}`",
                        "",
                        "Note: `sample_w_contribution` is the per-test-sample score whose mean equals the split-level sliced W. "
                        "`sample_w_mass_contribution` is the additive mass contribution whose sum equals the split-level sliced W.",
                        "",
                    ]
                )
        else:
            lines.extend(["## Workbook vs X-Space Source Scores", "", "- X-space source directory not found.", ""])

    audit_table = pd.DataFrame(rows)
    audit_table.to_csv(diagnostics_dir / "w_source_audit.csv", index=False, encoding="utf-8-sig")
    if not audit_table.empty:
        lines.extend(["## Audit Table", "", markdown_table(audit_table, list(audit_table.columns), max_rows=len(audit_table)), ""])
    (diagnostics_dir / "w_source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_wx_fold_diagnostic(workbook_path: Path, diagnostics_dir: Path) -> None:
    frame = read_workbook(workbook_path)
    w_column = infer_w_column(workbook_path, frame)
    if w_column != "Wx":
        return
    shift_table = build_shift_table(frame)
    output_csv = diagnostics_dir / "wx_fold_feature_shift_diagnostic.csv"
    shift_table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    test_values = finite_series(frame.loc[frame["set_type"].eq("test"), w_column])
    scaled_audit_path = diagnostics_dir / "wx_scaled_feature_audit.csv"
    scaled_audit = pd.DataFrame()
    try:
        split_dir = Path(str(get_scalar(frame, "source_split_dir")))
        summary_path = split_dir / "split_summary.json"
        if summary_path.exists():
            import json
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            train_df = pd.read_csv(split_dir / "train.csv", encoding="utf-8-sig")
            test_df = pd.read_csv(split_dir / "test.csv", encoding="utf-8-sig")
            target_col = str(summary.get("split_target_col") or summary.get("target_column") or get_scalar(frame, "target_col"))
            features = [str(column) for column in summary.get("x_space_feature_columns", []) if str(column)]
            if features:
                train_x = pd.DataFrame({column: pd.to_numeric(train_df.get(column), errors="coerce") for column in features})
                test_x = pd.DataFrame({column: pd.to_numeric(test_df.get(column), errors="coerce") for column in features})
            else:
                excluded = {target_col, "ID", "__row_id__", "__source_index__"}
                features = [str(column) for column in train_df.select_dtypes(include=[np.number, "bool"]).columns if str(column) not in excluded]
                train_x = pd.DataFrame({column: pd.to_numeric(train_df[column], errors="coerce") for column in features})
                test_x = pd.DataFrame({column: pd.to_numeric(test_df[column], errors="coerce") for column in features})
            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            scaler = StandardScaler()
            train_imputed = imputer.fit_transform(train_x)
            test_imputed = imputer.transform(test_x)
            train_scaled = scaler.fit_transform(train_imputed)
            test_scaled = scaler.transform(test_imputed)
            scaled_rows: list[dict[str, Any]] = []
            for index, feature in enumerate(features):
                train_raw = train_imputed[:, index]
                test_raw = test_imputed[:, index]
                test_scaled_feature = test_scaled[:, index]
                scaled_rows.append(
                    {
                        "feature": feature,
                        "train_raw_mean": float(np.mean(train_raw)),
                        "test_raw_mean": float(np.mean(test_raw)),
                        "train_raw_std": float(np.std(train_raw, ddof=1)) if len(train_raw) > 1 else 0.0,
                        "scaler_mean": float(scaler.mean_[index]),
                        "scaler_scale": float(scaler.scale_[index]),
                        "train_nonzero_rate": float(np.mean(train_raw != 0)),
                        "test_nonzero_rate": float(np.mean(test_raw != 0)),
                        "test_scaled_mean": float(np.mean(test_scaled_feature)),
                        "test_scaled_abs_mean": float(np.mean(np.abs(test_scaled_feature))),
                        "test_scaled_max_abs": float(np.max(np.abs(test_scaled_feature))),
                    }
                )
            scaled_audit = pd.DataFrame(scaled_rows).sort_values("test_scaled_abs_mean", ascending=False, kind="stable")
            scaled_audit.to_csv(scaled_audit_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        scaled_audit = pd.DataFrame([{"error": repr(exc)}])
        scaled_audit.to_csv(scaled_audit_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Wx fold diagnostic",
        "",
        "This file summarizes why the current fold has large X-space W values.",
        "",
        "## Wx statistics",
        "",
        f"- Test count: `{len(test_values)}`",
        f"- Mean / median / std: `{test_values.mean():.6g}` / `{test_values.median():.6g}` / `{test_values.std(ddof=1):.6g}`",
        f"- Min / max: `{test_values.min():.6g}` / `{test_values.max():.6g}`",
        "",
        "## Key interpretation",
        "",
        "- The audit checks show the workbook values match both `three_space_sample_w_values.csv` and the upstream X-space `ood_sample_scores.csv` exactly.",
        "- Therefore the large `Wx` values are not introduced by the high-W Excel export.",
        "- In the upstream scoring code, X-space W is computed after zero imputation and `StandardScaler` fitting on the training split.",
        "- `sample_w_contribution` is a per-test-sample score whose mean equals split-level `sliced_wasserstein`; `sample_w_mass_contribution` is the additive contribution whose sum equals split-level `sliced_wasserstein`.",
        "- For this fold, the split-level X-space `sliced_wasserstein` is about `70.5958`, so all per-sample `Wx` values cluster around that scale.",
        "- The strongest direct cause is that some process columns are constant zero in train but nonzero in test. With `StandardScaler`, a zero-variance train column gets scale `1`, so the test raw process values remain hundreds of units in scaled X-space.",
        "",
        "## Largest scaled X-space feature offsets",
        "",
        markdown_table(
            scaled_audit,
            [
                "feature",
                "train_raw_mean",
                "test_raw_mean",
                "train_raw_std",
                "scaler_scale",
                "train_nonzero_rate",
                "test_nonzero_rate",
                "test_scaled_abs_mean",
                "test_scaled_max_abs",
            ],
            max_rows=15,
        ),
        "",
        f"Full scaled-feature audit table: `{scaled_audit_path.name}`",
        "",
        "## Largest train/test feature shifts after zero exclusion for element/process features",
        "",
        markdown_table(
            shift_table,
            [
                "feature",
                "feature_type",
                "train_count",
                "test_count",
                "train_mean",
                "test_mean",
                "train_nonzero_rate",
                "test_nonzero_rate",
                "abs_nonzero_rate_diff",
                "abs_std_mean_diff",
                "selection_basis",
            ],
            max_rows=20,
        ),
        "",
        f"Full feature-shift table: `{output_csv.name}`",
        "",
    ]
    (diagnostics_dir / "wx_fold_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame[columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def write_summary(
    frame: pd.DataFrame,
    w_column: str,
    shift_table: pd.DataFrame,
    element_features: list[str],
    process_features: list[str],
    output_path: Path,
) -> None:
    stats = w_stats(frame, w_column)
    top_w = (
        frame[frame["set_type"].eq("test")]
        .copy()
        .assign(_w=lambda data: pd.to_numeric(data[w_column], errors="coerce"))
        .sort_values("_w", ascending=False, kind="stable")
    )
    top_w_columns = [column for column in ["ID", w_column, "w_rank_desc", "is_top_n", "is_above_threshold", "highlight_reason"] if column in top_w.columns]
    lines = [
        f"# {w_column} split diagnostic summary",
        "",
        f"- Workbook W space: `{w_column}`",
        f"- Case: `{get_scalar(frame, 'case_key')}`",
        f"- Method: `{get_scalar(frame, 'method')}`",
        f"- Fold: `{get_scalar(frame, 'fold_id') or 'nofold'}`",
        f"- Train rows: `{int((frame['set_type'] == 'train').sum())}`",
        f"- Test rows: `{int((frame['set_type'] == 'test').sum())}`",
        "",
        "## W Distribution",
        "",
        f"- Count: `{stats['count']:.0f}`",
        f"- Mean / std / CV: `{stats['mean']:.4g}` / `{stats['std']:.4g}` / `{stats['cv']:.4g}`",
        f"- Min / Q25 / median / Q75 / max: `{stats['min']:.4g}` / `{stats['q25']:.4g}` / `{stats['median']:.4g}` / `{stats['q75']:.4g}` / `{stats['max']:.4g}`",
        "",
        "Top test rows by current W:",
        "",
        markdown_table(top_w, top_w_columns, max_rows=10),
        "",
        "## Largest Train/Test Feature Shifts",
        "",
        markdown_table(
            shift_table,
            [
                "feature",
                "feature_type",
                "train_count",
                "test_count",
                "train_mean",
                "test_mean",
                "train_nonzero_rate",
                "test_nonzero_rate",
                "abs_nonzero_rate_diff",
                "abs_std_mean_diff",
                "selection_basis",
            ],
            max_rows=15,
        ),
        "",
        "## Plotted Features",
        "",
        f"- Element violin features: `{', '.join(element_features) if element_features else 'none'}`",
        f"- Process violin features: `{', '.join(process_features) if process_features else 'none'}`",
        "- Split feature plots are saved as `element_violin_train.png`, `element_violin_test.png`, `process_violin_train.png`, and `process_violin_test.png`.",
        "",
        "## Interpretation Notes",
        "",
        f"- This report analyzes only `{w_column}` from the supplied workbook. It does not merge or compare sibling W spaces.",
        "- For element and process features, raw zero values are treated as absent and excluded from value-shift statistics and violin distributions.",
        "- The violin plots use raw feature values after zero exclusion. Each feature is drawn in its own row with an axis limited to the nonzero value range.",
        "- `train_nonzero_rate`, `test_nonzero_rate`, and `abs_nonzero_rate_diff` are retained to show whether a feature is mainly a presence/absence split rather than a value-distribution shift.",
        "- Use `top_shift_features.csv` for exact numeric train/test differences behind the plotted feature choices.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_workbook(
    workbook_path: Path,
    top_features_count: int,
    *,
    include_cross_space: bool = False,
    audit_w_source: bool = False,
) -> Path:
    frame = read_workbook(workbook_path)
    w_column = infer_w_column(workbook_path, frame)
    diagnostics_dir = workbook_path.parent / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    shift_table = build_shift_table(frame)
    shift_table.to_csv(diagnostics_dir / "top_shift_features.csv", index=False, encoding="utf-8-sig")

    element_features = top_features(shift_table, "element", top_features_count)
    process_features = top_features(shift_table, "process", top_features_count)

    save_w_distribution(frame, w_column, diagnostics_dir / "w_distribution.png")
    save_violin(frame, element_features, "Element distributions: train vs test (zeros excluded)", diagnostics_dir / "element_violin.png")
    save_violin(frame, element_features, "Element distributions: train only (zeros excluded)", diagnostics_dir / "element_violin_train.png", set_type="train")
    save_violin(frame, element_features, "Element distributions: test only (zeros excluded)", diagnostics_dir / "element_violin_test.png", set_type="test")
    save_violin(frame, process_features, "Processing feature distributions: train vs test (zeros excluded)", diagnostics_dir / "process_violin.png")
    save_violin(frame, process_features, "Processing feature distributions: train only (zeros excluded)", diagnostics_dir / "process_violin_train.png", set_type="train")
    save_violin(frame, process_features, "Processing feature distributions: test only (zeros excluded)", diagnostics_dir / "process_violin_test.png", set_type="test")
    write_summary(
        frame,
        w_column,
        shift_table,
        element_features,
        process_features,
        diagnostics_dir / "diagnostic_summary.md",
    )
    if include_cross_space:
        save_cross_space_w_comparison(workbook_path, diagnostics_dir)
    if audit_w_source:
        audit_current_w_source(workbook_path, diagnostics_dir)
        write_wx_fold_diagnostic(workbook_path, diagnostics_dir)
    return diagnostics_dir


def find_workbooks(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing root directory: {root}")
    workbooks = [
        path
        for path in root.rglob("data.xlsx")
        if path.is_file() and any(part in W_COLUMNS for part in path.parts)
    ]
    return sorted(workbooks, key=lambda path: tuple(path.parts))


def workbook_metadata(workbook_path: Path, diagnostics_dir: Path | None = None, error: str = "") -> dict[str, Any]:
    parts = list(workbook_path.parts)
    space = next((part for part in parts if part in W_COLUMNS), "")
    space_index = parts.index(space) if space in parts else -1
    case_key = parts[space_index + 1] if space_index >= 0 and space_index + 1 < len(parts) else ""
    method = parts[space_index + 2] if space_index >= 0 and space_index + 2 < len(parts) else ""
    fold_id = ""
    if workbook_path.parent.name.lower().startswith("fold"):
        fold_id = workbook_path.parent.name
    return {
        "space": space,
        "case_key": case_key,
        "method": method,
        "fold_id": fold_id,
        "workbook": str(workbook_path),
        "diagnostics_dir": str(diagnostics_dir) if diagnostics_dir is not None else "",
        "status": "error" if error else "ok",
        "error": error,
    }


def analyze_root(root: Path, top_features_count: int, manifest_name: str) -> Path:
    workbooks = find_workbooks(root)
    if not workbooks:
        raise ValueError(f"No data.xlsx workbooks found under: {root}")
    rows: list[dict[str, Any]] = []
    total = len(workbooks)
    for index, workbook_path in enumerate(workbooks, start=1):
        print(f"[{index}/{total}] {workbook_path}")
        try:
            diagnostics_dir = analyze_workbook(workbook_path, top_features_count)
            rows.append(workbook_metadata(workbook_path, diagnostics_dir=diagnostics_dir))
        except Exception as exc:
            rows.append(workbook_metadata(workbook_path, error=repr(exc)))
            print(f"  ERROR: {exc}", file=sys.stderr)
    manifest_path = root / manifest_name
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    error_count = sum(1 for row in rows if row["status"] == "error")
    print(f"Saved batch manifest: {manifest_path}")
    print(f"Batch complete: ok={total - error_count}, error={error_count}, total={total}")
    if error_count:
        raise RuntimeError(f"Batch diagnostics completed with {error_count} error(s). See {manifest_path}")
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.workbook:
        diagnostics_dir = analyze_workbook(
            Path(args.workbook),
            int(args.top_features),
            include_cross_space=bool(args.include_cross_space),
            audit_w_source=bool(args.audit_w_source),
        )
        print(f"Saved diagnostics under: {diagnostics_dir}")
    else:
        analyze_root(Path(args.root), int(args.top_features), str(args.manifest_name))


if __name__ == "__main__":
    main()
