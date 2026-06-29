from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SPACE_ORDER = ["X-space", "Y-space", "Z-space"]
ERROR_METRICS = ["abs_error", "relative_error_pct"]
ERROR_METRIC_LABELS = {
    "abs_error": "MAE slope",
    "relative_error_pct": "Relative error slope",
}
METHOD_ORDER = ["RandCV", "Extra.", "LOCO", "SX-sgl", "SX-cls", "SY-sgl", "SY-cls"]
HYBRID_SUBSETS = ("combined", "test_extrapolation_high20", "test_inner_ood")
INPUT_COLUMNS = [
    "scope",
    "task_key",
    "task_id",
    "method_short",
    "model",
    "space",
    "ID",
    "sample_order",
    "source_split_dir",
    "split_w",
    "sample_w_contribution",
    "abs_error",
    "relative_error_pct",
]
OOD_SEVERITY_COLUMNS = [
    "scope",
    "subset",
    "task_key",
    "task_id",
    "method_short",
    "space",
    "ood_severity",
    "ood_severity_n_samples",
    "ood_severity_n_splits",
    "ood_severity_status",
]
OOD_SEVERITY_METRICS = [
    "ood_severity",
    "ood_severity_n_samples",
    "ood_severity_n_splits",
    "ood_severity_status",
]
SLOPE_COLUMNS = [
    "scope",
    "subset",
    "task_key",
    "task_id",
    "method_short",
    "model",
    "space",
    "error_metric",
    "slope",
    "intercept",
    "n_points",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "slope_status",
    "ood_severity",
    "ood_severity_n_samples",
    "ood_severity_n_splits",
    "ood_severity_status",
]
BY_TASK_METHOD_COLUMNS = [
    "scope",
    "subset",
    "task_key",
    "task_id",
    "method_short",
    "model",
    "space",
    "error_metric",
    "slope",
    "intercept",
    "n_points",
    "slope_status",
    "ood_severity",
    "ood_severity_n_samples",
    "ood_severity_n_splits",
    "ood_severity_status",
]
BY_TASK_MODEL_COLUMNS = [
    "scope",
    "subset",
    "task_key",
    "task_id",
    "model",
    "method_short",
    "space",
    "error_metric",
    "slope",
    "intercept",
    "n_points",
    "slope_status",
    "ood_severity",
    "ood_severity_n_samples",
    "ood_severity_n_splits",
    "ood_severity_status",
]
FIGURE_MANIFEST_COLUMNS = ["figure_group", "scope", "subset", "task_id", "method_short", "model", "figure", "format"]
METHOD_COLORS = {
    "RandCV": "#4c78a8",
    "Extra.": "#f58518",
    "LOCO": "#54a24b",
    "SX-sgl": "#b279a2",
    "SX-cls": "#e45756",
    "SY-sgl": "#72b7b2",
    "SY-cls": "#ff9da6",
}


def final_results_root() -> Path:
    return (
        Path("D:/XJTU")
        / "\u5df2\u5b8c\u6210\u8bba\u6587\u6570\u636e\u6c47\u603b"
        / "Fewshot"
        / "\u9884\u5904\u7406\u6c47\u603b\u6570\u636e"
        / "\u6700\u7ec8\u7ed3\u679c\u56fe"
    )


DEFAULT_OOD_OUTPUT = final_results_root() / "OOD" / "w_error_relationship"
DEFAULT_HYBRID_OUTPUT = final_results_root() / "OOD HYBIRD" / "w_error_relationship"


def parse_max_relative_error_pct(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    try:
        threshold = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-relative-error-pct must be a non-negative number or none") from exc
    if threshold < 0:
        raise argparse.ArgumentTypeError("--max-relative-error-pct must be non-negative")
    return threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize W-error linear trend slopes from generated long tables.")
    parser.add_argument("--scope", choices=["ood", "hybrid", "both"], default="both")
    parser.add_argument("--ood-output", default=str(DEFAULT_OOD_OUTPUT))
    parser.add_argument("--hybrid-output", default=str(DEFAULT_HYBRID_OUTPUT))
    parser.add_argument("--output-suffix", default="", help="Append suffix to output directory names for smoke runs.")
    parser.add_argument("--case-contains", default=None, help="Optional task_id/task_key filter for smoke runs.")
    parser.add_argument("--max-relative-error-pct", type=parse_max_relative_error_pct, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args(argv)


def output_with_suffix(path: Path, suffix: str) -> Path:
    suffix = str(suffix or "").strip()
    if not suffix:
        return path
    return path.with_name(f"{path.name}{suffix}")


def read_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, usecols=usecols)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, usecols=usecols)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def safe_filename(text: object) -> str:
    value = "" if text is None else str(text).strip()
    value = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "unknown"


def ordered_methods(methods: Iterable[object]) -> list[str]:
    values = [str(value) for value in pd.Series(list(methods)).dropna().unique()]
    return sorted(values, key=lambda item: METHOD_ORDER.index(item) if item in METHOD_ORDER else 999)


def filter_by_relative_error(frame: pd.DataFrame, max_relative_error_pct: float | None) -> pd.DataFrame:
    if max_relative_error_pct is None or frame.empty or "relative_error_pct" not in frame.columns:
        return frame.copy()
    relative_error = pd.to_numeric(frame["relative_error_pct"], errors="coerce")
    return frame[relative_error.isna() | relative_error.le(max_relative_error_pct)].copy()


def filter_by_case(frame: pd.DataFrame, case_contains: str | None) -> pd.DataFrame:
    if not case_contains or frame.empty:
        return frame.copy()
    needle = str(case_contains).lower()
    task_id = frame.get("task_id", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    task_key = frame.get("task_key", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    return frame[task_id.str.contains(needle, regex=False) | task_key.str.contains(needle, regex=False)].copy()


def fit_slope(x: Iterable[object], y: Iterable[object]) -> dict[str, object]:
    x_values = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    n_points = int(valid.sum())
    base = {
        "slope": np.nan,
        "intercept": np.nan,
        "n_points": n_points,
        "x_min": np.nan,
        "x_max": np.nan,
        "y_min": np.nan,
        "y_max": np.nan,
        "slope_status": "insufficient_points",
    }
    if n_points == 0:
        return base
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    base.update(
        {
            "x_min": float(np.min(x_valid)),
            "x_max": float(np.max(x_valid)),
            "y_min": float(np.min(y_valid)),
            "y_max": float(np.max(y_valid)),
        }
    )
    if n_points < 2:
        return base
    if np.unique(x_valid).size < 2:
        base["slope_status"] = "constant_w"
        return base
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        base["slope_status"] = "invalid_fit"
        return base
    base.update({"slope": float(slope), "intercept": float(intercept), "slope_status": "ok"})
    return base


def fit_severity_trend(x: Iterable[object], y: Iterable[object]) -> dict[str, float] | None:
    x_values = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2:
        return None
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if np.unique(x_valid).size < 2:
        return None
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "x_min": float(np.min(x_valid)),
        "x_max": float(np.max(x_valid)),
    }


def sample_identity(frame: pd.DataFrame) -> pd.Series:
    index_key = pd.Series(frame.index.astype(str), index=frame.index, dtype="object")
    if "sample_order" in frame.columns:
        sample_order = frame["sample_order"]
        sample_order_key = sample_order.where(sample_order.notna()).astype("object")
        sample_order_key = sample_order_key.astype(str)
        sample_order_key = sample_order_key.where(sample_order.notna() & sample_order_key.ne(""), index_key)
    else:
        sample_order_key = index_key
    if "ID" in frame.columns:
        sample_id = frame["ID"]
        sample_id_key = sample_id.where(sample_id.notna()).astype("object")
        sample_id_key = sample_id_key.astype(str)
        return sample_id_key.where(sample_id.notna() & sample_id_key.ne(""), sample_order_key)
    return sample_order_key


def summarize_ood_severity_group(group: pd.DataFrame) -> dict[str, object]:
    if "split_w" not in group.columns:
        return {
            "ood_severity": np.nan,
            "ood_severity_n_samples": 0,
            "ood_severity_n_splits": 0,
            "ood_severity_status": "missing_split_w",
        }
    work = group.copy()
    work["_split_w_numeric"] = pd.to_numeric(work["split_w"], errors="coerce")
    work = work[np.isfinite(work["_split_w_numeric"].to_numpy(dtype=float, na_value=np.nan))].copy()
    if work.empty:
        return {
            "ood_severity": np.nan,
            "ood_severity_n_samples": 0,
            "ood_severity_n_splits": 0,
            "ood_severity_status": "missing_split_w",
        }
    work["_source_split_key"] = (
        work.get("source_split_dir", pd.Series("", index=work.index)).fillna("").astype(str).replace("", "__missing_split_dir__")
    )
    work["_sample_key"] = sample_identity(work)
    dedup_samples = work.drop_duplicates(["_source_split_key", "space", "_split_w_numeric", "_sample_key"])
    split_summary = (
        dedup_samples.groupby(["_source_split_key", "space", "_split_w_numeric"], dropna=False, sort=True)["_sample_key"]
        .nunique()
        .reset_index(name="n_samples")
    )
    split_summary = split_summary[split_summary["n_samples"].gt(0)].copy()
    if split_summary.empty:
        return {
            "ood_severity": np.nan,
            "ood_severity_n_samples": 0,
            "ood_severity_n_splits": 0,
            "ood_severity_status": "missing_split_w",
        }
    weights = pd.to_numeric(split_summary["n_samples"], errors="coerce").to_numpy(dtype=float)
    split_w = pd.to_numeric(split_summary["_split_w_numeric"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(weights) & np.isfinite(split_w) & (weights > 0)
    if not finite.any():
        return {
            "ood_severity": np.nan,
            "ood_severity_n_samples": 0,
            "ood_severity_n_splits": 0,
            "ood_severity_status": "missing_split_w",
        }
    weights = weights[finite]
    split_w = split_w[finite]
    return {
        "ood_severity": float(np.average(split_w, weights=weights)),
        "ood_severity_n_samples": int(weights.sum()),
        "ood_severity_n_splits": int(finite.sum()),
        "ood_severity_status": "ok",
    }


def build_ood_severity_table(frame: pd.DataFrame, *, subset: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=OOD_SEVERITY_COLUMNS)
    group_cols = ["scope", "task_key", "task_id", "method_short", "space"]
    work = frame.copy()
    for column in group_cols:
        if column not in work.columns:
            work[column] = ""
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        row_base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rows.append({**row_base, "subset": subset, **summarize_ood_severity_group(group)})
    return pd.DataFrame(rows, columns=OOD_SEVERITY_COLUMNS)


def build_slope_table(frame: pd.DataFrame, *, subset: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SLOPE_COLUMNS)
    rows: list[dict[str, object]] = []
    group_cols = ["scope", "task_key", "task_id", "method_short", "model", "space"]
    work = frame.copy()
    for column in group_cols:
        if column not in work.columns:
            work[column] = ""
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        row_base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for metric in ERROR_METRICS:
            result = fit_slope(group["sample_w_contribution"], group[metric])
            rows.append({**row_base, "subset": subset, "error_metric": metric, **result})
    slopes = pd.DataFrame(rows)
    severity = build_ood_severity_table(work, subset=subset)
    merge_cols = ["scope", "subset", "task_key", "task_id", "method_short", "space"]
    slopes = slopes.merge(severity, on=merge_cols, how="left")
    for column in OOD_SEVERITY_METRICS:
        if column not in slopes.columns:
            slopes[column] = np.nan if column != "ood_severity_status" else "missing_split_w"
    return slopes.reindex(columns=SLOPE_COLUMNS)


def build_comparison_tables(slopes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if slopes.empty:
        return pd.DataFrame(columns=BY_TASK_METHOD_COLUMNS), pd.DataFrame(columns=BY_TASK_MODEL_COLUMNS)
    by_task_method = slopes[BY_TASK_METHOD_COLUMNS].sort_values(
        ["scope", "subset", "task_id", "method_short", "model", "error_metric", "space"], kind="stable"
    )
    by_task_model = slopes[BY_TASK_MODEL_COLUMNS].sort_values(
        ["scope", "subset", "task_id", "model", "method_short", "error_metric", "space"], kind="stable"
    )
    return by_task_method.reset_index(drop=True), by_task_model.reset_index(drop=True)


def slope_axis(ax: plt.Axes, panel: pd.DataFrame, x_column: str, x_values: list[str], space: str, metric: str) -> None:
    values = []
    for value in x_values:
        match = panel[
            panel[x_column].astype(str).eq(value)
            & panel["space"].astype(str).eq(space)
            & panel["error_metric"].astype(str).eq(metric)
            & panel["slope_status"].astype(str).eq("ok")
        ]
        slope = pd.to_numeric(match["slope"], errors="coerce").dropna()
        values.append(float(slope.iloc[0]) if not slope.empty else np.nan)
    positions = np.arange(len(x_values))
    ax.axhline(0.0, color="#555555", linewidth=0.9, alpha=0.75)
    finite = np.isfinite(np.array(values, dtype=float))
    colors = ["#4c78a8" if value >= 0 else "#e45756" for value in np.nan_to_num(values, nan=0.0)]
    ax.bar(positions[finite], np.array(values, dtype=float)[finite], color=np.array(colors, dtype=object)[finite], alpha=0.82)
    if not finite.any():
        ax.text(0.5, 0.5, "No valid slope", ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_title(f"{space} / {ERROR_METRIC_LABELS[metric]}", loc="left", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("slope")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def severity_slope_axis(ax: plt.Axes, panel: pd.DataFrame, space: str, metric: str) -> None:
    match = panel[
        panel["space"].astype(str).eq(space)
        & panel["error_metric"].astype(str).eq(metric)
        & panel["slope_status"].astype(str).eq("ok")
        & panel["ood_severity_status"].astype(str).eq("ok")
    ].copy()
    match["ood_severity"] = pd.to_numeric(match["ood_severity"], errors="coerce")
    match["slope"] = pd.to_numeric(match["slope"], errors="coerce")
    finite = np.isfinite(match["ood_severity"].to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(
        match["slope"].to_numpy(dtype=float, na_value=np.nan)
    )
    match = match.loc[finite].sort_values(["ood_severity", "method_short"], kind="stable")

    ax.axhline(0.0, color="#555555", linewidth=0.9, alpha=0.75)
    if match.empty:
        ax.text(0.5, 0.5, "No valid severity/slope", ha="center", va="center", transform=ax.transAxes, color="#666666")
    else:
        for _, row in match.iterrows():
            method = str(row.get("method_short", ""))
            color = METHOD_COLORS.get(method, "#4c78a8")
            x_value = float(row["ood_severity"])
            y_value = float(row["slope"])
            ax.scatter([x_value], [y_value], s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(
                method,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#333333",
            )
        trend = fit_severity_trend(match["ood_severity"], match["slope"])
        if trend is not None:
            x_line = np.linspace(trend["x_min"], trend["x_max"], 100)
            y_line = trend["slope"] * x_line + trend["intercept"]
            ax.plot(x_line, y_line, color="#222222", linewidth=1.4, linestyle="-", alpha=0.85)
    ax.set_title(f"{space} / {ERROR_METRIC_LABELS[metric]}", loc="left", fontsize=10)
    ax.set_xlabel("OOD severity (split W)")
    ax.set_ylabel("slope")
    ax.grid(True, axis="both", color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_slope_comparison_grid(
    group: pd.DataFrame,
    *,
    title: str,
    x_column: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[dict[str, object]]:
    if x_column == "method_short":
        x_values = []
    else:
        x_values = sorted(str(value) for value in group[x_column].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for row_idx, metric in enumerate(ERROR_METRICS):
        for col_idx, space in enumerate(SPACE_ORDER):
            if x_column == "method_short":
                severity_slope_axis(axes[row_idx, col_idx], group, space, metric)
            else:
                slope_axis(axes[row_idx, col_idx], group, x_column, x_values, space, metric)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    first = group.iloc[0]
    for fmt in formats:
        path = output_base.parent / f"{output_base.name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        rows.append(
            {
                "scope": first.get("scope", ""),
                "subset": first.get("subset", ""),
                "task_id": first.get("task_id", ""),
                "method_short": first.get("method_short", np.nan) if x_column == "model" else np.nan,
                "model": first.get("model", np.nan) if x_column == "method_short" else np.nan,
                "figure": str(path),
                "format": fmt,
            }
        )
    plt.close(fig)
    return rows


def plot_slope_comparisons(slopes: pd.DataFrame, output_dir: Path, formats: Iterable[str], dpi: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if slopes.empty:
        return pd.DataFrame(columns=FIGURE_MANIFEST_COLUMNS)
    for (task_id, method), group in slopes.groupby(["task_id", "method_short"], dropna=False, sort=True):
        base = (
            output_dir
            / "figures"
            / "slope_comparison"
            / "by_task_method"
            / safe_filename(task_id)
            / f"{safe_filename(method)}_slope_by_model"
        )
        for row in plot_slope_comparison_grid(
            group,
            title=f"{task_id} | {method} | slope by model",
            x_column="model",
            output_base=base,
            formats=formats,
            dpi=dpi,
        ):
            rows.append({"figure_group": "slope_by_task_method", **row})
    for (task_id, model), group in slopes.groupby(["task_id", "model"], dropna=False, sort=True):
        base = (
            output_dir
            / "figures"
            / "slope_comparison"
            / "by_task_model"
            / safe_filename(task_id)
            / f"{safe_filename(model)}_slope_by_method"
        )
        for row in plot_slope_comparison_grid(
            group,
            title=f"{task_id} | {model} | slope by OOD severity",
            x_column="method_short",
            output_base=base,
            formats=formats,
            dpi=dpi,
        ):
            rows.append({"figure_group": "slope_by_task_model", **row})
    return pd.DataFrame(rows, columns=FIGURE_MANIFEST_COLUMNS)


def write_report(output_dir: Path, slopes: pd.DataFrame, manifest: pd.DataFrame) -> None:
    ok = int(slopes["slope_status"].astype(str).eq("ok").sum()) if "slope_status" in slopes.columns else 0
    lines = [
        "# W-error trend slope summary",
        "",
        f"- Slope rows: {len(slopes)}",
        f"- Valid slope rows: {ok}",
        f"- Figure files: {len(manifest)}",
        "",
        "## Files",
        "",
        "- `csv/w_error_trend_slopes_long.csv`",
        "- `csv/w_error_trend_slopes_by_task_method.csv`",
        "- `csv/w_error_trend_slopes_by_task_model.csv`",
        "- `csv/slope_figure_manifest.csv`",
        "- `figures/slope_comparison/by_task_method/`",
        "- `figures/slope_comparison/by_task_model/`",
    ]
    (output_dir / "slope_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_output(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    subset: str,
    formats: list[str],
    dpi: int,
    max_relative_error_pct: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = filter_by_relative_error(frame, max_relative_error_pct)
    slopes = build_slope_table(filtered, subset=subset)
    by_task_method, by_task_model = build_comparison_tables(slopes)
    manifest = plot_slope_comparisons(slopes, output_dir, formats, dpi)
    write_csv(slopes, output_dir / "csv" / "w_error_trend_slopes_long.csv")
    write_csv(by_task_method, output_dir / "csv" / "w_error_trend_slopes_by_task_method.csv")
    write_csv(by_task_model, output_dir / "csv" / "w_error_trend_slopes_by_task_model.csv")
    write_csv(manifest, output_dir / "csv" / "slope_figure_manifest.csv")
    write_report(output_dir, slopes, manifest)


def read_long_table(output_dir: Path, *, case_contains: str | None) -> pd.DataFrame:
    path = output_dir / "csv" / "w_error_samples_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing W-error long table: {path}")
    frame = read_csv(path, usecols=INPUT_COLUMNS)
    return filter_by_case(frame, case_contains)


def run_standard(args: argparse.Namespace) -> None:
    output_dir = output_with_suffix(Path(args.ood_output), args.output_suffix)
    source_dir = Path(args.ood_output)
    frame = read_long_table(source_dir, case_contains=args.case_contains)
    run_one_output(
        frame,
        output_dir,
        subset="standard",
        formats=args.formats,
        dpi=args.dpi,
        max_relative_error_pct=args.max_relative_error_pct,
    )


def run_hybrid(args: argparse.Namespace) -> None:
    source_root = Path(args.hybrid_output)
    output_root = output_with_suffix(source_root, args.output_suffix)
    for subset in HYBRID_SUBSETS:
        source_dir = source_root / subset
        output_dir = output_root / subset
        frame = read_long_table(source_dir, case_contains=args.case_contains)
        run_one_output(
            frame,
            output_dir,
            subset=subset,
            formats=args.formats,
            dpi=args.dpi,
            max_relative_error_pct=args.max_relative_error_pct,
        )


def main() -> None:
    args = parse_args()
    if args.scope in {"ood", "both"}:
        run_standard(args)
    if args.scope in {"hybrid", "both"}:
        run_hybrid(args)


if __name__ == "__main__":
    main()
