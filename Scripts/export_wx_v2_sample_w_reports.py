from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts import export_three_space_w_ood_report as three_space

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_WX_V2_ROOT = Path("output") / "ood_xspace_scores" / "wx_v2_mixed_ot"
DEFAULT_OUTPUT_ROOT = Path("output") / "ood_xspace_scores" / "sample_w_reports_v2"
DEFAULT_ORIGINAL_SAMPLE_W_REPORTS = Path("output") / "ood_xspace_scores" / "sample_w_reports"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sample_w_reports_v2 by replacing X-space sample W with Wx_v2 "
            "Mixed-OT outputs while reusing the existing Y/Z-space report flow."
        )
    )
    parser.add_argument("--wx-v2-root", default=str(DEFAULT_WX_V2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--original-sample-w-reports",
        default=str(DEFAULT_ORIGINAL_SAMPLE_W_REPORTS),
        help="Existing sample_w_reports directory used to locate ood_split_summary.csv by default.",
    )
    parser.add_argument(
        "--split-summary",
        default=None,
        help="X-space ood_split_summary.csv. Defaults to <original-sample-w-reports>/../ood_split_summary.csv.",
    )
    parser.add_argument("--yz-summary", default=str(three_space.DEFAULT_YZ_SUMMARY))
    parser.add_argument("--embedding-data-dir", default=str(three_space.DEFAULT_EMBEDDING_DATA_DIR))
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def required_sample_columns() -> list[str]:
    return [
        *CASE_COLUMNS,
        "method",
        "split_id",
        "fold_id",
        "source_split_dir",
        "__row_id__",
        "__source_index__",
        "target_col",
        "target_value",
        "wx_v2_sample_score",
        "wx_v2_mass_contribution",
        "wx_v2_rank_desc",
    ]


def validate_wx_v2_inputs(samples: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    missing_samples = [column for column in required_sample_columns() if column not in samples.columns]
    if missing_samples:
        raise ValueError(f"Wx_v2 sample table is missing required columns: {missing_samples}")
    if "source_split_dir" not in split_summary.columns or "wx_v2" not in split_summary.columns:
        raise ValueError("Wx_v2 split summary must contain source_split_dir and wx_v2.")


def case_label_from_row(row: pd.Series) -> str:
    return f"{row['alloy_family']} / {row['dataset_name']} / {row['property']}"


def build_v2_sample_w_table(samples: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in required_sample_columns() if column not in samples.columns]
    if missing:
        raise ValueError(f"Wx_v2 sample table is missing required columns: {missing}")

    columns = [
        *CASE_COLUMNS,
        "method",
        "split_id",
        "fold_id",
        "source_split_dir",
        "__row_id__",
        "__source_index__",
        "target_col",
        "target_value",
        "wx_v2_sample_score",
        "wx_v2_mass_contribution",
        "wx_v2_rank_desc",
    ]
    if "ID" in samples.columns:
        columns.insert(columns.index("target_col"), "ID")
    table = samples[columns].copy()
    table = table.rename(
        columns={
            "wx_v2_sample_score": "sample_w_contribution",
            "wx_v2_mass_contribution": "sample_w_mass_contribution",
            "wx_v2_rank_desc": "sample_w_rank_desc",
        }
    )
    table["ood_score"] = table["sample_w_contribution"]
    table["ood_percentile_vs_train"] = np.nan
    table["case_label"] = table.apply(case_label_from_row, axis=1)
    table["sample_w_contribution"] = pd.to_numeric(table["sample_w_contribution"], errors="coerce")
    table["sample_w_mass_contribution"] = pd.to_numeric(table["sample_w_mass_contribution"], errors="coerce")
    table["sample_w_rank_desc"] = pd.to_numeric(table["sample_w_rank_desc"], errors="coerce").astype("Int64")
    sort_cols = [*CASE_COLUMNS, "method", "fold_id", "sample_w_rank_desc"]
    return table.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def top10_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if finite.empty:
        return float("nan")
    top_n = max(1, int(math.ceil(len(finite) * 0.1)))
    return float(finite.tail(top_n).mean())


def ordered_methods(methods: Iterable[str]) -> list[str]:
    available = {str(method) for method in methods}
    known = [method for method in METHOD_ORDER if method in available]
    unknown = sorted(available - set(known))
    return known + unknown


def summarize_v2_sample_w(table: pd.DataFrame) -> pd.DataFrame:
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
    return summary.sort_values([*CASE_COLUMNS, "_method_order", "method"], kind="stable").drop(columns="_method_order")


def build_v2_split_summary_for_three_space(wx_v2_split_summary: pd.DataFrame, original_split_summary: pd.DataFrame) -> pd.DataFrame:
    split_w = wx_v2_split_summary[["source_split_dir", "wx_v2"]].copy()
    split_w["source_split_dir"] = split_w["source_split_dir"].astype(str)
    original = original_split_summary.copy()
    original["source_split_dir"] = original["source_split_dir"].astype(str)
    merged = original.drop(columns=["sliced_wasserstein"], errors="ignore").merge(split_w, on="source_split_dir", how="left")
    if merged["wx_v2"].isna().any():
        missing = int(merged["wx_v2"].isna().sum())
        raise ValueError(f"Missing Wx_v2 split values for {missing} original split rows.")
    merged["sliced_wasserstein"] = pd.to_numeric(merged["wx_v2"], errors="coerce")
    return merged.drop(columns=["wx_v2"])


def build_top_samples(sample_values: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in sample_values.groupby([*CASE_COLUMNS, "method"], sort=True, dropna=False):
        rows.append(group.sort_values("sample_w_contribution", ascending=False, na_position="last", kind="stable").head(top_n))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=sample_values.columns)


def combine_v2_x_with_cached_yz(v2_x_samples: pd.DataFrame, cached_three_space_samples: pd.DataFrame) -> pd.DataFrame:
    cached_yz = cached_three_space_samples[cached_three_space_samples["space"].isin(["Y-space", "Z-space"])].copy()
    combined = pd.concat([v2_x_samples, cached_yz], ignore_index=True, sort=False)
    space_order = {"X-space": 0, "Y-space": 1, "Z-space": 2}
    method_order = {method: idx for idx, method in enumerate(three_space.METHOD_ORDER)}
    combined["space_w_rank_desc"] = (
        combined.groupby(["case_key", "method", "space"], dropna=False)["sample_w_contribution"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    combined["_space_order"] = combined["space"].map(space_order).fillna(999)
    combined["_method_order"] = combined["method"].map(method_order).fillna(999)
    sort_columns = ["case_key", "_space_order", "_method_order", "method", "fold_id", "space_w_rank_desc"]
    existing_sort_columns = [column for column in sort_columns if column in combined.columns]
    return (
        combined.sort_values(existing_sort_columns, kind="stable")
        .drop(columns=["_space_order", "_method_order"])
        .reset_index(drop=True)
    )


def build_summary_from_cached_yz(x_summary: pd.DataFrame, cached_summary: pd.DataFrame) -> pd.DataFrame:
    x_metrics = three_space.build_x_metric_table(x_summary)
    summary = cached_summary.copy()
    drop_columns = [
        "x_sample_count",
        "X_mean_W",
        "X_median_W",
        "X_q25_W",
        "X_q75_W",
        "X_p90_W",
        "X_max_W",
        "X_top10pct_mean_W",
        "X_ratio",
        "X_top10pct_mean_ratio",
        "three_space_mean_ratio",
    ]
    summary = summary.drop(columns=[column for column in drop_columns if column in summary.columns], errors="ignore")
    summary = summary.merge(x_metrics, on=["case_key", "method"], how="left")
    summary["x_sample_count"] = pd.to_numeric(summary["x_sample_count"], errors="coerce")
    summary["yz_n_test"] = pd.to_numeric(summary.get("yz_n_test"), errors="coerce")
    summary["n_test"] = summary["x_sample_count"].where(summary["x_sample_count"].notna(), summary["yz_n_test"])

    for ratio_column, metric_column in [
        ("X_ratio", "X_median_W"),
        ("X_top10pct_mean_ratio", "X_top10pct_mean_W"),
    ]:
        summary[ratio_column] = np.nan
        for _, indices in summary.groupby("case_key").groups.items():
            group = summary.loc[indices]
            baseline = group.loc[group["method"] == "random_cv_baseline", metric_column]
            baseline_value = float(baseline.iloc[0]) if len(baseline) and pd.notna(baseline.iloc[0]) else float("nan")
            if not math.isfinite(baseline_value) or baseline_value <= 0:
                continue
            summary.loc[indices, ratio_column] = pd.to_numeric(group[metric_column], errors="coerce") / baseline_value

    for ratio_column in ["Y_ratio", "Z_ratio"]:
        if ratio_column not in summary.columns:
            summary[ratio_column] = np.nan
    summary["three_space_mean_ratio"] = summary[["Y_ratio", "Z_ratio", "X_ratio"]].mean(axis=1, skipna=True)
    return three_space.sort_summary_by_case_and_ood(summary)


def generate_reports(
    sample_values: pd.DataFrame,
    x_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    yz_summary: pd.DataFrame,
    embedding_data_dir: Path,
    output_root: Path,
    formats: list[str],
    dpi: int,
    top_n: int,
    yz_summary_path: Path,
) -> None:
    summary = three_space.sort_summary_by_case_and_ood(three_space.build_three_space_summary(x_summary, yz_summary))
    prepared_samples = three_space.prepare_sample_values(sample_values)
    top_samples = build_top_samples(prepared_samples, top_n)
    temp_split_summary_path = output_root / "_wx_v2_split_summary_for_three_space.csv"
    write_csv(split_summary, temp_split_summary_path)
    three_space_samples = three_space.build_three_space_sample_values(
        prepared_samples,
        temp_split_summary_path,
        embedding_data_dir,
    )
    three_space_sample_summary = three_space.summarize_three_space_sample_values(three_space_samples)

    write_csv(sample_values, output_root / "all_sample_w_values.csv")
    write_csv(x_summary, output_root / "case_method_w_summary.csv")
    write_csv(summary, output_root / "three_space_w_ratio_summary.csv")
    write_csv(top_samples, output_root / "xspace_top_w_samples.csv")
    write_csv(three_space_samples, output_root / "three_space_sample_w_values.csv")
    write_csv(three_space_sample_summary, output_root / "three_space_sample_w_summary.csv")

    for case_key, case_summary in summary.groupby("case_key", sort=False):
        case_dir = output_root / "cases" / case_key
        case_samples = three_space_samples[three_space_samples["case_key"] == case_key].copy()
        write_csv(case_summary, case_dir / "task_w_summary.csv")
        three_space.plot_task_ratio(case_summary, case_dir / "task_w_ratio", formats, dpi)
        three_space.plot_task_curves(case_summary, case_samples, case_dir / "task_w_curves", formats, dpi)
        three_space.remove_stale_task_dashboard_figures(case_dir)

    report = three_space.build_report(summary, top_samples, output_root, output_root, yz_summary_path)
    report_path = output_root / "three_space_w_ood_report.md"
    report_path.write_text(report, encoding="utf-8")

    temp_split_summary_path.unlink(missing_ok=True)


def generate_reports_from_cached_yz(
    sample_values: pd.DataFrame,
    x_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    original_sample_w_reports: Path,
    output_root: Path,
    formats: list[str],
    dpi: int,
    top_n: int,
    yz_summary_path: Path,
) -> None:
    cached_samples_path = original_sample_w_reports / "three_space_sample_w_values.csv"
    cached_summary_path = original_sample_w_reports / "three_space_w_ratio_summary.csv"
    if not cached_samples_path.exists():
        raise FileNotFoundError(f"Missing cached three-space sample table: {cached_samples_path}")
    if not cached_summary_path.exists():
        raise FileNotFoundError(f"Missing cached three-space summary table: {cached_summary_path}")

    prepared_samples = three_space.prepare_sample_values(sample_values)
    top_samples = build_top_samples(prepared_samples, top_n)
    v2_x_samples = three_space.x_space_long_samples(prepared_samples, split_summary)
    cached_three_space_samples = read_csv(cached_samples_path)
    three_space_samples = combine_v2_x_with_cached_yz(v2_x_samples, cached_three_space_samples)
    three_space_sample_summary = three_space.summarize_three_space_sample_values(three_space_samples)
    summary = build_summary_from_cached_yz(x_summary, read_csv(cached_summary_path))

    write_csv(sample_values, output_root / "all_sample_w_values.csv")
    write_csv(x_summary, output_root / "case_method_w_summary.csv")
    write_csv(summary, output_root / "three_space_w_ratio_summary.csv")
    write_csv(top_samples, output_root / "xspace_top_w_samples.csv")
    write_csv(three_space_samples, output_root / "three_space_sample_w_values.csv")
    write_csv(three_space_sample_summary, output_root / "three_space_sample_w_summary.csv")

    for case_key, case_summary in summary.groupby("case_key", sort=False):
        case_dir = output_root / "cases" / case_key
        case_samples = three_space_samples[three_space_samples["case_key"] == case_key].copy()
        write_csv(case_summary, case_dir / "task_w_summary.csv")
        three_space.plot_task_ratio(case_summary, case_dir / "task_w_ratio", formats, dpi)
        three_space.plot_task_curves(case_summary, case_samples, case_dir / "task_w_curves", formats, dpi)
        three_space.remove_stale_task_dashboard_figures(case_dir)

    report = three_space.build_report(summary, top_samples, output_root, output_root, yz_summary_path)
    report_path = output_root / "three_space_w_ood_report.md"
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    wx_v2_root = Path(args.wx_v2_root)
    output_root = Path(args.output_root)
    original_sample_w_reports = Path(args.original_sample_w_reports)
    split_summary_path = (
        Path(args.split_summary)
        if args.split_summary
        else original_sample_w_reports.parent / "ood_split_summary.csv"
    )
    yz_summary_path = Path(args.yz_summary)
    embedding_data_dir = Path(args.embedding_data_dir)

    sample_path = wx_v2_root / "all_wx_v2_sample_scores.csv"
    split_path = wx_v2_root / "wx_v2_split_summary.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing Wx_v2 sample table: {sample_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing Wx_v2 split summary: {split_path}")
    if not split_summary_path.exists():
        raise FileNotFoundError(f"Missing original split summary: {split_summary_path}")
    wx_v2_samples = read_csv(sample_path)
    wx_v2_split_summary = read_csv(split_path)
    original_split_summary = read_csv(split_summary_path)
    validate_wx_v2_inputs(wx_v2_samples, wx_v2_split_summary)

    output_root.mkdir(parents=True, exist_ok=True)
    sample_values = build_v2_sample_w_table(wx_v2_samples)
    x_summary = summarize_v2_sample_w(sample_values)
    split_summary = build_v2_split_summary_for_three_space(wx_v2_split_summary, original_split_summary)

    if yz_summary_path.exists():
        generate_reports(
            sample_values=sample_values,
            x_summary=x_summary,
            split_summary=split_summary,
            yz_summary=read_csv(yz_summary_path),
            embedding_data_dir=embedding_data_dir,
            output_root=output_root,
            formats=list(args.formats),
            dpi=int(args.dpi),
            top_n=int(args.top_n),
            yz_summary_path=yz_summary_path,
        )
    else:
        print(f"Missing Y/Z-space source; reusing cached Y/Z rows from: {original_sample_w_reports}")
        generate_reports_from_cached_yz(
            sample_values=sample_values,
            x_summary=x_summary,
            split_summary=split_summary,
            original_sample_w_reports=original_sample_w_reports,
            output_root=output_root,
            formats=list(args.formats),
            dpi=int(args.dpi),
            top_n=int(args.top_n),
            yz_summary_path=yz_summary_path,
        )

    print(f"Saved Wx_v2 sample W values: {output_root / 'all_sample_w_values.csv'}")
    print(f"Saved Wx_v2 method summary: {output_root / 'case_method_w_summary.csv'}")
    print(f"Saved three-space V2 sample W values: {output_root / 'three_space_sample_w_values.csv'}")
    print(f"Saved three-space V2 report: {output_root / 'three_space_w_ood_report.md'}")
    print(f"Saved case figures under: {output_root / 'cases'}")


if __name__ == "__main__":
    main()
