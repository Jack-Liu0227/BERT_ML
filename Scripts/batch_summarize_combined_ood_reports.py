from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _external_ood_sources import load_external_ood_sources
from _ood_summary_common import (
    OOD_METHOD_ORDER,
    normalize_alloy_family_name,
    plot_case_metric,
    reset_output_dir,
    resolve_case_level_artifact,
    safe_name,
    save_csv,
)


SUMMARY_FILES = {
    "Traditional": "all_traditional_ood_model_summary.csv",
    "BERT": "all_bert_ood_model_summary.csv",
    "TabPFN": "all_tabpfn_ood_model_summary.csv",
}

TASK_KEYS = ["alloy_family", "dataset_name", "property"]
CASE_KEYS = TASK_KEYS + ["ood_method"]
FAMILY_KEYS = CASE_KEYS + ["model_family"]
CANONICAL_COLUMNS = [
    "alloy_family",
    "dataset_name",
    "property",
    "ood_method",
    "model_family",
    "model",
    "display_label",
    "model_dir",
    "source_dir",
    "trial_count",
    "fold_count",
    "summary_test_r2",
    "summary_test_r2_std",
    "summary_test_mae",
    "summary_test_mae_std",
    "summary_test_rmse",
    "summary_test_rmse_std",
    "representative_selection_mode",
    "representative_trial_id",
    "representative_fold",
    "representative_test_r2",
    "representative_test_mae",
    "representative_test_rmse",
    "representative_predictions_file",
    "representative_plot_file",
    "artifact_selection_mode",
    "artifact_predictions_file",
    "artifact_expected_split_file",
    "artifact_test_r2",
    "artifact_test_mae",
    "artifact_test_rmse",
    "artifact_test_row_count",
    "plot_test_r2",
    "plot_test_mae",
    "plot_test_rmse",
    "family_best_metric",
    "family_rank_score",
    "rank_within_family",
    "is_family_best",
    "source_family_dir",
]
NUMERIC_COLUMNS = [
    "trial_count",
    "fold_count",
    "summary_test_r2",
    "summary_test_r2_std",
    "summary_test_mae",
    "summary_test_mae_std",
    "summary_test_rmse",
    "summary_test_rmse_std",
    "representative_fold",
    "representative_test_r2",
    "representative_test_mae",
    "representative_test_rmse",
    "artifact_test_r2",
    "artifact_test_mae",
    "artifact_test_rmse",
    "artifact_test_row_count",
    "plot_test_r2",
    "plot_test_mae",
    "plot_test_rmse",
    "family_rank_score",
    "rank_within_family",
]


def _coalesce_column(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
    default: object = pd.NA,
) -> None:
    if target in df.columns:
        series = df[target]
    else:
        series = pd.Series(default, index=df.index, dtype="object")

    for candidate in candidates:
        if candidate not in df.columns:
            continue
        series = series.where(series.notna(), df[candidate])

    df[target] = series


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _add_family_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ranked_parts: list[pd.DataFrame] = []
    for _, family_df in df.groupby(FAMILY_KEYS, dropna=False, sort=False):
        working_df = family_df.copy()
        working_df["_sort_r2"] = pd.to_numeric(working_df["summary_test_r2"], errors="coerce").fillna(-np.inf)
        working_df["_sort_r2_std"] = pd.to_numeric(working_df["summary_test_r2_std"], errors="coerce").fillna(np.inf)
        working_df["_sort_mae"] = pd.to_numeric(working_df["summary_test_mae"], errors="coerce").fillna(np.inf)
        working_df = working_df.sort_values(
            ["_sort_r2", "_sort_r2_std", "_sort_mae", "model"],
            ascending=[False, True, True, True],
            kind="mergesort",
            na_position="last",
        )
        working_df["family_best_metric"] = "summary_test_r2"
        working_df["family_rank_score"] = pd.to_numeric(working_df["summary_test_r2"], errors="coerce")
        working_df["rank_within_family"] = np.arange(1, len(working_df) + 1, dtype=int)
        working_df["is_family_best"] = working_df["rank_within_family"].eq(1)
        ranked_parts.append(working_df.drop(columns=["_sort_r2", "_sort_r2_std", "_sort_mae"]))

    return pd.concat(ranked_parts, ignore_index=True)


def canonicalize_family_summary(df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    if "model_family" in working_df.columns:
        working_df["model_family"] = working_df["model_family"].fillna(family_name)
    else:
        working_df["model_family"] = family_name

    for column in ["alloy_family", "dataset_name", "property", "ood_method", "model", "model_dir", "source_dir"]:
        if column not in working_df.columns:
            working_df[column] = pd.NA
    if "display_label" not in working_df.columns:
        working_df["display_label"] = working_df["model"]

    single_run_defaults = {
        "trial_count": 1 if family_name == "TabPFN" else pd.NA,
        "fold_count": 1 if family_name == "TabPFN" else pd.NA,
        "summary_test_r2_std": 0.0 if family_name == "TabPFN" else pd.NA,
        "summary_test_mae_std": 0.0 if family_name == "TabPFN" else pd.NA,
        "summary_test_rmse_std": 0.0 if family_name == "TabPFN" else pd.NA,
        "representative_selection_mode": "single_run" if family_name == "TabPFN" else pd.NA,
    }

    _coalesce_column(working_df, "trial_count", [], default=single_run_defaults["trial_count"])
    _coalesce_column(working_df, "fold_count", [], default=single_run_defaults["fold_count"])
    _coalesce_column(working_df, "summary_test_r2", ["final_test_r2", "test_r2"])
    _coalesce_column(working_df, "summary_test_r2_std", [], default=single_run_defaults["summary_test_r2_std"])
    _coalesce_column(working_df, "summary_test_mae", ["final_test_mae", "test_mae"])
    _coalesce_column(working_df, "summary_test_mae_std", [], default=single_run_defaults["summary_test_mae_std"])
    _coalesce_column(working_df, "summary_test_rmse", ["final_test_rmse", "test_rmse"])
    _coalesce_column(working_df, "summary_test_rmse_std", [], default=single_run_defaults["summary_test_rmse_std"])

    _coalesce_column(working_df, "representative_selection_mode", ["selection_mode"], default=single_run_defaults["representative_selection_mode"])
    _coalesce_column(working_df, "representative_trial_id", ["selected_trial_id"])
    _coalesce_column(working_df, "representative_fold", ["selected_fold"])
    _coalesce_column(working_df, "representative_test_r2", ["test_r2", "final_test_r2"])
    _coalesce_column(working_df, "representative_test_mae", ["test_mae", "final_test_mae"])
    _coalesce_column(working_df, "representative_test_rmse", ["test_rmse", "final_test_rmse"])
    _coalesce_column(working_df, "representative_predictions_file", ["predictions_file"])
    _coalesce_column(working_df, "representative_plot_file", ["plot_file"])

    _coalesce_column(working_df, "family_best_metric", [], default="summary_test_r2")
    _coalesce_column(working_df, "family_rank_score", ["summary_test_r2"])
    _coalesce_column(working_df, "rank_within_family", [])
    _coalesce_column(working_df, "is_family_best", [], default=False)

    for column in NUMERIC_COLUMNS:
        if column not in working_df.columns:
            working_df[column] = np.nan
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    if family_name == "TabPFN":
        for std_col in ["summary_test_r2_std", "summary_test_mae_std", "summary_test_rmse_std"]:
            working_df.loc[working_df[std_col].isna(), std_col] = 0.0
        for count_col in ["trial_count", "fold_count"]:
            working_df.loc[working_df[count_col].isna(), count_col] = 1
        working_df.loc[
            working_df["representative_selection_mode"].isna(),
            "representative_selection_mode",
        ] = "single_run"
    working_df["is_family_best"] = _normalize_bool_series(working_df["is_family_best"])

    working_df = _add_family_ranking_columns(working_df)
    return working_df


def load_family_summary(reports_root: Path, family_name: str) -> pd.DataFrame:
    summary_file = reports_root / family_name / "00_summary_tables" / SUMMARY_FILES[family_name]
    if not summary_file.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.read_csv(summary_file)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = canonicalize_family_summary(df, family_name)
    df["source_family_dir"] = str((reports_root / family_name).resolve())
    return df.reindex(columns=CANONICAL_COLUMNS)


def enrich_with_artifact_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    existing_artifact_defaults = {
        "artifact_selection_mode": pd.Series(pd.NA, index=working_df.index, dtype="object"),
        "artifact_predictions_file": pd.Series(pd.NA, index=working_df.index, dtype="object"),
        "artifact_expected_split_file": pd.Series(pd.NA, index=working_df.index, dtype="object"),
        "artifact_test_r2": pd.Series(np.nan, index=working_df.index, dtype="float64"),
        "artifact_test_mae": pd.Series(np.nan, index=working_df.index, dtype="float64"),
        "artifact_test_rmse": pd.Series(np.nan, index=working_df.index, dtype="float64"),
        "artifact_test_row_count": pd.Series(np.nan, index=working_df.index, dtype="float64"),
    }
    artifact_columns = {
        column: (
            working_df[column].copy()
            if column in working_df.columns
            else default_series.copy()
        )
        for column, default_series in existing_artifact_defaults.items()
    }

    for idx, row in working_df.iterrows():
        has_existing_artifact = any(
            pd.notna(working_df.at[idx, col])
            for col in ["artifact_test_r2", "artifact_test_mae", "artifact_test_rmse", "artifact_predictions_file"]
            if col in working_df.columns
        )
        if has_existing_artifact:
            continue
        artifact_info = resolve_case_level_artifact(row)
        if not artifact_info:
            continue
        artifact_columns["artifact_selection_mode"].iat[idx] = artifact_info.get("source_mode", pd.NA)
        artifact_columns["artifact_predictions_file"].iat[idx] = artifact_info.get("predictions_file", pd.NA)
        artifact_columns["artifact_expected_split_file"].iat[idx] = artifact_info.get("expected_split_file", pd.NA)
        artifact_columns["artifact_test_r2"].iat[idx] = pd.to_numeric(pd.Series([artifact_info.get("test_r2")]), errors="coerce").iloc[0]
        artifact_columns["artifact_test_mae"].iat[idx] = pd.to_numeric(pd.Series([artifact_info.get("test_mae")]), errors="coerce").iloc[0]
        artifact_columns["artifact_test_rmse"].iat[idx] = pd.to_numeric(pd.Series([artifact_info.get("test_rmse")]), errors="coerce").iloc[0]
        artifact_columns["artifact_test_row_count"].iat[idx] = pd.to_numeric(pd.Series([artifact_info.get("test_row_count")]), errors="coerce").iloc[0]

    for column, series in artifact_columns.items():
        working_df[column] = series

    if "plot_test_r2" not in working_df.columns:
        working_df["plot_test_r2"] = np.nan
    if "plot_test_mae" not in working_df.columns:
        working_df["plot_test_mae"] = np.nan
    if "plot_test_rmse" not in working_df.columns:
        working_df["plot_test_rmse"] = np.nan

    existing_plot_r2 = pd.to_numeric(working_df["plot_test_r2"], errors="coerce")
    existing_plot_mae = pd.to_numeric(working_df["plot_test_mae"], errors="coerce")
    existing_plot_rmse = pd.to_numeric(working_df["plot_test_rmse"], errors="coerce")
    representative_r2 = pd.to_numeric(working_df["representative_test_r2"], errors="coerce")
    representative_mae = pd.to_numeric(working_df["representative_test_mae"], errors="coerce")
    representative_rmse = pd.to_numeric(working_df["representative_test_rmse"], errors="coerce")
    artifact_r2 = pd.to_numeric(working_df["artifact_test_r2"], errors="coerce")
    artifact_mae = pd.to_numeric(working_df["artifact_test_mae"], errors="coerce")
    artifact_rmse = pd.to_numeric(working_df["artifact_test_rmse"], errors="coerce")
    summary_r2 = pd.to_numeric(working_df["summary_test_r2"], errors="coerce")
    summary_mae = pd.to_numeric(working_df["summary_test_mae"], errors="coerce")
    summary_rmse = pd.to_numeric(working_df["summary_test_rmse"], errors="coerce")

    # Combined plots/data should align with the exported alloy-case artifacts under
    # output/ood_summary_reports/*/01_alloy_cases/... . Those selected files now track
    # the original experiment's actual OOD test split, i.e. artifact_test_*.
    # Keep representative_* separately for tracing/diagnostics, but do not use it as
    # the primary Combined plotting value.
    working_df["plot_test_r2"] = existing_plot_r2.where(existing_plot_r2.notna(), artifact_r2)
    working_df["plot_test_r2"] = pd.to_numeric(working_df["plot_test_r2"], errors="coerce").where(
        pd.to_numeric(working_df["plot_test_r2"], errors="coerce").notna(),
        representative_r2,
    ).where(
        lambda s: s.notna(),
        summary_r2,
    )
    working_df["plot_test_mae"] = existing_plot_mae.where(existing_plot_mae.notna(), artifact_mae)
    working_df["plot_test_mae"] = pd.to_numeric(working_df["plot_test_mae"], errors="coerce").where(
        pd.to_numeric(working_df["plot_test_mae"], errors="coerce").notna(),
        representative_mae,
    ).where(
        lambda s: s.notna(),
        summary_mae,
    )
    working_df["plot_test_rmse"] = existing_plot_rmse.where(existing_plot_rmse.notna(), artifact_rmse)
    working_df["plot_test_rmse"] = pd.to_numeric(working_df["plot_test_rmse"], errors="coerce").where(
        pd.to_numeric(working_df["plot_test_rmse"], errors="coerce").notna(),
        representative_rmse,
    ).where(
        lambda s: s.notna(),
        summary_rmse,
    )
    return working_df.reindex(columns=CANONICAL_COLUMNS)


def build_case_stem(alloy_family: str, dataset_name: str, property_name: str) -> str:
    return "__".join(
        [
            safe_name(str(alloy_family)),
            safe_name(str(dataset_name)),
            safe_name(str(property_name)),
        ]
    )


def build_combined_df(reports_root: Path, external_sources_config: Path | None = None) -> pd.DataFrame:
    frames = [load_family_summary(reports_root, family_name) for family_name in SUMMARY_FILES]
    external_df = load_external_ood_sources(external_sources_config)
    if not external_df.empty:
        for column in CANONICAL_COLUMNS:
            if column not in external_df.columns:
                external_df[column] = pd.NA
        frames.append(external_df.reindex(columns=CANONICAL_COLUMNS))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["alloy_family"] = combined_df["alloy_family"].map(normalize_alloy_family_name)
    combined_df["ood_method"] = pd.Categorical(
        combined_df["ood_method"],
        categories=OOD_METHOD_ORDER,
        ordered=True,
    )
    combined_df = combined_df.sort_values(
        ["alloy_family", "dataset_name", "property", "ood_method", "model_family", "model"],
        na_position="last",
    ).reset_index(drop=True)
    return enrich_with_artifact_metrics(combined_df)


def summarize_case_anomalies(case_df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    model_count = int(
        case_df.assign(model_key=case_df["model_family"].astype(str) + "::" + case_df["model"].astype(str))["model_key"]
        .nunique()
    )
    available_methods = set(case_df["ood_method"].astype(str))
    missing_methods = [method for method in OOD_METHOD_ORDER if method not in available_methods]

    for metric_col, metric_name in [
        ("plot_test_r2", "R2"),
        ("plot_test_mae", "MAE"),
        ("plot_test_rmse", "RMSE"),
    ]:
        metric_df = case_df[["ood_method", "model_family", "model", metric_col]].copy()
        metric_df[metric_col] = pd.to_numeric(metric_df[metric_col], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric_col]).reset_index(drop=True)
        if metric_df.empty:
            continue

        values = metric_df[metric_col].astype(float)
        median = float(values.median())
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            if metric_col == "plot_test_r2":
                outlier_mask = values < (q1 - 1.5 * iqr)
            else:
                outlier_mask = values > (q3 + 1.5 * iqr)
        else:
            outlier_mask = pd.Series(False, index=metric_df.index)

        missing_model_count_by_method = {
            method: model_count - int((case_df["ood_method"].astype(str) == method).sum())
            for method in OOD_METHOD_ORDER
            if method in available_methods
        }

        for idx, row in metric_df.iterrows():
            notes: list[str] = []
            missing_count = missing_model_count_by_method.get(str(row["ood_method"]), 0)
            if missing_count > 0:
                notes.append(f"{row['ood_method']} missing {missing_count} models")
            if bool(outlier_mask.iloc[idx]):
                notes.append(f"{metric_name} outlier")
            if metric_col == "plot_test_r2" and float(row[metric_col]) < 0:
                notes.append("R2<0")
            if metric_col in {"plot_test_mae", "plot_test_rmse"} and median > 0 and float(row[metric_col]) >= 2.0 * median:
                notes.append(f"{metric_name} >= 2x median")

            if not notes and not missing_methods:
                continue

            if not notes and missing_methods:
                notes.append(f"missing methods in case: {', '.join(missing_methods)}")

            records.append(
                {
                    "alloy_family": str(case_df.iloc[0]["alloy_family"]),
                    "dataset_name": str(case_df.iloc[0]["dataset_name"]),
                    "property": str(case_df.iloc[0]["property"]),
                    "metric": metric_name,
                    "ood_method": str(row["ood_method"]),
                    "model_family": str(row["model_family"]),
                    "model": str(row["model"]),
                    "value": float(row[metric_col]),
                    "issue_notes": " | ".join(notes),
                    "missing_methods": ", ".join(missing_methods) if missing_methods else pd.NA,
                }
            )

    if not records and missing_methods:
        records.append(
            {
                "alloy_family": str(case_df.iloc[0]["alloy_family"]),
                "dataset_name": str(case_df.iloc[0]["dataset_name"]),
                "property": str(case_df.iloc[0]["property"]),
                "metric": "ALL",
                "ood_method": pd.NA,
                "model_family": pd.NA,
                "model": pd.NA,
                "value": pd.NA,
                "issue_notes": f"missing methods in case: {', '.join(missing_methods)}",
                "missing_methods": ", ".join(missing_methods),
            }
        )
    return records


def export_combined_case_outputs(combined_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    data_root = output_root / "data"
    image_root = output_root / "figure"
    audit_records: list[dict] = []

    grouped = combined_df.groupby(TASK_KEYS, sort=True)
    for alloy_family, dataset_name, property_name in grouped.groups:
        case_df = (
            grouped.get_group((alloy_family, dataset_name, property_name))
            .sort_values(["ood_method", "model_family", "model"])
            .reset_index(drop=True)
        )

        case_stem = build_case_stem(alloy_family, dataset_name, property_name)
        save_csv(case_df, data_root / f"{case_stem}__combined_model_family_summary.csv")

        pivot_df = case_df.pivot_table(
            index=["model_family", "model"],
            columns="ood_method",
            values=["plot_test_r2", "plot_test_mae", "plot_test_rmse"],
            aggfunc="first",
            observed=False,
        )
        if not pivot_df.empty:
            pivot_df.columns = [f"{method}_{metric}" for metric, method in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            save_csv(pivot_df, data_root / f"{case_stem}__combined_metric_pivot.csv")

        plot_case_metric(case_df, image_root / f"{case_stem}__combined_r2_summary.png", "plot_test_r2", "R2")
        plot_case_metric(case_df, image_root / f"{case_stem}__combined_mae_summary.png", "plot_test_mae", "MAE")
        plot_case_metric(case_df, image_root / f"{case_stem}__combined_rmse_summary.png", "plot_test_rmse", "RMSE")

        audit_records.extend(summarize_case_anomalies(case_df))

    audit_df = pd.DataFrame(audit_records)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(
            ["alloy_family", "dataset_name", "property", "metric", "ood_method", "model_family", "model"],
            na_position="last",
        ).reset_index(drop=True)
        save_csv(audit_df, data_root / "combined_plot_audit.csv")
    return audit_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Traditional, BERT, and TabPFN OOD summaries into one report.")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("output/ood_summary_reports"),
        help="Root directory containing Traditional, BERT, and TabPFN summary folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Combined summary output directory. Defaults to <reports-root>/Combined.",
    )
    parser.add_argument(
        "--external-sources-config",
        type=Path,
        default=Path("Scripts/external_ood_model_sources.yaml"),
        help="Optional YAML config describing extra external OOD model sources to merge into the combined summary.",
    )
    args = parser.parse_args()

    if not args.reports_root.exists():
        raise FileNotFoundError(f"Reports root not found: {args.reports_root}")

    output_root = args.output_dir or (args.reports_root / "Combined")
    reset_output_dir(output_root)

    external_config = args.external_sources_config if args.external_sources_config and args.external_sources_config.exists() else None
    combined_df = build_combined_df(args.reports_root, external_config)
    if combined_df.empty:
        print(f"No family summaries found under: {args.reports_root}")
        return

    save_csv(combined_df.reindex(columns=CANONICAL_COLUMNS), output_root / "data" / "all_model_families_ood_summary.csv")
    audit_df = export_combined_case_outputs(combined_df, output_root)

    print(f"Combined OOD summary complete: {output_root}")
    print(f"Cases: {combined_df[TASK_KEYS].drop_duplicates().shape[0]}")
    print(f"Rows: {len(combined_df)}")
    if audit_df is not None and not audit_df.empty:
        print(f"Audit rows: {len(audit_df)}")


if __name__ == "__main__":
    main()
