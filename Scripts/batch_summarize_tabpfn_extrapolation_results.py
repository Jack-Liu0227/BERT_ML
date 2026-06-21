from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _ood_summary_common import (
    HYBRID_TEST_SET_COLUMN_PREFIXES,
    align_summary_metrics_to_artifact,
    annotate_family_ranks,
    create_global_exports,
    export_case_outputs,
    load_hybrid_prediction_subset_metrics,
    normalize_alloy_family_name,
    normalize_ood_method,
    reset_output_dir,
    resolve_case_level_artifact,
    save_subset_labeling_audit,
)


VALID_METHODS = {
    "random_cv_baseline",
    "target_extrapolation",
    "loco",
    "sparse_x_cluster",
    "sparse_x_single",
    "sparse_y_cluster",
    "sparse_y_single",
    "hybrid_extrapolation_loco",
    "hybrid_extrapolation_random_cv",
    "hybrid_extrapolation_sparse_x_cluster",
    "hybrid_extrapolation_sparse_x_single",
    "hybrid_extrapolation_sparse_y_cluster",
    "hybrid_extrapolation_sparse_y_single",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_raw_ood_method(payload: dict) -> str:
    return str(
        payload.get("split_strategy")
        or payload.get("split_summary", {}).get("split_strategy")
        or ""
    ).strip().lower()


def _metric_missing(value: object) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.isna(numeric)


def _raw_hybrid_subset_metrics_complete(row: dict[str, object]) -> bool:
    for summary_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
        raw_prefix = summary_prefix.replace("summary_", "raw_")
        for metric in ("r2", "mae", "rmse", "n_samples"):
            if _metric_missing(row.get(f"{raw_prefix}_{metric}", pd.NA)):
                return False
    return True


def _fill_raw_subset_metrics_from_summary(
    row: dict[str, object],
    recovered_metrics: dict[str, object],
) -> None:
    for summary_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
        raw_prefix = summary_prefix.replace("summary_", "raw_")
        for metric in ("r2", "mae", "rmse", "n_samples"):
            summary_col = f"{summary_prefix}_{metric}"
            raw_col = f"{raw_prefix}_{metric}"
            if summary_col not in recovered_metrics:
                continue
            if raw_col not in row or _metric_missing(row.get(raw_col, pd.NA)):
                row[raw_col] = recovered_metrics[summary_col]


def infer_case_context(base_dir: Path, metrics_path: Path, payload: dict) -> tuple[str, str, str, str, Path, int | None]:
    relative_parts = metrics_path.relative_to(base_dir).parts
    if len(relative_parts) < 5:
        raise ValueError(f"Unexpected TabPFN metrics path: {metrics_path}")

    raw_ood_method = extract_raw_ood_method(payload)
    fold_index = payload.get("fold_index")
    method_dir_aliases = {
        "loco_k5": "loco",
        "sparse_x_cluster_k5": "sparse_x_cluster",
        "sparse_x_single_k5": "sparse_x_single",
        "sparse_y_cluster_k5": "sparse_y_cluster",
        "sparse_y_single_k5": "sparse_y_single",
        "hybrid_extrapolation_loco_k5": "hybrid_extrapolation_loco",
        "hybrid_extrapolation_random_cv_k5": "hybrid_extrapolation_random_cv",
        "hybrid_extrapolation_sparse_x_cluster_k5": "hybrid_extrapolation_sparse_x_cluster",
        "hybrid_extrapolation_sparse_x_single_k5": "hybrid_extrapolation_sparse_x_single",
        "hybrid_extrapolation_sparse_y_cluster_k5": "hybrid_extrapolation_sparse_y_cluster",
        "hybrid_extrapolation_sparse_y_single_k5": "hybrid_extrapolation_sparse_y_single",
    }
    first_part = str(relative_parts[0]).strip().lower()
    first_part_method = method_dir_aliases.get(first_part, first_part)

    if raw_ood_method not in VALID_METHODS:
        candidate = first_part_method
        if candidate not in VALID_METHODS or len(relative_parts) < 6:
            raise ValueError(f"Unsupported or missing OOD method for: {metrics_path}")
        raw_ood_method = candidate

    if first_part_method in VALID_METHODS and len(relative_parts) >= 6:
        alloy_family, dataset_name, property_name = relative_parts[1:4]
    else:
        alloy_family, dataset_name, property_name = relative_parts[0:3]

    if "folds" in relative_parts:
        fold_pos = relative_parts.index("folds")
        model_dir = base_dir.joinpath(*relative_parts[: fold_pos + 2])
    else:
        model_dir = metrics_path.parent.parent

    return alloy_family, dataset_name, property_name, raw_ood_method, model_dir, (None if fold_index is None else int(fold_index))


def collect_raw_rows(base_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    subset_audit_rows: list[dict[str, object]] = []
    for metrics_path in sorted(base_dir.rglob("metrics/metrics_summary.json")):
        payload = load_json(metrics_path)
        try:
            alloy_family, dataset_name, property_name, raw_ood_method, model_dir, fold_index = infer_case_context(base_dir, metrics_path, payload)
        except ValueError:
            continue

        test_metrics = payload.get("ood_test", {})
        if not test_metrics:
            continue

        feature_mode = str(payload.get("feature_mode", "")).strip().capitalize() or "Unknown"
        model_name = f"{payload.get('model_name', 'TabPFN')}-{feature_mode}"
        target_col = str(payload.get("target_col", property_name))
        ood_method = normalize_ood_method(raw_ood_method)
        row = {
            "alloy_family": normalize_alloy_family_name(alloy_family),
            "dataset_name": dataset_name,
            "property": target_col,
            "ood_method": ood_method,
            "model_family": "TabPFN",
            "model": model_name,
            "model_dir": str(model_dir),
            "source_dir": str(base_dir),
            "fold_index": fold_index,
            "raw_test_r2": test_metrics.get("r2"),
            "raw_test_rmse": test_metrics.get("rmse"),
            "raw_test_mae": test_metrics.get("mae"),
            "predictions_file": str(metrics_path.parent.parent / "predictions" / "all_predictions.csv"),
            "plot_file": str(metrics_path.parent.parent / "plots" / f"{property_name.replace('/', '_')}_predictions.png"),
        }
        test_set_metrics = payload.get("test_set_metrics", {})
        if isinstance(test_set_metrics, dict):
            for test_set_name, summary_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.items():
                metrics = test_set_metrics.get(test_set_name)
                if not isinstance(metrics, dict):
                    continue
                raw_prefix = summary_prefix.replace("summary_", "raw_")
                for metric in ("r2", "mae", "rmse", "n_samples"):
                    row[f"{raw_prefix}_{metric}"] = metrics.get(metric)
        if str(row["ood_method"]).startswith("HybridHigh20+") and not _raw_hybrid_subset_metrics_complete(row):
            recovered_metrics, audit_rows = load_hybrid_prediction_subset_metrics(
                model_dir,
                target_col,
                context={
                    "alloy_family": normalize_alloy_family_name(alloy_family),
                    "dataset_name": dataset_name,
                    "property": target_col,
                    "ood_method": ood_method,
                    "model_family": "TabPFN",
                    "model": model_name,
                    "source_dir": str(base_dir),
                },
            )
            _fill_raw_subset_metrics_from_summary(row, recovered_metrics)
            subset_audit_rows.extend(audit_rows)
        rows.append(row)
    return rows, subset_audit_rows


def _std_or_zero(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    std_value = numeric.std()
    if pd.isna(std_value):
        return 0.0
    return float(std_value)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    raw_df = pd.DataFrame(rows)
    aggregated_rows: list[dict[str, object]] = []
    group_cols = ["alloy_family", "dataset_name", "property", "ood_method", "model_family", "model"]

    for group_values, group_df in raw_df.groupby(group_cols, sort=True, dropna=False):
        summary_test_r2 = float(pd.to_numeric(group_df["raw_test_r2"], errors="coerce").mean())
        summary_test_mae = float(pd.to_numeric(group_df["raw_test_mae"], errors="coerce").mean())
        summary_test_rmse = float(pd.to_numeric(group_df["raw_test_rmse"], errors="coerce").mean())

        representative_df = group_df.assign(
            _distance_to_summary_test_r2=(pd.to_numeric(group_df["raw_test_r2"], errors="coerce") - summary_test_r2).abs(),
            _fold_sort=pd.to_numeric(group_df["fold_index"], errors="coerce").fillna(-1),
        ).sort_values(
            ["_distance_to_summary_test_r2", "raw_test_r2", "_fold_sort", "predictions_file"],
            ascending=[True, False, True, True],
            na_position="last",
            kind="stable",
        )
        representative_row = representative_df.iloc[0]

        selection_mode = "single_run"
        if len(group_df) > 1:
            selection_mode = "closest_summary_test_r2_fold"

        tabpfn_loco_fold_details: list[dict[str, object]] = []
        if str(group_values[3]) in {"LOCO", "RandomCV", "HybridHigh20+LOCO", "HybridHigh20+RandCV"}:
            detail_df = group_df.assign(
                _fold_sort=pd.to_numeric(group_df["fold_index"], errors="coerce").fillna(np.inf),
            ).sort_values(
                ["_fold_sort", "predictions_file"],
                ascending=[True, True],
                na_position="last",
                kind="stable",
            )
            for _, fold_row in detail_df.iterrows():
                tabpfn_loco_fold_details.append(
                    {
                        "fold_index": (
                            int(fold_row["fold_index"])
                            if pd.notna(fold_row["fold_index"])
                            else None
                        ),
                        "model_dir": str(fold_row["model_dir"]),
                        "predictions_file": str(fold_row["predictions_file"]),
                        "test_r2": (
                            float(fold_row["raw_test_r2"])
                            if pd.notna(fold_row["raw_test_r2"])
                            else None
                        ),
                        "test_mae": (
                            float(fold_row["raw_test_mae"])
                            if pd.notna(fold_row["raw_test_mae"])
                            else None
                        ),
                        "test_rmse": (
                            float(fold_row["raw_test_rmse"])
                            if pd.notna(fold_row["raw_test_rmse"])
                            else None
                        ),
                    }
                )

        record = dict(zip(group_cols, group_values))
        method_label = str(group_values[3])
        selection_mode = "single_run"
        if len(group_df) > 1:
            selection_mode = (
                "closest_summary_test_r2_outer_fold"
                if method_label in {"LOCO", "RandomCV", "HybridHigh20+LOCO", "HybridHigh20+RandCV"}
                else "closest_summary_test_r2_fold"
            )
        record.update(
            {
                "model_dir": str(representative_row["model_dir"]),
                "source_dir": str(representative_row["source_dir"]),
                "trial_count": 1,
                "fold_count": int(len(group_df)),
                "summary_test_r2": summary_test_r2,
                "summary_test_r2_std": _std_or_zero(group_df["raw_test_r2"]),
                "summary_test_mae": summary_test_mae,
                "summary_test_mae_std": _std_or_zero(group_df["raw_test_mae"]),
                "summary_test_rmse": summary_test_rmse,
                "summary_test_rmse_std": _std_or_zero(group_df["raw_test_rmse"]),
                "representative_selection_mode": selection_mode,
                "representative_trial_id": pd.NA,
                "representative_fold": representative_row["fold_index"] if pd.notna(representative_row["fold_index"]) else pd.NA,
                "representative_test_r2": float(representative_row["raw_test_r2"]),
                "representative_test_mae": float(representative_row["raw_test_mae"]),
                "representative_test_rmse": float(representative_row["raw_test_rmse"]),
                "representative_predictions_file": str(representative_row["predictions_file"]),
                "representative_plot_file": str(representative_row["plot_file"]),
                "tabpfn_loco_fold_count": len(tabpfn_loco_fold_details) if tabpfn_loco_fold_details else pd.NA,
                "tabpfn_loco_fold_details_json": (
                    json.dumps(tabpfn_loco_fold_details, ensure_ascii=False)
                    if tabpfn_loco_fold_details
                    else pd.NA
                ),
            }
        )
        for summary_prefix in HYBRID_TEST_SET_COLUMN_PREFIXES.values():
            raw_prefix = summary_prefix.replace("summary_", "raw_")
            for metric in ("r2", "mae", "rmse"):
                raw_col = f"{raw_prefix}_{metric}"
                if raw_col not in group_df.columns:
                    continue
                numeric = pd.to_numeric(group_df[raw_col], errors="coerce").dropna()
                if numeric.empty:
                    continue
                record[f"{summary_prefix}_{metric}"] = float(numeric.mean())
                record[f"{summary_prefix}_{metric}_std"] = _std_or_zero(numeric)
            raw_count_col = f"{raw_prefix}_n_samples"
            if raw_count_col in group_df.columns:
                counts = pd.to_numeric(group_df[raw_count_col], errors="coerce").dropna()
                if not counts.empty:
                    record[f"{summary_prefix}_n_samples"] = int(counts.sum())
        aggregated_rows.append(record)

    return aggregated_rows


def add_artifact_and_plot_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    working_df = summary_df.copy()
    for column in [
        "artifact_selection_mode",
        "artifact_predictions_file",
        "artifact_expected_split_file",
    ]:
        if column not in working_df.columns:
            working_df[column] = pd.NA
    for column in [
        "artifact_test_r2",
        "artifact_test_mae",
        "artifact_test_rmse",
        "artifact_test_row_count",
        "plot_test_r2",
        "plot_test_mae",
        "plot_test_rmse",
    ]:
        if column not in working_df.columns:
            working_df[column] = pd.NA

    for idx, row in working_df.iterrows():
        artifact_info = resolve_case_level_artifact(row)
        if not artifact_info:
            continue
        working_df.at[idx, "artifact_selection_mode"] = artifact_info.get("source_mode", pd.NA)
        working_df.at[idx, "artifact_predictions_file"] = artifact_info.get("predictions_file", pd.NA)
        working_df.at[idx, "artifact_expected_split_file"] = artifact_info.get("expected_split_file", pd.NA)
        working_df.at[idx, "artifact_test_r2"] = artifact_info.get("test_r2", pd.NA)
        working_df.at[idx, "artifact_test_mae"] = artifact_info.get("test_mae", pd.NA)
        working_df.at[idx, "artifact_test_rmse"] = artifact_info.get("test_rmse", pd.NA)
        working_df.at[idx, "artifact_test_row_count"] = artifact_info.get("test_row_count", pd.NA)

    for metric in ["r2", "mae", "rmse"]:
        plot_col = f"plot_test_{metric}"
        artifact_col = f"artifact_test_{metric}"
        representative_col = f"representative_test_{metric}"
        summary_col = f"summary_test_{metric}"
        plot_series = pd.to_numeric(working_df[plot_col], errors="coerce")
        artifact_series = pd.to_numeric(working_df[artifact_col], errors="coerce")
        representative_series = pd.to_numeric(working_df[representative_col], errors="coerce")
        summary_series = pd.to_numeric(working_df[summary_col], errors="coerce")
        plot_series = plot_series.where(plot_series.notna(), artifact_series)
        plot_series = plot_series.where(plot_series.notna(), representative_series)
        plot_series = plot_series.where(plot_series.notna(), summary_series)
        working_df[plot_col] = plot_series

    return working_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize TabPFN 2.5-Plus multi-OOD experiment results.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional legacy single base directory. When provided, only this directory is scanned.",
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=None,
        help="One or more TabPFN result roots. Defaults to Numeric + Text 2.5-Plus roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to output/ood_summary_reports/TabPFN.",
    )
    args = parser.parse_args()

    if args.base_dirs:
        base_dirs = [Path(item) for item in args.base_dirs]
    elif args.base_dir is not None:
        base_dirs = [args.base_dir]
    else:
        base_dirs = [
            Path("output/ood_results_TabPFN-2.5-Plus-Numeric"),
            Path("output/ood_results_TabPFN-2.5-Plus-Text"),
        ]

    missing_dirs = [str(path) for path in base_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Base directory not found: {', '.join(missing_dirs)}")

    summary_root = args.output_dir or Path("output/ood_summary_reports/TabPFN")
    reset_output_dir(summary_root)

    raw_rows: list[dict[str, object]] = []
    subset_audit_rows: list[dict[str, object]] = []
    for base_dir in base_dirs:
        family_raw_rows, family_subset_audit_rows = collect_raw_rows(base_dir)
        raw_rows.extend(family_raw_rows)
        subset_audit_rows.extend(family_subset_audit_rows)
    rows = aggregate_rows(raw_rows)
    if not rows:
        print(f"No TabPFN multi-OOD metrics were collected under: {base_dirs}")
        return

    summary_df = align_summary_metrics_to_artifact(pd.DataFrame(rows))
    summary_df = annotate_family_ranks(summary_df)
    summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    create_global_exports(summary_df, summary_root, "all_tabpfn_ood_model_summary.csv")
    save_subset_labeling_audit(subset_audit_rows, summary_root)
    export_case_outputs(summary_df, summary_root)

    print(f"TabPFN OOD summary complete: {summary_root}")
    print(f"Cases: {summary_df[['alloy_family', 'dataset_name', 'property']].drop_duplicates().shape[0]}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
