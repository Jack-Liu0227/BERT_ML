from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _ood_summary_common import (
    align_summary_metrics_to_artifact,
    annotate_family_ranks,
    aggregate_hybrid_test_set_metrics,
    create_global_exports,
    export_case_outputs,
    load_hybrid_test_set_metrics,
    normalize_alloy_family_name,
    normalize_ood_method,
    reset_output_dir,
)


EXPERIMENT_PREFIX = "experiment3_llmprop_"
MODEL_FAMILY = "LLMProp"
MODEL_NAME = "LLM-Prop"
DISPLAY_LABEL = "LLM-Prop"
SUMMARY_FILENAME = "all_llmprop_ood_model_summary.csv"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _std_or_zero(values: list[float]) -> float:
    numeric = pd.to_numeric(pd.Series(values, dtype="float64"), errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    value = numeric.std()
    return 0.0 if pd.isna(value) else float(value)


def _metric_prefix(target_column: str) -> str:
    return f"test_{target_column}_"


def _extract_test_metrics(metrics_path: Path, target_column: str) -> dict[str, float]:
    payload = _read_json(metrics_path)
    prefix = _metric_prefix(target_column)
    required = {
        "r2": f"{prefix}r2",
        "mae": f"{prefix}mae",
        "rmse": f"{prefix}rmse",
    }
    missing = [key for key in required.values() if key not in payload]
    if missing:
        raise KeyError(f"Missing LLMProp test metrics in {metrics_path}: {missing}")
    return {
        "r2": float(payload[required["r2"]]),
        "mae": float(payload[required["mae"]]),
        "rmse": float(payload[required["rmse"]]),
    }


def _outer_fold_index(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        name = candidate.name
        if not name.startswith("fold_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _infer_case_from_model_root(base_dir: Path, model_root: Path) -> tuple[str, str, str, str, Path]:
    relative_parts = model_root.relative_to(base_dir).parts
    if len(relative_parts) < 6:
        raise ValueError(f"Unexpected LLMProp result path: {model_root}")
    experiment_name, alloy_family, dataset_name, property_name, raw_ood_method = relative_parts[:5]
    return alloy_family, dataset_name, property_name, raw_ood_method, base_dir / experiment_name


def _single_run_row(base_dir: Path, model_root: Path) -> dict[str, Any]:
    alloy_family, dataset_name, property_name, raw_ood_method, experiment_dir = _infer_case_from_model_root(base_dir, model_root)
    metrics = _extract_test_metrics(model_root / "final_evaluation_metrics.json", property_name)
    predictions_file = model_root / "predictions" / "all_predictions.csv"
    plot_file = model_root / "predictions" / "test_predictions.csv"
    row = {
        "alloy_family": normalize_alloy_family_name(alloy_family),
        "dataset_name": dataset_name,
        "property": property_name,
        "ood_method": normalize_ood_method(raw_ood_method),
        "model_family": MODEL_FAMILY,
        "model": MODEL_NAME,
        "display_label": DISPLAY_LABEL,
        "model_dir": str(model_root),
        "source_dir": str(experiment_dir),
        "trial_count": 1,
        "fold_count": 1,
        "summary_test_r2": metrics["r2"],
        "summary_test_r2_std": 0.0,
        "summary_test_mae": metrics["mae"],
        "summary_test_mae_std": 0.0,
        "summary_test_rmse": metrics["rmse"],
        "summary_test_rmse_std": 0.0,
        "representative_selection_mode": "single_run",
        "representative_trial_id": pd.NA,
        "representative_fold": pd.NA,
        "representative_test_r2": metrics["r2"],
        "representative_test_mae": metrics["mae"],
        "representative_test_rmse": metrics["rmse"],
        "representative_predictions_file": str(predictions_file) if predictions_file.exists() else pd.NA,
        "representative_plot_file": str(plot_file) if plot_file.exists() else pd.NA,
        "loco_outer_fold_best_count": pd.NA,
        "loco_outer_fold_best_details_json": pd.NA,
        "artifact_selection_mode": "llmprop_single_run",
        "artifact_predictions_file": str(predictions_file) if predictions_file.exists() else pd.NA,
        "artifact_expected_split_file": str(model_root / "split_data" / "test.csv")
        if (model_root / "split_data" / "test.csv").exists()
        else pd.NA,
        "artifact_test_r2": metrics["r2"],
        "artifact_test_mae": metrics["mae"],
        "artifact_test_rmse": metrics["rmse"],
        "artifact_test_row_count": _count_test_rows(predictions_file),
        "plot_test_r2": metrics["r2"],
        "plot_test_mae": metrics["mae"],
        "plot_test_rmse": metrics["rmse"],
    }
    row.update(load_hybrid_test_set_metrics(model_root / "final_evaluation_metrics.json"))
    return row


def _count_test_rows(predictions_file: Path) -> int | float:
    if not predictions_file.exists():
        return np.nan
    try:
        df = pd.read_csv(predictions_file, low_memory=False)
    except Exception:
        return np.nan
    if "Dataset" not in df.columns:
        return int(len(df))
    dataset = (
        df["Dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )
    return int(dataset.isin({"test", "testing", "oodtest", "ood", "oodtesting"}).sum())


def _multi_fold_row(base_dir: Path, model_root: Path) -> dict[str, Any] | None:
    alloy_family, dataset_name, property_name, raw_ood_method, experiment_dir = _infer_case_from_model_root(base_dir, model_root)
    fold_records: list[dict[str, Any]] = []
    hybrid_metric_payloads: list[dict[str, Any]] = []
    for fold_dir in sorted((model_root / "folds").glob("fold_*")):
        metrics_path = fold_dir / "final_evaluation_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _extract_test_metrics(metrics_path, property_name)
        try:
            hybrid_metric_payloads.append(_read_json(metrics_path))
        except Exception:
            pass
        predictions_file = fold_dir / "predictions" / "all_predictions.csv"
        fold_records.append(
            {
                "outer_fold_index": _outer_fold_index(fold_dir),
                "model_dir": str(fold_dir),
                "outer_predictions_file": str(predictions_file) if predictions_file.exists() else pd.NA,
                "outer_test_r2": metrics["r2"],
                "outer_test_mae": metrics["mae"],
                "outer_test_rmse": metrics["rmse"],
                "outer_test_row_count": _count_test_rows(predictions_file),
            }
        )
    if not fold_records:
        return None

    fold_df = pd.DataFrame(fold_records)
    for metric in ["outer_test_r2", "outer_test_mae", "outer_test_rmse"]:
        fold_df[metric] = pd.to_numeric(fold_df[metric], errors="coerce")
    fold_df = fold_df.dropna(subset=["outer_test_r2", "outer_test_mae", "outer_test_rmse"]).reset_index(drop=True)
    if fold_df.empty:
        return None

    summary_mae = float(fold_df["outer_test_mae"].mean())
    representative = (
        fold_df.assign(
            _distance_to_summary_mae=(fold_df["outer_test_mae"] - summary_mae).abs(),
            _sort_fold=pd.to_numeric(fold_df["outer_fold_index"], errors="coerce").fillna(np.inf),
        )
        .sort_values(
            ["_distance_to_summary_mae", "outer_test_mae", "outer_test_r2", "_sort_fold"],
            ascending=[True, True, False, True],
            na_position="last",
            kind="mergesort",
        )
        .iloc[0]
    )
    details = fold_df.to_dict(orient="records")
    representative_fold = representative["outer_fold_index"]
    representative_fold = int(representative_fold) if pd.notna(representative_fold) else pd.NA
    row = {
        "alloy_family": normalize_alloy_family_name(alloy_family),
        "dataset_name": dataset_name,
        "property": property_name,
        "ood_method": normalize_ood_method(raw_ood_method),
        "model_family": MODEL_FAMILY,
        "model": MODEL_NAME,
        "display_label": DISPLAY_LABEL,
        "model_dir": str(model_root),
        "source_dir": str(experiment_dir),
        "trial_count": int(len(fold_df)),
        "fold_count": int(len(fold_df)),
        "summary_test_r2": float(fold_df["outer_test_r2"].mean()),
        "summary_test_r2_std": _std_or_zero(fold_df["outer_test_r2"].tolist()),
        "summary_test_mae": summary_mae,
        "summary_test_mae_std": _std_or_zero(fold_df["outer_test_mae"].tolist()),
        "summary_test_rmse": float(fold_df["outer_test_rmse"].mean()),
        "summary_test_rmse_std": _std_or_zero(fold_df["outer_test_rmse"].tolist()),
        "representative_selection_mode": "closest_summary_test_mae_outer_fold_oodtest",
        "representative_trial_id": pd.NA,
        "representative_fold": representative_fold,
        "representative_test_r2": float(representative["outer_test_r2"]),
        "representative_test_mae": float(representative["outer_test_mae"]),
        "representative_test_rmse": float(representative["outer_test_rmse"]),
        "representative_predictions_file": representative["outer_predictions_file"],
        "representative_plot_file": pd.NA,
        "loco_outer_fold_best_count": int(len(fold_df)),
        "loco_outer_fold_best_details_json": json.dumps(details, ensure_ascii=False),
        "artifact_selection_mode": "llmprop_outer_fold_aggregate",
        "artifact_predictions_file": representative["outer_predictions_file"],
        "artifact_expected_split_file": pd.NA,
        "artifact_test_r2": float(fold_df["outer_test_r2"].mean()),
        "artifact_test_mae": summary_mae,
        "artifact_test_rmse": float(fold_df["outer_test_rmse"].mean()),
        "artifact_test_row_count": float(pd.to_numeric(fold_df["outer_test_row_count"], errors="coerce").sum()),
        "plot_test_r2": float(fold_df["outer_test_r2"].mean()),
        "plot_test_mae": summary_mae,
        "plot_test_rmse": float(fold_df["outer_test_rmse"].mean()),
    }
    row.update(aggregate_hybrid_test_set_metrics(hybrid_metric_payloads))
    return row


def collect_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiment_dirs = [
        *base_dir.glob(f"{EXPERIMENT_PREFIX}*"),
        *base_dir.glob("experiment_hybrid_llmprop_*"),
    ]
    for experiment_dir in sorted(set(experiment_dirs)):
        if not experiment_dir.is_dir():
            continue
        for model_root in sorted(experiment_dir.glob("*/*/*/*/llmprop")):
            if (model_root / "folds").is_dir():
                row = _multi_fold_row(base_dir, model_root)
                if row is not None:
                    rows.append(row)
            elif (model_root / "final_evaluation_metrics.json").exists():
                rows.append(_single_run_row(base_dir, model_root))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize LLM-Prop OOD experiment results.")
    parser.add_argument("--base-dir", type=Path, default=Path("output/ood_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ood_summary_reports/LLMProp"))
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    reset_output_dir(args.output_dir)
    rows = collect_rows(args.base_dir)
    if not rows:
        print(f"No LLMProp OOD metrics were collected under: {args.base_dir}")
        return

    summary_df = pd.DataFrame(rows)
    summary_df = align_summary_metrics_to_artifact(summary_df)
    summary_df = annotate_family_ranks(summary_df)
    summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    create_global_exports(summary_df, args.output_dir, SUMMARY_FILENAME)
    export_case_outputs(summary_df, args.output_dir)

    print(f"LLMProp OOD summary complete: {args.output_dir}")
    print(f"Cases: {summary_df[['alloy_family', 'dataset_name', 'property']].drop_duplicates().shape[0]}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
