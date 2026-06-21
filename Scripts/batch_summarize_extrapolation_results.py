from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _ood_summary_common import (
    align_summary_metrics_to_artifact,
    annotate_family_ranks,
    aggregate_hybrid_test_set_metrics,
    create_global_exports,
    export_case_outputs,
    fill_missing_hybrid_test_set_metrics,
    iter_experiment_dirs,
    load_hybrid_prediction_subset_metrics,
    load_hybrid_test_set_metrics,
    normalize_alloy_family_name,
    normalize_ood_method,
    reset_output_dir,
    save_subset_labeling_audit,
)
from _raw_prediction_stats import (
    load_case_level_test_metrics,
    summarize_optuna_model_trials,
    summarize_optuna_model_trials_from_dirs,
)


MODEL_MAP = {
    "catboost_results": "CatBoost",
    "lightgbm_results": "LightGBM",
    "mlp_results": "MLP",
    "sklearn_rf_results": "RF",
    "xgboost_results": "XGB",
}

CANONICAL_RAW_METHOD_BY_EXPERIMENT_SUFFIX = {
    "random_cv_baseline": {"random_cv_baseline"},
    "extrapolation": {"target_extrapolation"},
    "loco": {"loco_k5", "loco"},
    "sparse_x_cluster": {"sparse_x_cluster_k5", "sparse_x_cluster"},
    "sparse_x_single": {"sparse_x_single_k5", "sparse_x_single"},
    "sparse_y_cluster": {"sparse_y_cluster_k5", "sparse_y_cluster"},
    "sparse_y_single": {"sparse_y_single_k5", "sparse_y_single"},
    "hybrid_all_ml_models_loco": {"hybrid_extrapolation_loco_k5", "hybrid_extrapolation_loco"},
    "hybrid_all_ml_models_random_cv": {"hybrid_extrapolation_random_cv_k5", "hybrid_extrapolation_random_cv"},
    "hybrid_all_ml_models_sparse_x_cluster": {"hybrid_extrapolation_sparse_x_cluster_k5", "hybrid_extrapolation_sparse_x_cluster"},
    "hybrid_all_ml_models_sparse_x_single": {"hybrid_extrapolation_sparse_x_single_k5", "hybrid_extrapolation_sparse_x_single"},
    "hybrid_all_ml_models_sparse_y_cluster": {"hybrid_extrapolation_sparse_y_cluster_k5", "hybrid_extrapolation_sparse_y_cluster"},
    "hybrid_all_ml_models_sparse_y_single": {"hybrid_extrapolation_sparse_y_single_k5", "hybrid_extrapolation_sparse_y_single"},
}


def expected_raw_methods_for_experiment(experiment_name: str) -> set[str]:
    prefixes = ("experiment1_all_ml_models_", "experiment_")
    suffix = experiment_name
    for prefix in prefixes:
        if experiment_name.startswith(prefix):
            suffix = experiment_name[len(prefix):]
            break
    return CANONICAL_RAW_METHOD_BY_EXPERIMENT_SUFFIX.get(suffix, set())


def build_summary_row(
    alloy_family: str,
    dataset_name: str,
    property_name: str,
    ood_method: str,
    model_label: str,
    model_dir: Path,
    experiment_dir: Path,
    summary: dict[str, object],
    subset_audit_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    representative_model_dir = summary.get("representative_model_dir") or str(model_dir)
    representative_model_path = Path(str(representative_model_dir))
    loco_outer_fold_best_details = summary.get("loco_outer_fold_best_details") or []
    row = {
        "alloy_family": normalize_alloy_family_name(alloy_family),
        "dataset_name": dataset_name,
        "property": property_name,
        "ood_method": ood_method,
        "model_family": "Traditional",
        "model": model_label,
        "model_dir": str(representative_model_path),
        "source_dir": str(experiment_dir),
        "trial_count": summary.get("trial_count"),
        "fold_count": summary.get("fold_count"),
        "summary_test_r2": summary.get("summary_test_r2"),
        "summary_test_r2_std": summary.get("summary_test_r2_std"),
        "summary_test_mae": summary.get("summary_test_mae"),
        "summary_test_mae_std": summary.get("summary_test_mae_std"),
        "summary_test_rmse": summary.get("summary_test_rmse"),
        "summary_test_rmse_std": summary.get("summary_test_rmse_std"),
        "representative_selection_mode": summary.get("representative_selection_mode"),
        "representative_trial_id": summary.get("representative_trial_id"),
        "representative_fold": summary.get("representative_fold"),
        "representative_test_r2": summary.get("representative_test_r2"),
        "representative_test_mae": summary.get("representative_test_mae"),
        "representative_test_rmse": summary.get("representative_test_rmse"),
        "representative_predictions_file": summary.get("representative_predictions_file"),
        "representative_plot_file": str(representative_model_path / "plots" / "final_model_evaluation_all_sets_comparison.png"),
        "loco_outer_fold_best_count": len(loco_outer_fold_best_details),
        "loco_outer_fold_best_details_json": json.dumps(loco_outer_fold_best_details, ensure_ascii=False),
    }
    row.update(load_hybrid_test_set_metrics(model_dir / "final_evaluation_metrics.json"))
    for key, value in summary.items():
        if str(key).startswith("summary_test_extrapolation_high20_") or str(key).startswith("summary_test_inner_ood_"):
            row[key] = value
    if str(ood_method).startswith("HybridHigh20+") and str(ood_method) != "HybridHigh20+LOCO":
        recovered_metrics, audit_rows = load_hybrid_prediction_subset_metrics(
            model_dir,
            property_name,
            context={
                "alloy_family": normalize_alloy_family_name(alloy_family),
                "dataset_name": dataset_name,
                "property": property_name,
                "ood_method": ood_method,
                "model_family": "Traditional",
                "model": model_label,
                "source_dir": str(experiment_dir),
            },
        )
        fill_missing_hybrid_test_set_metrics(row, recovered_metrics)
        if subset_audit_rows is not None:
            subset_audit_rows.extend(audit_rows)
    return row


def _outer_fold_index(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        name = str(candidate.name)
        if not name.startswith("fold_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _std_or_zero(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    value = numeric.std()
    return 0.0 if pd.isna(value) else float(value)


def summarize_outer_fold_case_level_metrics(
    outer_model_dirs: list[Path],
    *,
    property_name: str,
) -> dict[str, dict[str, object]]:
    """Summarize RandomCV/LOCO outer folds from final OOD-test artifacts.

    The older path used ``summarize_outer_fold_final_test_metrics`` which also
    scans every inner Optuna trial/fold only to record selected-trial details.
    For publication figures we need the final outer-fold OOD metrics, and those
    are already present in each fold's final prediction artifact. Reading only
    those files keeps the full OOD refresh tractable and avoids missing folded
    RandomCV/LOCO rows.
    """

    fold_rows: list[dict[str, object]] = []
    hybrid_metric_payloads: list[dict[str, object]] = []
    for model_dir in sorted(outer_model_dirs, key=lambda path: (_outer_fold_index(path) is None, _outer_fold_index(path) or 0, str(path))):
        metrics = load_case_level_test_metrics(model_dir, property_name)
        if metrics is None:
            continue
        metrics_json = model_dir / "final_evaluation_metrics.json"
        if metrics_json.exists():
            with metrics_json.open("r", encoding="utf-8") as handle:
                hybrid_metric_payloads.append(json.load(handle))
        fold_rows.append(
            {
                "outer_fold_index": _outer_fold_index(model_dir),
                "model_dir": str(model_dir),
                "selected_trial_id": None,
                "selected_trial_num": None,
                "selected_trial_fold_count": None,
                "selected_mean_test_r2": None,
                "selected_std_test_r2": None,
                "selected_mean_test_mae": None,
                "selected_std_test_mae": None,
                "selected_mean_test_rmse": None,
                "selected_std_test_rmse": None,
                "selected_inner_predictions_file": None,
                "outer_predictions_file": metrics["predictions_file"],
                "outer_test_r2": float(metrics["test_r2"]),
                "outer_test_mae": float(metrics["test_mae"]),
                "outer_test_rmse": float(metrics["test_rmse"]),
                "outer_test_row_count": int(metrics["test_row_count"]),
            }
        )

    if not fold_rows:
        return {}

    fold_df = pd.DataFrame(fold_rows)
    for metric in ["outer_test_r2", "outer_test_mae", "outer_test_rmse"]:
        fold_df[metric] = pd.to_numeric(fold_df[metric], errors="coerce")
    fold_df = fold_df.dropna(subset=["outer_test_r2", "outer_test_mae", "outer_test_rmse"]).reset_index(drop=True)
    if fold_df.empty:
        return {}

    summary_test_mae = float(fold_df["outer_test_mae"].mean())
    representative_df = fold_df.assign(
        _distance_to_summary_test_mae=(fold_df["outer_test_mae"] - summary_test_mae).abs(),
        _sort_outer_fold=pd.to_numeric(fold_df["outer_fold_index"], errors="coerce").fillna(float("inf")),
    ).sort_values(
        ["_distance_to_summary_test_mae", "outer_test_mae", "outer_test_r2", "_sort_outer_fold"],
        ascending=[True, True, False, True],
        na_position="last",
        kind="mergesort",
    )
    representative_row = representative_df.iloc[0]
    representative_fold = representative_row["outer_fold_index"]
    representative_fold = int(representative_fold) if pd.notna(representative_fold) else pd.NA

    summary_payload = {
        property_name: {
            "trial_count": int(len(fold_df)),
            "fold_count": int(len(fold_df)),
            "summary_test_r2": float(fold_df["outer_test_r2"].mean()),
            "summary_test_r2_std": _std_or_zero(fold_df["outer_test_r2"]),
            "summary_test_mae": summary_test_mae,
            "summary_test_mae_std": _std_or_zero(fold_df["outer_test_mae"]),
            "summary_test_rmse": float(fold_df["outer_test_rmse"].mean()),
            "summary_test_rmse_std": _std_or_zero(fold_df["outer_test_rmse"]),
            "representative_selection_mode": "closest_summary_test_mae_outer_fold_oodtest_fast",
            "representative_trial_id": pd.NA,
            "representative_fold": representative_fold,
            "representative_test_r2": float(representative_row["outer_test_r2"]),
            "representative_test_mae": float(representative_row["outer_test_mae"]),
            "representative_test_rmse": float(representative_row["outer_test_rmse"]),
            "representative_predictions_file": representative_row["outer_predictions_file"],
            "representative_model_dir": str(representative_row["model_dir"]),
            "loco_outer_fold_best_details": fold_rows,
        }
    }
    summary_payload[property_name].update(aggregate_hybrid_test_set_metrics(hybrid_metric_payloads))
    return summary_payload


def collect_rows(base_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    subset_audit_rows: list[dict[str, object]] = []
    loco_case_models: dict[tuple[str, str, str, str, str, Path], list[Path]] = {}

    experiment_dirs = [
        *iter_experiment_dirs(base_dir, "experiment1_all_ml_models_"),
        *iter_experiment_dirs(base_dir, "experiment_hybrid_all_ml_models_"),
    ]
    for experiment_dir in sorted(set(experiment_dirs)):
        expected_raw_methods = expected_raw_methods_for_experiment(experiment_dir.name)
        # The output layout is fixed; scanning the exact Traditional depth is
        # much faster than a recursive walk over every trial artifact.
        model_comparison_dirs = [
            *experiment_dir.glob("*/*/*/*/tradition/model_comparison"),
            *experiment_dir.glob("*/*/*/*/tradition/folds/fold_*/model_comparison"),
        ]
        for model_comparison_dir in sorted(set(model_comparison_dirs)):
            relative_parts = model_comparison_dir.relative_to(experiment_dir).parts
            if len(relative_parts) == 6 and relative_parts[4] == "tradition":
                alloy_family, dataset_name, property_name, raw_ood_method, _, _ = relative_parts
                if expected_raw_methods and raw_ood_method not in expected_raw_methods:
                    continue
                ood_method = normalize_ood_method(raw_ood_method)

                for model_dir in sorted(model_comparison_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model_label = MODEL_MAP.get(model_dir.name)
                    if model_label is None:
                        continue

                    summaries = summarize_optuna_model_trials(
                        model_dir / "predictions" / "optuna_trials",
                        property_name=property_name,
                    )
                    if not summaries:
                        continue

                    for summary_property, summary in summaries.items():
                        rows.append(
                            build_summary_row(
                                alloy_family=alloy_family,
                                dataset_name=dataset_name,
                                property_name=summary_property,
                                ood_method=ood_method,
                                model_label=model_label,
                                model_dir=model_dir,
                                experiment_dir=experiment_dir,
                                summary=summary,
                                subset_audit_rows=subset_audit_rows,
                            )
                        )
                continue

            if len(relative_parts) == 8 and relative_parts[4] == "tradition" and relative_parts[5] == "folds":
                alloy_family, dataset_name, property_name, raw_ood_method, _, _, _, _ = relative_parts
                if expected_raw_methods and raw_ood_method not in expected_raw_methods:
                    continue
                for model_dir in sorted(model_comparison_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model_label = MODEL_MAP.get(model_dir.name)
                    if model_label is None:
                        continue
                    key = (
                        normalize_alloy_family_name(alloy_family),
                        dataset_name,
                        property_name,
                        normalize_ood_method(raw_ood_method),
                        model_label,
                        experiment_dir,
                    )
                    loco_case_models.setdefault(key, []).append(model_dir)

    for (alloy_family, dataset_name, property_name, ood_method, model_label, experiment_dir), model_dirs in sorted(loco_case_models.items()):
        if str(ood_method) in {"LOCO", "RandomCV", "HybridHigh20+LOCO", "HybridHigh20+RandCV"}:
            summaries = summarize_outer_fold_case_level_metrics(
                model_dirs,
                property_name=property_name,
            )
        else:
            summaries = summarize_optuna_model_trials_from_dirs(
                [model_dir / "predictions" / "optuna_trials" for model_dir in model_dirs],
                property_name=property_name,
            )
        if not summaries:
            continue

        for summary_property, summary in summaries.items():
            representative_model_dir = summary.get("representative_model_dir") or str(model_dirs[0])
            rows.append(
                build_summary_row(
                    alloy_family=alloy_family,
                    dataset_name=dataset_name,
                    property_name=summary_property,
                    ood_method=ood_method,
                    model_label=model_label,
                    model_dir=Path(str(representative_model_dir)),
                    experiment_dir=experiment_dir,
                    summary=summary,
                    subset_audit_rows=subset_audit_rows,
                )
            )

    return rows, subset_audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize Traditional multi-OOD experiment results.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/ood_results"),
        help="Base directory containing Traditional multi-OOD result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to output/ood_summary_reports/Traditional.",
    )
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    summary_root = args.output_dir or Path("output/ood_summary_reports/Traditional")
    reset_output_dir(summary_root)

    rows, subset_audit_rows = collect_rows(args.base_dir)
    if not rows:
        print(f"No Traditional multi-OOD metrics were collected under: {args.base_dir}")
        return

    summary_df = align_summary_metrics_to_artifact(pd.DataFrame(rows))
    summary_df = annotate_family_ranks(summary_df)
    summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    create_global_exports(summary_df, summary_root, "all_traditional_ood_model_summary.csv")
    save_subset_labeling_audit(subset_audit_rows, summary_root)
    export_case_outputs(summary_df, summary_root)

    print(f"Traditional OOD summary complete: {summary_root}")
    print(f"Cases: {summary_df[['alloy_family', 'dataset_name', 'property']].drop_duplicates().shape[0]}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
