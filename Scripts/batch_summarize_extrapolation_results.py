from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _ood_summary_common import (
    annotate_family_ranks,
    create_global_exports,
    export_case_outputs,
    iter_experiment_dirs,
    normalize_alloy_family_name,
    normalize_ood_method,
    reset_output_dir,
)
from _raw_prediction_stats import (
    summarize_loco_outer_fold_best_trials,
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


def build_summary_row(
    alloy_family: str,
    dataset_name: str,
    property_name: str,
    ood_method: str,
    model_label: str,
    model_dir: Path,
    experiment_dir: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    representative_model_dir = summary.get("representative_model_dir") or str(model_dir)
    representative_model_path = Path(str(representative_model_dir))
    loco_outer_fold_best_details = summary.get("loco_outer_fold_best_details") or []
    return {
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


def collect_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    loco_case_models: dict[tuple[str, str, str, str, str, Path], list[Path]] = {}

    for experiment_dir in iter_experiment_dirs(base_dir, "experiment1_all_ml_models_"):
        for model_comparison_dir in experiment_dir.rglob("model_comparison"):
            relative_parts = model_comparison_dir.relative_to(experiment_dir).parts
            if len(relative_parts) == 6 and relative_parts[4] == "tradition":
                alloy_family, dataset_name, property_name, raw_ood_method, _, _ = relative_parts
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
                            )
                        )
                continue

            if len(relative_parts) == 8 and relative_parts[4] == "tradition" and relative_parts[5] == "folds":
                alloy_family, dataset_name, property_name, raw_ood_method, _, _, _, _ = relative_parts
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
        if str(ood_method) == "LOCO":
            summaries = summarize_loco_outer_fold_best_trials(
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
                )
            )

    return rows


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

    rows = collect_rows(args.base_dir)
    if not rows:
        print(f"No Traditional multi-OOD metrics were collected under: {args.base_dir}")
        return

    summary_df = annotate_family_ranks(pd.DataFrame(rows))
    summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    create_global_exports(summary_df, summary_root, "all_traditional_ood_model_summary.csv")
    export_case_outputs(summary_df, summary_root)

    print(f"Traditional OOD summary complete: {summary_root}")
    print(f"Cases: {summary_df[['alloy_family', 'dataset_name', 'property']].drop_duplicates().shape[0]}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
