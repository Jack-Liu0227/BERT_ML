from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _ood_summary_common import (
    align_summary_metrics_to_artifact,
    annotate_family_ranks,
    create_global_exports,
    export_case_outputs,
    normalize_alloy_family_name,
    normalize_ood_method,
    reset_output_dir,
)
from _raw_prediction_stats import (
    summarize_outer_fold_final_test_metrics,
    summarize_optuna_model_trials,
    summarize_optuna_model_trials_from_dirs,
)


MODEL_MAP = {
    "scibert": "SciBERT",
    "matscibert": "MatSciBERT",
    "steelbert": "SteelBERT",
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
        "model_family": "BERT",
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
        "representative_plot_file": str(representative_model_path / "plots" / "best_model_all_sets_comparison.png"),
        "loco_outer_fold_best_count": len(loco_outer_fold_best_details),
        "loco_outer_fold_best_details_json": json.dumps(loco_outer_fold_best_details, ensure_ascii=False),
    }


def collect_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    experiment_dirs = sorted(
        path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("experiment2")
    )
    for experiment_dir in experiment_dirs:
        for model_name_raw, model_label in MODEL_MAP.items():
            for model_dir in experiment_dir.rglob(model_name_raw):
                relative_parts = model_dir.relative_to(experiment_dir).parts
                if len(relative_parts) != 5:
                    continue

                alloy_family, dataset_name, property_name, raw_ood_method, _ = relative_parts
                ood_method = normalize_ood_method(raw_ood_method)
                optuna_trials_dir = model_dir / "predictions" / "optuna_trials"
                if optuna_trials_dir.exists():
                    summaries = summarize_optuna_model_trials(
                        optuna_trials_dir,
                        property_name=property_name,
                    )
                elif ood_method in {"LOCO", "RandomCV"}:
                    loco_model_dirs = sorted(
                        fold_dir
                        for fold_dir in (model_dir / "folds").glob("fold_*")
                        if (fold_dir / "predictions" / "optuna_trials").exists()
                    ) if (model_dir / "folds").exists() else []
                    summaries = summarize_outer_fold_final_test_metrics(
                        loco_model_dirs,
                        property_name=property_name,
                    )
                else:
                    loco_trial_dirs = sorted(
                        fold_dir / "predictions" / "optuna_trials"
                        for fold_dir in (model_dir / "folds").glob("fold_*")
                        if (fold_dir / "predictions" / "optuna_trials").exists()
                    ) if (model_dir / "folds").exists() else []
                    summaries = summarize_optuna_model_trials_from_dirs(
                        loco_trial_dirs,
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch summarize BERT multi-OOD experiment results.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/ood_results"),
        help="Base directory containing BERT multi-OOD result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to output/ood_summary_reports/BERT.",
    )
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    summary_root = args.output_dir or Path("output/ood_summary_reports/BERT")
    reset_output_dir(summary_root)

    rows = collect_rows(args.base_dir)
    if not rows:
        print(f"No BERT multi-OOD metrics were collected under: {args.base_dir}")
        return

    summary_df = align_summary_metrics_to_artifact(pd.DataFrame(rows))
    summary_df = annotate_family_ranks(summary_df)
    summary_df["alloy_family"] = summary_df["alloy_family"].map(normalize_alloy_family_name)
    create_global_exports(summary_df, summary_root, "all_bert_ood_model_summary.csv")
    export_case_outputs(summary_df, summary_root)

    print(f"BERT OOD summary complete: {summary_root}")
    print(f"Cases: {summary_df[['alloy_family', 'dataset_name', 'property']].drop_duplicates().shape[0]}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()
