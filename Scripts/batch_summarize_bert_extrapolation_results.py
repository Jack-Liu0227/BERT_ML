from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def load_progress_json(progress_json: Path | None) -> dict[str, dict[str, str]]:
    if progress_json is None:
        return {}
    if not progress_json.exists():
        raise FileNotFoundError(f"Progress JSON not found: {progress_json}")
    with progress_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Progress JSON must be an object: {progress_json}")
    progress: dict[str, dict[str, str]] = {}
    for config_name, task_map in payload.items():
        if isinstance(task_map, dict):
            progress[str(config_name)] = {str(task): str(status) for task, status in task_map.items()}
    return progress


def build_progress_task_key(alloy_family: str, property_name: str) -> str:
    return f"{normalize_alloy_family_name(alloy_family)}_{property_name}"


def resolve_progress_status(
    progress: dict[str, dict[str, str]],
    experiment_name: str,
    alloy_family: str,
    property_name: str,
) -> tuple[str, str]:
    task_key = build_progress_task_key(alloy_family, property_name)
    config_progress = progress.get(experiment_name)
    if config_progress is None:
        return task_key, "untracked"
    return task_key, str(config_progress.get(task_key, "missing"))


def expected_raw_ood_method_for_tracked_config(experiment_name: str) -> str | None:
    for method_name in (
        "sparse_x_cluster",
        "sparse_x_single",
        "sparse_y_cluster",
        "sparse_y_single",
        "loco",
    ):
        if experiment_name.endswith(f"_{method_name}"):
            return f"{method_name}_k5"
    return None


def build_summary_row(
    alloy_family: str,
    dataset_name: str,
    property_name: str,
    ood_method: str,
    model_label: str,
    model_dir: Path,
    experiment_dir: Path,
    summary: dict[str, object],
    progress_task_key: str | None = None,
    progress_status: str | None = None,
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
        "batch_progress_task_key": progress_task_key,
        "batch_progress_status": progress_status,
    }


def collect_rows(
    base_dir: Path,
    *,
    progress: dict[str, dict[str, str]] | None = None,
    success_only: bool = False,
    tracked_configs: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, Any]] = []
    progress = progress or {}
    tracked_configs = tracked_configs or set()
    experiment_dirs = sorted(
        path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("experiment2")
    )
    for experiment_dir in experiment_dirs:
        config_is_tracked = experiment_dir.name in tracked_configs or (not tracked_configs and experiment_dir.name in progress)
        for model_name_raw, model_label in MODEL_MAP.items():
            for model_dir in experiment_dir.rglob(model_name_raw):
                relative_parts = model_dir.relative_to(experiment_dir).parts
                if len(relative_parts) != 5:
                    continue

                alloy_family, dataset_name, property_name, raw_ood_method, _ = relative_parts
                expected_raw_ood_method = (
                    expected_raw_ood_method_for_tracked_config(experiment_dir.name)
                    if config_is_tracked
                    else None
                )
                if expected_raw_ood_method is not None and raw_ood_method != expected_raw_ood_method:
                    skipped_rows.append(
                        {
                            "experiment": experiment_dir.name,
                            "alloy_family": normalize_alloy_family_name(alloy_family),
                            "dataset_name": dataset_name,
                            "property": property_name,
                            "ood_method": normalize_ood_method(raw_ood_method),
                            "model": model_label,
                            "batch_progress_task_key": build_progress_task_key(alloy_family, property_name),
                            "batch_progress_status": "legacy_non_k5_artifact",
                            "model_dir": str(model_dir),
                            "reason": f"tracked k5 config expects raw method {expected_raw_ood_method}",
                        }
                    )
                    continue
                progress_task_key, progress_status = resolve_progress_status(
                    progress,
                    experiment_dir.name,
                    alloy_family,
                    property_name,
                )
                if success_only and config_is_tracked and progress_status != "success":
                    skipped_rows.append(
                        {
                            "experiment": experiment_dir.name,
                            "alloy_family": normalize_alloy_family_name(alloy_family),
                            "dataset_name": dataset_name,
                            "property": property_name,
                            "ood_method": normalize_ood_method(raw_ood_method),
                            "model": model_label,
                            "batch_progress_task_key": progress_task_key,
                            "batch_progress_status": progress_status,
                            "model_dir": str(model_dir),
                            "reason": "progress status is not success",
                        }
                    )
                    continue
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
                            progress_task_key=progress_task_key,
                            progress_status=progress_status,
                        )
                    )
    return rows, skipped_rows


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
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional run-batch progress JSON used to exclude tracked incomplete/running tasks.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="When --progress-json is provided, include tracked configs only when task status is success.",
    )
    parser.add_argument(
        "--tracked-config",
        action="append",
        default=[],
        help=(
            "Experiment config name governed by --progress-json. "
            "Can be passed multiple times; untracked experiment2 dirs are summarized from artifacts."
        ),
    )
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    summary_root = args.output_dir or Path("output/ood_summary_reports/BERT")
    reset_output_dir(summary_root)

    progress = load_progress_json(args.progress_json)
    rows, skipped_rows = collect_rows(
        args.base_dir,
        progress=progress,
        success_only=args.success_only,
        tracked_configs=set(args.tracked_config),
    )
    if skipped_rows:
        skipped_output = summary_root / "00_summary_tables" / "skipped_bert_tasks.csv"
        skipped_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped_rows).sort_values(["experiment", "batch_progress_task_key", "model"]).to_csv(
            skipped_output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Skipped BERT tasks: {len(skipped_rows)} -> {skipped_output}")
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
