"""
Independent RandomCV vs LOCO k-sweep runner.

This module intentionally does not modify or route through the existing OOD
batch runner/progress files. It creates an isolated experiment tree under
``output/cv_k_sweep/...`` and summarizes pooled held-out Pearson R vs k.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_processing.strength_ood_common import (
    PreparedFold,
    prepare_loco_folds,
    prepare_random_cv_baseline_folds,
    save_json,
)
from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD, get_alloy_config_ood, list_available_alloys_ood
from src.pipelines.ood_pipeline import _run_single_split_flow

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


LOGGER = logging.getLogger(__name__)

SUPPORTED_METHODS = ("random_cv_baseline", "loco")
DEFAULT_MODELS = ("xgboost", "sklearn_rf", "lightgbm", "mlp", "catboost")
DEFAULT_K_VALUES = tuple(range(2, 11))
DEFAULT_SEEDS = tuple(range(10))
METHOD_LABELS = {
    "random_cv_baseline": "RandomCV",
    "loco": "LOCO",
}


@dataclass(frozen=True)
class SweepTask:
    method: str
    k: int
    seed: int
    alloy_type: str
    dataset_name: str
    target_column: str
    model_name: str
    data_file: Path
    task_root: Path


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _safe_component(value: str) -> str:
    sanitized = str(value)
    for old, new in {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }.items():
        sanitized = sanitized.replace(old, new)
    return sanitized.strip() or "unknown"


def _dataset_name_from_config(raw_data: str | Path) -> str:
    dataset_name = Path(raw_data).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        dataset_name = dataset_name.replace(suffix, "")
    return dataset_name


def _resolve_path(base_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_path / path


def _parse_int_list(values: Sequence[int] | None, default_values: Sequence[int]) -> List[int]:
    resolved = list(default_values if values is None else values)
    resolved = sorted(dict.fromkeys(int(value) for value in resolved))
    if not resolved:
        raise ValueError("integer list cannot be empty")
    return resolved


def _resolve_alloy_types(requested: Sequence[str] | None) -> List[str]:
    available = list_available_alloys_ood()
    if requested is None:
        return available
    invalid = [alloy for alloy in requested if alloy not in available]
    if invalid:
        raise ValueError(f"Unsupported alloy types: {', '.join(invalid)}. Available: {', '.join(available)}")
    return list(requested)


def _resolve_targets_for_alloy(alloy_type: str, requested_targets: Sequence[str] | None) -> List[str]:
    configured_targets = list(get_alloy_config_ood(alloy_type).get("targets") or [])
    if requested_targets is None:
        return configured_targets
    requested = set(requested_targets)
    return [target for target in configured_targets if target in requested]


def _validate_methods(methods: Sequence[str]) -> List[str]:
    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {', '.join(invalid)}. Available: {', '.join(SUPPORTED_METHODS)}")
    return list(dict.fromkeys(methods))


def _validate_models(models: Sequence[str]) -> List[str]:
    invalid = [model for model in models if model not in DEFAULT_MODELS]
    if invalid:
        raise ValueError(f"Unsupported traditional ML models: {', '.join(invalid)}. Available: {', '.join(DEFAULT_MODELS)}")
    return list(dict.fromkeys(models))


def _build_training_args(
    *,
    task: SweepTask,
    alloy_config: Dict[str, Any],
    method: str,
    k: int,
    seed: int,
    mlp_max_iter: int,
    evaluate_after_train: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_file=str(task.data_file),
        result_dir=str(task.task_root),
        target_column=task.target_column,
        processing_cols=alloy_config.get("processing_cols") or [],
        processing_text_column=alloy_config.get("processing_text_column"),
        use_composition_feature=True,
        use_element_embedding=False,
        use_process_embedding=False,
        use_temperature=False,
        embedding_type="tradition",
        models=[task.model_name],
        use_nn=False,
        cross_validate=False,
        num_folds=1,
        test_size=0.2,
        random_state=seed,
        epochs=200,
        patience=30,
        batch_size=256,
        use_optuna=False,
        n_trials=0,
        mlp_max_iter=mlp_max_iter,
        evaluate_after_train=evaluate_after_train,
        run_shap_analysis=False,
        split_strategy=method,
        extrapolation_side="low_to_high",
        sparse_candidate_pool_size=500,
        sparse_cluster_count=50,
        sparse_samples_per_cluster=1,
        sparse_kde_bandwidth=None,
        sparse_neighbors_per_seed=5,
        loco_cluster_count=k,
        baseline_num_folds=k,
    )


def _prepare_folds(raw_df: pd.DataFrame, target_column: str, method: str, k: int, seed: int) -> List[PreparedFold]:
    if method == "random_cv_baseline":
        return prepare_random_cv_baseline_folds(
            df=raw_df,
            target_col=target_column,
            num_folds=k,
            random_state=seed,
        )
    if method == "loco":
        return prepare_loco_folds(
            df=raw_df,
            target_col=target_column,
            cluster_count=k,
            random_state=seed,
        )
    raise ValueError(f"Unsupported method: {method}")


def _prediction_path_for_fold(fold_root: Path, model_name: str) -> Path:
    return fold_root / "model_comparison" / f"{model_name}_results" / "predictions" / "all_predictions.csv"


def _resolve_prediction_columns(df: pd.DataFrame, target_column: str) -> tuple[str, str, str | None]:
    dataset_col = "Dataset" if "Dataset" in df.columns else ("set" if "set" in df.columns else None)
    exact_actual = f"{target_column}_Actual"
    exact_pred = f"{target_column}_Predicted"
    if exact_actual in df.columns and exact_pred in df.columns:
        return exact_actual, exact_pred, dataset_col

    actual_candidates = [col for col in df.columns if col.endswith("_Actual") or col.startswith("True_")]
    pred_candidates = [col for col in df.columns if col.endswith("_Predicted") or col.startswith("Pred_")]
    if len(actual_candidates) == 1 and len(pred_candidates) == 1:
        return actual_candidates[0], pred_candidates[0], dataset_col
    raise ValueError(
        f"Could not resolve prediction columns for target '{target_column}'. "
        f"Columns: {', '.join(df.columns)}"
    )


def _load_ood_predictions(prediction_path: Path, target_column: str) -> pd.DataFrame:
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    df = pd.read_csv(prediction_path)
    actual_col, pred_col, dataset_col = _resolve_prediction_columns(df, target_column)
    if dataset_col is not None:
        labels = df[dataset_col].astype(str).str.lower()
        mask = labels.isin({"oodtest", "test", "extrapolationtest"})
        df = df.loc[mask].copy()
    out = pd.DataFrame(
        {
            "y_true": pd.to_numeric(df[actual_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
    return out


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _compute_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    if predictions.empty:
        return {
            "pearson_r": float("nan"),
            "sklearn_r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "n_test": 0,
        }
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    if len(y_true) < 2:
        sklearn_r2 = float("nan")
    else:
        sklearn_r2 = float(r2_score(y_true, y_pred))
    return {
        "pearson_r": _pearson_r(y_true, y_pred),
        "sklearn_r2": sklearn_r2,
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def _metrics_json_path(task_root: Path) -> Path:
    return task_root / "seed_metrics.json"


def _read_existing_metrics(task_root: Path) -> Dict[str, Any] | None:
    path = _metrics_json_path(task_root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(output_root: Path, status_rows: List[Dict[str, Any]]) -> None:
    if not status_rows:
        return
    status_path = output_root / "run_status.csv"
    pd.DataFrame(status_rows).to_csv(status_path, index=False, encoding="utf-8-sig")


def _run_task(
    *,
    task: SweepTask,
    alloy_config: Dict[str, Any],
    mlp_max_iter: int,
    evaluate_after_train: bool,
) -> Dict[str, Any]:
    task.task_root.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(task.data_file)
    folds = _prepare_folds(
        raw_df=raw_df,
        target_column=task.target_column,
        method=task.method,
        k=task.k,
        seed=task.seed,
    )
    training_args = _build_training_args(
        task=task,
        alloy_config=alloy_config,
        method=task.method,
        k=task.k,
        seed=task.seed,
        mlp_max_iter=mlp_max_iter,
        evaluate_after_train=evaluate_after_train,
    )

    prediction_frames: List[pd.DataFrame] = []
    fold_details: List[Dict[str, Any]] = []
    for prepared_fold in folds:
        fold_root = task.task_root / "folds" / f"fold_{prepared_fold.fold_index}"
        _run_single_split_flow(
            args=training_args,
            prepared_split=prepared_fold.split,
            run_root=fold_root,
            extra_manifest={
                "fold_index": prepared_fold.fold_index,
                "held_out_cluster_id": prepared_fold.held_out_cluster_id,
                "k_sweep": {
                    "method": task.method,
                    "k": task.k,
                    "seed": task.seed,
                    "model": task.model_name,
                },
            },
        )
        pred_path = _prediction_path_for_fold(fold_root, task.model_name)
        fold_predictions = _load_ood_predictions(pred_path, task.target_column)
        fold_predictions["fold_index"] = prepared_fold.fold_index
        prediction_frames.append(fold_predictions)
        fold_details.append(
            {
                "fold_index": int(prepared_fold.fold_index),
                "held_out_cluster_id": int(prepared_fold.held_out_cluster_id),
                "prediction_file": str(pred_path),
                "n_test": int(len(fold_predictions)),
                **prepared_fold.metadata,
            }
        )

    pooled_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = _compute_metrics(pooled_predictions)
    metrics_payload: Dict[str, Any] = {
        "method": task.method,
        "method_label": METHOD_LABELS.get(task.method, task.method),
        "k": int(task.k),
        "seed": int(task.seed),
        "alloy_type": task.alloy_type,
        "dataset_name": task.dataset_name,
        "target_column": task.target_column,
        "model": task.model_name,
        "data_file": str(task.data_file),
        "task_root": str(task.task_root),
        "fold_count": int(len(folds)),
        "actual_k": int(len(folds)),
        "metrics_protocol": "pooled held-out predictions across all outer folds",
        **metrics,
        "fold_details": fold_details,
    }
    save_json(_metrics_json_path(task.task_root), metrics_payload)
    return metrics_payload


def _build_tasks(args: argparse.Namespace, output_root: Path, base_path: Path) -> List[SweepTask]:
    methods = _validate_methods(args.methods)
    models = _validate_models(args.models)
    k_values = _parse_int_list(args.k_values, DEFAULT_K_VALUES)
    seeds = _parse_int_list(args.seeds, DEFAULT_SEEDS)
    alloy_types = _resolve_alloy_types(args.alloy_types)

    tasks: List[SweepTask] = []
    skipped_alloy_targets: List[str] = []
    for alloy_type in alloy_types:
        alloy_config = get_alloy_config_ood(alloy_type)
        targets = _resolve_targets_for_alloy(alloy_type, args.targets)
        if not targets:
            skipped_alloy_targets.append(alloy_type)
            continue
        data_file = _resolve_path(base_path, alloy_config["raw_data"])
        dataset_name = _dataset_name_from_config(alloy_config["raw_data"])
        for method in methods:
            for k in k_values:
                for seed in seeds:
                    for target_column in targets:
                        for model_name in models:
                            task_root = (
                                output_root
                                / method
                                / f"k_{k}"
                                / f"seed_{seed}"
                                / alloy_type
                                / dataset_name
                                / _safe_component(target_column)
                                / model_name
                            )
                            tasks.append(
                                SweepTask(
                                    method=method,
                                    k=int(k),
                                    seed=int(seed),
                                    alloy_type=alloy_type,
                                    dataset_name=dataset_name,
                                    target_column=target_column,
                                    model_name=model_name,
                                    data_file=data_file,
                                    task_root=task_root,
                                )
                            )
    if skipped_alloy_targets:
        LOGGER.warning(
            "No requested targets matched for alloys: %s",
            ", ".join(skipped_alloy_targets),
        )
    if not tasks:
        raise ValueError("No k-sweep tasks were generated. Check --alloy_types/--targets/--methods/--models.")
    return tasks


def _sample_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return float("nan")
    return float(numeric.std(ddof=1))


def _write_summaries(seed_metrics: List[Dict[str, Any]], output_root: Path) -> Dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    seed_df = pd.DataFrame(seed_metrics)
    if seed_df.empty:
        raise ValueError("No successful seed metrics available for summary")

    fold_details_json = seed_df.get("fold_details")
    if fold_details_json is not None:
        seed_df = seed_df.copy()
        seed_df["fold_details_json"] = seed_df["fold_details"].apply(lambda value: json.dumps(value, ensure_ascii=False))
        seed_df = seed_df.drop(columns=["fold_details"])

    seed_path = output_root / "k_sweep_seed_metrics.csv"
    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")

    group_cols = ["method", "method_label", "alloy_type", "dataset_name", "target_column", "model", "k"]
    by_k = (
        seed_df.groupby(group_cols, dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", _sample_std),
            pearson_r_median=("pearson_r", "median"),
            sklearn_r2_mean=("sklearn_r2", "mean"),
            sklearn_r2_std=("sklearn_r2", _sample_std),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", _sample_std),
            mae_mean=("mae", "mean"),
            mae_std=("mae", _sample_std),
            n_test_mean=("n_test", "mean"),
            fold_count_mean=("fold_count", "mean"),
            actual_k_mean=("actual_k", "mean"),
        )
        .reset_index()
        .sort_values(["model", "alloy_type", "target_column", "method", "k"], kind="stable")
    )
    by_k_path = output_root / "k_sweep_by_k_summary.csv"
    by_k.to_csv(by_k_path, index=False, encoding="utf-8-sig")

    across_cols = ["method", "method_label", "alloy_type", "dataset_name", "target_column", "model"]
    across_k = (
        by_k.groupby(across_cols, dropna=False)
        .agg(
            k_count=("k", "nunique"),
            pearson_r_median_across_k=("pearson_r_mean", "median"),
            pearson_r_mean_across_k=("pearson_r_mean", "mean"),
            pearson_r_std_across_k=("pearson_r_mean", _sample_std),
            pearson_r_min_across_k=("pearson_r_mean", "min"),
            pearson_r_max_across_k=("pearson_r_mean", "max"),
            sklearn_r2_median_across_k=("sklearn_r2_mean", "median"),
            rmse_mean_across_k=("rmse_mean", "mean"),
            mae_mean_across_k=("mae_mean", "mean"),
        )
        .reset_index()
        .sort_values(["model", "alloy_type", "target_column", "method"], kind="stable")
    )
    across_k_path = output_root / "k_sweep_across_k_summary.csv"
    across_k.to_csv(across_k_path, index=False, encoding="utf-8-sig")

    plot_paths = _write_plots(by_k, output_root / "plots")
    return {
        "seed_metrics": str(seed_path),
        "by_k_summary": str(by_k_path),
        "across_k_summary": str(across_k_path),
        **plot_paths,
    }


def _write_plots(by_k: pd.DataFrame, plots_dir: Path) -> Dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if by_k.empty:
        return written

    for model_name, model_df in by_k.groupby("model", sort=False):
        cases = (
            model_df[["alloy_type", "dataset_name", "target_column"]]
            .drop_duplicates()
            .sort_values(["alloy_type", "target_column"], kind="stable")
            .to_dict(orient="records")
        )
        if not cases:
            continue

        n_cases = len(cases)
        n_cols = min(3, n_cases)
        n_rows = int(math.ceil(n_cases / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax_index, case in enumerate(cases):
            ax = axes_flat[ax_index]
            case_df = model_df[
                (model_df["alloy_type"] == case["alloy_type"])
                & (model_df["dataset_name"] == case["dataset_name"])
                & (model_df["target_column"] == case["target_column"])
            ]
            for method in SUPPORTED_METHODS:
                method_df = case_df[case_df["method"] == method].sort_values("k")
                if method_df.empty:
                    continue
                yerr = method_df["pearson_r_std"].to_numpy(dtype=float)
                yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                ax.errorbar(
                    method_df["k"].to_numpy(dtype=int),
                    method_df["pearson_r_mean"].to_numpy(dtype=float),
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_title(f"{case['alloy_type']} / {case['target_column']}", fontsize=10)
            ax.set_xlabel("k")
            ax.set_ylabel("Pearson R")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        for ax in axes_flat[n_cases:]:
            ax.axis("off")

        fig.suptitle(f"RandomCV vs LOCO Pearson R vs k - {model_name}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = plots_dir / f"{_safe_component(model_name)}_pearson_r_vs_k.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written[f"plot_{model_name}"] = str(plot_path)
    return written


def _collect_existing_seed_metrics(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("seed_metrics.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            LOGGER.warning("Failed to read existing metrics %s: %s", path, exc)
    return rows


def _write_manifest(
    *,
    args: argparse.Namespace,
    output_root: Path,
    base_path: Path,
    tasks: Sequence[SweepTask] | None,
    summary_paths: Dict[str, str] | None,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": "src.pipelines.run_cv_k_sweep",
        "description": "Independent RandomCV vs LOCO k-sweep; does not use existing OOD batch progress/output roots.",
        "base_path": str(base_path),
        "output_root": str(output_root),
        "methods": list(args.methods),
        "k_values": list(args.k_values if args.k_values is not None else DEFAULT_K_VALUES),
        "seeds": list(args.seeds if args.seeds is not None else DEFAULT_SEEDS),
        "alloy_types": list(args.alloy_types) if args.alloy_types is not None else list_available_alloys_ood(),
        "targets": list(args.targets) if args.targets is not None else "all configured targets",
        "models": list(args.models),
        "metric_primary": "pooled held-out Pearson R",
        "auxiliary_metrics": ["sklearn_r2", "rmse", "mae"],
        "training_protocol": {
            "traditional_ml_only": True,
            "embedding_type": "tradition",
            "use_composition_feature": True,
            "use_optuna": False,
            "cross_validate": False,
            "run_shap_analysis": False,
        },
        "task_count": None if tasks is None else len(tasks),
        "summary_paths": summary_paths or {},
        "alloy_configs": ALLOY_CONFIGS_OOD,
    }
    save_json(output_root / "manifest.json", payload)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent RandomCV vs LOCO k-sweep for traditional ML models.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output_root",
        default="output/cv_k_sweep/experiment1_all_ml_models",
        help="Isolated output root for this k-sweep.",
    )
    parser.add_argument("--base_path", default=str(Path(__file__).resolve().parents[2]), help="Repository base path.")
    parser.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS), choices=SUPPORTED_METHODS)
    parser.add_argument("--k_values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--alloy_types", nargs="+", default=None, choices=list_available_alloys_ood())
    parser.add_argument("--targets", nargs="+", default=None, help="Optional target-name filter applied to each alloy.")
    parser.add_argument("--mlp_max_iter", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Skip tasks with existing seed_metrics.json.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned tasks without training or writing outputs.")
    parser.add_argument("--summarize_only", action="store_true", help="Only summarize existing seed_metrics.json files.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop after the first failed task.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional cap for smoke/debug runs.")
    parser.add_argument("--no_evaluate_after_train", action="store_true", help="Train without final test evaluation.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    base_path = Path(args.base_path).resolve()
    output_root = _resolve_path(base_path, args.output_root).resolve()

    if args.summarize_only:
        seed_metrics = _collect_existing_seed_metrics(output_root)
        summary_paths = _write_summaries(seed_metrics, output_root)
        _write_manifest(args=args, output_root=output_root, base_path=base_path, tasks=None, summary_paths=summary_paths)
        LOGGER.info("Summary-only complete: %s", output_root)
        print(json.dumps({"output_root": str(output_root), **summary_paths}, indent=2, ensure_ascii=False))
        return

    tasks = _build_tasks(args, output_root, base_path)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]

    if args.dry_run:
        LOGGER.info("Dry run: %d tasks would be executed", len(tasks))
        preview = [
            {
                "method": task.method,
                "k": task.k,
                "seed": task.seed,
                "alloy_type": task.alloy_type,
                "target_column": task.target_column,
                "model": task.model_name,
                "task_root": str(task.task_root),
            }
            for task in tasks[:20]
        ]
        print(json.dumps({"task_count": len(tasks), "preview": preview}, indent=2, ensure_ascii=False))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(args=args, output_root=output_root, base_path=base_path, tasks=tasks, summary_paths=None)

    seed_metrics: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, task in enumerate(tasks, start=1):
        task_started = time.time()
        LOGGER.info(
            "[%d/%d] method=%s k=%d seed=%d alloy=%s target=%s model=%s",
            index,
            len(tasks),
            task.method,
            task.k,
            task.seed,
            task.alloy_type,
            task.target_column,
            task.model_name,
        )
        status = "success"
        error = ""
        try:
            existing = _read_existing_metrics(task.task_root) if args.resume else None
            if existing is not None:
                metrics_payload = existing
                status = "skipped"
                LOGGER.info("  skipped existing metrics: %s", _metrics_json_path(task.task_root))
            else:
                alloy_config = get_alloy_config_ood(task.alloy_type)
                metrics_payload = _run_task(
                    task=task,
                    alloy_config=alloy_config,
                    mlp_max_iter=args.mlp_max_iter,
                    evaluate_after_train=not args.no_evaluate_after_train,
                )
            seed_metrics.append(metrics_payload)
            LOGGER.info(
                "  %s | Pearson R=%.4f R2=%.4f RMSE=%.4f MAE=%.4f n=%s",
                status,
                metrics_payload.get("pearson_r", float("nan")),
                metrics_payload.get("sklearn_r2", float("nan")),
                metrics_payload.get("rmse", float("nan")),
                metrics_payload.get("mae", float("nan")),
                metrics_payload.get("n_test", "NA"),
            )
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("  failed: %s", error)
            if args.verbose:
                LOGGER.error(traceback.format_exc())
            if args.stop_on_error:
                status_rows.append(
                    {
                        "status": status,
                        "error": error,
                        "elapsed_seconds": round(time.time() - task_started, 3),
                        **task.__dict__,
                    }
                )
                _write_status(output_root, status_rows)
                raise
        finally:
            row = {
                "status": status,
                "error": error,
                "elapsed_seconds": round(time.time() - task_started, 3),
                "method": task.method,
                "k": task.k,
                "seed": task.seed,
                "alloy_type": task.alloy_type,
                "dataset_name": task.dataset_name,
                "target_column": task.target_column,
                "model": task.model_name,
                "data_file": str(task.data_file),
                "task_root": str(task.task_root),
            }
            status_rows.append(row)
            _write_status(output_root, status_rows)

    summary_paths = _write_summaries(seed_metrics, output_root)
    _write_manifest(args=args, output_root=output_root, base_path=base_path, tasks=tasks, summary_paths=summary_paths)
    payload = {
        "output_root": str(output_root),
        "task_count": len(tasks),
        "success_or_skipped": sum(1 for row in status_rows if row["status"] in {"success", "skipped"}),
        "failed": sum(1 for row in status_rows if row["status"] == "failed"),
        "elapsed_seconds": round(time.time() - started, 3),
        **summary_paths,
    }
    LOGGER.info("k-sweep complete: %s", json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
