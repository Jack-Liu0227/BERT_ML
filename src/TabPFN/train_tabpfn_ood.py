"""
Train and evaluate TabPFN on supported single-split OOD strategies.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing.strength_ood_common import PreparedFold, PreparedSplit, save_json, save_prepared_split

warnings.filterwarnings("ignore")

try:
    from .ood_data_processor import TabPFNOODDataProcessor
    from .model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from .prediction_alignment import align_df_to_reference_id_order
    from .tabpfn_configs import TABPFN_MODEL_CONFIG
    from .tabpfn_ood_configs import (
        get_all_tabpfn_ood_alloys,
        get_tabpfn_ood_config,
    )
except ImportError:  # pragma: no cover
    from ood_data_processor import TabPFNOODDataProcessor
    from model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from prediction_alignment import align_df_to_reference_id_order
    from tabpfn_configs import TABPFN_MODEL_CONFIG
    from tabpfn_ood_configs import (
        get_all_tabpfn_ood_alloys,
        get_tabpfn_ood_config,
    )


logger = logging.getLogger(__name__)
TABPFN_AVAILABLE = True


def configure_logging(log_file: Optional[Path] = None) -> None:
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in handlers:
        logger.addHandler(handler)


def sanitize_target_name(target_col: str) -> str:
    return (
        target_col.replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        .replace("/", "_")
    )


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.asarray(y_true) != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask]) / np.asarray(y_true)[mask])) * 100)


def get_ood_output_root(
    base_path: str | Path,
    model_info: Dict[str, str],
    output_root: str | None = None,
) -> Path:
    if output_root:
        return Path(base_path) / output_root
    return Path(base_path) / "output" / f"ood_results_{model_info['model_run_dirname']}"


OOD_METHOD_OUTPUT_SUFFIXES = {
    "sparse_x_single": "sparse_x_single_k5",
    "sparse_y_single": "sparse_y_single_k5",
    "sparse_x_cluster": "sparse_x_cluster_k5",
    "sparse_y_cluster": "sparse_y_cluster_k5",
    "loco": "loco_k5",
    "hybrid_extrapolation_sparse_x_single": "hybrid_extrapolation_sparse_x_single_k5",
    "hybrid_extrapolation_sparse_y_single": "hybrid_extrapolation_sparse_y_single_k5",
    "hybrid_extrapolation_sparse_x_cluster": "hybrid_extrapolation_sparse_x_cluster_k5",
    "hybrid_extrapolation_sparse_y_cluster": "hybrid_extrapolation_sparse_y_cluster_k5",
    "hybrid_extrapolation_loco": "hybrid_extrapolation_loco_k5",
    "hybrid_extrapolation_random_cv": "hybrid_extrapolation_random_cv_k5",
}


def get_ood_method_output_root(
    base_path: str | Path,
    model_info: Dict[str, str],
    split_strategy: str,
    output_root: str | None = None,
) -> Path:
    method_dir = OOD_METHOD_OUTPUT_SUFFIXES.get(split_strategy, split_strategy)
    return get_ood_output_root(base_path, model_info, output_root) / method_dir


class TabPFNOODTrainer:
    def __init__(
        self,
        alloy_type: str,
        target_col: str,
        base_path: str = ".",
        output_root: str | None = None,
        test_size: Optional[float] = None,
        outer_test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        split_strategy: str = "target_extrapolation",
        extrapolation_side: Optional[str] = None,
        sparse_candidate_pool_size: int = 500,
        sparse_cluster_count: int = 5,
        sparse_samples_per_cluster: int = 1,
        sparse_kde_bandwidth: float | None = None,
        sparse_neighbors_per_seed: int = 5,
        loco_cluster_count: int = 5,
        baseline_num_folds: int = 5,
        backend: str = "auto",
        model_version: Optional[str] = None,
        feature_mode: str | None = None,
    ):
        self.alloy_type = alloy_type
        self.target_col = target_col
        self.base_path = Path(base_path)
        self.output_root = output_root
        self.backend = backend
        self.model_version = model_version
        self.feature_mode = feature_mode
        self.split_strategy = split_strategy
        self.runtime_info = get_tabpfn_runtime_config(
            base_path=self.base_path,
            backend=backend,
            preferred_model_version=model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        self.config = get_tabpfn_ood_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
            base_path=str(self.base_path),
        )
        self.resolved_backend = str(self.runtime_info.get("resolved_backend", backend))
        if test_size is not None:
            self.config["test_size"] = test_size
        if outer_test_size is not None:
            self.config["outer_test_size"] = outer_test_size
        if random_state is not None:
            self.config["random_state"] = random_state
        self.config["split_strategy"] = split_strategy
        if extrapolation_side is not None:
            self.config["extrapolation_side"] = extrapolation_side
        self.config["sparse_candidate_pool_size"] = sparse_candidate_pool_size
        self.config["sparse_cluster_count"] = sparse_cluster_count
        self.config["sparse_samples_per_cluster"] = sparse_samples_per_cluster
        self.config["sparse_kde_bandwidth"] = sparse_kde_bandwidth
        self.config["sparse_neighbors_per_seed"] = sparse_neighbors_per_seed
        self.config["loco_cluster_count"] = loco_cluster_count
        self.config["baseline_num_folds"] = baseline_num_folds

        self.processor = TabPFNOODDataProcessor(self.config, base_path=str(self.base_path))
        self.model = None
        self.model_info: Dict[str, Any] = dict(self.runtime_info)
        self.results: Dict[str, Any] = {}
        self.test_result_key = "ood_test"
        self.dataset_name = Path(self.config["raw_data"]).stem
        self.result_dir = (
            get_ood_method_output_root(
                self.base_path,
                self.runtime_info,
                str(self.config["split_strategy"]),
                self.output_root,
            )
            / self.alloy_type
            / self.dataset_name
            / self.target_col
        )
        self.split_artifacts: Dict[str, str] = {}
        self.summary = None

    def prepare_data(self, scale: bool = True, drop_na: bool = True) -> None:
        logger.info("Preparing data for %s - %s", self.alloy_type, self.target_col)
        logger.info(
            "Backend selection: requested=%s resolved=%s feature_mode=%s",
            self.backend,
            self.resolved_backend,
            self.runtime_info.get("feature_mode"),
        )
        logger.info("Raw data: %s", self.base_path / self.config["raw_data"])
        raw_df = self.processor.load_data()
        feature_frame = self.processor.prepare_feature_frame(self.target_col, drop_na=drop_na)
        train_df, test_df, summary = self.processor.prepare_ood_split(
            feature_frame=feature_frame,
            target_col=self.target_col,
            split_strategy=str(self.config["split_strategy"]),
            test_size=float(self.config["test_size"]),
            outer_test_size=float(self.config.get("outer_test_size", 0.2)),
            extrapolation_side=str(self.config["extrapolation_side"]),
            sparse_candidate_pool_size=int(self.config["sparse_candidate_pool_size"]),
            sparse_cluster_count=int(self.config["sparse_cluster_count"]),
            sparse_samples_per_cluster=int(self.config["sparse_samples_per_cluster"]),
            sparse_kde_bandwidth=self.config["sparse_kde_bandwidth"],
            sparse_neighbors_per_seed=int(self.config["sparse_neighbors_per_seed"]),
            loco_cluster_count=int(self.config["loco_cluster_count"]),
            baseline_num_folds=int(self.config["baseline_num_folds"]),
        )

        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.split_artifacts = self.processor.save_split_artifacts(
            train_df=train_df,
            test_df=test_df,
            summary=summary,
            output_dir=str(self.result_dir / "split_data"),
        )
        self.summary = summary
        self.raw_row_count = len(raw_df)
        logger.info(
            "Loaded %d raw rows, retained %d rows with %d usable features",
            len(raw_df),
            len(feature_frame),
            len(self.processor.available_feature_cols),
        )
        logger.info(
            "Split complete: strategy=%s train=%d %s=%d train_max=%.4f test_min=%.4f",
            summary["split_strategy"],
            summary["train_size"],
            summary["test_label"],
            summary["test_size"],
            summary["train_target_max"],
            summary["test_target_min"],
        )
        logger.info("Split artifacts saved under %s", self.result_dir / "split_data")
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.ids_train,
            self.ids_test,
        ) = self.processor.build_model_inputs(train_df, test_df, self.target_col, scale=scale)

    def train_regression(self) -> Dict[str, Any]:
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is not available. Please install it first.")

        logger.info("Initializing TabPFN regressor")
        self.model, self.model_info = create_tabpfn_regressor(
            base_path=self.base_path,
            preferred_model_version=self.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            backend=self.backend,
            feature_mode=self.feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        logger.info(
            "Using model: %s (%s)",
            self.model_info.get("model_name", "unknown"),
            self.model_info.get("feature_mode", "unknown"),
        )

        logger.info("Fitting model on %d training rows", len(self.y_train))
        try:
            self.model.fit(self.X_train, self.y_train)
        except Exception as e:
            backend = self.model_info.get("backend", self.backend)
            model_version = self.model_info.get(
                "preferred_model_version",
                self.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            )
            if backend == "local":
                raise RuntimeError(
                    "TabPFN V2 local fit failed. "
                    f"Resolved model version: `{model_version}`. "
                    "Local mode only supports numeric features."
                ) from e

            raise RuntimeError(
                "TabPFN-2.5-Plus API fit failed. Make sure the `.env` file contains a valid "
                "`TABPFN_API_KEY` or `PRIORLABS_API_KEY`."
            ) from e
        logger.info("Generating predictions for train/test splits")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        self.evaluate_regression(y_pred_train, y_pred_test)
        return self.results

    def _create_regressor(self) -> tuple[Any, Dict[str, Any]]:
        logger.info("Initializing TabPFN regressor")
        model, model_info = create_tabpfn_regressor(
            base_path=self.base_path,
            preferred_model_version=self.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            backend=self.backend,
            feature_mode=self.feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        logger.info(
            "Using model: %s (%s)",
            model_info.get("model_name", "unknown"),
            model_info.get("feature_mode", "unknown"),
        )
        return model, model_info

    def _fit_and_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> tuple[Any, Dict[str, Any], np.ndarray, np.ndarray]:
        model, model_info = self._create_regressor()
        logger.info("Fitting model on %d training rows", len(y_train))
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            backend = model_info.get("backend", self.backend)
            model_version = model_info.get(
                "preferred_model_version",
                self.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            )
            if backend == "local":
                raise RuntimeError(
                    "TabPFN V2 local fit failed. "
                    f"Resolved model version: `{model_version}`. "
                    "Local mode only supports numeric features."
                ) from e

            raise RuntimeError(
                "TabPFN-2.5-Plus API fit failed. Make sure the `.env` file contains a valid "
                "`TABPFN_API_KEY` or `PRIORLABS_API_KEY`."
            ) from e

        logger.info("Generating predictions for train/test splits")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        return model, model_info, np.asarray(y_pred_train), np.asarray(y_pred_test)

    def _build_result_payload(
        self,
        model_info: Dict[str, Any],
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "model_info": model_info,
            "train": {
                "mae": float(mean_absolute_error(y_train, y_pred_train)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "r2": float(r2_score(y_train, y_pred_train)),
                "mape": safe_mape(y_train, y_pred_train),
                "n_samples": int(len(y_train)),
            },
            self.test_result_key: {
                "mae": float(mean_absolute_error(y_test, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "r2": float(r2_score(y_test, y_pred_test)),
                "mape": safe_mape(y_test, y_pred_test),
                "n_samples": int(len(y_test)),
            },
            "predictions": {
                "train": np.asarray(y_pred_train),
                "test": np.asarray(y_pred_test),
            },
        }

    def _metrics_payload(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": safe_mape(y_true, y_pred),
            "n_samples": int(len(y_true)),
        }

    def _run_prepared_split(
        self,
        prepared_split: PreparedSplit,
        result_dir: Path,
        align_predictions: bool,
        fold_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        split_artifacts = save_prepared_split(prepared_split, result_dir / "split_data")
        X_train, X_test, y_train, y_test, ids_train, ids_test = self.processor.build_model_inputs(
            prepared_split.train_df,
            prepared_split.test_df,
            self.target_col,
            scale=True,
        )
        model, model_info, y_pred_train, y_pred_test = self._fit_and_predict(X_train, y_train, X_test)
        result = self._build_result_payload(model_info, y_train, y_pred_train, y_test, y_pred_test)
        result["split_summary"] = prepared_split.summary
        result["split_artifacts"] = split_artifacts
        if fold_metadata:
            result.update(fold_metadata)

        test_set_predictions: Dict[str, Dict[str, Any]] = {}
        test_set_metrics: Dict[str, Dict[str, Any]] = {}
        for test_set_name, test_set_df in prepared_split.test_sets.items():
            X_test_set, y_test_set, ids_test_set = self.processor.build_additional_test_inputs(
                test_set_df,
                self.target_col,
                scale=True,
            )
            y_pred_test_set = np.asarray(model.predict(X_test_set))
            test_set_metrics[test_set_name] = self._metrics_payload(y_test_set, y_pred_test_set)
            test_set_predictions[test_set_name] = {
                "ids": ids_test_set,
                "y_true": y_test_set,
                "y_pred": y_pred_test_set,
            }
        if test_set_metrics:
            result["test_set_metrics"] = test_set_metrics

        target_name = sanitize_target_name(self.target_col)
        self._plot_predictions_for_split(
            y_test=y_test,
            y_pred_test=y_pred_test,
            save_path=result_dir / "plots" / f"{target_name}_predictions.png",
        )
        align_ref = None
        if align_predictions:
            align_ref_raw = self.config.get("align_reference_predictions_csv")
            if align_ref_raw:
                align_ref = str(self.base_path / align_ref_raw)
        self._save_predictions_for_split(
            save_path=result_dir / "predictions" / "all_predictions.csv",
            ids_train=ids_train,
            y_train=y_train,
            y_pred_train=y_pred_train,
            ids_test=ids_test,
            y_test=y_test,
            y_pred_test=y_pred_test,
            align_reference_csv=align_ref,
        )
        for test_set_name, payload in test_set_predictions.items():
            self._save_test_set_predictions(
                save_path=result_dir / "predictions" / "test_sets" / f"{test_set_name}_predictions.csv",
                test_set_name=test_set_name,
                ids_test=payload["ids"],
                y_test=payload["y_true"],
                y_pred_test=payload["y_pred"],
            )
        self._save_metrics_payload(
            result=result,
            save_path=result_dir / "metrics" / "metrics_summary.json",
        )
        self._save_manifest_payload(
            result_dir=result_dir,
            model_info=model_info,
            split_artifacts=split_artifacts,
            split_summary=prepared_split.summary,
            fold_metadata=fold_metadata,
        )
        return result

    def evaluate_regression(self, y_pred_train: np.ndarray, y_pred_test: np.ndarray) -> None:
        self.results = {
            "model_info": self.model_info,
            "train": {
                "mae": float(mean_absolute_error(self.y_train, y_pred_train)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_train, y_pred_train))),
                "r2": float(r2_score(self.y_train, y_pred_train)),
                "mape": safe_mape(self.y_train, y_pred_train),
                "n_samples": int(len(self.y_train)),
            },
            self.test_result_key: {
                "mae": float(mean_absolute_error(self.y_test, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_test, y_pred_test))),
                "r2": float(r2_score(self.y_test, y_pred_test)),
                "mape": safe_mape(self.y_test, y_pred_test),
                "n_samples": int(len(self.y_test)),
            },
            "predictions": {
                "train": np.asarray(y_pred_train),
                "test": np.asarray(y_pred_test),
            },
        }
        logger.info(
            "Finished evaluation | train_r2=%.4f test_r2=%.4f test_mae=%.4f test_rmse=%.4f",
            self.results["train"]["r2"],
            self.results[self.test_result_key]["r2"],
            self.results[self.test_result_key]["mae"],
            self.results[self.test_result_key]["rmse"],
        )

    def plot_predictions(self, save_path: str) -> None:
        if not self.results:
            raise ValueError("No results available to plot")
        self._plot_predictions_for_split(
            y_test=self.y_test,
            y_pred_test=self.results["predictions"]["test"],
            save_path=save_path,
        )

    def _plot_predictions_for_split(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        save_path: str | Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors="k", linewidths=0.5)

        min_val = min(np.min(y_test), np.min(y_pred_test))
        max_val = max(np.max(y_test), np.max(y_pred_test))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

        ax.set_xlabel(f"Actual {self.target_col}")
        ax.set_ylabel(f"Predicted {self.target_col}")
        ax.set_title(
            f"{self.alloy_type} - {self.target_col}\n"
            f"OOD Prediction Overview"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal", adjustable="box")

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path_obj, dpi=300, bbox_inches="tight")
        plt.close()

    def save_predictions_csv(self, save_path: str, align_reference_csv: Optional[str] = None) -> pd.DataFrame:
        if not self.results:
            raise ValueError("No results available to save")
        return self._save_predictions_for_split(
            save_path=save_path,
            ids_train=self.ids_train,
            y_train=self.y_train,
            y_pred_train=self.results["predictions"]["train"],
            ids_test=self.ids_test,
            y_test=self.y_test,
            y_pred_test=self.results["predictions"]["test"],
            align_reference_csv=align_reference_csv,
        )

    def _save_predictions_for_split(
        self,
        save_path: str | Path,
        ids_train: np.ndarray,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        ids_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        align_reference_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        train_df = pd.DataFrame(
            {
                "ID": ids_train,
                f"{self.target_col}_Actual": y_train,
                f"{self.target_col}_Predicted": y_pred_train,
                "Dataset": "Train",
            }
        )
        test_df = pd.DataFrame(
            {
                "ID": ids_test,
                f"{self.target_col}_Actual": y_test,
                f"{self.target_col}_Predicted": y_pred_test,
                "Dataset": "OODTest",
            }
        )
        all_data = pd.concat([train_df, test_df], ignore_index=True)

        if align_reference_csv:
            ref_path = Path(align_reference_csv)
            if ref_path.exists():
                ref_df = pd.read_csv(ref_path)
                try:
                    all_data = align_df_to_reference_id_order(all_data, ref_df, id_col="ID")
                except Exception:
                    all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)
            else:
                all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)
        else:
            all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        all_data.to_csv(save_path_obj, index=False)
        logger.info("Predictions saved to %s", save_path_obj)
        return all_data

    def _save_test_set_predictions(
        self,
        save_path: str | Path,
        test_set_name: str,
        ids_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> pd.DataFrame:
        test_df = pd.DataFrame(
            {
                "ID": ids_test,
                f"{self.target_col}_Actual": y_test,
                f"{self.target_col}_Predicted": y_pred_test,
                "Dataset": test_set_name,
            }
        )
        test_df = test_df.sort_values("ID", kind="stable").reset_index(drop=True)
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(save_path_obj, index=False)
        logger.info("Test-set predictions saved to %s", save_path_obj)
        return test_df

    def save_metrics(self, save_path: str) -> None:
        self._save_metrics_payload(self.results, save_path)

    def _save_metrics_payload(self, result: Dict[str, Any], save_path: str | Path) -> None:
        model_info = result["model_info"]
        metrics_payload = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "raw_row_count": getattr(self, "raw_row_count", None),
            "model_name": model_info["model_name"],
            "model_dirname": model_info["model_dirname"],
            "feature_mode": model_info["feature_mode"],
            "feature_mode_dirname": model_info["feature_mode_dirname"],
            "requested_backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "model_info": model_info,
            "train": result["train"],
            self.test_result_key: result[self.test_result_key],
            "split_summary": result.get("split_summary", self.summary),
        }
        if result.get("test_set_metrics"):
            metrics_payload["test_set_metrics"] = result["test_set_metrics"]
        if "fold_index" in result:
            metrics_payload["fold_index"] = result["fold_index"]
        if "held_out_cluster_id" in result:
            metrics_payload["held_out_cluster_id"] = result["held_out_cluster_id"]

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_path_obj.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Metrics saved to %s", save_path_obj)

    def save_manifest(self) -> None:
        self._save_manifest_payload(
            result_dir=self.result_dir,
            model_info=self.model_info,
            split_artifacts=self.split_artifacts,
            split_summary=self.summary,
        )

    def _save_manifest_payload(
        self,
        result_dir: Path,
        model_info: Dict[str, Any],
        split_artifacts: Dict[str, str],
        split_summary: Dict[str, Any] | None,
        fold_metadata: Dict[str, Any] | None = None,
    ) -> None:
        manifest = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "raw_data": self.config["raw_data"],
            "result_dir": str(result_dir),
            "test_size": self.config["test_size"],
            "outer_test_size": self.config.get("outer_test_size", 0.2),
            "random_state": self.config["random_state"],
            "split_strategy": self.config["split_strategy"],
            "extrapolation_side": self.config["extrapolation_side"],
            "sparse_candidate_pool_size": self.config["sparse_candidate_pool_size"],
            "sparse_cluster_count": self.config["sparse_cluster_count"],
            "sparse_samples_per_cluster": self.config["sparse_samples_per_cluster"],
            "sparse_kde_bandwidth": self.config["sparse_kde_bandwidth"],
            "sparse_neighbors_per_seed": self.config["sparse_neighbors_per_seed"],
            "loco_cluster_count": self.config["loco_cluster_count"],
            "baseline_num_folds": self.config["baseline_num_folds"],
            "model_name": model_info["model_name"],
            "model_dirname": model_info["model_dirname"],
            "feature_mode": model_info["feature_mode"],
            "feature_mode_dirname": model_info["feature_mode_dirname"],
            "requested_backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "model_info": model_info,
            "feature_cols": self.processor.available_feature_cols,
            "split_artifacts": split_artifacts,
            "split_summary": split_summary,
        }
        if fold_metadata:
            manifest.update(fold_metadata)
        (result_dir / "pipeline_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Manifest saved to %s", result_dir / "pipeline_manifest.json")

    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        train_metrics = ["mae", "rmse", "r2", "mape", "n_samples"]
        test_metrics = ["mae", "rmse", "r2", "mape", "n_samples"]
        aggregated_train = {
            metric: float(np.mean([fold["train"][metric] for fold in fold_results]))
            for metric in train_metrics
        }
        aggregated_test = {
            metric: float(np.mean([fold[self.test_result_key][metric] for fold in fold_results]))
            for metric in test_metrics
        }
        aggregated: Dict[str, Any] = {
            "fold_count": len(fold_results),
            "train": aggregated_train,
            self.test_result_key: aggregated_test,
        }
        test_set_names = sorted(
            {
                test_set_name
                for fold in fold_results
                for test_set_name in fold.get("test_set_metrics", {}).keys()
            }
        )
        if test_set_names:
            aggregated["test_set_metrics"] = {}
            for test_set_name in test_set_names:
                aggregated["test_set_metrics"][test_set_name] = {
                    metric: float(
                        np.mean(
                            [
                                fold["test_set_metrics"][test_set_name][metric]
                                for fold in fold_results
                                if test_set_name in fold.get("test_set_metrics", {})
                            ]
                        )
                    )
                    for metric in test_metrics
                }
        return aggregated

    def _run_multi_fold(self, align_predictions: bool, drop_na: bool) -> Dict[str, Any]:
        split_strategy = str(self.config["split_strategy"])
        logger.info("Preparing %s folds for %s - %s", split_strategy, self.alloy_type, self.target_col)
        raw_df = self.processor.load_data()
        feature_frame = self.processor.prepare_feature_frame(self.target_col, drop_na=drop_na)
        prepared_result = self.processor.prepare_ood_result(
            feature_frame=feature_frame,
            target_col=self.target_col,
            split_strategy=split_strategy,
            test_size=float(self.config["test_size"]),
            outer_test_size=float(self.config.get("outer_test_size", 0.2)),
            extrapolation_side=str(self.config["extrapolation_side"]),
            sparse_candidate_pool_size=int(self.config["sparse_candidate_pool_size"]),
            sparse_cluster_count=int(self.config["sparse_cluster_count"]),
            sparse_samples_per_cluster=int(self.config["sparse_samples_per_cluster"]),
            sparse_kde_bandwidth=self.config["sparse_kde_bandwidth"],
            sparse_neighbors_per_seed=int(self.config["sparse_neighbors_per_seed"]),
            loco_cluster_count=int(self.config["loco_cluster_count"]),
            baseline_num_folds=int(self.config["baseline_num_folds"]),
        )
        if not isinstance(prepared_result, list):
            raise ValueError(f"{split_strategy} expected a list of prepared folds")

        self.raw_row_count = len(raw_df)
        fold_results: List[Dict[str, Any]] = []
        for prepared_fold in prepared_result:
            fold_dir = self.result_dir / "folds" / f"fold_{prepared_fold.fold_index}"
            fold_metadata = {
                "fold_index": prepared_fold.fold_index,
                "held_out_cluster_id": prepared_fold.held_out_cluster_id,
            }
            logger.info(
                "Running %s fold %d (held_out_cluster_id=%s)",
                split_strategy,
                prepared_fold.fold_index,
                prepared_fold.held_out_cluster_id,
            )
            fold_results.append(
                self._run_prepared_split(
                    prepared_split=prepared_fold.split,
                    result_dir=fold_dir,
                    align_predictions=align_predictions,
                    fold_metadata=fold_metadata,
                )
            )

        aggregated = self._aggregate_fold_results(fold_results)
        multi_fold_manifest = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "split_strategy": split_strategy,
            "fold_count": len(fold_results),
            "folds": [
                {
                    "fold_index": fold["fold_index"],
                    "held_out_cluster_id": fold["held_out_cluster_id"],
                    "split_summary": fold["split_summary"],
                    "train": fold["train"],
                    self.test_result_key: fold[self.test_result_key],
                    "test_set_metrics": fold.get("test_set_metrics", {}),
                }
                for fold in fold_results
            ],
            "aggregated_metrics": {
                "train": aggregated["train"],
                self.test_result_key: aggregated[self.test_result_key],
                "test_set_metrics": aggregated.get("test_set_metrics", {}),
            },
        }
        manifest_name = "loco_manifest.json" if split_strategy == "loco" else f"{split_strategy}_manifest.json"
        save_json(self.result_dir / manifest_name, multi_fold_manifest)

        result = {
            "status": "success",
            "split_strategy": split_strategy,
            "model_info": fold_results[0]["model_info"] if fold_results else self.runtime_info,
            "train": aggregated["train"],
            self.test_result_key: aggregated[self.test_result_key],
            "test_set_metrics": aggregated.get("test_set_metrics", {}),
            "fold_results": fold_results,
            "split_summary": {
                "split_strategy": split_strategy,
                "fold_count": len(fold_results),
            },
            "multi_fold_manifest": str(self.result_dir / manifest_name),
        }
        logger.info(
            "%s completed: folds=%d mean_test_r2=%.4f mean_test_mae=%.4f",
            split_strategy,
            len(fold_results),
            result[self.test_result_key]["r2"],
            result[self.test_result_key]["mae"],
        )
        return result

    def run(self, scale: bool = True, drop_na: bool = True, align_predictions: bool = True) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info(
            "Starting single-target OOD run: %s - %s (%s)",
            self.alloy_type,
            self.target_col,
            self.config["split_strategy"],
        )
        logger.info("Result directory: %s", self.result_dir)
        if self.config["split_strategy"] in {"loco", "random_cv_baseline", "hybrid_extrapolation_loco", "hybrid_extrapolation_random_cv"}:
            return self._run_multi_fold(align_predictions=align_predictions, drop_na=drop_na)
        self.prepare_data(scale=scale, drop_na=drop_na)
        self.train_regression()

        target_name = sanitize_target_name(self.target_col)
        plots_dir = self.result_dir / "plots"
        predictions_dir = self.result_dir / "predictions"
        metrics_dir = self.result_dir / "metrics"

        self.plot_predictions(str(plots_dir / f"{target_name}_predictions.png"))

        align_ref = None
        if align_predictions:
            align_ref_raw = self.config.get("align_reference_predictions_csv")
            if align_ref_raw:
                align_ref = str(self.base_path / align_ref_raw)
        self.save_predictions_csv(
            save_path=str(predictions_dir / "all_predictions.csv"),
            align_reference_csv=align_ref,
        )
        self.save_metrics(str(metrics_dir / "metrics_summary.json"))
        self.save_manifest()
        logger.info("Run completed successfully for %s - %s", self.alloy_type, self.target_col)
        return self.results


def run_single_ood_experiment(
    alloy_type: str,
    target_col: str,
    base_path: str = ".",
    output_root: str | None = None,
    test_size: Optional[float] = None,
    outer_test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    split_strategy: str = "target_extrapolation",
    extrapolation_side: Optional[str] = None,
    sparse_candidate_pool_size: int = 500,
    sparse_cluster_count: int = 5,
    sparse_samples_per_cluster: int = 1,
    sparse_kde_bandwidth: float | None = None,
    sparse_neighbors_per_seed: int = 5,
    loco_cluster_count: int = 5,
    baseline_num_folds: int = 5,
    align_predictions: bool = True,
    backend: str = "auto",
    model_version: Optional[str] = None,
    feature_mode: str | None = None,
) -> Dict[str, Any]:
    trainer = TabPFNOODTrainer(
        alloy_type=alloy_type,
        target_col=target_col,
        base_path=base_path,
        output_root=output_root,
        test_size=test_size,
        outer_test_size=outer_test_size,
        random_state=random_state,
        split_strategy=split_strategy,
        extrapolation_side=extrapolation_side,
        sparse_candidate_pool_size=sparse_candidate_pool_size,
        sparse_cluster_count=sparse_cluster_count,
        sparse_samples_per_cluster=sparse_samples_per_cluster,
        sparse_kde_bandwidth=sparse_kde_bandwidth,
        sparse_neighbors_per_seed=sparse_neighbors_per_seed,
        loco_cluster_count=loco_cluster_count,
        baseline_num_folds=baseline_num_folds,
        backend=backend,
        model_version=model_version,
        feature_mode=feature_mode,
    )
    return trainer.run(align_predictions=align_predictions)


def run_all_ood_experiments(
    base_path: str = ".",
    output_root: str | None = None,
    test_size: Optional[float] = None,
    outer_test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    split_strategy: str = "target_extrapolation",
    extrapolation_side: Optional[str] = None,
    sparse_candidate_pool_size: int = 500,
    sparse_cluster_count: int = 5,
    sparse_samples_per_cluster: int = 1,
    sparse_kde_bandwidth: float | None = None,
    sparse_neighbors_per_seed: int = 5,
    loco_cluster_count: int = 5,
    baseline_num_folds: int = 5,
    align_predictions: bool = True,
    backend: str = "auto",
    model_version: Optional[str] = None,
    feature_mode: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    all_results: Dict[str, Dict[str, Any]] = {}
    alloy_types = get_all_tabpfn_ood_alloys(
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    total_targets = 0
    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_ood_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        total_targets += len(alloy_config["targets"])

    logger.info(
        "Starting batch OOD run across %d alloys and %d targets",
        len(alloy_types),
        total_targets,
    )

    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_ood_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        logger.info("Processing alloy %s with targets %s", alloy_type, alloy_config["targets"])
        alloy_results: Dict[str, Any] = {}
        for target_col in alloy_config["targets"]:
            try:
                alloy_results[target_col] = run_single_ood_experiment(
                    alloy_type=alloy_type,
                    target_col=target_col,
                    base_path=base_path,
                    output_root=output_root,
                    test_size=test_size,
                    outer_test_size=outer_test_size,
                    random_state=random_state,
                    split_strategy=split_strategy,
                    extrapolation_side=extrapolation_side,
                    sparse_candidate_pool_size=sparse_candidate_pool_size,
                    sparse_cluster_count=sparse_cluster_count,
                    sparse_samples_per_cluster=sparse_samples_per_cluster,
                    sparse_kde_bandwidth=sparse_kde_bandwidth,
                    sparse_neighbors_per_seed=sparse_neighbors_per_seed,
                    loco_cluster_count=loco_cluster_count,
                    baseline_num_folds=baseline_num_folds,
                    align_predictions=align_predictions,
                    backend=backend,
                    model_version=model_version,
                    feature_mode=feature_mode,
                )
            except Exception as exc:
                logger.exception(
                    "Error in OOD run for %s - %s: %s",
                    alloy_type,
                    target_col,
                    exc,
                )
                alloy_results[target_col] = {
                    "status": "failed",
                    "error": str(exc),
                }
        all_results[alloy_type] = alloy_results

    return all_results


def save_ood_summary_results(
    all_results: Dict[str, Dict[str, Any]],
    base_path: str,
    runtime_info: Dict[str, Any],
    output_root: str | None = None,
) -> Path:
    summary_rows: list[Dict[str, Any]] = []
    for alloy_type, alloy_results in all_results.items():
        for target_col, result in alloy_results.items():
            if not result:
                continue

            if result.get("status") == "failed":
                summary_rows.append(
                    {
                        "Alloy": alloy_type,
                        "Target": target_col,
                        "Status": "failed",
                        "Error": result.get("error"),
                    }
                )
                continue

            model_info = result["model_info"]
            summary_rows.append(
                {
                    "Alloy": alloy_type,
                    "Target": target_col,
                    "Status": "success",
                    "Split_Strategy": result.get("split_strategy", result.get("split_summary", {}).get("split_strategy")),
                    "Fold_Count": result.get("fold_count", 1),
                    "Model_Name": model_info["model_name"],
                    "Model_Dirname": model_info["model_dirname"],
                    "Feature_Mode": model_info["feature_mode"],
                    "Feature_Mode_Dirname": model_info["feature_mode_dirname"],
                    "Requested_Backend": model_info["requested_backend"],
                    "Resolved_Backend": model_info["resolved_backend"],
                    "Effective_Model_Version": model_info["effective_model_version"],
                    "Train_N": result["train"]["n_samples"],
                    "Train_MAE": result["train"]["mae"],
                    "Train_RMSE": result["train"]["rmse"],
                    "Train_R2": result["train"]["r2"],
                    "Train_MAPE": result["train"]["mape"],
                    "OOD_Test_N": result["ood_test"]["n_samples"],
                    "OOD_Test_MAE": result["ood_test"]["mae"],
                    "OOD_Test_RMSE": result["ood_test"]["rmse"],
                    "OOD_Test_R2": result["ood_test"]["r2"],
                    "OOD_Test_MAPE": result["ood_test"]["mape"],
                    "High20_Test_N": result.get("test_set_metrics", {})
                    .get("test_extrapolation_high20", {})
                    .get("n_samples"),
                    "High20_Test_MAE": result.get("test_set_metrics", {})
                    .get("test_extrapolation_high20", {})
                    .get("mae"),
                    "High20_Test_RMSE": result.get("test_set_metrics", {})
                    .get("test_extrapolation_high20", {})
                    .get("rmse"),
                    "High20_Test_R2": result.get("test_set_metrics", {})
                    .get("test_extrapolation_high20", {})
                    .get("r2"),
                    "Inner_OOD_Test_N": result.get("test_set_metrics", {})
                    .get("test_inner_ood", {})
                    .get("n_samples"),
                    "Inner_OOD_Test_MAE": result.get("test_set_metrics", {})
                    .get("test_inner_ood", {})
                    .get("mae"),
                    "Inner_OOD_Test_RMSE": result.get("test_set_metrics", {})
                    .get("test_inner_ood", {})
                    .get("rmse"),
                    "Inner_OOD_Test_R2": result.get("test_set_metrics", {})
                    .get("test_inner_ood", {})
                    .get("r2"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    first_success = next(
        (
            result
            for alloy_results in all_results.values()
            for result in alloy_results.values()
            if result and result.get("status") != "failed"
        ),
        None,
    )
    split_strategy = "target_extrapolation"
    if first_success:
        split_strategy = str(
            first_success.get(
                "split_strategy",
                first_success.get("split_summary", {}).get("split_strategy", split_strategy),
            )
        )

    output_dir = get_ood_method_output_root(base_path, runtime_info, split_strategy, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Batch OOD summary saved to %s", summary_path)
    return summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TabPFN OOD trainer for single-target or all-target runs",
        allow_abbrev=False,
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--alloy_type", required=False, type=str)
    parser.add_argument("--target_col", required=False, type=str)
    parser.add_argument("--backend", choices=["auto", "api", "local"], default="auto")
    parser.add_argument("--model_version", choices=["latest", "v2", "v2.5", "v2.6"], default=None)
    parser.add_argument("--feature_mode", choices=["numeric", "text"], default=None)
    parser.add_argument("--base_path", default=str(Path(__file__).resolve().parents[2]), type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--test_size", default=None, type=float)
    parser.add_argument("--outer_test_size", default=None, type=float)
    parser.add_argument("--random_state", default=None, type=int)
    parser.add_argument("--split_strategy", default="target_extrapolation", type=str)
    parser.add_argument("--extrapolation_side", default=None, type=str)
    parser.add_argument("--sparse_candidate_pool_size", default=500, type=int)
    parser.add_argument("--sparse_cluster_count", default=5, type=int)
    parser.add_argument("--sparse_samples_per_cluster", default=1, type=int)
    parser.add_argument("--sparse_kde_bandwidth", default=None, type=float)
    parser.add_argument("--sparse_neighbors_per_seed", default=5, type=int)
    parser.add_argument("--loco_cluster_count", default=5, type=int)
    parser.add_argument("--baseline_num_folds", default=5, type=int)
    parser.add_argument("--disable_alignment", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.all and (args.alloy_type or args.target_col):
        parser.error("`--all` cannot be combined with `--alloy_type` or `--target_col`.")
    if (args.alloy_type is None) ^ (args.target_col is None):
        parser.error("`--alloy_type` and `--target_col` must be provided together unless `--all` is used.")
    if not args.all and not (args.alloy_type and args.target_col):
        parser.error("Provide either `--all` or both `--alloy_type` and `--target_col`.")

    runtime_info = get_tabpfn_runtime_config(
        base_path=args.base_path,
        backend=args.backend,
        preferred_model_version=args.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
        feature_mode=args.feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
    )

    if args.all:
        default_log_file = (
            get_ood_method_output_root(
                args.base_path,
                runtime_info,
                args.split_strategy,
                args.output_root,
            )
            / "batch_logs"
            / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
        resolved_config = get_tabpfn_ood_config(
            args.alloy_type,
            backend=args.backend,
            feature_mode=args.feature_mode,
            base_path=args.base_path,
        )
        default_log_file = (
            get_ood_method_output_root(
                args.base_path,
                runtime_info,
                args.split_strategy,
                args.output_root,
            )
            / args.alloy_type
            / Path(resolved_config["raw_data"]).stem
            / args.target_col
            / "logs"
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    configure_logging(default_log_file)

    try:
        if args.all:
            all_results = run_all_ood_experiments(
                base_path=args.base_path,
                output_root=args.output_root,
                test_size=args.test_size,
                outer_test_size=args.outer_test_size,
                random_state=args.random_state,
                split_strategy=args.split_strategy,
                extrapolation_side=args.extrapolation_side,
                sparse_candidate_pool_size=args.sparse_candidate_pool_size,
                sparse_cluster_count=args.sparse_cluster_count,
                sparse_samples_per_cluster=args.sparse_samples_per_cluster,
                sparse_kde_bandwidth=args.sparse_kde_bandwidth,
                sparse_neighbors_per_seed=args.sparse_neighbors_per_seed,
                loco_cluster_count=args.loco_cluster_count,
                baseline_num_folds=args.baseline_num_folds,
                align_predictions=not args.disable_alignment,
                backend=args.backend,
                model_version=args.model_version,
                feature_mode=args.feature_mode,
            )
            summary_path = save_ood_summary_results(
                all_results,
                base_path=args.base_path,
                runtime_info=runtime_info,
                output_root=args.output_root,
            )
            success_count = 0
            failure_count = 0
            for alloy_results in all_results.values():
                for result in alloy_results.values():
                    if result.get("status") == "failed":
                        failure_count += 1
                    else:
                        success_count += 1
            payload = {
                "mode": "all",
                "split_strategy": args.split_strategy,
                "success_count": success_count,
                "failure_count": failure_count,
                "summary_results_csv": str(summary_path),
            }
            logger.info("Batch OOD summary:\n%s", json.dumps(payload, indent=2, ensure_ascii=False))
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            results = run_single_ood_experiment(
                alloy_type=args.alloy_type,
                target_col=args.target_col,
                base_path=args.base_path,
                output_root=args.output_root,
                test_size=args.test_size,
                outer_test_size=args.outer_test_size,
                random_state=args.random_state,
                split_strategy=args.split_strategy,
                extrapolation_side=args.extrapolation_side,
                sparse_candidate_pool_size=args.sparse_candidate_pool_size,
                sparse_cluster_count=args.sparse_cluster_count,
                sparse_samples_per_cluster=args.sparse_samples_per_cluster,
                sparse_kde_bandwidth=args.sparse_kde_bandwidth,
                sparse_neighbors_per_seed=args.sparse_neighbors_per_seed,
                loco_cluster_count=args.loco_cluster_count,
                baseline_num_folds=args.baseline_num_folds,
                align_predictions=not args.disable_alignment,
                backend=args.backend,
                model_version=args.model_version,
                feature_mode=args.feature_mode,
            )
            logger.info("Final OOD metrics:\n%s", json.dumps(results["ood_test"], indent=2, ensure_ascii=False))
            print(json.dumps(results["ood_test"], indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.exception("Error in OOD run: %s", exc)
        print(f"Error in OOD run: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
# python src\TabPFN\train_tabpfn_ood.py --all --backend api --feature_mode text
# python src\TabPFN\train_tabpfn_ood.py --all --backend api --feature_mode numeric
# python src\TabPFN\train_tabpfn_ood.py --all --backend local --feature_mode numeric --model_version v2
