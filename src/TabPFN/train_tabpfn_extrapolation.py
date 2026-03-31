"""
Train and evaluate TabPFN on low-to-high single-target extrapolation splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    from .extrapolation_data_processor import TabPFNExtrapolationDataProcessor
    from .model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from .prediction_alignment import align_df_to_reference_id_order
    from .tabpfn_configs import TABPFN_MODEL_CONFIG
    from .tabpfn_extrapolation_configs import (
        get_all_tabpfn_extrapolation_alloys,
        get_tabpfn_extrapolation_config,
    )
except ImportError:  # pragma: no cover
    from extrapolation_data_processor import TabPFNExtrapolationDataProcessor
    from model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from prediction_alignment import align_df_to_reference_id_order
    from tabpfn_configs import TABPFN_MODEL_CONFIG
    from tabpfn_extrapolation_configs import (
        get_all_tabpfn_extrapolation_alloys,
        get_tabpfn_extrapolation_config,
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


def get_extrapolation_output_root(
    base_path: str | Path,
    model_info: Dict[str, str],
    output_root: str | None = None,
) -> Path:
    if output_root:
        return Path(base_path) / output_root
    return Path(base_path) / "output" / f"extrapolation_results_{model_info['model_run_dirname']}"


class TabPFNExtrapolationTrainer:
    def __init__(
        self,
        alloy_type: str,
        target_col: str,
        base_path: str = ".",
        output_root: str | None = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        extrapolation_side: Optional[str] = None,
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
        self.runtime_info = get_tabpfn_runtime_config(
            base_path=self.base_path,
            backend=backend,
            preferred_model_version=model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        self.config = get_tabpfn_extrapolation_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
            base_path=str(self.base_path),
        )
        self.resolved_backend = str(self.runtime_info.get("resolved_backend", backend))
        if test_size is not None:
            self.config["test_size"] = test_size
        if random_state is not None:
            self.config["random_state"] = random_state
        if extrapolation_side is not None:
            self.config["extrapolation_side"] = extrapolation_side

        self.processor = TabPFNExtrapolationDataProcessor(self.config, base_path=str(self.base_path))
        self.model = None
        self.model_info: Dict[str, Any] = dict(self.runtime_info)
        self.results: Dict[str, Any] = {}
        self.dataset_name = Path(self.config["raw_data"]).stem
        self.result_dir = (
            get_extrapolation_output_root(self.base_path, self.runtime_info, self.output_root)
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
        train_df, test_df, summary = self.processor.split_low_high(
            feature_frame=feature_frame,
            target_col=self.target_col,
            test_size=float(self.config["test_size"]),
            extrapolation_side=str(self.config["extrapolation_side"]),
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
            "Split complete: train=%d, extrapolation_test=%d, train_max=%.4f, test_min=%.4f",
            summary.train_size,
            summary.test_size_rows,
            summary.train_target_max,
            summary.test_target_min,
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
            "extrapolation_test": {
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
            self.results["extrapolation_test"]["r2"],
            self.results["extrapolation_test"]["mae"],
            self.results["extrapolation_test"]["rmse"],
        )

    def plot_predictions(self, save_path: str) -> None:
        if not self.results:
            raise ValueError("No results available to plot")

        y_pred_test = self.results["predictions"]["test"]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.y_test, y_pred_test, alpha=0.6, s=50, edgecolors="k", linewidths=0.5)

        min_val = min(np.min(self.y_test), np.min(y_pred_test))
        max_val = max(np.max(self.y_test), np.max(y_pred_test))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

        ax.set_xlabel(f"Actual {self.target_col}")
        ax.set_ylabel(f"Predicted {self.target_col}")
        ax.set_title(
            f"{self.alloy_type} - {self.target_col}\n"
            f"Extrapolation R2 = {self.results['extrapolation_test']['r2']:.4f}, "
            f"MAE = {self.results['extrapolation_test']['mae']:.2f}"
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

        train_df = pd.DataFrame(
            {
                "ID": self.ids_train,
                f"{self.target_col}_Actual": self.y_train,
                f"{self.target_col}_Predicted": self.results["predictions"]["train"],
                "Dataset": "Train",
            }
        )
        test_df = pd.DataFrame(
            {
                "ID": self.ids_test,
                f"{self.target_col}_Actual": self.y_test,
                f"{self.target_col}_Predicted": self.results["predictions"]["test"],
                "Dataset": "ExtrapolationTest",
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

    def save_metrics(self, save_path: str) -> None:
        metrics_payload = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "raw_row_count": self.raw_row_count,
            "model_name": self.model_info["model_name"],
            "model_dirname": self.model_info["model_dirname"],
            "feature_mode": self.model_info["feature_mode"],
            "feature_mode_dirname": self.model_info["feature_mode_dirname"],
            "requested_backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "model_info": self.model_info,
            "train": self.results["train"],
            "extrapolation_test": self.results["extrapolation_test"],
            "split_summary": self.summary.to_dict() if self.summary else None,
        }
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_path_obj.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Metrics saved to %s", save_path_obj)

    def save_manifest(self) -> None:
        manifest = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "raw_data": self.config["raw_data"],
            "result_dir": str(self.result_dir),
            "test_size": self.config["test_size"],
            "random_state": self.config["random_state"],
            "extrapolation_side": self.config["extrapolation_side"],
            "model_name": self.model_info["model_name"],
            "model_dirname": self.model_info["model_dirname"],
            "feature_mode": self.model_info["feature_mode"],
            "feature_mode_dirname": self.model_info["feature_mode_dirname"],
            "requested_backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "model_info": self.model_info,
            "feature_cols": self.processor.available_feature_cols,
            "split_artifacts": self.split_artifacts,
        }
        (self.result_dir / "pipeline_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Manifest saved to %s", self.result_dir / "pipeline_manifest.json")

    def run(self, scale: bool = True, drop_na: bool = True, align_predictions: bool = True) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("Starting single-target extrapolation run: %s - %s", self.alloy_type, self.target_col)
        logger.info("Result directory: %s", self.result_dir)
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


def run_single_extrapolation_experiment(
    alloy_type: str,
    target_col: str,
    base_path: str = ".",
    output_root: str | None = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    extrapolation_side: Optional[str] = None,
    align_predictions: bool = True,
    backend: str = "auto",
    model_version: Optional[str] = None,
    feature_mode: str | None = None,
) -> Dict[str, Any]:
    trainer = TabPFNExtrapolationTrainer(
        alloy_type=alloy_type,
        target_col=target_col,
        base_path=base_path,
        output_root=output_root,
        test_size=test_size,
        random_state=random_state,
        extrapolation_side=extrapolation_side,
        backend=backend,
        model_version=model_version,
        feature_mode=feature_mode,
    )
    return trainer.run(align_predictions=align_predictions)


def run_all_extrapolation_experiments(
    base_path: str = ".",
    output_root: str | None = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    extrapolation_side: Optional[str] = None,
    align_predictions: bool = True,
    backend: str = "auto",
    model_version: Optional[str] = None,
    feature_mode: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    all_results: Dict[str, Dict[str, Any]] = {}
    alloy_types = get_all_tabpfn_extrapolation_alloys(
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    total_targets = 0
    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_extrapolation_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        total_targets += len(alloy_config["targets"])

    logger.info(
        "Starting batch extrapolation run across %d alloys and %d targets",
        len(alloy_types),
        total_targets,
    )

    for alloy_type in alloy_types:
        alloy_config = get_tabpfn_extrapolation_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        )
        logger.info("Processing alloy %s with targets %s", alloy_type, alloy_config["targets"])
        alloy_results: Dict[str, Any] = {}
        for target_col in alloy_config["targets"]:
            try:
                alloy_results[target_col] = run_single_extrapolation_experiment(
                    alloy_type=alloy_type,
                    target_col=target_col,
                    base_path=base_path,
                    output_root=output_root,
                    test_size=test_size,
                    random_state=random_state,
                    extrapolation_side=extrapolation_side,
                    align_predictions=align_predictions,
                    backend=backend,
                    model_version=model_version,
                    feature_mode=feature_mode,
                )
            except Exception as exc:
                logger.exception(
                    "Error in extrapolation run for %s - %s: %s",
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


def save_extrapolation_summary_results(
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
                    "Extrapolation_Test_N": result["extrapolation_test"]["n_samples"],
                    "Extrapolation_Test_MAE": result["extrapolation_test"]["mae"],
                    "Extrapolation_Test_RMSE": result["extrapolation_test"]["rmse"],
                    "Extrapolation_Test_R2": result["extrapolation_test"]["r2"],
                    "Extrapolation_Test_MAPE": result["extrapolation_test"]["mape"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    output_dir = get_extrapolation_output_root(base_path, runtime_info, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Batch extrapolation summary saved to %s", summary_path)
    return summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TabPFN extrapolation trainer for single-target or all-target runs",
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
    parser.add_argument("--random_state", default=None, type=int)
    parser.add_argument("--extrapolation_side", default=None, type=str)
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
            get_extrapolation_output_root(args.base_path, runtime_info, args.output_root)
            / "batch_logs"
            / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    else:
        resolved_config = get_tabpfn_extrapolation_config(
            args.alloy_type,
            backend=args.backend,
            feature_mode=args.feature_mode,
            base_path=args.base_path,
        )
        default_log_file = (
            get_extrapolation_output_root(args.base_path, runtime_info, args.output_root)
            / args.alloy_type
            / Path(resolved_config["raw_data"]).stem
            / args.target_col
            / "logs"
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    configure_logging(default_log_file)

    try:
        if args.all:
            all_results = run_all_extrapolation_experiments(
                base_path=args.base_path,
                output_root=args.output_root,
                test_size=args.test_size,
                random_state=args.random_state,
                extrapolation_side=args.extrapolation_side,
                align_predictions=not args.disable_alignment,
                backend=args.backend,
                model_version=args.model_version,
                feature_mode=args.feature_mode,
            )
            summary_path = save_extrapolation_summary_results(
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
                "success_count": success_count,
                "failure_count": failure_count,
                "summary_results_csv": str(summary_path),
            }
            logger.info("Batch extrapolation summary:\n%s", json.dumps(payload, indent=2, ensure_ascii=False))
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            results = run_single_extrapolation_experiment(
                alloy_type=args.alloy_type,
                target_col=args.target_col,
                base_path=args.base_path,
                output_root=args.output_root,
                test_size=args.test_size,
                random_state=args.random_state,
                extrapolation_side=args.extrapolation_side,
                align_predictions=not args.disable_alignment,
                backend=args.backend,
                model_version=args.model_version,
                feature_mode=args.feature_mode,
            )
            logger.info("Final extrapolation metrics:\n%s", json.dumps(results["extrapolation_test"], indent=2, ensure_ascii=False))
            print(json.dumps(results["extrapolation_test"], indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.exception("Error in extrapolation run: %s", exc)
        print(f"Error in extrapolation run: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
# python src\TabPFN\train_tabpfn_extrapolation.py --all --backend api --feature_mode text
# python src\TabPFN\train_tabpfn_extrapolation.py --all --backend api --feature_mode numeric
# python src\TabPFN\train_tabpfn_extrapolation.py --all --backend local --feature_mode numeric --model_version v2
