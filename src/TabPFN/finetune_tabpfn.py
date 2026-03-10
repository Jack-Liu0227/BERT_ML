"""
TabPFN Fine-tuning Script

Fine-tune and evaluate TabPFN regressor models on alloy datasets.
"""

from __future__ import annotations

import warnings
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.finetuning.finetuned_base import EvalResult
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    TABPFN_FINETUNE_AVAILABLE = True
except ImportError:
    print("Warning: TabPFN finetuning not available. Please install/update tabpfn.")
    TABPFN_FINETUNE_AVAILABLE = False

try:
    from .data_processor import TabPFNDataProcessor
    from .prediction_alignment import align_df_to_reference_id_order
    from .tabpfn_configs import get_all_alloy_types, get_tabpfn_config
except ImportError:  # pragma: no cover
    from data_processor import TabPFNDataProcessor
    from prediction_alignment import align_df_to_reference_id_order
    from tabpfn_configs import get_all_alloy_types, get_tabpfn_config


DEFAULT_FINETUNE_PARAMS: Dict[str, Any] = {
    "device": "cuda",
    "epochs": 300,
    "learning_rate": 1e-5,
    "validation_split_ratio": 0.1,
    "early_stopping": True,
    "early_stopping_patience": 5,
    "n_estimators_finetune": 1,
}


def resolve_regressor_model_path() -> str:
    """
    Prefer a local open v2 regressor checkpoint to avoid gated v2.5 downloads.
    """
    user_set_raw = os.environ.get("TABPFN_REGRESSOR_MODEL_PATH", "").strip()
    if user_set_raw:
        user_set = Path(user_set_raw).expanduser()
    else:
        user_set = Path("")
    if user_set_raw and user_set.exists():
        return str(user_set)

    candidates = [
        Path.home() / "AppData" / "Roaming" / "tabpfn" / "tabpfn-v2-regressor.ckpt",
        Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-regressor.ckpt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return "tabpfn-v2-regressor.ckpt"


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that ignores zero targets to avoid division-by-zero."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class TrackingFinetunedTabPFNRegressor(FinetunedTabPFNRegressor):
    """
    Extend TabPFN finetuning regressor to track per-epoch train loss, val MSE, and val R2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_history_: list[dict[str, float]] = []

    def _evaluate_model(self, eval_config, X_train, y_train, X_val, y_val):  # type: ignore[override]
        eval_regressor = clone_model_for_evaluation(
            self.finetuned_estimator_,
            eval_config,
            TabPFNRegressor,
        )
        eval_regressor.fit(X_train, y_train)

        try:
            train_predictions = eval_regressor.predict(X_train)  # type: ignore
            predictions = eval_regressor.predict(X_val)  # type: ignore
            train_mse = mean_squared_error(y_train, train_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            mse = mean_squared_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
        except (ValueError, RuntimeError, AttributeError):
            train_mse = np.nan
            train_r2 = np.nan
            mse = np.nan
            r2 = np.nan

        return EvalResult(
            primary=mse,
            secondary={
                "r2": r2,
                "train_mse": train_mse,
                "train_r2": train_r2,
            },
        )

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):  # type: ignore[override]
        train_loss = np.nan if mean_train_loss is None else float(mean_train_loss)
        val_mse = float(eval_result.primary)
        val_r2 = float(eval_result.secondary.get("r2", np.nan))

        self.epoch_history_.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "train_mse": float(eval_result.secondary.get("train_mse", np.nan)),
                "val_mse": val_mse,
                "train_r2": float(eval_result.secondary.get("train_r2", np.nan)),
                "val_r2": val_r2,
            }
        )

        train_loss_text = "N/A" if np.isnan(train_loss) else f"{train_loss:.4f}"
        val_r2_text = "NaN" if np.isnan(val_r2) else f"{val_r2:.4f}"
        print(
            f"Epoch {epoch + 1}: Val MSE={val_mse:.4f}, Val R2={val_r2_text}, Train Loss={train_loss_text}"
        )

    def _get_checkpoint_metrics(self, eval_result):  # type: ignore[override]
        return {
            "mse": float(eval_result.primary),
            "r2": float(eval_result.secondary.get("r2", np.nan)),
            "train_mse": float(eval_result.secondary.get("train_mse", np.nan)),
            "train_r2": float(eval_result.secondary.get("train_r2", np.nan)),
        }


class TabPFNFinetuneTrainer:
    def __init__(
        self,
        alloy_type: str,
        target_col: str,
        base_path: str = ".",
        finetune_params: Optional[Dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ):
        self.alloy_type = alloy_type
        self.target_col = target_col
        self.base_path = Path(base_path)
        self.config = get_tabpfn_config(alloy_type)
        self.data_processor = TabPFNDataProcessor(self.config)
        self.finetune_params = {**DEFAULT_FINETUNE_PARAMS, **(finetune_params or {})}
        extra_kwargs = dict(self.finetune_params.get("extra_regressor_kwargs", {}) or {})
        extra_kwargs.setdefault("model_path", resolve_regressor_model_path())
        self.finetune_params["extra_regressor_kwargs"] = extra_kwargs
        if self.finetune_params.get("device") == "cuda" and torch is not None and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU for fine-tuning.")
            self.finetune_params["device"] = "cpu"
        self.checkpoint_root = (
            Path(checkpoint_root)
            if checkpoint_root
            else self.base_path / "output" / "TabPFN_finetune_results" / alloy_type / "checkpoints"
        )
        self.model = None
        self.results: Dict[str, Any] = {}
        self.training_history = pd.DataFrame()

    def prepare_data(self, scale: bool = True, drop_na: bool = True):
        self.X_train, self.X_test, self.y_train, self.y_test, self.ids_train, self.ids_test = (
            self.data_processor.get_full_pipeline(
                target_col=self.target_col,
                base_path=str(self.base_path),
                scale=scale,
                drop_na=drop_na,
            )
        )

    def finetune_regression(self):
        if not TABPFN_FINETUNE_AVAILABLE:
            raise ImportError("tabpfn.finetuning is not available. Please install/update tabpfn.")

        print(f"\n{'=' * 60}")
        print("Fine-tuning TabPFN Regression Model")
        print(f"{'=' * 60}")
        print(f"Finetune params: {self.finetune_params}")

        self.model = TrackingFinetunedTabPFNRegressor(**self.finetune_params)

        target_name = (
            self.target_col.replace("(", "")
            .replace(")", "")
            .replace("%", "percent")
            .replace("/", "_")
        )
        output_dir = self.checkpoint_root / target_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Fine-tune checkpoint/output dir: {output_dir}")
        self.model.fit(self.X_train, self.y_train, output_dir=output_dir)
        self.training_history = pd.DataFrame(getattr(self.model, "epoch_history_", []))

        print("Making predictions...")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        print("Predictions completed")

        self.evaluate_regression(y_pred_train, y_pred_test)
        return self.results

    def save_training_history(self, save_dir: str, target_name: str):
        if self.training_history.empty:
            print("No epoch history found. Skip saving training curves.")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / f"{target_name}_epoch_metrics.csv"
        loss_plot_path = save_dir / f"{target_name}_loss_curve.png"
        r2_plot_path = save_dir / f"{target_name}_r2_curve.png"

        history = self.training_history.copy()
        history["epoch"] = history["epoch"].astype(int)
        history = history.sort_values("epoch").reset_index(drop=True)
        history["val_loss"] = history["val_mse"]
        history.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        ax.plot(history["epoch"], history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(f"{self.alloy_type} - {self.target_col} Loss Curve")
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["epoch"], history["train_r2"], label="Train R2", color="#2ca02c", linewidth=2)
        ax.plot(history["epoch"], history["val_r2"], label="Val R2", color="#d62728", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R2")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(f"{self.alloy_type} - {self.target_col} R2 Curve")
        plt.tight_layout()
        plt.savefig(r2_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Epoch metrics saved to: {csv_path}")
        print(f"Loss curve saved to: {loss_plot_path}")
        print(f"R2 curve saved to: {r2_plot_path}")

    def evaluate_regression(self, y_pred_train: np.ndarray, y_pred_test: np.ndarray):
        print(f"\n{'=' * 60}")
        print("Evaluation Results")
        print(f"{'=' * 60}")

        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        train_mape = safe_mape(self.y_train, y_pred_train)

        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mape = safe_mape(self.y_test, y_pred_test)

        self.results = {
            "train": {"mae": train_mae, "rmse": train_rmse, "r2": train_r2, "mape": train_mape},
            "test": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2, "mape": test_mape},
            "predictions": {"train": y_pred_train, "test": y_pred_test},
        }

        print("\nTraining Set:")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R2:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%" if np.isfinite(train_mape) else "  MAPE: NaN (target contains only zeros)")

        print("\nTest Set:")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R2:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%" if np.isfinite(test_mape) else "  MAPE: NaN (target contains only zeros)")

    def plot_predictions(self, save_path: str = None):
        if not self.results:
            print("No results to plot. Fine-tune the model first.")
            return

        y_pred_test = self.results["predictions"]["test"]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(self.y_test, y_pred_test, alpha=0.6, s=50, edgecolors="k", linewidths=0.5)

        min_val = min(self.y_test.min(), y_pred_test.min())
        max_val = max(self.y_test.max(), y_pred_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")

        ax.set_xlabel(f"Actual {self.target_col}", fontsize=12)
        ax.set_ylabel(f"Predicted {self.target_col}", fontsize=12)
        ax.set_title(
            f"{self.alloy_type} - {self.target_col}\nR2 = {self.results['test']['r2']:.4f}, MAE = {self.results['test']['mae']:.2f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        plt.close()

    def save_predictions_csv(self, save_path: str = None, align_reference_csv: str = None):
        if not self.results:
            print("No results to save. Fine-tune the model first.")
            return

        y_pred_train = self.results["predictions"]["train"]
        y_pred_test = self.results["predictions"]["test"]

        train_data = pd.DataFrame(
            {
                "ID": self.ids_train,
                f"{self.target_col}_Actual": self.y_train,
                f"{self.target_col}_Predicted": y_pred_train,
                "Dataset": "Train",
            }
        )
        test_data = pd.DataFrame(
            {
                "ID": self.ids_test,
                f"{self.target_col}_Actual": self.y_test,
                f"{self.target_col}_Predicted": y_pred_test,
                "Dataset": "Test",
            }
        )

        all_data = pd.concat([train_data, test_data], ignore_index=True)
        if align_reference_csv:
            ref_path = Path(align_reference_csv)
            if ref_path.exists():
                try:
                    ref_df = pd.read_csv(ref_path)
                    all_data = align_df_to_reference_id_order(all_data, ref_df, id_col="ID")
                    print(f"Aligned predictions row order to reference CSV: {ref_path}")
                except Exception as e:
                    print(f"Warning: failed to align to reference CSV ({ref_path}): {e}")
                    all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)
            else:
                print(f"Warning: reference CSV not found for alignment: {ref_path}")
                all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)
        else:
            all_data = all_data.sort_values("ID", kind="stable").reset_index(drop=True)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            all_data.to_csv(save_path, index=False)
            print(f"Predictions saved to: {save_path}")

        return all_data


def run_single_finetune_experiment(
    alloy_type: str,
    target_col: str,
    base_path: str = ".",
    finetune_params: Optional[Dict[str, Any]] = None,
):
    print(f"\n{'#' * 60}")
    print(f"# Fine-tune Experiment: {alloy_type} - {target_col}")
    print(f"{'#' * 60}")

    trainer = TabPFNFinetuneTrainer(
        alloy_type=alloy_type,
        target_col=target_col,
        base_path=base_path,
        finetune_params=finetune_params,
    )
    trainer.prepare_data(scale=True, drop_na=True)
    results = trainer.finetune_regression()

    output_dir = Path(base_path) / "output" / "TabPFN_finetune_results" / alloy_type
    target_name = target_col.replace("(", "").replace(")", "").replace("%", "percent").replace("/", "_")

    plot_path = output_dir / f"{target_name}_predictions.png"
    trainer.save_training_history(save_dir=str(output_dir), target_name=target_name)
    trainer.plot_predictions(save_path=str(plot_path))

    csv_path = output_dir / f"{target_name}_all_predictions.csv"
    align_ref = trainer.config.get("align_reference_predictions_csv")
    if align_ref:
        align_ref = str(Path(base_path) / align_ref)
    trainer.save_predictions_csv(save_path=str(csv_path), align_reference_csv=align_ref)

    return results


def run_all_finetune_experiments(base_path: str = ".", finetune_params: Optional[Dict[str, Any]] = None):
    if not TABPFN_FINETUNE_AVAILABLE:
        print("TabPFN finetuning not available. Please install/update tabpfn first.")
        return

    all_results: Dict[str, Any] = {}
    for alloy_type in get_all_alloy_types():
        config = get_tabpfn_config(alloy_type)
        targets = config["targets"]

        alloy_results: Dict[str, Any] = {}
        for target in targets:
            try:
                results = run_single_finetune_experiment(alloy_type, target, base_path, finetune_params)
                alloy_results[target] = results
            except Exception as e:
                print(f"\nError in {alloy_type} - {target}: {e}")
                import traceback

                traceback.print_exc()
                continue
        all_results[alloy_type] = alloy_results

    save_summary_results(all_results, base_path)
    return all_results


def save_summary_results(all_results: Dict[str, Any], base_path: str = "."):
    summary_data = []
    for alloy_type, alloy_results in all_results.items():
        for target, results in alloy_results.items():
            if not results:
                continue
            summary_data.append(
                {
                    "Alloy": alloy_type,
                    "Target": target,
                    "Train_MAE": results["train"]["mae"],
                    "Train_RMSE": results["train"]["rmse"],
                    "Train_R2": results["train"]["r2"],
                    "Train_MAPE": results["train"]["mape"],
                    "Test_MAE": results["test"]["mae"],
                    "Test_RMSE": results["test"]["rmse"],
                    "Test_R2": results["test"]["r2"],
                    "Test_MAPE": results["test"]["mape"],
                }
            )

    df_summary = pd.DataFrame(summary_data)
    output_dir = Path(base_path) / "output" / "TabPFN_finetune_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_results.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("Fine-tune Summary Results")
    print(f"{'=' * 60}")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent

    print("TabPFN Fine-tuning for Alloy Datasets")
    print("=" * 60)
    print("\nAvailable options:")
    print("1. Run all fine-tune experiments")
    print("2. Run single fine-tune experiment")

    choice = input("\nEnter your choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        run_all_finetune_experiments(base_path)
    elif choice == "2":
        print("\nAvailable alloy types:", get_all_alloy_types())
        alloy_type = input("Enter alloy type (e.g., Ti): ").strip()
        config = get_tabpfn_config(alloy_type)
        print(f"Available targets: {config['targets']}")
        target_col = input("Enter target column: ").strip()
        run_single_finetune_experiment(alloy_type, target_col, base_path)
    else:
        print("Invalid choice!")

    print("\n" + "=" * 60)
    print("Done!")
