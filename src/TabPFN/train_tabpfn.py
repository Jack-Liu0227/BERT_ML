"""
TabPFN Model Training Script
TabPFN 模型训练脚本

Train and evaluate TabPFN models on alloy datasets
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Local imports (support both package import and direct script execution)
try:
    from .tabpfn_configs import TABPFN_CONFIGS, TABPFN_MODEL_CONFIG, get_tabpfn_config, get_all_alloy_types
    from .data_processor import TabPFNDataProcessor, create_data_processor
    from .model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from .prediction_alignment import align_df_to_reference_id_order
except ImportError:  # pragma: no cover
    from tabpfn_configs import TABPFN_CONFIGS, TABPFN_MODEL_CONFIG, get_tabpfn_config, get_all_alloy_types
    from data_processor import TabPFNDataProcessor, create_data_processor
    from model_factory import create_tabpfn_regressor, get_tabpfn_runtime_config
    from prediction_alignment import align_df_to_reference_id_order


TABPFN_AVAILABLE = True


def get_training_output_root(base_path: str | Path, model_info: Dict[str, str]) -> Path:
    return Path(base_path) / "output" / f"TabPFN_results_{model_info['model_run_dirname']}"


class TabPFNTrainer:
    """TabPFN 模型训练器 / TabPFN Model Trainer"""
    
    def __init__(
        self,
        alloy_type: str,
        target_col: str,
        base_path: str = ".",
        backend: str = "auto",
        model_version: str | None = None,
        feature_mode: str | None = None,
    ):
        """
        初始化训练器
        
        Args:
            alloy_type: 合金类型 ("Ti", "Al", "HEA", "Steel")
            target_col: 目标列名
            base_path: 项目根目录
        """
        self.alloy_type = alloy_type
        self.target_col = target_col
        self.base_path = Path(base_path)
        self.backend = backend
        self.model_version = model_version
        self.feature_mode = feature_mode
        self.runtime_info = get_tabpfn_runtime_config(
            base_path=self.base_path,
            backend=backend,
            preferred_model_version=model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        
        # 获取配置
        self.config = get_tabpfn_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
            base_path=str(self.base_path),
        )
        
        # 初始化数据处理器
        self.data_processor = TabPFNDataProcessor(self.config)
        
        # 初始化模型
        self.model = None
        self.model_info = dict(self.runtime_info)
        self.results = {}
        
    def prepare_data(self, scale: bool = True, drop_na: bool = True):
        """准备数据 / Prepare data"""
        self.X_train, self.X_test, self.y_train, self.y_test, self.ids_train, self.ids_test = \
            self.data_processor.get_full_pipeline(
                target_col=self.target_col,
                base_path=str(self.base_path),
                scale=scale,
                drop_na=drop_na
            )
    
    def train_regression(self):
        """
        训练回归模型
        Train regression model using TabPFNRegressor
        """
        print(f"\n{'='*60}")
        print("Training TabPFN Regression Model")
        print(f"{'='*60}")

        self.model, self.model_info = create_tabpfn_regressor(
            base_path=self.base_path,
            preferred_model_version=self.model_version or TABPFN_MODEL_CONFIG.get("model_version"),
            backend=self.backend,
            feature_mode=self.feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
        )
        print(
            "Initialized model: "
            f"{self.model_info.get('model_name', 'unknown')} "
            f"({self.model_info.get('feature_mode', 'unknown')})"
        )

        # 训练模型
        print("Fitting model...")
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
        print("Model fitted successfully")
        
        # 预测
        print("Making predictions...")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        print("Predictions completed")
        
        # 评估
        self.evaluate_regression(y_pred_train, y_pred_test)
        
        return self.results
    
    def evaluate_regression(self, y_pred_train: np.ndarray, y_pred_test: np.ndarray):
        """
        评估回归模型
        Evaluate regression model
        """
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        
        # 训练集评估
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        train_mape = np.mean(np.abs((self.y_train - y_pred_train) / self.y_train)) * 100
        
        # 测试集评估
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mape = np.mean(np.abs((self.y_test - y_pred_test) / self.y_test)) * 100
        
        # 保存结果
        self.results = {
            'model_info': self.model_info,
            'train': {
                'mae': train_mae,
                'rmse': train_rmse,
                'r2': train_r2,
                'mape': train_mape
            },
            'test': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'mape': test_mape
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        # 打印结果
        print("\nTraining Set:")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R2:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")
        
        print("\nTest Set:")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R2:   {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.2f}%")
        
    def plot_predictions(self, save_path: str = None):
        """
        绘制预测结果（仅对角线图）
        Plot predictions (diagonal plot only)
        """
        if not self.results:
            print("No results to plot. Train the model first.")
            return
        
        y_pred_test = self.results['predictions']['test']
        
        # 创建单个图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制散点图
        ax.scatter(self.y_test, y_pred_test, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
        
        # 绘制对角线
        min_val = min(self.y_test.min(), y_pred_test.min())
        max_val = max(self.y_test.max(), y_pred_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect prediction')
        
        # 设置标签和标题
        ax.set_xlabel(f'Actual {self.target_col}', fontsize=12)
        ax.set_ylabel(f'Predicted {self.target_col}', fontsize=12)
        ax.set_title(f'{self.alloy_type} - {self.target_col}\nR2 = {self.results["test"]["r2"]:.4f}, MAE = {self.results["test"]["mae"]:.2f}', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置相同的坐标轴范围
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.close()
    
    def save_predictions_csv(self, save_path: str = None, align_reference_csv: str = None):
        """
        保存预测结果为CSV文件
        Save predictions to CSV file
        """
        if not self.results:
            print("No results to save. Train the model first.")
            return
        
        y_pred_train = self.results['predictions']['train']
        y_pred_test = self.results['predictions']['test']
        
        # 创建训练集数据（使用原始 ID）
        train_data = pd.DataFrame({
            'ID': self.ids_train,
            f'{self.target_col}_Actual': self.y_train,
            f'{self.target_col}_Predicted': y_pred_train,
            'Dataset': 'Train'
        })
        
        # 创建测试集数据（使用原始 ID）
        test_data = pd.DataFrame({
            'ID': self.ids_test,
            f'{self.target_col}_Actual': self.y_test,
            f'{self.target_col}_Predicted': y_pred_test,
            'Dataset': 'Test'
        })
        
        # 合并数据
        all_data = pd.concat([train_data, test_data], ignore_index=True)
        # Align row order for easier cross-model comparison:
        # - If a reference CSV is provided and exists: follow its ID order.
        # - Otherwise: fallback to deterministic sorting by ID.
        if align_reference_csv:
            ref_path = Path(align_reference_csv)
            if ref_path.exists():
                try:
                    ref_df = pd.read_csv(ref_path)
                    all_data = align_df_to_reference_id_order(all_data, ref_df, id_col="ID")
                    print(f"Aligned predictions row order to reference CSV: {ref_path}")
                except Exception as e:
                    print(f"Warning: failed to align to reference CSV ({ref_path}): {e}")
                    all_data = all_data.sort_values('ID', kind='stable').reset_index(drop=True)
            else:
                print(f"Warning: reference CSV not found for alignment: {ref_path}")
                all_data = all_data.sort_values('ID', kind='stable').reset_index(drop=True)
        else:
            all_data = all_data.sort_values('ID', kind='stable').reset_index(drop=True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            all_data.to_csv(save_path, index=False)
            print(f"Predictions saved to: {save_path}")
        
        return all_data

    def save_manifest(self, save_path: str) -> None:
        available_feature_cols = list(getattr(self.data_processor, "feature_names", []))
        numeric_feature_cols = list(getattr(self.data_processor, "numeric_feature_names", []))
        non_numeric_feature_cols = list(getattr(self.data_processor, "non_numeric_feature_names", []))
        manifest = {
            "alloy_type": self.alloy_type,
            "target_col": self.target_col,
            "raw_data": self.config["raw_data"],
            "model_name": self.model_info.get("model_name"),
            "model_dirname": self.model_info.get("model_dirname"),
            "feature_mode": self.model_info.get("feature_mode"),
            "feature_mode_dirname": self.model_info.get("feature_mode_dirname"),
            "requested_backend": self.backend,
            "resolved_backend": self.model_info.get("resolved_backend", self.backend),
            "model_info": self.model_info,
            "configured_feature_cols": self.config.get("feature_cols", []),
            "element_feature_cols": self.config.get("element_feature_cols", []),
            "processing_feature_cols": self.config.get("processing_feature_cols", []),
            "numeric_process_feature_cols": self.config.get("numeric_process_feature_cols", []),
            "text_feature_cols": self.config.get("text_feature_cols", []),
            "available_feature_cols": available_feature_cols,
            "numeric_feature_cols": numeric_feature_cols,
            "non_numeric_feature_cols": non_numeric_feature_cols,
        }
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_path_obj.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Manifest saved to: {save_path_obj}")


def run_single_experiment(
    alloy_type: str,
    target_col: str,
    base_path: str = ".",
    backend: str = "auto",
    model_version: str | None = None,
    feature_mode: str | None = None,
):
    """
    运行单个实验
    Run single experiment
    
    Args:
        alloy_type: 合金类型
        target_col: 目标列
        base_path: 项目根目录
    """
    print(f"\n{'#'*60}")
    print(f"# Experiment: {alloy_type} - {target_col}")
    print(f"{'#'*60}")
    
    # 初始化训练器
    trainer = TabPFNTrainer(
        alloy_type,
        target_col,
        base_path,
        backend=backend,
        model_version=model_version,
        feature_mode=feature_mode,
    )
    
    # 准备数据
    trainer.prepare_data(scale=True, drop_na=True)
    
    # 训练模型
    results = trainer.train_regression()
    
    # 创建输出目录
    output_dir = get_training_output_root(base_path, trainer.model_info) / alloy_type
    target_name = target_col.replace('(', '').replace(')', '').replace('%', 'percent').replace('/', '_')
    
    # 绘制预测图
    plot_path = output_dir / f"{target_name}_predictions.png"
    trainer.plot_predictions(save_path=str(plot_path))
    
    # 保存预测数据
    csv_path = output_dir / f"{target_name}_all_predictions.csv"
    align_ref = trainer.config.get("align_reference_predictions_csv")
    if align_ref:
        align_ref = str(Path(base_path) / align_ref)
    trainer.save_predictions_csv(save_path=str(csv_path), align_reference_csv=align_ref)

    manifest_path = output_dir / f"{target_name}_manifest.json"
    trainer.save_manifest(save_path=str(manifest_path))
    
    return results


def run_all_experiments(
    base_path: str = ".",
    backend: str = "auto",
    model_version: str | None = None,
    feature_mode: str | None = None,
):
    """
    运行所有实验
    Run all experiments
    """
    if not TABPFN_AVAILABLE:
        print("TabPFN not available. Please install it first.")
        return
    
    runtime_info = get_tabpfn_runtime_config(
        base_path=base_path,
        backend=backend,
        preferred_model_version=model_version or TABPFN_MODEL_CONFIG.get("model_version"),
        feature_mode=feature_mode or TABPFN_MODEL_CONFIG.get("feature_mode"),
    )
    all_results = {}
    
    for alloy_type in get_all_alloy_types(
        backend=backend,
        base_path=str(base_path),
    ):
        config = get_tabpfn_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=str(base_path),
        )
        targets = config["targets"]
        
        alloy_results = {}
        for target in targets:
            try:
                results = run_single_experiment(
                    alloy_type,
                    target,
                    base_path,
                    backend=backend,
                    model_version=model_version,
                    feature_mode=feature_mode,
                )
                alloy_results[target] = results
            except Exception as e:
                print(f"\nError in {alloy_type} - {target}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[alloy_type] = alloy_results
    
    # 保存汇总结果
    save_summary_results(all_results, base_path, runtime_info=runtime_info)
    
    return all_results


def save_summary_results(
    all_results: Dict,
    base_path: str = ".",
    runtime_info: Dict | None = None,
):
    """
    保存汇总结果
    Save summary results
    """
    summary_data = []
    
    for alloy_type, alloy_results in all_results.items():
        for target, results in alloy_results.items():
            if results:
                summary_data.append({
                    'Alloy': alloy_type,
                    'Target': target,
                    'Model_Name': results['model_info']['model_name'],
                    'Model_Dirname': results['model_info']['model_dirname'],
                    'Feature_Mode': results['model_info']['feature_mode'],
                    'Feature_Mode_Dirname': results['model_info']['feature_mode_dirname'],
                    'Backend': results['model_info']['backend'],
                    'Requested_Backend': results['model_info']['requested_backend'],
                    'Resolved_Backend': results['model_info']['resolved_backend'],
                    'Effective_Model_Version': results['model_info']['effective_model_version'],
                    'Train_MAE': results['train']['mae'],
                    'Train_RMSE': results['train']['rmse'],
                    'Train_R2': results['train']['r2'],
                    'Train_MAPE': results['train']['mape'],
                    'Test_MAE': results['test']['mae'],
                    'Test_RMSE': results['test']['rmse'],
                    'Test_R2': results['test']['r2'],
                    'Test_MAPE': results['test']['mape'],
                })
    
    df_summary = pd.DataFrame(summary_data)
    
    # 保存到CSV
    resolved_runtime_info = runtime_info
    if resolved_runtime_info is None:
        for alloy_results in all_results.values():
            for results in alloy_results.values():
                if results:
                    resolved_runtime_info = results.get('model_info')
                    break
            if resolved_runtime_info is not None:
                break
    if resolved_runtime_info is None:
        raise ValueError("Could not determine TabPFN runtime metadata for saving summary results.")

    output_dir = get_training_output_root(base_path, resolved_runtime_info)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "summary_results.csv"
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("Summary Results")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to: {summary_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate TabPFN models on alloy datasets.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "api", "local"],
        default="auto",
        help=(
            "Backend selection. `auto` prefers API when a key is available, "
            "otherwise falls back to the local model."
        ),
    )
    parser.add_argument(
        "--model_version",
        choices=["latest", "v2", "v2.5", "v2.6"],
        default=None,
        help=(
            "Compatibility flag. Local mode only supports `v2`, while API mode "
            "ignores this value and always uses TabPFN-2.5-Plus."
        ),
    )
    parser.add_argument(
        "--feature_mode",
        choices=["numeric", "text"],
        default=None,
        help=(
            "Feature mode. Local defaults to `numeric` and does not support `text`; "
            "API defaults to `text` but also supports `numeric`."
        ),
    )
    parser.add_argument(
        "--alloy_type",
        type=str,
        default=None,
        help="Run a single alloy type non-interactively.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Run a single target column non-interactively.",
    )
    parser.add_argument(
        "--all",
        "--run_all",
        dest="run_all",
        action="store_true",
        help=(
            "Run all configured alloy/target experiments non-interactively. "
            "`--run_all` is kept as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root path.",
    )
    return parser


def run_interactive(
    base_path: str,
    backend: str,
    model_version: str | None,
    feature_mode: str | None,
) -> None:
    print("TabPFN Model Training for Alloy Datasets")
    print("=" * 60)
    print(f"Backend mode: {backend}")
    if model_version:
        print(f"Model version override: {model_version}")
    if feature_mode:
        print(f"Feature mode override: {feature_mode}")

    print("\nAvailable options:")
    print("1. Run all experiments")
    print("2. Run single experiment")

    choice = input("\nEnter your choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        run_all_experiments(
            base_path,
            backend=backend,
            model_version=model_version,
            feature_mode=feature_mode,
        )
    elif choice == "2":
        print(
            "\nAvailable alloy types:",
            get_all_alloy_types(backend=backend, base_path=str(base_path)),
        )
        alloy_type = input("Enter alloy type (e.g., Ti): ").strip()

        config = get_tabpfn_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=str(base_path),
        )
        print(f"Available targets: {config['targets']}")
        target_col = input("Enter target column: ").strip()

        run_single_experiment(
            alloy_type,
            target_col,
            base_path,
            backend=backend,
            model_version=model_version,
            feature_mode=feature_mode,
        )
    else:
        print("Invalid choice!")

    print("\n" + "=" * 60)
    print("Done!")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.run_all and (args.alloy_type or args.target_col):
        parser.error("`--all`/`--run_all` cannot be combined with `--alloy_type` or `--target_col`.")

    if (args.alloy_type is None) ^ (args.target_col is None):
        parser.error("`--alloy_type` and `--target_col` must be provided together.")

    if args.run_all:
        run_all_experiments(
            args.base_path,
            backend=args.backend,
            model_version=args.model_version,
            feature_mode=args.feature_mode,
        )
        return

    if args.alloy_type and args.target_col:
        run_single_experiment(
            alloy_type=args.alloy_type,
            target_col=args.target_col,
            base_path=args.base_path,
            backend=args.backend,
            model_version=args.model_version,
            feature_mode=args.feature_mode,
        )
        return

    run_interactive(
        args.base_path,
        backend=args.backend,
        model_version=args.model_version,
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
