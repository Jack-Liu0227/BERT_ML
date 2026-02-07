"""
TabPFN Model Training Script
TabPFN 模型训练脚本

Train and evaluate TabPFN models on alloy datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TabPFN imports
try:
    from tabpfn import TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion
    except ImportError:
        ModelVersion = None
    TABPFN_AVAILABLE = True
except ImportError:
    print("Warning: TabPFN not installed. Please install with: pip install tabpfn")
    TABPFN_AVAILABLE = False
    ModelVersion = None

# Local imports (support both package import and direct script execution)
try:
    from .tabpfn_configs import TABPFN_CONFIGS, TABPFN_MODEL_CONFIG, get_tabpfn_config, get_all_alloy_types
    from .data_processor import TabPFNDataProcessor, create_data_processor
    from .prediction_alignment import align_df_to_reference_id_order
except ImportError:  # pragma: no cover
    from tabpfn_configs import TABPFN_CONFIGS, TABPFN_MODEL_CONFIG, get_tabpfn_config, get_all_alloy_types
    from data_processor import TabPFNDataProcessor, create_data_processor
    from prediction_alignment import align_df_to_reference_id_order


class TabPFNTrainer:
    """TabPFN 模型训练器 / TabPFN Model Trainer"""
    
    def __init__(self, alloy_type: str, target_col: str, base_path: str = "."):
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
        
        # 获取配置
        self.config = get_tabpfn_config(alloy_type)
        
        # 初始化数据处理器
        self.data_processor = TabPFNDataProcessor(self.config)
        
        # 初始化模型
        self.model = None
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
        print(f"Training TabPFN Regression Model (v2)")
        print(f"{'='*60}")
        
        # 初始化回归器 - 使用 v2 版本（开放模型，无需认证）
        # v2.5 需要在 HuggingFace 接受条款并认证
        try:
            from tabpfn.constants import ModelVersion
            self.model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
            print("TabPFNRegressor (v2) initialized - using open model")
        except Exception as e:
            print(f"Warning: Could not use v2 explicitly, trying default: {e}")
            self.model = TabPFNRegressor(model_version="v2")
            print("TabPFNRegressor initialized with v2")
        
        # 训练模型
        print("Fitting model...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model fitted successfully")
        
        # 预测
        print("Making predictions...")
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        print("✓ Predictions completed")
        
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
        print(f"  R²:   {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.2f}%")
        
        print("\nTest Set:")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²:   {test_r2:.4f}")
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
        ax.set_title(f'{self.alloy_type} - {self.target_col}\nR² = {self.results["test"]["r2"]:.4f}, MAE = {self.results["test"]["mae"]:.2f}', 
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


def run_single_experiment(alloy_type: str, target_col: str, base_path: str = "."):
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
    trainer = TabPFNTrainer(alloy_type, target_col, base_path)
    
    # 准备数据
    trainer.prepare_data(scale=True, drop_na=True)
    
    # 训练模型
    results = trainer.train_regression()
    
    # 创建输出目录
    output_dir = Path(base_path) / "output" / "TabPFN_results" / alloy_type
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
    
    return results


def run_all_experiments(base_path: str = "."):
    """
    运行所有实验
    Run all experiments
    """
    if not TABPFN_AVAILABLE:
        print("TabPFN not available. Please install it first.")
        return
    
    all_results = {}
    
    for alloy_type in get_all_alloy_types():
        config = get_tabpfn_config(alloy_type)
        targets = config["targets"]
        
        alloy_results = {}
        for target in targets:
            try:
                results = run_single_experiment(alloy_type, target, base_path)
                alloy_results[target] = results
            except Exception as e:
                print(f"\nError in {alloy_type} - {target}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[alloy_type] = alloy_results
    
    # 保存汇总结果
    save_summary_results(all_results, base_path)
    
    return all_results


def save_summary_results(all_results: Dict, base_path: str = "."):
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
    output_dir = Path(base_path) / "output" / "TabPFN_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "summary_results.csv"
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("Summary Results")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    # 设置项目根目录（相对于此脚本的位置）
    base_path = Path(__file__).parent.parent.parent
    
    print("TabPFN Model Training for Alloy Datasets")
    print("=" * 60)
    
    # 选择运行模式
    print("\nAvailable options:")
    print("1. Run all experiments")
    print("2. Run single experiment")
    
    choice = input("\nEnter your choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        # 运行所有实验
        all_results = run_all_experiments(base_path)
        
    elif choice == "2":
        # 选择合金类型和目标
        print("\nAvailable alloy types:", get_all_alloy_types())
        alloy_type = input("Enter alloy type (e.g., Ti): ").strip()
        
        config = get_tabpfn_config(alloy_type)
        print(f"Available targets: {config['targets']}")
        target_col = input("Enter target column: ").strip()
        
        # 运行单个实验
        results = run_single_experiment(alloy_type, target_col, base_path)
    
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("Done!")
