"""
模型对比器模块
Model Comparator Module

提供多模型对比的核心功能
Provides core functionality for multi-model comparison
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

# 设置标准输出和标准错误为UTF-8编码
# Set stdout and stderr to UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

try:
    # 尝试相对导入（当作为模块运行时）
    from .pipeline import MLTrainingPipeline
    from .utils import to_long_path
    from .model_comparison_plots import create_model_comparison_plots
except ImportError:
    # 尝试直接导入（当直接运行时）
    try:
        from pipeline import MLTrainingPipeline
        from utils import to_long_path
        from model_comparison_plots import create_model_comparison_plots
    except ImportError:
        # 使用完整路径导入
        from src.models.ml_train_eval_pipeline.pipeline import MLTrainingPipeline
        from src.models.ml_train_eval_pipeline.utils import to_long_path
        from src.models.ml_train_eval_pipeline.model_comparison_plots import create_model_comparison_plots


class ModelComparator:
    """
    多模型对比器
    Multi-Model Comparator
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        初始化模型对比器
        Initialize model comparator
        
        Args:
            base_config: 基础配置字典
        """
        self.base_config = base_config
        self.results = {}
        self.comparison_dir = None
        
    def compare_models(self, 
                      models_to_compare: List[str], 
                      use_optuna: bool = False,
                      n_trials: int = 50,
                      n_repeats: int = 1) -> Dict[str, Any]:
        """
        对比多个模型的性能
        Compare performance of multiple models
        """
        print("=" * 80)
        print("开始多模型对比 / Starting Multi-Model Comparison")
        print("=" * 80)
        
        # 创建对比结果目录
        base_result_dir = self.base_config['result_dir']
        self.comparison_dir = os.path.join(base_result_dir, "model_comparison")
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        for model_type in models_to_compare:
            print(f"\n{'='*60}")
            print(f"训练模型 / Training Model: {model_type.upper()}")
            print(f"{'='*60}")
            
            try:
                # 为每个模型创建基础结果目录
                model_base_dir = os.path.join(self.comparison_dir, f"{model_type}_results")
                
                repeat_results = []
                
                for i in range(n_repeats):
                    if n_repeats > 1:
                        print(f"\n  >> 运行重复实验 {i+1}/{n_repeats} (Run {i+1}/{n_repeats})")
                        model_result_dir = os.path.join(model_base_dir, f"repeat_{i}")
                    else:
                        model_result_dir = model_base_dir
                    
                    # 创建模型特定的配置
                    model_config = self.base_config.copy()
                    model_config['model_type'] = model_type
                    model_config['result_dir'] = model_result_dir
                    model_config['use_optuna'] = use_optuna
                    model_config['n_trials'] = n_trials
                    
                    # 更新随机种子以保证每次运行不同
                    if 'random_state' in model_config:
                        model_config['random_state'] = model_config['random_state'] + i
                    
                    # 训练模型
                    result = self._train_single_model(model_config)
                    repeat_results.append(result)
                
                # 如果有多次重复，聚合结果
                if n_repeats > 1:
                    print(f"\n[聚合] 计算 {n_repeats} 次运行的平均指标...")
                    final_result = self._aggregate_repeat_results(repeat_results)
                    self.results[model_type] = final_result
                else:
                    self.results[model_type] = repeat_results[0]

                print(f"[OK] 模型 {model_type} 训练完成")

            except Exception as e:
                print(f"[FAIL] 模型 {model_type} 训练失败: {str(e)}")
                self.results[model_type] = {'error': str(e)}
        
        # 生成对比报告
        self._generate_comparison_report()
        
        return self.results
    
    def _train_single_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """训练单个模型"""
        # 创建模拟的命令行参数
        class Args:
            def __init__(self, config_dict):
                # 设置所有必需的默认属性
                self.data_file = None
                self.result_dir = None
                self.model_type = 'xgboost'
                self.target_columns = []
                self.processing_cols = []
                self.use_composition_feature = False
                self.use_temperature = False
                self.other_features_name = None
                self.test_size = 0.2
                self.random_state = 42
                self.evaluate_after_train = True
                self.run_shap_analysis = False
                self.cross_validate = True
                self.num_folds = 5
                self.use_optuna = False
                self.n_trials = 50
                self.study_name = 'ml_hyperparameter_optimization'
                self.mlp_max_iter = 200

                # 用配置字典中的值覆盖默认值
                for key, value in config_dict.items():
                    setattr(self, key, value)

        args = Args(config)
        
        # 处理Windows长路径
        args.result_dir = to_long_path(args.result_dir)
        
        # 创建并运行管道
        pipeline = MLTrainingPipeline(args)
        pipeline.run()
        
        # 收集结果
        result = self._collect_model_results(args.result_dir)
        return result
    
    def _aggregate_repeat_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多次重复实验的结果"""
        aggregated = {}
        
        # 收集所有有效的最终评估指标
        all_metrics = {}
        valid_results_count = 0
        
        for res in results:
            if 'error' in res:
                continue
                
            valid_results_count += 1
            
            # 从final_evaluation提取
            if 'final_evaluation' in res:
                metrics = res['final_evaluation']
                for k, v in metrics.items():
                    # 只聚合数值型指标
                    if isinstance(v, (int, float)):
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(v)
            
            # 从evaluation提取(备用)
            elif 'evaluation' in res and 'test_metrics' in res['evaluation']:
                test_metrics = res['evaluation']['test_metrics']
                for target, metrics in test_metrics.items():
                    for m_name, m_val in metrics.items():
                        key = f"test_{target}_{m_name}"
                        if key not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(m_val)

        # 计算均值和标准差
        final_evaluation = {}
        for k, values in all_metrics.items():
            if values:
                final_evaluation[k] = float(np.mean(values))
                final_evaluation[f"{k}_std"] = float(np.std(values))
                final_evaluation[f"{k}_all_values"] = values
        
        aggregated['final_evaluation'] = final_evaluation
        aggregated['repeat_results'] = results
        aggregated['n_repeats'] = len(results)
        aggregated['valid_repeats'] = valid_results_count
        
        return aggregated

    def _collect_model_results(self, result_dir: str) -> Dict[str, Any]:
        """收集模型训练结果"""
        result = {}

        try:
            # 读取最终评估结果
            final_eval_file = os.path.join(result_dir, "final_evaluation_metrics.json")
            if os.path.exists(final_eval_file):
                with open(final_eval_file, 'r', encoding='utf-8') as f:
                    final_metrics = json.load(f)
                    result['final_evaluation'] = final_metrics

            # 读取评估结果（保持兼容性）
            eval_file = os.path.join(result_dir, "evaluation_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf-8') as f:
                    result['evaluation'] = json.load(f)

            # 读取交叉验证结果
            cv_file = os.path.join(result_dir, "cross_validation_results.json")
            if os.path.exists(cv_file):
                with open(cv_file, 'r', encoding='utf-8') as f:
                    result['cross_validation'] = json.load(f)

            # 读取交叉验证平均指标（新格式）
            cv_avg_file = os.path.join(result_dir, "cv_avg_metrics.json")
            if os.path.exists(cv_avg_file):
                with open(cv_avg_file, 'r', encoding='utf-8') as f:
                    cv_avg_metrics = json.load(f)
                    result['cv_avg_metrics'] = cv_avg_metrics

            # 读取Optuna最佳参数
            optuna_file = os.path.join(result_dir, "optuna_best_params.json")
            if os.path.exists(optuna_file):
                with open(optuna_file, 'r', encoding='utf-8') as f:
                    result['best_params'] = json.load(f)

            # 检查模型文件是否存在
            model_file = os.path.join(result_dir, "best_model.pkl")
            result['model_saved'] = os.path.exists(model_file)

        except Exception as e:
            result['collection_error'] = str(e)

        return result
    
    def _generate_comparison_report(self):
        """生成模型对比报告"""
        print(f"\n{'='*80}")
        print("生成对比报告 / Generating Comparison Report")
        print(f"{'='*80}")
        
        # 创建对比表格
        comparison_data = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
                
            row = {'Model': model_name}

            # Flag to track if we found test set results
            found_test_results = False

            # 优先提取最终测试集评估指标 (Priority 1: Final test set evaluation)
            if 'final_evaluation' in result:
                final_metrics = result['final_evaluation']
                for metric_key, metric_value in final_metrics.items():
                    if '_test_' in metric_key:
                        parts = metric_key.split('_')
                        if len(parts) >= 4:
                            target_start_idx = parts.index('test') + 1
                            metric_type = parts[-1].upper()
                            target_name = '_'.join(parts[target_start_idx:-1])
                            row[f'{target_name}_{metric_type}'] = metric_value
                            found_test_results = True

            # 备选：提取评估指标中的测试集结果 (Priority 2: Test metrics from evaluation)
            if not found_test_results and 'evaluation' in result:
                eval_data = result['evaluation']
                if 'test_metrics' in eval_data:
                    for target, metrics in eval_data['test_metrics'].items():
                        row[f'{target}_R2'] = metrics.get('r2', 'N/A')
                        row[f'{target}_RMSE'] = metrics.get('rmse', 'N/A')
                        row[f'{target}_MAE'] = metrics.get('mae', 'N/A')
                        found_test_results = True

            # 最后备选：使用交叉验证平均结果 (Priority 3: CV average as fallback)
            if not found_test_results and 'cv_avg_metrics' in result:
                cv_avg = result['cv_avg_metrics']
                # Extract target-specific metrics from CV averages
                for key, value in cv_avg.items():
                    if '_' in key:  # Target-specific metrics like "UTS(MPa)_r2"
                        parts = key.split('_')
                        if len(parts) >= 2:
                            target_name = '_'.join(parts[:-1])
                            metric_type = parts[-1].upper()
                            row[f'{target_name}_{metric_type}'] = value
                            found_test_results = True

            # 记录交叉验证结果用于参考 (Keep CV results for reference)
            if 'cross_validation' in result:
                cv_data = result['cross_validation']
                for target, cv_result in cv_data.items():
                    if isinstance(cv_result, dict):
                        row[f'{target}_CV_R2_mean'] = cv_result.get('r2_mean', 'N/A')
                        row[f'{target}_CV_R2_std'] = cv_result.get('r2_std', 'N/A')
            
            comparison_data.append(row)
        
        # 保存对比表格
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            csv_path = os.path.join(self.comparison_dir, "model_comparison_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"[结果] 对比结果已保存到: {csv_path}")
            
            # 生成可视化图表
            plot_results = create_model_comparison_plots(df, self.comparison_dir)
            if plot_results.get('comprehensive_plot'):
                print(f"[图表] 综合对比图表已生成")
                print(f"   - 包含 MAE、R²、RMSE 对比和综合排名")
                print(f"   - 支持 {plot_results.get('targets_count', 0)} 个目标变量")
                print(f"   - 对比 {plot_results.get('models_count', 0)} 个模型")

            if plot_results.get('individual_plots'):
                print(f"[图表] 单指标对比图表: {len(plot_results['individual_plots'])} 个")
        
        # 保存详细重复实验结果 (Detailed Repeats CSV)
        detailed_data = []
        for model_name, result in self.results.items():
            if 'error' in result or 'repeat_results' not in result:
                continue
            
            for i, rep_res in enumerate(result['repeat_results']):
                d_row = {'Model': model_name, 'Repeat': i + 1}
                
                # 提取各项指 (Extract metrics)
                if 'final_evaluation' in rep_res:
                    for k, v in rep_res['final_evaluation'].items():
                        if isinstance(v, (int, float, str)):
                            d_row[k] = v
                            
                elif 'evaluation' in rep_res and 'test_metrics' in rep_res['evaluation']:
                     # Fallback
                     for target, metrics in rep_res['evaluation']['test_metrics'].items():
                         for m_k, m_v in metrics.items():
                             d_row[f"{target}_{m_k}"] = m_v
                             
                detailed_data.append(d_row)
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            # Sort columns
            cols = ['Model', 'Repeat'] + [c for c in detailed_df.columns if c not in ['Model', 'Repeat']]
            detailed_df = detailed_df[cols]
            
            detailed_path = os.path.join(self.comparison_dir, "model_comparison_detailed_repeats.csv")
            detailed_df.to_csv(detailed_path, index=False)
            print(f"[结果] 详细重复实验结果已保存到: {detailed_path}")

        # 生成文本报告
        self._create_text_report()
    
    def _create_text_report(self):
        """创建文本格式的对比报告"""
        report_path = os.path.join(self.comparison_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("机器学习模型对比报告\n")
            f.write("Machine Learning Model Comparison Report\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"模型 / Model: {model_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                if 'error' in result:
                    f.write(f"❌ 训练失败 / Training Failed: {result['error']}\n\n")
                    continue
                
                # 写入最终评估结果 (优先使用final_evaluation)
                if 'final_evaluation' in result:
                    f.write("测试集评估结果 / Test Set Evaluation:\n")
                    final_metrics = result['final_evaluation']

                    # 提取目标变量名
                    targets = set()
                    for key in final_metrics.keys():
                        if '_test_' in key:
                            parts = key.split('_')
                            if len(parts) >= 4:
                                target_start_idx = parts.index('test') + 1
                                target_name = '_'.join(parts[target_start_idx:-1])
                                targets.add(target_name)

                    for target in sorted(targets):
                        f.write(f"  {target}:\n")
                        r2_key = f"final_model_evaluation_test_{target}_r2"
                        rmse_key = f"final_model_evaluation_test_{target}_rmse"
                        mae_key = f"final_model_evaluation_test_{target}_mae"

                        r2_val = final_metrics.get(r2_key, 'N/A')
                        rmse_val = final_metrics.get(rmse_key, 'N/A')
                        mae_val = final_metrics.get(mae_key, 'N/A')

                        if r2_val != 'N/A':
                            f.write(f"    R² Score: {r2_val:.4f}\n")
                        else:
                            f.write(f"    R² Score: N/A\n")

                        if rmse_val != 'N/A':
                            f.write(f"    RMSE: {rmse_val:.4f}\n")
                        else:
                            f.write(f"    RMSE: N/A\n")

                        if mae_val != 'N/A':
                            f.write(f"    MAE: {mae_val:.4f}\n")
                        else:
                            f.write(f"    MAE: N/A\n")

                # 备用：写入旧格式评估结果
                elif 'evaluation' in result:
                    f.write("测试集评估结果 / Test Set Evaluation:\n")
                    eval_data = result['evaluation']
                    if 'test_metrics' in eval_data:
                        for target, metrics in eval_data['test_metrics'].items():
                            f.write(f"  {target}:\n")
                            f.write(f"    R² Score: {metrics.get('r2', 'N/A'):.4f}\n")
                            f.write(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}\n")
                            f.write(f"    MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                
                # 写入交叉验证结果
                if 'cross_validation' in result:
                    f.write("\n交叉验证结果 / Cross-Validation Results:\n")
                    cv_data = result['cross_validation']
                    for target, cv_result in cv_data.items():
                        if isinstance(cv_result, dict):
                            f.write(f"  {target}:\n")
                            f.write(f"    R² Mean: {cv_result.get('r2_mean', 'N/A'):.4f}\n")
                            f.write(f"    R² Std: {cv_result.get('r2_std', 'N/A'):.4f}\n")

                # 写入最佳参数信息
                if 'best_params' in result:
                    f.write("\n最佳超参数 / Best Hyperparameters:\n")
                    best_params = result['best_params']
                    for param, value in best_params.items():
                        f.write(f"  {param}: {value}\n")

                # 生成符合要求的性能摘要 (Performance Summary)
                # 优先使用交叉验证结果计算不确定度 (Use CV results for uncertainty)
                target_stats = {}
                
                if 'cross_validation' in result:
                    cv_data = result['cross_validation']
                    for target, metrics in cv_data.items():
                        if isinstance(metrics, dict):
                            if target not in target_stats:
                                target_stats[target] = {}
                                
                            # Map CV metric names to standard names
                            # CV often has: r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std
                            for m in ['r2', 'rmse', 'mae', 'mape']:
                                mean_key = f"{m}_mean"
                                std_key = f"{m}_std"
                                
                                if mean_key in metrics:
                                    target_stats[target][m] = {
                                        'mean': metrics[mean_key],
                                        'std': metrics.get(std_key, 0.0)
                                    }

                # 如果没有CV结果，尝试使用Test结果 (Fallback to Test results if no CV)
                if not target_stats and 'final_evaluation' in result:
                    final_metrics = result['final_evaluation']
                    # Parses keys like "final_model_evaluation_test_{Target}_{metric}"
                    for k, v in final_metrics.items():
                        if "_test_" in k:
                            parts = k.split('_')
                            try:
                                test_idx = parts.index('test')
                            except ValueError:
                                continue
                            
                            suffix = parts[-1]
                            # Only handle single values here, as std usually comes from CV or repeats
                            # If individual test result, std is 0
                            if suffix != 'std': 
                                metric = suffix
                                target = "_".join(parts[test_idx+1:-1])
                                
                                if target not in target_stats:
                                    target_stats[target] = {}
                                if metric not in target_stats[target]:
                                    target_stats[target][metric] = {'mean': v, 'std': 0.0}

                if target_stats:
                    f.write("\n性能摘要 / Performance Summary:\n")
                    summary_lines = []
                    for metric in ['r2', 'mae', 'rmse', 'mape']:
                         targets_with_m = [t for t in target_stats if metric in target_stats[t] and 'mean' in target_stats[t][metric]]
                         if not targets_with_m:
                             continue
                         
                         targets_with_m.sort()
                         val_strs = []
                         t_names = []
                         
                         for t in targets_with_m:
                             stats = target_stats[t][metric]
                             m_val = stats.get('mean', 0)
                             s_val = stats.get('std', 0)
                             
                             if metric.lower() == 'r2':
                                 # Format as percentage if it looks like r2 (<=1.0)
                                 # Use absolute value check to allow for negative R2
                                 if abs(m_val) <= 1.0:
                                     val_strs.append(f"{m_val*100:.2f}% (±{s_val*100:.2f}%)")
                                 else:
                                     # Already percentage? Unlikely for R2 std implementation usually, strictly assumes 0-1
                                     # But just in case
                                     val_strs.append(f"{m_val:.2f}% (±{s_val:.2f}%)")
                             else:
                                 # For MAE/RMSE, use raw values
                                 val_strs.append(f"{m_val:.4f} (±{s_val:.4f})")
                             
                             t_names.append(t)
                         
                         if val_strs:
                             if len(val_strs) > 1:
                                 v_join = ", ".join(val_strs[:-1]) + " and " + val_strs[-1]
                                 n_join = ", ".join(t_names[:-1]) + " and " + t_names[-1]
                             else:
                                 v_join = val_strs[0]
                                 n_join = t_names[0]
                             
                             summary_lines.append(f"{metric.upper()} of {v_join} for {n_join}, respectively.")

                    if summary_lines:
                        f.write("\n".join(summary_lines) + "\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"📄 详细报告已保存到: {report_path}")
