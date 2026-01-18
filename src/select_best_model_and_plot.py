"""
自动选择性能最好的模型，并从该模型中选择接近平均值的结果进行绘制
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
import seaborn as sns
from typing import List, Dict, Tuple
import shutil

# 设置字体
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300


def find_all_predictions_by_model(base_dir: str) -> Dict[str, List[Path]]:
    """
    查找所有模型的 fold 预测文件 (all_predictions.csv) 并按模型分组
    允许 optuna_trials，以便我们能从中选出最佳 trial
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        字典，键为模型名，值为文件路径列表
    """
    base_path = Path(base_dir)
    # 查找所有包含 all_predictions.csv 的文件
    all_files = list(base_path.rglob("*all_predictions.csv"))
    
    # 按模型分组
    model_files = {}
    model_map = {
        'catboost_results': 'CatBoost',
        'lightgbm_results': 'LightGBM',
        'xgboost_results': 'XGB',
        'sklearn_rf_results': 'RF',
        'mlp_results': 'MLP'
    }
    
    valid_files = []
    
    for file_path in all_files:
        path_str = str(file_path)
        
        # 排除 顶层的 all_predictions.csv (即不包含 fold_ 也不包含 optuna 的)
        # 通常有效的交叉验证文件都在 fold_ 下
        
        # 必须包含 'fold_'
        is_fold = 'fold_' in path_str.lower()
        
        if not is_fold:
            # print(f"DEBUG: Skipping non-fold: {path_str}")
            continue
        
        valid_files.append(file_path)

        for key, model_name in model_map.items():
            if key in path_str:
                if model_name not in model_files:
                    model_files[model_name] = []
                model_files[model_name].append(file_path)
                break
    
    print(f"\n找到 {len(all_files)} 个 all_predictions.csv 文件")
    print(f"保留 {len(valid_files)} 个包含 fold 信息的文件")
    
    for model, files in model_files.items():
        print(f"  {model}: {len(files)} 个文件")
    
    return model_files


def calculate_r2_for_file(file_path: Path, target_property: str = None) -> Dict:
    """
    计算单个文件的 R2 值 (仅计算测试集)
    """
    try:
        df = pd.read_csv(file_path)
        
        # 只保留测试集数据
        if 'Dataset' in df.columns:
            preferred_splits = ['Test', 'Validation', 'Valid', 'Val', 'test', 'validation', 'val']
            split_used = None
            for split in preferred_splits:
                if (df['Dataset'] == split).any():
                    split_used = split
                    break
            if split_used:
                df = df[df['Dataset'] == split_used].copy()
            else:
                df = df.copy()
        
        if len(df) == 0:
            return {}
        
        # 查找所有属性的实际值列
        actual_cols = [col for col in df.columns if col.endswith('_Actual')]
        
        results = {}
        for actual_col in actual_cols:
            property_name = actual_col.replace('_Actual', '')
            
            if target_property and property_name != target_property:
                continue
            
            pred_col = f"{property_name}_Predicted"
            if pred_col not in df.columns:
                continue
            
            valid_df = df[[actual_col, pred_col]].dropna()
            if len(valid_df) == 0:
                continue
            
            r2 = r2_score(valid_df[actual_col], valid_df[pred_col])
            results[property_name] = {
                'r2': r2,
                'n_samples': len(valid_df),
                'file_path': str(file_path)
            }
        
        return results
    except Exception as e:
        # print(f"错误: 处理文件 {file_path} 时出错: {e}") 
        return {}


def calculate_model_performance(model_files: Dict[str, List[Path]]) -> pd.DataFrame:
    """
    计算每个模型的性能统计。
    逻辑：
    1. 识别 Trial (如果不含 optuna_trials，则视为 default trial)
    2. 按 Trial 分组计算平均 R2
    3. 选出最佳 Trial
    4. 仅返回最佳 Trial 的 Fold 统计信息
    
    Args:
        model_files: 模型文件字典
        
    Returns:
        包含模型性能统计的DataFrame
    """
    import re
    all_metrics = []
    
    for model_name, file_paths in model_files.items():
        if not file_paths:
            continue
            
        print(f"正在处理模型 {model_name}... (共 {len(file_paths)} 个文件)")
        
        # 1. 解析文件并按 Trial 分组
        trials_data = {}  # {trial_id: {property: [ (r2, file_path), ... ] }}
        
        for file_path in file_paths:
            path_str = str(file_path)
            
            # 提取 Trial ID
            match = re.search(r'trial_(\d+)', path_str)
            if match:
                trial_id = f"trial_{match.group(1)}"
            else:
                 trial_id = 'default_trial'

            
            # 计算该文件的 R2
            r2_results = calculate_r2_for_file(file_path)
            
            if not r2_results:
                continue
                
            if trial_id not in trials_data:
                trials_data[trial_id] = {}
            
            for prop, metrics in r2_results.items():
                if prop not in trials_data[trial_id]:
                    trials_data[trial_id][prop] = []
                trials_data[trial_id][prop].append({
                    'r2': metrics['r2'],
                    'n_samples': metrics['n_samples'],
                    'file_path': metrics['file_path']
                })
        
        # 2. 对每个属性，找到最佳 Trial
        # 我们假设所有 Trial 都处理相同的属性集合
        
        # 获取所有属性
        all_props = set()
        for t_data in trials_data.values():
            all_props.update(t_data.keys())
            
        for prop in all_props:
            best_trial_id = None
            best_trial_mean_r2 = -float('inf')
            
            # 遍历该模型所有 Trial，找该属性 R2 最高的
            for trial_id, prop_map in trials_data.items():
                if prop not in prop_map:
                    continue
                
                # 计算该 Trial 所有 Fold 的平均 R2
                folds_r2 = [item['r2'] for item in prop_map[prop]]
                if not folds_r2: 
                    continue
                
                num_folds = len(folds_r2)
                mean_r2 = sum(folds_r2) / num_folds
                
                # 更新最佳 Trial
                # 策略：优先选择 Fold 数更多的（完整运行的），其次选 R2 更高的
                # 如果当前 Trial 的 Fold 数比之前记录的最佳 Trial 更多，直接替换
                if best_trial_id is None or num_folds > len(trials_data[best_trial_id][prop]):
                    best_trial_mean_r2 = mean_r2
                    best_trial_id = trial_id
                # 如果 Fold 数相同，则比较 R2
                elif num_folds == len(trials_data[best_trial_id][prop]):
                    if mean_r2 > best_trial_mean_r2:
                        best_trial_mean_r2 = mean_r2
                        best_trial_id = trial_id
            
            if best_trial_id:
                fold_count = len(trials_data[best_trial_id][prop])
                print(f"  属性 {prop}: 最佳 Trial 是 {best_trial_id} (Mean R2: {best_trial_mean_r2:.4f}, Folds: {fold_count})")
                
                # 3. 数据收集
                # 遍历该属性下所有 Trial
                for trial_id, prop_map in trials_data.items():
                    if prop not in prop_map:
                        continue
                        
                    folds_data = prop_map[prop]
                    for item in folds_data:
                        # A. 添加到 "All Trials" 统计 (用于对比)
                        all_metrics.append({
                            'model': f"{model_name} (All Trials)",
                            'property': prop,
                            'r2': item['r2'],
                            'n_samples': item['n_samples'],
                            'file_path': item['file_path'],
                            'trial_id': trial_id
                        })
                        
                        # B. 如果是最佳 Trial，添加到 "Main" 统计 (用于绘图和代表性选择)
                        if trial_id == best_trial_id:
                            all_metrics.append({
                                'model': model_name,
                                'property': prop,
                                'r2': item['r2'],
                                'n_samples': item['n_samples'],
                                'file_path': item['file_path'],
                                'trial_id': trial_id
                            })
    
    df = pd.DataFrame(all_metrics)
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 计算统计信息 (基于最佳 Trial 的 Folds)
    stats = df.groupby(['model', 'property'])['r2'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    return df, stats



def plot_model_comparison(stats_df: pd.DataFrame, output_dir: str):
    """
    绘制模型对比图（生成两版：最佳Trial和所有Trials）
    
    Args:
        stats_df: 模型统计DataFrame
        output_dir: 输出目录
    """
    if stats_df.empty or 'property' not in stats_df.columns or 'model' not in stats_df.columns:
        print("Warning: stats data missing; skip model comparison plots")
        return

    # 定义模型顺序和颜色
    model_order = ['CatBoost', 'LightGBM', 'MLP', 'RF', 'XGB']
    color_map = {
        'Ep(mV)': '#f4c542',  # Gold
        'UTS(MPa)': '#aacfef',  # Light Blue
        'El(%)': '#ffb3a7',  # Light Pink
        'YS(MPa)': '#c5e1a5'  # Light Green
    }
    
    properties = sorted(stats_df['property'].unique())
    
    # 分离数据
    # 1. Best Trials (模型名不包含 "All Trials")
    df_best = stats_df[~stats_df['model'].str.contains("All Trials")].copy()
    
    # 2. All Trials (模型名包含 "All Trials")
    df_all = stats_df[stats_df['model'].str.contains("All Trials")].copy()
    # 清理模型名以便绘图 (去掉后缀)
    df_all['model'] = df_all['model'].str.replace(r" \(All Trials\)", "", regex=True)
    
    def _plot_single_comparison(data_df: pd.DataFrame, suffix: str, title_prefix: str = ""):
        if data_df.empty:
            return

        # ???????????????????????????????????????????????????
        data_props = [prop for prop in properties if not data_df[data_df['property'] == prop].empty]
        if not data_props:
            return

        x = np.arange(len(model_order))
        group_width = 0.8
        bar_width = group_width / max(len(data_props), 1)

        plt.figure(figsize=(10, 8))

        for idx, prop in enumerate(data_props):
            prop_data = data_df[data_df['property'] == prop]
            if prop_data.empty:
                continue

            # ???????????????????????????
            prop_data = prop_data.set_index('model').reindex(model_order).reset_index()
            prop_data = prop_data.dropna(subset=['mean'])
            if prop_data.empty:
                continue

            means = prop_data['mean'].tolist()
            stds = prop_data['std'].tolist()

            offset = (idx - (len(data_props) - 1) / 2) * bar_width
            plt.bar(
                x + offset,
                means,
                yerr=stds,
                width=bar_width,
                capsize=4,
                color=color_map.get(prop, '#aacfef'),
                edgecolor='black',
                linewidth=1.2,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                label=prop
            )

        plt.xlabel('Predictive models', fontweight='bold', fontsize=18)
        plt.ylabel('R??', fontweight='bold', fontsize=18)
        plt.ylim(0, 1.0)
        plt.xticks(x, model_order, fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.legend(frameon=True, fontsize=12, edgecolor='black')

        # ???????????????????????????
        # plt.title(f"{title_prefix}Model Comparison", fontsize=20, fontweight='bold', pad=20)

        plt.tight_layout()

        output_path = Path(output_dir) / f"model_comparison_all_properties_{suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"?????????????????????({suffix}): {output_path}")
        plt.close()

    # 绘制两版图表
    print("\n绘制 Best Trials 对比图...")
    _plot_single_comparison(df_best, "Best")
    
    print("\n绘制 All Trials 对比图...")
    _plot_single_comparison(df_all, "AllTrials")


def select_best_model_representative(all_results_df: pd.DataFrame, property_name: str) -> Tuple[str, Path, float]:
    """
    选择性能最好的模型，并从中选择最接近平均值的结果
    
    Args:
        all_results_df: 所有结果的DataFrame
        property_name: 目标属性名称
        
    Returns:
        (最佳模型名, 代表性文件路径, R2值)
    """
    # 筛选指定属性的数据
    prop_data = all_results_df[all_results_df['property'] == property_name]
    
    # 计算每个模型的平均R2
    model_means = prop_data.groupby('model')['r2'].mean().sort_values(ascending=False)
    
    if len(model_means) == 0:
        raise ValueError(f"未找到属性 {property_name} 的数据")
    
    # 选择最佳模型
    best_model = model_means.index[0]
    best_mean_r2 = model_means.iloc[0]
    
    print(f"\n属性 {property_name}:")
    print(f"  性能最好的模型: {best_model} (平均 R² = {best_mean_r2:.4f})")
    
    # 从最佳模型中选择最接近平均值的文件
    best_model_data = prop_data[prop_data['model'] == best_model]
    mean_r2 = best_model_data['r2'].mean()
    
    # 找到最接近平均值的结果
    closest_idx = (best_model_data['r2'] - mean_r2).abs().idxmin()
    representative_file = best_model_data.loc[closest_idx, 'file_path']
    representative_r2 = best_model_data.loc[closest_idx, 'r2']
    
    print(f"  选择的代表性结果: R² = {representative_r2:.4f} (接近平均值 {mean_r2:.4f})")
    print(f"  文件路径: {representative_file}")
    
    return best_model, Path(representative_file), representative_r2


def plot_diagonal_chart(file_path: Path, property_name: str, model_name: str, 
                       r2_value: float, output_dir: str):
    """
    绘制对角线图 (预测值 vs 实际值)
    
    Args:
        file_path: CSV文件路径
        property_name: 属性名称
        model_name: 模型名称
        r2_value: R2值
        output_dir: 输出目录
    """
    df = pd.read_csv(file_path)
    
    # 只保留测试集数据
    if 'Dataset' in df.columns:
        df = df[df['Dataset'] == 'Test'].copy()
    
    # 查找实际值和预测值列
    actual_col = f"{property_name}_Actual"
    pred_col = f"{property_name}_Predicted"
    
    if actual_col not in df.columns or pred_col not in df.columns:
        print(f"警告: 未找到 {property_name} 的列")
        return
    
    # 移除缺失值
    valid_df = df[[actual_col, pred_col]].dropna()
    
    if len(valid_df) == 0:
        print(f"警告: 没有有效数据")
        return
    
    # 绘制对角线图
    plt.figure(figsize=(8, 8))
    
    # 散点图
    plt.scatter(valid_df[actual_col], valid_df[pred_col], 
               alpha=0.6, s=80, edgecolors='black', linewidth=0.8,
               c='#4CAF50', label='Test Set Predictions')
    
    # 绘制对角线 (y=x)
    min_val = min(valid_df[actual_col].min(), valid_df[pred_col].min())
    max_val = max(valid_df[actual_col].max(), valid_df[pred_col].max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', linewidth=2.5, label='Perfect Prediction (y=x)')
    
    # 设置标签和标题
    property_label = property_name.replace('_', ' ')
    plt.xlabel(f'Experimental {property_label}', fontsize=16, fontweight='bold')
    plt.ylabel(f'Predicted {property_label}', fontsize=16, fontweight='bold')
    plt.title(f'{model_name} - {property_label}\nR² = {r2_value:.4f}', 
             fontsize=18, fontweight='bold')
    
    plt.legend(loc='upper left', fontsize=14, frameon=True, 
              edgecolor='black', fancybox=False)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置相等的坐标轴比例
    plt.axis('equal')
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    
    # 保存图表
    safe_prop_name = property_name.replace('(', '').replace(')', '').replace('%', 'pct')
    output_path = Path(output_dir) / f"best_model_diagonal_{safe_prop_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存对角线图: {output_path}")
    plt.close()


def find_model_comparison_dirs(base_dir: str) -> List[Path]:
    """
    递归查找所有 model_comparison 目录
    
    Args:
        base_dir: 基础目录
        
    Returns:
        model_comparison 目录列表
    """
    base_path = Path(base_dir)
    model_dirs = []
    
    for root, dirs, files in os.walk(base_path):
        if 'model_comparison' in dirs:
            model_dirs.append(Path(root) / 'model_comparison')
    
    return model_dirs


def process_single_directory(base_dir: str, output_dir: str):
    """
    处理单个 model_comparison 目录
    
    Args:
        base_dir: model_comparison 目录路径
        output_dir: 输出目录路径
        
    Returns:
        包含最佳模型信息的字典，失败返回None
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"分析目录: {base_dir}")
    print("=" * 80)
    
    # 1. 查找所有模型的文件
    model_files = find_all_predictions_by_model(base_dir)
    
    if not model_files:
        print("  警告: 未找到任何模型文件，跳过\n")
        return None
    
    # 2. 计算所有模型的性能
    print("\n计算模型性能统计...")
    
    all_results_df, stats_df = calculate_model_performance(model_files)
    
    # 保存统计结果
    stats_path = Path(output_dir) / "model_performance_statistics.csv"
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    print(f"保存统计结果: {stats_path}")
    
    # 显示统计结果
    print("\n模型性能统计:")
    print(stats_df.to_string(index=False))
    
    # 3. 绘制模型对比图
    print("\n绘制模型对比图...")
    plot_model_comparison(stats_df, output_dir)
    
    # 4. 对每个属性，选择最佳模型的代表性结果
    print("\n选择最佳模型的代表性结果...")
    
    if all_results_df.empty or 'property' not in all_results_df.columns:
        print("  Warning: no performance data available; skipping best model selection\n")
        return None

    
    properties = all_results_df['property'].unique()

    best_results = []
    best_models_info = {}  # 存储每个属性的最佳模型信息
    
    for prop in properties:
        try:
            best_model, repr_file, repr_r2 = select_best_model_representative(
                all_results_df, prop
            )
            
            best_results.append({
                'property': prop,
                'best_model': best_model,
                'representative_r2': repr_r2,
                'file_path': str(repr_file)
            })
            
            # 记录最佳模型信息
            best_models_info[prop] = best_model
            
            # 绘制对角线图
            plot_diagonal_chart(repr_file, prop, best_model, repr_r2, output_dir)
            
            # 复制代表性文件
            safe_prop_name = prop.replace('(', '').replace(')', '').replace('%', 'pct')
            dst_path = Path(output_dir) / f"best_model_{best_model}_{safe_prop_name}.csv"
            shutil.copy2(repr_file, dst_path)
            print(f"  复制代表性文件: {dst_path}")
            
        except Exception as e:
            print(f"  警告: 处理属性 {prop} 时出错: {e}")
    
    # 保存最佳结果汇总
    if best_results:
        best_summary_df = pd.DataFrame(best_results)
        summary_path = Path(output_dir) / "best_models_summary.csv"
        best_summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"保存最佳模型汇总: {summary_path}")
    
    print(f"\n完成! 结果已保存到: {output_dir}\n")
    
    # 返回最佳模型信息
    return {
        'success': True,
        'best_models': best_models_info,  # {property: model_name}
        'properties': list(properties)
    }


def main():
    """主函数"""
    import sys
    
    # 确定基础目录
    if len(sys.argv) > 1:
        base_search_dir = sys.argv[1]
    else:
        # 默认处理所有合金
        base_search_dir = r"output\new_results_withuncertainty"
    
    base_search_path = Path(base_search_dir)
    
    if not base_search_path.exists():
        print(f"错误: 目录不存在: {base_search_dir}")
        return
    
    print("=" * 80)
    print("开始分析所有合金的模型性能并选择最佳代表性结果")
    print("=" * 80)
    print(f"搜索目录: {base_search_dir}\n")
    
    # 创建汇总目录
    summary_dir = base_search_path / "all_alloys_best_models_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"汇总目录: {summary_dir}\n")
    
    # 查找所有 model_comparison 目录
    model_comparison_dirs = find_model_comparison_dirs(base_search_dir)
    
    if not model_comparison_dirs:
        print(f"错误: 在 {base_search_dir} 中未找到任何 model_comparison 目录")
        return
    
    print(f"找到 {len(model_comparison_dirs)} 个 model_comparison 目录:\n")
    for i, dir_path in enumerate(model_comparison_dirs, 1):
        rel_path = dir_path.relative_to(base_search_path) if dir_path.is_relative_to(base_search_path) else dir_path
        print(f"  {i}. {rel_path}")
    print()
    
    # 处理每个目录
    successful = 0
    failed = 0
    all_summaries = []  # 收集所有汇总信息
    
    for i, model_dir in enumerate(model_comparison_dirs, 1):
        print(f"\n{'='*80}")
        print(f"处理进度: {i}/{len(model_comparison_dirs)}")
        print(f"{'='*80}\n")
        
        # 创建输出目录（在 model_comparison 目录内）
        output_dir = model_dir / "best_model_analysis"
        
        try:
            result = process_single_directory(str(model_dir), str(output_dir))
            if result and result.get('success'):
                successful += 1
                
                # 提取合金名称（用于命名）
                try:
                    rel_path = model_dir.relative_to(base_search_path)
                    alloy_name = rel_path.parts[0] if len(rel_path.parts) > 0 else f"alloy_{i}"
                    dataset_name = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
                    prefix = f"{dataset_name}" if dataset_name else "default"
                    prefix = prefix.replace(" ", "_").replace("(", "").replace(")", "")
                except:
                    alloy_name = f"alloy_{i}"
                    prefix = "default"
                
                # 获取最佳模型信息
                best_models = result.get('best_models', {})
                
                # 为该合金创建子文件夹
                alloy_summary_dir = summary_dir / alloy_name
                alloy_summary_dir.mkdir(parents=True, exist_ok=True)
                
                # 复制结果到汇总目录
                print(f"\n复制结果到汇总目录: {alloy_summary_dir.name}/...")
                
                # 复制图表 - 根据文件名识别属性并添加模型名
                for file in output_dir.glob("*.png"):
                    # 从文件名中提取属性信息
                    filename = file.name
                    model_suffix = ""
                    
                    # 查找匹配的属性和模型
                    for prop, model in best_models.items():
                        safe_prop = prop.replace('(', '').replace(')', '').replace('%', 'pct')
                        if safe_prop in filename:
                            model_suffix = f"_{model}"
                            break
                    
                    # 重新组合文件名：dataset_model_原文件名
                    if prefix != "default" and model_suffix:
                        dst_name = f"{prefix}{model_suffix}_{filename}"
                    elif model_suffix:
                        dst_name = f"{model_suffix[1:]}_{filename}"  # 去掉前导下划线
                    elif prefix != "default":
                        dst_name = f"{prefix}_{filename}"
                    else:
                        dst_name = filename
                    
                    dst = alloy_summary_dir / dst_name
                    shutil.copy2(file, dst)
                    print(f"  图表: {alloy_name}/{dst_name}")
                
                # 复制最佳模型汇总CSV
                summary_csv = output_dir / "best_models_summary.csv"
                if summary_csv.exists():
                    # 读取并添加合金信息
                    df = pd.read_csv(summary_csv)
                    df.insert(0, 'alloy', alloy_name)
                    if dataset_name:
                        df.insert(1, 'dataset', dataset_name)
                    all_summaries.append(df)
                    
                    # 复制到合金子文件夹
                    dst_name = f"{prefix}_best_models_summary.csv" if prefix != "default" else "best_models_summary.csv"
                    dst = alloy_summary_dir / dst_name
                    shutil.copy2(summary_csv, dst)
                    print(f"  汇总: {alloy_name}/{dst_name}")
                
                # 复制统计CSV
                stats_csv = output_dir / "model_performance_statistics.csv"
                if stats_csv.exists():
                    dst_name = f"{prefix}_model_performance_statistics.csv" if prefix != "default" else "model_performance_statistics.csv"
                    dst = alloy_summary_dir / dst_name
                    shutil.copy2(stats_csv, dst)
                    print(f"  统计: {alloy_name}/{dst_name}")
                
                # 复制预测数据CSV文件 - 文件名已经包含模型名
                for pred_file in output_dir.glob("best_model_*.csv"):
                    dst_name = f"{prefix}_{pred_file.name}" if prefix != "default" else pred_file.name
                    dst = alloy_summary_dir / dst_name
                    shutil.copy2(pred_file, dst)
                    print(f"  数据: {alloy_name}/{dst_name}")
                
            else:
                failed += 1
        except Exception as e:
            print(f"错误: 处理目录 {model_dir} 时出错: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 创建总的汇总表格
    if all_summaries:
        print(f"\n{'='*80}")
        print("创建跨合金总汇总表格")
        print(f"{'='*80}\n")
        
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_path = summary_dir / "ALL_ALLOYS_BEST_MODELS_SUMMARY.csv"
        combined_summary.to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"总汇总表格: {combined_path}")
        print(f"\n总汇总内容预览:")
        print(combined_summary.to_string(index=False))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("批量处理总结")
    print("=" * 80)
    print(f"总共处理: {len(model_comparison_dirs)} 个目录")
    print(f"成功: {successful}")
    print(f"失败/跳过: {failed}")
    print(f"\n所有汇总结果已保存到: {summary_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
