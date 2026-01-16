"""
配置管理模块
Configuration Manager Module

处理配置文件加载、参数解析和验证
Handles configuration file loading, parameter parsing and validation
"""

import json
import argparse
from typing import Dict, Any


def get_config_parser() -> argparse.ArgumentParser:
    """
    创建配置解析器
    Create configuration parser
    """
    parser = argparse.ArgumentParser(
        description='机器学习模型对比工具 / ML Model Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法 / Example Usage:

1. 基础对比 / Basic comparison:
   python model_comparison_cli.py \\
       --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \\
       --result_dir "output/results" \\
       --target_columns "UTS(MPa)" "El(%)" \\
       --models xgboost sklearn_rf mlp

2. 带优化的对比 / Comparison with optimization:
   python model_comparison_cli.py \\
       --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \\
       --result_dir "output/results" \\
       --target_columns "UTS(MPa)" \\
       --models xgboost lightgbm \\
       --use_optuna --n_trials 10

3. 自定义特征 / Custom features:
   python model_comparison_cli.py \\
       --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \\
       --result_dir "output/results" \\
       --target_columns "UTS(MPa)" "El(%)" \\
       --models sklearn_rf sklearn_svr \\
       --processing_cols "Solution Temperature()" "Solution Time(h)" \\
       --use_composition_feature
        """
    )
    
    # 配置文件选项
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    
    # 必需参数
    parser.add_argument('--data_file', type=str, required=True,
                        help='数据文件路径 / Path to data file')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='结果保存目录 / Results output directory')
    parser.add_argument('--target_columns', type=str, nargs='+', required=True,
                        help='目标列名 / Target column names')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        choices=['xgboost', 'lightgbm', 'sklearn_gpr', 'catboost', 'sklearn_rf', 'sklearn_svr', 'mlp'],
                        help='要对比的模型 / Models to compare')
    
    # 特征设置
    parser.add_argument('--processing_cols', type=str, nargs='*', default=[],
                        help='处理列名 / Processing column names')
    parser.add_argument('--use_composition_feature', action='store_true',
                        help='使用成分特征 / Use composition features')
    parser.add_argument('--use_temperature', action='store_true',
                        help='使用温度特征 / Use temperature features')
    parser.add_argument('--other_features_name', type=str, nargs='*', default=None,
                        help='其他特征名 / Other feature names')
    
    # 训练设置
    parser.add_argument('--cross_validate', action='store_true', default=True,
                        help='启用交叉验证 / Enable cross validation')
    parser.add_argument('--num_folds', type=int, default=3,
                        help='交叉验证折数 / Number of CV folds')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例 / Test set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子 / Random seed')
    parser.add_argument('--evaluate_after_train', action='store_true', default=True,
                        help='训练后评估 / Evaluate after training')
    parser.add_argument('--run_shap_analysis', action='store_true',
                        help='运行SHAP分析 / Run SHAP analysis')
    
    # Optuna设置
    parser.add_argument('--use_optuna', action='store_true',
                        help='使用Optuna优化 / Use Optuna optimization')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Optuna试验次数 / Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='model_comparison_optimization',
                        help='Optuna研究名称 / Optuna study name')
    
    # MLP特定参数
    parser.add_argument('--mlp_max_iter', type=int, default=500,
                        help='MLP最大迭代次数 / MLP max iterations')
    
    return parser


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置
    Load configuration from JSON file
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {str(e)}")
        raise


def merge_configs(file_config: Dict[str, Any], args_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并文件配置和命令行配置（命令行优先）
    Merge file config and command line config (command line takes priority)
    """
    merged = file_config.copy()
    
    # 只更新非None的命令行参数
    for key, value in args_config.items():
        if value is not None:
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    Get default configuration
    """
    return {
        'data_file': "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv",
        'result_dir': "output/results/model_comparison_example",
        'models': ['xgboost', 'sklearn_rf', 'mlp', 'sklearn_svr', 'lightgbm', 'catboost'],
        'target_columns': ["UTS(MPa)", "El(%)"],
        'processing_cols': [
            "Solution Temperature()", "Solution Time(h)",
            "Aging Temperature()", "Aging Time(h)",
            "Thermo-Mechanical Treatment Temperature()", "Deformation(%)"
        ],
        'use_composition_feature': True,
        'use_temperature': False,
        'other_features_name': None,
        'cross_validate': True,
        'num_folds': 3,
        'test_size': 0.2,
        'random_state': 42,
        'evaluate_after_train': True,
        'run_shap_analysis': False,
        'use_optuna': True,
        'n_trials': 20,
        'study_name': 'model_comparison_optimization',
        'mlp_max_iter': 500
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置参数
    Validate configuration parameters
    """
    required_params = ['data_file', 'result_dir', 'target_columns', 'models']
    for param in required_params:
        if param not in config:
            raise ValueError(f"缺少必要的配置参数: {param}")
    
    # 验证模型列表
    valid_models = ['xgboost', 'lightgbm', 'sklearn_gpr', 'catboost', 'sklearn_rf', 'sklearn_svr', 'mlp']
    for model in config['models']:
        if model not in valid_models:
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {valid_models}")
    
    print("✅ 配置验证通过")
