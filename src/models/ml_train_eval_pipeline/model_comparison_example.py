"""
简化的模型对比脚本
Simplified Model Comparison Script

提供模型对比的核心功能接口
Provides core model comparison functionality interface
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    # 尝试相对导入（当作为模块运行时）
    from .model_comparator import ModelComparator
    from .config_manager import (
        get_config_parser,
        load_config_from_file,
        merge_configs,
        get_default_config,
        validate_config
    )
except ImportError:
    # 尝试直接导入（当直接运行时）
    try:
        from model_comparator import ModelComparator
        from config_manager import (
            get_config_parser,
            load_config_from_file,
            merge_configs,
            get_default_config,
            validate_config
        )
    except ImportError:
        # 使用完整路径导入
        from src.models.ml_train_eval_pipeline.model_comparator import ModelComparator
        from src.models.ml_train_eval_pipeline.config_manager import (
            get_config_parser,
            load_config_from_file,
            merge_configs,
            get_default_config,
            validate_config
        )


def run_model_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行模型对比的核心函数
    Core function to run model comparison
    
    Args:
        config: 配置字典，包含所有必要的参数
        
    Returns:
        包含所有模型结果的字典
    """
    print("🚀 机器学习模型对比 / ML Model Comparison")
    print("=" * 80)
    
    # 验证配置
    validate_config(config)
    
    # 提取模型对比参数
    models_to_compare = config['models']
    use_optuna = config.get('use_optuna', False)
    n_trials = config.get('n_trials', 20)
    
    print(f"📋 将对比以下模型: {', '.join(models_to_compare)}")
    print(f"🔧 使用Optuna优化: {'是' if use_optuna else '否'}")
    if use_optuna:
        print(f"🔢 Optuna试验次数: {n_trials}")
    
    # 创建模型对比器
    comparator = ModelComparator(config)
    
    # 开始对比
    results = comparator.compare_models(
        models_to_compare=models_to_compare,
        use_optuna=use_optuna,
        n_trials=n_trials
    )
    
    print(f"\n[完成] 模型对比完成！结果保存在: {comparator.comparison_dir}")
    print("\n[结果] 查看以下文件获取详细结果:")
    print("  - model_comparison_results.csv: 对比表格")
    print("  - model_comparison_plots.png: 可视化图表")
    print("  - comparison_report.txt: 详细文本报告")
    print("  - 各模型子目录: 包含每个模型的详细训练结果")
    
    return results


def main():
    """
    主函数 - 解析参数并运行模型对比
    Main function - Parse arguments and run model comparison
    """
    parser = get_config_parser()
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = {}
        
        # 如果指定了配置文件，先加载文件配置
        if args.config:
            config = load_config_from_file(args.config)
        
        # 将命令行参数转换为字典
        args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
        
        # 合并配置（命令行参数优先）
        if args_dict:
            config = merge_configs(config, args_dict)
        
        # 如果没有配置，使用默认配置
        if not config:
            config = get_default_config()
        
        # 运行模型对比
        run_model_comparison(config)
        
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        print("\n💡 提示:")
        print("- 检查配置文件格式是否正确")
        print("- 确保数据文件路径存在")
        print("- 使用 --help 查看参数说明")
        sys.exit(1)


if __name__ == '__main__':
    main()
