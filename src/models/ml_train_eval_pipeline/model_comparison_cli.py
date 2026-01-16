#!/usr/bin/env python3
"""
机器学习模型对比命令行工具
ML Model Comparison Command Line Tool

简化版本，专注于命令行参数运行
Simplified version focused on command line arguments
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    # 尝试相对导入（当作为模块运行时）
    from .model_comparator import ModelComparator
except ImportError:
    # 尝试直接导入（当直接运行时）
    try:
        from model_comparator import ModelComparator
    except ImportError:
        # 使用完整路径导入
        from src.models.ml_train_eval_pipeline.model_comparator import ModelComparator


def create_cli_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器
    Create command line argument parser
    """
    parser = argparse.ArgumentParser(
        description='机器学习模型对比工具 / ML Model Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法 / Example Usage:

1. 基础对比 / Basic comparison:
   python -m 'src.models.ml_train_eval_pipeline.model_comparison_cli \
       --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
       --result_dir "output/results/Ti_alloys/Xue/ID/model_comparison_example" \
       --target_columns "UTS(MPa)" "El(%)" \
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
    
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='重复实验次数 / Number of experiment repeats')
    
    return parser


def run_cli_comparison(args) -> Dict[str, Any]:
    """
    运行命令行模式的模型对比
    Run model comparison in CLI mode
    """
    print("🚀 机器学习模型对比 / ML Model Comparison")
    print("=" * 80)
    
    # 构建配置字典
    config = {
        'data_file': args.data_file,
        'result_dir': args.result_dir,
        'target_columns': args.target_columns,
        'processing_cols': args.processing_cols,
        'use_composition_feature': args.use_composition_feature,
        'use_temperature': args.use_temperature,
        'other_features_name': args.other_features_name,
        'cross_validate': args.cross_validate,
        'num_folds': args.num_folds,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'evaluate_after_train': args.evaluate_after_train,
        'run_shap_analysis': args.run_shap_analysis,
        'study_name': args.study_name,
        'mlp_max_iter': args.mlp_max_iter
    }
    
    # 显示配置信息
    print(f"📁 数据文件: {args.data_file}")
    print(f"📂 结果目录: {args.result_dir}")
    print(f"🎯 目标列: {', '.join(args.target_columns)}")
    print(f"[模型] 对比模型: {', '.join(args.models)}")
    print(f"[参数] 使用Optuna优化: {'是' if args.use_optuna else '否'}")
    if args.use_optuna:
        print(f"[参数] Optuna试验次数: {args.n_trials}")
    print(f"[参数] 交叉验证: {'是' if args.cross_validate else '否'} ({args.num_folds} 折)")
    print(f"[实验] 重复次数: {args.n_repeats}")
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")
    
    # 创建模型对比器
    comparator = ModelComparator(config)
    
    # 运行对比
    results = comparator.compare_models(
        models_to_compare=args.models,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        n_repeats=args.n_repeats
    )
    
    print(f"\n[完成] 模型对比完成！结果保存在: {comparator.comparison_dir}")
    print("\n[结果] 查看以下文件获取详细结果:")
    print("  - model_comparison_results.csv: 对比表格")
    print("  - model_comparison_plots.png: 可视化图表")
    print("  - comparison_report.txt: 详细文本报告")
    
    return results


def main():
    """
    主函数
    Main function
    """
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        results = run_cli_comparison(args)
        print(f"\n✅ 成功对比了 {len([r for r in results.values() if 'error' not in r])} 个模型")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {str(e)}")
        print("\n💡 提示:")
        print("- 检查数据文件路径是否正确")
        print("- 确保目标列名存在于数据中")
        print("- 使用 --help 查看参数说明")
        sys.exit(1)





if __name__ == '__main__':

    main()


"""
=============================================================================
真实数据集模型对比示例命令 / Real Dataset Model Comparison Examples
=============================================================================

1. 钛合金数据集标准对比 / Titanium Alloys Standard Comparison:

python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
    --result_dir "output/results/Ti_alloys/Xue/ID/" \
    --target_columns "UTS(MPa)" "El(%)" \
    --processing_cols "Solution Temperature(℃)" "Solution Time(h)" "Aging Temperature(℃)" "Aging Time(h)" "Thermo-Mechanical Treatment Temperature(℃)" "Deformation(%)" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50
2. 铝合金数据集对比 / Aluminum Alloys Comparison:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Al_Alloys/USTB/USTB_Al_alloys_processed_split_withID.csv" \
    --result_dir "output/results/Al_alloys/USTB_new/ID/" \
    --target_columns "UTS(MPa)" \
    --processing_cols "ST1" "TIME1" "ST2" "TIME2" "ST3" "TIME3" "Cold_Deformation_percent" "First_Aging_Temp_C" "First_Aging_Time_h" "Second_Aging_Temp_C" "Second_Aging_Time_h" "Third_Aging_Temp_C" "Third_Aging_Time_h" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

3. 铌合金数据集对比（含温度特征）/ Niobium Alloys Comparison (with Temperature):
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Nb_Alloys/Nb_cleandata/Nb_clean_with_processing_sequence_withID.csv" \
    --result_dir "output/results/Nb_alloys/Nb_cleandata/ID/withTemp" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --processing_cols "Temperature((K))" "Anealing Temperature((K))" "Anealing times(h)" "Thermo-Mechanical Treatment Temperature((K))" "Deformation(%)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --use_temperature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

4. 铌合金数据集对比（不含温度特征）/ Niobium Alloys Comparison (without Temperature):
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Nb_Alloys/Nb_cleandata/Nb_clean_with_processing_sequence_withID.csv" \
    --result_dir "output/results/Nb_alloys/Nb_cleandata/ID/noTemp" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --processing_cols "Anealing Temperature((K))" "Anealing times(h)" "Thermo-Mechanical Treatment Temperature((K))" "Deformation(%)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

5. 高熵合金数据集对比 / High Entropy Alloys Comparison:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/HEA_data/RoomTemperature_HEA_with_ID.csv" \
    --result_dir "output/results/HEA_data/yasir_data/ID/" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --processing_cols "Hom_Temp(K)" "CR(%)" "recrystalize temperature/K" "recrystalize time/mins" "Anneal_Temp(K)" "Anneal_Time(h)" "aging temperature/K" "aging time/hours" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

    
    5. 高熵合金数据集对比 / High Entropy Alloys Comparison:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/HEA_data/RoomTemperature_HEA_with_ID.csv" \
    --result_dir "output/results/HEA_data/yasir_data/ID/" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --processing_cols "Hom_Temp(K)" "CR(%)" "recrystalize temperature/K" "recrystalize time/mins" "Anneal_Temp(K)" "Anneal_Time(h)" "aging temperature/K" "aging time/hours" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

    python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets\HEA_data\RoomTemperature_HEA_train_with_ID.csv" \
    --result_dir "output/results/HEA_data/yasir_data_half/ID/" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --processing_cols "Hom_Temp(K)" "CR(%)" "recrystalize temperature/K" "recrystalize time/mins" "Anneal_Temp(K)" "Anneal_Time(h)" "aging temperature/K" "aging time/hours" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

6. 铌合金高温压缩强度对比 / Niobium Alloys High-Temperature Compressive Strength:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Nb_Alloys/Harbin/HTC_processed_withID.csv" \
    --result_dir "output/results/Nb_alloys/Harbin/HTC/" \
    --target_columns "high-temperature compressive strength(MPa)" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

7. 铌合金断裂韧性对比 / Niobium Alloys Fracture Toughness:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Nb_Alloys/Harbin/KQ_processed_withID.csv" \
    --result_dir "output/results/Nb_alloys/Harbin/KQ/" \
    --target_columns "KQ(MPa·m^(1/2))" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --use_composition_feature \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

8. 铌合金断裂韧性对比 加入特征 / Niobium Alloys Fracture Toughness:
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Nb_Alloys/Harbin/5 Selected Features for KQ_with_ID.csv" \
    --result_dir "output/results/Nb_alloys/Harbin/KQ_enhanced/" \
    --target_columns "KQ(MPa·m^(1/2))" \
    --models xgboost sklearn_rf mlp lightgbm catboost \
    --processing_cols "PT4m" "Ω" "SL1m" "Λ"	"JN1" \
    --cross_validate \
    --num_folds 9 \
    --test_size 0.2 \
    --random_state 42 \
    --evaluate_after_train \
    --run_shap_analysis \
    --use_optuna \
    --n_trials 50

=============================================================================
注意事项 / Notes:
- 所有命令都已转换为模型对比格式，支持多模型同时对比
- 保留了原始的数据集路径和参数设置
- 添加了多个模型选项：xgboost, sklearn_rf, mlp, lightgbm, catboost
- 可以根据需要调整模型列表和参数
- 使用 --help 查看所有可用参数
=============================================================================
"""
