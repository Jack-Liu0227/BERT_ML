#!/usr/bin/env python3
"""
端到端机器学习流水线 / End-to-End Machine Learning Pipeline

整合特征工程、模型训练（传统ML和神经网络）、评估为统一工作流
Integrates feature engineering, model training (traditional ML and neural networks), 
and evaluation into a unified workflow.

支持的功能 / Supported Features:
- 特征生成（组分特征、嵌入特征）/ Feature generation (composition, embeddings)
- 传统ML模型训练 / Traditional ML model training
- 神经网络模型训练 / Neural network model training
- 超参数优化（Optuna）/ Hyperparameter optimization (Optuna)
- 交叉验证 / Cross-validation
- 模型评估和SHAP分析 / Model evaluation and SHAP analysis
"""

import os
import sys
import argparse

# 设置标准输出和标准错误为UTF-8编码
# Set stdout and stderr to UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 延迟导入以避免依赖问题
# 这些模块将在需要时才导入
# from src.feature_engineering.feature_processor import FeatureProcessor
# from src.models.ml_train_eval_pipeline.model_comparator import ModelComparator
# from src.models.nn_train_eval_pipeline.pipeline import TrainingPipeline as NNTrainingPipeline


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 导入合金数据配置 / Import Alloy Data Configuration
# ============================================================================
# 从 batch_configs.py 导入合金数据配置和辅助函数
# Import alloy data configuration and helper functions from batch_configs.py
from src.pipelines.batch_configs import (
    ALLOY_CONFIGS,
    get_alloy_config,
    list_available_alloys
)


def str2bool(v):
    """将字符串转换为布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='端到端机器学习流水线 / End-to-End ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # ========== 数据配置 / Data Configuration ==========
    parser.add_argument('--data_file', type=str, required=True,
                        help='输入数据文件路径（原始数据或特征文件）/ Input data file path')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='结果输出目录 / Output directory for results')
    parser.add_argument('--target_columns', type=str, nargs='+', required=True,
                        help='目标预测列名 / Target prediction columns')
    parser.add_argument('--processing_cols', type=str, nargs='*', default=[],
                        help='处理参数数值列名（用于ML模型特征）/ Processing parameter numerical columns (for ML features)')
    parser.add_argument('--processing_text_column', type=str, default=None,
                        help='工艺描述文本列名（用于BERT嵌入），默认为"Processing_Description" / Processing description text column (for BERT embeddings), default "Processing_Description"')
    parser.add_argument('--alloy_type', type=str, default=None,
                        help='合金类型（用于特征目录构建）/ Alloy type (for feature directory construction)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='数据集名称（用于特征目录构建）/ Dataset name (for feature directory construction)')

    # ========== 特征配置 / Feature Configuration ==========
    parser.add_argument('--use_composition_feature', type=str2bool, default=False,
                        help='是否使用组分特征 / Use composition features')
    parser.add_argument('--use_element_embedding', type=str2bool, default=False,
                        help='是否使用元素嵌入 / Use element embeddings')
    parser.add_argument('--use_process_embedding', type=str2bool, default=False,
                        help='是否使用工艺嵌入 / Use process embeddings')
    parser.add_argument('--use_temperature', type=str2bool, default=False,
                        help='是否使用温度特征 / Use temperature features')
    parser.add_argument('--embedding_type', type=str, required=True,
                        choices=['tradition', 'scibert', 'steelbert', 'matscibert'],
                        help='嵌入类型 / Embedding type')
    
    # ========== 模型配置 / Model Configuration ==========
    parser.add_argument('--models', type=str, nargs='*', default=None,
                        choices=['xgboost', 'sklearn_rf', 'mlp', 'lightgbm', 'catboost'],
                        help='传统ML模型列表 / Traditional ML models')
    parser.add_argument('--use_nn', action='store_true',
                        help='使用神经网络模型 / Use neural network model')
    
    # ========== 训练配置 / Training Configuration ==========
    parser.add_argument('--cross_validate', action='store_true', default=False,
                        help='启用交叉验证 / Enable cross-validation')
    parser.add_argument('--num_folds', type=int, default=9,
                        help='交叉验证折数 / Number of CV folds')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例 / Test set size')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子 / Random seed')
    
    # ========== 神经网络特定参数 / Neural Network Specific ==========
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数 / Maximum training epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='早停耐心值 / Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='训练批次大小 / Training batch size')
    
    # ========== 优化配置 / Optimization Configuration ==========
    parser.add_argument('--use_optuna', action='store_true', default=False,
                        help='启用Optuna超参数优化 / Enable Optuna optimization')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna试验次数 / Number of Optuna trials')
    
    # ========== 评估配置 / Evaluation Configuration ==========
    parser.add_argument('--evaluate_after_train', action='store_true', default=True,
                        help='训练后评估 / Evaluate after training')
    parser.add_argument('--run_shap_analysis', action='store_true', default=True,
                        help='运行SHAP分析 / Run SHAP analysis')
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    验证命令行参数的有效性
    Validate command line arguments
    """
    # 验证模型选择互斥性
    if args.use_nn and args.models:
        raise ValueError("--use_nn 和 --models 不能同时指定 / Cannot specify both --use_nn and --models")
    
    if not args.use_nn and not args.models:
        raise ValueError("必须指定 --use_nn 或 --models 之一 / Must specify either --use_nn or --models")
    
    # 验证嵌入类型配置
    if args.embedding_type != 'tradition':
        if not args.use_element_embedding and not args.use_process_embedding:
            raise ValueError(
                f"使用嵌入类型 '{args.embedding_type}' 时，必须指定 --use_element_embedding 或 --use_process_embedding / "
                f"When using embedding type '{args.embedding_type}', must specify --use_element_embedding or --use_process_embedding"
            )
    
    # 验证数据文件存在
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在 / Data file not found: {args.data_file}")
    
    logger.info("参数验证通过 / Arguments validated successfully")


def infer_alloy_type_and_dataset(data_file: str, result_dir: str) -> Tuple[str, str]:
    """
    从文件路径推断合金类型和数据集名称
    Infer alloy type and dataset name from file paths
    """
    supported_alloys = ['Ti_alloys', 'Al_alloys', 'Nb_alloys', 'HEA', 'Steel']

    # 尝试从data_file路径推断
    data_path = Path(data_file)
    for alloy in supported_alloys:
        if alloy in str(data_path):
            # 找到合金类型，尝试提取数据集名称
            parts = data_path.parts
            try:
                alloy_idx = parts.index(alloy)
                if alloy_idx + 1 < len(parts):
                    dataset_name = parts[alloy_idx + 1]
                    logger.info(f"从数据文件推断: 合金类型={alloy}, 数据集={dataset_name}")
                    return alloy, dataset_name
            except (ValueError, IndexError):
                pass

    # 尝试从result_dir路径推断
    result_path = Path(result_dir)
    for alloy in supported_alloys:
        if alloy in str(result_path):
            parts = result_path.parts
            try:
                alloy_idx = parts.index(alloy)
                if alloy_idx + 1 < len(parts):
                    dataset_name = parts[alloy_idx + 1]
                    logger.info(f"从结果目录推断: 合金类型={alloy}, 数据集={dataset_name}")
                    return alloy, dataset_name
            except (ValueError, IndexError):
                pass

    # 无法推断，使用默认值
    logger.warning("无法推断合金类型和数据集名称，使用默认值")
    return "Unknown", "Unknown"


def is_feature_file(file_path: str) -> bool:
    """
    判断输入文件是否为特征文件
    Check if input file is a feature file
    """
    path = Path(file_path)

    # 检查是否在Features目录下
    if 'Features' in path.parts:
        logger.info(f"检测到特征文件（位于Features目录）: {file_path}")
        return True

    # 检查文件名是否包含特征文件标识
    if 'features' in path.name.lower() or 'feature' in path.name.lower():
        logger.info(f"检测到特征文件（文件名包含'feature'）: {file_path}")
        return True

    return False


def generate_features_stage(
    args: argparse.Namespace,
    alloy_type: str,
    dataset_name: str
) -> str:
    """
    特征生成阶段
    Feature generation stage

    Returns:
        str: 特征文件路径 / Path to feature file
    """
    logger.info("=" * 80)
    logger.info("阶段 1: 特征生成 / Stage 1: Feature Generation")
    logger.info("=" * 80)

    # 如果输入已经是特征文件，跳过特征生成
    if is_feature_file(args.data_file):
        logger.info("输入文件已是特征文件，跳过特征生成阶段")
        return args.data_file

    # 构建特征保存目录
    # 目录结构: Features/{alloy_type}/{dataset_name}/{embedding_type}/
    feature_dir = Path("Features") / alloy_type / dataset_name / args.embedding_type
    feature_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"特征保存目录: {feature_dir}")

    # 检查特征文件是否已存在
    feature_file = feature_dir / "features_with_id.csv"
    if feature_file.exists():
        logger.info(f"特征文件已存在: {feature_file}")
        return str(feature_file)

    # 执行特征生成
    logger.info("开始特征生成...")
    log_dir = feature_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入FeatureProcessor
    from src.feature_engineering.feature_processor import FeatureProcessor

    # 映射embedding_type到model_name
    model_name_map = {
        'tradition': 'steelbert',  # 传统特征也需要模型名称（但不使用嵌入）
        'scibert': 'scibert',
        'steelbert': 'steelbert',
        'matscibert': 'matscibert'
    }
    model_name = model_name_map.get(args.embedding_type, 'steelbert')

    # 尝试从配置中获取工艺描述列名
    processing_text_column = "Processing_Description"  # 默认值
    if hasattr(args, 'processing_text_column') and args.processing_text_column:
        processing_text_column = args.processing_text_column
    else:
        # 尝试从合金配置中获取
        try:
            alloy_config = get_alloy_config(alloy_type)
            if 'processing_text_column' in alloy_config:
                processing_text_column = alloy_config['processing_text_column']
        except:
            pass  # 使用默认值

    logger.info(f"工艺描述列名（用于BERT嵌入）: {processing_text_column}")
    logger.info(f"工艺参数列（用于ML特征）: {args.processing_cols if args.processing_cols else '无'}")

    processor = FeatureProcessor(
        data_path=args.data_file,
        use_process_embedding=args.use_process_embedding,
        use_element_embedding=args.use_element_embedding,
        use_composition_feature=args.use_composition_feature,
        use_temperature=args.use_temperature,
        standardize_features=False,  # 标准化在训练时处理
        feature_dir=str(feature_dir),
        log_dir=str(log_dir),
        target_columns=args.target_columns,
        model_name=model_name,
        other_features_name=args.processing_cols if args.processing_cols else None,
        processing_text_column=processing_text_column
    )

    X, y = processor.process()
    logger.info(f"特征生成完成: {X.shape[1]} 个特征, {X.shape[0]} 个样本")

    return str(feature_file)


def train_traditional_ml_models(args: argparse.Namespace, feature_file: str) -> Dict[str, Any]:
    """
    训练传统ML模型
    Train traditional ML models
    """
    logger.info("=" * 80)
    logger.info("阶段 2: 传统ML模型训练 / Stage 2: Traditional ML Model Training")
    logger.info("=" * 80)

    # 延迟导入ModelComparator
    from src.models.ml_train_eval_pipeline.model_comparator import ModelComparator

    # 构建配置
    config = {
        'data_file': feature_file,
        'result_dir': args.result_dir,
        'target_columns': args.target_columns,
        'processing_cols': args.processing_cols,
        'use_composition_feature': args.use_composition_feature,
        'use_temperature': args.use_temperature,
        'other_features_name': args.processing_cols if args.processing_cols else None,
        'cross_validate': args.cross_validate,
        'num_folds': args.num_folds,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'evaluate_after_train': args.evaluate_after_train,
        'run_shap_analysis': args.run_shap_analysis,
        'study_name': 'ml_hyperparameter_optimization',
        'mlp_max_iter': getattr(args, 'mlp_max_iter', 300)  # 从args读取，默认300
    }

    # 创建模型对比器
    comparator = ModelComparator(config)

    # 运行模型对比
    results = comparator.compare_models(
        models_to_compare=args.models,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials
    )

    logger.info(f"传统ML模型训练完成，结果保存在: {comparator.comparison_dir}")
    return results


def train_neural_network_model(args: argparse.Namespace, feature_file: str) -> None:
    """
    训练神经网络模型
    Train neural network model
    """
    logger.info("=" * 80)
    logger.info("阶段 2: 神经网络模型训练 / Stage 2: Neural Network Model Training")
    logger.info("=" * 80)

    # 延迟导入NNTrainingPipeline
    from src.models.nn_train_eval_pipeline.pipeline import TrainingPipeline as NNTrainingPipeline

    # 创建模拟的命令行参数对象
    class NNArgs:
        def __init__(self, pipeline_args: argparse.Namespace, feature_file: str):
            # 数据和目录
            self.data_file = feature_file
            self.result_dir = pipeline_args.result_dir

            # 模型选择
            self.model_type = 'nn'

            # 模型架构
            self.emb_hidden_dim = 256
            self.feature1_hidden_dim = 256
            self.feature2_hidden_dim = 256
            self.other_features_hidden_dim = 0
            self.hidden_dims = [256, 128]
            self.dropout_rate = 0.2

            # 训练参数
            self.epochs = pipeline_args.epochs
            self.batch_size = pipeline_args.batch_size
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = pipeline_args.patience
            self.use_lr_scheduler = False
            self.lr_scheduler_patience = 10
            self.lr_scheduler_factor = 0.5

            # 特征选择
            self.target_columns = pipeline_args.target_columns
            self.use_process_embedding = pipeline_args.use_process_embedding
            self.use_joint_composition_process_embedding = False
            self.use_element_embedding = pipeline_args.use_element_embedding
            self.use_composition_feature = pipeline_args.use_composition_feature
            self.use_feature1 = False
            self.use_feature2 = False
            self.other_features_name = pipeline_args.processing_cols if pipeline_args.processing_cols else None
            self.use_temperature = pipeline_args.use_temperature

            # 执行控制
            self.test_size = pipeline_args.test_size
            self.random_state = pipeline_args.random_state
            self.evaluate_after_train = pipeline_args.evaluate_after_train
            self.device = 'cuda:0'
            self.use_multi_gpu = False
            self.cross_validate = pipeline_args.cross_validate
            self.num_folds = pipeline_args.num_folds

            # Optuna优化
            self.use_optuna = pipeline_args.use_optuna
            self.n_trials = pipeline_args.n_trials
            self.study_name = 'nn_hyperparameter_optimization'

    nn_args = NNArgs(args, feature_file)

    # 创建并运行神经网络训练流水线
    pipeline = NNTrainingPipeline(nn_args)
    pipeline.run()

    logger.info(f"神经网络模型训练完成，结果保存在: {args.result_dir}")


def main():
    """主函数 / Main function"""
    # 解析命令行参数
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # 验证参数
        validate_arguments(args)

        # 确定合金类型和数据集名称
        # 优先使用命令行参数，如果没有则尝试推断
        if args.alloy_type and args.dataset_name:
            alloy_type = args.alloy_type
            dataset_name = args.dataset_name
            logger.info(f"使用命令行参数: 合金类型={alloy_type}, 数据集={dataset_name}")
        else:
            alloy_type, dataset_name = infer_alloy_type_and_dataset(args.data_file, args.result_dir)
            logger.info(f"推断得到: 合金类型={alloy_type}, 数据集={dataset_name}")

        # 创建结果目录
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"结果目录: {result_dir}")

        # 阶段1: 特征生成
        feature_file = generate_features_stage(args, alloy_type, dataset_name)

        # 阶段2: 模型训练
        if args.models:
            # 训练传统ML模型
            train_traditional_ml_models(args, feature_file)
        elif args.use_nn:
            # 训练神经网络模型
            train_neural_network_model(args, feature_file)

        logger.info("=" * 80)
        logger.info("流水线执行完成！/ Pipeline execution completed!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"流水线执行失败 / Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


"""
============================================================================
使用示例 / Usage Examples
============================================================================

示例 1: 传统ML模型 + 组分特征（钛合金）
Example 1: Traditional ML with Composition Features (Titanium Alloys)
----------------------------------------------------------------------------
python -m src.pipelines.end_to_end_pipeline `
    --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" `
    --result_dir "output/results/Ti_alloys/Xue/tradition/" `
    --target_columns "UTS(MPa)" "El(%)" `
    --processing_cols "Solution Temperature(℃)" "Solution Time(h)" "Aging Temperature(℃)" "Aging Time(h)" "Thermo-Mechanical Treatment Temperature(℃)" "Deformation(%)" `
    --models xgboost sklearn_rf mlp lightgbm catboost `
    --use_composition_feature True `
    --embedding_type tradition `
    --cross_validate --num_folds 9 `
    --test_size 0.2 `
    --random_state 42 `
    --evaluate_after_train `
    --run_shap_analysis `
    --use_optuna `
    --n_trials 50

示例 2: 神经网络 + 嵌入特征（钢铁 - MatSciBERT）
Example 2: Neural Network with Embeddings (Steel - MatSciBERT)
----------------------------------------------------------------------------
python -m src.pipelines.end_to_end_pipeline `
    --data_file "Features/Steel/USTB_steel/all_features_withID/features_with_id.csv" `
    --result_dir "output/results/Steel/USTB_steel/matscibert/NN_opt" `
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
    --use_nn `
    --cross_validate `
    --num_folds 9 `
    --epochs 200 `
    --patience 200 `
    --batch_size 256 `
    --use_element_embedding True `
    --use_process_embedding True `
    --use_temperature False `
    --embedding_type matscibert `
    --evaluate_after_train `
    --use_optuna `
    --n_trials 50

示例 3: 传统ML模型 + SteelBERT嵌入（高熵合金）
Example 3: Traditional ML with SteelBERT Embeddings (HEA)
----------------------------------------------------------------------------
python -m src.pipelines.end_to_end_pipeline `
    --data_file "datasets/HEA_data/RoomTemperature_HEA_withID.csv" `
    --result_dir "output/results/HEA/yasir_data/steelbert/" `
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
    --processing_cols "Hom_Temp(K)" "CR(%)" "recrystalize temperature/K" "recrystalize time/mins" "Anneal_Temp(K)" "Anneal_Time(h)" "aging temperature/K" "aging time/hours" `
    --models xgboost lightgbm catboost `
    --use_element_embedding True `
    --use_process_embedding True `
    --embedding_type steelbert `
    --cross_validate --num_folds 9 `
    --evaluate_after_train `
    --use_optuna `
    --n_trials 30

示例 4: 神经网络 + SciBERT嵌入（铝合金）
Example 4: Neural Network with SciBERT Embeddings (Aluminum Alloys)
----------------------------------------------------------------------------
python -m src.pipelines.end_to_end_pipeline `
    --data_file "datasets/Al_Alloys/USTB/USTB_Al_alloys_processed_withID.csv" `
    --result_dir "output/results/Al_alloys/USTB/scibert/NN" `
    --target_columns "UTS(MPa)" `
    --processing_cols "ST1" "TIME1" "ST2" "TIME2" "ST3" "TIME3" "Cold_Deformation_percent" "First_Aging_Temp_C" "First_Aging_Time_h" "Second_Aging_Temp_C" "Second_Aging_Time_h" "Third_Aging_Temp_C" "Third_Aging_Time_h" `
    --use_nn `
    --use_element_embedding True `
    --use_process_embedding True `
    --embedding_type scibert `
    --cross_validate --num_folds 9 `
    --epochs 200 `
    --batch_size 128 `
    --evaluate_after_train

示例 5: 传统ML模型 + 组分特征 + 温度特征（铌合金）
Example 5: Traditional ML with Composition + Temperature Features (Niobium Alloys)
----------------------------------------------------------------------------
python -m src.pipelines.end_to_end_pipeline `
    --data_file "datasets/Nb_Alloys/Nb_cleandata/Nb_clean_with_processing_sequence_withID.csv" `
    --result_dir "output/results/Nb_alloys/Nb_cleandata/tradition/withTemp" `
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
    --processing_cols "Temperature((K))" "Anealing Temperature((K))" "Anealing times(h)" "Thermo-Mechanical Treatment Temperature((K))" "Deformation(%)" `
    --models xgboost sklearn_rf catboost `
    --use_composition_feature True `
    --use_temperature True `
    --embedding_type tradition `
    --cross_validate --num_folds 9 `
    --evaluate_after_train `
    --run_shap_analysis

============================================================================
参数说明 / Parameter Description
============================================================================

必需参数 / Required Parameters:
  --data_file           输入数据文件路径（原始数据或特征文件）
  --result_dir          结果输出目录
  --target_columns      目标预测列名（可多个）
  --embedding_type      嵌入类型: tradition/scibert/steelbert/matscibert

模型选择 / Model Selection (二选一 / Choose one):
  --models              传统ML模型列表: xgboost, sklearn_rf, mlp, lightgbm, catboost
  --use_nn              使用神经网络模型

特征配置 / Feature Configuration:
  --use_composition_feature    使用组分特征
  --use_element_embedding      使用元素嵌入
  --use_process_embedding      使用工艺嵌入
  --use_temperature            使用温度特征
  --processing_cols            处理参数列名

训练配置 / Training Configuration:
  --cross_validate      启用交叉验证
  --num_folds           交叉验证折数（默认9）
  --test_size           测试集比例（默认0.2）
  --random_state        随机种子（默认42）

神经网络参数 / Neural Network Parameters:
  --epochs              最大训练轮数（默认200）
  --patience            早停耐心值（默认200）
  --batch_size          训练批次大小（默认256）

优化配置 / Optimization Configuration:
  --use_optuna          启用Optuna超参数优化
  --n_trials            Optuna试验次数（默认50）

评估配置 / Evaluation Configuration:
  --evaluate_after_train    训练后评估
  --run_shap_analysis       运行SHAP分析（仅传统ML）

============================================================================
目录结构说明 / Directory Structure
============================================================================

输入特征目录结构 / Input Feature Structure:
Features/{alloy_type}/{dataset_name}/
├── tradition/           # Traditional composition features only
├── scibert/            # Features with SciBERT embeddings
├── steelbert/          # Features with SteelBERT embeddings
└── matscibert/         # Features with MatSciBERT embeddings
    ├── features_with_id.csv
    ├── target_with_id.csv
    └── feature_names.txt

输出结果目录结构 / Output Result Structure:
output/results/{alloy_type}/{dataset_name}/{embedding_type}/
├── tradition/          # Results with traditional features
├── scibert/           # Results with SciBERT features
├── steelbert/         # Results with SteelBERT features
└── matscibert/        # Results with MatSciBERT features
    ├── models/
    ├── predictions/
    ├── evaluations/
    └── logs/

支持的合金类型 / Supported Alloy Types:
- Ti_alloys (钛合金)
- Al_alloys (铝合金)
- Nb_alloys (铌合金)
- HEA (高熵合金)
- Steel (钢铁)

============================================================================
工作流程 / Workflow
============================================================================

1. 参数验证 / Parameter Validation
   - 检查参数有效性
   - 验证文件存在性
   - 验证参数组合

2. 特征生成 / Feature Generation (如需要 / If needed)
   - 检测输入是否为特征文件
   - 如果是原始数据，生成特征
   - 保存到标准化目录结构

3. 模型训练 / Model Training
   - 传统ML: 支持多模型对比、交叉验证、Optuna优化
   - 神经网络: 支持交叉验证、Optuna优化、早停

4. 模型评估 / Model Evaluation (如启用 / If enabled)
   - 计算评估指标（R², RMSE, MAE）
   - 生成预测图表
   - SHAP分析（传统ML）

============================================================================
注意事项 / Notes
============================================================================

1. 模型选择互斥：不能同时指定 --use_nn 和 --models
2. 嵌入类型非tradition时，必须指定至少一种嵌入特征
3. 特征文件自动检测：位于Features/目录或文件名包含'feature'
4. 结果目录自动创建：遵循标准化目录结构
5. 交叉验证推荐折数：9折（可根据数据量调整）
6. Optuna优化推荐试验次数：30-50次

============================================================================
"""

