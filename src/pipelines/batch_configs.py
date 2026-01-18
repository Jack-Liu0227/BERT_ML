"""
# 批量运行配置文件
Batch run configuration file

# 包含两类配置：
1. ALLOY_CONFIGS: 合金数据配置（数据文件路径、目标列、工艺参数等）
2. BATCH_CONFIGS: 批量运行任务配置（实验参数、模型选择等）

Contains two types of configurations:
1. ALLOY_CONFIGS: Alloy data configuration (data file paths, target columns, processing parameters, etc.)
2. BATCH_CONFIGS: Batch run task configuration (experiment parameters, model selection, etc.)
"""

from typing import Dict, Any, List

# ============================================================================
# 合金数据配置 / Alloy Data Configuration
# ============================================================================
# 此配置字典集中管理所有合金类型的数据文件路径和目标列名
# This configuration dictionary centrally manages data file paths and target columns for all alloy types
#
# 配置项说明 / Configuration Items:
#   - raw_data: 原始数据文件路径 / Raw data file path
#   - targets: 目标预测列名列表 / Target prediction column names
#   - processing_cols: 工艺参数数值列名列表，用于传统ML模型作为特征
#                     Processing parameter numerical columns for traditional ML models
#   - nn_additional_features: 神经网络模型可选择使用的额外数值特征列名列表
#                            Optional additional numerical features for neural network models
#   - processing_text_column: 工艺描述文本列名，用于BERT模型生成嵌入向量
#                            Processing description text column for BERT embedding generation
#   - description: 数据集描述 / Dataset description
#
# 使用方法 / Usage:
#   config = get_alloy_config("Ti")
#   data_file = config["raw_data"]
#   targets = config["targets"]
#   processing_cols = config["processing_cols"]  # 用于ML模型 / For ML models
#   text_col = config["processing_text_column"]  # 用于BERT嵌入 / For BERT embeddings
# ============================================================================

ALLOY_CONFIGS = {
    # 钛合金 / Titanium Alloys
    # "Ti": {
    #     "raw_data": "datasets/Ti_alloys/titanium.csv",
    #     "targets": ["UTS(MPa)", "El(%)"],
    #     "processing_cols": [
    #         "Solution Temperature(℃)", "Solution Time(h)",
    #         "Aging Temperature(℃)", "Aging Time(h)",
    #         "Thermo-Mechanical Treatment Temperature(℃)", "Deformation(%)"
    #     ],
    #     "processing_text_column": "Processing_Description",
    #     "description": "钛合金力学性能数据集 / Titanium alloy mechanical properties dataset"
    # },

    # # # 铝合金 / Aluminum Alloys
    # "Al": {
    #     "raw_data": "datasets/Al_Alloys/aluminum.csv",
    #     "targets": ["UTS(MPa)"],
    #     "processing_cols": [
    #         "ST1", "TIME1", "ST2", "TIME2", "ST3", "TIME3", 
    #         "Cold_Deformation_percent", 
    #         "First_Aging_Temp_C", "First_Aging_Time_h", 
    #         "Second_Aging_Temp_C", "Second_Aging_Time_h", 
    #         "Third_Aging_Temp_C", "Third_Aging_Time_h"
    #     ],
    #     "processing_text_column": "Processing_Description",
    #     "description": "铝合金力学性能数据集 / Aluminum alloy mechanical properties dataset"
    # },

    # "HEA_half": {
    #     "raw_data": "datasets/HEA_data/hea.csv",
    #     "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
    #     "processing_cols": [
    #         "Hom_Temp(K)", "CR(%)",
    #         "recrystalize temperature/K", "recrystalize time/mins",
    #         "Anneal_Temp(K)", "Anneal_Time(h)",
    #         "aging temperature/K", "aging time/hours"
    #     ],
    #     "processing_text_column": "Processing_Description",
    #     "description": "高熵合金室温力学性能数据集的一半 / HEA room temperature mechanical properties dataset (half)"
    # },


    # # 钢铁 / Steel
    # "Steel": {
    #     "raw_data": "datasets/Steel/steel.csv",
    #     "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
    #     "processing_cols": [],  # Steel数据集的工艺参数列需要根据具体数据确定
    #     "processing_text_column": "Processing_Description",
    #     "description": "钢铁力学性能数据集 / Steel mechanical properties dataset"
    # },

# 高熵合金腐蚀 / HEA Corrosion
    "HEA_corrosion": {
        "raw_data": "datasets/HEA_data/Pitting_potential_data_xiongjie.csv",
        "targets": ["Ep(mV)"],
        "processing_cols": [
            "Temperature", "Cl Concentration", "PH"
        ],
        "processing_text_column": None,  # 该数据集无文本描述列
        "description": "高熵合金点蚀电位数据集 / HEA pitting potential dataset"
    }
}


def get_alloy_config(alloy_type: str) -> Dict[str, Any]:
    """
# 获取指定合金类型的配置
    Get configuration for specified alloy type

    Args:
        alloy_type: 合金类型简称 / Alloy type abbreviation
# 支持的类型 / Supported types: "Ti", "Al", "HEA_full", "HEA_half", "Nb", "Steel"

    Returns:
        Dict[str, Any]: 包含数据文件路径、目标列名等配置信息
                       Configuration dict containing data file path, target columns, etc.

    Raises:
        ValueError: 如果合金类型不支持 / If alloy type is not supported

    Example:
        >>> config = get_alloy_config("Ti")
        >>> print(config["raw_data"])
        datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv
        >>> print(config["targets"])
        ['UTS(MPa)', 'El(%)']
    """
    if alloy_type not in ALLOY_CONFIGS:
        supported_types = ", ".join(ALLOY_CONFIGS.keys())
        raise ValueError(
            f"不支持的合金类型: {alloy_type} / Unsupported alloy type: {alloy_type}\n"
            f"支持的类型 / Supported types: {supported_types}"
        )

    return ALLOY_CONFIGS[alloy_type].copy()


def list_available_alloys() -> List[str]:
    """
# 列出所有可用的合金类型
    List all available alloy types

    Returns:
        List[str]: 合金类型列表 / List of alloy types
    """
    return list(ALLOY_CONFIGS.keys())


# ============================================================================
# 批量运行任务配置 / Batch Run Task Configurations
# ============================================================================
BATCH_CONFIGS = {
# ========================================================================
# 完整模型对比实验配置
    # Complete Model Comparison Experiment Configurations
# ========================================================================

# 实验1: 所有合金 + 5个传统ML模型对比（tradition嵌入）
    "experiment1_all_ml_models": {
        "description": "实验1：所有合金 + 5个传统ML模型对比（XGBoost, RF, MLP, LightGBM, CatBoost）",
        "alloy_types": None,  # None表示所有合金（Ti, Al, HEA_full, HEA_half, Nb, Steel）
        "exclude_alloys": [],
        "embedding_type": "tradition",
        "use_composition_feature": True,
        "use_element_embedding": False,
        "use_process_embedding": False,
        "use_temperature": False,
        "models":  ["xgboost", "sklearn_rf","lightgbm", "mlp","catboost"],# ["catboost"],  # 5个模型 ,
        "use_nn": False,
        "cross_validate": True,
        "num_folds": 9,
        "use_optuna": True,
        "n_trials": 30,
        "n_repeats": 10,
        "mlp_max_iter": 300,
        "evaluate_after_train": True,
        "run_shap_analysis": True,
    },

    # 实验2a: 所有合金 + 神经网络 + SciBERT嵌入
    "experiment2a_all_nn_scibert": {
        "description": "【实验2a】所有合金 × 神经网络 + SciBERT嵌入",
        "alloy_types": None,
        "exclude_alloys": [],
        "embedding_type": "scibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "processing_cols": [],  # 神经网络不需要处理工艺参数列，只需要BERT嵌入
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "n_repeats": 10,
        "evaluate_after_train": True,
        "run_shap_analysis": True,
    },

    # 实验2b: 所有合金 + 神经网络 + SteelBERT嵌入
    "experiment2b_all_nn_steelbert": {
        "description": "【实验2b】所有合金 × 神经网络 + SteelBERT嵌入",
        "alloy_types": None,
        "exclude_alloys": [],
        "embedding_type": "steelbert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "processing_cols": [],  # 神经网络不需要处理工艺参数列，只需要BERT嵌入
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "n_repeats": 10,
        "evaluate_after_train": True,
        "run_shap_analysis": True,
    },

    # 实验2c: 所有合金 + 神经网络 + MatSciBERT嵌入
    "experiment2c_all_nn_matscibert": {
        "description": "【实验2c】所有合金 × 神经网络 + MatSciBERT嵌入",
        "alloy_types": None,
        "exclude_alloys": [],
        "embedding_type": "matscibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "processing_cols": [],  # 神经网络不需要处理工艺参数列，只需要BERT嵌入
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "n_repeats": 10,
        "evaluate_after_train": True,
        "run_shap_analysis": True,
    },
}
