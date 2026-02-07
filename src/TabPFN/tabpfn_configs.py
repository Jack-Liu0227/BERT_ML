"""
TabPFN Model Configuration
基于 batch_configs.py 的配置文件，用于 TabPFN 模型测试

Configuration file for TabPFN model testing based on batch_configs.py
"""

from typing import Dict, Any, List

# ============================================================================
# TabPFN 数据集配置 / TabPFN Dataset Configuration
# ============================================================================

TABPFN_CONFIGS = {
    # 钛合金 / Titanium Alloys
    "Ti": {
        "raw_data": "datasets/Ti_alloys/titanium.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        # Optional: align exported TabPFN prediction CSV row order to an existing reference CSV by ID
        # so that different models' `all_predictions.csv` can be compared row-by-row.
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Ti/titanium/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "feature_cols": [
            # 元素列 / Element columns
            "Al(wt%)", "Cr(wt%)", "Fe(wt%)", "Mo(wt%)", "Nb(wt%)", 
            "Sn(wt%)", "Ti(wt%)", "V(wt%)", "Zr(wt%)",
            # 工艺参数 / Processing parameters
            "Solution Temperature(℃)", "Solution Time(h)",
            "Aging Temperature(℃)", "Aging Time(h)",
            "Thermo-Mechanical Treatment Temperature(℃)", "Deformation(%)"
        ],
        "description": "钛合金力学性能数据集 / Titanium alloy mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42
    },

    # 铝合金 / Aluminum Alloys
    "Al": {
        "raw_data": "datasets/Al_Alloys/aluminum.csv",
        "targets": ["UTS(MPa)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Al/aluminum/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "feature_cols": [
            # 元素列 / Element columns
            "Ag(wt%)", "Al(wt%)", "Be(wt%)", "Ce(wt%)", "Cr(wt%)", "Cu(wt%)", 
            "Fe(wt%)", "Li(wt%)", "Mg(wt%)", "Mn(wt%)", "Ni(wt%)", "Re(wt%)", 
            "Si(wt%)", "Sn(wt%)", "Ti(wt%)", "V(wt%)", "Zn(wt%)", "Zr(wt%)",
            # 工艺参数 / Processing parameters
            "ST1", "TIME1", "ST2", "TIME2", "ST3", "TIME3", 
            "Cold_Deformation_percent", 
            "First_Aging_Temp_C", "First_Aging_Time_h", 
            "Second_Aging_Temp_C", "Second_Aging_Time_h", 
            "Third_Aging_Temp_C", "Third_Aging_Time_h"
        ],
        "description": "铝合金力学性能数据集 / Aluminum alloy mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42
    },

    # 高熵合金 / High Entropy Alloys
    "HEA": {
        "raw_data": "datasets/HEA_data/hea.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/HEA_half/hea/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "feature_cols": [
            # 元素列 / Element columns (at%)
            "Al(at%)", "C(at%)", "Co(at%)", "Cr(at%)", "Cu(at%)", 
            "Fe(at%)", "Mn(at%)", "Mo(at%)", "Nb(at%)", "Ni(at%)", 
            "Ta(at%)", "Ti(at%)", "V(at%)", "W(at%)",
            # 工艺参数 / Processing parameters
            "Hom_Temp(K)", "CR(%)",
            "recrystalize temperature/K", "recrystalize time/mins",
            "Anneal_Temp(K)", "Anneal_Time(h)",
            "aging temperature/K", "aging time/hours"
        ],
        "description": "高熵合金室温力学性能数据集 / HEA room temperature mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42
    },

    # 钢铁 / Steel
    "Steel": {
        "raw_data": "datasets/Steel/steel.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Steel/steel/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "feature_cols": [
            # 元素列 / Element columns
            "Al(wt%)", "As(wt%)", "B(wt%)", "Bi(wt%)", "C(wt%)", "Ca(wt%)", 
            "Ce(wt%)", "Cl(wt%)", "Co(wt%)", "Cr(wt%)", "Cu(wt%)", "F(wt%)", 
            "Fe(wt%)", "H(wt%)", "La(wt%)", "Mg(wt%)", "Mn(wt%)", "Mo(wt%)", 
            "N(wt%)", "Na(wt%)", "Nb(wt%)", "Ni(wt%)", "O(wt%)", "P(wt%)", 
            "Pb(wt%)", "S(wt%)", "Sb(wt%)", "Si(wt%)", "Sn(wt%)", "Ta(wt%)", 
            "Ti(wt%)", "V(wt%)", "W(wt%)", "Y(wt%)", "Zn(wt%)", "Zr(wt%)"
            # 注意：Steel数据集的工艺参数在 Processing_Description 中，可能需要单独提取
        ],
        "description": "钢铁力学性能数据集 / Steel mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42
    },
}


# ============================================================================
# TabPFN 模型配置 / TabPFN Model Configuration
# ============================================================================

TABPFN_MODEL_CONFIG = {
    # 模型版本 / Model version
    # 使用 v2 版本（开放模型，无需认证）
    # Using v2 (open model, no authentication required)
    # v2.5 requires HuggingFace authentication
    "model_version": "v2",
    
    # 任务类型 / Task type
    # TabPFN 支持分类和回归任务
    # TabPFN supports both classification and regression tasks
    "task_type": "regression",  # "classification" or "regression"
    
    # 评估指标 / Evaluation metrics
    "metrics": {
        "regression": ["mae", "rmse", "r2", "mape"],
        "classification": ["accuracy", "roc_auc", "f1"]
    }
}


def get_tabpfn_config(alloy_type: str) -> Dict[str, Any]:
    """
    获取指定合金类型的配置
    Get configuration for specified alloy type
    
    Args:
        alloy_type: 合金类型 ("Ti", "Al", "HEA", "Steel")
        
    Returns:
        配置字典 / Configuration dictionary
    """
    if alloy_type not in TABPFN_CONFIGS:
        raise ValueError(f"Unknown alloy type: {alloy_type}. Available types: {list(TABPFN_CONFIGS.keys())}")
    
    return TABPFN_CONFIGS[alloy_type]


def get_all_alloy_types() -> List[str]:
    """获取所有可用的合金类型 / Get all available alloy types"""
    return list(TABPFN_CONFIGS.keys())
