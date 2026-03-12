"""
Independent batch configuration for low-strength train / high-strength extrapolation runs.
"""

from __future__ import annotations

from typing import Any, Dict, List


ALLOY_CONFIGS_EXTRAPOLATION: Dict[str, Dict[str, Any]] = {
    "Ti": {
        "raw_data": "datasets/Ti_alloys/titanium.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        "processing_cols": [
            "Solution Temperature(℃)", "Solution Time(h)",
            "Aging Temperature(℃)", "Aging Time(h)",
            "Thermo-Mechanical Treatment Temperature(℃)", "Deformation(%)",
        ],
        "processing_text_column": "Processing_Description",
        "description": "Titanium alloy extrapolation dataset",
    },
    "Al": {
        "raw_data": "datasets/Al_Alloys/aluminum.csv",
        "targets": ["UTS(MPa)"],
        "processing_cols": [
            "ST1", "TIME1", "ST2", "TIME2", "ST3", "TIME3",
            "Cold_Deformation_percent",
            "First_Aging_Temp_C", "First_Aging_Time_h",
            "Second_Aging_Temp_C", "Second_Aging_Time_h",
            "Third_Aging_Temp_C", "Third_Aging_Time_h",
        ],
        "processing_text_column": "Processing_Description",
        "description": "Aluminum alloy extrapolation dataset",
    },
    "HEA_half": {
        "raw_data": "datasets/HEA_data/hea.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "processing_cols": [
            "Hom_Temp(K)", "CR(%)",
            "recrystalize temperature/K", "recrystalize time/mins",
            "Anneal_Temp(K)", "Anneal_Time(h)",
            "aging temperature/K", "aging time/hours",
        ],
        "processing_text_column": "Processing_Description",
        "description": "HEA extrapolation dataset",
    },
    "Steel": {
        "raw_data": "datasets/Steel/steel.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "processing_cols": [],
        "processing_text_column": "Processing_Description",
        "description": "Steel extrapolation dataset",
    },
    "HEA_corrosion": {
        "raw_data": "datasets/HEA_data/Pitting_potential_data_xiongjie.csv",
        "targets": ["Ep(mV)"],
        "processing_cols": ["Temperature", "Cl Concentration", "PH"],
        "processing_text_column": None,
        "description": "HEA corrosion extrapolation dataset",
    },
}


def get_alloy_config_extrapolation(alloy_type: str) -> Dict[str, Any]:
    if alloy_type not in ALLOY_CONFIGS_EXTRAPOLATION:
        raise ValueError(
            f"Unsupported alloy type for extrapolation: {alloy_type}. "
            f"Supported types: {', '.join(ALLOY_CONFIGS_EXTRAPOLATION.keys())}"
        )
    return ALLOY_CONFIGS_EXTRAPOLATION[alloy_type].copy()


def list_available_alloys_extrapolation() -> List[str]:
    return list(ALLOY_CONFIGS_EXTRAPOLATION.keys())


BATCH_CONFIGS_EXTRAPOLATION: Dict[str, Dict[str, Any]] = {
    "experiment1_all_ml_models_extrapolation": {
        "description": "Alloy extrapolation experiment with traditional ML models",
        "alloy_types": None,
        "exclude_alloys": ["HEA_corrosion"],
        "embedding_type": "tradition",
        "use_composition_feature": True,
        "use_element_embedding": False,
        "use_process_embedding": False,
        "use_temperature": False,
        "models": ["xgboost", "sklearn_rf", "lightgbm", "mlp", "catboost"],
        "use_nn": False,
        "cross_validate": True,
        "num_folds": 9,
        "use_optuna": True,
        "n_trials": 30,
        "mlp_max_iter": 300,
        "evaluate_after_train": True,
        "run_shap_analysis": True,
        "test_size": 0.2,
        "split_strategy": "target_extrapolation",
        "extrapolation_side": "low_to_high",
    },
    "experiment2a_all_nn_scibert_extrapolation": {
        "description": "Alloy extrapolation experiment with NN + SciBERT",
        "alloy_types": None,
        "exclude_alloys": ["HEA_corrosion"],
        "embedding_type": "scibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "evaluate_after_train": True,
        "run_shap_analysis": False,
        "test_size": 0.2,
        "split_strategy": "target_extrapolation",
        "extrapolation_side": "low_to_high",
    },
    "experiment2b_all_nn_steelbert_extrapolation": {
        "description": "Alloy extrapolation experiment with NN + SteelBERT",
        "alloy_types": None,
        "exclude_alloys": ["HEA_corrosion"],
        "embedding_type": "steelbert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "evaluate_after_train": True,
        "run_shap_analysis": False,
        "test_size": 0.2,
        "split_strategy": "target_extrapolation",
        "extrapolation_side": "low_to_high",
    },
    "experiment2c_all_nn_matscibert_extrapolation": {
        "description": "Alloy extrapolation experiment with NN + MatSciBERT",
        "alloy_types": None,
        "exclude_alloys": ["HEA_corrosion"],
        "embedding_type": "matscibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "evaluate_after_train": True,
        "run_shap_analysis": False,
        "test_size": 0.2,
        "split_strategy": "target_extrapolation",
        "extrapolation_side": "low_to_high",
    },
}
