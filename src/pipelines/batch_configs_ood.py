"""
Unified batch configuration source of truth for OOD runs.
"""

from __future__ import annotations

from typing import Any, Dict, List


ALLOY_CONFIGS_OOD: Dict[str, Dict[str, Any]] = {
    "Ti": {
        "raw_data": "datasets/Ti_alloys/titanium.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        "processing_cols": [
            "Solution Temperature(℃)", "Solution Time(h)",
            "Aging Temperature(℃)", "Aging Time(h)",
            "Thermo-Mechanical Treatment Temperature(℃)", "Deformation(%)",
        ],
        "processing_text_column": "Processing_Description",
        "description": "Titanium alloy OOD dataset",
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
        "description": "Aluminum alloy OOD dataset",
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
        "description": "HEA OOD dataset",
    },
    "Steel": {
        "raw_data": "datasets/Steel/steel.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "processing_cols": [],
        "processing_text_column": "Processing_Description",
        "description": "Steel OOD dataset",
    },
    # "HEA_corrosion": {
    #     "raw_data": "datasets/HEA_data/Pitting_potential_data_xiongjie.csv",
    #     "targets": ["Ep(mV)"],
    #     "processing_cols": ["Temperature", "Cl Concentration", "PH"],
    #     "processing_text_column": None,
    #     "description": "HEA corrosion OOD dataset",
    # },
}


COMMON_OOD_DEFAULTS: Dict[str, Any] = {
    "alloy_types": None,
    "exclude_alloys": ["HEA_corrosion"],
    "cross_validate": True,
    "num_folds": 9,
    "use_optuna": True,
    "n_trials": 30,
    "evaluate_after_train": True,
    "test_size": 0.2,
    "random_state": 42,
}


OOD_METHODS: Dict[str, Dict[str, Any]] = {
    "target_extrapolation": {
        "display_name": "target extrapolation",
        "config_suffix": "extrapolation",
        "result_dir_suffix": "target_extrapolation",
        "summary_file_name": "split_summary.json",
        "is_multi_fold": False,
        "default_params": {
            "split_strategy": "target_extrapolation",
            "extrapolation_side": "low_to_high",
        },
        "cli_args": ["split_strategy", "test_size", "extrapolation_side"],
    },
    "sparse_x_single": {
        "display_name": "sparse X single",
        "config_suffix": "sparse_x_single",
        "result_dir_suffix": "sparse_x_single",
        "summary_file_name": "split_summary.json",
        "is_multi_fold": False,
        "default_params": {
            "split_strategy": "sparse_x_single",
            "sparse_candidate_pool_size": 500,
            "sparse_cluster_count": 50,
            "sparse_samples_per_cluster": 1,
            "sparse_kde_bandwidth": None,
        },
        "cli_args": [
            "split_strategy",
            "test_size",
            "sparse_candidate_pool_size",
            "sparse_cluster_count",
            "sparse_samples_per_cluster",
            "sparse_kde_bandwidth",
        ],
    },
    "sparse_y_single": {
        "display_name": "sparse Y single",
        "config_suffix": "sparse_y_single",
        "result_dir_suffix": "sparse_y_single",
        "summary_file_name": "split_summary.json",
        "is_multi_fold": False,
        "default_params": {
            "split_strategy": "sparse_y_single",
            "sparse_candidate_pool_size": 500,
            "sparse_cluster_count": 50,
            "sparse_samples_per_cluster": 1,
            "sparse_kde_bandwidth": None,
        },
        "cli_args": [
            "split_strategy",
            "test_size",
            "sparse_candidate_pool_size",
            "sparse_cluster_count",
            "sparse_samples_per_cluster",
            "sparse_kde_bandwidth",
        ],
    },
    "sparse_x_cluster": {
        "display_name": "sparse X cluster",
        "config_suffix": "sparse_x_cluster",
        "result_dir_suffix": "sparse_x_cluster",
        "summary_file_name": "split_summary.json",
        "is_multi_fold": False,
        "default_params": {
            "split_strategy": "sparse_x_cluster",
            "sparse_candidate_pool_size": 500,
            "sparse_cluster_count": 50,
            "sparse_neighbors_per_seed": 5,
        },
        "cli_args": [
            "split_strategy",
            "test_size",
            "sparse_candidate_pool_size",
            "sparse_cluster_count",
            "sparse_neighbors_per_seed",
        ],
    },
    "sparse_y_cluster": {
        "display_name": "sparse Y cluster",
        "config_suffix": "sparse_y_cluster",
        "result_dir_suffix": "sparse_y_cluster",
        "summary_file_name": "split_summary.json",
        "is_multi_fold": False,
        "default_params": {
            "split_strategy": "sparse_y_cluster",
            "sparse_candidate_pool_size": 500,
            "sparse_cluster_count": 50,
            "sparse_neighbors_per_seed": 5,
        },
        "cli_args": [
            "split_strategy",
            "test_size",
            "sparse_candidate_pool_size",
            "sparse_cluster_count",
            "sparse_neighbors_per_seed",
        ],
    },
    "loco": {
        "display_name": "LOCO",
        "config_suffix": "loco",
        "result_dir_suffix": "loco",
        "summary_file_name": "loco_manifest.json",
        "is_multi_fold": True,
        "default_params": {
            "split_strategy": "loco",
            "loco_cluster_count": 5,
        },
        "cli_args": ["split_strategy", "test_size", "loco_cluster_count"],
    },
}


EXPERIMENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "experiment1_all_ml_models": {
        "description_prefix": "Alloy OOD experiment with traditional ML models",
        "embedding_type": "tradition",
        "use_composition_feature": True,
        "use_element_embedding": False,
        "use_process_embedding": False,
        "use_temperature": False,
        "models": ["xgboost", "sklearn_rf", "lightgbm", "mlp", "catboost"],
        "use_nn": False,
        "mlp_max_iter": 300,
        "run_shap_analysis": True,
    },
    "experiment2a_all_nn_scibert": {
        "description_prefix": "Alloy OOD experiment with NN + SciBERT",
        "embedding_type": "scibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "run_shap_analysis": False,
    },
    "experiment2b_all_nn_steelbert": {
        "description_prefix": "Alloy OOD experiment with NN + SteelBERT",
        "embedding_type": "steelbert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "run_shap_analysis": False,
    },
    "experiment2c_all_nn_matscibert": {
        "description_prefix": "Alloy OOD experiment with NN + MatSciBERT",
        "embedding_type": "matscibert",
        "use_composition_feature": False,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,
        "use_nn": True,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "run_shap_analysis": False,
    },
}


def get_alloy_config_ood(alloy_type: str) -> Dict[str, Any]:
    if alloy_type not in ALLOY_CONFIGS_OOD:
        raise ValueError(
            f"Unsupported alloy type for OOD: {alloy_type}. "
            f"Supported types: {', '.join(ALLOY_CONFIGS_OOD.keys())}"
        )
    return ALLOY_CONFIGS_OOD[alloy_type].copy()


def list_available_alloys_ood() -> List[str]:
    return list(ALLOY_CONFIGS_OOD.keys())


def get_ood_method_meta(method_name: str) -> Dict[str, Any]:
    if method_name not in OOD_METHODS:
        raise ValueError(
            f"Unsupported OOD method: {method_name}. "
            f"Supported methods: {', '.join(OOD_METHODS.keys())}"
        )
    return OOD_METHODS[method_name].copy()


def list_available_ood_methods() -> List[str]:
    return list(OOD_METHODS.keys())


def _build_batch_configs() -> Dict[str, Dict[str, Any]]:
    batch_configs: Dict[str, Dict[str, Any]] = {}
    for template_name, template in EXPERIMENT_TEMPLATES.items():
        for method_name, method_meta in OOD_METHODS.items():
            config_name = f"{template_name}_{method_meta['config_suffix']}"
            batch_configs[config_name] = {
                **COMMON_OOD_DEFAULTS,
                **method_meta["default_params"],
                **template,
                "ood_method": method_name,
                "is_multi_fold": method_meta["is_multi_fold"],
                "summary_file_name": method_meta["summary_file_name"],
                "result_dir_suffix": method_meta["result_dir_suffix"],
                "description": f"{template['description_prefix']} ({method_meta['display_name']})",
            }
    return batch_configs


BATCH_CONFIGS_OOD: Dict[str, Dict[str, Any]] = _build_batch_configs()
