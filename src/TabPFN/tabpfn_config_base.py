"""
Shared TabPFN dataset configuration building blocks.

This module keeps all alloy metadata and feature definitions in one place so
backend-specific and feature-mode-specific config modules can stay thin and
consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List


PROCESSING_TEXT_COLUMN = "Processing_Description"
MATBENCH_STEELS_TEXT_COLUMN = "composition"


COMMON_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Ti": {
        "raw_data": "datasets/Ti_alloys/titanium.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Ti/titanium/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "description": "钛合金力学性能数据集 / Titanium alloy mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42,
    },
    "Al": {
        "raw_data": "datasets/Al_Alloys/aluminum.csv",
        "targets": ["UTS(MPa)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Al/aluminum/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "description": "铝合金力学性能数据集 / Aluminum alloy mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42,
    },
    "HEA": {
        "raw_data": "datasets/HEA_data/hea.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/HEA_half/hea/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "description": "高熵合金室温力学性能数据集 / HEA room temperature mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42,
    },
    "Steel": {
        "raw_data": "datasets/Steel/steel.csv",
        "targets": ["YS(MPa)", "UTS(MPa)", "El(%)"],
        "align_reference_predictions_csv": "output/new_results_withuncertainty/Steel/steel/tradition/model_comparison/catboost_results/predictions/all_predictions.csv",
        "description": "钢铁力学性能数据集 / Steel mechanical properties dataset",
        "test_size": 0.2,
        "random_state": 42,
    },
    "MatbenchSteels": {
        "raw_data": "datasets/matbench_convert/matbench_steels_ood.csv",
        "targets": ["yield strength"],
        "description": "Matbench steels yield-strength dataset",
        "test_size": 0.2,
        "random_state": 42,
        "source_data": "datasets/matbench_convert/matbench_steels.csv",
        "text_feature_cols": [MATBENCH_STEELS_TEXT_COLUMN],
    },
}


ELEMENT_FEATURE_COLS: Dict[str, List[str]] = {
    "Ti": [
        "Al(wt%)",
        "Cr(wt%)",
        "Fe(wt%)",
        "Mo(wt%)",
        "Nb(wt%)",
        "Sn(wt%)",
        "Ti(wt%)",
        "V(wt%)",
        "Zr(wt%)",
    ],
    "Al": [
        "Ag(wt%)",
        "Al(wt%)",
        "Be(wt%)",
        "Ce(wt%)",
        "Cr(wt%)",
        "Cu(wt%)",
        "Fe(wt%)",
        "Li(wt%)",
        "Mg(wt%)",
        "Mn(wt%)",
        "Ni(wt%)",
        "Re(wt%)",
        "Si(wt%)",
        "Sn(wt%)",
        "Ti(wt%)",
        "V(wt%)",
        "Zn(wt%)",
        "Zr(wt%)",
    ],
    "HEA": [
        "Al(at%)",
        "C(at%)",
        "Co(at%)",
        "Cr(at%)",
        "Cu(at%)",
        "Fe(at%)",
        "Mn(at%)",
        "Mo(at%)",
        "Nb(at%)",
        "Ni(at%)",
        "Ta(at%)",
        "Ti(at%)",
        "V(at%)",
        "W(at%)",
    ],
    "Steel": [
        "Al(wt%)",
        "As(wt%)",
        "B(wt%)",
        "Bi(wt%)",
        "C(wt%)",
        "Ca(wt%)",
        "Ce(wt%)",
        "Cl(wt%)",
        "Co(wt%)",
        "Cr(wt%)",
        "Cu(wt%)",
        "F(wt%)",
        "Fe(wt%)",
        "H(wt%)",
        "La(wt%)",
        "Mg(wt%)",
        "Mn(wt%)",
        "Mo(wt%)",
        "N(wt%)",
        "Na(wt%)",
        "Nb(wt%)",
        "Ni(wt%)",
        "O(wt%)",
        "P(wt%)",
        "Pb(wt%)",
        "S(wt%)",
        "Sb(wt%)",
        "Si(wt%)",
        "Sn(wt%)",
        "Ta(wt%)",
        "Ti(wt%)",
        "V(wt%)",
        "W(wt%)",
        "Y(wt%)",
        "Zn(wt%)",
        "Zr(wt%)",
    ],
    "MatbenchSteels": [
        "Al(at%)",
        "C(at%)",
        "Co(at%)",
        "Cr(at%)",
        "Fe(at%)",
        "Mn(at%)",
        "Mo(at%)",
        "N(at%)",
        "Nb(at%)",
        "Ni(at%)",
        "Si(at%)",
        "Ti(at%)",
        "V(at%)",
        "W(at%)",
    ],
}


LOCAL_PROCESS_FEATURE_COLS: Dict[str, List[str]] = {
    "Ti": [
        "Solution Temperature(\u2103)",
        "Solution Time(h)",
        "Aging Temperature(\u2103)",
        "Aging Time(h)",
        "Thermo-Mechanical Treatment Temperature(\u2103)",
        "Deformation(%)",
    ],
    "Al": [
        "ST1",
        "TIME1",
        "ST2",
        "TIME2",
        "ST3",
        "TIME3",
        "Cold_Deformation_percent",
        "First_Aging_Temp_C",
        "First_Aging_Time_h",
        "Second_Aging_Temp_C",
        "Second_Aging_Time_h",
        "Third_Aging_Temp_C",
        "Third_Aging_Time_h",
    ],
    "HEA": [
        "Hom_Temp(K)",
        "CR(%)",
        "recrystalize temperature/K",
        "recrystalize time/mins",
        "Anneal_Temp(K)",
        "Anneal_Time(h)",
        "aging temperature/K",
        "aging time/hours",
    ],
    "Steel": [],
    "MatbenchSteels": [],
}


def build_tabpfn_configs(feature_mode: str) -> Dict[str, Dict[str, Any]]:
    if feature_mode not in {"numeric", "text"}:
        raise ValueError(f"Unsupported TabPFN feature mode: {feature_mode}")

    configs: Dict[str, Dict[str, Any]] = {}
    for alloy_type, shared_config in COMMON_DATASET_CONFIGS.items():
        element_feature_cols = list(ELEMENT_FEATURE_COLS[alloy_type])
        numeric_process_feature_cols = list(LOCAL_PROCESS_FEATURE_COLS.get(alloy_type, []))

        if feature_mode == "text":
            text_feature_cols = list(shared_config.get("text_feature_cols", [PROCESSING_TEXT_COLUMN]))
            processing_feature_cols = text_feature_cols
        else:
            processing_feature_cols = numeric_process_feature_cols
            text_feature_cols = []

        feature_cols = element_feature_cols + processing_feature_cols
        if feature_mode == "numeric" and PROCESSING_TEXT_COLUMN in feature_cols:
            raise ValueError(
                f"Numeric TabPFN config for {alloy_type} must not include {PROCESSING_TEXT_COLUMN}."
            )
        if feature_mode == "text" and any(
            col in numeric_process_feature_cols for col in feature_cols
        ):
            raise ValueError(
                f"Text TabPFN config for {alloy_type} must not include numeric processing columns."
            )

        config = {
            **shared_config,
            "feature_cols": feature_cols,
            "element_feature_cols": element_feature_cols,
            "processing_feature_cols": processing_feature_cols,
            "numeric_process_feature_cols": numeric_process_feature_cols,
            "text_feature_cols": text_feature_cols,
        }

        config["feature_mode"] = feature_mode

        configs[alloy_type] = config

    return configs
