"""
Independent configuration for TabPFN target extrapolation experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tabpfn_configs import TABPFN_CONFIGS


DEFAULT_EXTRAPOLATION_SETTINGS: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "extrapolation_side": "low_to_high",
}


TABPFN_EXTRAPOLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_EXTRAPOLATION_SETTINGS,
    }
    for alloy_type, config in TABPFN_CONFIGS.items()
}


TABPFN_EXTRAPOLATION_BATCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tabpfn_all_extrapolation": {
        "description": "Run low-to-high TabPFN extrapolation for all supported alloys and targets.",
        "alloy_types": None,
        "exclude_alloys": [],
        "test_size": 0.2,
        "random_state": 42,
        "extrapolation_side": "low_to_high",
        "output_root": "output/extrapolation_results_tabpfn",
        "align_predictions": True,
    }
}


def get_tabpfn_extrapolation_config(alloy_type: str) -> Dict[str, Any]:
    if alloy_type not in TABPFN_EXTRAPOLATION_CONFIGS:
        raise ValueError(
            f"Unknown alloy type: {alloy_type}. "
            f"Available types: {list(TABPFN_EXTRAPOLATION_CONFIGS.keys())}"
        )
    return TABPFN_EXTRAPOLATION_CONFIGS[alloy_type].copy()


def get_all_tabpfn_extrapolation_alloys() -> List[str]:
    return list(TABPFN_EXTRAPOLATION_CONFIGS.keys())

