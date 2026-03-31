"""
Backend-aware configuration for TabPFN target extrapolation experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from .tabpfn_configs import get_all_alloy_types, get_tabpfn_config, get_tabpfn_configs
except ImportError:  # pragma: no cover
    from model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from tabpfn_configs import get_all_alloy_types, get_tabpfn_config, get_tabpfn_configs


DEFAULT_EXTRAPOLATION_SETTINGS: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "extrapolation_side": "low_to_high",
}


TABPFN_EXTRAPOLATION_CONFIGS_LOCAL_NUMERIC: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_EXTRAPOLATION_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="local",
        feature_mode="numeric",
    ).items()
}


TABPFN_EXTRAPOLATION_CONFIGS_API_TEXT: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_EXTRAPOLATION_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="api",
        feature_mode="text",
    ).items()
}


TABPFN_EXTRAPOLATION_CONFIGS_API_NUMERIC: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_EXTRAPOLATION_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="api",
        feature_mode="numeric",
    ).items()
}


TABPFN_EXTRAPOLATION_CONFIGS_BY_BACKEND: Dict[str, Dict[str, Dict[str, Any]]] = {
    "local:numeric": TABPFN_EXTRAPOLATION_CONFIGS_LOCAL_NUMERIC,
    "api:text": TABPFN_EXTRAPOLATION_CONFIGS_API_TEXT,
    "api:numeric": TABPFN_EXTRAPOLATION_CONFIGS_API_NUMERIC,
}

# Backward-compatible default for legacy imports.
TABPFN_EXTRAPOLATION_CONFIGS = TABPFN_EXTRAPOLATION_CONFIGS_LOCAL_NUMERIC


TABPFN_EXTRAPOLATION_BATCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tabpfn_all_extrapolation": {
        "description": "Run low-to-high TabPFN extrapolation for all supported alloys and targets.",
        "alloy_types": None,
        "exclude_alloys": [],
        "test_size": 0.2,
        "random_state": 42,
        "extrapolation_side": "low_to_high",
        "output_root": None,
        "align_predictions": True,
    }
}


def get_tabpfn_extrapolation_config(
    alloy_type: str,
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> Dict[str, Any]:
    available_alloys = get_all_alloy_types(backend=backend, base_path=base_path)
    if alloy_type not in available_alloys:
        raise ValueError(
            f"Unknown alloy type: {alloy_type}. "
            f"Available types: {available_alloys}"
        )
    return {
        **get_tabpfn_config(
            alloy_type,
            backend=backend,
            feature_mode=feature_mode,
            base_path=base_path,
        ),
        **DEFAULT_EXTRAPOLATION_SETTINGS,
    }


def get_tabpfn_extrapolation_configs(
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> Dict[str, Dict[str, Any]]:
    config_backend = resolve_tabpfn_backend(base_path=base_path, backend=backend)
    config_feature_mode = resolve_tabpfn_feature_mode(
        base_path=base_path,
        backend=config_backend,
        feature_mode=feature_mode,
    )
    return TABPFN_EXTRAPOLATION_CONFIGS_BY_BACKEND[f"{config_backend}:{config_feature_mode}"]


def get_all_tabpfn_extrapolation_alloys(
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> List[str]:
    get_tabpfn_extrapolation_configs(
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    return get_all_alloy_types(backend=backend, base_path=base_path)
