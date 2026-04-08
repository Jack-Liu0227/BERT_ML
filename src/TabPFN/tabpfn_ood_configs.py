"""
Backend-aware configuration for TabPFN OOD experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from .tabpfn_configs import get_all_alloy_types, get_tabpfn_config, get_tabpfn_configs
except ImportError:  # pragma: no cover
    from model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from tabpfn_configs import get_all_alloy_types, get_tabpfn_config, get_tabpfn_configs


DEFAULT_OOD_SETTINGS: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "split_strategy": "target_extrapolation",
    "extrapolation_side": "low_to_high",
    "sparse_candidate_pool_size": 500,
    "sparse_cluster_count": 50,
    "sparse_samples_per_cluster": 1,
    "sparse_kde_bandwidth": None,
    "sparse_neighbors_per_seed": 5,
    "loco_cluster_count": 5,
}


TABPFN_OOD_CONFIGS_LOCAL_NUMERIC: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_OOD_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="local",
        feature_mode="numeric",
    ).items()
}


TABPFN_OOD_CONFIGS_API_TEXT: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_OOD_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="api",
        feature_mode="text",
    ).items()
}


TABPFN_OOD_CONFIGS_API_NUMERIC: Dict[str, Dict[str, Any]] = {
    alloy_type: {
        **config,
        **DEFAULT_OOD_SETTINGS,
    }
    for alloy_type, config in get_tabpfn_configs(
        backend="api",
        feature_mode="numeric",
    ).items()
}


TABPFN_OOD_CONFIGS_BY_BACKEND: Dict[str, Dict[str, Dict[str, Any]]] = {
    "local:numeric": TABPFN_OOD_CONFIGS_LOCAL_NUMERIC,
    "api:text": TABPFN_OOD_CONFIGS_API_TEXT,
    "api:numeric": TABPFN_OOD_CONFIGS_API_NUMERIC,
}

TABPFN_OOD_CONFIGS = TABPFN_OOD_CONFIGS_LOCAL_NUMERIC


OOD_METHOD_BATCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "target_extrapolation": {
        "description": "Run target-extrapolation OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "target_extrapolation",
        "extrapolation_side": "low_to_high",
    },
    "sparse_x_single": {
        "description": "Run sparse_x_single OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "sparse_x_single",
        "sparse_candidate_pool_size": 500,
        "sparse_cluster_count": 50,
        "sparse_samples_per_cluster": 1,
        "sparse_kde_bandwidth": None,
    },
    "sparse_y_single": {
        "description": "Run sparse_y_single OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "sparse_y_single",
        "sparse_candidate_pool_size": 500,
        "sparse_cluster_count": 50,
        "sparse_samples_per_cluster": 1,
        "sparse_kde_bandwidth": None,
    },
    "sparse_x_cluster": {
        "description": "Run sparse_x_cluster OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "sparse_x_cluster",
        "sparse_candidate_pool_size": 500,
        "sparse_cluster_count": 50,
        "sparse_neighbors_per_seed": 5,
    },
    "sparse_y_cluster": {
        "description": "Run sparse_y_cluster OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "sparse_y_cluster",
        "sparse_candidate_pool_size": 500,
        "sparse_cluster_count": 50,
        "sparse_neighbors_per_seed": 5,
    },
    "loco": {
        "description": "Run LOCO OOD TabPFN for all supported alloys and targets.",
        "split_strategy": "loco",
        "loco_cluster_count": 5,
    },
}


def _build_batch_configs() -> Dict[str, Dict[str, Any]]:
    batch_configs: Dict[str, Dict[str, Any]] = {}
    for method_name, method_defaults in OOD_METHOD_BATCH_DEFAULTS.items():
        batch_configs[f"tabpfn_all_{method_name}"] = {
            "alloy_types": None,
            "exclude_alloys": [],
            "test_size": 0.2,
            "random_state": 42,
            "output_root": None,
            "align_predictions": True,
            **method_defaults,
        }

    batch_configs["tabpfn_all_ood"] = batch_configs["tabpfn_all_target_extrapolation"].copy()
    return batch_configs


TABPFN_OOD_BATCH_CONFIGS: Dict[str, Dict[str, Any]] = _build_batch_configs()


def get_tabpfn_ood_config(
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
        **DEFAULT_OOD_SETTINGS,
    }


def get_tabpfn_ood_configs(
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
    return TABPFN_OOD_CONFIGS_BY_BACKEND[f"{config_backend}:{config_feature_mode}"]


def get_all_tabpfn_ood_alloys(
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> List[str]:
    get_tabpfn_ood_configs(
        backend=backend,
        feature_mode=feature_mode,
        base_path=base_path,
    )
    return get_all_alloy_types(backend=backend, base_path=base_path)
