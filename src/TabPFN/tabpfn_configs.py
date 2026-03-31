"""
TabPFN configuration dispatcher.

This module keeps the public config API stable while routing callers to the
explicit API or local config files.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from .tabpfn_configs_api import (
        TABPFN_CONFIGS as TABPFN_CONFIGS_API,
        TABPFN_CONFIGS_API_NUMERIC,
        TABPFN_CONFIGS_API_TEXT,
    )
    from .tabpfn_configs_local import TABPFN_CONFIGS as TABPFN_CONFIGS_LOCAL
except ImportError:  # pragma: no cover
    from model_factory import resolve_tabpfn_backend, resolve_tabpfn_feature_mode
    from tabpfn_configs_api import (
        TABPFN_CONFIGS as TABPFN_CONFIGS_API,
        TABPFN_CONFIGS_API_NUMERIC,
        TABPFN_CONFIGS_API_TEXT,
    )
    from tabpfn_configs_local import TABPFN_CONFIGS as TABPFN_CONFIGS_LOCAL


TABPFN_CONFIGS_BY_RUNTIME: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {
    ("local", "numeric"): TABPFN_CONFIGS_LOCAL,
    ("api", "numeric"): TABPFN_CONFIGS_API_NUMERIC,
    ("api", "text"): TABPFN_CONFIGS_API_TEXT,
}

# Backward-compatible default for legacy imports that do not pass a backend.
TABPFN_CONFIGS = TABPFN_CONFIGS_LOCAL


TABPFN_MODEL_CONFIG = {
    "model_version": "v2",
    "feature_mode": None,
    "task_type": "regression",
    "metrics": {
        "regression": ["mae", "rmse", "r2", "mape"],
        "classification": ["accuracy", "roc_auc", "f1"],
    },
}


def get_tabpfn_configs(
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> Dict[str, Dict[str, Any]]:
    resolved_backend = resolve_tabpfn_backend(base_path=base_path, backend=backend)
    resolved_feature_mode = resolve_tabpfn_feature_mode(
        base_path=base_path,
        backend=resolved_backend,
        feature_mode=feature_mode,
    )
    return TABPFN_CONFIGS_BY_RUNTIME[(resolved_backend, resolved_feature_mode)]


def get_tabpfn_config(
    alloy_type: str,
    backend: str = "local",
    feature_mode: str | None = None,
    base_path: str = ".",
) -> Dict[str, Any]:
    resolved_backend = resolve_tabpfn_backend(base_path=base_path, backend=backend)
    resolved_feature_mode = resolve_tabpfn_feature_mode(
        base_path=base_path,
        backend=resolved_backend,
        feature_mode=feature_mode,
    )
    backend_configs = TABPFN_CONFIGS_BY_RUNTIME[(resolved_backend, resolved_feature_mode)]
    if alloy_type not in backend_configs:
        raise ValueError(
            f"Unknown alloy type: {alloy_type}. "
            f"Available types: {list(backend_configs.keys())}"
        )

    config = backend_configs[alloy_type].copy()
    config["requested_backend"] = backend
    config["config_backend"] = resolved_backend
    config["requested_feature_mode"] = feature_mode
    config["feature_mode"] = resolved_feature_mode
    return config


def get_all_alloy_types(
    backend: str = "local",
    base_path: str = ".",
) -> List[str]:
    return list(get_tabpfn_configs(backend=backend, base_path=base_path).keys())
