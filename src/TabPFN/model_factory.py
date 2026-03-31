"""
Helpers for choosing TabPFN runtime options and creating API/local regressors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from tabpfn_client import TabPFNRegressor as ApiTabPFNRegressor
    from tabpfn_client import set_access_token

    TABPFN_API_CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ApiTabPFNRegressor = None
    set_access_token = None
    TABPFN_API_CLIENT_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor as LocalTabPFNRegressor
    from tabpfn.constants import ModelVersion as LocalModelVersion

    TABPFN_LOCAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    LocalTabPFNRegressor = None
    LocalModelVersion = None
    TABPFN_LOCAL_AVAILABLE = False


TABPFN_API_KEY_ENV_VARS = ("TABPFN_API_KEY", "PRIORLABS_API_KEY")
TABPFN_FEATURE_MODES = ("numeric", "text")


def load_tabpfn_env(base_path: str | Path = ".") -> None:
    if load_dotenv is None:
        return

    env_path = Path(base_path) / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_tabpfn_api_key(base_path: str | Path = ".") -> Optional[str]:
    load_tabpfn_env(base_path)
    for env_var in TABPFN_API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value and value.strip():
            return value.strip()
    return None


def resolve_tabpfn_backend(
    *,
    base_path: str | Path = ".",
    backend: str = "auto",
) -> str:
    backend = backend.strip().lower()
    if backend not in {"auto", "api", "local"}:
        raise ValueError(
            f"Unsupported TabPFN backend: {backend}. "
            "Expected one of: auto, api, local."
        )

    if backend == "auto":
        return "api" if get_tabpfn_api_key(base_path) is not None else "local"
    return backend


def resolve_tabpfn_feature_mode(
    *,
    base_path: str | Path = ".",
    backend: str = "auto",
    feature_mode: str | None = None,
) -> str:
    resolved_backend = resolve_tabpfn_backend(base_path=base_path, backend=backend)
    if feature_mode is None:
        return "text" if resolved_backend == "api" else "numeric"

    normalized = feature_mode.strip().lower()
    if normalized not in TABPFN_FEATURE_MODES:
        raise ValueError(
            f"Unsupported TabPFN feature mode: {feature_mode}. "
            "Expected one of: numeric, text."
        )
    if resolved_backend == "local" and normalized != "numeric":
        raise ValueError(
            "Local TabPFN V2 does not support text feature mode. "
            "Use `--feature_mode numeric` for local, or switch to `--backend api`."
        )
    return normalized


def get_tabpfn_runtime_config(
    *,
    base_path: str | Path = ".",
    backend: str = "auto",
    preferred_model_version: str | None = None,
    feature_mode: str | None = None,
) -> Dict[str, Any]:
    resolved_backend = resolve_tabpfn_backend(base_path=base_path, backend=backend)
    resolved_feature_mode = resolve_tabpfn_feature_mode(
        base_path=base_path,
        backend=resolved_backend,
        feature_mode=feature_mode,
    )

    if resolved_backend == "api":
        model_name = "TabPFN-2.5-Plus"
        model_dirname = "TabPFN-2.5-Plus"
        effective_model_version = "api_2.5_plus"
    else:
        requested_version = preferred_model_version.strip().lower() if preferred_model_version else "v2"
        if requested_version != "v2":
            raise ValueError(
                "Local TabPFN is fixed to `TabPFN V2`. "
                "Use `--model_version v2` or omit the flag."
            )
        model_name = "TabPFN V2"
        model_dirname = "TabPFN-V2"
        effective_model_version = "v2"

    feature_mode_dirname = resolved_feature_mode.title()

    return {
        "backend": resolved_backend,
        "requested_backend": backend,
        "resolved_backend": resolved_backend,
        "requested_feature_mode": feature_mode,
        "feature_mode": resolved_feature_mode,
        "feature_mode_dirname": feature_mode_dirname,
        "requested_model_version": preferred_model_version,
        "preferred_model_version": effective_model_version,
        "effective_model_version": effective_model_version,
        "model_name": model_name,
        "model_dirname": model_dirname,
        "model_run_dirname": f"{model_dirname}-{feature_mode_dirname}",
    }


def create_tabpfn_regressor(
    *,
    base_path: str | Path = ".",
    preferred_model_version: str | None = None,
    backend: str = "auto",
    feature_mode: str | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    runtime_config = get_tabpfn_runtime_config(
        base_path=base_path,
        backend=backend,
        preferred_model_version=preferred_model_version,
        feature_mode=feature_mode,
    )
    resolved_backend = runtime_config["resolved_backend"]

    api_key = get_tabpfn_api_key(base_path)
    should_use_api = resolved_backend == "api" and api_key is not None
    if should_use_api:
        if not TABPFN_API_CLIENT_AVAILABLE:
            raise ImportError(
                "Found a TabPFN API key, but tabpfn-client is not installed. "
                "Install it with `pip install -U tabpfn-client`."
            )
        assert set_access_token is not None
        assert ApiTabPFNRegressor is not None
        set_access_token(api_key)
        return (
            ApiTabPFNRegressor(),
            {
                **runtime_config,
                "api_key_env_var": next(
                    env_var
                    for env_var in TABPFN_API_KEY_ENV_VARS
                    if os.getenv(env_var)
                ),
            },
        )

    if resolved_backend == "api":
        if api_key is None:
            raise RuntimeError(
                "TabPFN backend 'api' was requested, but no API key was found. "
                "Add `TABPFN_API_KEY` or `PRIORLABS_API_KEY` to `.env`."
            )

    if not TABPFN_LOCAL_AVAILABLE:
        raise ImportError(
            "Neither a TabPFN API backend nor a local tabpfn installation is available."
        )

    assert LocalTabPFNRegressor is not None
    if LocalModelVersion is None:
        raise ImportError(
            "tabpfn is installed, but ModelVersion could not be imported."
        )
    version = getattr(LocalModelVersion, "V2")
    return (
        LocalTabPFNRegressor.create_default_for_version(version),
        runtime_config,
    )
