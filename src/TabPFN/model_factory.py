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
    from tabpfn_client.constants import ModelVersion as ApiModelVersion
    from tabpfn_client import set_access_token
    from tabpfn_client.client import ServiceClient as ApiServiceClient
    import httpx

    TABPFN_API_CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ApiTabPFNRegressor = None
    ApiModelVersion = None
    set_access_token = None
    ApiServiceClient = None
    httpx = None
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
DEFAULT_TABPFN_API_TIMEOUT_SECONDS = 300.0


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


def configure_tabpfn_api_client_timeout() -> float:
    raw = os.getenv(
        "TABPFN_API_TIMEOUT_SECONDS",
        str(DEFAULT_TABPFN_API_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid TABPFN_API_TIMEOUT_SECONDS: {raw!r}") from exc
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("TABPFN_API_TIMEOUT_SECONDS must be in (0, 3600].")
    if ApiServiceClient is None or httpx is None:
        raise ImportError("tabpfn-client/httpx is unavailable for API timeout configuration.")
    ApiServiceClient.httpx_timeout_s = timeout_seconds
    ApiServiceClient.httpx_client.timeout = httpx.Timeout(timeout_seconds)
    return timeout_seconds


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

    if backend == "local":
        raise ValueError(
            "Only TabPFN API 2.5 Plus is allowed; local TabPFN V2 is disabled."
        )
    return "api"


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

    requested_version = (
        preferred_model_version.strip().lower()
        if preferred_model_version
        else "v2.5"
    )
    if requested_version not in {"v2.5", "api_2.5_plus"}:
        raise ValueError(
            "Only TabPFN API 2.5 Plus is allowed; use `--model_version v2.5` "
            "or omit the model-version flag."
        )
    model_name = "TabPFN-2.5-Plus"
    model_dirname = "TabPFN-2.5-Plus"
    effective_model_version = "api_2.5_plus"

    feature_mode_dirname = resolved_feature_mode.title()

    return {
        "backend": resolved_backend,
        "requested_backend": backend,
        "resolved_backend": resolved_backend,
        "requested_feature_mode": feature_mode,
        "feature_mode": resolved_feature_mode,
        "feature_mode_dirname": feature_mode_dirname,
        "requested_model_version": preferred_model_version,
        "requested_model_path": "v2.5_default",
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
    random_state: int | None = None,
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
        if ApiModelVersion is None:
            raise ImportError(
                "tabpfn-client is installed, but its ModelVersion enum is unavailable."
            )
        api_timeout_seconds = configure_tabpfn_api_client_timeout()
        set_access_token(api_key)
        overrides = {"random_state": int(random_state)} if random_state is not None else {}
        model = ApiTabPFNRegressor.create_default_for_version(
            ApiModelVersion.V2_5,
            **overrides,
        )
        if str(getattr(model, "model_path", "")) != "v2.5_default":
            raise RuntimeError(
                "TabPFN API client did not resolve the explicitly requested v2.5_default model."
            )
        return (
            model,
            {
                **runtime_config,
                "random_state": random_state,
                "api_timeout_seconds": api_timeout_seconds,
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

    raise RuntimeError("Only TabPFN API 2.5 Plus is allowed.")
