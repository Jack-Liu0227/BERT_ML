"""
Backward-compatible alias for the value-based TabPFN configs.

This module now reuses the local numeric-feature configs instead of keeping a
separate duplicated copy.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from .tabpfn_configs import TABPFN_MODEL_CONFIG
    from .tabpfn_configs_local import TABPFN_CONFIGS
except ImportError:  # pragma: no cover
    from tabpfn_configs import TABPFN_MODEL_CONFIG
    from tabpfn_configs_local import TABPFN_CONFIGS


def get_tabpfn_config(alloy_type: str) -> Dict[str, Any]:
    if alloy_type not in TABPFN_CONFIGS:
        raise ValueError(
            f"Unknown alloy type: {alloy_type}. "
            f"Available types: {list(TABPFN_CONFIGS.keys())}"
        )
    return TABPFN_CONFIGS[alloy_type]


def get_all_alloy_types() -> List[str]:
    return list(TABPFN_CONFIGS.keys())
