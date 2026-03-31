"""
TabPFN dataset configs for the API backend.

The API backend can keep raw processing text so the hosted model can use it
directly.
"""

from __future__ import annotations

try:
    from .tabpfn_config_base import build_tabpfn_configs
except ImportError:  # pragma: no cover
    from tabpfn_config_base import build_tabpfn_configs


TABPFN_CONFIGS_API_TEXT = build_tabpfn_configs("text")
TABPFN_CONFIGS_API_NUMERIC = build_tabpfn_configs("numeric")

# Backward-compatible default for API callers that do not specify feature mode.
TABPFN_CONFIGS = TABPFN_CONFIGS_API_TEXT
