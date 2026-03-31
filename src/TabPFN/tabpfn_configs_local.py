"""
TabPFN dataset configs for the local backend.

Local/open checkpoints use numeric processing columns instead of raw
processing text.
"""

from __future__ import annotations

try:
    from .tabpfn_config_base import build_tabpfn_configs
except ImportError:  # pragma: no cover
    from tabpfn_config_base import build_tabpfn_configs


TABPFN_CONFIGS_LOCAL = build_tabpfn_configs("numeric")
TABPFN_CONFIGS = TABPFN_CONFIGS_LOCAL
