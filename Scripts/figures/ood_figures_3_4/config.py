from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FigureConfig:
    config_path: Path
    output_root: Path
    raw: dict[str, Any]


def default_config_path() -> Path:
    return REPO_ROOT / "configs" / "figures" / "ood_figures_3_4.config.yaml"


def resolve_path(value: str | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: str | Path, output_root_override: str | Path | None = None) -> FigureConfig:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    output_value = output_root_override if output_root_override is not None else raw.get("output_root", "output/ood_figures_3_4")
    output_root = resolve_path(output_value)
    return FigureConfig(config_path=path, output_root=output_root, raw=raw)


def list_section_sources(cfg: FigureConfig, section: str, key: str) -> list[dict[str, Any]]:
    value = cfg.raw.get(section, {}).get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{section}.{key} must be a list")
    return [entry for entry in value if isinstance(entry, dict)]


def configured_required_names(cfg: FigureConfig, section: str) -> set[str]:
    value = cfg.raw.get(section, {}).get("required_sources", [])
    return {str(item) for item in value or []}
