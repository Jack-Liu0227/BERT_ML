from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from .common import StageResult, ensure_dirs
from .config import FigureConfig
from .discovery import _protocol_to_axis
from .figure3 import SAMPLE_COLUMNS, _standardize_source
from .manifest import write_manifest


def _discovered_path(cfg: FigureConfig) -> Path:
    return cfg.output_root / "audit" / "discovered_sources.csv"


def _metadata_from_discovered(row: pd.Series) -> dict[str, object]:
    return {
        "task_id": row.get("task_id"),
        "alloy_family": row.get("alloy_family"),
        "dataset_name": row.get("dataset_name"),
        "property": row.get("property"),
        "protocol": row.get("protocol"),
        "failure_axis": row.get("failure_axis") or _protocol_to_axis(row.get("protocol")),
        "model": row.get("model"),
        "model_family": row.get("model_family"),
    }


def prepare_figure3_from_discovery(cfg: FigureConfig) -> pd.DataFrame:
    path = _discovered_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    discovered = pd.read_csv(path)
    frames = []
    for _, row in discovered.iterrows():
        if row.get("usable_for") != "figure3":
            continue
        missing_reason = row.get("missing_reason")
        if pd.notna(missing_reason) and str(missing_reason):
            continue
        source_path = Path(str(row.get("path")))
        if not source_path.exists():
            continue
        entry = {
            "name": row.get("source_name"),
            "path": str(source_path),
            "required": True,
            "confidence_source": row.get("confidence_source") or "semantic",
            "metadata": _metadata_from_discovered(row),
        }
        frames.append(_standardize_source(entry))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def run_prepare_figure3(cfg: FigureConfig) -> StageResult:
    stage_dir = cfg.output_root / "figure3_calibration"
    ensure_dirs(stage_dir)
    legacy_csv_dir = stage_dir / "csv"
    if legacy_csv_dir.exists():
        shutil.rmtree(legacy_csv_dir)
    sample = prepare_figure3_from_discovery(cfg)
    write_manifest(
        stage_dir / "prepare_manifest.json",
        stage="prepare-figure3",
        config_path=cfg.config_path,
        output_root=cfg.output_root,
        payload={"row_count": int(len(sample))},
        source_files=[_discovered_path(cfg)] if _discovered_path(cfg).exists() else [],
    )
    return StageResult(stage="prepare-figure3", output_dir=stage_dir, row_count=len(sample))
