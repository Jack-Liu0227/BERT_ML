from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


STANDARD_SAMPLE_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "sample_id",
    "y_true",
    "y_pred",
    "abs_error",
    "uncertainty_score",
    "confidence_tier",
    "parameter_snapshot_id",
    "source_path",
]

PARAMETER_AUDIT_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "source_root",
    "model_dir",
    "status",
    "parameter_config_path",
    "parameter_evidence_path",
    "split_manifest_path",
    "train_split_hash",
    "test_split_hash",
    "parameter_selection_basis",
    "reporting_selection_mode",
    "reporting_uses_ood_test",
    "parameter_payload_json",
    "message",
]


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    project_root: Path
    output_root: Path
    raw: dict[str, Any]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path


def default_config_path() -> Path:
    return Path(__file__).with_name("calibration.config.yaml")


def load_config(path: str | Path | None = None, output_root: str | Path | None = None) -> CalibrationConfig:
    config_path = Path(path) if path else default_config_path()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    configured_project_root = raw.get("project_root")
    if configured_project_root:
        candidate = Path(configured_project_root)
        project_root = candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()
    else:
        project_root = Path(__file__).resolve().parents[2]

    output_value = output_root or raw.get("output_root", "output/calibration")
    output_path = Path(output_value)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    return CalibrationConfig(
        path=config_path,
        project_root=project_root,
        output_root=output_path.resolve(),
        raw=raw,
    )
