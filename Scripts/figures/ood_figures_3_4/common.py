from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FigureConfig, resolve_path


TIER_ORDER = ["High", "Medium", "Low"]
TIER_RISK_SCORE = {"High": 1.0, "Medium": 2.0, "Low": 3.0}


@dataclass(frozen=True)
class StageResult:
    stage: str
    output_dir: Path
    row_count: int
    missing_count: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "output_dir": str(self.output_dir),
            "row_count": self.row_count,
            "missing_count": self.missing_count,
        }


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def source_path(entry: dict[str, Any]) -> Path:
    return resolve_path(entry["path"])


def read_source_csv(entry: dict[str, Any]) -> pd.DataFrame:
    path = source_path(entry)
    frame = pd.read_csv(path)
    column_map = entry.get("column_map") or {}
    if column_map:
        frame = frame.rename(columns={source: target for source, target in column_map.items()})
    for key, value in (entry.get("metadata") or {}).items():
        if key not in frame.columns:
            frame[key] = value
    if "source_name" not in frame.columns:
        frame["source_name"] = entry.get("name", path.stem)
    if "source_file" not in frame.columns:
        frame["source_file"] = str(path)
    return frame


def coalesce_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])
    return None


def normalize_confidence_tier(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"high", "h", "confident", "low_uncertainty"}:
        return "High"
    if normalized in {"medium", "med", "m", "moderate"}:
        return "Medium"
    if normalized in {"low", "l", "uncertain", "high_uncertainty"}:
        return "Low"
    return str(value).strip()


def assign_uncertainty_tertiles(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = numeric.dropna().sort_values(kind="mergesort")
    n = len(valid)
    if n == 0:
        return result
    labels = ["High", "Medium", "Low"]
    for rank, index in enumerate(valid.index):
        bucket = min(int(rank * 3 / n), 2)
        result.loc[index] = labels[bucket]
    return result


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    delta = 1.96 * se
    return mean, mean - delta, mean + delta


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    paired = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(paired) < 2 or paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
        return np.nan
    return float(paired["x"].rank().corr(paired["y"].rank()))


def empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)
