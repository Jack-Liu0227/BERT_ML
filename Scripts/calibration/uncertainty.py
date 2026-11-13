from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd

from .schemas import STANDARD_SAMPLE_COLUMNS, CalibrationConfig


BASE_PREDICTION_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "sample_id",
    "repeat_id",
    "y_true",
    "y_pred",
    "uncertainty_score",
    "source_path",
]


def _column(frame: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit and explicit in frame.columns:
        return explicit
    lowered = {str(name).lower(): str(name) for name in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _metadata_value(frame: pd.DataFrame, metadata: dict[str, Any], name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([metadata.get(name, "")] * len(frame), index=frame.index)


def standardize_prediction_source(entry: dict[str, Any], cfg: CalibrationConfig) -> pd.DataFrame:
    source_path = cfg.resolve(entry["path"])
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    frame = pd.read_csv(source_path)
    column_map = entry.get("column_map", {}) or {}
    metadata = entry.get("metadata", {}) or {}
    sample_col = _column(frame, column_map.get("sample_id"), ("sample_id", "ID", "id", "__row_id__"))
    repeat_col = _column(frame, column_map.get("repeat_id"), ("repeat_id", "seed", "member_id", "trial_id"))
    true_col = _column(frame, column_map.get("y_true"), ("y_true", "true_value", "actual", "actual_value"))
    pred_col = _column(frame, column_map.get("y_pred"), ("y_pred", "prediction", "predicted_value", "pred"))
    uncertainty_col = _column(
        frame,
        column_map.get("uncertainty_score"),
        ("uncertainty_score", "predictive_std", "sigma", "std"),
    )
    if true_col is None or pred_col is None:
        raise ValueError(f"Prediction source needs y_true and y_pred columns: {source_path}")

    result = pd.DataFrame(index=frame.index)
    for name in ("task_id", "protocol", "split_id", "model_family", "model"):
        result[name] = _metadata_value(frame, metadata, name)
    result["sample_id"] = frame[sample_col] if sample_col else np.arange(len(frame))
    result["repeat_id"] = frame[repeat_col] if repeat_col else pd.NA
    result["y_true"] = pd.to_numeric(frame[true_col], errors="coerce")
    result["y_pred"] = pd.to_numeric(frame[pred_col], errors="coerce")
    result["uncertainty_score"] = (
        pd.to_numeric(frame[uncertainty_col], errors="coerce") if uncertainty_col else np.nan
    )
    result["source_path"] = str(source_path)
    return result[BASE_PREDICTION_COLUMNS]


def aggregate_repeated_predictions(frame: pd.DataFrame, minimum_repeats: int = 2) -> pd.DataFrame:
    group_columns = ["task_id", "protocol", "split_id", "model_family", "model", "sample_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        predictions = pd.to_numeric(group["y_pred"], errors="coerce").dropna()
        existing_uncertainty = pd.to_numeric(group["uncertainty_score"], errors="coerce").dropna()
        if not existing_uncertainty.empty:
            uncertainty = float(existing_uncertainty.iloc[0])
            uncertainty_source = "provided"
        elif len(predictions) >= minimum_repeats:
            uncertainty = float(predictions.std(ddof=1))
            uncertainty_source = "repeat_std"
        else:
            uncertainty = np.nan
            uncertainty_source = "missing"
        truth = pd.to_numeric(group["y_true"], errors="coerce").dropna()
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "y_true": float(truth.iloc[0]) if not truth.empty else np.nan,
                "y_pred": float(predictions.mean()) if not predictions.empty else np.nan,
                "uncertainty_score": uncertainty,
                "repeat_count": int(len(predictions)),
                "uncertainty_source": uncertainty_source,
                "source_path": ";".join(sorted(set(group["source_path"].dropna().astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def run_uncertainty(cfg: CalibrationConfig, parameter_snapshot_id: str = "") -> pd.DataFrame:
    output_dir = cfg.output_root / "uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = cfg.raw.get("uncertainty", {}) or {}
    entries = settings.get("prediction_sources", []) or []
    minimum_repeats = int(settings.get("minimum_repeats", 2))
    frames: list[pd.DataFrame] = []
    source_errors: list[dict[str, str]] = []
    for entry in entries:
        try:
            frames.append(standardize_prediction_source(entry, cfg))
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            source_errors.append({"path": str(entry.get("path", "")), "message": str(exc)})

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_PREDICTION_COLUMNS)
    aggregated = aggregate_repeated_predictions(raw, minimum_repeats) if not raw.empty else pd.DataFrame()
    if aggregated.empty:
        sample = pd.DataFrame(columns=STANDARD_SAMPLE_COLUMNS + ["repeat_count", "uncertainty_source"])
    else:
        sample = aggregated.copy()
        sample["abs_error"] = (sample["y_pred"] - sample["y_true"]).abs()
        sample["confidence_tier"] = pd.NA
        sample["parameter_snapshot_id"] = parameter_snapshot_id
        sample = sample[STANDARD_SAMPLE_COLUMNS + ["repeat_count", "uncertainty_source"]]

    sample_path = output_dir / "sample_level_uncertainty.csv"
    sample.to_csv(sample_path, index=False)
    missing = sample[sample["uncertainty_score"].isna()].copy() if not sample.empty else sample.copy()
    missing.to_csv(output_dir / "missing_uncertainty.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "stage": "uncertainty",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_count": len(entries),
                "sample_count": int(len(sample)),
                "missing_uncertainty_count": int(sample["uncertainty_score"].isna().sum()) if not sample.empty else 0,
                "source_errors": source_errors,
                "output": str(sample_path),
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    return sample
