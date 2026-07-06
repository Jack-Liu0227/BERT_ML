from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import frame_join_ids, join_id_column, read_csv

TRAIN_LABELS = {"train", "training"}


def find_true_pred_columns(frame: pd.DataFrame) -> tuple[str, str]:
    columns = list(frame.columns)
    true_candidates = [column for column in columns if column.lower().startswith(("true_", "actual_"))]
    pred_candidates = [column for column in columns if column.lower().startswith(("pred_", "prediction_"))]
    if true_candidates and pred_candidates:
        return true_candidates[0], pred_candidates[0]

    actual_by_prefix = {}
    for column in columns:
        lower = column.lower()
        for suffix in ("_actual", "_true"):
            if lower.endswith(suffix):
                actual_by_prefix[column[: -len(suffix)]] = column
    predicted_by_prefix = {
        column[: -len("_Predicted")]: column
        for column in columns
        if column.lower().endswith("_predicted")
    }
    for prefix, true_column in actual_by_prefix.items():
        pred_column = predicted_by_prefix.get(prefix)
        if pred_column:
            return true_column, pred_column

    true_candidates = [column for column in columns if column.lower() in {"true", "actual", "y_true", "target"}]
    pred_candidates = [column for column in columns if column.lower() in {"pred", "prediction", "y_pred", "predicted"}]
    if true_candidates and pred_candidates:
        return true_candidates[0], pred_candidates[0]

    raise ValueError(f"Could not identify true/prediction columns from: {columns}")


def parse_prediction_errors(prediction_file: Path, *, allowed_ids: set[str] | None = None, test_set: str = "") -> pd.DataFrame:
    predictions = read_csv(prediction_file)
    id_column = join_id_column(predictions)
    if id_column is None:
        raise ValueError(f"{prediction_file} is missing a supported join ID column.")
    true_col, pred_col = find_true_pred_columns(predictions)
    work = predictions.copy()
    if "set" in work.columns:
        set_values = work["set"].astype(str).str.lower()
        work = work[~set_values.isin(TRAIN_LABELS)].copy()
    work["join_id"] = frame_join_ids(work)
    if allowed_ids is not None:
        work = work[work["join_id"].isin(allowed_ids)].copy()

    true_values = pd.to_numeric(work[true_col], errors="coerce")
    predicted_values = pd.to_numeric(work[pred_col], errors="coerce")
    signed_error = predicted_values - true_values
    abs_error = signed_error.abs()
    denominator = true_values.abs()
    return pd.DataFrame(
        {
            "ID": work[id_column].to_numpy(),
            "join_id": work["join_id"].to_numpy(),
            "true_value": true_values.to_numpy(),
            "predicted_value": predicted_values.to_numpy(),
            "signed_error": signed_error.to_numpy(),
            "abs_error": abs_error.to_numpy(),
            "relative_error_pct": np.where(denominator > 0, abs_error / denominator * 100.0, np.nan),
            "prediction_set": work["set"].astype(str).to_numpy() if "set" in work.columns else test_set,
            "test_set": test_set,
        }
    )


def prediction_inventory(source, *, allowed_ids: set[str] | None, test_set: str) -> dict[str, object]:
    predictions = read_csv(source.prediction_file)
    if "set" in predictions.columns:
        sets = predictions["set"].astype(str).str.lower()
        train_rows = int(sets.isin(TRAIN_LABELS).sum())
    else:
        train_rows = 0
    errors = parse_prediction_errors(source.prediction_file, allowed_ids=allowed_ids, test_set=test_set)
    return {
        "ood_scope": source.scope,
        "task_key": source.task_key,
        "model": source.model,
        "method": source.method,
        "fold_id": source.fold_id,
        "test_set": test_set,
        "prediction_file": str(source.prediction_file),
        "exists": source.prediction_file.exists(),
        "total_rows": int(len(predictions)),
        "train_rows": train_rows,
        "used_error_rows": int(len(errors)),
    }
