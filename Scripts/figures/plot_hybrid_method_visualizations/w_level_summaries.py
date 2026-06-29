from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


SAMPLE_LEVEL = "sample_level"
DISTRIBUTION_LEVEL = "distribution_level"
W_LEVELS = (SAMPLE_LEVEL, DISTRIBUTION_LEVEL)
W_LEVEL_ALIASES = {
    "sample": SAMPLE_LEVEL,
    SAMPLE_LEVEL: SAMPLE_LEVEL,
    "distribution": DISTRIBUTION_LEVEL,
    DISTRIBUTION_LEVEL: DISTRIBUTION_LEVEL,
}
W_SOURCES = {
    SAMPLE_LEVEL: "sample_w_contribution",
    DISTRIBUTION_LEVEL: "split_w",
}


def normalize_w_level(value: object) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key not in W_LEVEL_ALIASES:
        raise ValueError(f"Unknown W level: {value}")
    return W_LEVEL_ALIASES[key]


def selected_w_levels(value: object) -> list[str]:
    key = str(value or "both").strip().lower().replace("-", "_")
    if key == "both":
        return list(W_LEVELS)
    return [normalize_w_level(key)]


def weighted_quantile(values: Iterable[float], weights: Iterable[float], quantile: float) -> float:
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    finite = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not finite.any():
        return float("nan")
    value_array = value_array[finite]
    weight_array = weight_array[finite]
    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    cutoff = float(quantile) * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    index = min(index, len(ordered_values) - 1)
    return float(ordered_values[index])


def sample_identity(frame: pd.DataFrame) -> pd.Series:
    index_key = pd.Series(frame.index.astype(str), index=frame.index, dtype=object)
    if "sample_order" in frame.columns:
        sample_order = frame["sample_order"]
        sample_order_key = sample_order.where(sample_order.notna()).astype("object").astype(str)
        fallback = sample_order_key.where(sample_order.notna() & sample_order_key.ne(""), index_key)
    else:
        fallback = index_key
    if "ID" not in frame.columns:
        return fallback
    sample_id = frame["ID"]
    sample_id_key = sample_id.where(sample_id.notna()).astype("object").astype(str)
    return sample_id_key.where(sample_id.notna() & sample_id_key.ne(""), fallback)


def sample_occurrence_frame(group: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column
        for column in ["fold_id", "ID", "sample_order", "true_value", "predicted_value", "prediction_file"]
        if column in group.columns
    ]
    if not keys:
        return group.copy()
    return group.drop_duplicates(subset=keys).copy()


def split_count(group: pd.DataFrame) -> int:
    if "split_w" not in group.columns:
        return 0
    split_frame = split_level_frame(group)
    return int(len(split_frame))


def split_level_frame(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty or "split_w" not in group.columns:
        return pd.DataFrame(columns=["split_w", "n_samples", "relative_error_pct", "abs_error"])
    work = group.copy()
    work["_split_w_numeric"] = pd.to_numeric(work["split_w"], errors="coerce")
    finite = np.isfinite(work["_split_w_numeric"].to_numpy(dtype=float, na_value=np.nan))
    work = work[finite].copy()
    if work.empty:
        return pd.DataFrame(columns=["split_w", "n_samples", "relative_error_pct", "abs_error"])
    work["_source_split_key"] = work.get("source_split_dir", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_source_split_key"] = work["_source_split_key"].replace("", "__missing_split_dir__")
    work["_fold_key"] = work.get("fold_id", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_space_key"] = work.get("space", pd.Series("", index=work.index)).fillna("").astype(str)
    work["_sample_key"] = sample_identity(work)
    dedup = work.drop_duplicates(
        ["_source_split_key", "_fold_key", "_space_key", "_split_w_numeric", "_sample_key"]
    ).copy()
    split_cols = ["_source_split_key", "_fold_key", "_space_key", "_split_w_numeric"]
    grouped = (
        dedup.groupby(split_cols, dropna=False, sort=True)
        .agg(
            n_samples=("_sample_key", "nunique"),
            relative_error_pct=("relative_error_pct", "mean"),
            abs_error=("abs_error", "mean"),
        )
        .reset_index()
        .rename(columns={"_split_w_numeric": "split_w"})
    )
    return grouped[grouped["n_samples"].gt(0)].reset_index(drop=True)


def fit_slope(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2:
        return float("nan")
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    if np.unique(x_valid).size < 2:
        return float("nan")
    slope, _ = np.polyfit(x_valid, y_valid, 1)
    return float(slope) if math.isfinite(float(slope)) else float("nan")


def r2_score_safe(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_values = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    pred_values = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(true_values) & np.isfinite(pred_values)
    if int(valid.sum()) < 2:
        return float("nan")
    y = true_values[valid]
    p = pred_values[valid]
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denominator)


def base_error_metrics(group: pd.DataFrame) -> dict[str, object]:
    samples = sample_occurrence_frame(group)
    abs_error = pd.to_numeric(samples.get("abs_error", pd.Series(dtype=float)), errors="coerce")
    rel_error = pd.to_numeric(samples.get("relative_error_pct", pd.Series(dtype=float)), errors="coerce")
    true_values = pd.to_numeric(samples.get("true_value", pd.Series(np.nan, index=samples.index)), errors="coerce")
    pred_values = pd.to_numeric(samples.get("predicted_value", pd.Series(np.nan, index=samples.index)), errors="coerce")
    signed = pd.to_numeric(samples.get("signed_error", pred_values - true_values), errors="coerce")
    return {
        "MAE": float(abs_error.mean()) if abs_error.notna().any() else np.nan,
        "RMSE": float(np.sqrt(np.nanmean(abs_error.to_numpy(dtype=float) ** 2))) if abs_error.notna().any() else np.nan,
        "R2": r2_score_safe(true_values, pred_values),
        "relative_error_mean": float(rel_error.mean()) if rel_error.notna().any() else np.nan,
        "ceiling_bias": float(signed.mean()) if signed.notna().any() else np.nan,
        "underprediction_ratio": float((signed < 0).mean()) if signed.notna().any() else np.nan,
    }


def summarize_sample_w(group: pd.DataFrame) -> dict[str, object]:
    w = pd.to_numeric(group.get("sample_w_contribution", pd.Series(dtype=float)), errors="coerce")
    rel_error = pd.to_numeric(group.get("relative_error_pct", pd.Series(dtype=float)), errors="coerce")
    metrics = base_error_metrics(group)
    spearman_ready = w.notna().sum() >= 2 and rel_error.notna().sum() >= 2 and w.nunique(dropna=True) >= 2 and rel_error.nunique(dropna=True) >= 2
    return {
        "W_level": SAMPLE_LEVEL,
        "W_source": W_SOURCES[SAMPLE_LEVEL],
        "W_mean": float(w.mean()) if w.notna().any() else np.nan,
        "W_median": float(w.median()) if w.notna().any() else np.nan,
        "W_p90": float(w.quantile(0.9)) if w.notna().any() else np.nan,
        **metrics,
        "slope": fit_slope(w, rel_error),
        "spearman": float(w.corr(rel_error, method="spearman")) if spearman_ready else np.nan,
        "n_samples": int(len(group)),
        "n_splits": split_count(group),
    }


def summarize_distribution_w(group: pd.DataFrame) -> dict[str, object]:
    split_frame = split_level_frame(group)
    metrics = base_error_metrics(group)
    if split_frame.empty:
        return {
            "W_level": DISTRIBUTION_LEVEL,
            "W_source": W_SOURCES[DISTRIBUTION_LEVEL],
            "W_mean": np.nan,
            "W_median": np.nan,
            "W_p90": np.nan,
            **metrics,
            "slope": np.nan,
            "spearman": np.nan,
            "n_samples": 0,
            "n_splits": 0,
        }
    split_w = pd.to_numeric(split_frame["split_w"], errors="coerce")
    weights = pd.to_numeric(split_frame["n_samples"], errors="coerce")
    rel_error = pd.to_numeric(split_frame["relative_error_pct"], errors="coerce")
    spearman_ready = (
        split_w.notna().sum() >= 2
        and rel_error.notna().sum() >= 2
        and split_w.nunique(dropna=True) >= 2
        and rel_error.nunique(dropna=True) >= 2
    )
    return {
        "W_level": DISTRIBUTION_LEVEL,
        "W_source": W_SOURCES[DISTRIBUTION_LEVEL],
        "W_mean": float(np.average(split_w.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float))),
        "W_median": weighted_quantile(split_w, weights, 0.5),
        "W_p90": weighted_quantile(split_w, weights, 0.9),
        **metrics,
        "slope": fit_slope(split_w, rel_error),
        "spearman": float(split_w.corr(rel_error, method="spearman")) if spearman_ready else np.nan,
        "n_samples": int(weights.sum()),
        "n_splits": int(len(split_frame)),
    }


def summarize_w_group(group: pd.DataFrame, w_level: str) -> dict[str, object]:
    level = normalize_w_level(w_level)
    if level == DISTRIBUTION_LEVEL:
        return summarize_distribution_w(group)
    return summarize_sample_w(group)
