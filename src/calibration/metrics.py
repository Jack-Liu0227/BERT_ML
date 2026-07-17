from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .confidence import TIER_ORDER, assign_confidence_tiers
from .schemas import STANDARD_SAMPLE_COLUMNS, CalibrationConfig


SUMMARY_GROUPS = ["task_id", "protocol", "split_id", "model_family", "model"]


def mean_ci95(values: pd.Series) -> tuple[float, float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    se = sd / np.sqrt(len(numeric))
    return mean, sd, se, mean - 1.96 * se, mean + 1.96 * se


def summarize_tier_error(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame()
    working = sample.dropna(subset=["confidence_tier", "abs_error"]).copy()
    if working.empty:
        return pd.DataFrame()
    overall = working.groupby(SUMMARY_GROUPS, dropna=False)["abs_error"].mean().rename("overall_mae")
    rows: list[dict[str, object]] = []
    for keys, group in working.groupby(SUMMARY_GROUPS + ["confidence_tier"], observed=True, dropna=False):
        mean, sd, se, low, high = mean_ci95(group["abs_error"])
        base_key = keys[:-1]
        denominator = float(overall.loc[base_key])
        rows.append(
            {
                **dict(zip(SUMMARY_GROUPS, base_key)),
                "confidence_tier": str(keys[-1]),
                "n": int(len(group)),
                "mae": mean,
                "sd": sd,
                "se": se,
                "ci95_low": low,
                "ci95_high": high,
                "overall_mae": denominator,
                "normalized_mae": mean / denominator if denominator else np.nan,
                "normalized_ci95_low": low / denominator if denominator else np.nan,
                "normalized_ci95_high": high / denominator if denominator else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["confidence_tier"] = pd.Categorical(result["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return result.sort_values(SUMMARY_GROUPS + ["confidence_tier"]).astype({"confidence_tier": "object"})


def uncertainty_error_metrics(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in sample.groupby(SUMMARY_GROUPS, dropna=False):
        valid = group[["uncertainty_score", "abs_error"]].apply(pd.to_numeric, errors="coerce").dropna()
        correlation = float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) > 1 else np.nan
        rows.append(
            {
                **dict(zip(SUMMARY_GROUPS, keys)),
                "sample_n": int(len(valid)),
                "spearman_uncertainty_error": correlation,
            }
        )
    return pd.DataFrame(rows)


def run_calibration(cfg: CalibrationConfig, uncertainty_path: str | Path | None = None) -> pd.DataFrame:
    source_path = Path(uncertainty_path) if uncertainty_path else cfg.output_root / "uncertainty" / "sample_level_uncertainty.csv"
    output_dir = cfg.output_root / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        sample = pd.read_csv(source_path)
    else:
        sample = pd.DataFrame(columns=STANDARD_SAMPLE_COLUMNS)
    if not sample.empty:
        sample["abs_error"] = (pd.to_numeric(sample["y_pred"], errors="coerce") - pd.to_numeric(sample["y_true"], errors="coerce")).abs()
        sample = assign_confidence_tiers(sample, cfg.raw.get("confidence", {}) or {})
    for column in STANDARD_SAMPLE_COLUMNS:
        if column not in sample.columns:
            sample[column] = pd.NA
    sample = sample[STANDARD_SAMPLE_COLUMNS + [column for column in sample.columns if column not in STANDARD_SAMPLE_COLUMNS]]
    sample_path = output_dir / "sample_level_calibration.csv"
    sample.to_csv(sample_path, index=False)
    summary = summarize_tier_error(sample)
    summary.to_csv(output_dir / "tier_error_summary.csv", index=False)
    continuous = uncertainty_error_metrics(sample)
    continuous.to_csv(output_dir / "uncertainty_error_metrics.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "stage": "calibration",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": int(len(sample)),
                "tier_summary_count": int(len(summary)),
                "source": str(source_path),
                "sample_output": str(sample_path),
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    return sample
