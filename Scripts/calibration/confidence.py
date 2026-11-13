from __future__ import annotations

import pandas as pd


TIER_ORDER = ["High", "Medium", "Low"]
GROUP_COLUMNS = ["task_id", "protocol", "split_id", "model_family", "model"]


def _quantile_tiers(values: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    if valid.empty:
        return result
    low_threshold = float(valid.quantile(lower))
    high_threshold = float(valid.quantile(upper))
    result.loc[numeric <= low_threshold] = "High"
    result.loc[(numeric > low_threshold) & (numeric <= high_threshold)] = "Medium"
    result.loc[numeric > high_threshold] = "Low"
    return result


def _absolute_tiers(values: pd.Series, high_max: float, medium_max: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    result.loc[numeric <= high_max] = "High"
    result.loc[(numeric > high_max) & (numeric <= medium_max)] = "Medium"
    result.loc[numeric > medium_max] = "Low"
    return result


def assign_confidence_tiers(frame: pd.DataFrame, settings: dict[str, object] | None = None) -> pd.DataFrame:
    settings = settings or {}
    method = str(settings.get("method", "quantile")).lower()
    output = frame.copy()
    output["confidence_tier"] = pd.NA
    for _, index in output.groupby(GROUP_COLUMNS, dropna=False).groups.items():
        values = output.loc[index, "uncertainty_score"]
        if method == "quantile":
            lower = float(settings.get("lower_quantile", 1.0 / 3.0))
            upper = float(settings.get("upper_quantile", 2.0 / 3.0))
            tiers = _quantile_tiers(values, lower, upper)
        elif method == "absolute":
            tiers = _absolute_tiers(
                values,
                float(settings["high_max"]),
                float(settings["medium_max"]),
            )
        elif method == "continuous":
            tiers = pd.Series(pd.NA, index=index, dtype="object")
        else:
            raise ValueError(f"Unsupported confidence method: {method}")
        output.loc[index, "confidence_tier"] = tiers
    output["confidence_tier"] = pd.Categorical(output["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return output
