from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .common import FORBIDDEN_TOKENS, SPLIT_SOURCE, WX_MODE, normalize_id, source_uses_global_ood_splits
from .predictions import parse_prediction_errors
from .sources import ResultSource
from .splits import test_id_set
from .wx import compute_source_wx, w_inventory_row


OUTPUT_COLUMNS = [
    "ood_scope",
    "task_key",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "model",
    "method",
    "fold_id",
    "test_set",
    "ID",
    "wx_mode",
    "split_source",
    "prediction_file",
    "source_split_dir",
    "wx",
    "wx_sample_score",
    "wx_mass_contribution",
    "wx_rank_desc",
    "target_value",
    "true_value",
    "predicted_value",
    "signed_error",
    "abs_error",
    "relative_error_pct",
    "prediction_set",
    "join_mode",
]


def process_source(source: ResultSource, *, hybrid_subset: str = "combined", wx_cache: dict[tuple[str, str], tuple] | None = None):
    cache_key = (str(source.split_dir), hybrid_subset)
    if wx_cache is not None and cache_key in wx_cache:
        wx_result, train, test, summary, feature_columns, test_set = wx_cache[cache_key]
    else:
        wx_result, train, test, summary, feature_columns, test_set = compute_source_wx(source, hybrid_subset=hybrid_subset)
        if wx_cache is not None:
            wx_cache[cache_key] = (wx_result, train, test, summary, feature_columns, test_set)
    allowed_ids = test_id_set(test)
    errors = parse_prediction_errors(source.prediction_file, allowed_ids=allowed_ids, test_set=test_set)
    samples = wx_result.sample_scores.copy()
    samples["join_id"] = [normalize_id(value) for value in samples["ID"].tolist()]
    joined = samples.merge(errors, on="join_id", how="inner", suffixes=("_w", ""))
    if "ID_w" in joined.columns:
        joined["ID"] = joined["ID_w"]

    joined["ood_scope"] = source.scope
    joined["task_key"] = source.task_key
    joined["task_id"] = source.task_id
    joined["alloy_family"] = source.family
    joined["dataset_name"] = source.dataset
    joined["property"] = source.property_name
    joined["model"] = source.model
    joined["method"] = source.method
    joined["fold_id"] = source.fold_id
    joined["test_set"] = test_set
    joined["wx_mode"] = WX_MODE
    joined["split_source"] = SPLIT_SOURCE
    joined["prediction_file"] = str(source.prediction_file)
    joined["source_split_dir"] = str(source.split_dir)
    joined["wx"] = float(wx_result.split_row["wx"])
    joined["join_mode"] = "fold_id_id"
    for column in OUTPUT_COLUMNS:
        if column not in joined.columns:
            joined[column] = pd.NA
    joined = joined[OUTPUT_COLUMNS].copy()

    matched_ids = set(joined["ID"].map(normalize_id).tolist())
    w_ids = set(samples["ID"].map(normalize_id).tolist())
    error_ids = set(errors["ID"].map(normalize_id).tolist())
    coverage = {
        "ood_scope": source.scope,
        "task_key": source.task_key,
        "model": source.model,
        "method": source.method,
        "fold_id": source.fold_id,
        "test_set": test_set,
        "source_split_dir": str(source.split_dir),
        "prediction_file": str(source.prediction_file),
        "w_rows": int(len(samples)),
        "error_rows": int(len(errors)),
        "matched_rows": int(len(joined)),
        "unmatched_w_rows": int(len(w_ids - matched_ids)),
        "unmatched_error_rows": int(len(error_ids - matched_ids)),
        "unmatched_w_ids": ";".join(sorted(w_ids - matched_ids)),
        "unmatched_error_ids": ";".join(sorted(error_ids - matched_ids)),
    }
    inventory = w_inventory_row(source, wx_result, train, test, feature_columns, test_set)
    return joined, coverage, inventory


def build_audit_summary(joined: pd.DataFrame, coverage: pd.DataFrame, w_inventory: pd.DataFrame, prediction_inventory: pd.DataFrame, *, scope: str, test_set: str) -> dict[str, Any]:
    forbidden_columns = [column for column in joined.columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    global_rows = (
        int(joined["source_split_dir"].map(source_uses_global_ood_splits).sum())
        if "source_split_dir" in joined.columns
        else 0
    )
    mean_diff = (
        pd.to_numeric(w_inventory.get("mean_wx_sample_score", pd.Series(dtype=float)), errors="coerce")
        - pd.to_numeric(w_inventory.get("wx", pd.Series(dtype=float)), errors="coerce")
    ).abs()
    mass_diff = (
        pd.to_numeric(w_inventory.get("sum_wx_mass_contribution", pd.Series(dtype=float)), errors="coerce")
        - pd.to_numeric(w_inventory.get("wx", pd.Series(dtype=float)), errors="coerce")
    ).abs()
    unmatched_w = int(pd.to_numeric(coverage.get("unmatched_w_rows", pd.Series(dtype=float)), errors="coerce").sum())
    unmatched_error = int(pd.to_numeric(coverage.get("unmatched_error_rows", pd.Series(dtype=float)), errors="coerce").sum())
    failures: list[str] = []
    if forbidden_columns:
        failures.append(f"Forbidden output columns: {forbidden_columns}")
    if global_rows:
        failures.append(f"Rows using global output/ood_splits: {global_rows}")
    if mean_diff.max(skipna=True) > 1e-8:
        failures.append(f"mean(wx_sample_score) conservation failed: {float(mean_diff.max())}")
    if mass_diff.max(skipna=True) > 1e-8:
        failures.append(f"sum(wx_mass_contribution) conservation failed: {float(mass_diff.max())}")
    warnings: list[str] = []
    if unmatched_error:
        failures.append(f"Unmatched prediction/error rows: error={unmatched_error}")
    if unmatched_w:
        warnings.append(f"W rows without prediction/error rows: w={unmatched_w}")
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "warnings": warnings,
        "ood_scope": scope,
        "test_set": test_set,
        "wx_mode": WX_MODE,
        "split_source": SPLIT_SOURCE,
        "output_rows": int(len(joined)),
        "prediction_files": int(len(prediction_inventory)),
        "w_sources": int(len(w_inventory)),
        "max_mean_wx_sample_score_diff": float(mean_diff.max(skipna=True)) if not mean_diff.empty else math.nan,
        "max_sum_wx_mass_contribution_diff": float(mass_diff.max(skipna=True)) if not mass_diff.empty else math.nan,
        "unmatched_w_rows": unmatched_w,
        "unmatched_error_rows": unmatched_error,
        "global_ood_splits_rows": global_rows,
        "forbidden_columns": forbidden_columns,
    }
