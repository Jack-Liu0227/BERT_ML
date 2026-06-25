from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DELTA_COLUMNS,
    METHOD_ORDER,
    PURE_HYBRID_COLUMNS,
    RAW_SET_SUBSETS,
    SET_AB_AUDIT_COLUMNS,
    SET_AB_FAMILY_BEST_COLUMNS,
    SET_AB_MASTER_COLUMNS,
    SET_AB_MODEL_SUMMARY_COLUMNS,
    SET_FILESYSTEM_NAMES,
    SUMMARY_COLUMNS,
    SUBSET_LABELS,
    TRIPTYCH_MODEL_REVIEW_COLUMNS,
    YZ_OOD_MAP_COLUMNS,
)
from .io import clean_text, read_csv


def normalize_method(value: object) -> str:
    raw = clean_text(value)
    key = raw.lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    key = key.replace("randcv", "random_cv")
    for prefix in ("hybridhigh20_", "hybrid_high20_", "hybrid_extrapolation_"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    mapping = {
        "random_cv": "RandCV",
        "random_cv_baseline": "RandCV",
        "hybridhigh20_randcv": "RandCV",
        "hybridhigh20_randomcv": "RandCV",
        "loco": "LOCO",
        "loco_k5": "LOCO",
        "hybridhigh20_loco": "LOCO",
        "target_extrapolation": "Extra.",
        "extrapolation": "Extra.",
        "sparse_x_single": "SX-sgl",
        "sparse_x_single_k5": "SX-sgl",
        "hybridhigh20_sparsexsingle": "SX-sgl",
        "hybridhigh20_sparse_x_single": "SX-sgl",
        "sparse_x_cluster": "SX-cls",
        "sparse_x_cluster_k5": "SX-cls",
        "hybridhigh20_sparsexcluster": "SX-cls",
        "hybridhigh20_sparse_x_cluster": "SX-cls",
        "sparse_y_single": "SY-sgl",
        "sparse_y_single_k5": "SY-sgl",
        "hybridhigh20_sparseysingle": "SY-sgl",
        "hybridhigh20_sparse_y_single": "SY-sgl",
        "sparse_y_cluster": "SY-cls",
        "sparse_y_cluster_k5": "SY-cls",
        "hybridhigh20_sparseycluster": "SY-cls",
        "hybridhigh20_sparse_y_cluster": "SY-cls",
    }
    return mapping.get(key, raw)


def normalize_subset(value: object) -> str:
    text = clean_text(value)
    key = text.lower()
    return SUBSET_LABELS.get(key, text or "Unknown")


def subset_filesystem_name(value: object) -> str:
    label = normalize_subset(value)
    return SET_FILESYSTEM_NAMES.get(label, label.replace(" ", "_") or "Unknown")


def method_sort_key(value: object) -> tuple[int, str]:
    method = clean_text(value)
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else 999, method)


def model_family_for(model: object, explicit: object = "") -> str:
    text = clean_text(model).lower()
    if "gpt" in text:
        return "GPT"
    explicit_text = clean_text(explicit)
    if explicit_text:
        return explicit_text
    if any(token in text for token in ("bert", "scibert", "matscibert", "steelbert")):
        return "BERT"
    if "tabpfn" in text:
        return "TabPFN"
    if "llm" in text or "prop" in text:
        return "LLMProp"
    return "Traditional"


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


def sample_occurrence_frame(group: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ["fold_id", "ID", "sample_order", "true_value", "predicted_value", "prediction_file"] if column in group.columns]
    if not keys:
        return group.copy()
    return group.drop_duplicates(subset=keys).copy()


def build_set_ab_master_sample_long(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SET_AB_MASTER_COLUMNS)
    work = frame.copy()
    work["case"] = work.get("task_id", work.get("task_key", "case")).fillna("").astype(str)
    work["task_key"] = work.get("task_key", work["case"]).fillna("").astype(str)
    work["model"] = work.get("model", "").fillna("").astype(str)
    work["model_family"] = [
        model_family_for(model, family)
        for model, family in zip(work["model"], work.get("model_family", pd.Series("", index=work.index)))
    ]
    work["method"] = work.get("method_short", work.get("method", "")).map(normalize_method)
    work["raw_subset"] = work.get("raw_subset", work.get("test_set", "")).fillna("").astype(str)
    work["set_label"] = work["raw_subset"].map(normalize_subset)
    work["subset"] = work["set_label"]
    work["subset_fs"] = work["set_label"].map(subset_filesystem_name)
    work["aggregation"] = "fold_weighted"
    work = work[work["raw_subset"].isin(RAW_SET_SUBSETS) & work["set_label"].isin(["Set A", "Set B"])].copy()
    for column in SET_AB_MASTER_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan
    for column in ["sample_w_contribution", "sample_w_mass_contribution", "split_w", "true_value", "predicted_value", "signed_error", "abs_error", "relative_error_pct"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work["signed_error"].isna().all():
        work["signed_error"] = work["predicted_value"] - work["true_value"]
    if work["abs_error"].isna().all():
        work["abs_error"] = work["signed_error"].abs()
    return work[SET_AB_MASTER_COLUMNS].sort_values(["case", "method", "model", "set_label", "space"], kind="stable").reset_index(drop=True)


def summarize_model_group(group: pd.DataFrame, keys: tuple[object, ...], key_cols: list[str]) -> dict[str, object]:
    values = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
    samples = sample_occurrence_frame(group)
    abs_error = pd.to_numeric(samples["abs_error"], errors="coerce")
    rel_error = pd.to_numeric(samples["relative_error_pct"], errors="coerce")
    true_values = pd.to_numeric(samples["true_value"], errors="coerce")
    pred_values = pd.to_numeric(samples["predicted_value"], errors="coerce")
    signed = pd.to_numeric(samples["signed_error"], errors="coerce")
    fold_values = samples.get("fold_id", pd.Series(index=samples.index, dtype=object))
    id_values = samples.get("ID", pd.Series(index=samples.index, dtype=object))
    return {
        **values,
        "aggregation": "fold_weighted",
        "MAE": float(abs_error.mean()) if abs_error.notna().any() else np.nan,
        "RMSE": float(np.sqrt(np.nanmean(abs_error.to_numpy(dtype=float) ** 2))) if abs_error.notna().any() else np.nan,
        "R2": r2_score_safe(true_values, pred_values),
        "relative_error_mean": float(rel_error.mean()) if rel_error.notna().any() else np.nan,
        "ceiling_bias": float(signed.mean()) if signed.notna().any() else np.nan,
        "underprediction_ratio": float((signed < 0).mean()) if signed.notna().any() else np.nan,
        "n_samples": int(len(samples)),
        "n_folds": int(fold_values.dropna().nunique()) if len(fold_values) else 0,
        "n_ids": int(id_values.dropna().nunique()) if len(id_values) else 0,
        "MAE_sample": float(abs_error.mean()) if abs_error.notna().any() else np.nan,
        "MAE_source": "sample_w_error",
        "triptych_source_csv": "",
    }


def build_triptych_model_review(triptych_root: Path | None, case_contains: str | None = None) -> pd.DataFrame:
    if triptych_root is None or not Path(triptych_root).exists():
        return pd.DataFrame(columns=TRIPTYCH_MODEL_REVIEW_COLUMNS)
    rows: list[dict[str, object]] = []
    for raw_subset in RAW_SET_SUBSETS:
        csv_dir = Path(triptych_root) / raw_subset / "csv"
        if not csv_dir.exists():
            continue
        for path in csv_dir.glob("*.csv"):
            name = path.name.lower()
            if not (name.endswith("_mae.csv") or name.endswith("_models_mae.csv")):
                continue
            frame = read_csv(path)
            if frame.empty:
                continue
            if "model" in frame.columns:
                model_col = "model"
                if "bert_models_mae" in name:
                    family = "BERT"
                elif "traditional_models_mae" in name:
                    family = "Traditional"
                else:
                    family = ""
            elif "aggregate_label" in frame.columns:
                model_col = "aggregate_label"
                family = ""
            else:
                continue
            for _, item in frame.iterrows():
                model = clean_text(item.get(model_col))
                if model in {"BERT-best", "Traditional-best"}:
                    continue
                model_family = model_family_for(model, family)
                for column in frame.columns:
                    if not str(column).startswith("HybridHigh20+"):
                        continue
                    mae = pd.to_numeric(pd.Series([item.get(column)]), errors="coerce").iloc[0]
                    if pd.isna(mae):
                        continue
                    rows.append(
                        {
                            "case": "",
                            "model_family": model_family,
                            "model": model,
                            "method": normalize_method(column),
                            "set_label": normalize_subset(raw_subset),
                            "raw_subset": raw_subset,
                            "subset_fs": subset_filesystem_name(raw_subset),
                            "triptych_MAE": float(mae),
                            "source_csv": str(path),
                        }
                    )
    result = pd.DataFrame(rows, columns=TRIPTYCH_MODEL_REVIEW_COLUMNS)
    if result.empty:
        return result
    result["case"] = result["source_csv"].map(infer_case_from_triptych_path)
    if case_contains:
        needle = case_contains.lower()
        result = result[result["case"].astype(str).str.lower().str.contains(needle, regex=False)].copy()
    return result.sort_values(["case", "method", "model", "set_label"], kind="stable").reset_index(drop=True)


def infer_case_from_triptych_path(path_value: object) -> str:
    name = Path(clean_text(path_value)).name.lower()
    stem = name
    for suffix in ["_bert_models_mae.csv", "_traditional_models_mae.csv", "_mae.csv"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.replace("ood_", "").replace("_triptych_abcde", "")
    mapping = {
        "al_aluminum_utsmpa": "Al-UTS",
        "hea_hea_elpct": "HEA-El",
        "hea_hea_utsmpa": "HEA-UTS",
        "hea_hea_yspct": "HEA-YS",
        "matbench_steel_yield_strength": "Matbench_Steel-YS",
        "steel_steel_elpct": "Steel-El",
        "steel_steel_utsmpa": "Steel-UTS",
        "steel_steel_yspct": "Steel-YS",
        "ti_titanium_elpct": "Ti-El",
        "ti_titanium_utsmpa": "Ti-UTS",
    }
    return mapping.get(stem, stem)


def build_set_ab_model_summary(master: pd.DataFrame, triptych_review: pd.DataFrame | None = None) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=SET_AB_MODEL_SUMMARY_COLUMNS)
    key_cols = [
        "case",
        "task_key",
        "alloy_family",
        "dataset_name",
        "property",
        "model",
        "model_family",
        "method",
        "set_label",
        "subset",
        "raw_subset",
        "subset_fs",
    ]
    rows = [summarize_model_group(group, keys, key_cols) for keys, group in master.groupby(key_cols, dropna=False, sort=True)]
    result = pd.DataFrame(rows, columns=SET_AB_MODEL_SUMMARY_COLUMNS)
    if triptych_review is not None and not triptych_review.empty:
        triptych = triptych_review.rename(columns={"triptych_MAE": "_triptych_MAE", "source_csv": "_triptych_source_csv"})
        merge_cols = ["case", "model_family", "model", "method", "set_label"]
        result = result.merge(triptych[[*merge_cols, "_triptych_MAE", "_triptych_source_csv"]], on=merge_cols, how="left")
        mask = result["_triptych_MAE"].notna()
        result.loc[mask, "MAE"] = result.loc[mask, "_triptych_MAE"]
        result.loc[mask, "MAE_source"] = "triptych_aligned"
        result.loc[mask, "triptych_source_csv"] = result.loc[mask, "_triptych_source_csv"]
        result = result.drop(columns=["_triptych_MAE", "_triptych_source_csv"])
    return result[SET_AB_MODEL_SUMMARY_COLUMNS].sort_values(["case", "method", "model", "set_label"], kind="stable").reset_index(drop=True)


def build_set_ab_space_summary_long(master: pd.DataFrame, model_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    summary = build_hybrid_summary_long(master)
    if summary.empty or model_summary is None or model_summary.empty:
        return summary
    metric_cols = ["MAE", "RMSE", "R2", "relative_error_mean", "ceiling_bias", "underprediction_ratio", "n_samples"]
    merge_cols = ["case", "task_key", "alloy_family", "dataset_name", "property", "model", "model_family", "method"]
    model_metrics = model_summary.drop(columns=["subset"], errors="ignore").rename(columns={"set_label": "subset"})[[*merge_cols, "subset", *metric_cols]]
    summary = summary.drop(columns=[column for column in metric_cols if column in summary.columns])
    summary = summary.merge(model_metrics, on=[*merge_cols, "subset"], how="left")
    return summary[SUMMARY_COLUMNS].sort_values(["case", "method", "model", "subset", "space"], kind="stable").reset_index(drop=True)


def build_yz_ood_map_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=YZ_OOD_MAP_COLUMNS)
    key_cols = [
        "case",
        "task_key",
        "alloy_family",
        "dataset_name",
        "property",
        "model",
        "model_family",
        "method",
        "subset",
    ]
    work = summary[summary["subset"].isin(["Set A", "Set B"]) & summary["space"].isin(["Y-space", "Z-space"])].copy()
    if work.empty:
        return pd.DataFrame(columns=YZ_OOD_MAP_COLUMNS)
    work["W_mean"] = pd.to_numeric(work["W_mean"], errors="coerce")
    pivot = (
        work.pivot_table(index=key_cols, columns="space", values="W_mean", aggfunc="mean")
        .rename(columns={"Y-space": "Wy_mean", "Z-space": "Wz_mean"})
        .reset_index()
    )
    for column in ["Wy_mean", "Wz_mean"]:
        if column not in pivot.columns:
            pivot[column] = np.nan
    metric_cols = ["aggregation", "MAE", "RMSE", "R2", "relative_error_mean", "n_samples"]
    metrics = work[work["space"].eq("Y-space")][[*key_cols, *metric_cols]].drop_duplicates(subset=key_cols)
    if metrics.empty:
        metrics = work[[*key_cols, *metric_cols]].drop_duplicates(subset=key_cols)
    result = pivot.merge(metrics, on=key_cols, how="left")
    result["subset_fs"] = result["subset"].map(subset_filesystem_name)
    for column in YZ_OOD_MAP_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result[YZ_OOD_MAP_COLUMNS].sort_values(["case", "method", "model", "subset"], kind="stable").reset_index(drop=True)


def build_set_ab_family_best_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame(columns=SET_AB_FAMILY_BEST_COLUMNS)
    rows: list[dict[str, object]] = []
    for keys, group in model_summary.groupby(["case", "model_family", "method", "set_label", "raw_subset", "subset_fs"], dropna=False, sort=True):
        case, family, method, set_label, raw_subset, subset_fs = keys
        ranked = group.copy()
        ranked["MAE"] = pd.to_numeric(ranked["MAE"], errors="coerce")
        ranked = ranked.dropna(subset=["MAE"]).sort_values(["MAE", "model"], kind="stable")
        if ranked.empty:
            continue
        row = ranked.iloc[0]
        family_label = f"{family}-best" if family in {"BERT", "Traditional"} else clean_text(row["model"])
        rows.append(
            {
                "case": case,
                "family_label": family_label,
                "model_family": family,
                "method": method,
                "set_label": set_label,
                "raw_subset": raw_subset,
                "subset_fs": subset_fs,
                "selected_model": row["model"],
                "MAE": row["MAE"],
                "selection_rule": "min_MAE_within_task_method_set_family",
                "source_csv": row.get("triptych_source_csv", ""),
            }
        )
    return pd.DataFrame(rows, columns=SET_AB_FAMILY_BEST_COLUMNS).sort_values(["case", "method", "model_family", "set_label"], kind="stable").reset_index(drop=True)


def build_set_ab_source_audit(model_summary: pd.DataFrame, triptych_review: pd.DataFrame, old_summary: pd.DataFrame | None = None) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame(columns=SET_AB_AUDIT_COLUMNS)
    base_cols = ["case", "model_family", "model", "method", "set_label", "raw_subset", "n_samples", "n_folds", "n_ids", "MAE_sample"]
    audit = model_summary[base_cols].rename(columns={"MAE_sample": "master_MAE_sample"}).copy()
    if triptych_review is not None and not triptych_review.empty:
        trip = triptych_review.rename(columns={"triptych_MAE": "triptych_MAE"})
        audit = audit.merge(
            trip[["case", "model_family", "model", "method", "set_label", "triptych_MAE", "source_csv"]],
            on=["case", "model_family", "model", "method", "set_label"],
            how="left",
        )
    else:
        audit["triptych_MAE"] = np.nan
        audit["source_csv"] = ""
    if old_summary is not None and not old_summary.empty:
        old = old_summary[old_summary["space"].eq("Y-space")][["case", "model_family", "model", "method", "subset", "MAE"]].rename(
            columns={"subset": "set_label", "MAE": "old_summary_MAE"}
        )
        audit = audit.merge(old, on=["case", "model_family", "model", "method", "set_label"], how="left")
    else:
        audit["old_summary_MAE"] = np.nan
    audit["diff_master_triptych"] = pd.to_numeric(audit["master_MAE_sample"], errors="coerce") - pd.to_numeric(audit["triptych_MAE"], errors="coerce")
    audit["diff_old_triptych"] = pd.to_numeric(audit["old_summary_MAE"], errors="coerce") - pd.to_numeric(audit["triptych_MAE"], errors="coerce")
    audit["verdict"] = np.where(audit["triptych_MAE"].isna(), "missing_triptych", np.where(audit["diff_master_triptych"].abs() <= 1e-6, "match", "mismatch"))
    return audit[SET_AB_AUDIT_COLUMNS].sort_values(["case", "method", "model", "set_label"], kind="stable").reset_index(drop=True)


def deduplicate_group_for_subset(group: pd.DataFrame, subset: str) -> tuple[pd.DataFrame, str]:
    return group.copy(), "fold_weighted"


def summarize_one_group(group: pd.DataFrame, keys: tuple[object, ...], key_cols: list[str]) -> dict[str, object]:
    values = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
    subset = clean_text(values.get("subset"))
    work, aggregation = deduplicate_group_for_subset(group, subset)
    w = pd.to_numeric(work["sample_w_contribution"], errors="coerce")
    abs_error = pd.to_numeric(work["abs_error"], errors="coerce")
    rel_error = pd.to_numeric(work["relative_error_pct"], errors="coerce")
    true_values = pd.to_numeric(work.get("true_value", pd.Series(np.nan, index=work.index)), errors="coerce")
    pred_values = pd.to_numeric(work.get("predicted_value", pd.Series(np.nan, index=work.index)), errors="coerce")
    signed = pd.to_numeric(work.get("signed_error", pred_values - true_values), errors="coerce")
    return {
        **values,
        "aggregation": aggregation,
        "W_mean": float(w.mean()) if w.notna().any() else np.nan,
        "W_median": float(w.median()) if w.notna().any() else np.nan,
        "W_p90": float(w.quantile(0.9)) if w.notna().any() else np.nan,
        "MAE": float(abs_error.mean()) if abs_error.notna().any() else np.nan,
        "RMSE": float(np.sqrt(np.nanmean(abs_error.to_numpy(dtype=float) ** 2))) if abs_error.notna().any() else np.nan,
        "R2": r2_score_safe(true_values, pred_values),
        "relative_error_mean": float(rel_error.mean()) if rel_error.notna().any() else np.nan,
        "slope": fit_slope(w, rel_error),
        "spearman": float(w.corr(rel_error, method="spearman")) if w.notna().sum() >= 2 and rel_error.notna().sum() >= 2 else np.nan,
        "ceiling_bias": float(signed.mean()) if signed.notna().any() else np.nan,
        "underprediction_ratio": float((signed < 0).mean()) if signed.notna().any() else np.nan,
        "n_samples": int(len(work)),
    }


def build_hybrid_summary_long(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    work = build_set_ab_master_sample_long(frame) if "set_label" not in frame.columns else frame.copy()
    work["case"] = work.get("case", work.get("task_id", work.get("task_key", "case"))).fillna("").astype(str)
    work["task_key"] = work.get("task_key", work["case"]).fillna("").astype(str)
    work["model"] = work.get("model", "").fillna("").astype(str)
    work["model_family"] = [
        model_family_for(model, family)
        for model, family in zip(work["model"], work.get("model_family", pd.Series("", index=work.index)))
    ]
    work["method"] = work.get("method_short", work.get("method", "")).map(normalize_method)
    work["subset"] = work.get("set_label", work.get("test_set", "")).map(normalize_subset)
    work = work[work["subset"].isin(["Set A", "Set B"])].copy()
    for column in ["alloy_family", "dataset_name", "property", "space"]:
        if column not in work.columns:
            work[column] = ""
    key_cols = [
        "case",
        "task_key",
        "alloy_family",
        "dataset_name",
        "property",
        "model",
        "model_family",
        "method",
        "subset",
        "space",
    ]
    rows = [summarize_one_group(group, keys, key_cols) for keys, group in work.groupby(key_cols, dropna=False, sort=True)]
    result = pd.DataFrame(rows)
    return result[SUMMARY_COLUMNS].sort_values(["case", "method", "model", "subset", "space"], kind="stable").reset_index(drop=True)


def build_ab_delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=DELTA_COLUMNS)
    base_cols = ["case", "task_key", "alloy_family", "dataset_name", "property", "model", "model_family", "method"]
    base = summary[base_cols].drop_duplicates().copy()
    for metric, output_col, space in [
        ("MAE", "delta_MAE", "Y-space"),
        ("W_p90", "delta_X_W", "X-space"),
        ("W_p90", "delta_Y_W", "Y-space"),
        ("W_p90", "delta_Z_W", "Z-space"),
        ("slope", "delta_slope_X", "X-space"),
        ("slope", "delta_slope_Y", "Y-space"),
        ("slope", "delta_slope_Z", "Z-space"),
    ]:
        set_a = summary[(summary["subset"].eq("Set A")) & (summary["space"].eq(space))][[*base_cols, metric]].rename(columns={metric: f"{output_col}_A"})
        set_b = summary[(summary["subset"].eq("Set B")) & (summary["space"].eq(space))][[*base_cols, metric]].rename(columns={metric: f"{output_col}_B"})
        base = base.merge(set_a, on=base_cols, how="left").merge(set_b, on=base_cols, how="left")
        base[output_col] = pd.to_numeric(base[f"{output_col}_A"], errors="coerce") - pd.to_numeric(base[f"{output_col}_B"], errors="coerce")
        base = base.drop(columns=[f"{output_col}_A", f"{output_col}_B"])
    ceiling = summary[(summary["subset"].eq("Set A")) & (summary["space"].eq("Y-space"))][
        [*base_cols, "ceiling_bias", "underprediction_ratio"]
    ].rename(columns={"ceiling_bias": "ceiling_bias_SetA", "underprediction_ratio": "underprediction_ratio_SetA"})
    base = base.merge(ceiling, on=base_cols, how="left")
    return base[DELTA_COLUMNS].sort_values(["case", "method", "model"], kind="stable").reset_index(drop=True)


def build_pure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=PURE_HYBRID_COLUMNS)
    work = frame.copy()
    work["case"] = work.get("task_id", work.get("task_key", "case")).fillna("").astype(str)
    work["task_key"] = work.get("task_key", work["case"]).fillna("").astype(str)
    work["model"] = work.get("model", "").fillna("").astype(str)
    work["model_family"] = [
        model_family_for(model, family)
        for model, family in zip(work["model"], work.get("model_family", pd.Series("", index=work.index)))
    ]
    work["method_group"] = work.get("method_short", work.get("method", "")).map(normalize_method)
    work["split_type"] = "Pure"
    work["subset"] = "Pure " + work["method_group"].astype(str)
    key_cols = ["case", "task_key", "model", "model_family", "method_group", "split_type", "subset", "space"]
    rows: list[dict[str, object]] = []
    for keys, group in work.groupby(key_cols, dropna=False, sort=True):
        values = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        w = pd.to_numeric(group["sample_w_contribution"], errors="coerce")
        abs_error = pd.to_numeric(group["abs_error"], errors="coerce")
        rel_error = pd.to_numeric(group["relative_error_pct"], errors="coerce")
        rows.append(
            {
                **values,
                "W_mean": float(w.mean()) if w.notna().any() else np.nan,
                "W_median": float(w.median()) if w.notna().any() else np.nan,
                "W_p90": float(w.quantile(0.9)) if w.notna().any() else np.nan,
                "MAE": float(abs_error.mean()) if abs_error.notna().any() else np.nan,
                "relative_error_mean": float(rel_error.mean()) if rel_error.notna().any() else np.nan,
                "n_samples": int(group["ID"].nunique()) if "ID" in group.columns else int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=PURE_HYBRID_COLUMNS)


def build_pure_vs_hybrid_summary(summary: pd.DataFrame, pure_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    hybrid = summary.copy()
    if hybrid.empty:
        hybrid_rows = pd.DataFrame(columns=PURE_HYBRID_COLUMNS)
    else:
        hybrid_rows = hybrid.rename(columns={"method": "method_group"})[
            ["case", "task_key", "model", "model_family", "method_group", "subset", "space", "W_mean", "W_median", "W_p90", "MAE", "relative_error_mean", "n_samples"]
        ].copy()
        hybrid_rows["split_type"] = "Hybrid"
        hybrid_rows = hybrid_rows[PURE_HYBRID_COLUMNS]
    pure_rows = build_pure_summary(pure_frame if pure_frame is not None else pd.DataFrame())
    frames = [frame for frame in [pure_rows, hybrid_rows] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=PURE_HYBRID_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)[PURE_HYBRID_COLUMNS]
