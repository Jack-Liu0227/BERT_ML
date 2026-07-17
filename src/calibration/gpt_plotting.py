from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .full_plotting import (
    PROTOCOL_ORDER,
    TIER_ORDER,
    _configure_matplotlib,
    _file_sha256,
    _heatmap,
    _publish_package,
    _save_figure,
    _tree_sha256,
    _validate_output_file,
    validate_full_plot_inputs,
)


METHOD_ORDER = (
    "Traditional-best",
    "BERT-best",
    "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
    "GPT-5.4",
)
METHOD_COLORS = {
    "Traditional-best": "#0072B2",
    "BERT-best": "#D55E00",
    "LLM-Prop": "#009E73",
    "TabPFN-2.5-Plus-Numeric": "#CC79A7",
    "TabPFN-2.5-Plus-Text": "#56B4E9",
    "GPT-5.4": "#000000",
}
METHOD_LABELS = {
    "Traditional-best": "Traditional-best",
    "BERT-best": "BERT-best",
    "LLM-Prop": "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric": "TabPFN Numeric",
    "TabPFN-2.5-Plus-Text": "TabPFN Text",
    "GPT-5.4": "GPT-5.4",
}
TIER_RISK_SCORE = {"High": 1.0, "Medium": 2.0, "Low": 3.0}

COMMON_TASKS = (
    "Al__aluminum__UTS_MPa",
    "HEA__hea__El",
    "HEA__hea__UTS_MPa",
    "Matbench_Steel__matbench_steels_ood__yield_strength",
    "Steel__steel__El",
    "Steel__steel__UTS_MPa",
    "Ti__titanium__El",
    "Ti__titanium__UTS_MPa",
)
EXCLUDED_TASKS = {
    "HEA__hea__YS_MPa": "GPT result is unavailable.",
    "Steel__steel__YS_MPa": "GPT covers only RandomCV and Extrapolation.",
}
TASK_METADATA_TO_ID = {
    ("Al", "aluminum", "UTS(MPa)"): "Al__aluminum__UTS_MPa",
    ("HEA", "hea", "El(%)"): "HEA__hea__El",
    ("HEA", "hea", "UTS(MPa)"): "HEA__hea__UTS_MPa",
    ("HEA", "hea", "YS(MPa)"): "HEA__hea__YS_MPa",
    (
        "Matbench Steel",
        "matbench_steels_ood",
        "yield strength",
    ): "Matbench_Steel__matbench_steels_ood__yield_strength",
    ("Steel", "steel", "El(%)"): "Steel__steel__El",
    ("Steel", "steel", "UTS(MPa)"): "Steel__steel__UTS_MPa",
    ("Steel", "steel", "YS(MPa)"): "Steel__steel__YS_MPa",
    ("Ti", "titanium", "El(%)"): "Ti__titanium__El",
    ("Ti", "titanium", "UTS(MPa)"): "Ti__titanium__UTS_MPa",
}
TASK_ID_ALIASES = {
    "MatbenchSteels__matbench_steels_ood__YS_MPa": (
        "Matbench_Steel__matbench_steels_ood__yield_strength"
    ),
}
EXPECTED_REPRESENTATIVE_COUNTS = {
    "BERT-best": {
        "MatSciBERT": 14,
        "SciBERT": 26,
        "SteelBERT": 16,
    },
    "Traditional-best": {
        "CatBoost": 19,
        "LightGBM": 15,
        "MLP": 8,
        "RF": 2,
        "XGB": 12,
    },
}
DEFAULT_SETTINGS = {
    "output_dir": "figures/final_with_gpt",
    "formats": ["png", "pdf", "svg"],
    "dpi": 300,
    "bootstrap_iterations": 10000,
    "bootstrap_seed": 20260715,
    "include_task_appendix": True,
    "prediction_inventory": (
        "output/w_error_relationship_full/OOD/csv/prediction_source_inventory.csv"
    ),
    "combined_summary": (
        "output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv"
    ),
}


def normalize_task_id(value: object) -> str:
    task_id = str(value)
    return TASK_ID_ALIASES.get(task_id, task_id)


def _canonical_summary_task_id(row: pd.Series) -> str | None:
    key = (
        str(row.get("alloy_family", "")).strip(),
        str(row.get("dataset_name", "")).strip(),
        str(row.get("property", "")).strip(),
    )
    return TASK_METADATA_TO_ID.get(key)


def _normalize_protocol(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if "random" in lowered:
        return "RandomCV"
    if "loco" in lowered:
        return "LOCO"
    if "extrapolation" in lowered:
        return "Extrapolation"
    lookup = {protocol.lower(): protocol for protocol in PROTOCOL_ORDER}
    return lookup.get(lowered, text)


def select_representative_models(
    summary: pd.DataFrame,
    *,
    common_tasks: Sequence[str] = COMMON_TASKS,
    protocols: Sequence[str] = PROTOCOL_ORDER,
    strict: bool = True,
) -> pd.DataFrame:
    required = {
        "alloy_family",
        "dataset_name",
        "property",
        "ood_method",
        "model_family",
        "model",
        "plot_test_mae",
        "summary_test_mae_std",
        "summary_test_r2",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Combined OOD summary is missing columns: {missing}")

    working = summary[summary["model_family"].astype(str).isin({"Traditional", "BERT"})].copy()
    working["task_id"] = working.apply(_canonical_summary_task_id, axis=1)
    working["protocol"] = working["ood_method"].map(_normalize_protocol)
    working = working[
        working["task_id"].isin(set(common_tasks))
        & working["protocol"].isin(set(protocols))
    ].copy()
    for column in ("plot_test_mae", "summary_test_mae_std", "summary_test_r2"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["_mae"] = working["plot_test_mae"].fillna(np.inf)
    working["_mae_std"] = working["summary_test_mae_std"].fillna(np.inf)
    working["_r2"] = working["summary_test_r2"].fillna(-np.inf)
    working = working.sort_values(
        ["task_id", "protocol", "model_family", "_mae", "_mae_std", "_r2", "model"],
        ascending=[True, True, True, True, True, False, True],
        kind="mergesort",
    )
    selected = working.groupby(
        ["task_id", "protocol", "model_family"],
        as_index=False,
        sort=False,
    ).head(1).copy()
    selected["method"] = selected["model_family"].map(
        {"Traditional": "Traditional-best", "BERT": "BERT-best"}
    )
    selected = selected.rename(columns={"model": "selected_model"})
    columns = [
        "task_id",
        "protocol",
        "method",
        "model_family",
        "selected_model",
        "plot_test_mae",
        "summary_test_mae_std",
        "summary_test_r2",
        "alloy_family",
        "dataset_name",
        "property",
    ]
    selected = selected[columns].sort_values(
        ["method", "task_id", "protocol"], kind="stable"
    ).reset_index(drop=True)

    if strict:
        expected_cases = len(tuple(common_tasks)) * len(tuple(protocols))
        observed = selected.groupby("method").size().to_dict()
        if observed != {"BERT-best": expected_cases, "Traditional-best": expected_cases}:
            raise ValueError(
                f"Representative model coverage is incomplete: expected={expected_cases} observed={observed}"
            )
        compositions = {
            method: selected[selected["method"].eq(method)]["selected_model"]
            .value_counts()
            .sort_index()
            .to_dict()
            for method in ("BERT-best", "Traditional-best")
        }
        expected_compositions = {
            method: dict(sorted(counts.items()))
            for method, counts in EXPECTED_REPRESENTATIVE_COUNTS.items()
        }
        if compositions != expected_compositions:
            raise ValueError(
                "Representative model composition changed: "
                f"expected={expected_compositions} observed={compositions}"
            )
    return selected


def _coalesce_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _normalize_confidence_tier(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"high", "h", "confident", "low_uncertainty"}:
        return "High"
    if normalized in {"medium", "med", "m", "moderate"}:
        return "Medium"
    if normalized in {"low", "l", "uncertain", "high_uncertainty"}:
        return "Low"
    return None


def standardize_gpt_source(entry: Mapping[str, object]) -> pd.DataFrame:
    path = Path(str(entry["path"])).resolve()
    frame = pd.read_csv(path)
    id_column = _coalesce_column(
        frame,
        ("sample_id", "ID", "id", "sample_index", "__row_id__", "__test_index__"),
    )
    true_column = _coalesce_column(frame, ("y_true", "true_value", "actual", "actual_value"))
    pred_column = _coalesce_column(frame, ("y_pred", "predicted_value", "prediction", "pred"))
    if true_column is None:
        candidates = [column for column in frame.columns if str(column).lower().endswith("_true")]
        true_column = str(candidates[0]) if candidates else None
    if pred_column is None:
        candidates = [
            column
            for column in frame.columns
            if str(column).lower().endswith(("_predicted", "_pred"))
        ]
        pred_column = str(candidates[0]) if candidates else None
    confidence_column = _coalesce_column(
        frame, ("confidence_tier", "confidence", "semantic_confidence")
    )
    if true_column is None or pred_column is None or confidence_column is None:
        raise ValueError(f"GPT source lacks true, predicted, or confidence columns: {path}")

    output = pd.DataFrame(index=frame.index)
    output["task_id"] = normalize_task_id(entry.get("task_id"))
    output["protocol"] = _normalize_protocol(entry.get("protocol"))
    output["method"] = "GPT-5.4"
    output["model_family"] = "GPT"
    output["selected_model"] = "GPT-5.4"
    output["sample_id"] = frame[id_column].astype(str) if id_column else frame.index.astype(str)
    output["y_true"] = pd.to_numeric(frame[true_column], errors="coerce")
    output["y_pred"] = pd.to_numeric(frame[pred_column], errors="coerce")
    output["abs_error"] = (output["y_pred"] - output["y_true"]).abs()
    output["confidence_tier"] = frame[confidence_column].map(_normalize_confidence_tier)
    output["confidence_source"] = "semantic"
    output["uncertainty_score"] = np.nan
    output["member_count"] = 1
    output["uncertainty_method"] = "semantic_confidence_tier"
    output["source_path"] = str(path)
    output = output.dropna(subset=["y_true", "y_pred", "abs_error", "confidence_tier"])
    return output.reset_index(drop=True)


def load_gpt_samples_from_inventory(
    inventory_path: str | Path,
    *,
    project_root: str | Path,
    common_tasks: Sequence[str] = COMMON_TASKS,
    protocols: Sequence[str] = PROTOCOL_ORDER,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[Path]]:
    inventory_file = Path(inventory_path).resolve()
    if not inventory_file.exists():
        raise FileNotFoundError(inventory_file)
    inventory = pd.read_csv(inventory_file)
    required = {"model", "prediction_file", "method"}
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValueError(f"GPT prediction inventory is missing columns: {missing}")
    task_column = "task_id" if "task_id" in inventory.columns else "task_key"
    if task_column not in inventory.columns:
        raise ValueError("GPT prediction inventory needs task_id or task_key.")

    root = Path(project_root).resolve()
    gpt_inventory = inventory[
        inventory["model"].astype(str).str.contains("GPT-5.4", case=False, na=False)
    ].copy()
    entries: list[dict[str, object]] = []
    for row in gpt_inventory.itertuples(index=False):
        raw_path = Path(str(getattr(row, "prediction_file")))
        path = raw_path if raw_path.is_absolute() else (root / raw_path).resolve()
        entries.append(
            {
                "path": str(path),
                "task_id": normalize_task_id(getattr(row, task_column)),
                "protocol": _normalize_protocol(getattr(row, "method")),
                "model": "GPT-5.4",
            }
        )
    unique_entries = {
        (str(entry["path"]), str(entry["task_id"]), str(entry["protocol"])): entry
        for entry in entries
    }
    all_entries = list(unique_entries.values())
    included_entries = [
        entry
        for entry in all_entries
        if entry["task_id"] in set(common_tasks) and entry["protocol"] in set(protocols)
    ]
    missing_paths = [Path(str(entry["path"])) for entry in included_entries if not Path(str(entry["path"])).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"GPT prediction files are missing: {missing_paths[:5]}")
    frames = [standardize_gpt_source(entry) for entry in included_entries]
    if not frames:
        raise ValueError("No GPT-5.4 prediction sources matched the common task/protocol scope.")
    sample = pd.concat(frames, ignore_index=True)
    sample = sample.drop_duplicates(
        ["source_path", "sample_id", "task_id", "protocol", "method"],
        keep="first",
    ).sort_values(
        ["task_id", "protocol", "source_path", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    sources = sorted({Path(str(entry["path"])).resolve() for entry in included_entries})

    if strict:
        if len(all_entries) != 126:
            raise ValueError(f"GPT source discovery count changed: expected=126 observed={len(all_entries)}")
        if len(sources) != 120:
            raise ValueError(f"Included GPT source count changed: expected=120 observed={len(sources)}")
        if len(sample) != 9231:
            raise ValueError(f"GPT sample count changed: expected=9231 observed={len(sample)}")
        case_count = sample[["task_id", "protocol"]].drop_duplicates().shape[0]
        if case_count != 56:
            raise ValueError(f"GPT task/protocol coverage changed: expected=56 observed={case_count}")
    return sample, sources


def _all_gpt_source_paths(
    inventory_path: str | Path,
    *,
    project_root: str | Path,
) -> list[Path]:
    inventory = pd.read_csv(Path(inventory_path).resolve())
    root = Path(project_root).resolve()
    gpt = inventory[
        inventory["model"].astype(str).str.contains("GPT-5.4", case=False, na=False)
    ]
    paths = []
    for value in gpt["prediction_file"].dropna().astype(str):
        raw = Path(value)
        paths.append(raw.resolve() if raw.is_absolute() else (root / raw).resolve())
    return sorted(set(paths))


def build_non_gpt_selected_sample(
    sample: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    common_tasks: Sequence[str] = COMMON_TASKS,
    protocols: Sequence[str] = PROTOCOL_ORDER,
    strict: bool = True,
) -> pd.DataFrame:
    required_sample = {
        "task_id",
        "protocol",
        "model_family",
        "model",
        "sample_id",
        "y_true",
        "y_pred",
        "abs_error",
        "confidence_tier",
    }
    missing = sorted(required_sample.difference(sample.columns))
    if missing:
        raise ValueError(f"Non-GPT calibration sample is missing columns: {missing}")
    required_selection = {"task_id", "protocol", "method", "selected_model"}
    missing_selection = sorted(required_selection.difference(selection.columns))
    if missing_selection:
        raise ValueError(f"Representative selection is missing columns: {missing_selection}")

    working = sample[
        sample["task_id"].astype(str).isin(set(common_tasks))
        & sample["protocol"].astype(str).isin(set(protocols))
    ].copy()
    family = working[working["model_family"].astype(str).isin({"Traditional", "BERT"})].merge(
        selection[["task_id", "protocol", "method", "selected_model"]],
        left_on=["task_id", "protocol", "model"],
        right_on=["task_id", "protocol", "selected_model"],
        how="inner",
        validate="many_to_one",
    )
    fixed_map = {
        "LLM-Prop": "LLM-Prop",
        "TabPFN-2.5-Plus-Numeric": "TabPFN-2.5-Plus-Numeric",
        "TabPFN-2.5-Plus-Text": "TabPFN-2.5-Plus-Text",
    }
    fixed = working[working["model"].astype(str).isin(fixed_map)].copy()
    fixed["method"] = fixed["model"].map(fixed_map)
    fixed["selected_model"] = fixed["model"].astype(str)
    integrated = pd.concat([family, fixed], ignore_index=True, sort=False)
    integrated["confidence_source"] = "uncertainty_tertile"
    integrated["method"] = pd.Categorical(
        integrated["method"], categories=METHOD_ORDER[:-1], ordered=True
    )
    integrated = integrated.sort_values(
        ["method", "task_id", "protocol", "sample_id"], kind="stable"
    ).astype({"method": "object"}).reset_index(drop=True)

    if strict:
        expected_methods = METHOD_ORDER[:-1]
        case_counts = (
            integrated[["method", "task_id", "protocol"]]
            .drop_duplicates()
            .groupby("method")
            .size()
            .to_dict()
        )
        sample_counts = integrated.groupby("method").size().to_dict()
        expected_case_counts = {method: 56 for method in expected_methods}
        expected_sample_counts = {method: 9231 for method in expected_methods}
        if case_counts != expected_case_counts:
            raise ValueError(
                f"Selected non-GPT case coverage changed: expected={expected_case_counts} observed={case_counts}"
            )
        if sample_counts != expected_sample_counts:
            raise ValueError(
                f"Selected non-GPT sample counts changed: expected={expected_sample_counts} observed={sample_counts}"
            )
    return integrated


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    numeric = np.asarray(list(values), dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(numeric.mean())
    if numeric.size == 1 or iterations <= 1:
        return mean, mean, mean
    sampled = rng.choice(numeric, size=(int(iterations), numeric.size), replace=True)
    bootstrap_means = sampled.mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _tier_spearman(group: pd.DataFrame) -> tuple[float, str]:
    paired = pd.DataFrame(
        {
            "risk": group["confidence_tier"].map(TIER_RISK_SCORE),
            "error": pd.to_numeric(group["abs_error"], errors="coerce"),
        }
    ).dropna()
    if len(paired) < 2:
        return np.nan, "fewer_than_two_samples"
    if paired["risk"].nunique() < 2:
        return np.nan, "fewer_than_two_tiers"
    if paired["error"].nunique() < 2:
        return np.nan, "constant_error"
    return float(paired["risk"].corr(paired["error"], method="spearman")), "valid"


def prepare_integrated_tables(
    sample: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260715,
    methods: Sequence[str] = METHOD_ORDER,
    protocols: Sequence[str] = PROTOCOL_ORDER,
) -> dict[str, pd.DataFrame]:
    required = {
        "task_id",
        "protocol",
        "method",
        "selected_model",
        "sample_id",
        "abs_error",
        "confidence_tier",
        "confidence_source",
    }
    missing = sorted(required.difference(sample.columns))
    if missing:
        raise ValueError(f"Integrated sample table is missing columns: {missing}")
    working = sample.copy()
    working["abs_error"] = pd.to_numeric(working["abs_error"], errors="raise")
    keys = ["task_id", "protocol", "method"]
    working["overall_mae"] = working.groupby(keys, dropna=False)["abs_error"].transform("mean")

    tier_rows: list[dict[str, object]] = []
    for group_keys, group in working.groupby(keys + ["confidence_tier"], dropna=False):
        overall_mae = float(group["overall_mae"].iloc[0])
        mae = float(group["abs_error"].mean())
        tier_rows.append(
            {
                "task_id": group_keys[0],
                "protocol": group_keys[1],
                "method": group_keys[2],
                "selected_model": str(group["selected_model"].iloc[0]),
                "confidence_tier": group_keys[3],
                "n": int(len(group)),
                "mae": mae,
                "overall_mae": overall_mae,
                "normalized_mae": mae / overall_mae if overall_mae > 0 else np.nan,
            }
        )
    tier_summary = pd.DataFrame(tier_rows)
    tier_summary["confidence_tier"] = pd.Categorical(
        tier_summary["confidence_tier"], categories=TIER_ORDER, ordered=True
    )
    tier_summary = tier_summary.sort_values(
        ["task_id", "protocol", "method", "confidence_tier"], kind="stable"
    ).astype({"confidence_tier": "object"}).reset_index(drop=True)

    pivot = tier_summary.pivot_table(
        index=keys,
        columns="confidence_tier",
        values="normalized_mae",
        aggfunc="first",
        observed=False,
    ).reset_index()
    for tier in TIER_ORDER:
        if tier not in pivot.columns:
            pivot[tier] = np.nan

    task_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for group_keys, group in working.groupby(keys, sort=False, dropna=False):
        tier_values = pivot[
            pivot["task_id"].astype(str).eq(str(group_keys[0]))
            & pivot["protocol"].astype(str).eq(str(group_keys[1]))
            & pivot["method"].astype(str).eq(str(group_keys[2]))
        ].iloc[0]
        high = float(tier_values["High"]) if pd.notna(tier_values["High"]) else np.nan
        medium = float(tier_values["Medium"]) if pd.notna(tier_values["Medium"]) else np.nan
        low = float(tier_values["Low"]) if pd.notna(tier_values["Low"]) else np.nan
        discrimination = low - high if np.isfinite(high) and np.isfinite(low) else np.nan
        spearman, reason = _tier_spearman(group)
        present_tiers = set(group["confidence_tier"].dropna().astype(str))
        row = {
            "task_id": group_keys[0],
            "protocol": group_keys[1],
            "method": group_keys[2],
            "selected_model": str(group["selected_model"].iloc[0]),
            "High": high,
            "Medium": medium,
            "Low": low,
            "discrimination": discrimination,
            "tier_spearman_error": spearman,
            "sample_n": int(len(group)),
            "confidence_source": str(group["confidence_source"].iloc[0]),
        }
        task_rows.append(row)
        coverage_rows.append(
            {
                **row,
                "record_type": "case",
                "tier_count": len(present_tiers),
                "complete_three_tiers": set(TIER_ORDER).issubset(present_tiers),
                "has_high_low_pair": {"High", "Low"}.issubset(present_tiers),
                "tier_spearman_valid": np.isfinite(spearman),
                "tier_spearman_validity_reason": reason,
                "coverage_status": "included",
            }
        )
    task_summary = pd.DataFrame(task_rows).sort_values(keys, kind="stable").reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows).sort_values(keys, kind="stable").reset_index(drop=True)

    protocol_rows: list[dict[str, object]] = []
    for (method, protocol), group in task_summary.groupby(
        ["method", "protocol"], sort=False, dropna=False
    ):
        discrimination = pd.to_numeric(group["discrimination"], errors="coerce")
        spearman = pd.to_numeric(group["tier_spearman_error"], errors="coerce")
        protocol_rows.append(
            {
                "method": method,
                "protocol": protocol,
                "discrimination_mean": float(discrimination.mean())
                if discrimination.notna().any()
                else np.nan,
                "discrimination_valid_count": int(discrimination.notna().sum()),
                "discrimination_total_count": int(len(discrimination)),
                "tier_spearman_mean": float(spearman.mean()) if spearman.notna().any() else np.nan,
                "tier_spearman_valid_count": int(spearman.notna().sum()),
                "tier_spearman_total_count": int(len(spearman)),
                "tier_spearman_valid_percent": 100.0 * float(spearman.notna().mean()),
            }
        )
    method_protocol = pd.DataFrame(protocol_rows)

    rng = np.random.default_rng(int(bootstrap_seed))
    method_tier_rows: list[dict[str, object]] = []
    groupings: list[tuple[str, pd.DataFrame]] = [("All", tier_summary)]
    groupings.extend(
        (protocol, tier_summary[tier_summary["protocol"].astype(str).eq(protocol)])
        for protocol in protocols
    )
    for protocol, frame in groupings:
        for method in methods:
            method_frame = frame[frame["method"].astype(str).eq(method)]
            for tier in TIER_ORDER:
                values = pd.to_numeric(
                    method_frame.loc[
                        method_frame["confidence_tier"].astype(str).eq(tier),
                        "normalized_mae",
                    ],
                    errors="coerce",
                )
                mean, low, high = _bootstrap_mean_ci(
                    values,
                    rng=rng,
                    iterations=int(bootstrap_iterations),
                )
                method_tier_rows.append(
                    {
                        "protocol": protocol,
                        "method": method,
                        "confidence_tier": tier,
                        "normalized_mae_mean": mean,
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "observation_count": int(values.notna().sum()),
                    }
                )
    method_tier = pd.DataFrame(method_tier_rows)
    validity = coverage[
        [
            "task_id",
            "protocol",
            "method",
            "selected_model",
            "tier_spearman_error",
            "tier_spearman_valid",
            "tier_spearman_validity_reason",
            "sample_n",
            "confidence_source",
        ]
    ].copy()
    return {
        "method_tier_summary": method_tier,
        "method_protocol_summary": method_protocol,
        "task_protocol_method_summary": task_summary,
        "tier_spearman_validity": validity,
        "coverage_audit": coverage,
        "tier_detail": tier_summary,
    }


def _close_and_return(
    figure: Any,
    base_path: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    paths = _save_figure(figure, base_path, formats=formats, dpi=dpi)
    plt.close(figure)
    return paths


def _method_row_labels() -> dict[str, str]:
    return {method: METHOD_LABELS[method] for method in METHOD_ORDER}


def _plot_main(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    method_tier = tables["method_tier_summary"]
    method_protocol = tables["method_protocol_summary"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.35), constrained_layout=True)
    pooled = method_tier[method_tier["protocol"].astype(str).eq("All")]
    x = np.arange(len(TIER_ORDER))
    for method in METHOD_ORDER:
        group = pooled[pooled["method"].astype(str).eq(method)].set_index(
            "confidence_tier"
        ).reindex(TIER_ORDER)
        mean = group["normalized_mae_mean"].to_numpy(float)
        low = group["bootstrap_ci95_low"].to_numpy(float)
        high = group["bootstrap_ci95_high"].to_numpy(float)
        axes[0, 0].plot(
            x,
            mean,
            marker="o",
            linewidth=1.25,
            markersize=3.5,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[0, 0].fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.12)
    axes[0, 0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, TIER_ORDER)
    axes[0, 0].set_ylabel("Normalized MAE")
    axes[0, 0].set_title("Confidence-tier error discrimination")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=5.4)
    axes[0, 0].grid(axis="y", alpha=0.22)

    discrimination_annotations = {
        (str(row.method), str(row.protocol)): (
            "NA\n0/0"
            if not np.isfinite(row.discrimination_mean)
            else f"{row.discrimination_mean:.2f}\n{int(row.discrimination_valid_count)}/{int(row.discrimination_total_count)}"
        )
        for row in method_protocol.itertuples()
    }
    discrimination = method_protocol.rename(columns={"method": "row", "protocol": "column"})
    image = _heatmap(
        axes[0, 1],
        discrimination,
        row_order=METHOD_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="discrimination_mean",
        vmin=-1.75,
        vmax=1.75,
        cmap_name="RdBu_r",
        title="Low-tier minus high-tier normalized MAE",
        annotation=discrimination_annotations,
        row_labels=_method_row_labels(),
    )
    fig.colorbar(image, ax=axes[0, 1], fraction=0.045, pad=0.02)

    spearman_annotations = {
        (str(row.method), str(row.protocol)): (
            "NA\n0/0"
            if not np.isfinite(row.tier_spearman_mean)
            else f"{row.tier_spearman_mean:.2f}\n{int(row.tier_spearman_valid_count)}/{int(row.tier_spearman_total_count)}"
        )
        for row in method_protocol.itertuples()
    }
    spearman = method_protocol.rename(columns={"method": "row", "protocol": "column"})
    image = _heatmap(
        axes[1, 0],
        spearman,
        row_order=METHOD_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="tier_spearman_mean",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        title="Mean Spearman tier-risk-error correlation",
        annotation=spearman_annotations,
        row_labels=_method_row_labels(),
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.045, pad=0.02)

    validity_annotations = {
        (str(row.method), str(row.protocol)): (
            f"{row.tier_spearman_valid_percent:.0f}%\n"
            f"{int(row.tier_spearman_valid_count)}/{int(row.tier_spearman_total_count)}"
        )
        for row in method_protocol.itertuples()
    }
    validity = method_protocol.rename(columns={"method": "row", "protocol": "column"})
    image = _heatmap(
        axes[1, 1],
        validity,
        row_order=METHOD_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="tier_spearman_valid_percent",
        vmin=0.0,
        vmax=100.0,
        cmap_name="Blues",
        title="Valid tier Spearman coverage",
        annotation=validity_annotations,
        row_labels=_method_row_labels(),
    )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.02)
    for label, ax in zip("ABCD", axes.ravel()):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    return _close_and_return(fig, output_dir / "calibration_main", formats, dpi)


def _plot_by_protocol(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    summary = tables["method_tier_summary"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.4), sharex=True, sharey=True)
    x = np.arange(len(TIER_ORDER))
    handles: list[Any] = []
    labels: list[str] = []
    for ax, protocol in zip(axes.ravel(), PROTOCOL_ORDER):
        panel = summary[summary["protocol"].astype(str).eq(protocol)]
        for method in METHOD_ORDER:
            group = panel[panel["method"].astype(str).eq(method)].set_index(
                "confidence_tier"
            ).reindex(TIER_ORDER)
            line = ax.plot(
                x,
                group["normalized_mae_mean"].to_numpy(float),
                marker="o",
                markersize=2.6,
                linewidth=0.9,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )[0]
            if method not in labels:
                handles.append(line)
                labels.append(method)
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.6)
        ax.set_title(protocol)
        ax.set_xticks(x, TIER_ORDER)
        ax.grid(axis="y", alpha=0.18)
    axes[1, 3].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized MAE")
    fig.legend(
        handles,
        [METHOD_LABELS[label] for label in labels],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=5.6,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return _close_and_return(fig, output_dir / "calibration_by_protocol", formats, dpi)


def _plot_by_method(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    summary = tables["method_protocol_summary"].rename(
        columns={"method": "row", "protocol": "column"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), constrained_layout=True)
    discrimination_annotations = {
        (str(row.method), str(row.protocol)): (
            "NA"
            if not np.isfinite(row.discrimination_mean)
            else f"{row.discrimination_mean:.2f}\n{int(row.discrimination_valid_count)}/{int(row.discrimination_total_count)}"
        )
        for row in tables["method_protocol_summary"].itertuples()
    }
    image = _heatmap(
        axes[0],
        summary,
        row_order=METHOD_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="discrimination_mean",
        vmin=-1.75,
        vmax=1.75,
        cmap_name="RdBu_r",
        title="Confidence discrimination by method",
        annotation=discrimination_annotations,
        row_labels=_method_row_labels(),
    )
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    spearman_annotations = {
        (str(row.method), str(row.protocol)): (
            "NA"
            if not np.isfinite(row.tier_spearman_mean)
            else f"{row.tier_spearman_mean:.2f}\n{int(row.tier_spearman_valid_count)}/{int(row.tier_spearman_total_count)}"
        )
        for row in tables["method_protocol_summary"].itertuples()
    }
    image = _heatmap(
        axes[1],
        summary,
        row_order=METHOD_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="tier_spearman_mean",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        title="Tier-risk-error correlation by method",
        annotation=spearman_annotations,
        row_labels=_method_row_labels(),
    )
    fig.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
    for label, ax in zip("AB", axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    return _close_and_return(fig, output_dir / "calibration_by_method", formats, dpi)


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "item"


def _plot_task_appendix(
    tier_detail: pd.DataFrame,
    output_root: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    from matplotlib import pyplot as plt

    output_paths: list[Path] = []
    task_dirs: dict[str, str] = {}
    markers = ("o", "s", "^", "D", "v", "P")
    x = np.arange(len(TIER_ORDER))
    for task_id in COMMON_TASKS:
        task = tier_detail[tier_detail["task_id"].astype(str).eq(task_id)]
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.8), sharex=True, sharey=True)
        handles: list[Any] = []
        labels: list[str] = []
        for ax, protocol in zip(axes.ravel(), PROTOCOL_ORDER):
            panel = task[task["protocol"].astype(str).eq(protocol)]
            for index, method in enumerate(METHOD_ORDER):
                group = panel[panel["method"].astype(str).eq(method)].copy()
                if group.empty:
                    continue
                group["confidence_tier"] = pd.Categorical(
                    group["confidence_tier"], categories=TIER_ORDER, ordered=True
                )
                group = group.set_index("confidence_tier").reindex(TIER_ORDER)
                line = ax.plot(
                    x,
                    group["normalized_mae"].to_numpy(float),
                    color=METHOD_COLORS[method],
                    marker=markers[index],
                    linewidth=0.85,
                    markersize=2.6,
                    alpha=0.9,
                    label=METHOD_LABELS[method],
                )[0]
                if method not in labels:
                    handles.append(line)
                    labels.append(method)
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.6)
            ax.set_title(protocol)
            ax.set_xticks(x, TIER_ORDER)
            ax.grid(axis="y", alpha=0.18)
        axes[1, 3].axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("Normalized MAE")
        fig.suptitle(task_id, y=0.99, fontsize=9)
        fig.legend(
            handles,
            [METHOD_LABELS[label] for label in labels],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=5.4,
        )
        fig.tight_layout(rect=(0, 0.105, 1, 0.965))
        task_dir = output_root / _safe_component(task_id)
        task_dirs[task_id] = str(Path("by_task") / _safe_component(task_id))
        output_paths.extend(
            _close_and_return(
                fig,
                task_dir / "calibration_by_protocol",
                formats,
                dpi,
            )
        )
    return output_paths, task_dirs


def _write_readme(path: Path) -> None:
    path.write_text(
        """# GPT-Integrated Calibration Figure Package

This package compares six method series on eight tasks with complete seven-protocol coverage.

- Traditional-best selects one of CatBoost, LightGBM, MLP, RF, or XGB for every task and protocol using the established paper rule: lowest OOD-test MAE, then lowest MAE standard deviation, highest R2, and model name.
- BERT-best selects one of MatSciBERT, SciBERT, or SteelBERT with the same rule.
- Traditional uncertainty is the sample standard deviation of ten ensemble members with seeds 42-51.
- BERT uncertainty is the sample standard deviation of thirty MC-dropout passes with seeds 42-71.
- LLM-Prop and both TabPFN 2.5 Plus variants are shown separately.
- GPT-5.4 uses its original semantic High, Medium, and Low confidence labels.

Normalized MAE is the tier MAE divided by the overall MAE for the same task, protocol, and method. Confidence discrimination is Low-tier normalized MAE minus High-tier normalized MAE. Spearman correlation uses the shared tier-risk mapping High=1, Medium=2, Low=3 for every method.

Missing GPT tiers are not imputed. Curves use available tier cases, discrimination requires paired High and Low tiers, and correlation requires at least two distinct tiers. Exact counts are stored in `data/coverage_audit.csv`.

HEA YS is excluded because GPT is unavailable. Steel YS is excluded because GPT covers only RandomCV and Extrapolation. The verified non-GPT aggregate and the original `figures/final` package are not modified.
""",
        encoding="ascii",
    )


def _resolve_setting_path(value: object, project_root: Path) -> Path:
    raw = Path(str(value))
    return raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()


def _append_excluded_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    result = coverage.copy()
    for task_id, reason in EXCLUDED_TASKS.items():
        row: dict[str, object] = {}
        for column in result.columns:
            if pd.api.types.is_bool_dtype(result[column].dtype):
                row[column] = False
            elif pd.api.types.is_numeric_dtype(result[column].dtype):
                row[column] = np.nan
            else:
                row[column] = ""
        row.update(
            {
                "task_id": task_id,
                "protocol": "All",
                "method": "All",
                "selected_model": "",
                "record_type": "excluded_task",
                "coverage_status": reason,
                "tier_spearman_validity_reason": "excluded_from_common_task_scope",
            }
        )
        result.loc[len(result)] = row
    return result


def _paths_hash(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in sorted(set(paths))}


def build_gpt_calibration_figures(
    output_root: str | Path,
    *,
    project_root: str | Path,
    scope: str = "all",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scope != "all":
        raise ValueError("GPT-integrated calibration plotting supports only scope=all.")
    validated = validate_full_plot_inputs(output_root, scope=scope)
    output = Path(validated["output_root"])
    project = Path(project_root).resolve()
    options = {**DEFAULT_SETTINGS, **dict(settings or {})}
    formats = tuple(str(value).lower() for value in options["formats"])
    if not formats or any(value not in {"png", "pdf", "svg"} for value in formats):
        raise ValueError(f"Unsupported GPT plot formats: {formats}")
    dpi = int(options["dpi"])
    bootstrap_iterations = int(options["bootstrap_iterations"])
    bootstrap_seed = int(options["bootstrap_seed"])
    include_task_appendix = bool(options["include_task_appendix"])
    if dpi < 300:
        raise ValueError("GPT-integrated PNG delivery requires at least 300 DPI.")
    if bootstrap_iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")

    inventory_path = _resolve_setting_path(options["prediction_inventory"], project)
    summary_path = _resolve_setting_path(options["combined_summary"], project)
    for path in (inventory_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = pd.read_csv(summary_path)
    selection = select_representative_models(summary)
    non_gpt_source = pd.read_csv(validated["paths"]["sample"])
    non_gpt_sample = build_non_gpt_selected_sample(non_gpt_source, selection)
    gpt_sample, included_gpt_sources = load_gpt_samples_from_inventory(
        inventory_path,
        project_root=project,
    )
    all_gpt_sources = _all_gpt_source_paths(inventory_path, project_root=project)
    if len(all_gpt_sources) != 126 or any(not path.is_file() for path in all_gpt_sources):
        raise ValueError("The complete GPT source inventory must contain 126 existing files.")
    integrated = pd.concat([non_gpt_sample, gpt_sample], ignore_index=True, sort=False)
    integrated["method"] = pd.Categorical(
        integrated["method"], categories=METHOD_ORDER, ordered=True
    )
    integrated = integrated.sort_values(
        ["method", "task_id", "protocol", "sample_id"], kind="stable"
    ).astype({"method": "object"}).reset_index(drop=True)
    if len(integrated) != 55386:
        raise ValueError(f"Integrated sample count changed: expected=55386 observed={len(integrated)}")
    expected_method_samples = {method: 9231 for method in METHOD_ORDER}
    observed_method_samples = integrated.groupby("method").size().to_dict()
    if observed_method_samples != expected_method_samples:
        raise ValueError(
            f"Integrated method sample counts changed: expected={expected_method_samples} observed={observed_method_samples}"
        )

    tables = prepare_integrated_tables(
        integrated,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    expected_table_counts = {
        "tier_detail": 989,
        "task_protocol_method_summary": 336,
        "method_protocol_summary": 42,
        "tier_spearman_validity": 336,
        "method_tier_summary": 144,
    }
    observed_table_counts = {name: len(tables[name]) for name in expected_table_counts}
    if observed_table_counts != expected_table_counts:
        raise ValueError(
            f"Integrated summary counts changed: expected={expected_table_counts} observed={observed_table_counts}"
        )
    gpt_coverage = tables["coverage_audit"][
        tables["coverage_audit"]["method"].astype(str).eq("GPT-5.4")
    ]
    if int(gpt_coverage["complete_three_tiers"].sum()) != 39:
        raise ValueError("GPT complete three-tier case count changed.")
    if int(gpt_coverage["has_high_low_pair"].sum()) != 40:
        raise ValueError("GPT paired High/Low case count changed.")
    tables["coverage_audit"] = _append_excluded_coverage(tables["coverage_audit"])

    manifest = validated["root_manifest"]
    split_root = Path(str(manifest["split_root"])).resolve()
    protected_paths = {
        "aggregate": output / "aggregate",
        "frozen_v1": validated["paths"]["frozen"],
        "canonical_splits": split_root,
        "failed_attempts": output / "failed_attempts",
        "existing_final": output / "figures" / "final",
    }
    protected_before = {
        name: _file_sha256(path) if Path(path).is_file() else _tree_sha256(path)
        for name, path in protected_paths.items()
    }
    source_files = [inventory_path, summary_path, *all_gpt_sources]
    source_hashes_before = _paths_hash(source_files)

    _configure_matplotlib()
    figures_root = output / "figures"
    final_dir = output / str(options["output_dir"])
    if final_dir.parent != figures_root:
        raise ValueError(f"GPT plot output must be a direct figures child: {final_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ready_dir = figures_root / f".final_with_gpt.ready_{stamp}"
    if ready_dir.exists():
        raise FileExistsError(ready_dir)
    ready_dir.mkdir(parents=True)
    try:
        data_dir = ready_dir / "data"
        overview_dir = ready_dir / "overview"
        by_task_dir = ready_dir / "by_task"
        for directory in (data_dir, overview_dir, by_task_dir):
            directory.mkdir(parents=True, exist_ok=True)

        integrated_columns = [
            "task_id",
            "protocol",
            "method",
            "model_family",
            "selected_model",
            "sample_id",
            "y_true",
            "y_pred",
            "abs_error",
            "uncertainty_score",
            "confidence_tier",
            "confidence_source",
            "member_count",
            "uncertainty_method",
            "source_path",
        ]
        for column in integrated_columns:
            if column not in integrated.columns:
                integrated[column] = pd.NA
        data_frames = {
            "representative_model_selection": selection,
            "integrated_sample_level": integrated[integrated_columns],
            "method_tier_summary": tables["method_tier_summary"],
            "method_protocol_summary": tables["method_protocol_summary"],
            "task_protocol_method_summary": tables["task_protocol_method_summary"],
            "tier_spearman_validity": tables["tier_spearman_validity"],
            "coverage_audit": tables["coverage_audit"],
        }
        for name, frame in data_frames.items():
            frame.to_csv(data_dir / f"{name}.csv", index=False)

        figure_paths: list[Path] = []
        figure_paths.extend(_plot_main(tables, overview_dir, formats=formats, dpi=dpi))
        figure_paths.extend(_plot_by_protocol(tables, overview_dir, formats=formats, dpi=dpi))
        figure_paths.extend(_plot_by_method(tables, overview_dir, formats=formats, dpi=dpi))
        task_dirs: dict[str, str] = {}
        if include_task_appendix:
            task_paths, task_dirs = _plot_task_appendix(
                tables["tier_detail"],
                by_task_dir,
                formats=formats,
                dpi=dpi,
            )
            figure_paths.extend(task_paths)
        _write_readme(ready_dir / "README.md")

        protected_after = {
            name: _file_sha256(path) if Path(path).is_file() else _tree_sha256(path)
            for name, path in protected_paths.items()
        }
        source_hashes_after = _paths_hash(source_files)
        if protected_before != protected_after:
            raise ValueError(
                "Protected calibration sources changed during GPT plotting: "
                f"before={protected_before} after={protected_after}"
            )
        if source_hashes_before != source_hashes_after:
            raise ValueError("GPT or selection source files changed during plotting.")

        qa = {
            str(path.relative_to(ready_dir)): _validate_output_file(path, dpi)
            for path in figure_paths
        }
        expected_logical = 3 + (len(COMMON_TASKS) if include_task_appendix else 0)
        expected_files = expected_logical * len(formats)
        if len(figure_paths) != expected_files:
            raise ValueError(
                f"GPT figure output count mismatch: expected={expected_files} observed={len(figure_paths)}"
            )
        output_files = sorted(
            path
            for path in ready_dir.rglob("*")
            if path.is_file() and path.name != "figure_manifest.json"
        )
        output_hashes = {
            str(path.relative_to(ready_dir)): _file_sha256(path) for path in output_files
        }
        selection_composition = {
            method: selection[selection["method"].eq(method)]["selected_model"]
            .value_counts()
            .sort_index()
            .to_dict()
            for method in ("Traditional-best", "BERT-best")
        }
        payload = {
            "schema_version": 1,
            "stage": "full-plot-gpt",
            "scope": scope,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": validated["snapshot_id"],
            "source_verification_status": validated["report"]["verification_status"],
            "verified_unit_count": int(validated["report"]["verified_unit_count"]),
            "common_tasks": list(COMMON_TASKS),
            "excluded_tasks": EXCLUDED_TASKS,
            "protocols": list(PROTOCOL_ORDER),
            "methods": list(METHOD_ORDER),
            "confidence_tiers": list(TIER_ORDER),
            "integrated_sample_count": len(integrated),
            "sample_count_by_method": observed_method_samples,
            "representative_selection_count": len(selection),
            "representative_model_composition": selection_composition,
            "gpt_source_count_discovered": len(all_gpt_sources),
            "gpt_source_count_included": len(included_gpt_sources),
            "gpt_complete_three_tier_cases": int(gpt_coverage["complete_three_tiers"].sum()),
            "gpt_high_low_paired_cases": int(gpt_coverage["has_high_low_pair"].sum()),
            "summary_row_counts": observed_table_counts,
            "statistics": {
                "normalized_mae_weighting": "equal task-protocol observations",
                "bootstrap_iterations": bootstrap_iterations,
                "bootstrap_seed": bootstrap_seed,
                "discrimination_definition": "Low normalized MAE - High normalized MAE",
                "tier_risk_mapping": TIER_RISK_SCORE,
                "spearman_definition": "sample-level tier risk versus absolute error",
                "missing_tier_policy": "retain missing; require paired High/Low for discrimination",
            },
            "formats": list(formats),
            "png_dpi": dpi,
            "logical_figure_count": expected_logical,
            "figure_output_count": len(figure_paths),
            "task_outputs": task_dirs,
            "protected_source_hashes": protected_before,
            "input_files": {
                "prediction_inventory": {
                    "path": str(inventory_path),
                    "sha256": source_hashes_before[str(inventory_path)],
                },
                "combined_summary": {
                    "path": str(summary_path),
                    "sha256": source_hashes_before[str(summary_path)],
                },
                "non_gpt_sample": {
                    "path": str(validated["paths"]["sample"]),
                    "sha256": _file_sha256(validated["paths"]["sample"]),
                },
                "gpt_prediction_files": [
                    {"path": str(path), "sha256": source_hashes_before[str(path)]}
                    for path in all_gpt_sources
                ],
            },
            "output_file_hashes": output_hashes,
            "image_qa": qa,
        }
        (ready_dir / "figure_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        _publish_package(ready_dir, final_dir)
    except Exception:
        shutil.rmtree(ready_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(final_dir),
        "logical_figure_count": expected_logical,
        "output_file_count": len(figure_paths),
        "task_count": len(COMMON_TASKS),
        "integrated_sample_count": len(integrated),
        "manifest_path": str(final_dir / "figure_manifest.json"),
    }
