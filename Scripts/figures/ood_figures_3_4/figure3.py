from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    TIER_ORDER,
    TIER_RISK_SCORE,
    StageResult,
    assign_uncertainty_tertiles,
    coalesce_column,
    empty_csv,
    ensure_dirs,
    mean_ci,
    normalize_confidence_tier,
    read_source_csv,
    source_path,
    spearman_corr,
)
from .config import FigureConfig, list_section_sources
from .discovery import _protocol_to_axis
from .manifest import write_manifest


SAMPLE_COLUMNS = [
    "source_name",
    "source_file",
    "sample_id",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "protocol",
    "failure_axis",
    "model_family",
    "model",
    "y_true",
    "y_pred",
    "abs_error",
    "uncertainty_score",
    "confidence_source",
    "confidence_tier",
]

PROTOCOL_ORDER = [
    "Extrapolation",
    "LOCO",
    "RandomCV",
    "SparseXsingle",
    "SparseXcluster",
    "SparseYsingle",
    "SparseYcluster",
]

BY_METHOD_SUMMARY_COLUMNS = [
    "protocol",
    "task_id",
    "model_family",
    "model",
    "confidence_tier",
    "tier_n",
    "mae",
    "ci95_low",
    "ci95_high",
    "model_overall_mae",
    "normalized_mae",
]

BY_METHOD_METRIC_COLUMNS = ["protocol", "task_id", "model_family", "model", "metric_name", "metric_value"]

BY_TASK_METHOD_COLUMNS = BY_METHOD_SUMMARY_COLUMNS + ["safe_task_id", "normalized_ci95_low", "normalized_ci95_high"]

OVERVIEW_COLUMNS = [
    "protocol",
    "model_family",
    "model",
    "confidence_tier",
    "task_n",
    "sample_n",
    "normalized_mae",
    "ci95_low",
    "ci95_high",
    "included_task_ids",
]

TASK_METHOD_COVERAGE_COLUMNS = [
    "task_id",
    "safe_task_id",
    "protocol",
    "protocol_present",
    "tier_count",
    "sample_count",
    "missing_tiers",
]


def _standardize_source(entry: dict[str, Any]) -> pd.DataFrame:
    frame = read_source_csv(entry)
    id_col = coalesce_column(frame, ["sample_id", "ID", "id"])
    true_col = coalesce_column(frame, ["y_true", "true_value", "actual", "actual_value"])
    pred_col = coalesce_column(frame, ["y_pred", "predicted_value", "prediction", "pred"])
    if true_col is None:
        true_candidates = [column for column in frame.columns if str(column).lower().endswith("_true")]
        true_col = true_candidates[0] if true_candidates else None
    if pred_col is None:
        pred_candidates = [column for column in frame.columns if str(column).lower().endswith(("_predicted", "_pred"))]
        pred_col = pred_candidates[0] if pred_candidates else None
    confidence_col = coalesce_column(frame, ["confidence_tier", "confidence", "semantic_confidence"])
    uncertainty_col = coalesce_column(frame, ["uncertainty_score", "sigma", "std", "predictive_std"])
    if true_col is None or pred_col is None:
        raise ValueError(f"Figure 3 source {entry.get('name')} must include y_true/y_pred columns")

    output = pd.DataFrame(index=frame.index)
    output["source_name"] = frame.get("source_name", entry.get("name", "source"))
    output["source_file"] = frame.get("source_file", str(source_path(entry)))
    output["sample_id"] = frame[id_col] if id_col else np.arange(len(frame))
    for column in ["task_id", "alloy_family", "dataset_name", "property", "protocol", "failure_axis", "model_family", "model"]:
        output[column] = frame[column] if column in frame.columns else entry.get("metadata", {}).get(column, pd.NA)
    output["y_true"] = pd.to_numeric(frame[true_col], errors="coerce")
    output["y_pred"] = pd.to_numeric(frame[pred_col], errors="coerce")
    output["abs_error"] = (output["y_pred"] - output["y_true"]).abs()
    output["uncertainty_score"] = pd.to_numeric(frame[uncertainty_col], errors="coerce") if uncertainty_col else np.nan
    output["confidence_source"] = entry.get("confidence_source") or frame.get("confidence_source", "semantic" if confidence_col else "uncertainty_tertile")

    if confidence_col:
        output["confidence_tier"] = frame[confidence_col].map(normalize_confidence_tier)
    else:
        output["confidence_tier"] = pd.NA

    is_gpt = output["model"].astype(str).str.contains("gpt", case=False, na=False)
    needs_tertile = output["confidence_tier"].isna() & ~is_gpt
    if needs_tertile.any():
        group_cols = ["task_id", "protocol", "failure_axis", "model_family", "model"]
        for _, index in output[needs_tertile].groupby(group_cols, dropna=False).groups.items():
            output.loc[index, "confidence_tier"] = assign_uncertainty_tertiles(output.loc[index, "uncertainty_score"])
    return output[SAMPLE_COLUMNS]


def build_sample_level_table(cfg: FigureConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in list_section_sources(cfg, "figure3", "sample_level_sources"):
        path = source_path(entry)
        if not path.exists():
            continue
        frames.append(_standardize_source(entry))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    sample = pd.concat(frames, ignore_index=True)
    aliases = cfg.raw.get("model_family_aliases", {}) or {}
    sample["model_family"] = sample.apply(
        lambda row: aliases.get(str(row["model"]), row["model_family"]),
        axis=1,
    )
    sample["confidence_tier"] = pd.Categorical(sample["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return sample.sort_values(["failure_axis", "task_id", "model_family", "model", "confidence_tier", "sample_id"])


def _discovered_path(cfg: FigureConfig) -> Path:
    return cfg.output_root / "audit" / "discovered_sources.csv"


def _metadata_from_discovered(row: pd.Series) -> dict[str, object]:
    return {
        "task_id": row.get("task_id"),
        "alloy_family": row.get("alloy_family"),
        "dataset_name": row.get("dataset_name"),
        "property": row.get("property"),
        "protocol": row.get("protocol"),
        "failure_axis": row.get("failure_axis") or _protocol_to_axis(row.get("protocol")),
        "model": row.get("model"),
        "model_family": row.get("model_family"),
    }


def build_sample_level_table_from_discovery(cfg: FigureConfig) -> pd.DataFrame:
    path = _discovered_path(cfg)
    if not path.exists():
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    discovered = pd.read_csv(path)
    frames: list[pd.DataFrame] = []
    for _, row in discovered.iterrows():
        if row.get("usable_for") != "figure3":
            continue
        missing_reason = row.get("missing_reason")
        if pd.notna(missing_reason) and str(missing_reason):
            continue
        discovered_source_path = Path(str(row.get("path")))
        if not discovered_source_path.exists():
            continue
        entry = {
            "name": row.get("source_name"),
            "path": str(discovered_source_path),
            "required": True,
            "confidence_source": row.get("confidence_source") or "semantic",
            "metadata": _metadata_from_discovered(row),
        }
        frames.append(_standardize_source(entry))
    if not frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    sample = pd.concat(frames, ignore_index=True)
    sample["confidence_tier"] = pd.Categorical(sample["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return sample.sort_values(["failure_axis", "task_id", "model_family", "model", "confidence_tier", "sample_id"])


def summarize_tiers(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(
            columns=[
                "failure_axis",
                "task_id",
                "model_family",
                "model",
                "confidence_tier",
                "tier_n",
                "mae",
                "ci95_low",
                "ci95_high",
                "model_overall_mae",
                "normalized_mae",
            ]
        )
    working = sample.copy()
    working["model_overall_mae"] = working.groupby(
        ["failure_axis", "task_id", "model_family", "model"],
        dropna=False,
    )["abs_error"].transform("mean")
    rows = []
    for keys, group in working.dropna(subset=["confidence_tier"]).groupby(
        ["failure_axis", "task_id", "model_family", "model", "confidence_tier"],
        observed=True,
        dropna=False,
    ):
        mean, low, high = mean_ci(group["abs_error"])
        model_mae = float(pd.to_numeric(group["model_overall_mae"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "failure_axis": keys[0],
                "task_id": keys[1],
                "model_family": keys[2],
                "model": keys[3],
                "confidence_tier": str(keys[4]),
                "tier_n": int(len(group)),
                "mae": mean,
                "ci95_low": low,
                "ci95_high": high,
                "model_overall_mae": model_mae,
                "normalized_mae": mean / model_mae if model_mae and np.isfinite(model_mae) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _ordered_protocols(values: pd.Series) -> list[str]:
    present = [str(value) for value in values.fillna("Unknown").unique()]
    ordered = [protocol for protocol in PROTOCOL_ORDER if protocol in present]
    ordered.extend(sorted(protocol for protocol in present if protocol not in set(PROTOCOL_ORDER)))
    return ordered


def safe_task_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or "task"


def safe_task_id_map(task_ids: pd.Series) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for task_id in sorted(str(value) for value in task_ids.dropna().unique()):
        base = safe_task_id(task_id)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapping[task_id] = candidate
        used.add(candidate)
    return mapping


def summarize_tiers_by_method(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(columns=BY_METHOD_SUMMARY_COLUMNS)
    working = sample.copy()
    working["protocol"] = working["protocol"].fillna("Unknown")
    working["model_overall_mae"] = working.groupby(
        ["protocol", "task_id", "model_family", "model"],
        dropna=False,
    )["abs_error"].transform("mean")
    rows = []
    for keys, group in working.dropna(subset=["confidence_tier"]).groupby(
        ["protocol", "task_id", "model_family", "model", "confidence_tier"],
        observed=True,
        dropna=False,
    ):
        mean, low, high = mean_ci(group["abs_error"])
        model_mae = float(pd.to_numeric(group["model_overall_mae"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "protocol": keys[0],
                "task_id": keys[1],
                "model_family": keys[2],
                "model": keys[3],
                "confidence_tier": str(keys[4]),
                "tier_n": int(len(group)),
                "mae": mean,
                "ci95_low": low,
                "ci95_high": high,
                "model_overall_mae": model_mae,
                "normalized_mae": mean / model_mae if model_mae and np.isfinite(model_mae) else np.nan,
            }
        )
    summary = pd.DataFrame(rows, columns=BY_METHOD_SUMMARY_COLUMNS)
    if summary.empty:
        return summary
    summary["protocol"] = pd.Categorical(summary["protocol"], categories=_ordered_protocols(summary["protocol"]), ordered=True)
    summary["confidence_tier"] = pd.Categorical(summary["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return summary.sort_values(["protocol", "task_id", "model_family", "model", "confidence_tier"]).astype(
        {"protocol": "object", "confidence_tier": "object"}
    )


def confidence_metrics(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sample.empty:
        return pd.DataFrame(columns=["failure_axis", "task_id", "model_family", "model", "metric_name", "metric_value"])
    working = sample.copy()
    working["tier_risk_score"] = working["confidence_tier"].map(TIER_RISK_SCORE)
    for keys, group in working.groupby(["failure_axis", "task_id", "model_family", "model"], dropna=False):
        rows.append({**dict(zip(["failure_axis", "task_id", "model_family", "model"], keys)), "metric_name": "spearman_confidence_error", "metric_value": spearman_corr(group["tier_risk_score"], group["abs_error"])})
        tier_mae = group.groupby("confidence_tier", observed=True)["abs_error"].mean()
        high = tier_mae.get("High", np.nan)
        low = tier_mae.get("Low", np.nan)
        ratio = low / high if high and np.isfinite(high) else np.nan
        rows.append({**dict(zip(["failure_axis", "task_id", "model_family", "model"], keys)), "metric_name": "low_vs_high_mae_ratio", "metric_value": ratio})
    return pd.DataFrame(rows)


def confidence_metrics_by_method(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if sample.empty:
        return pd.DataFrame(columns=BY_METHOD_METRIC_COLUMNS)
    working = sample.copy()
    working["protocol"] = working["protocol"].fillna("Unknown")
    working["tier_risk_score"] = working["confidence_tier"].map(TIER_RISK_SCORE)
    for keys, group in working.groupby(["protocol", "task_id", "model_family", "model"], dropna=False):
        rows.append(
            {
                **dict(zip(["protocol", "task_id", "model_family", "model"], keys)),
                "metric_name": "spearman_confidence_error",
                "metric_value": spearman_corr(group["tier_risk_score"], group["abs_error"]),
            }
        )
        tier_mae = group.groupby("confidence_tier", observed=True)["abs_error"].mean()
        high = tier_mae.get("High", np.nan)
        low = tier_mae.get("Low", np.nan)
        ratio = low / high if high and np.isfinite(high) else np.nan
        rows.append(
            {
                **dict(zip(["protocol", "task_id", "model_family", "model"], keys)),
                "metric_name": "low_vs_high_mae_ratio",
                "metric_value": ratio,
            }
        )
    metrics = pd.DataFrame(rows, columns=BY_METHOD_METRIC_COLUMNS)
    if metrics.empty:
        return metrics
    metrics["protocol"] = pd.Categorical(metrics["protocol"], categories=_ordered_protocols(metrics["protocol"]), ordered=True)
    return metrics.sort_values(["protocol", "task_id", "model_family", "model", "metric_name"]).astype({"protocol": "object"})


def al_uts_diagnostic(sample: pd.DataFrame, cfg: FigureConfig) -> pd.DataFrame:
    section = cfg.raw.get("figure3", {})
    filters = section.get("al_uts_filter", {}) or {}
    task_text = str(filters.get("task_id_contains", "Al-UTS"))
    model_text = str(filters.get("model_contains", "GPT"))
    mask = sample["task_id"].astype(str).str.contains(task_text, case=False, na=False) & sample["model"].astype(str).str.contains(model_text, case=False, na=False)
    diagnostic = summarize_tiers(sample.loc[mask].copy())
    return diagnostic


def write_missing_uncertainty_report(sample: pd.DataFrame, cfg: FigureConfig) -> int:
    audit_dir = cfg.output_root / "audit"
    ensure_dirs(audit_dir)
    path = audit_dir / "missing_required_inputs.csv"
    columns = ["section", "source_type", "source_name", "path", "required", "exists", "size_bytes", "status", "message"]
    is_gpt = sample["model"].astype(str).str.contains("gpt", case=False, na=False)
    missing = sample[~is_gpt & sample["confidence_tier"].isna()].copy()
    if missing.empty:
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        return 0

    rows = []
    for (source_name, source_file), group in missing.groupby(["source_name", "source_file"], dropna=False):
        source_path = Path(str(source_file))
        rows.append(
            {
                "section": "figure3",
                "source_type": "sample_level",
                "source_name": source_name,
                "path": str(source_path),
                "required": True,
                "exists": source_path.exists(),
                "size_bytes": source_path.stat().st_size if source_path.exists() else 0,
                "status": "missing_uncertainty",
                "message": "Non-GPT Figure 3 source needs confidence_tier or uncertainty_score/sigma/std/predictive_std.",
            }
        )

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["section", "source_type", "source_name", "path", "status"], keep="last")
    combined.to_csv(path, index=False)
    return len(rows)


def plot_figure3(tier_summary: pd.DataFrame, figures_dir: Path) -> None:
    if tier_summary.empty:
        return
    axes_values = list(tier_summary["failure_axis"].dropna().unique()) or ["all"]
    fig, axes = plt.subplots(1, len(axes_values), figsize=(6 * len(axes_values), 4), squeeze=False)
    x_lookup = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
    for ax, axis_value in zip(axes[0], axes_values):
        panel = tier_summary if axis_value == "all" else tier_summary[tier_summary["failure_axis"].eq(axis_value)]
        plot_rows = []
        for keys, group in panel.groupby(["model_family", "model", "confidence_tier"], observed=True, dropna=False):
            mean, low, high = mean_ci(group["normalized_mae"])
            plot_rows.append(
                {
                    "model_family": keys[0],
                    "model": keys[1],
                    "confidence_tier": keys[2],
                    "normalized_mae": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        plot_data = pd.DataFrame(plot_rows)
        if not plot_data.empty:
            plot_data["confidence_tier"] = pd.Categorical(plot_data["confidence_tier"], categories=TIER_ORDER, ordered=True)
        for (model_family, model), group in plot_data.groupby(["model_family", "model"], dropna=False):
            ordered = group.sort_values("confidence_tier")
            x = [x_lookup[str(tier)] for tier in ordered["confidence_tier"]]
            y = pd.to_numeric(ordered["normalized_mae"], errors="coerce")
            yerr = np.vstack(
                [
                    y - pd.to_numeric(ordered["ci95_low"], errors="coerce"),
                    pd.to_numeric(ordered["ci95_high"], errors="coerce") - y,
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=f"{model_family}: {model}",
            )
        ax.set_title(str(axis_value))
        ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER)
        ax.set_ylabel("Normalized MAE")
        ax.set_xlabel("Confidence / uncertainty tier")
        ax.grid(alpha=0.25)
    axes[0, -1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(figures_dir / f"figure3_confidence_error_discrimination.{ext}", dpi=300)
    plt.close(fig)


def by_method_manifest_payload(sample: pd.DataFrame, by_method_summary: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "included_protocols": [],
            "task_count_by_protocol": {},
            "sample_count_by_protocol": {},
            "missing_tiers_by_protocol": {},
            "is_gpt_only": False,
        }
    working = sample.copy()
    working["protocol"] = working["protocol"].fillna("Unknown")
    protocols = _ordered_protocols(working["protocol"])
    task_counts = working.groupby("protocol", dropna=False)["task_id"].nunique().to_dict()
    sample_counts = working.groupby("protocol", dropna=False).size().to_dict()
    missing_tiers = {}
    for protocol in protocols:
        present = set(
            by_method_summary.loc[by_method_summary["protocol"].astype(str).eq(protocol), "confidence_tier"].dropna().astype(str)
        )
        missing = [tier for tier in TIER_ORDER if tier not in present]
        if missing:
            missing_tiers[protocol] = missing
    is_gpt_only = sample["model"].astype(str).str.contains("gpt", case=False, na=False).all()
    return {
        "included_protocols": protocols,
        "task_count_by_protocol": {str(key): int(value) for key, value in task_counts.items()},
        "sample_count_by_protocol": {str(key): int(value) for key, value in sample_counts.items()},
        "missing_tiers_by_protocol": missing_tiers,
        "is_gpt_only": bool(is_gpt_only),
    }


def figure3_excluded_tasks(cfg: FigureConfig) -> list[str]:
    section = cfg.raw.get("figure3", {}) or {}
    return [str(task_id) for task_id in section.get("exclude_tasks_from_aggregate", []) or []]


def overview_aggregate_table(by_method_summary: pd.DataFrame) -> pd.DataFrame:
    if by_method_summary.empty:
        return pd.DataFrame(columns=OVERVIEW_COLUMNS)
    rows = []
    working = by_method_summary.copy()
    working["protocol"] = working["protocol"].fillna("Unknown")
    for keys, group in working.groupby(["protocol", "model_family", "model", "confidence_tier"], observed=True, dropna=False):
        mean, low, high = mean_ci(group["normalized_mae"])
        rows.append(
            {
                "protocol": keys[0],
                "model_family": keys[1],
                "model": keys[2],
                "confidence_tier": str(keys[3]),
                "task_n": int(group["task_id"].nunique()),
                "sample_n": int(pd.to_numeric(group["tier_n"], errors="coerce").fillna(0).sum()),
                "normalized_mae": mean,
                "ci95_low": low,
                "ci95_high": high,
                "included_task_ids": ";".join(sorted(str(value) for value in group["task_id"].dropna().unique())),
            }
        )
    overview = pd.DataFrame(rows, columns=OVERVIEW_COLUMNS)
    if overview.empty:
        return overview
    overview["protocol"] = pd.Categorical(overview["protocol"], categories=_ordered_protocols(overview["protocol"]), ordered=True)
    overview["confidence_tier"] = pd.Categorical(overview["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return overview.sort_values(["protocol", "model_family", "model", "confidence_tier"]).astype(
        {"protocol": "object", "confidence_tier": "object"}
    )


def plot_overview_from_csv(overview_csv: Path, overview_dir: Path) -> str | None:
    if not overview_csv.exists():
        return None
    overview = pd.read_csv(overview_csv)
    if overview.empty:
        return None
    protocols = _ordered_protocols(overview["protocol"])
    ncols = min(3, len(protocols))
    nrows = int(np.ceil(len(protocols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), squeeze=False)
    x_lookup = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
    for ax in axes.ravel()[len(protocols) :]:
        ax.axis("off")
    for ax, protocol in zip(axes.ravel(), protocols):
        plot_data = overview[overview["protocol"].astype(str).eq(protocol)].copy()
        if not plot_data.empty:
            plot_data["confidence_tier"] = pd.Categorical(plot_data["confidence_tier"], categories=TIER_ORDER, ordered=True)
        for (model_family, model), group in plot_data.groupby(["model_family", "model"], dropna=False):
            ordered = group.sort_values("confidence_tier")
            x = [x_lookup[str(tier)] for tier in ordered["confidence_tier"]]
            y = pd.to_numeric(ordered["normalized_mae"], errors="coerce")
            yerr = np.vstack(
                [
                    y - pd.to_numeric(ordered["ci95_low"], errors="coerce"),
                    pd.to_numeric(ordered["ci95_high"], errors="coerce") - y,
                ]
            )
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"{model_family}: {model}")
        ax.axhline(1.0, color="0.75", linewidth=0.8, linestyle="--")
        ax.set_title(str(protocol))
        ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER)
        ax.set_xlabel("Confidence tier")
        ax.set_ylabel("Normalized MAE")
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if handles else 1))
    output_path = overview_dir / "figure3_confidence_error_by_method.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path)


def task_by_method_table(by_method_summary: pd.DataFrame) -> pd.DataFrame:
    if by_method_summary.empty:
        return pd.DataFrame(columns=BY_TASK_METHOD_COLUMNS)
    table = by_method_summary.copy()
    mapping = safe_task_id_map(table["task_id"])
    table["safe_task_id"] = table["task_id"].astype(str).map(mapping)
    table["normalized_ci95_low"] = pd.to_numeric(table["ci95_low"], errors="coerce") / pd.to_numeric(
        table["model_overall_mae"], errors="coerce"
    )
    table["normalized_ci95_high"] = pd.to_numeric(table["ci95_high"], errors="coerce") / pd.to_numeric(
        table["model_overall_mae"], errors="coerce"
    )
    return table[BY_TASK_METHOD_COLUMNS]


def task_method_coverage(task_method_summary: pd.DataFrame) -> pd.DataFrame:
    if task_method_summary.empty:
        return pd.DataFrame(columns=TASK_METHOD_COVERAGE_COLUMNS)
    rows = []
    protocols = PROTOCOL_ORDER + [
        protocol
        for protocol in _ordered_protocols(task_method_summary["protocol"])
        if protocol not in set(PROTOCOL_ORDER)
    ]
    for task_id, task_group in task_method_summary.groupby("task_id", dropna=False):
        safe_id = str(task_group["safe_task_id"].iloc[0])
        for protocol in protocols:
            protocol_group = task_group[task_group["protocol"].astype(str).eq(protocol)]
            present_tiers = set(protocol_group["confidence_tier"].dropna().astype(str))
            protocol_present = not protocol_group.empty
            missing = [tier for tier in TIER_ORDER if tier not in present_tiers]
            rows.append(
                {
                    "task_id": task_id,
                    "safe_task_id": safe_id,
                    "protocol": protocol,
                    "protocol_present": bool(protocol_present),
                    "tier_count": int(len(present_tiers)),
                    "sample_count": int(pd.to_numeric(protocol_group["tier_n"], errors="coerce").fillna(0).sum()),
                    "missing_tiers": ",".join(missing),
                }
            )
    return pd.DataFrame(rows, columns=TASK_METHOD_COVERAGE_COLUMNS)


def write_task_by_method_outputs(task_method_summary: pd.DataFrame, by_task_root: Path) -> dict[str, dict[str, str]]:
    if task_method_summary.empty:
        return {}
    ensure_dirs(by_task_root)
    outputs: dict[str, dict[str, str]] = {}
    x_lookup = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
    for task_id, task_group in task_method_summary.groupby("task_id", dropna=False):
        safe_id = str(task_group["safe_task_id"].iloc[0])
        task_dir = by_task_root / safe_id
        ensure_dirs(task_dir)
        csv_path = task_dir / "figure3_confidence_error_by_method.csv"
        task_group.to_csv(csv_path, index=False)
        protocols = _ordered_protocols(task_group["protocol"])
        ncols = min(3, len(protocols))
        nrows = int(np.ceil(len(protocols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), squeeze=False)
        for ax in axes.ravel()[len(protocols) :]:
            ax.axis("off")
        for ax, protocol in zip(axes.ravel(), protocols):
            panel = task_group[task_group["protocol"].astype(str).eq(protocol)]
            for (model_family, model), group in panel.groupby(["model_family", "model"], dropna=False):
                ordered = group.copy()
                ordered["confidence_tier"] = pd.Categorical(ordered["confidence_tier"], categories=TIER_ORDER, ordered=True)
                ordered = ordered.sort_values("confidence_tier")
                x = [x_lookup[str(tier)] for tier in ordered["confidence_tier"]]
                y = pd.to_numeric(ordered["normalized_mae"], errors="coerce")
                low = pd.to_numeric(ordered["normalized_ci95_low"], errors="coerce")
                high = pd.to_numeric(ordered["normalized_ci95_high"], errors="coerce")
                yerr = np.vstack([y - low, high - y])
                ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"{model_family}: {model}")
            ax.axhline(1.0, color="0.75", linewidth=0.8, linestyle="--")
            ax.set_title(str(protocol))
            ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER)
            ax.set_xlabel("Confidence tier")
            ax.set_ylabel("Normalized MAE")
            ax.grid(alpha=0.25)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if len(labels) > 1:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
        fig.suptitle(str(task_id), y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94 if len(labels) > 1 else 0.96))
        png_path = task_dir / "figure3_confidence_error_by_method.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        outputs[str(task_id)] = {"csv": str(csv_path), "png": str(png_path)}
    return outputs


def task_by_method_manifest_payload(coverage: pd.DataFrame, outputs: dict[str, dict[str, str]]) -> dict[str, Any]:
    if coverage.empty:
        return {
            "figure_count": 0,
            "protocol_count_by_task": {},
            "missing_tiers_by_task_protocol": {},
            "outputs": {},
        }
    present = coverage[coverage["protocol_present"].astype(bool)]
    protocol_counts = present.groupby("task_id")["protocol"].nunique().to_dict()
    missing: dict[str, dict[str, list[str]]] = {}
    for _, row in present.iterrows():
        missing_tiers = [tier for tier in str(row["missing_tiers"]).split(",") if tier]
        if missing_tiers:
            missing.setdefault(str(row["task_id"]), {})[str(row["protocol"])] = missing_tiers
    return {
        "figure_count": int(coverage["task_id"].nunique()),
        "protocol_count_by_task": {str(key): int(value) for key, value in protocol_counts.items()},
        "missing_tiers_by_task_protocol": missing,
        "outputs": outputs,
    }


def _clean_generated_figure3_dirs(stage_dir: Path) -> None:
    for name in ["csv", "figures", "overview", "by_task"]:
        target = stage_dir / name
        if target.exists():
            shutil.rmtree(target)


def run_figure3(cfg: FigureConfig) -> StageResult:
    stage_dir = cfg.output_root / "figure3_calibration"
    _clean_generated_figure3_dirs(stage_dir)
    overview_dir = stage_dir / "overview"
    by_task_dir = stage_dir / "by_task"
    ensure_dirs(overview_dir, by_task_dir)

    configured_sources = list_section_sources(cfg, "figure3", "sample_level_sources")
    sample_frames: list[pd.DataFrame] = []
    if configured_sources:
        configured_sample = build_sample_level_table(cfg)
        if not configured_sample.empty:
            sample_frames.append(configured_sample)
    discovered_sample = build_sample_level_table_from_discovery(cfg)
    if not discovered_sample.empty:
        sample_frames.append(discovered_sample)
    if sample_frames:
        sample = pd.concat(sample_frames, ignore_index=True).drop_duplicates(
            subset=["source_file", "sample_id", "task_id", "protocol", "model_family", "model"],
            keep="first",
        )
        sample["confidence_tier"] = pd.Categorical(sample["confidence_tier"], categories=TIER_ORDER, ordered=True)
        sample = sample.sort_values(
            ["failure_axis", "task_id", "model_family", "model", "confidence_tier", "sample_id"]
        )
    else:
        sample = pd.DataFrame(columns=SAMPLE_COLUMNS)

    by_method_summary = summarize_tiers_by_method(sample)
    task_method_summary = task_by_method_table(by_method_summary)
    task_coverage = task_method_coverage(task_method_summary)
    missing_uncertainty_count = write_missing_uncertainty_report(sample, cfg)

    excluded_tasks = figure3_excluded_tasks(cfg)
    overview_method_summary = by_method_summary[
        ~by_method_summary["task_id"].astype(str).isin(excluded_tasks)
    ].copy() if not by_method_summary.empty else by_method_summary
    overview_sample = sample[
        ~sample["task_id"].astype(str).isin(excluded_tasks)
    ].copy() if not sample.empty else sample
    overview_table = overview_aggregate_table(overview_method_summary)
    overview_csv = overview_dir / "figure3_confidence_error_by_method.csv"
    overview_table.to_csv(overview_csv, index=False)
    overview_png = plot_overview_from_csv(overview_csv, overview_dir)

    task_outputs = write_task_by_method_outputs(task_method_summary, by_task_dir)

    if configured_sources:
        sources = [source_path(entry) for entry in configured_sources if source_path(entry).exists()]
    elif not sample.empty and "source_file" in sample:
        sources = [Path(str(path)) for path in sorted(sample["source_file"].dropna().astype(str).unique())]
    elif _discovered_path(cfg).exists():
        sources = [_discovered_path(cfg)]
    else:
        sources = []
    included_task_ids = sorted(str(value) for value in overview_method_summary["task_id"].dropna().unique()) if not overview_method_summary.empty else []
    actual_excluded_task_ids = sorted(
        task_id for task_id in excluded_tasks if not sample.empty and sample["task_id"].astype(str).eq(task_id).any()
    )
    write_manifest(
        stage_dir / "manifest.json",
        stage="figure3",
        config_path=cfg.config_path,
        output_root=cfg.output_root,
        payload={
            "row_count": int(len(sample)),
            "tier_summary_by_method_rows": int(len(overview_method_summary)),
            "missing_uncertainty_count": int(missing_uncertainty_count),
            "overview": {
                **by_method_manifest_payload(overview_sample, overview_method_summary),
                "csv_path": str(overview_csv),
                "png_path": overview_png,
                "included_task_ids": included_task_ids,
                "excluded_task_ids": actual_excluded_task_ids,
            },
            "task_by_method": task_by_method_manifest_payload(task_coverage, task_outputs),
        },
        source_files=sources,
    )
    return StageResult(stage="figure3", output_dir=stage_dir, row_count=len(sample), missing_count=missing_uncertainty_count)
