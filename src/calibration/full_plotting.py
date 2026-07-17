from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .full_pipeline import _replace_with_retry


TIER_ORDER = ("High", "Medium", "Low")
PROTOCOL_ORDER = (
    "RandomCV",
    "LOCO",
    "Extrapolation",
    "SparseXsingle",
    "SparseXcluster",
    "SparseYsingle",
    "SparseYcluster",
)
FAMILY_ORDER = ("Traditional", "BERT", "LLMProp", "TabPFN")
MODEL_ORDER = (
    "CatBoost",
    "LightGBM",
    "MLP",
    "RF",
    "XGB",
    "MatSciBERT",
    "SciBERT",
    "SteelBERT",
    "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
)
FAMILY_COLORS = {
    "Traditional": "#0072B2",
    "BERT": "#D55E00",
    "LLMProp": "#009E73",
    "TabPFN": "#CC79A7",
}
MODEL_FAMILIES = {
    "CatBoost": "Traditional",
    "LightGBM": "Traditional",
    "MLP": "Traditional",
    "RF": "Traditional",
    "XGB": "Traditional",
    "MatSciBERT": "BERT",
    "SciBERT": "BERT",
    "SteelBERT": "BERT",
    "LLM-Prop": "LLMProp",
    "TabPFN-2.5-Plus-Numeric": "TabPFN",
    "TabPFN-2.5-Plus-Text": "TabPFN",
}
MODEL_LABELS = {
    "CatBoost": "Trad. | CatBoost",
    "LightGBM": "Trad. | LightGBM",
    "MLP": "Trad. | MLP",
    "RF": "Trad. | RF",
    "XGB": "Trad. | XGB",
    "MatSciBERT": "BERT | MatSciBERT",
    "SciBERT": "BERT | SciBERT",
    "SteelBERT": "BERT | SteelBERT",
    "LLM-Prop": "LLM | LLM-Prop",
    "TabPFN-2.5-Plus-Numeric": "TabPFN | Numeric",
    "TabPFN-2.5-Plus-Text": "TabPFN | Text",
}
DEFAULT_SETTINGS = {
    "output_dir": "figures/final",
    "formats": ["png", "pdf", "svg"],
    "dpi": 300,
    "bootstrap_iterations": 10000,
    "bootstrap_seed": 20260715,
    "include_task_appendix": True,
}
EXPECTED_COUNTS = {
    "all": {
        "verified_units": 1650,
        "tier_rows": 2310,
        "metric_rows": 770,
        "coverage_rows": 770,
        "tasks": 10,
        "spearman_na": 127,
    },
    "ti": {
        "verified_units": 330,
        "tier_rows": 462,
        "metric_rows": 154,
        "coverage_rows": 154,
        "tasks": 2,
    },
}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: str | Path) -> str:
    root = Path(path).resolve()
    digest = hashlib.sha256()
    if not root.exists():
        digest.update(b"MISSING")
        return digest.hexdigest()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "item"


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
    if numeric.size == 1 or int(iterations) <= 1:
        return mean, mean, mean
    sampled = rng.choice(numeric, size=(int(iterations), numeric.size), replace=True)
    bootstrap_means = sampled.mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _summarize_family_tiers(
    tier_summary: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(bootstrap_seed))
    rows: list[dict[str, Any]] = []
    groupings: list[tuple[str, pd.DataFrame]] = [("All", tier_summary)]
    groupings.extend(
        (protocol, tier_summary[tier_summary["protocol"].astype(str).eq(protocol)])
        for protocol in PROTOCOL_ORDER
    )
    for protocol, frame in groupings:
        for family in FAMILY_ORDER:
            family_frame = frame[frame["model_family"].astype(str).eq(family)]
            for tier in TIER_ORDER:
                values = pd.to_numeric(
                    family_frame.loc[
                        family_frame["confidence_tier"].astype(str).eq(tier),
                        "normalized_mae",
                    ],
                    errors="coerce",
                )
                mean, low, high = _bootstrap_mean_ci(
                    values,
                    rng=rng,
                    iterations=bootstrap_iterations,
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "model_family": family,
                        "confidence_tier": tier,
                        "normalized_mae_mean": mean,
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "observation_count": int(values.notna().sum()),
                    }
                )
    result = pd.DataFrame(rows)
    result["protocol"] = pd.Categorical(
        result["protocol"],
        categories=("All",) + PROTOCOL_ORDER,
        ordered=True,
    )
    result["model_family"] = pd.Categorical(
        result["model_family"], categories=FAMILY_ORDER, ordered=True
    )
    result["confidence_tier"] = pd.Categorical(
        result["confidence_tier"], categories=TIER_ORDER, ordered=True
    )
    return result.sort_values(
        ["protocol", "model_family", "confidence_tier"], kind="stable"
    ).astype(
        {
            "protocol": "object",
            "model_family": "object",
            "confidence_tier": "object",
        }
    ).reset_index(drop=True)


def prepare_plot_tables(
    tier_summary: pd.DataFrame,
    uncertainty_metrics: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260715,
) -> dict[str, pd.DataFrame]:
    required_tier = {
        "task_id",
        "protocol",
        "model_family",
        "model",
        "confidence_tier",
        "normalized_mae",
    }
    required_metric = {
        "task_id",
        "protocol",
        "model_family",
        "model",
        "spearman_uncertainty_error",
    }
    missing_tier = sorted(required_tier.difference(tier_summary.columns))
    missing_metric = sorted(required_metric.difference(uncertainty_metrics.columns))
    if missing_tier:
        raise ValueError(f"Tier summary is missing columns: {missing_tier}")
    if missing_metric:
        raise ValueError(f"Uncertainty metrics are missing columns: {missing_metric}")

    tier = tier_summary.copy()
    metric = uncertainty_metrics.copy()
    tier["confidence_tier"] = pd.Categorical(
        tier["confidence_tier"], categories=TIER_ORDER, ordered=True
    )
    tier["normalized_mae"] = pd.to_numeric(tier["normalized_mae"], errors="raise")
    metric["spearman_uncertainty_error"] = pd.to_numeric(
        metric["spearman_uncertainty_error"], errors="coerce"
    )

    keys = ["task_id", "protocol", "model_family", "model"]
    pivot = tier.pivot_table(
        index=keys,
        columns="confidence_tier",
        values="normalized_mae",
        aggfunc="first",
        observed=False,
    ).reset_index()
    for tier_name in TIER_ORDER:
        if tier_name not in pivot.columns:
            pivot[tier_name] = np.nan
    pivot["discrimination"] = pivot["Low"] - pivot["High"]
    task_summary = pivot.merge(
        metric[keys + ["spearman_uncertainty_error"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    task_summary = task_summary[
        keys
        + [
            "High",
            "Medium",
            "Low",
            "discrimination",
            "spearman_uncertainty_error",
        ]
    ].sort_values(keys, kind="stable").reset_index(drop=True)

    model_rows: list[dict[str, Any]] = []
    for (protocol, family, model), group in task_summary.groupby(
        ["protocol", "model_family", "model"], sort=False, dropna=False
    ):
        spearman = pd.to_numeric(group["spearman_uncertainty_error"], errors="coerce")
        model_rows.append(
            {
                "protocol": str(protocol),
                "model_family": str(family),
                "model": str(model),
                "discrimination_mean": float(group["discrimination"].mean()),
                "discrimination_count": int(group["discrimination"].notna().sum()),
                "spearman_mean": float(spearman.mean()) if spearman.notna().any() else np.nan,
                "spearman_valid_count": int(spearman.notna().sum()),
                "spearman_total_count": int(len(spearman)),
            }
        )
    model_summary = pd.DataFrame(model_rows)
    model_summary["protocol"] = pd.Categorical(
        model_summary["protocol"], categories=PROTOCOL_ORDER, ordered=True
    )
    model_summary["model"] = pd.Categorical(
        model_summary["model"], categories=MODEL_ORDER, ordered=True
    )
    model_summary = model_summary.sort_values(
        ["model", "protocol"], kind="stable"
    ).astype({"protocol": "object", "model": "object"}).reset_index(drop=True)

    validity = metric.copy()
    validity["spearman_valid"] = validity["spearman_uncertainty_error"].notna()
    validity["validity_reason"] = np.where(
        validity["spearman_valid"], "valid", "undefined_constant_uncertainty"
    )
    validity = validity.sort_values(keys, kind="stable").reset_index(drop=True)

    return {
        "family_tier_summary": _summarize_family_tiers(
            tier,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "model_protocol_summary": model_summary,
        "task_protocol_model_summary": task_summary,
        "spearman_validity": validity,
    }


def validate_full_plot_inputs(
    output_root: str | Path,
    *,
    scope: str = "all",
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    audit_root = output / "audit"
    aggregate_root = output / "aggregate"
    report_path = audit_root / "full_verification_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("verification_status")) != "passed":
        raise ValueError("Full calibration verification must pass before plotting.")
    if str(report.get("scope")) != str(scope):
        raise ValueError(
            f"Full plot scope mismatch: verified={report.get('scope')} requested={scope}"
        )
    expected = EXPECTED_COUNTS.get(str(scope))
    if expected is None:
        raise ValueError(f"Unsupported full plot scope: {scope}")
    if int(report.get("verified_unit_count", -1)) != int(expected["verified_units"]):
        raise ValueError("Full calibration verified unit count is not exact.")
    if int(report.get("failed_unit_count", -1)) != 0:
        raise ValueError("Full calibration verification contains failed units.")

    paths = {
        "tier": aggregate_root / "all_tier_error_summary.csv",
        "metrics": aggregate_root / "all_uncertainty_error_metrics.csv",
        "coverage": aggregate_root / "all_coverage_summary.csv",
        "sample": aggregate_root / "all_sample_level_calibration.csv",
        "root_manifest": output / "manifest.json",
        "verification": report_path,
        "frozen": output / "parameter_freeze" / "frozen_model_params_v1.yaml",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    tier = pd.read_csv(paths["tier"])
    metrics = pd.read_csv(paths["metrics"])
    coverage = pd.read_csv(paths["coverage"])
    root_manifest = json.loads(paths["root_manifest"].read_text(encoding="utf-8"))
    frozen = yaml.safe_load(paths["frozen"].read_text(encoding="utf-8")) or {}
    if len(tier) != int(expected["tier_rows"]):
        raise ValueError(f"Tier row count is not exact: {len(tier)}")
    if len(metrics) != int(expected["metric_rows"]):
        raise ValueError(f"Metric row count is not exact: {len(metrics)}")
    if len(coverage) != int(expected["coverage_rows"]):
        raise ValueError(f"Coverage row count is not exact: {len(coverage)}")
    if int(pd.to_numeric(coverage["missing_units"], errors="raise").sum()) != 0:
        raise ValueError("Full plot requires zero missing calibration units.")
    if tier["task_id"].astype(str).nunique() != int(expected["tasks"]):
        raise ValueError("Full plot task count is not exact.")
    if set(tier["protocol"].astype(str)) != set(PROTOCOL_ORDER):
        raise ValueError("Full plot protocols differ from the canonical order.")
    if set(tier["model_family"].astype(str)) != set(FAMILY_ORDER):
        raise ValueError("Full plot model families are incomplete.")
    if set(tier["model"].astype(str)) != set(MODEL_ORDER):
        raise ValueError("Full plot models are incomplete.")
    expected_na = expected.get("spearman_na")
    observed_na = int(metrics["spearman_uncertainty_error"].isna().sum())
    if expected_na is not None and observed_na != int(expected_na):
        raise ValueError(
            f"Spearman NA count changed: expected={expected_na} observed={observed_na}"
        )
    snapshot_ids = {
        str(report.get("snapshot_id", "")),
        str(root_manifest.get("snapshot_id", "")),
        str(frozen.get("snapshot_id", "")),
    }
    if len(snapshot_ids) != 1 or "" in snapshot_ids:
        raise ValueError(f"Frozen snapshot identity mismatch: {sorted(snapshot_ids)}")

    return {
        "output_root": output,
        "report": report,
        "root_manifest": root_manifest,
        "snapshot_id": snapshot_ids.pop(),
        "tier": tier,
        "metrics": metrics,
        "coverage": coverage,
        "paths": paths,
        "spearman_na_count": observed_na,
    }


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(
    figure: Any,
    base_path: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        output_paths.append(path)
    return output_paths


def _heatmap(
    ax: Any,
    frame: pd.DataFrame,
    *,
    row_order: tuple[str, ...],
    column_order: tuple[str, ...],
    value_column: str,
    vmin: float,
    vmax: float,
    cmap_name: str,
    title: str,
    annotation: dict[tuple[str, str], str] | None = None,
    row_labels: dict[str, str] | None = None,
) -> Any:
    from matplotlib import pyplot as plt

    pivot = frame.pivot(index="row", columns="column", values=value_column).reindex(
        index=row_order,
        columns=column_order,
    )
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(column_order)), column_order, rotation=45, ha="right")
    labels = [row_labels.get(row, row) if row_labels else row for row in row_order]
    ax.set_yticks(np.arange(len(row_order)), labels)
    ax.set_title(title)
    for row_index, row in enumerate(row_order):
        for column_index, column in enumerate(column_order):
            value = values[row_index, column_index]
            text = annotation.get((row, column), "") if annotation else ""
            if not text:
                text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and abs(value) > 0.62 * max(abs(vmin), abs(vmax)) else "black"
            ax.text(column_index, row_index, text, ha="center", va="center", fontsize=5.2, color=color)
    ax.set_xticks(np.arange(-0.5, len(column_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def _family_protocol_summary(task_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, protocol), group in task_summary.groupby(
        ["model_family", "protocol"], sort=False, dropna=False
    ):
        spearman = pd.to_numeric(group["spearman_uncertainty_error"], errors="coerce")
        rows.append(
            {
                "model_family": str(family),
                "protocol": str(protocol),
                "discrimination_mean": float(group["discrimination"].mean()),
                "spearman_mean": float(spearman.mean()) if spearman.notna().any() else np.nan,
                "spearman_valid_count": int(spearman.notna().sum()),
                "spearman_total_count": int(len(spearman)),
                "spearman_valid_percent": 100.0 * float(spearman.notna().mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_main(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    family_tier = tables["family_tier_summary"]
    task_summary = tables["task_protocol_model_summary"]
    family_protocol = _family_protocol_summary(task_summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)

    pooled = family_tier[family_tier["protocol"].astype(str).eq("All")]
    x = np.arange(len(TIER_ORDER))
    for family in FAMILY_ORDER:
        group = pooled[pooled["model_family"].astype(str).eq(family)].set_index(
            "confidence_tier"
        ).reindex(TIER_ORDER)
        mean = group["normalized_mae_mean"].to_numpy(float)
        low = group["bootstrap_ci95_low"].to_numpy(float)
        high = group["bootstrap_ci95_high"].to_numpy(float)
        axes[0, 0].plot(
            x,
            mean,
            marker="o",
            linewidth=1.5,
            color=FAMILY_COLORS[family],
            label=family,
        )
        axes[0, 0].fill_between(x, low, high, color=FAMILY_COLORS[family], alpha=0.14)
    axes[0, 0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, TIER_ORDER)
    axes[0, 0].set_ylabel("Normalized MAE")
    axes[0, 0].set_title("Confidence-tier error discrimination")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 0].grid(axis="y", alpha=0.22)

    discrimination = family_protocol.rename(
        columns={"model_family": "row", "protocol": "column"}
    )
    image = _heatmap(
        axes[0, 1],
        discrimination,
        row_order=FAMILY_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="discrimination_mean",
        vmin=-1.25,
        vmax=1.25,
        cmap_name="RdBu_r",
        title="Low-tier minus high-tier normalized MAE",
    )
    fig.colorbar(image, ax=axes[0, 1], fraction=0.045, pad=0.02)

    spearman_annotations = {
        (str(row.model_family), str(row.protocol)): (
            "NA"
            if not np.isfinite(row.spearman_mean)
            else f"{row.spearman_mean:.2f}\n{int(row.spearman_valid_count)}/{int(row.spearman_total_count)}"
        )
        for row in family_protocol.itertuples()
    }
    spearman = family_protocol.rename(
        columns={"model_family": "row", "protocol": "column"}
    )
    image = _heatmap(
        axes[1, 0],
        spearman,
        row_order=FAMILY_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="spearman_mean",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        title="Mean Spearman uncertainty-error correlation",
        annotation=spearman_annotations,
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.045, pad=0.02)

    validity = family_protocol.rename(
        columns={"model_family": "row", "protocol": "column"}
    )
    validity_annotations = {
        (str(row.model_family), str(row.protocol)): (
            f"{row.spearman_valid_percent:.0f}%\n{int(row.spearman_valid_count)}/{int(row.spearman_total_count)}"
        )
        for row in family_protocol.itertuples()
    }
    image = _heatmap(
        axes[1, 1],
        validity,
        row_order=FAMILY_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="spearman_valid_percent",
        vmin=0.0,
        vmax=100.0,
        cmap_name="Blues",
        title="Valid Spearman coverage",
        annotation=validity_annotations,
    )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.02)

    for label, ax in zip("ABCD", axes.ravel()):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    return _close_and_return(
        fig,
        output_dir / "calibration_main",
        formats,
        dpi,
    )


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


def _plot_by_protocol(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    summary = tables["family_tier_summary"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.3), sharex=True, sharey=True)
    x = np.arange(len(TIER_ORDER))
    handles: list[Any] = []
    labels: list[str] = []
    for ax, protocol in zip(axes.ravel(), PROTOCOL_ORDER):
        panel = summary[summary["protocol"].astype(str).eq(protocol)]
        for family in FAMILY_ORDER:
            group = panel[panel["model_family"].astype(str).eq(family)].set_index(
                "confidence_tier"
            ).reindex(TIER_ORDER)
            line = ax.plot(
                x,
                group["normalized_mae_mean"],
                marker="o",
                linewidth=1.25,
                markersize=3,
                color=FAMILY_COLORS[family],
                label=family,
            )[0]
            ax.fill_between(
                x,
                group["bootstrap_ci95_low"].to_numpy(float),
                group["bootstrap_ci95_high"].to_numpy(float),
                color=FAMILY_COLORS[family],
                alpha=0.12,
            )
            if not handles:
                pass
            if family not in labels:
                handles.append(line)
                labels.append(family)
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(protocol)
        ax.set_xticks(x, TIER_ORDER)
        ax.grid(axis="y", alpha=0.2)
    axes[1, 3].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized MAE")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _close_and_return(
        fig,
        output_dir / "calibration_by_protocol",
        formats,
        dpi,
    )


def _plot_by_model(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    summary = tables["model_protocol_summary"].rename(
        columns={"model": "row", "protocol": "column"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.3), constrained_layout=True)
    image = _heatmap(
        axes[0],
        summary,
        row_order=MODEL_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="discrimination_mean",
        vmin=-1.25,
        vmax=1.25,
        cmap_name="RdBu_r",
        title="Confidence discrimination by model",
        row_labels=MODEL_LABELS,
    )
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    annotations = {
        (str(row.row), str(row.column)): (
            "NA"
            if not np.isfinite(row.spearman_mean)
            else f"{row.spearman_mean:.2f}\n{int(row.spearman_valid_count)}/{int(row.spearman_total_count)}"
        )
        for row in summary.itertuples()
    }
    image = _heatmap(
        axes[1],
        summary,
        row_order=MODEL_ORDER,
        column_order=PROTOCOL_ORDER,
        value_column="spearman_mean",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="RdBu_r",
        title="Mean Spearman correlation by model",
        annotation=annotations,
        row_labels=MODEL_LABELS,
    )
    fig.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
    for label, ax in zip("AB", axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")
    return _close_and_return(
        fig,
        output_dir / "calibration_by_model",
        formats,
        dpi,
    )


def _model_styles() -> dict[str, dict[str, Any]]:
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")
    line_styles = ("-", "--", "-.", ":")
    return {
        model: {
            "color": FAMILY_COLORS[MODEL_FAMILIES[model]],
            "marker": markers[index],
            "linestyle": line_styles[index % len(line_styles)],
        }
        for index, model in enumerate(MODEL_ORDER)
    }


def _plot_task_appendix(
    tier_summary: pd.DataFrame,
    output_root: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    from matplotlib import pyplot as plt

    output_paths: list[Path] = []
    task_dirs: dict[str, str] = {}
    styles = _model_styles()
    tasks = sorted(tier_summary["task_id"].astype(str).unique())
    x = np.arange(len(TIER_ORDER))
    for task_id in tasks:
        task = tier_summary[tier_summary["task_id"].astype(str).eq(task_id)]
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.8), sharex=True, sharey=True)
        handles: list[Any] = []
        labels: list[str] = []
        for ax, protocol in zip(axes.ravel(), PROTOCOL_ORDER):
            panel = task[task["protocol"].astype(str).eq(protocol)]
            for model in MODEL_ORDER:
                group = panel[panel["model"].astype(str).eq(model)].copy()
                if group.empty:
                    continue
                group["confidence_tier"] = pd.Categorical(
                    group["confidence_tier"], categories=TIER_ORDER, ordered=True
                )
                group = group.sort_values("confidence_tier")
                style = styles[model]
                line = ax.plot(
                    x,
                    group["normalized_mae"].to_numpy(float),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=0.85,
                    markersize=2.5,
                    alpha=0.86,
                    label=model,
                )[0]
                if model not in labels:
                    handles.append(line)
                    labels.append(model)
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.6)
            ax.set_title(protocol)
            ax.set_xticks(x, TIER_ORDER)
            ax.grid(axis="y", alpha=0.18)
        axes[1, 3].axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("Normalized MAE")
        fig.suptitle(task_id, y=0.99, fontsize=9)
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=5.5)
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


def _validate_png(path: Path, dpi: int) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        width, height = image.size
        observed_dpi = image.info.get("dpi", (0, 0))
        grayscale = image.convert("L")
        variance = float(ImageStat.Stat(grayscale).var[0])
    if width < 1000 or height < 700:
        raise ValueError(f"PNG dimensions are too small: {path} -> {width}x{height}")
    if min(float(observed_dpi[0]), float(observed_dpi[1])) < float(dpi) - 2:
        raise ValueError(f"PNG DPI is too small: {path} -> {observed_dpi}")
    if variance <= 1.0:
        raise ValueError(f"PNG appears blank: {path}")
    return {
        "width": width,
        "height": height,
        "dpi": [float(observed_dpi[0]), float(observed_dpi[1])],
        "grayscale_variance": variance,
    }


def _validate_output_file(path: Path, dpi: int) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size < 1024:
        raise ValueError(f"Figure output is missing or empty: {path}")
    if path.suffix.lower() == ".png":
        return _validate_png(path, dpi)
    if path.suffix.lower() == ".pdf" and not path.read_bytes().startswith(b"%PDF"):
        raise ValueError(f"Invalid PDF output: {path}")
    if path.suffix.lower() == ".svg":
        prefix = path.read_text(encoding="utf-8")[:1000].lower()
        if "<svg" not in prefix:
            raise ValueError(f"Invalid SVG output: {path}")
    return {"size_bytes": path.stat().st_size}


def _write_readme(path: Path) -> None:
    path.write_text(
        """# Full Calibration Figure Package

This package is generated only from the verified full calibration aggregate.

- `overview/calibration_main`: pooled family-level calibration summary.
- `overview/calibration_by_protocol`: family curves for all seven protocols.
- `overview/calibration_by_model`: model-level discrimination and correlation heatmaps.
- `by_task`: ten task-specific appendix figures.
- `data`: exact plotting tables used for the figures.

Normalized MAE is relative to the overall MAE for each task/protocol/model. The
confidence discrimination score is Low-tier normalized MAE minus High-tier
normalized MAE. Positive values indicate that uncertainty tiers successfully
separate higher-error samples. Undefined Spearman correlations are shown as NA
and retained in `data/spearman_validity.csv`.

The diagnostic `failed_attempts` directory is not a source for any figure in
this package.
""",
        encoding="ascii",
    )


def _publish_package(ready_dir: Path, final_dir: Path) -> None:
    figures_root = final_dir.parent
    archive_root = figures_root / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archived: Path | None = None
    if final_dir.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = archive_root / stamp
        _replace_with_retry(final_dir, archived)
    try:
        _replace_with_retry(ready_dir, final_dir)
    except Exception:
        if archived is not None and archived.exists() and not final_dir.exists():
            _replace_with_retry(archived, final_dir)
        raise


def build_full_calibration_figures(
    output_root: str | Path,
    *,
    scope: str = "all",
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = validate_full_plot_inputs(output_root, scope=scope)
    output = Path(validated["output_root"])
    options = {**DEFAULT_SETTINGS, **dict(settings or {})}
    formats = tuple(str(value).lower() for value in options["formats"])
    if not formats or any(value not in {"png", "pdf", "svg"} for value in formats):
        raise ValueError(f"Unsupported full plot formats: {formats}")
    dpi = int(options["dpi"])
    bootstrap_iterations = int(options["bootstrap_iterations"])
    bootstrap_seed = int(options["bootstrap_seed"])
    include_task_appendix = bool(options["include_task_appendix"])
    if dpi < 300:
        raise ValueError("Full calibration PNG delivery requires at least 300 DPI.")
    if bootstrap_iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")

    _configure_matplotlib()
    figures_root = output / "figures"
    final_dir = output / str(options["output_dir"])
    if final_dir.parent != figures_root:
        raise ValueError(f"Full plot output must be a direct figures child: {final_dir}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ready_dir = figures_root / f".final.ready_{stamp}"
    if ready_dir.exists():
        raise FileExistsError(ready_dir)
    ready_dir.mkdir(parents=True)

    manifest = validated["root_manifest"]
    split_root = Path(str(manifest["split_root"])).resolve()
    protected_paths = {
        "aggregate": output / "aggregate",
        "frozen_v1": validated["paths"]["frozen"],
        "canonical_splits": split_root,
        "failed_attempts": output / "failed_attempts",
    }
    before_hashes = {
        name: _file_sha256(path) if Path(path).is_file() else _tree_sha256(path)
        for name, path in protected_paths.items()
    }

    tables = prepare_plot_tables(
        validated["tier"],
        validated["metrics"],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    data_dir = ready_dir / "data"
    overview_dir = ready_dir / "overview"
    by_task_dir = ready_dir / "by_task"
    for directory in (data_dir, overview_dir, by_task_dir):
        directory.mkdir(parents=True, exist_ok=True)
    data_paths = {
        name: data_dir / f"{name}.csv"
        for name in (
            "family_tier_summary",
            "model_protocol_summary",
            "task_protocol_model_summary",
            "spearman_validity",
        )
    }
    for name, path in data_paths.items():
        tables[name].to_csv(path, index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(_plot_main(tables, overview_dir, formats=formats, dpi=dpi))
    figure_paths.extend(
        _plot_by_protocol(tables, overview_dir, formats=formats, dpi=dpi)
    )
    figure_paths.extend(_plot_by_model(tables, overview_dir, formats=formats, dpi=dpi))
    task_dirs: dict[str, str] = {}
    if include_task_appendix:
        task_paths, task_dirs = _plot_task_appendix(
            validated["tier"],
            by_task_dir,
            formats=formats,
            dpi=dpi,
        )
        figure_paths.extend(task_paths)
    _write_readme(ready_dir / "README.md")

    after_hashes = {
        name: _file_sha256(path) if Path(path).is_file() else _tree_sha256(path)
        for name, path in protected_paths.items()
    }
    if before_hashes != after_hashes:
        raise ValueError(
            f"Protected calibration sources changed during plotting: before={before_hashes} after={after_hashes}"
        )

    qa = {
        str(path.relative_to(ready_dir)): _validate_output_file(path, dpi)
        for path in figure_paths
    }
    expected_logical = 3 + (int(validated["tier"]["task_id"].nunique()) if include_task_appendix else 0)
    expected_files = expected_logical * len(formats)
    if len(figure_paths) != expected_files:
        raise ValueError(
            f"Figure output count mismatch: expected={expected_files} observed={len(figure_paths)}"
        )

    output_files = sorted(
        path for path in ready_dir.rglob("*") if path.is_file() and path.name != "figure_manifest.json"
    )
    output_hashes = {
        str(path.relative_to(ready_dir)): _file_sha256(path) for path in output_files
    }
    payload = {
        "schema_version": 1,
        "stage": "full-plot",
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": validated["snapshot_id"],
        "source_verification_status": validated["report"]["verification_status"],
        "verified_unit_count": int(validated["report"]["verified_unit_count"]),
        "tier_summary_count": int(len(validated["tier"])),
        "uncertainty_metric_count": int(len(validated["metrics"])),
        "spearman_na_count": int(validated["spearman_na_count"]),
        "task_count": int(validated["tier"]["task_id"].nunique()),
        "protocols": list(PROTOCOL_ORDER),
        "model_families": list(FAMILY_ORDER),
        "models": list(MODEL_ORDER),
        "confidence_tiers": list(TIER_ORDER),
        "statistics": {
            "normalized_mae_weighting": "equal task-model observations",
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "discrimination_definition": "Low normalized MAE - High normalized MAE",
            "spearman_na_policy": "retain as NA and report valid/total counts",
        },
        "formats": list(formats),
        "png_dpi": dpi,
        "logical_figure_count": expected_logical,
        "figure_output_count": len(figure_paths),
        "task_outputs": task_dirs,
        "protected_source_hashes": before_hashes,
        "source_files": {
            name: {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for name, path in validated["paths"].items()
        },
        "output_file_hashes": output_hashes,
        "image_qa": qa,
    }
    (ready_dir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _publish_package(ready_dir, final_dir)
    return {
        "output_dir": str(final_dir),
        "logical_figure_count": expected_logical,
        "output_file_count": len(figure_paths),
        "task_count": int(validated["tier"]["task_id"].nunique()),
        "spearman_na_count": int(validated["spearman_na_count"]),
        "manifest_path": str(final_dir / "figure_manifest.json"),
    }
