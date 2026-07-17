from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import yaml

from src.training_size_ood.aggregation import (
    degradation_ratio_summary,
    family_curve_summary,
    sensitivity_summary,
)
from src.training_size_ood.core import FAILURE_AXIS_LOCO, FAILURE_AXIS_TOP20

from .common import StageResult, empty_csv, ensure_dirs, read_source_csv, source_path
from .config import FigureConfig, list_section_sources
from .manifest import write_manifest


REQUIRED_COLUMNS = [
    "task_id",
    "model_family",
    "model",
    "failure_axis",
    "training_fraction",
    "repeat_id",
    "random_cv_mae",
    "ood_mae",
]

FAMILY_ORDER = ["Traditional", "BERT", "TabPFN", "LLMProp"]
FAMILY_COLORS = {
    "Traditional": "#3572A5",
    "BERT": "#D1495B",
    "TabPFN": "#2A9D8F",
    "LLMProp": "#8E6C8A",
}


def load_degradation_by_repeat(cfg: FigureConfig) -> pd.DataFrame:
    frames = []
    for entry in list_section_sources(cfg, "figure4", "degradation_sources"):
        path = source_path(entry)
        if not path.exists():
            continue
        frame = read_source_csv(entry)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Figure 4 source {entry.get('name')} missing columns: {missing}")
        frames.append(frame)
    generated_path = _training_output_root(cfg) / "csv" / "degradation_ratio_by_repeat.csv"
    strict = bool(dict(cfg.raw.get("figure4") or {}).get("require_complete_training_size", False))
    if not frames and generated_path.exists() and not strict:
        frames.append(pd.read_csv(generated_path))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["degradation_ratio"])
    data = pd.concat(frames, ignore_index=True)
    for column in ["training_fraction", "repeat_id", "random_cv_mae", "ood_mae"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["training_fraction", "random_cv_mae", "ood_mae"])
    if (data["random_cv_mae"] <= 0).any():
        raise ValueError("Figure 4 source contains non-positive Random-CV MAE values")
    data["degradation_ratio"] = data["ood_mae"] / data["random_cv_mae"]
    return data


def validate_complete_training_source(cfg: FigureConfig, data: pd.DataFrame) -> None:
    figure_cfg = dict(cfg.raw.get("figure4") or {})
    if not bool(figure_cfg.get("require_complete_training_size", False)):
        return
    sources = [
        source_path(entry)
        for entry in list_section_sources(cfg, "figure4", "degradation_sources")
        if source_path(entry).exists()
    ]
    if len(sources) != 1:
        raise ValueError("Strict Figure 4 requires exactly one canonical degradation source")
    coverage_path = sources[0].parent / "coverage_report.csv"
    if not coverage_path.exists():
        raise ValueError(f"Strict Figure 4 coverage report is missing: {coverage_path}")
    coverage = pd.read_csv(coverage_path)
    if len(coverage) != 1:
        raise ValueError("Strict Figure 4 coverage report must contain exactly one row")
    row = coverage.iloc[0]
    expected_runs = int(figure_cfg.get("expected_run_count", 4510))
    expected_rows = int(figure_cfg.get("expected_ratio_row_count", expected_runs * 2))
    incomplete_counts = {
        name: int(pd.to_numeric(row.get(name, 0), errors="raise"))
        for name in [
            "failed_run_count",
            "missing_run_count",
            "stale_run_count",
            "incomplete_run_count",
        ]
    }
    if (
        int(pd.to_numeric(row.get("expected_run_count"), errors="raise")) != expected_runs
        or int(pd.to_numeric(row.get("complete_run_count"), errors="raise")) != expected_runs
        or any(incomplete_counts.values())
    ):
        raise ValueError(
            "Figure 4 training-size coverage is incomplete: "
            f"expected={expected_runs}, complete={row.get('complete_run_count')}, "
            f"non_complete={incomplete_counts}"
        )
    strict_columns = {
        "snapshot_id",
        "parameter_hash",
        "subset_hash",
        "random_cv_test_hash",
        "top20_test_hash",
        "loco_test_hash",
        "training_pool_hash",
    }
    missing = strict_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Strict Figure 4 source is missing columns: {sorted(missing)}")
    if len(data) != expected_rows:
        raise ValueError(f"Strict Figure 4 requires {expected_rows} ratio rows, found {len(data)}")
    run_keys = ["task_id", "model_family", "model", "training_fraction", "repeat_id"]
    if data.duplicated(run_keys + ["failure_axis"]).any():
        raise ValueError("Strict Figure 4 source contains duplicate run/failure-axis rows")
    axis_sets = data.groupby(run_keys, dropna=False)["failure_axis"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    expected_axes = frozenset({FAILURE_AXIS_LOCO, FAILURE_AXIS_TOP20})
    if len(axis_sets) != expected_runs or not axis_sets.map(lambda value: value == expected_axes).all():
        raise ValueError("Strict Figure 4 source does not contain both failure axes for every run")
    snapshots = data["snapshot_id"].dropna().astype(str).unique()
    if len(snapshots) != 1 or not snapshots[0]:
        raise ValueError("Strict Figure 4 source must contain one non-empty snapshot ID")
    training_root_value = figure_cfg.get("training_output_root")
    frozen_path_value = figure_cfg.get("frozen_parameters_path")
    if not training_root_value or not frozen_path_value:
        raise ValueError("Strict Figure 4 requires training_output_root and frozen_parameters_path")
    training_root = Path(str(training_root_value))
    frozen_path = Path(str(frozen_path_value))
    if not training_root.is_absolute():
        training_root = Path.cwd() / training_root
    if not frozen_path.is_absolute():
        frozen_path = Path.cwd() / frozen_path
    experiment_manifest_path = training_root / "experiment_manifest.json"
    locked_frozen_path = training_root / "locked_inputs" / "frozen_model_params_v1.yaml"
    if not experiment_manifest_path.exists() or not frozen_path.exists() or not locked_frozen_path.exists():
        raise ValueError("Strict Figure 4 snapshot-lock artifacts are missing")
    experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
    frozen_payload = yaml.safe_load(frozen_path.read_text(encoding="utf-8")) or {}
    expected_snapshot = str(experiment_manifest.get("snapshot_id", ""))
    if expected_snapshot != snapshots[0] or str(frozen_payload.get("snapshot_id", "")) != snapshots[0]:
        raise ValueError("Strict Figure 4 snapshot ID does not match the locked experiment")
    if frozen_path.read_bytes() != locked_frozen_path.read_bytes():
        raise ValueError("Strict Figure 4 frozen-parameter file differs from the locked snapshot")
    for column in [
        "random_cv_test_hash",
        "top20_test_hash",
        "loco_test_hash",
        "training_pool_hash",
    ]:
        if data.groupby("task_id", dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Strict Figure 4 source changes fixed {column} within a task")
    subset_groups = data.groupby(
        ["task_id", "training_fraction", "repeat_id"],
        dropna=False,
    )["subset_hash"].nunique(dropna=False)
    if subset_groups.gt(1).any():
        raise ValueError("Strict Figure 4 source changes training subsets across models")


def build_figure4(family_summary: pd.DataFrame, sensitivity: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    _draw_protocol_panel(ax_a)
    _draw_curve_panel(
        ax_b,
        family_summary[family_summary["failure_axis"].eq(FAILURE_AXIS_LOCO)],
        title="Compositional extreme (farthest LOCO)",
    )
    _draw_curve_panel(
        ax_c,
        family_summary[family_summary["failure_axis"].eq(FAILURE_AXIS_TOP20)],
        title="Property extreme (top 20%)",
    )
    _draw_sensitivity_panel(ax_d, sensitivity)
    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    return fig


def model_level_curve_summary(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_family",
        "model",
        "failure_axis",
        "training_fraction",
        "task_count",
        "mean_degradation_ratio",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    task_means = (
        data.groupby(
            ["task_id", "model_family", "model", "failure_axis", "training_fraction"],
            dropna=False,
        )["degradation_ratio"]
        .mean()
        .reset_index()
    )
    summary = (
        task_means.groupby(
            ["model_family", "model", "failure_axis", "training_fraction"],
            dropna=False,
        )
        .agg(
            task_count=("task_id", "nunique"),
            mean_degradation_ratio=("degradation_ratio", "mean"),
        )
        .reset_index()
    )
    return summary[columns]


def plot_figure4(
    family_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if family_summary.empty:
        return
    fig = build_figure4(family_summary, sensitivity)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            figures_dir / f"figure4_degradation_vs_training_size.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_model_level_supplement(data: pd.DataFrame, figures_dir: Path) -> None:
    if data.empty:
        return
    summary = model_level_curve_summary(data)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, axis, title in [
        (axes[0], FAILURE_AXIS_LOCO, "Farthest LOCO"),
        (axes[1], FAILURE_AXIS_TOP20, "Top 20% property extreme"),
    ]:
        panel = summary[summary["failure_axis"].eq(axis)]
        for (family, model), group in panel.groupby(["model_family", "model"], dropna=False):
            ordered = group.sort_values("training_fraction")
            ax.plot(
                ordered["training_fraction"] * 100,
                ordered["mean_degradation_ratio"],
                marker="o",
                linewidth=1.1,
                markersize=3.5,
                color=FAMILY_COLORS.get(str(family), "#555555"),
                alpha=0.8,
                label=f"{family}: {model}",
            )
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Training-set size (%)")
        ax.set_ylabel("MAE(OOD) / MAE(Random-CV)")
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(figures_dir / f"figure4_model_level_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task_level_supplements(data: pd.DataFrame, figures_dir: Path) -> None:
    if data.empty:
        return
    task_dir = figures_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for task_id, task_data in data.groupby("task_id", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
        for ax, axis, title in [
            (axes[0], FAILURE_AXIS_LOCO, "Farthest LOCO"),
            (axes[1], FAILURE_AXIS_TOP20, "Top 20% property extreme"),
        ]:
            panel = task_data[task_data["failure_axis"].eq(axis)]
            summary = (
                panel.groupby(["model_family", "training_fraction"], dropna=False)[
                    "degradation_ratio"
                ]
                .mean()
                .reset_index()
            )
            for family in FAMILY_ORDER:
                group = summary[summary["model_family"].eq(family)].sort_values("training_fraction")
                if group.empty:
                    continue
                ax.plot(
                    group["training_fraction"] * 100,
                    group["degradation_ratio"],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=FAMILY_COLORS[family],
                    label=family,
                )
            ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Training-set size (%)")
            ax.set_ylabel("MAE(OOD) / MAE(Random-CV)")
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.grid(axis="y", alpha=0.2)
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(str(task_id), fontsize=11)
        filename = _safe_filename(str(task_id))
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(task_dir / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_task_level_raw_mae_supplements(data: pd.DataFrame, figures_dir: Path) -> None:
    if data.empty:
        return
    task_dir = figures_dir / "raw_mae_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for task_id, task_data in data.groupby("task_id", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        for ax, axis, title in [
            (axes[0], FAILURE_AXIS_LOCO, "Farthest LOCO"),
            (axes[1], FAILURE_AXIS_TOP20, "Top 20% property extreme"),
        ]:
            panel = task_data[task_data["failure_axis"].eq(axis)]
            summary = (
                panel.groupby(["model_family", "training_fraction"], dropna=False)[
                    ["random_cv_mae", "ood_mae"]
                ]
                .mean()
                .reset_index()
            )
            for family in FAMILY_ORDER:
                group = summary[summary["model_family"].eq(family)].sort_values(
                    "training_fraction"
                )
                if group.empty:
                    continue
                x = group["training_fraction"] * 100
                color = FAMILY_COLORS[family]
                ax.plot(
                    x,
                    group["ood_mae"],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=color,
                    label=f"{family} OOD",
                )
                ax.plot(
                    x,
                    group["random_cv_mae"],
                    linewidth=1.2,
                    linestyle="--",
                    color=color,
                    alpha=0.85,
                    label=f"{family} Random-CV",
                )
            ax.set_title(title)
            ax.set_xlabel("Training-set size (%)")
            ax.set_ylabel("MAE (task units)")
            ax.set_xticks([20, 40, 60, 80, 100])
            ax.grid(axis="y", alpha=0.2)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
        fig.suptitle(str(task_id), fontsize=11)
        filename = _safe_filename(str(task_id))
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(task_dir / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_figure4(cfg: FigureConfig) -> StageResult:
    stage_dir = _training_output_root(cfg)
    csv_dir = stage_dir / "csv"
    figures_dir = stage_dir / "figures"
    ensure_dirs(csv_dir, figures_dir)
    data = load_degradation_by_repeat(cfg)
    validate_complete_training_source(cfg, data)
    figure_cfg = dict(cfg.raw.get("figure4") or {})
    iterations = int(figure_cfg.get("bootstrap_iterations", 10_000))
    random_seed = int(figure_cfg.get("random_seed", 20260707))
    model_summary = degradation_ratio_summary(data)
    family_summary = family_curve_summary(
        data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    sensitivity = sensitivity_summary(
        data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    empty_csv(csv_dir / "training_size_sample_level_predictions.csv", [])
    data.to_csv(csv_dir / "degradation_ratio_by_repeat.csv", index=False)
    model_summary.to_csv(csv_dir / "degradation_ratio_summary.csv", index=False)
    model_summary.to_csv(csv_dir / "task_model_curve_summary.csv", index=False)
    family_summary.to_csv(csv_dir / "family_curve_summary.csv", index=False)
    sensitivity.to_csv(csv_dir / "sensitivity_summary.csv", index=False)
    plot_figure4(family_summary, sensitivity, figures_dir)
    plot_model_level_supplement(data, figures_dir)
    plot_task_level_supplements(data, figures_dir)
    plot_task_level_raw_mae_supplements(data, figures_dir)
    sources = [
        source_path(entry)
        for entry in list_section_sources(cfg, "figure4", "degradation_sources")
        if source_path(entry).exists()
    ]
    write_manifest(
        stage_dir / "manifest.json",
        stage="figure4",
        config_path=cfg.config_path,
        output_root=stage_dir,
        payload={
            "row_count": int(len(data)),
            "model_summary_rows": int(len(model_summary)),
            "family_summary_rows": int(len(family_summary)),
            "sensitivity_rows": int(len(sensitivity)),
            "bootstrap_iterations": iterations,
            "random_seed": random_seed,
        },
        source_files=sources,
    )
    return StageResult(stage="figure4", output_dir=stage_dir, row_count=len(data))


def _training_output_root(cfg: FigureConfig) -> Path:
    value = dict(cfg.raw.get("figure4") or {}).get(
        "training_output_root",
        "output/training_size_ood",
    )
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _draw_protocol_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    boxes = {
        "full": (0.05, 0.78, 0.34, 0.12, "Full dataset"),
        "top": (0.61, 0.78, 0.34, 0.12, "OOD: top 20% target"),
        "remaining": (0.05, 0.54, 0.34, 0.12, "Remaining ~80%"),
        "loco": (0.61, 0.54, 0.34, 0.12, "OOD: farthest LOCO"),
        "candidate": (0.05, 0.30, 0.34, 0.12, "Remaining candidates"),
        "random": (0.61, 0.30, 0.34, 0.12, "Fixed Random-CV test"),
        "train": (0.29, 0.06, 0.42, 0.12, "Training pool: 20/40/60/80/100%"),
    }
    fills = {
        "top": "#F4D6D9",
        "loco": "#D6E8F4",
        "random": "#DCEED8",
        "train": "#F2E8C9",
    }
    for name, (x, y, width, height, label) in boxes.items():
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=fills.get(name, "#F3F3F3"),
                edgecolor="#4A4A4A",
                linewidth=0.9,
            )
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    arrows = [
        ("full", "top"),
        ("full", "remaining"),
        ("remaining", "loco"),
        ("remaining", "candidate"),
        ("candidate", "random"),
        ("candidate", "train"),
    ]
    for source, target in arrows:
        sx, sy, sw, sh, _ = boxes[source]
        tx, ty, tw, th, _ = boxes[target]
        start = (sx + sw / 2, sy)
        end = (tx + tw / 2, ty + th)
        if abs((tx + tw / 2) - (sx + sw / 2)) > 0.2:
            start = (sx + sw, sy + sh / 2)
            end = (tx, ty + th / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10, linewidth=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fixed split and training-size protocol", fontsize=11)


def _draw_curve_panel(ax: plt.Axes, panel: pd.DataFrame, *, title: str) -> None:
    for family in FAMILY_ORDER:
        group = panel[panel["model_family"].eq(family)].sort_values("training_fraction")
        if group.empty:
            continue
        x = group["training_fraction"].to_numpy(dtype=float) * 100
        y = group["mean_degradation_ratio"].to_numpy(dtype=float)
        low = group["ci95_low"].to_numpy(dtype=float)
        high = group["ci95_high"].to_numpy(dtype=float)
        color = FAMILY_COLORS[family]
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.5, color=color, label=family)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Training-set size (%)")
    ax.set_ylabel("MAE(OOD) / MAE(Random-CV)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def _draw_sensitivity_panel(ax: plt.Axes, sensitivity: pd.DataFrame) -> None:
    positions = np.arange(len(FAMILY_ORDER), dtype=float)
    axis_styles = [
        (FAILURE_AXIS_LOCO, -0.12, "o", "Farthest LOCO"),
        (FAILURE_AXIS_TOP20, 0.12, "s", "Top 20% target"),
    ]
    for axis, offset, marker, label in axis_styles:
        panel = sensitivity[sensitivity["failure_axis"].eq(axis)].set_index("model_family")
        xs: list[float] = []
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        colors: list[str] = []
        for index, family in enumerate(FAMILY_ORDER):
            if family not in panel.index:
                continue
            row = panel.loc[family]
            xs.append(float(index + offset))
            ys.append(float(row["sensitivity"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))
            colors.append(FAMILY_COLORS[family])
        if xs:
            y = np.asarray(ys)
            yerr = np.vstack([y - np.asarray(lows), np.asarray(highs) - y])
            for x_value, y_value, error, color in zip(xs, ys, yerr.T, colors):
                ax.errorbar(
                    x_value,
                    y_value,
                    yerr=np.asarray(error).reshape(2, 1),
                    marker=marker,
                    color=color,
                    markerfacecolor=color,
                    markersize=5,
                    capsize=3,
                    linewidth=1.2,
                )
            ax.plot([], [], marker=marker, color="#555555", linestyle="none", label=label)
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(positions, FAMILY_ORDER)
    ax.set_ylabel("Sensitivity (-slope)")
    ax.set_title("Data-scarcity sensitivity")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
