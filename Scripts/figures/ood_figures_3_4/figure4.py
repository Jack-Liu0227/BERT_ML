from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml

from src.training_size_ood.aggregation import (
    degradation_ratio_summary,
    family_curve_summary,
    selected_method_curve_summary,
    selected_method_sensitivity_summary,
    sensitivity_summary,
)
from src.training_size_ood.core import FAILURE_AXIS_LOCO, FAILURE_AXIS_TOP20
from src.training_size_ood.explainer import build_figure4_explainer
from src.training_size_ood.selection import (
    DISPLAY_METHODS,
    apply_representative_selection,
    select_representative_models,
    validate_display_methods,
)

from .common import StageResult, empty_csv, ensure_dirs, read_source_csv, source_path
from .config import FigureConfig, list_section_sources
from .figure4_panel_a import draw_panel_a
from .manifest import write_manifest
from .training_size_task_figures import (
    DEFAULT_EXCLUDED_TASKS,
    prepare_display_data,
    render_training_size_task_outputs,
)


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

DISPLAY_METHOD_ORDER = list(DISPLAY_METHODS)
DISPLAY_METHOD_COLORS = {
    "Traditional-best": "#356F9F",
    "BERT-best": "#D0525F",
    "TabPFN Numeric": "#238B7E",
    "TabPFN Text": "#71B7A8",
    "LLM-Prop": "#8B6F8F",
}
DISPLAY_METHOD_MARKERS = {
    "Traditional-best": "o",
    "BERT-best": "s",
    "TabPFN Numeric": "^",
    "TabPFN Text": "D",
    "LLM-Prop": "P",
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


def build_figure4(
    family_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    selected_summary: pd.DataFrame | None = None,
    selected_sensitivity: pd.DataFrame | None = None,
) -> plt.Figure:
    selected = selected_summary is not None and not selected_summary.empty
    curve_summary = selected_summary if selected else family_summary
    sensitivity_data = selected_sensitivity if selected else sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_panel_a(ax_a)
    _draw_curve_panel(
        ax_b,
        curve_summary[curve_summary["failure_axis"].eq(FAILURE_AXIS_LOCO)],
        title="Farthest LOCO",
        selected=selected,
    )
    _draw_curve_panel(
        ax_c,
        curve_summary[curve_summary["failure_axis"].eq(FAILURE_AXIS_TOP20)],
        title="Top-20% property extreme",
        selected=selected,
    )
    _draw_sensitivity_panel(ax_d, sensitivity_data, selected=selected)
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


def task_raw_ood_mae_curve_summary(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    """Summarize absolute OOD MAE without combining differently scaled tasks."""
    columns = [
        "task_id",
        "display_method",
        "failure_axis",
        "training_fraction",
        "repeat_count",
        "mean_ood_mae",
        "ci95_low",
        "ci95_high",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    working = _with_display_method(data)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    group_columns = ["task_id", "display_method", "failure_axis", "training_fraction"]
    for keys, group in working.groupby(group_columns, dropna=False, sort=True):
        values = group.groupby("repeat_id", dropna=False)["ood_mae"].mean().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        point = float(values.mean())
        if len(values) == 1 or bootstrap_iterations <= 0:
            low = high = point
        else:
            sampled = rng.choice(values, size=(bootstrap_iterations, len(values)), replace=True)
            draws = sampled.mean(axis=1)
            low, high = np.percentile(draws, [2.5, 97.5]).astype(float)
        rows.append(
            {
                "task_id": keys[0],
                "display_method": keys[1],
                "failure_axis": keys[2],
                "training_fraction": float(keys[3]),
                "repeat_count": int(len(values)),
                "mean_ood_mae": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def task_raw_ood_mae_sensitivity_summary(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    """Fit absolute OOD MAE = a + b log(training fraction) within each task."""
    columns = [
        "task_id",
        "display_method",
        "failure_axis",
        "fraction_count",
        "sensitivity",
        "ci95_low",
        "ci95_high",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    working = _with_display_method(data)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for keys, group in working.groupby(
        ["task_id", "display_method", "failure_axis"],
        dropna=False,
        sort=True,
    ):
        fractions: list[float] = []
        repeat_values: list[np.ndarray] = []
        for fraction, fraction_group in group.groupby("training_fraction", dropna=False, sort=True):
            values = (
                fraction_group.groupby("repeat_id", dropna=False)["ood_mae"]
                .mean()
                .to_numpy(dtype=float)
            )
            values = values[np.isfinite(values)]
            if float(fraction) > 0 and len(values):
                fractions.append(float(fraction))
                repeat_values.append(values)
        if len(fractions) < 2:
            continue
        x = np.log(np.asarray(fractions, dtype=float))
        point_y = np.asarray([values.mean() for values in repeat_values], dtype=float)
        point = -_linear_slope(x, point_y)
        if bootstrap_iterations <= 0 or all(len(values) == 1 for values in repeat_values):
            low = high = point
        else:
            draws = np.empty((bootstrap_iterations, len(repeat_values)), dtype=float)
            for index, values in enumerate(repeat_values):
                sampled = rng.choice(values, size=(bootstrap_iterations, len(values)), replace=True)
                draws[:, index] = sampled.mean(axis=1)
            x_centered = x - x.mean()
            denominator = float(np.sum(x_centered**2))
            slopes = (draws @ x_centered) / denominator
            sensitivity_draws = -slopes
            low, high = np.percentile(sensitivity_draws, [2.5, 97.5]).astype(float)
        rows.append(
            {
                "task_id": keys[0],
                "display_method": keys[1],
                "failure_axis": keys[2],
                "fraction_count": int(len(fractions)),
                "sensitivity": float(point),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_absolute_mae_figure(
    task_id: str,
    curve_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> plt.Figure:
    """Build a task-specific Figure 4 whose quantitative axes use raw OOD MAE."""
    task_curves = curve_summary[curve_summary["task_id"].eq(task_id)]
    task_sensitivity = sensitivity[sensitivity["task_id"].eq(task_id)]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_panel_a(ax_a)
    _draw_raw_mae_curve_panel(
        ax_b,
        task_curves[task_curves["failure_axis"].eq(FAILURE_AXIS_LOCO)],
        title="Farthest LOCO",
    )
    _draw_raw_mae_curve_panel(
        ax_c,
        task_curves[task_curves["failure_axis"].eq(FAILURE_AXIS_TOP20)],
        title="Top-20% property extreme",
    )
    _draw_raw_mae_sensitivity_panel(ax_d, task_sensitivity)
    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    fig.suptitle(f"Absolute OOD MAE | {task_id}", fontsize=13, fontweight="bold")
    return fig


def plot_absolute_mae_task_figures(
    data: pd.DataFrame,
    figures_dir: Path,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write task-specific raw-MAE figures and one multipage PDF."""
    curve_summary = task_raw_ood_mae_curve_summary(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    sensitivity = task_raw_ood_mae_sensitivity_summary(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    if curve_summary.empty:
        return curve_summary, sensitivity
    task_dir = figures_dir / "figure4_absolute_mae_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    multipage_path = figures_dir / "figure4_absolute_ood_mae_by_task.pdf"
    with PdfPages(multipage_path) as pdf:
        for task_id in sorted(curve_summary["task_id"].dropna().astype(str).unique()):
            fig = build_absolute_mae_figure(task_id, curve_summary, sensitivity)
            filename = _safe_filename(task_id)
            for extension in ["png", "pdf", "svg"]:
                fig.savefig(task_dir / f"{filename}.{extension}", dpi=600, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    _plot_absolute_mae_overview(curve_summary, figures_dir)
    return curve_summary, sensitivity


def plot_figure4(
    family_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figures_dir: Path,
    *,
    selected_summary: pd.DataFrame | None = None,
    selected_sensitivity: pd.DataFrame | None = None,
) -> None:
    if family_summary.empty and (selected_summary is None or selected_summary.empty):
        return
    fig = build_figure4(
        family_summary,
        sensitivity,
        selected_summary=selected_summary,
        selected_sensitivity=selected_sensitivity,
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            figures_dir / f"figure4_degradation_vs_training_size.{ext}",
            dpi=600,
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
    selected = "display_method" in data.columns
    key_column = "display_method" if selected else "model_family"
    order = DISPLAY_METHOD_ORDER if selected else FAMILY_ORDER
    palette = DISPLAY_METHOD_COLORS if selected else FAMILY_COLORS
    for task_id, task_data in data.groupby("task_id", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
        for ax, axis, title in [
            (axes[0], FAILURE_AXIS_LOCO, "Farthest LOCO"),
            (axes[1], FAILURE_AXIS_TOP20, "Top 20% property extreme"),
        ]:
            panel = task_data[task_data["failure_axis"].eq(axis)]
            summary = (
                panel.groupby([key_column, "training_fraction"], dropna=False)[
                    "degradation_ratio"
                ]
                .mean()
                .reset_index()
            )
            for method in order:
                group = summary[summary[key_column].eq(method)].sort_values("training_fraction")
                if group.empty:
                    continue
                ax.plot(
                    group["training_fraction"] * 100,
                    group["degradation_ratio"],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=palette[method],
                    label=method,
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
    selected = "display_method" in data.columns
    key_column = "display_method" if selected else "model_family"
    order = DISPLAY_METHOD_ORDER if selected else FAMILY_ORDER
    palette = DISPLAY_METHOD_COLORS if selected else FAMILY_COLORS
    for task_id, task_data in data.groupby("task_id", dropna=False):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        for ax, axis, title in [
            (axes[0], FAILURE_AXIS_LOCO, "Farthest LOCO"),
            (axes[1], FAILURE_AXIS_TOP20, "Top 20% property extreme"),
        ]:
            panel = task_data[task_data["failure_axis"].eq(axis)]
            summary = (
                panel.groupby([key_column, "training_fraction"], dropna=False)[
                    ["random_cv_mae", "ood_mae"]
                ]
                .mean()
                .reset_index()
            )
            for method in order:
                group = summary[summary[key_column].eq(method)].sort_values(
                    "training_fraction"
                )
                if group.empty:
                    continue
                x = group["training_fraction"] * 100
                color = palette[method]
                ax.plot(
                    x,
                    group["ood_mae"],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=color,
                    label=f"{method} OOD",
                )
                ax.plot(
                    x,
                    group["random_cv_mae"],
                    linewidth=1.2,
                    linestyle="--",
                    color=color,
                    alpha=0.85,
                    label=f"{method} Random-CV",
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
    selection = select_representative_models(data)
    selected_data = apply_representative_selection(data, selection)
    if bool(figure_cfg.get("require_complete_training_size", False)):
        validate_display_methods(selected_data)
    selected_summary = selected_method_curve_summary(
        selected_data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    selected_sensitivity = selected_method_sensitivity_summary(
        selected_data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    excluded_tasks = tuple(
        str(value)
        for value in figure_cfg.get(
            "exclude_tasks_from_display",
            DEFAULT_EXCLUDED_TASKS,
        )
    )
    display_data = prepare_display_data(data, excluded_tasks=excluded_tasks)
    if bool(figure_cfg.get("require_complete_training_size", False)):
        expected_display_tasks = int(figure_cfg.get("expected_display_task_count", 8))
        observed_display_tasks = int(display_data["task_id"].nunique())
        if observed_display_tasks != expected_display_tasks:
            raise ValueError(
                f"Figure 4 requires {expected_display_tasks} display tasks, "
                f"found {observed_display_tasks}"
            )
    raw_mae_summary = task_raw_ood_mae_curve_summary(
        display_data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    raw_mae_sensitivity = task_raw_ood_mae_sensitivity_summary(
        display_data,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    empty_csv(csv_dir / "training_size_sample_level_predictions.csv", [])
    data.to_csv(csv_dir / "degradation_ratio_by_repeat.csv", index=False)
    model_summary.to_csv(csv_dir / "degradation_ratio_summary.csv", index=False)
    model_summary.to_csv(csv_dir / "task_model_curve_summary.csv", index=False)
    family_summary.to_csv(csv_dir / "family_curve_summary.csv", index=False)
    sensitivity.to_csv(csv_dir / "sensitivity_summary.csv", index=False)
    selection.to_csv(csv_dir / "representative_model_selection.csv", index=False)
    selected_summary.to_csv(csv_dir / "selected_method_curve_summary.csv", index=False)
    selected_sensitivity.to_csv(csv_dir / "selected_method_sensitivity_summary.csv", index=False)
    selected_data.to_csv(csv_dir / "selected_degradation_ratio_by_repeat.csv", index=False)
    raw_mae_summary.to_csv(csv_dir / "selected_method_raw_ood_mae_by_task.csv", index=False)
    raw_mae_sensitivity.to_csv(
        csv_dir / "selected_method_raw_ood_mae_sensitivity_by_task.csv",
        index=False,
    )
    panel_a_value = figure_cfg.get(
        "panel_a_drawio",
        figures_dir / "sources" / "figure4_panel_a_source.drawio",
    )
    panel_a_drawio = Path(str(panel_a_value))
    if not panel_a_drawio.is_absolute():
        panel_a_drawio = (Path.cwd() / panel_a_drawio).resolve()
    render_result = render_training_size_task_outputs(
        display_data,
        output_dir=figures_dir,
        panel_a_drawio=panel_a_drawio,
        drawio_executable=figure_cfg.get("drawio_executable"),
        clean=True,
        bootstrap_iterations=iterations,
        random_seed=random_seed,
    )
    explainer_cfg = dict(figure_cfg.get("explainer") or {})
    if bool(explainer_cfg.get("enabled", False)):
        build_figure4_explainer(
            stage_dir,
            output_dir=explainer_cfg.get("output_dir"),
            excluded_tasks=excluded_tasks,
        )
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
            "representative_selection_rows": int(len(selection)),
            "selected_summary_rows": int(len(selected_summary)),
            "selected_sensitivity_rows": int(len(selected_sensitivity)),
            "raw_mae_summary_rows": int(len(raw_mae_summary)),
            "raw_mae_sensitivity_rows": int(len(raw_mae_sensitivity)),
            "display_task_count": int(display_data["task_id"].nunique()),
            "excluded_display_tasks": list(excluded_tasks),
            "figure_output_contract": render_result,
            "selected_display_methods": DISPLAY_METHOD_ORDER,
            "selection_rule": "lowest mean Random-CV MAE per task and training fraction; tie-break by SD then model name",
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


def _draw_curve_panel(ax: plt.Axes, panel: pd.DataFrame, *, title: str, selected: bool = False) -> None:
    key_column = "display_method" if selected else "model_family"
    order = DISPLAY_METHOD_ORDER if selected else FAMILY_ORDER
    colors = DISPLAY_METHOD_COLORS if selected else FAMILY_COLORS
    for method in order:
        group = panel[panel[key_column].eq(method)].sort_values("training_fraction")
        if group.empty:
            continue
        x = group["training_fraction"].to_numpy(dtype=float) * 100
        y = group["mean_degradation_ratio"].to_numpy(dtype=float)
        low = group["ci95_low"].to_numpy(dtype=float)
        high = group["ci95_high"].to_numpy(dtype=float)
        color = colors[method]
        marker = DISPLAY_METHOD_MARKERS[method] if selected else "o"
        ax.plot(x, y, marker=marker, linewidth=1.8, markersize=4.5, color=color, label=method)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Training-set size (%)")
    ax.set_ylabel("MAE(OOD) / MAE(Random-CV)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.2, loc="best")


def _draw_raw_mae_curve_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    title: str,
    show_legend: bool = True,
) -> None:
    for method in DISPLAY_METHOD_ORDER:
        group = panel[panel["display_method"].eq(method)].sort_values("training_fraction")
        if group.empty:
            continue
        x = group["training_fraction"].to_numpy(dtype=float) * 100
        y = group["mean_ood_mae"].to_numpy(dtype=float)
        low = group["ci95_low"].to_numpy(dtype=float)
        high = group["ci95_high"].to_numpy(dtype=float)
        color = DISPLAY_METHOD_COLORS[method]
        ax.plot(
            x,
            y,
            marker=DISPLAY_METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=method,
        )
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Training-set size (%)")
    ax.set_ylabel("OOD MAE (native task units)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.22)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.2, loc="best")


def _draw_raw_mae_sensitivity_panel(ax: plt.Axes, sensitivity: pd.DataFrame) -> None:
    positions = np.arange(len(DISPLAY_METHOD_ORDER), dtype=float)
    axis_styles = [
        (FAILURE_AXIS_LOCO, -0.11, "o", "Farthest LOCO"),
        (FAILURE_AXIS_TOP20, 0.11, "s", "Top-20% property"),
    ]
    for axis, offset, marker, label in axis_styles:
        panel = sensitivity[sensitivity["failure_axis"].eq(axis)].set_index("display_method")
        for index, method in enumerate(DISPLAY_METHOD_ORDER):
            if method not in panel.index:
                continue
            row = panel.loc[method]
            y = float(row["sensitivity"])
            yerr = np.asarray(
                [
                    max(0.0, y - float(row["ci95_low"])),
                    max(0.0, float(row["ci95_high"]) - y),
                ]
            ).reshape(2, 1)
            ax.errorbar(
                index + offset,
                y,
                yerr=yerr,
                marker=marker,
                color=DISPLAY_METHOD_COLORS[method],
                markerfacecolor=DISPLAY_METHOD_COLORS[method],
                markersize=5,
                capsize=3,
                linewidth=1.2,
            )
        ax.plot([], [], marker=marker, color="#555555", linestyle="none", label=label)
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(positions, DISPLAY_METHOD_ORDER, rotation=25, ha="right")
    ax.set_ylabel("MAE sensitivity (-slope)")
    ax.set_title("Absolute-MAE sensitivity")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def _plot_absolute_mae_overview(curve_summary: pd.DataFrame, figures_dir: Path) -> None:
    """Create a task-separated overview without averaging across target units."""
    tasks = sorted(curve_summary["task_id"].dropna().astype(str).unique())
    if not tasks:
        return
    fig, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(13.4, max(4.0, 2.6 * len(tasks))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, task_id in enumerate(tasks):
        task = curve_summary[curve_summary["task_id"].eq(task_id)]
        for column_index, (axis, title) in enumerate(
            [
                (FAILURE_AXIS_LOCO, "Farthest LOCO"),
                (FAILURE_AXIS_TOP20, "Top-20% property extreme"),
            ]
        ):
            ax = axes[row_index, column_index]
            _draw_raw_mae_curve_panel(
                ax,
                task[task["failure_axis"].eq(axis)],
                title=f"{task_id} | {title}",
                show_legend=row_index == 0 and column_index == 1,
            )
            if row_index != 0 or column_index != 1:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            figures_dir / f"figure4_absolute_ood_mae_overview.{extension}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)


def _with_display_method(data: pd.DataFrame) -> pd.DataFrame:
    working = data.copy()
    if "display_method" in working.columns:
        return working
    working["display_method"] = working.apply(
        lambda row: {
            "Traditional": "Traditional-best",
            "BERT": "BERT-best",
            "LLMProp": "LLM-Prop",
        }.get(
            str(row["model_family"]),
            "TabPFN Text" if "Text" in str(row["model"]) else "TabPFN Numeric",
        ),
        axis=1,
    )
    return working


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered * (y - y.mean())) / denominator)


def _draw_sensitivity_panel(ax: plt.Axes, sensitivity: pd.DataFrame, *, selected: bool = False) -> None:
    order = DISPLAY_METHOD_ORDER if selected else FAMILY_ORDER
    palette = DISPLAY_METHOD_COLORS if selected else FAMILY_COLORS
    positions = np.arange(len(order), dtype=float)
    axis_styles = [
        (FAILURE_AXIS_LOCO, -0.11, "o", "Farthest LOCO"),
        (FAILURE_AXIS_TOP20, 0.11, "s", "Top-20% property"),
    ]
    for axis, offset, marker, label in axis_styles:
        key_column = "display_method" if selected else "model_family"
        panel = sensitivity[sensitivity["failure_axis"].eq(axis)].set_index(key_column)
        xs: list[float] = []
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        point_colors: list[str] = []
        for index, method in enumerate(order):
            if method not in panel.index:
                continue
            row = panel.loc[method]
            xs.append(float(index + offset))
            ys.append(float(row["sensitivity"]))
            lows.append(float(row["ci95_low"]))
            highs.append(float(row["ci95_high"]))
            point_colors.append(palette[method])
        if xs:
            y = np.asarray(ys)
            yerr = np.vstack(
                [
                    np.maximum(0.0, y - np.asarray(lows)),
                    np.maximum(0.0, np.asarray(highs) - y),
                ]
            )
            for x_value, y_value, error, color in zip(xs, ys, yerr.T, point_colors):
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
    ax.set_xticks(positions, order, rotation=25 if selected else 0, ha="right" if selected else "center")
    ax.set_ylabel("Sensitivity (-slope)")
    ax.set_title("Data-scarcity sensitivity")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
