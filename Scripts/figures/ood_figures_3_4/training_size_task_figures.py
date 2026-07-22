from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training_size_ood.core import FAILURE_AXIS_LOCO, FAILURE_AXIS_TOP20
from src.training_size_ood.selection import (
    DISPLAY_METHODS,
    apply_representative_selection,
    select_representative_models,
)

from .figure4_panel_a import write_panel_a_drawio


DISPLAY_METHOD_ORDER = list(DISPLAY_METHODS)
DISPLAY_METHOD_COLORS = {
    "Traditional-best": "#356F9F",
    "BERT-best": "#D0525F",
    "TabPFN Numeric": "#238B7E",
    "TabPFN Text": "#71B7A8",
    "LLM-Prop": "#8B6F8F",
    "GPT-5.4": "#D1842C",
}
DISPLAY_METHOD_MARKERS = {
    "Traditional-best": "o",
    "BERT-best": "s",
    "TabPFN Numeric": "^",
    "TabPFN Text": "D",
    "LLM-Prop": "P",
    "GPT-5.4": "X",
}
DEFAULT_EXCLUDED_TASKS = (
    "HEA__hea__YS_MPa",
    "Steel__steel__YS_MPa",
)
REQUIRED_COLUMNS = {
    "task_id",
    "model_family",
    "model",
    "training_fraction",
    "repeat_id",
    "failure_axis",
    "random_cv_mae",
    "ood_mae",
}
MANAGED_DIRECTORIES = (
    "figure4_absolute_mae_tasks",
    "figure4_ratio_tasks",
    "random_cv_mae",
    "task_ood_curve_pairs",
)
LEGACY_DIRECTORIES = ("raw_mae_tasks", "tasks")
LEGACY_FILES = (
    "figure4_absolute_ood_mae_by_task.pdf",
    "figure4_absolute_ood_mae_overview.pdf",
    "figure4_absolute_ood_mae_overview.png",
    "figure4_absolute_ood_mae_overview.svg",
    "figure4_degradation_vs_training_size.pdf",
    "figure4_degradation_vs_training_size.png",
    "figure4_degradation_vs_training_size.svg",
    "figure4_model_level_curves.pdf",
    "figure4_model_level_curves.png",
    "figure4_model_level_curves.svg",
    "figure4_panel_a_source.drawio",
    "figure4_panel_a_source.pdf",
    "figure4_panel_a_source.png",
    "figure4_panel_a_source.svg",
)


def prepare_display_data(
    data: pd.DataFrame,
    *,
    excluded_tasks: Iterable[str] = DEFAULT_EXCLUDED_TASKS,
    excluded_display_methods: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate raw run rows, select representatives, and filter display tasks."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Training-size figure input is missing columns: {sorted(missing)}")
    working = data.copy()
    for column in ["training_fraction", "repeat_id", "random_cv_mae", "ood_mae"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(
        subset=["training_fraction", "repeat_id", "random_cv_mae", "ood_mae"]
    )
    if (working["random_cv_mae"] <= 0).any():
        raise ValueError("Random-CV MAE must be positive")
    working["degradation_ratio"] = working["ood_mae"] / working["random_cv_mae"]
    selection = select_representative_models(working)
    selected = apply_representative_selection(working, selection)
    selected = selected[~selected["task_id"].isin(set(excluded_tasks))].copy()
    excluded_methods = set(excluded_display_methods)
    unexpected_exclusions = excluded_methods.difference(DISPLAY_METHOD_ORDER)
    if unexpected_exclusions:
        raise ValueError(
            f"Unknown display methods requested for exclusion: {sorted(unexpected_exclusions)}"
        )
    selected = selected[~selected["display_method"].isin(excluded_methods)].copy()
    expected = set(DISPLAY_METHOD_ORDER).difference(excluded_methods)
    for task_id, task_data in selected.groupby("task_id", dropna=False):
        observed = set(task_data["display_method"].dropna().astype(str))
        if observed != expected:
            raise ValueError(
                f"Task {task_id} does not contain the expected display methods: "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )
    return selected.reset_index(drop=True)


def summarize_task_metric(
    data: pd.DataFrame,
    *,
    metric_column: str,
    output_column: str,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    columns = [
        "task_id",
        "display_method",
        "failure_axis",
        "training_fraction",
        "repeat_count",
        output_column,
        "ci95_low",
        "ci95_high",
    ]
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    keys = ["task_id", "display_method", "failure_axis", "training_fraction"]
    for values, group in data.groupby(keys, dropna=False, sort=True):
        repeat_values = (
            group.groupby("repeat_id", dropna=False)[metric_column]
            .mean()
            .to_numpy(dtype=float)
        )
        repeat_values = repeat_values[np.isfinite(repeat_values)]
        if not len(repeat_values):
            continue
        point, low, high = _bootstrap_mean_interval(
            repeat_values,
            rng=rng,
            iterations=bootstrap_iterations,
        )
        rows.append(
            {
                "task_id": values[0],
                "display_method": values[1],
                "failure_axis": values[2],
                "training_fraction": float(values[3]),
                "repeat_count": int(len(repeat_values)),
                output_column: point,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_task_sensitivity(
    data: pd.DataFrame,
    *,
    metric_column: str,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    columns = [
        "task_id",
        "display_method",
        "failure_axis",
        "fraction_count",
        "sensitivity",
        "ci95_low",
        "ci95_high",
    ]
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for values, group in data.groupby(
        ["task_id", "display_method", "failure_axis"],
        dropna=False,
        sort=True,
    ):
        fractions: list[float] = []
        repeats: list[np.ndarray] = []
        for fraction, fraction_group in group.groupby(
            "training_fraction", dropna=False, sort=True
        ):
            repeat_values = (
                fraction_group.groupby("repeat_id", dropna=False)[metric_column]
                .mean()
                .to_numpy(dtype=float)
            )
            repeat_values = repeat_values[np.isfinite(repeat_values)]
            if float(fraction) > 0 and len(repeat_values):
                fractions.append(float(fraction))
                repeats.append(repeat_values)
        if len(fractions) < 2:
            continue
        x = np.log(np.asarray(fractions, dtype=float))
        point = -_linear_slope(x, np.asarray([item.mean() for item in repeats]))
        if bootstrap_iterations <= 0 or all(len(item) == 1 for item in repeats):
            low = high = point
        else:
            draws = np.empty((bootstrap_iterations, len(repeats)), dtype=float)
            for index, item in enumerate(repeats):
                sampled = rng.choice(
                    item,
                    size=(bootstrap_iterations, len(item)),
                    replace=True,
                )
                draws[:, index] = sampled.mean(axis=1)
            x_centered = x - x.mean()
            slopes = (draws @ x_centered) / float(np.sum(x_centered**2))
            low, high = np.percentile(-slopes, [2.5, 97.5]).astype(float)
        rows.append(
            {
                "task_id": values[0],
                "display_method": values[1],
                "failure_axis": values[2],
                "fraction_count": len(fractions),
                "sensitivity": float(point),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_random_cv_mae(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    run_keys = [
        "task_id",
        "display_method",
        "model",
        "training_fraction",
        "repeat_id",
    ]
    consistency = data.groupby(run_keys, dropna=False)["random_cv_mae"].nunique(
        dropna=False
    )
    if consistency.gt(1).any():
        raise ValueError("Random-CV MAE differs between failure axes for one run")
    deduplicated = data.drop_duplicates(run_keys)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    keys = ["task_id", "display_method", "training_fraction"]
    for values, group in deduplicated.groupby(keys, dropna=False, sort=True):
        repeat_values = group["random_cv_mae"].to_numpy(dtype=float)
        repeat_values = repeat_values[np.isfinite(repeat_values)]
        point, low, high = _bootstrap_mean_interval(
            repeat_values,
            rng=rng,
            iterations=bootstrap_iterations,
        )
        rows.append(
            {
                "task_id": values[0],
                "display_method": values[1],
                "training_fraction": float(values[2]),
                "repeat_count": int(len(repeat_values)),
                "mean_random_cv_mae": point,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def _validate_display_method_coverage(
    data: pd.DataFrame,
    display_methods: Sequence[str],
    *,
    allow_missing_methods: bool = False,
) -> None:
    methods = list(display_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("display_methods must contain unique method names")
    unsupported = set(methods).difference(DISPLAY_METHOD_COLORS)
    if unsupported:
        raise ValueError(f"No plotting style is defined for methods: {sorted(unsupported)}")
    for task_id, task_data in data.groupby("task_id", dropna=False):
        observed = set(task_data["display_method"].dropna().astype(str))
        expected = set(methods)
        if (not allow_missing_methods and observed != expected) or (
            allow_missing_methods and not observed.issubset(expected)
        ):
            raise ValueError(
                f"Task {task_id} does not contain the expected display methods: "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )


def render_training_size_task_outputs(
    data: pd.DataFrame,
    *,
    output_dir: str | Path,
    panel_a_drawio: str | Path,
    drawio_executable: str | Path | None = None,
    panel_a_image: str | Path | None = None,
    clean: bool = False,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
    dpi: int = 600,
    display_methods: Sequence[str] = tuple(DISPLAY_METHOD_ORDER),
    allow_missing_methods: bool = False,
) -> dict[str, object]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    method_order = list(display_methods)
    _validate_display_method_coverage(
        data,
        method_order,
        allow_missing_methods=allow_missing_methods,
    )
    drawio_path = ensure_panel_a_source(output_root, Path(panel_a_drawio))
    if clean:
        clean_managed_outputs(output_root)

    mae_summary = summarize_task_metric(
        data,
        metric_column="ood_mae",
        output_column="mean_ood_mae",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    ratio_summary = summarize_task_metric(
        data,
        metric_column="degradation_ratio",
        output_column="mean_degradation_ratio",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    mae_sensitivity = summarize_task_sensitivity(
        data,
        metric_column="ood_mae",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    ratio_sensitivity = summarize_task_sensitivity(
        data,
        metric_column="degradation_ratio",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    random_summary = summarize_random_cv_mae(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )

    if panel_a_image is not None:
        panel_path = Path(panel_a_image).resolve()
        _render_all_outputs(
            data=data,
            output_root=output_root,
            panel_image=panel_path,
            mae_summary=mae_summary,
            ratio_summary=ratio_summary,
            mae_sensitivity=mae_sensitivity,
            ratio_sensitivity=ratio_sensitivity,
            random_summary=random_summary,
            dpi=dpi,
            display_methods=method_order,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="figure4-panel-a-") as temp_dir:
            panel_path = Path(temp_dir) / "panel_a.png"
            render_drawio_to_png(
                drawio_path,
                panel_path,
                drawio_executable=drawio_executable,
            )
            _render_all_outputs(
                data=data,
                output_root=output_root,
                panel_image=panel_path,
                mae_summary=mae_summary,
                ratio_summary=ratio_summary,
                mae_sensitivity=mae_sensitivity,
                ratio_sensitivity=ratio_sensitivity,
                random_summary=random_summary,
                dpi=dpi,
                display_methods=method_order,
            )
    return {
        "task_count": int(data["task_id"].nunique()),
        "display_methods": method_order,
        "absolute_mae_directory": str(output_root / "figure4_absolute_mae_tasks"),
        "ratio_directory": str(output_root / "figure4_ratio_tasks"),
        "curve_pair_degradation_directory": str(
            output_root / "task_ood_curve_pairs" / "degradation"
        ),
        "curve_pair_absolute_mae_directory": str(
            output_root / "task_ood_curve_pairs" / "absolute_mae"
        ),
        "random_cv_directory": str(output_root / "random_cv_mae"),
        "panel_a_drawio": str(drawio_path),
    }


def ensure_panel_a_source(output_root: Path, requested_path: Path) -> Path:
    path = requested_path.resolve()
    if path.exists():
        return path
    expected_parent = (output_root / "sources").resolve()
    if path.parent != expected_parent:
        raise ValueError(f"Panel A source must be inside {expected_parent}: {path}")
    legacy = output_root / "figure4_panel_a_source.drawio"
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy.exists():
        shutil.move(str(legacy), str(path))
    else:
        write_panel_a_drawio(path)
    return path


def resolve_drawio_executable(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for command in ["draw.io.exe", "drawio.exe", "draw.io", "drawio"]:
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))
    local_app_data = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "draw.io" / "draw.io.exe")
    if program_files:
        candidates.append(Path(program_files) / "draw.io" / "draw.io.exe")
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "draw.io" / "draw.io.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "draw.io CLI was not found. Install it with: "
        "winget install --id JGraph.Draw -e --accept-source-agreements "
        "--accept-package-agreements, or pass --drawio-executable."
    )


def render_drawio_to_png(
    drawio_path: Path,
    output_path: Path,
    *,
    drawio_executable: str | Path | None = None,
) -> Path:
    executable = resolve_drawio_executable(drawio_executable)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(executable),
            "-x",
            "-f",
            "png",
            "-t",
            "-s",
            "2",
            "-o",
            str(output_path),
            str(drawio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not output_path.exists():
        detail = (completed.stderr or completed.stdout or "unknown draw.io error").strip()
        raise RuntimeError(f"draw.io export failed: {detail}")
    return output_path


def clean_managed_outputs(output_root: Path) -> None:
    root = output_root.resolve()
    for name in [*MANAGED_DIRECTORIES, *LEGACY_DIRECTORIES]:
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError(f"Refusing to clean path outside figure root: {path}")
        if path.exists():
            shutil.rmtree(path)
    for name in LEGACY_FILES:
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError(f"Refusing to clean path outside figure root: {path}")
        if path.exists():
            path.unlink()


def _render_all_outputs(
    *,
    data: pd.DataFrame,
    output_root: Path,
    panel_image: Path,
    mae_summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    mae_sensitivity: pd.DataFrame,
    ratio_sensitivity: pd.DataFrame,
    random_summary: pd.DataFrame,
    dpi: int,
    display_methods: Sequence[str],
) -> None:
    tasks = sorted(data["task_id"].dropna().astype(str).unique())
    mae_dir = output_root / "figure4_absolute_mae_tasks"
    ratio_dir = output_root / "figure4_ratio_tasks"
    curve_pair_mae_dir = output_root / "task_ood_curve_pairs" / "absolute_mae"
    curve_pair_ratio_dir = output_root / "task_ood_curve_pairs" / "degradation"
    random_dir = output_root / "random_cv_mae"
    for directory in [
        mae_dir,
        ratio_dir,
        curve_pair_mae_dir,
        curve_pair_ratio_dir,
        random_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    image = plt.imread(panel_image)
    for task_id in tasks:
        mae_figure = _build_task_figure(
            task_id=task_id,
            panel_image=image,
            summary=mae_summary,
            sensitivity=mae_sensitivity,
            value_column="mean_ood_mae",
            ylabel="OOD MAE (native task units)",
            sensitivity_ylabel="MAE sensitivity (-slope)",
            sensitivity_title="Absolute-MAE sensitivity",
            suptitle=f"Absolute OOD MAE | {task_id}",
            reference_line=None,
            display_methods=display_methods,
        )
        ratio_figure = _build_task_figure(
            task_id=task_id,
            panel_image=image,
            summary=ratio_summary,
            sensitivity=ratio_sensitivity,
            value_column="mean_degradation_ratio",
            ylabel="MAE(OOD) / MAE(Random-CV)",
            sensitivity_ylabel="Ratio sensitivity (-slope)",
            sensitivity_title="Degradation-ratio sensitivity",
            suptitle=f"OOD degradation ratio | {task_id}",
            reference_line=1.0,
            display_methods=display_methods,
        )
        mae_curve_pair = _build_task_curve_pair(
            task_id=task_id,
            summary=mae_summary,
            value_column="mean_ood_mae",
            ylabel="OOD MAE (native task units)",
            suptitle=f"Absolute OOD MAE | {task_id}",
            reference_line=None,
            display_methods=display_methods,
        )
        ratio_curve_pair = _build_task_curve_pair(
            task_id=task_id,
            summary=ratio_summary,
            value_column="mean_degradation_ratio",
            ylabel="MAE(OOD) / MAE(Random-CV)",
            suptitle=f"OOD degradation ratio | {task_id}",
            reference_line=1.0,
            display_methods=display_methods,
        )
        filename = _safe_filename(task_id)
        _save_figure(mae_figure, mae_dir / filename, dpi=dpi)
        _save_figure(ratio_figure, ratio_dir / filename, dpi=dpi)
        _save_png(mae_curve_pair, curve_pair_mae_dir / filename, dpi=dpi)
        _save_png(ratio_curve_pair, curve_pair_ratio_dir / filename, dpi=dpi)
        plt.close(mae_figure)
        plt.close(ratio_figure)
        plt.close(mae_curve_pair)
        plt.close(ratio_curve_pair)
    random_figure = _build_random_cv_figure(
        random_summary, tasks, display_methods=display_methods
    )
    _save_figure(random_figure, random_dir / "random_cv_mae_all_tasks", dpi=dpi)
    random_summary.to_csv(random_dir / "random_cv_mae_all_tasks.csv", index=False)
    plt.close(random_figure)


def _build_task_figure(
    *,
    task_id: str,
    panel_image: np.ndarray,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    value_column: str,
    ylabel: str,
    sensitivity_ylabel: str,
    sensitivity_title: str,
    suptitle: str,
    reference_line: float | None,
    display_methods: Sequence[str],
) -> plt.Figure:
    task_summary = summary[summary["task_id"].eq(task_id)]
    task_sensitivity = sensitivity[sensitivity["task_id"].eq(task_id)]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    ax_a.imshow(panel_image)
    ax_a.set_axis_off()
    _draw_curve_panel(
        ax_b,
        task_summary[task_summary["failure_axis"].eq(FAILURE_AXIS_LOCO)],
        title="Farthest LOCO",
        value_column=value_column,
        ylabel=ylabel,
        reference_line=reference_line,
        display_methods=display_methods,
    )
    _draw_curve_panel(
        ax_c,
        task_summary[task_summary["failure_axis"].eq(FAILURE_AXIS_TOP20)],
        title="Top-20% property extreme",
        value_column=value_column,
        ylabel=ylabel,
        reference_line=reference_line,
        display_methods=display_methods,
    )
    _draw_sensitivity_panel(
        ax_d,
        task_sensitivity,
        title=sensitivity_title,
        ylabel=sensitivity_ylabel,
        display_methods=display_methods,
    )
    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            -0.10,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig


def _build_task_curve_pair(
    *,
    task_id: str,
    summary: pd.DataFrame,
    value_column: str,
    ylabel: str,
    suptitle: str,
    reference_line: float | None,
    display_methods: Sequence[str],
) -> plt.Figure:
    """Render only the two OOD curve panels for one task."""
    task_summary = summary[summary["task_id"].eq(task_id)]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)
    _draw_curve_panel(
        axes[0],
        task_summary[task_summary["failure_axis"].eq(FAILURE_AXIS_LOCO)],
        title="Farthest LOCO",
        value_column=value_column,
        ylabel=ylabel,
        reference_line=reference_line,
        display_methods=display_methods,
        show_legend=False,
    )
    _draw_curve_panel(
        axes[1],
        task_summary[task_summary["failure_axis"].eq(FAILURE_AXIS_TOP20)],
        title="Top-20% property extreme",
        value_column=value_column,
        ylabel=ylabel,
        reference_line=reference_line,
        display_methods=display_methods,
        show_legend=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(display_methods), 6),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig


def _draw_curve_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    title: str,
    value_column: str,
    ylabel: str,
    reference_line: float | None,
    display_methods: Sequence[str],
    show_legend: bool = True,
) -> None:
    for method in display_methods:
        group = panel[panel["display_method"].eq(method)].sort_values(
            "training_fraction"
        )
        if group.empty:
            continue
        x = group["training_fraction"].to_numpy(dtype=float) * 100
        y = group[value_column].to_numpy(dtype=float)
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
    if reference_line is not None:
        ax.axhline(reference_line, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Training-set size (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.22)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.2, loc="best")


def _draw_sensitivity_panel(
    ax: plt.Axes,
    sensitivity: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    display_methods: Sequence[str],
) -> None:
    ordered_methods = sensitivity_method_order(sensitivity, display_methods)
    positions = np.arange(len(ordered_methods), dtype=float)
    axis_styles = [
        (FAILURE_AXIS_LOCO, -0.11, "o", "Farthest LOCO"),
        (FAILURE_AXIS_TOP20, 0.11, "s", "Top-20% property"),
    ]
    for axis, offset, marker, label in axis_styles:
        panel = sensitivity[sensitivity["failure_axis"].eq(axis)].set_index(
            "display_method"
        )
        for index, method in enumerate(ordered_methods):
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
    ax.set_xticks(positions, ordered_methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)


def sensitivity_method_order(
    sensitivity: pd.DataFrame,
    display_methods: Sequence[str],
) -> list[str]:
    """Order methods by mean sensitivity across the available failure axes."""
    scores: dict[str, float] = {}
    for method in display_methods:
        values = pd.to_numeric(
            sensitivity.loc[sensitivity["display_method"].eq(method), "sensitivity"],
            errors="coerce",
        ).dropna()
        scores[method] = float(values.mean()) if not values.empty else float("inf")
    return sorted(
        display_methods,
        key=lambda method: (scores[method], list(display_methods).index(method)),
    )


def _build_random_cv_figure(
    summary: pd.DataFrame,
    tasks: list[str],
    *,
    display_methods: Sequence[str],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.8), constrained_layout=True)
    for ax, task_id in zip(axes.ravel(), tasks):
        panel = summary[summary["task_id"].eq(task_id)]
        for method in display_methods:
            group = panel[panel["display_method"].eq(method)].sort_values(
                "training_fraction"
            )
            if group.empty:
                continue
            x = group["training_fraction"].to_numpy(dtype=float) * 100
            y = group["mean_random_cv_mae"].to_numpy(dtype=float)
            low = group["ci95_low"].to_numpy(dtype=float)
            high = group["ci95_high"].to_numpy(dtype=float)
            color = DISPLAY_METHOD_COLORS[method]
            ax.plot(
                x,
                y,
                marker=DISPLAY_METHOD_MARKERS[method],
                linewidth=1.5,
                markersize=3.8,
                color=color,
                label=method,
            )
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.set_title(task_id, fontsize=9)
        ax.set_xlabel("Training-set size (%)")
        ax.set_ylabel("Random-CV MAE (native task units)")
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(display_methods), 6),
        frameon=False,
    )
    fig.suptitle("Random-CV MAE across training-set sizes", fontsize=14, fontweight="bold")
    return fig


def _save_figure(fig: plt.Figure, output_base: Path, *, dpi: int) -> None:
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
        )


def _save_png(fig: plt.Figure, output_base: Path, *, dpi: int) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    point = float(values.mean())
    if len(values) == 1 or iterations <= 0:
        return point, point, point
    sampled = rng.choice(values, size=(iterations, len(values)), replace=True)
    low, high = np.percentile(sampled.mean(axis=1), [2.5, 97.5]).astype(float)
    return point, float(low), float(high)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(centered * (y - y.mean())) / denominator)


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
