from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Scripts.figures.ood_figures_3_4.training_size_gpt54_adapter import (
    load_gpt54_training_size_rows,
)
from Scripts.figures.ood_figures_3_4.training_size_task_figures import (
    prepare_display_data,
)


METHOD_ORDER = (
    "Traditional-best",
    "BERT-best",
    "LLM-Prop",
    "TabPFN Numeric",
    "TabPFN Text",
    "GPT-5.4",
)
FAILURE_AXIS_ORDER = (
    "top_tail_distant_loco",
    "top20_property_extreme",
)
FAILURE_AXIS_LABELS = {
    "top_tail_distant_loco": "Compositional-novelty-extreme (Farthest-LOCO)",
    "top20_property_extreme": "Property-extremity-extreme (Top-20 held out)",
}
METHOD_COLORS = {
    "Traditional-best": "#4C78A8",
    "BERT-best": "#E17C45",
    "LLM-Prop": "#4C9A61",
    "TabPFN Numeric": "#C9494B",
    "TabPFN Text": "#756BB1",
    "GPT-5.4": "#111111",
}
METHOD_MARKERS = {
    "Traditional-best": "o",
    "BERT-best": "s",
    "LLM-Prop": "^",
    "TabPFN Numeric": "D",
    "TabPFN Text": "v",
    "GPT-5.4": "*",
}
METHOD_PLOT_LABELS = {
    "Traditional-best": "Traditional (best)",
    "BERT-best": "BERT (best)",
    "LLM-Prop": "LLM-Prop",
    "TabPFN Numeric": "TabPFN-2.5+ (Numeric)",
    "TabPFN Text": "TabPFN-2.5+ (Text)",
    "GPT-5.4": "GPT-5.4",
}
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
CACHE_SCHEMA_VERSION = 1
SELECTION_RULE_VERSION = "random_cv_mae_v1_shared_8_tasks"
CACHE_FILENAME = "figure4_overview_cache.json"
DEGRADATION_SUMMARY_FILENAME = "figure4_six_method_summary.csv"
DEGRADATION_TASK_MEANS_FILENAME = "figure4_six_method_task_repeat_means.csv"
DEGRADATION_BOOTSTRAP_FILENAME = "figure4_six_method_bootstrap_summary.csv"
ABSOLUTE_SUMMARY_FILENAME = "figure4_six_method_absolute_mae_summary.csv"
ABSOLUTE_TASK_MEANS_FILENAME = "figure4_six_method_absolute_mae_task_repeat_means.csv"
ABSOLUTE_BOOTSTRAP_FILENAME = "figure4_six_method_absolute_mae_bootstrap_summary.csv"
CACHE_DATA_FILENAMES = (
    DEGRADATION_SUMMARY_FILENAME,
    DEGRADATION_TASK_MEANS_FILENAME,
    DEGRADATION_BOOTSTRAP_FILENAME,
    ABSOLUTE_SUMMARY_FILENAME,
    ABSOLUTE_TASK_MEANS_FILENAME,
    ABSOLUTE_BOOTSTRAP_FILENAME,
)


def load_gpt54_rows(root: str | Path) -> pd.DataFrame:
    rows, _audit = load_gpt54_training_size_rows(root)
    return rows


def prepare_overview_data(
    raw: pd.DataFrame,
    gpt_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare ten-task non-GPT rows and the available eight-task GPT rows."""
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Figure 4 input is missing columns: {sorted(missing)}")

    # The overview uses the shared eight-task display set for every method.
    base = prepare_display_data(raw)
    base = base[base["display_method"].isin(METHOD_ORDER[:-1])].copy()

    if gpt_rows.empty:
        raise ValueError("GPT-5.4 rows are required for the six-method overview")
    gpt = gpt_rows.copy()
    if "display_method" not in gpt.columns:
        gpt["display_method"] = "GPT-5.4"
    gpt = gpt[gpt["display_method"].eq("GPT-5.4")].copy()
    combined = pd.concat([base, gpt], ignore_index=True, sort=False)

    for column in ["training_fraction", "repeat_id", "random_cv_mae", "ood_mae"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(
        subset=[
            "task_id",
            "display_method",
            "training_fraction",
            "repeat_id",
            "random_cv_mae",
            "ood_mae",
            "failure_axis",
        ]
    ).copy()
    if (combined["random_cv_mae"] <= 0).any():
        raise ValueError("Random-CV MAE must be positive")
    combined["degradation_ratio"] = (
        combined["ood_mae"] / combined["random_cv_mae"]
    )

    observed = set(combined["display_method"].astype(str))
    if observed != set(METHOD_ORDER):
        raise ValueError(
            f"Six-method overview coverage mismatch: missing={sorted(set(METHOD_ORDER) - observed)}, "
            f"unexpected={sorted(observed - set(METHOD_ORDER))}"
        )
    if set(combined["failure_axis"].astype(str)) != set(FAILURE_AXIS_ORDER):
        raise ValueError("Overview requires both Farthest-LOCO and Top-20 failure axes")
    task_sets = {
        method: set(group["task_id"].astype(str))
        for method, group in combined.groupby("display_method", sort=True)
    }
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        raise ValueError("All six overview methods must use the same eight display tasks")
    if len(next(iter(task_sets.values()))) != 8:
        raise ValueError("The Figure 4 overview requires exactly eight display tasks")
    return combined.reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_data_sha256(data: pd.DataFrame) -> str:
    columns = [
        "task_id",
        "display_method",
        "model_family",
        "model",
        "training_fraction",
        "repeat_id",
        "failure_axis",
        "random_cv_mae",
        "ood_mae",
        "degradation_ratio",
    ]
    working = data[columns].copy()
    working = working.sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = working.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_expectation(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selected_data_sha256": _selected_data_sha256(data),
        "selected_row_count": int(len(data)),
        "task_ids": sorted(data["task_id"].dropna().astype(str).unique()),
        "method_order": list(METHOD_ORDER),
        "failure_axis_order": list(FAILURE_AXIS_ORDER),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(random_seed),
    }


def _load_cache_manifest(cache_dir: Path | None) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / CACHE_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_is_valid(
    cache_dir: Path | None,
    manifest: dict[str, Any] | None,
    expectation: dict[str, Any],
) -> bool:
    if cache_dir is None or manifest is None:
        return False
    for key, expected in expectation.items():
        if manifest.get(key) != expected:
            return False
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict):
        return False
    for filename in CACHE_DATA_FILENAMES:
        path = cache_dir / filename
        if not path.exists() or file_hashes.get(filename) != _sha256_file(path):
            return False
    return True


def _load_cached_tables(cache_dir: Path) -> tuple[pd.DataFrame, ...]:
    return tuple(pd.read_csv(cache_dir / filename) for filename in CACHE_DATA_FILENAMES)


def summarize_task_repeat_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse any duplicate source rows to one repeat-level ratio per axis."""
    keys = [
        "display_method",
        "task_id",
        "failure_axis",
        "training_fraction",
        "repeat_id",
    ]
    grouped = (
        data.groupby(keys, dropna=False, sort=True)
        .agg(
            repeat_degradation_ratio=("degradation_ratio", "mean"),
            repeat_random_cv_mae=("random_cv_mae", "mean"),
            repeat_ood_mae=("ood_mae", "mean"),
            source_row_count=("degradation_ratio", "size"),
        )
        .reset_index()
    )
    recomputed = grouped["repeat_ood_mae"] / grouped["repeat_random_cv_mae"]
    if not np.allclose(
        grouped["repeat_degradation_ratio"].to_numpy(dtype=float),
        recomputed.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Overview ratio rows are inconsistent with OOD and Random-CV MAE")
    return grouped


def summarize_task_repeat_absolute_mae(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "display_method",
        "task_id",
        "failure_axis",
        "training_fraction",
        "repeat_id",
    ]
    return (
        data.groupby(keys, dropna=False, sort=True)
        .agg(
            repeat_ood_mae=("ood_mae", "mean"),
            source_row_count=("ood_mae", "size"),
        )
        .reset_index()
    )


def _hierarchical_bootstrap(
    task_repeat: pd.DataFrame,
    *,
    value_column: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, int, int]:
    task_values = {
        task: group[value_column].to_numpy(dtype=float)
        for task, group in task_repeat.groupby("task_id", sort=True)
    }
    if not task_values:
        raise ValueError("Cannot bootstrap an empty overview group")
    task_ids = list(task_values)
    task_means = np.asarray([values.mean() for values in task_values.values()])
    point = float(task_means.mean())
    repeat_counts = [len(values) for values in task_values.values()]
    if iterations <= 0:
        return point, point, point, len(task_ids), min(repeat_counts), max(repeat_counts)

    task_draws: list[np.ndarray] = []
    for task in task_ids:
        values = task_values[task]
        repeat_indices = rng.integers(
            0,
            len(values),
            size=(iterations, len(values)),
        )
        task_draws.append(values[repeat_indices].mean(axis=1))
    task_matrix = np.column_stack(task_draws)
    task_indices = rng.integers(
        0,
        len(task_ids),
        size=(iterations, len(task_ids)),
    )
    draws = np.take_along_axis(task_matrix, task_indices, axis=1).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5]).astype(float)
    return point, float(low), float(high), len(task_ids), min(repeat_counts), max(repeat_counts)


def _build_metric_overview_tables(
    task_repeat: pd.DataFrame,
    *,
    repeat_value_column: str,
    task_mean_column: str,
    summary_value_column: str,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)

    for keys, group in task_repeat.groupby(
        ["display_method", "task_id", "failure_axis", "training_fraction"],
        dropna=False,
        sort=True,
    ):
        task_rows.append(
            {
                "method": keys[0],
                "task_id": keys[1],
                "failure_axis": keys[2],
                "training_fraction": float(keys[3]),
                "repeat_count": int(group["repeat_id"].nunique()),
                task_mean_column: float(group[repeat_value_column].mean()),
            }
        )

    task_rows_frame = pd.DataFrame(task_rows)
    for keys, group in task_repeat.groupby(
        ["display_method", "failure_axis", "training_fraction"],
        dropna=False,
        sort=True,
    ):
        point, low, high, task_count, repeat_min, repeat_max = _hierarchical_bootstrap(
            group,
            value_column=repeat_value_column,
            iterations=bootstrap_iterations,
            rng=rng,
        )
        summary_row = {
            "method": keys[0],
            "failure_axis": keys[1],
            "training_fraction": float(keys[2]),
            "task_count": task_count,
            "repeat_count_min": repeat_min,
            "repeat_count_max": repeat_max,
            summary_value_column: point,
            "ci95_low": low,
            "ci95_high": high,
        }
        summary_rows.append(summary_row)
        bootstrap_rows.append(
            {
                **summary_row,
                "bootstrap_iterations": int(bootstrap_iterations),
                "bootstrap_seed": int(random_seed),
                "ci_method": "hierarchical task-and-repeat bootstrap",
            }
        )

    summary = pd.DataFrame(summary_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary["failure_axis_label"] = summary["failure_axis"].map(FAILURE_AXIS_LABELS)
    bootstrap["failure_axis_label"] = bootstrap["failure_axis"].map(FAILURE_AXIS_LABELS)
    task_rows_frame = task_rows_frame.sort_values(
        ["method", "failure_axis", "training_fraction", "task_id"]
    ).reset_index(drop=True)
    summary = summary.sort_values(
        ["method", "failure_axis", "training_fraction"]
    ).reset_index(drop=True)
    bootstrap = bootstrap.sort_values(
        ["method", "failure_axis", "training_fraction"]
    ).reset_index(drop=True)
    return summary, task_rows_frame, bootstrap


def build_overview_tables(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _build_metric_overview_tables(
        summarize_task_repeat_ratios(data),
        repeat_value_column="repeat_degradation_ratio",
        task_mean_column="task_mean_degradation_ratio",
        summary_value_column="mean_degradation_ratio",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )


def build_absolute_overview_tables(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _build_metric_overview_tables(
        summarize_task_repeat_absolute_mae(data),
        repeat_value_column="repeat_ood_mae",
        task_mean_column="task_mean_ood_mae",
        summary_value_column="mean_ood_mae",
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )


def _draw_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    axis_name: str,
    value_column: str,
    panel_letter: str,
    reference_line: float | None,
) -> None:
    panel = summary[summary["failure_axis"].eq(axis_name)]
    for method in METHOD_ORDER:
        group = panel[panel["method"].eq(method)].sort_values("training_fraction")
        if group.empty:
            continue
        x = group["training_fraction"].to_numpy(dtype=float) * 100
        y = group[value_column].to_numpy(dtype=float)
        lower = y - group["ci95_low"].to_numpy(dtype=float)
        upper = group["ci95_high"].to_numpy(dtype=float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.8,
            markersize=5.2,
            capsize=2.8,
            elinewidth=1.0,
            label=METHOD_PLOT_LABELS[method],
        )
    if reference_line is not None:
        ax.axhline(reference_line, color="#777777", linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_title(
        f"({panel_letter}) {FAILURE_AXIS_LABELS[axis_name]}",
        fontsize=11.2,
        fontweight="bold",
    )
    ax.set_xlabel("Training-set fraction used")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def _add_shared_legend(fig: plt.Figure, ax: plt.Axes, *, y: float) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=6,
        frameon=False,
        fontsize=8.2,
    )


def _render_two_panel_overview(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    reference_line: float | None,
    dpi: int = 600,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.45), sharey=True)
    for panel_index, (axis_name, ax) in enumerate(zip(FAILURE_AXIS_ORDER, axes)):
        _draw_metric_panel(
            ax,
            summary,
            axis_name=axis_name,
            value_column=value_column,
            panel_letter="a" if panel_index == 0 else "b",
            reference_line=reference_line,
        )
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    _add_shared_legend(fig, axes[0], y=0.015)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.20, wspace=0.12)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_overview(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 600,
) -> None:
    _render_two_panel_overview(
        summary,
        output_path,
        value_column="mean_degradation_ratio",
        ylabel="Degradation ratio\n(OOD MAE / Random-CV MAE)",
        title="Performance-degradation ratio vs. training-set size (all six model families)",
        reference_line=1.0,
        dpi=dpi,
    )


def render_absolute_overview(
    summary: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 600,
) -> None:
    _render_two_panel_overview(
        summary,
        output_path,
        value_column="mean_ood_mae",
        ylabel="Absolute OOD MAE\n(task-balanced native-unit mean)",
        title="Absolute OOD MAE vs. training-set size (all six model families)",
        reference_line=None,
        dpi=dpi,
    )


def render_combined_overview(
    degradation_summary: pd.DataFrame,
    absolute_summary: pd.DataFrame,
    output_path: str | Path,
    *,
    dpi: int = 600,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0), sharey="row")
    for index, axis_name in enumerate(FAILURE_AXIS_ORDER):
        _draw_metric_panel(
            axes[0, index],
            degradation_summary,
            axis_name=axis_name,
            value_column="mean_degradation_ratio",
            panel_letter="a" if index == 0 else "b",
            reference_line=1.0,
        )
        _draw_metric_panel(
            axes[1, index],
            absolute_summary,
            axis_name=axis_name,
            value_column="mean_ood_mae",
            panel_letter="c" if index == 0 else "d",
            reference_line=None,
        )
    axes[0, 0].set_ylabel("Degradation ratio\n(OOD MAE / Random-CV MAE)")
    axes[1, 0].set_ylabel("Absolute OOD MAE\n(task-balanced native-unit mean)")
    fig.suptitle("Training-size OOD overview (all six model families)", fontsize=14, fontweight="bold")
    _add_shared_legend(fig, axes[0, 0], y=0.012)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.105, hspace=0.34, wspace=0.12)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_calculation_process(
    output_path: str | Path,
    *,
    summary: pd.DataFrame,
    task_repeat: pd.DataFrame,
    absolute_summary: pd.DataFrame,
    absolute_task_repeat: pd.DataFrame,
    bootstrap_iterations: int,
    random_seed: int,
) -> None:
    method_counts = (
        task_repeat.groupby("method")["task_id"].nunique().reindex(METHOD_ORDER)
    )
    lines = [
        "# Figure 4 overview calculation process",
        "",
        "For each method, task, repeat, training fraction, and OOD axis, the degradation ratio was calculated as:",
        "",
        "`r = OOD MAE / Random-CV MAE`",
        "",
        "Repeat-level ratios were first averaged within each task. The plotted point is the arithmetic mean of the task means, so every task contributes equal weight:",
        "",
        "`task_mean = mean(r across repeats)`",
        "",
        "`overview_mean = mean(task_mean across tasks)`",
        "",
        f"The confidence interval used {bootstrap_iterations:,} hierarchical bootstrap iterations with seed `{random_seed}`. Each iteration resampled task IDs with replacement and then resampled repeat-level ratios within every selected task with replacement. The 2.5th and 97.5th percentiles of the bootstrap means were reported as the 95% CI.",
        "",
        "The 100% training fraction used the complete training pool and was evaluated once; therefore, its repeat count is one.",
        "",
        "## Coverage",
        "",
        f"- Degradation summary rows: {len(summary)} (six methods x two OOD axes x five fractions).",
        f"- Absolute-MAE summary rows: {len(absolute_summary)}.",
        f"- Degradation task-mean rows: {len(task_repeat)}.",
        f"- Absolute-MAE task-mean rows: {len(absolute_task_repeat)}.",
    ]
    for method, count in method_counts.items():
        lines.append(f"- {method}: {int(count)} tasks.")
    lines.extend(
        [
            "",
            "Traditional-best and BERT-best were selected separately for each task and training fraction using Random-CV MAE only. All six displayed methods use the same eight display tasks. GPT-5.4 rows were obtained from the saved few-shot evaluations and were not retrained during this calculation.",
            "",
            "The absolute-MAE overview averages native-unit OOD MAE values across tasks. Because the tasks represent different properties and units, it is provided for global inspection; the degradation-ratio overview is the preferred cross-task comparison.",
        ]
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_overview_from_sources(
    input_csv: str | Path,
    gpt_root: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
    dpi: int = 600,
    cache_dir: str | Path | None = None,
    recompute_bootstrap: bool = False,
) -> dict[str, Any]:
    raw = pd.read_csv(input_csv)
    gpt_rows = load_gpt54_rows(gpt_root)
    data = prepare_overview_data(raw, gpt_rows)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_source = Path(cache_dir).resolve() if cache_dir is not None else output
    expectation = _cache_expectation(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    cached_manifest = _load_cache_manifest(cache_source)
    cache_hit = not recompute_bootstrap and _cache_is_valid(
        cache_source,
        cached_manifest,
        expectation,
    )
    if cache_hit:
        (
            summary,
            task_repeat,
            bootstrap,
            absolute_summary,
            absolute_task_repeat,
            absolute_bootstrap,
        ) = _load_cached_tables(cache_source)
        cache_status = "hit"
    else:
        summary, task_repeat, bootstrap = build_overview_tables(
            data,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        absolute_summary, absolute_task_repeat, absolute_bootstrap = build_absolute_overview_tables(
            data,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        cache_status = "forced_recompute" if recompute_bootstrap else "miss"
    summary.to_csv(output / "figure4_six_method_summary.csv", index=False)
    task_repeat.to_csv(output / "figure4_six_method_task_repeat_means.csv", index=False)
    bootstrap.to_csv(output / "figure4_six_method_bootstrap_summary.csv", index=False)
    absolute_summary.to_csv(output / "figure4_six_method_absolute_mae_summary.csv", index=False)
    absolute_task_repeat.to_csv(
        output / "figure4_six_method_absolute_mae_task_repeat_means.csv", index=False
    )
    absolute_bootstrap.to_csv(
        output / "figure4_six_method_absolute_mae_bootstrap_summary.csv", index=False
    )
    write_calculation_process(
        output / "figure4_calculation_process.md",
        summary=summary,
        task_repeat=task_repeat,
        absolute_summary=absolute_summary,
        absolute_task_repeat=absolute_task_repeat,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    render_overview(
        summary,
        output / "figure4_performance_degradation_ratio.png",
        dpi=dpi,
    )
    render_absolute_overview(
        absolute_summary,
        output / "figure4_absolute_ood_mae_overview.png",
        dpi=dpi,
    )
    render_combined_overview(
        summary,
        absolute_summary,
        output / "figure4_all_overviews.png",
        dpi=dpi,
    )
    manifest = {
        **expectation,
        "cache_status": cache_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            filename: _sha256_file(output / filename)
            for filename in CACHE_DATA_FILENAMES
        },
    }
    (output / CACHE_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "output_dir": str(output),
        "summary_rows": int(len(summary)),
        "task_repeat_rows": int(len(task_repeat)),
        "absolute_summary_rows": int(len(absolute_summary)),
        "absolute_task_repeat_rows": int(len(absolute_task_repeat)),
        "methods": list(METHOD_ORDER),
        "task_counts": {
            method: int(task_repeat.loc[task_repeat["method"].eq(method), "task_id"].nunique())
            for method in METHOD_ORDER
        },
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(random_seed),
        "cache_status": cache_status,
        "selected_data_sha256": expectation["selected_data_sha256"],
        "cache_manifest": str((output / CACHE_FILENAME).resolve()),
        "images": {
            "degradation": str((output / "figure4_performance_degradation_ratio.png").resolve()),
            "absolute_mae": str((output / "figure4_absolute_ood_mae_overview.png").resolve()),
            "combined": str((output / "figure4_all_overviews.png").resolve()),
        },
    }
