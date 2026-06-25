from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .config import MANIFEST_COLUMNS, MODEL_FAMILY_COLORS, REPRESENTATION_SPACES, SPACE_ORDER, SUBSET_MARKERS
from .io import clean_text, safe_filename
from .summaries import method_sort_key


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    figure_id: str,
    formats: Iterable[str],
    dpi: int,
    *,
    scope: str = "overview",
    task_id: str = "all",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task_dir = "all" if scope == "overview" else safe_filename(task_id)
    figure_dir = output_dir / "figures" / scope if scope == "overview" else output_dir / "figures" / scope / task_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = figure_dir / f"{figure_id}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        rows.append({"scope": scope, "task_id": task_id, "figure_id": figure_id, "format": fmt, "figure": str(path)})
    plt.close(fig)
    return rows


def nonempty_or_message(ax: plt.Axes, frame: pd.DataFrame, message: str = "No data") -> bool:
    if frame.empty:
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#777777")
        ax.set_axis_off()
        return False
    return True


def family_color(family: object) -> str:
    return MODEL_FAMILY_COLORS.get(clean_text(family), "#333333")


def method_colors(methods: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {method: cmap(index % 10) for index, method in enumerate(methods)}


def legend_label_for_group(family: object, group: pd.DataFrame) -> str:
    family_text = clean_text(family)
    models = sorted(group.get("model", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    if family_text == "GPT" and len(models) == 1:
        return models[0]
    return family_text


def collect_figure_legend(fig: plt.Figure) -> tuple[list[object], list[str]]:
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in fig.axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label == "_nolegend_" or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def add_bottom_legend(fig: plt.Figure, *, max_cols: int = 6, bottom: float = 0.14) -> None:
    handles, labels = collect_figure_legend(fig)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(max_cols, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, bottom, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))


def is_gpt54_model(model: object, family: object) -> bool:
    text = f"{clean_text(model)} {clean_text(family)}".lower()
    compact = text.replace("-", "").replace("_", "").replace(" ", "")
    return "gpt" in compact and ("5.4" in compact or "gpt54" in compact or "gpt5o4" in compact)


def ordered_methods(frame: pd.DataFrame) -> list[str]:
    methods = frame.get("method", pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
    return sorted(methods, key=method_sort_key)


def ordered_models(frame: pd.DataFrame, metric: str, *, ascending: bool = True) -> list[str]:
    if frame.empty:
        return []
    work = frame.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    order = work.groupby("model")[metric].mean().sort_values(ascending=ascending, kind="stable").index.tolist()
    return [str(model) for model in order]


def centered_limit(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), 95))
    return limit if math.isfinite(limit) and limit > 0 else 1.0


def draw_centered_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    annotate: bool = False,
    fig: plt.Figure | None = None,
) -> object:
    values = matrix.to_numpy(dtype=float) if not matrix.empty else np.empty((0, 0))
    if values.size == 0:
        nonempty_or_message(ax, matrix)
        return None
    vmax = centered_limit(values)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.grid(False)
    ax.set_xlabel("Method")
    ax.set_ylabel("Model")
    if annotate and matrix.shape[0] * matrix.shape[1] <= 80:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    ax.text(col_idx, row_idx, f"{value:.2g}", ha="center", va="center", fontsize=6, color="#222222")
    if fig is not None:
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label=colorbar_label)
    return image


def pivot_metric_matrix(
    frame: pd.DataFrame,
    *,
    value_col: str,
    model_order: list[str],
    column_order: list[str],
    column_col: str = "method",
) -> pd.DataFrame:
    pivot = frame.pivot_table(index="model", columns=column_col, values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=model_order, columns=column_order)
    return pivot


def figure_1_severity_decomposition(summary: pd.DataFrame) -> plt.Figure:
    methods = sorted(summary["method"].dropna().unique(), key=method_sort_key) or ["LOCO"]
    fig, axes = plt.subplots(len(methods), len(SPACE_ORDER), figsize=(4.0 * len(SPACE_ORDER), 3.0 * len(methods)), squeeze=False)
    fig.suptitle("Hybrid Set A/B W Severity Decomposition (fold-weighted)", fontsize=14, fontweight="bold")
    for row_idx, method in enumerate(methods):
        for col_idx, space in enumerate(SPACE_ORDER):
            ax = axes[row_idx, col_idx]
            panel = summary[(summary["method"].eq(method)) & (summary["space"].eq(space))]
            if not nonempty_or_message(ax, panel):
                continue
            data = [pd.to_numeric(panel[panel["subset"].eq(subset)]["W_p90"], errors="coerce").dropna().to_numpy() for subset in ["Set A", "Set B"]]
            ax.boxplot(data, tick_labels=["Set A", "Set B"], patch_artist=True)
            for pos, values in enumerate(data, start=1):
                if len(values):
                    ax.scatter(np.full(len(values), pos), values, s=18, alpha=0.65, color="#444444")
            ax.set_title(f"{method} | {space}", loc="left", fontsize=10)
            ax.set_ylabel("W p90")
            ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_2_error_dumbbell(summary: pd.DataFrame) -> plt.Figure:
    metric_space = summary[summary["space"].eq("Y-space")].copy()
    methods = sorted(metric_space["method"].dropna().unique(), key=method_sort_key) or ["LOCO"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.8 * len(methods), max(4.0, 0.35 * metric_space["model"].nunique())), squeeze=False)
    fig.suptitle("Hybrid Set A/B Model Error Contrast (fold-weighted)", fontsize=14, fontweight="bold")
    for ax, method in zip(axes.ravel(), methods):
        panel = metric_space[metric_space["method"].eq(method)]
        if not nonempty_or_message(ax, panel):
            continue
        pivot = panel.pivot_table(index=["model", "model_family"], columns="subset", values="MAE", aggfunc="mean").reset_index()
        if "Set A" in pivot.columns:
            fallback_columns = [column for column in ["Set A", "Set B"] if column in pivot.columns]
            pivot["_sort_MAE"] = pd.to_numeric(pivot["Set A"], errors="coerce")
            if fallback_columns:
                fallback = pivot[fallback_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                pivot["_sort_MAE"] = pivot["_sort_MAE"].fillna(fallback)
            pivot = pivot.sort_values(["_sort_MAE", "model"], kind="stable").drop(columns=["_sort_MAE"]).reset_index(drop=True)
        else:
            pivot = pivot.sort_values("model", kind="stable").reset_index(drop=True)
        y = np.arange(len(pivot))
        ax.set_yticks(y)
        ax.set_yticklabels(pivot["model"], fontsize=8)
        for idx, row in pivot.iterrows():
            a = row.get("Set A", np.nan)
            b = row.get("Set B", np.nan)
            color = family_color(row.get("model_family"))
            if pd.notna(a) and pd.notna(b):
                ax.plot([a, b], [idx, idx], color="#999999", linewidth=1.2, zorder=1)
            if pd.notna(a):
                ax.scatter(a, idx, marker=SUBSET_MARKERS["Set A"], color=color, s=36, label="_nolegend_", zorder=2)
            if pd.notna(b):
                ax.scatter(b, idx, marker=SUBSET_MARKERS["Set B"], color=color, s=36, label="_nolegend_", zorder=2)
        ax.set_title(method, loc="left")
        ax.set_xlabel("MAE")
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.7)
    subset_handles = [
        Line2D([0], [0], marker=SUBSET_MARKERS["Set A"], linestyle="None", color="#555555", markerfacecolor="#555555", label="Set A"),
        Line2D([0], [0], marker=SUBSET_MARKERS["Set B"], linestyle="None", color="#555555", markerfacecolor="#555555", label="Set B"),
    ]
    fig.legend(subset_handles, [handle.get_label() for handle in subset_handles], loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return fig


def figure_1_severity_decomposition_task_compact(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, len(SPACE_ORDER), figsize=(4.2 * len(SPACE_ORDER), 4.2), squeeze=False, sharey=False)
    fig.suptitle("Task Severity Decomposition by Space (fold-weighted)", fontsize=13, fontweight="bold")
    subset_styles = {"Set A": ("o", "#4c78a8"), "Set B": ("s", "#f58518")}
    for ax, space in zip(axes.ravel(), SPACE_ORDER):
        panel = summary[summary["space"].eq(space)].copy()
        if not nonempty_or_message(ax, panel):
            continue
        method_order = (
            panel.groupby("method")["W_p90"]
            .mean()
            .sort_values(kind="stable")
            .index.tolist()
        )
        x = np.arange(len(method_order))
        offsets = {"Set A": -0.12, "Set B": 0.12}
        for subset, (marker, color) in subset_styles.items():
            values = []
            for method in method_order:
                vals = pd.to_numeric(panel[(panel["method"].eq(method)) & (panel["subset"].eq(subset))]["W_p90"], errors="coerce").dropna()
                values.append(float(vals.mean()) if not vals.empty else np.nan)
            ax.plot(x + offsets[subset], values, marker=marker, color=color, linewidth=1.2, label=subset)
        ax.set_title(space)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=35, ha="right")
        ax.set_xlabel("Method ordered by mean W p90")
        ax.set_ylabel("W p90")
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    add_bottom_legend(fig, max_cols=2, bottom=0.17)
    return fig


def figure_2_error_dumbbell_task_compact(summary: pd.DataFrame) -> plt.Figure:
    return figure_2_error_dumbbell(summary)


def figure_3_representation_severity_sensitivity_task_compact(summary: pd.DataFrame) -> plt.Figure:
    return figure_3_representation_severity_sensitivity(summary)


def figure_4_target_ceiling_task_compact(summary: pd.DataFrame, *, loco_only: bool = False) -> plt.Figure:
    return figure_4_target_ceiling(summary, loco_only=loco_only)


def figure_3_representation_severity_sensitivity(summary: pd.DataFrame) -> plt.Figure:
    methods = sorted(summary["method"].dropna().unique(), key=method_sort_key) or ["LOCO"]
    fig, axes = plt.subplots(len(methods), len(REPRESENTATION_SPACES), figsize=(4.4 * len(REPRESENTATION_SPACES), 3.2 * len(methods)), squeeze=False)
    fig.suptitle("Representation Severity-Sensitivity Map (fold-weighted)", fontsize=14, fontweight="bold")
    for row_idx, method in enumerate(methods):
        for col_idx, space in enumerate(REPRESENTATION_SPACES):
            ax = axes[row_idx, col_idx]
            panel = summary[(summary["method"].eq(method)) & (summary["space"].eq(space))]
            if not nonempty_or_message(ax, panel):
                continue
            for subset, marker in SUBSET_MARKERS.items():
                subset_panel = panel[panel["subset"].eq(subset)]
                for family, group in subset_panel.groupby("model_family", dropna=False):
                    sizes = 22 + pd.to_numeric(group["MAE"], errors="coerce").fillna(0).clip(lower=0, upper=200).to_numpy() * 0.25
                    label = f"{legend_label_for_group(family, group)} {subset}"
                    ax.scatter(group["W_p90"], group["slope"], s=sizes, marker=marker, alpha=0.72, color=family_color(family), label=label)
            ax.axhline(0.0, color="#666666", linewidth=0.8)
            ax.set_title(f"{method} | {space}", loc="left", fontsize=10)
            ax.set_xlabel("W p90")
            ax.set_ylabel("relative error slope")
            ax.grid(True, color="#e6e6e6", linewidth=0.7)
    handles, labels = collect_figure_legend(fig)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_4_target_ceiling(summary: pd.DataFrame, *, loco_only: bool) -> plt.Figure:
    panel = summary[(summary["subset"].eq("Set A")) & (summary["space"].eq("Y-space"))].copy()
    if loco_only:
        panel = panel[panel["method"].eq("LOCO")]
        title = "Target-Ceiling Extrapolation Map (LOCO, fold-weighted)"
        figure_methods = ["LOCO"]
    else:
        title = "Target-Ceiling Extrapolation Map by Method (fold-weighted)"
        figure_methods = sorted(panel["method"].dropna().unique(), key=method_sort_key) or ["LOCO"]
    fig, axes = plt.subplots(1, len(figure_methods), figsize=(4.6 * len(figure_methods), 4.0), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for ax, method in zip(axes.ravel(), figure_methods):
        method_panel = panel[panel["method"].eq(method)]
        if not nonempty_or_message(ax, method_panel):
            continue
        for family, group in method_panel.groupby("model_family", dropna=False):
            sizes = 28 + pd.to_numeric(group["MAE"], errors="coerce").fillna(0).clip(lower=0, upper=200).to_numpy() * 0.3
            ax.scatter(group["W_p90"], group["ceiling_bias"], s=sizes, color=family_color(family), alpha=0.75, label=legend_label_for_group(family, group))
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_title(method, loc="left")
        ax.set_xlabel("Y-space W p90")
        ax.set_ylabel("ceiling bias (predicted - true)")
        ax.grid(True, color="#e6e6e6", linewidth=0.7)
    handles, labels = collect_figure_legend(fig)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figure_5_delta_heatmap(delta: pd.DataFrame) -> plt.Figure:
    metrics = [
        "delta_MAE",
        "delta_X_W",
        "delta_Y_W",
        "delta_Z_W",
        "delta_slope_X",
        "delta_slope_Z",
        "ceiling_bias_SetA",
        "underprediction_ratio_SetA",
    ]
    work = delta.copy()
    work["row_label"] = work["model"].astype(str) + " | " + work["method"].astype(str)
    matrix = work.set_index("row_label")[metrics].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.28 * max(len(matrix), 1))))
    if matrix.empty:
        nonempty_or_message(ax, matrix)
        return fig
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
    vmax = vmax if math.isfinite(vmax) and vmax > 0 else 1.0
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title("Model x Method A/B Delta Heatmap (fold-weighted)", loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="delta value")
    fig.tight_layout()
    return fig


def figure_6_hybrid_failure_disentanglement(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    desired_groups = [
        ("LOCO", "Set A", "Hybrid LOCO Set A"),
        ("LOCO", "Set B", "Hybrid LOCO Set B"),
        ("RandCV", "Set A", "Hybrid RandCV Set A"),
        ("RandCV", "Set B", "Hybrid RandCV Set B"),
    ]
    work = summary.copy()
    groups = [label for method, subset, label in desired_groups if not work[(work["method"].eq(method)) & (work["subset"].eq(subset))].empty]
    if not groups:
        groups = [
            f"Hybrid {method} {subset}"
            for method, subset in work[["method", "subset"]].drop_duplicates().sort_values(["method", "subset"]).itertuples(index=False)
        ]
    if work.empty or not groups:
        for ax in axes:
            nonempty_or_message(ax, work)
        return fig

    mae_rows = []
    for method, subset, label in desired_groups:
        vals = pd.to_numeric(
            work[(work["method"].eq(method)) & (work["subset"].eq(subset)) & (work["space"].eq("Y-space"))]["MAE"],
            errors="coerce",
        ).dropna()
        if label in groups:
            mae_rows.append((label, float(vals.mean()) if not vals.empty else np.nan))
    x = np.arange(len(mae_rows))
    axes[0].bar(x, [value for _, value in mae_rows], color="#4c78a8", alpha=0.82)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Hybrid-only Failure Disentanglement", loc="left", fontsize=12, fontweight="bold")
    axes[0].grid(True, axis="y", color="#e6e6e6", linewidth=0.7)

    offsets = np.linspace(-0.22, 0.22, len(SPACE_ORDER))
    for offset, space in zip(offsets, SPACE_ORDER):
        values = []
        for method, subset, label in desired_groups:
            if label not in groups:
                continue
            vals = pd.to_numeric(
                work[(work["method"].eq(method)) & (work["subset"].eq(subset)) & (work["space"].eq(space))]["W_p90"],
                errors="coerce",
            ).dropna()
            values.append(float(vals.mean()) if not vals.empty else np.nan)
        axes[1].bar(x + offset, values, width=0.18, alpha=0.82, label=space)
    axes[1].set_ylabel("W p90")
    axes[1].grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name for name, _ in mae_rows], rotation=25, ha="right")
    axes[1].legend(loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def figure_6_pure_vs_hybrid(pure_hybrid: pd.DataFrame) -> plt.Figure:
    return figure_6_hybrid_failure_disentanglement(pure_hybrid)


def figure_7_method_contrast(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, subset in zip(axes, ["Set A", "Set B"]):
        panel = summary[summary["subset"].eq(subset)]
        if not nonempty_or_message(ax, panel):
            continue
        methods = sorted(panel["method"].dropna().unique(), key=method_sort_key)
        x = np.arange(len(methods))
        offsets = np.linspace(-0.22, 0.22, len(SPACE_ORDER))
        for offset, space in zip(offsets, SPACE_ORDER):
            values = []
            lows = []
            highs = []
            for method in methods:
                vals = pd.to_numeric(panel[(panel["method"].eq(method)) & (panel["space"].eq(space))]["W_p90"], errors="coerce").dropna()
                mean = float(vals.mean()) if not vals.empty else np.nan
                sem = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                values.append(mean)
                lows.append(sem)
                highs.append(sem)
            ax.errorbar(x + offset, values, yerr=[lows, highs], marker="o", linewidth=1.2, capsize=3, label=space)
        ax.set_title(subset, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel("W p90")
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout()
    fig.suptitle("Method-Level Diagnostic Contrast (fold-weighted)", fontsize=14, fontweight="bold", y=1.03)
    return fig


def figure_8_yz_ood_map(yz_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.suptitle("Hybrid Set A/B Wy-Wz OOD Map (fold-weighted W mean)", fontsize=14, fontweight="bold")
    if not nonempty_or_message(ax, yz_summary):
        return fig
    work = yz_summary.copy()
    work["Wy_mean"] = pd.to_numeric(work["Wy_mean"], errors="coerce")
    work["Wz_mean"] = pd.to_numeric(work["Wz_mean"], errors="coerce")
    work = work[work["Wy_mean"].notna() & work["Wz_mean"].notna()].copy()
    if not nonempty_or_message(ax, work):
        return fig
    plot_rows = []
    for keys, group in work.groupby(["method", "subset"], dropna=False, sort=True):
        method, subset = keys
        weights = pd.to_numeric(group.get("n_samples", pd.Series(1.0, index=group.index)), errors="coerce").fillna(1.0)
        weights = weights.mask(weights <= 0, 1.0)
        plot_rows.append(
            {
                "method": method,
                "subset": subset,
                "Wy_mean": float(np.average(group["Wy_mean"].to_numpy(dtype=float), weights=weights.to_numpy(dtype=float))),
                "Wz_mean": float(np.average(group["Wz_mean"].to_numpy(dtype=float), weights=weights.to_numpy(dtype=float))),
                "MAE": float(pd.to_numeric(group.get("MAE", pd.Series(np.nan, index=group.index)), errors="coerce").mean()),
            }
        )
    work = pd.DataFrame(plot_rows)

    subset_markers = {"Set A": "o", "Set B": "s"}
    subset_labels = {"Set A": "Set A/top20", "Set B": "Set B/inner"}
    methods = sorted(work["method"].dropna().astype(str).unique().tolist(), key=method_sort_key)
    colors = method_colors(methods)
    for subset, marker in subset_markers.items():
        subset_panel = work[work["subset"].eq(subset)]
        for method in methods:
            panel = subset_panel[subset_panel["method"].eq(method)]
            if panel.empty:
                continue
            sizes = 28 + pd.to_numeric(panel.get("MAE", pd.Series(0, index=panel.index)), errors="coerce").fillna(0).clip(lower=0, upper=200).to_numpy() * 0.16
            ax.scatter(
                panel["Wy_mean"],
                panel["Wz_mean"],
                s=sizes,
                marker=marker,
                color=colors[method],
                alpha=0.76,
                edgecolors="#ffffff",
                linewidths=0.35,
                rasterized=True,
            )
    ax.set_xlabel("Wy mean (Y-space)")
    ax.set_ylabel("Wz mean (Z-space)")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    subset_handles = [
        Line2D([0], [0], marker=marker, linestyle="None", color="#555555", markerfacecolor="#555555", label=subset_labels[subset])
        for subset, marker in subset_markers.items()
    ]
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=colors[method], markerfacecolor=colors[method], label=method)
        for method in methods
    ]
    handles = [*subset_handles, *method_handles]
    if handles:
        fig.legend(handles, [handle.get_label() for handle in handles], loc="lower center", ncol=min(5, len(handles)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_builders_for_scope(summary: pd.DataFrame, delta: pd.DataFrame, pure_hybrid: pd.DataFrame, yz_summary: pd.DataFrame) -> list[tuple[str, object]]:
    return [
        ("figure_1_severity_decomposition", lambda: figure_1_severity_decomposition(summary)),
        ("figure_2_error_dumbbell", lambda: figure_2_error_dumbbell(summary)),
        ("figure_3_representation_severity_sensitivity", lambda: figure_3_representation_severity_sensitivity(summary)),
        ("figure_4_target_ceiling_loco", lambda: figure_4_target_ceiling(summary, loco_only=True)),
        ("figure_4_target_ceiling_by_method", lambda: figure_4_target_ceiling(summary, loco_only=False)),
        ("figure_5_model_method_delta_heatmap", lambda: figure_5_delta_heatmap(delta)),
        ("figure_6_hybrid_failure_disentanglement", lambda: figure_6_hybrid_failure_disentanglement(summary)),
        ("figure_7_method_diagnostic_contrast", lambda: figure_7_method_contrast(summary)),
        ("figure_8_yz_ood_map", lambda: figure_8_yz_ood_map(yz_summary)),
    ]


def figure_builders_for_task_scope(summary: pd.DataFrame, delta: pd.DataFrame, pure_hybrid: pd.DataFrame, yz_summary: pd.DataFrame) -> list[tuple[str, object]]:
    return [
        ("figure_1_severity_decomposition", lambda: figure_1_severity_decomposition_task_compact(summary)),
        ("figure_2_error_dumbbell", lambda: figure_2_error_dumbbell_task_compact(summary)),
        ("figure_3_representation_severity_sensitivity", lambda: figure_3_representation_severity_sensitivity_task_compact(summary)),
        ("figure_4_target_ceiling_loco", lambda: figure_4_target_ceiling_task_compact(summary, loco_only=True)),
        ("figure_4_target_ceiling_by_method", lambda: figure_4_target_ceiling_task_compact(summary, loco_only=False)),
        ("figure_5_model_method_delta_heatmap", lambda: figure_5_delta_heatmap(delta)),
        ("figure_6_hybrid_failure_disentanglement", lambda: figure_6_hybrid_failure_disentanglement(summary)),
        ("figure_7_method_diagnostic_contrast", lambda: figure_7_method_contrast(summary)),
        ("figure_8_yz_ood_map", lambda: figure_8_yz_ood_map(yz_summary)),
    ]


def generate_overview_figures(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    pure_hybrid: pd.DataFrame,
    yz_summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for figure_id, builder in figure_builders_for_scope(summary, delta, pure_hybrid, yz_summary):
        rows.extend(save_figure(builder(), output_dir, figure_id, formats, dpi, scope="overview", task_id="all"))
    return rows


def generate_task_figures(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    pure_hybrid: pd.DataFrame,
    yz_summary: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    task_ids = sorted(summary["case"].dropna().astype(str).unique().tolist())
    for task_id in task_ids:
        task_summary = summary[summary["case"].astype(str).eq(task_id)].copy()
        task_delta = delta[delta["case"].astype(str).eq(task_id)].copy() if not delta.empty else delta.copy()
        task_pure_hybrid = pure_hybrid[pure_hybrid["case"].astype(str).eq(task_id)].copy() if not pure_hybrid.empty else pure_hybrid.copy()
        task_yz_summary = yz_summary[yz_summary["case"].astype(str).eq(task_id)].copy() if not yz_summary.empty else yz_summary.copy()
        for figure_id, builder in figure_builders_for_task_scope(task_summary, task_delta, task_pure_hybrid, task_yz_summary):
            rows.extend(save_figure(builder(), output_dir, figure_id, formats, dpi, scope="by_task", task_id=task_id))
    return rows
