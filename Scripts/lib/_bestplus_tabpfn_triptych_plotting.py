from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _direction_to_ascending(direction: str) -> bool:
    normalized = str(direction).strip().lower()
    if normalized in {"asc", "ascending", "low", "lower", "min", "minimum"}:
        return True
    if normalized in {"desc", "descending", "high", "higher", "max", "maximum"}:
        return False
    raise ValueError(f"Unsupported sort direction: {direction}")


def _compute_relative_to_baseline(
    value: float,
    baseline: float,
    *,
    mode: str,
    better_direction: str,
) -> float:
    if pd.isna(value) or pd.isna(baseline):
        return np.nan
    if float(baseline) == 0.0:
        return np.nan

    if mode == "delta":
        return float(value) - float(baseline)
    if mode == "improvement_pct":
        if str(better_direction).strip().lower() == "lower":
            return (float(baseline) - float(value)) / float(baseline) * 100.0
        return (float(value) - float(baseline)) / abs(float(baseline)) * 100.0
    if mode == "delta_pct":
        return (float(value) - float(baseline)) / abs(float(baseline)) * 100.0

    raise ValueError(f"Unsupported baseline relative mode: {mode}")


def _resolve_alias_map(config: dict, key: str) -> dict[str, str]:
    return {
        str(source): str(target)
        for source, target in config.get("labels", {}).get(key, {}).items()
    }


def _display_name(values: list[str], alias_map: dict[str, str]) -> list[str]:
    return [alias_map.get(value, value) for value in values]


def _panel_enabled(config: dict, panel_name: str) -> bool:
    return bool(config.get("panels", {}).get(panel_name, {}).get("enabled", True))


def _maybe_set_tick_fontsize(axis, fontsize: float | None) -> None:
    if fontsize is None:
        return
    axis.tick_params(axis="both", labelsize=fontsize)


def prepare_task_table(task_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    display_cfg = config["display"]
    baseline_metric = display_cfg["baseline_metric"]
    baseline_reference_label = display_cfg["baseline_reference_label"]
    baseline_relative_mode = display_cfg.get("baseline_relative_mode", "improvement_pct")
    baseline_better_direction = display_cfg.get("baseline_better_direction", "lower")

    rank_metric = display_cfg["rank_metric"]
    rank_metric_direction = display_cfg.get("rank_metric_direction", "desc")
    rank_std_metric = display_cfg["rank_std_metric"]
    rank_std_metric_direction = display_cfg.get("rank_std_metric_direction", "asc")
    rank_tiebreak_metric = display_cfg.get("rank_tiebreak_metric")
    rank_tiebreak_metric_direction = display_cfg.get("rank_tiebreak_metric_direction", "asc")

    rank_parts: list[pd.DataFrame] = []
    for _, method_df in task_df.groupby("ood_method", observed=True, sort=False):
        working_df = method_df.copy()

        baseline_row = working_df.loc[working_df["aggregate_label"] == baseline_reference_label, baseline_metric]
        baseline_value = pd.to_numeric(baseline_row, errors="coerce").iloc[0] if not baseline_row.empty else np.nan
        working_df["baseline_reference_label"] = baseline_reference_label
        working_df["baseline_reference_value"] = baseline_value
        working_df["baseline_delta"] = pd.to_numeric(working_df[baseline_metric], errors="coerce") - baseline_value
        working_df["baseline_relative_value"] = [
            _compute_relative_to_baseline(
                value,
                baseline_value,
                mode=baseline_relative_mode,
                better_direction=baseline_better_direction,
            )
            for value in pd.to_numeric(working_df[baseline_metric], errors="coerce")
        ]

        sort_columns: list[str] = []
        ascending: list[bool] = []

        working_df["_rank_metric"] = pd.to_numeric(working_df[rank_metric], errors="coerce")
        sort_columns.append("_rank_metric")
        ascending.append(_direction_to_ascending(rank_metric_direction))

        working_df["_rank_std_metric"] = pd.to_numeric(working_df[rank_std_metric], errors="coerce")
        sort_columns.append("_rank_std_metric")
        ascending.append(_direction_to_ascending(rank_std_metric_direction))

        if rank_tiebreak_metric:
            working_df["_rank_tiebreak_metric"] = pd.to_numeric(
                working_df[rank_tiebreak_metric], errors="coerce"
            )
            sort_columns.append("_rank_tiebreak_metric")
            ascending.append(_direction_to_ascending(rank_tiebreak_metric_direction))

        working_df = working_df.sort_values(
            sort_columns + ["aggregate_label"],
            ascending=ascending + [True],
            kind="mergesort",
            na_position="last",
        )
        working_df["rank_in_method"] = np.arange(1, len(working_df) + 1, dtype=int)
        cleanup_columns = ["_rank_metric", "_rank_std_metric", "_rank_tiebreak_metric"]
        rank_parts.append(working_df.drop(columns=[c for c in cleanup_columns if c in working_df.columns]))

    ranked_df = pd.concat(rank_parts, ignore_index=True)
    return ranked_df.sort_values(["ood_method", "aggregate_label"]).reset_index(drop=True)


def _plot_summary_panel(
    ax,
    *,
    pivot_summary: pd.DataFrame,
    pivot_summary_std: pd.DataFrame,
    method_order: list[str],
    method_labels: list[str],
    series_order: list[str],
    series_labels: list[str],
    palette: dict[str, str],
    config: dict,
) -> None:
    figure_cfg = config["figure"]
    labels_cfg = config["labels"]
    display_cfg = config["display"]

    x = np.arange(len(method_order))
    width = float(figure_cfg["bar_width"])
    offsets = np.linspace(
        -0.5 * width * (len(series_order) - 1),
        0.5 * width * (len(series_order) - 1),
        len(series_order),
    )

    show_errorbars = bool(figure_cfg.get("summary_show_errorbars", True))
    fallback_palette = sns.color_palette("tab10", n_colors=max(len(series_order), 1))
    containers = []
    for idx, label in enumerate(series_order):
        yerr = pivot_summary_std[label].to_numpy() if show_errorbars else None
        container = ax.bar(
            x + offsets[idx],
            pivot_summary[label].to_numpy(),
            width=width,
            yerr=yerr,
            capsize=figure_cfg.get("bar_capsize", 3),
            color=palette.get(label, fallback_palette[idx % len(fallback_palette)]),
            edgecolor=figure_cfg.get("bar_edgecolor", "black"),
            linewidth=figure_cfg.get("bar_linewidth", 0.8),
            label=series_labels[idx],
        )
        containers.append(container)

    if figure_cfg.get("summary_zero_line", False):
        ax.axhline(0, color="black", lw=1, alpha=0.6)

    if figure_cfg.get("summary_show_value_labels", False):
        label_fmt = "{:" + figure_cfg.get("summary_value_label_fmt", ".1f") + "}"
        for container in containers:
            ax.bar_label(
                container,
                fmt=label_fmt,
                fontsize=figure_cfg.get("summary_value_label_fontsize", 8),
                padding=figure_cfg.get("summary_value_label_padding", 3),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=figure_cfg.get("summary_xtick_rotation", 20))
    ax.set_xlabel(labels_cfg.get("summary_xlabel", "OOD method"), fontsize=figure_cfg.get("axis_label_fontsize"))
    ax.set_ylabel(labels_cfg.get("summary_ylabel", display_cfg["summary_metric"]), fontsize=figure_cfg.get("axis_label_fontsize"))
    ax.set_title(figure_cfg["summary_title"], fontsize=figure_cfg.get("panel_title_fontsize"))
    _maybe_set_tick_fontsize(ax, figure_cfg.get("tick_label_fontsize"))

    if figure_cfg.get("summary_ylim") is not None:
        ax.set_ylim(*figure_cfg["summary_ylim"])

    if figure_cfg.get("summary_show_legend", True):
        legend_bbox = figure_cfg.get("legend_bbox_to_anchor")
        ax.legend(
            title=figure_cfg.get("legend_title", "Series"),
            frameon=figure_cfg.get("legend_frameon", True),
            loc=figure_cfg.get("legend_loc", "best"),
            bbox_to_anchor=tuple(legend_bbox) if legend_bbox else None,
            ncol=int(figure_cfg.get("legend_ncol", 1)),
            fontsize=figure_cfg.get("legend_fontsize", 10),
            title_fontsize=figure_cfg.get("legend_title_fontsize", 10),
        )


def _plot_heatmap_panel(
    ax,
    *,
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    annotation_fmt: str,
    cmap: str,
    colorbar_label: str,
    config: dict,
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    cbar_shrink: float = 1.0,
) -> None:
    figure_cfg = config["figure"]
    heatmap_kwargs = {
        "data": data,
        "annot": True,
        "fmt": annotation_fmt,
        "cmap": cmap,
        "linewidths": figure_cfg.get("heatmap_linewidths", 0.5),
        "linecolor": figure_cfg.get("linecolor", "white"),
        "annot_kws": {
            "fontsize": figure_cfg.get("heatmap_annot_fontsize", 10),
            "color": figure_cfg.get("heatmap_annot_color", "black"),
        },
        "ax": ax,
    }
    if center is not None:
        heatmap_kwargs["center"] = center
    if vmin is not None:
        heatmap_kwargs["vmin"] = vmin
    if vmax is not None:
        heatmap_kwargs["vmax"] = vmax
    if show_colorbar:
        heatmap_kwargs["cbar_kws"] = {"label": colorbar_label, "shrink": cbar_shrink}
    else:
        heatmap_kwargs["cbar"] = False

    sns.heatmap(**heatmap_kwargs)
    ax.set_title(title, fontsize=figure_cfg.get("panel_title_fontsize"))
    ax.set_xlabel(xlabel, fontsize=figure_cfg.get("axis_label_fontsize"))
    ax.set_ylabel(ylabel, fontsize=figure_cfg.get("axis_label_fontsize"))
    ax.tick_params(axis="x", rotation=figure_cfg.get("heatmap_xtick_rotation", 20))
    ax.tick_params(axis="y", rotation=figure_cfg.get("heatmap_ytick_rotation", 0))
    _maybe_set_tick_fontsize(ax, figure_cfg.get("tick_label_fontsize"))


def plot_triptych(task_df: pd.DataFrame, title: str, output_path: Path, config: dict) -> None:
    figure_cfg = config["figure"]
    labels_cfg = config["labels"]
    sns.set_theme(
        style="whitegrid",
        font="DejaVu Sans",
        context=figure_cfg.get("context", "notebook"),
        font_scale=figure_cfg.get("font_scale", 1.0),
    )

    method_order = list(config["method_order"])
    if "active_series_order" in task_df.columns and task_df["active_series_order"].notna().any():
        series_order = [
            label
            for label in str(task_df["active_series_order"].dropna().iloc[0]).split(",")
            if label
        ]
    else:
        configured_order = [str(label) for label in config["series_order"]]
        present_labels = task_df["aggregate_label"].astype(str).drop_duplicates().tolist()
        series_order = [label for label in configured_order if label in present_labels]
        series_order.extend(sorted([label for label in present_labels if label not in series_order]))
    palette = config["palette"]
    display_cfg = config["display"]

    method_aliases = _resolve_alias_map(config, "method_aliases")
    series_aliases = _resolve_alias_map(config, "series_aliases")
    method_labels = _display_name(method_order, method_aliases)
    series_labels = _display_name(series_order, series_aliases)

    summary_metric = display_cfg["summary_metric"]
    summary_std_metric = display_cfg["summary_std_metric"]
    baseline_colorbar_label = display_cfg["baseline_colorbar_label"]

    table_df = prepare_task_table(task_df, config)
    pivot_summary = (
        table_df.pivot(index="ood_method", columns="aggregate_label", values=summary_metric)
        .reindex(index=method_order, columns=series_order)
    )
    pivot_summary_std = (
        table_df.pivot(index="ood_method", columns="aggregate_label", values=summary_std_metric)
        .reindex(index=method_order, columns=series_order)
        .fillna(0.0)
    )
    pivot_baseline = (
        table_df.pivot(index="aggregate_label", columns="ood_method", values="baseline_relative_value")
        .reindex(index=series_order, columns=method_order)
    )
    pivot_rank = (
        table_df.pivot(index="aggregate_label", columns="ood_method", values="rank_in_method")
        .reindex(index=series_order, columns=method_order)
    )

    pivot_baseline_display = pivot_baseline.copy()
    pivot_baseline_display.index = series_labels
    pivot_baseline_display.columns = method_labels

    pivot_rank_display = pivot_rank.copy()
    pivot_rank_display.index = series_labels
    pivot_rank_display.columns = method_labels

    enabled_panels = [panel for panel in ("summary", "baseline", "rank") if _panel_enabled(config, panel)]
    if not enabled_panels:
        raise ValueError("At least one panel must be enabled in config.panels")

    panel_layout = figure_cfg.get("panel_layout")
    if panel_layout:
        filtered_layout: list[list[str]] = []
        for row in panel_layout:
            filtered_row = [cell for cell in row if cell in enabled_panels]
            if filtered_row:
                filtered_layout.append(filtered_row)
        if not filtered_layout:
            raise ValueError("figure.panel_layout filtered down to zero panels; check enabled panels vs layout")

        width_ratios = [
            figure_cfg["panel_width_ratios"].get(cell, 1.0)
            for cell in filtered_layout[0]
        ]
        height_ratios_cfg = figure_cfg.get("panel_height_ratios")
        gridspec_kw = {"width_ratios": width_ratios}
        if height_ratios_cfg:
            gridspec_kw["height_ratios"] = height_ratios_cfg

        fig = plt.figure(figsize=tuple(figure_cfg["figsize"]))
        panel_axes = fig.subplot_mosaic(filtered_layout, gridspec_kw=gridspec_kw)
    else:
        width_ratios = [figure_cfg["panel_width_ratios"][panel] for panel in enabled_panels]
        fig, axes = plt.subplots(
            1,
            len(enabled_panels),
            figsize=tuple(figure_cfg["figsize"]),
            gridspec_kw={"width_ratios": width_ratios},
        )
        axes = np.atleast_1d(axes)
        panel_axes = dict(zip(enabled_panels, axes, strict=False))

    if "summary" in panel_axes:
        _plot_summary_panel(
            panel_axes["summary"],
            pivot_summary=pivot_summary,
            pivot_summary_std=pivot_summary_std,
            method_order=method_order,
            method_labels=method_labels,
            series_order=series_order,
            series_labels=series_labels,
            palette=palette,
            config=config,
        )

    if "baseline" in panel_axes:
        _plot_heatmap_panel(
            panel_axes["baseline"],
            data=pivot_baseline_display,
            title=figure_cfg["baseline_title"],
            xlabel=labels_cfg.get("baseline_xlabel", "OOD method"),
            ylabel=labels_cfg.get("baseline_ylabel", ""),
            annotation_fmt=figure_cfg.get("baseline_annotation_fmt", ".1f"),
            cmap=figure_cfg.get("baseline_heatmap_cmap", "RdBu"),
            colorbar_label=baseline_colorbar_label,
            config=config,
            center=figure_cfg.get("baseline_heatmap_center", 0.0),
            vmin=figure_cfg.get("baseline_vmin"),
            vmax=figure_cfg.get("baseline_vmax"),
            show_colorbar=figure_cfg.get("baseline_show_colorbar", True),
            cbar_shrink=float(figure_cfg.get("baseline_cbar_shrink", 1.0)),
        )

    if "rank" in panel_axes:
        _plot_heatmap_panel(
            panel_axes["rank"],
            data=pivot_rank_display,
            title=figure_cfg["rank_title"],
            xlabel=labels_cfg.get("rank_xlabel", "OOD method"),
            ylabel=labels_cfg.get("rank_ylabel", ""),
            annotation_fmt=figure_cfg.get("rank_annotation_fmt", ".0f"),
            cmap=figure_cfg.get("rank_cmap", "Blues_r"),
            colorbar_label=display_cfg["rank_colorbar_label"],
            config=config,
            center=None,
            vmin=figure_cfg.get("rank_vmin", 1),
            vmax=figure_cfg.get("rank_vmax", len(series_order)),
            show_colorbar=figure_cfg.get("rank_show_colorbar", True),
            cbar_shrink=1.0,
        )

    if figure_cfg.get("show_suptitle", True):
        fig.suptitle(title, fontsize=figure_cfg.get("title_fontsize", 16), y=figure_cfg.get("suptitle_y", 1.02))

    plt.tight_layout(pad=figure_cfg.get("tight_layout_pad", 1.08))
    fig.subplots_adjust(
        wspace=float(figure_cfg.get("wspace", 0.26)),
        hspace=float(figure_cfg.get("hspace", 0.22)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=figure_cfg.get("dpi", 300),
        bbox_inches=figure_cfg.get("bbox_inches", "tight"),
        facecolor=figure_cfg.get("save_facecolor", "white"),
        transparent=bool(figure_cfg.get("save_transparent", False)),
    )
    plt.close(fig)
