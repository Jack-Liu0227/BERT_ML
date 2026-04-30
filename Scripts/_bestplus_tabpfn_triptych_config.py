from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TRIPTYCH_CONFIG: dict[str, Any] = {
    "method_order": [
        "RandomCV",
        "Extrapolation",
        "LOCO",
        "SparseXcluster",
        "SparseXsingle",
        "SparseYcluster",
        "SparseYsingle",
    ],
    "series_order": [
        "BERT-best",
        "Traditional-best",
        "TabPFN-2.5-Plus-Numeric",
        "TabPFN-2.5-Plus-Text",
        "gpt-5.4",
    ],
    "palette": {
        "BERT-best": "#009E73",
        "Traditional-best": "#3B5BDB",
        "TabPFN-2.5-Plus-Numeric": "#E66100",
        "TabPFN-2.5-Plus-Text": "#D81B60",
        "gpt-5.4": "#4D4D4D",
    },
    "labels": {
        "method_aliases": {},
        "series_aliases": {},
        "summary_xlabel": "OOD method",
        "baseline_xlabel": "OOD method",
        "rank_xlabel": "OOD method",
        "summary_ylabel": "OOD-test MAE",
        "baseline_ylabel": "",
        "rank_ylabel": "",
    },
    "selection": {
        "family_best_metric": "plot_test_mae",
        "family_best_metric_direction": "asc",
        "family_best_std_metric": "summary_test_mae_std",
        "family_best_std_metric_direction": "asc",
        "family_best_tiebreak_metric": "summary_test_r2",
        "family_best_tiebreak_metric_direction": "desc",
    },
    "display": {
        "summary_metric": "plot_test_mae",
        "summary_std_metric": "summary_test_mae_std",
        "summary_metric_direction": "asc",
        "baseline_metric": "plot_test_mae",
        "baseline_reference_label": "Traditional-best",
        "baseline_relative_mode": "improvement_pct",
        "baseline_better_direction": "lower",
        "rank_metric": "plot_test_mae",
        "rank_metric_direction": "asc",
        "rank_std_metric": "summary_test_mae_std",
        "rank_std_metric_direction": "asc",
        "rank_tiebreak_metric": "summary_test_r2",
        "rank_tiebreak_metric_direction": "desc",
        "negative_flag_metric": "plot_test_r2",
        "baseline_colorbar_label": "MAE improvement vs Traditional-best (%)",
        "rank_colorbar_label": "rank in method (1=best)",
    },
    "panels": {
        "summary": {"enabled": True},
        "baseline": {"enabled": True},
        "rank": {"enabled": True},
    },
    "figure": {
        "context": "notebook",
        "font_scale": 1.0,
        "figsize": [18, 6],
        "panel_layout": None,
        "panel_width_ratios": {
            "summary": 2.2,
            "baseline": 1.1,
            "rank": 1.1,
        },
        "panel_height_ratios": None,
        "wspace": 0.26,
        "hspace": 0.22,
        "bar_width": 0.18,
        "bar_capsize": 3,
        "bar_edgecolor": "black",
        "bar_linewidth": 0.8,
        "summary_zero_line": False,
        "summary_show_errorbars": False,
        "summary_show_legend": True,
        "summary_show_value_labels": False,
        "summary_value_label_fmt": ".1f",
        "summary_value_label_fontsize": 8,
        "summary_value_label_padding": 3,
        "summary_ylim": None,
        "summary_xtick_rotation": 20,
        "heatmap_xtick_rotation": 20,
        "heatmap_ytick_rotation": 0,
        "dpi": 300,
        "legend_title": "Series",
        "legend_loc": "best",
        "legend_bbox_to_anchor": None,
        "legend_ncol": 1,
        "legend_fontsize": 10,
        "legend_title_fontsize": 10,
        "legend_frameon": True,
        "heatmap_linewidths": 0.5,
        "linecolor": "white",
        "heatmap_annot_fontsize": 10,
        "heatmap_annot_color": "black",
        "baseline_annotation_fmt": ".1f",
        "baseline_show_colorbar": True,
        "baseline_cbar_shrink": 1.0,
        "baseline_heatmap_cmap": "RdBu",
        "baseline_heatmap_center": 0.0,
        "baseline_vmin": None,
        "baseline_vmax": None,
        "rank_annotation_fmt": ".0f",
        "rank_show_colorbar": True,
        "rank_cmap": "Blues_r",
        "rank_vmin": 1,
        "rank_vmax": None,
        "title_fontsize": 16,
        "panel_title_fontsize": None,
        "axis_label_fontsize": None,
        "tick_label_fontsize": None,
        "show_suptitle": True,
        "suptitle_y": 1.02,
        "tight_layout_pad": 1.08,
        "summary_title": "MAE summary",
        "baseline_title": "MAE vs Traditional-best baseline (%)",
        "rank_title": "Per-method rank (by OOD-test MAE)",
        "save_facecolor": "white",
        "save_transparent": False,
        "bbox_inches": "tight",
    },
    "output": {
        "clean_output_dir": True,
        "data_subdir": "data",
        "figure_subdir": "figure",
        "title_template": "OOD summary for {alloy_family} | {dataset_name} | {property}",
        "formats": ["png"],
    },
}


def default_config_path() -> Path:
    yaml_path = Path(__file__).with_name("build_bestplus_tabpfn_triptych.config.yaml")
    if yaml_path.exists():
        return yaml_path
    return Path(__file__).with_name("build_bestplus_tabpfn_triptych.config.json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_override(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            loaded = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(handle) or {}
        else:
            raise ValueError(f"Unsupported config format: {config_path}")

    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping/object: {config_path}")
    return loaded


def load_triptych_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = config_path or default_config_path()
    config = deepcopy(DEFAULT_TRIPTYCH_CONFIG)
    if resolved_path.exists():
        override = _load_config_override(resolved_path)
        config = _deep_merge(config, override)

    missing_palette = [label for label in config["series_order"] if label not in config["palette"]]
    if missing_palette:
        raise ValueError(f"Palette is missing series labels: {missing_palette}")

    panel_ratios = config["figure"].get("panel_width_ratios", {})
    missing_ratios = [panel for panel in ("summary", "baseline", "rank") if panel not in panel_ratios]
    if missing_ratios:
        raise ValueError(f"Figure panel_width_ratios is missing panels: {missing_ratios}")

    panel_layout = config["figure"].get("panel_layout")
    if panel_layout is not None and not isinstance(panel_layout, list):
        raise ValueError("figure.panel_layout must be null or a list of rows")

    output_formats = config["output"].get("formats", ["png"])
    if not output_formats:
        raise ValueError("At least one output format must be provided in output.formats")

    return config


# python Scripts\batch_summarize_extrapolation_results.py
# python Scripts\batch_summarize_bert_extrapolation_results.py
# python Scripts\batch_summarize_tabpfn_extrapolation_results.py
# python Scripts\batch_summarize_combined_ood_reports.py
# python Scripts\build_bestplus_tabpfn_triptych.py --output-dir output\ood_summary_reports\Combined\figure\per_task_bestplus_tabpfn
