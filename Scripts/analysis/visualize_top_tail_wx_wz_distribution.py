from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.analysis.analyze_top_tail_loco_proximity import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_SPLITS_ROOT,
    CASE_COLUMNS,
    build_cluster_frames_from_hybrid_manifest,
    case_key,
    clear_existing_figures,
    discover_hybrid_manifests,
    load_analysis_config,
    load_task_dataframe,
    read_csv,
    save_figure,
    task_spec_from_payload,
    write_csv,
)
from Scripts.analysis import export_three_space_w_ood_report as three_space  # noqa: E402
from src.data_processing.strength_ood_common import (  # noqa: E402
    build_numeric_x_space,
    prepare_work_dataframe,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / "wx_wz_tail_bin_distribution"
DEFAULT_JUMP_DRIVER_CSV = DEFAULT_OUTPUT_PARENT / "sensitivity_comparison" / "csv" / "detected_jump_driver_ids.csv"
DEFAULT_EMBEDDING_DATA_DIR = three_space.DEFAULT_EMBEDDING_DATA_DIR
FOCUS_CASES = {
    "HEA__hea__UTS_MPa",
    "HEA__hea__YS_MPa",
    "Steel__steel__UTS_MPa",
    "MatbenchSteels__matbench_steels_ood__yield_strength",
}
BIN_SIZE = 0.05
BIN_COUNT = 20
BIN_LABELS = ["top 0-5%", *[f"{start}-{start + 5}%" for start in range(5, 100, 5)]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 5% target-tail bins in Wx/X-space t-SNE and Wz/Z-space embedding coordinates."
    )
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--jump-driver-csv", default=str(DEFAULT_JUMP_DRIVER_CSV))
    parser.add_argument("--embedding-data-dir", default=str(DEFAULT_EMBEDDING_DATA_DIR))
    parser.add_argument("--supplemental-embedding-data-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--focus-only", action="store_true")
    parser.add_argument("--label-jump-ids", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-matbench", action="store_true")
    parser.add_argument(
        '--marker-policy',
        choices=['all_detected', 'drop_only'],
        default='all_detected',
        help='Choose whether black outlines mark all detected entrants or only removable drop drivers.',
    )
    return parser.parse_args()


def assign_tail_bins(frame: pd.DataFrame, target_col: str, bin_size: float = BIN_SIZE) -> pd.DataFrame:
    if target_col not in frame.columns:
        raise ValueError(f"target column not found: {target_col}")
    work = frame.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).reset_index(drop=True)
    if work.empty:
        raise ValueError("at least one finite target value is required")

    ranked_index = work.sort_values(target_col, ascending=False, kind="mergesort").index
    ranks = pd.Series(np.arange(1, len(work) + 1, dtype=int), index=ranked_index).sort_index()
    bin_count = int(round(1.0 / float(bin_size)))
    rank_offsets = ranks.to_numpy(dtype=int) - 1
    percentiles = rank_offsets.astype(float) / float(len(work))
    bin_indices = np.floor((rank_offsets * bin_count) / float(len(work))).astype(int)
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    labels = ["top 0-5%", *[f"{start}-{start + 5}%" for start in range(5, 100, 5)]]
    work["target_rank"] = ranks.to_numpy(dtype=int)
    work["target_percentile"] = percentiles
    work["tail_bin_index"] = bin_indices
    work["tail_bin_start_pct"] = bin_indices * int(round(bin_size * 100))
    work["tail_bin_end_pct"] = (bin_indices + 1) * int(round(bin_size * 100))
    work["tail_bin_5pct"] = [labels[int(index)] for index in bin_indices]
    return work


def build_xspace_tsne_embedding(
    frame: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    work_df = prepare_work_dataframe(frame, target_col)
    context = build_numeric_x_space(work_df, target_col=target_col, random_state=random_state)
    projection = context.projection.rename(columns={"projection_x": "x_tsne_1", "projection_y": "x_tsne_2"}).copy()
    passthrough = [column for column in ["ID"] if column in work_df.columns]
    if passthrough:
        projection = projection.merge(work_df[["__row_id__", *passthrough]], on="__row_id__", how="left")
    return projection, list(context.feature_names)


def merge_jump_driver_flags(samples: pd.DataFrame, jump_drivers: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    result['is_detected_jump_entrant'] = False
    result['is_drop_jump_driver'] = False
    result["is_wx_jump_driver"] = False
    result["jump_interval"] = ""
    result["jump_status"] = ""
    result["driver_action"] = ""
    if jump_drivers.empty or "ID" not in result.columns or "ID" not in jump_drivers.columns:
        result["is_wx_jump_driver"] = result["is_wx_jump_driver"].astype(object)
        return result

    required = {"case_key", "ID"}
    if not required.issubset(result.columns) or not required.issubset(jump_drivers.columns):
        result["is_wx_jump_driver"] = result["is_wx_jump_driver"].astype(object)
        return result

    driver_map: dict[tuple[str, int], dict[str, str]] = {}
    for _, row in jump_drivers.iterrows():
        sample_id = pd.to_numeric(row.get("ID"), errors="coerce")
        if pd.isna(sample_id):
            continue
        key = (str(row.get("case_key")), int(sample_id))
        interval = f"{int(row.get('lower_threshold_pct', 0))}-{int(row.get('upper_threshold_pct', 0))}%"
        payload = driver_map.setdefault(key, {"jump_interval": "", "jump_status": "", "driver_action": ""})
        row_is_drop_driver = (
            str(row.get('driver_action', '')).strip() == 'drop'
            and str(row.get('jump_status', '')).strip() == 'removable'
        )
        payload['is_drop_jump_driver'] = bool(payload.get('is_drop_jump_driver', False) or row_is_drop_driver)
        for column, value in [
            ("jump_interval", interval),
            ("jump_status", str(row.get("jump_status", ""))),
            ("driver_action", str(row.get("driver_action", ""))),
        ]:
            existing = [item for item in payload[column].split(";") if item]
            if value and value not in existing:
                existing.append(value)
            payload[column] = ";".join(existing)

    for index, row in result.iterrows():
        sample_id = pd.to_numeric(row.get("ID"), errors="coerce")
        if pd.isna(sample_id):
            continue
        payload = driver_map.get((str(row.get("case_key")), int(sample_id)))
        if not payload:
            continue
        result.at[index, 'is_detected_jump_entrant'] = True
        result.at[index, 'is_drop_jump_driver'] = bool(payload.get('is_drop_jump_driver', False))
        result.at[index, "is_wx_jump_driver"] = True
        result.at[index, "jump_interval"] = payload["jump_interval"]
        result.at[index, "jump_status"] = payload["jump_status"]
        result.at[index, "driver_action"] = payload["driver_action"]
    for column in ['is_detected_jump_entrant', 'is_drop_jump_driver', 'is_wx_jump_driver']:
        result[column] = result[column].map(bool).astype(object)
    return result


def select_marker_rows(frame: pd.DataFrame, marker_policy: str) -> pd.DataFrame:
    marker_columns = {
        'all_detected': 'is_detected_jump_entrant',
        'drop_only': 'is_drop_jump_driver',
    }
    if marker_policy not in marker_columns:
        raise ValueError(f'Unknown marker policy: {marker_policy}')
    marker_column = marker_columns[marker_policy]
    if marker_column not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame[marker_column].map(bool)].copy()


def merge_zspace_coordinates(samples: pd.DataFrame, background: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    result = samples.copy()
    result["z_x"] = np.nan
    result["z_y"] = np.nan
    if background is None or background.empty:
        result["z_status"] = "missing_embedding_file"
        return result, "missing_embedding_file"
    if not {"x", "y"}.issubset(background.columns):
        result["z_status"] = "missing_embedding_xy"
        return result, "missing_embedding_xy"

    if "__id_key__" in background.columns and "ID" in result.columns:
        keys = pd.DataFrame({"__merge_order__": np.arange(len(result)), "__id_key__": pd.to_numeric(result["ID"], errors="coerce")})
        merged = keys.merge(background[["__id_key__", "x", "y"]], on="__id_key__", how="left", sort=False)
    elif "__source_index_key__" in background.columns and "__source_index__" in result.columns:
        keys = pd.DataFrame(
            {
                "__merge_order__": np.arange(len(result)),
                "__source_index_key__": pd.to_numeric(result["__source_index__"], errors="coerce"),
            }
        )
        merged = keys.merge(background[["__source_index_key__", "x", "y"]], on="__source_index_key__", how="left", sort=False)
    else:
        result["z_status"] = "missing_embedding_join_key"
        return result, "missing_embedding_join_key"

    merged = merged.sort_values("__merge_order__", kind="stable")
    result["z_x"] = pd.to_numeric(merged["x"], errors="coerce").to_numpy(dtype=float)
    result["z_y"] = pd.to_numeric(merged["y"], errors="coerce").to_numpy(dtype=float)
    status = "ok" if not result[["z_x", "z_y"]].isna().any().any() else "missing_embedding_rows"
    result["z_status"] = status
    return result, status


SUPPLEMENTAL_DATASET_TO_CASE = {
    "Steel__steel__YSMPa": ("Steel", "steel", "YS(MPa)"),
}


def _clean_embedding_background(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"x", "y"}.issubset(frame.columns) or not ({"ID", "__source_index__"} & set(frame.columns)):
        return pd.DataFrame()
    columns = [column for column in ["ID", "__source_index__", "x", "y"] if column in frame.columns]
    result = frame[columns].copy()
    if "ID" in result.columns:
        result["__id_key__"] = pd.to_numeric(result["ID"], errors="coerce")
    if "__source_index__" in result.columns:
        result["__source_index_key__"] = pd.to_numeric(result["__source_index__"], errors="coerce")
    result["x"] = pd.to_numeric(result["x"], errors="coerce")
    result["y"] = pd.to_numeric(result["y"], errors="coerce")
    if "__id_key__" in result.columns and result["__id_key__"].notna().any():
        return (
            result.dropna(subset=["__id_key__", "x", "y"])
            .sort_values("__id_key__", kind="stable")
            .drop_duplicates("__id_key__", keep="first")
            .reset_index(drop=True)
        )
    return (
        result.dropna(subset=["__source_index_key__", "x", "y"])
        .sort_values("__source_index_key__", kind="stable")
        .drop_duplicates("__source_index_key__", keep="first")
        .reset_index(drop=True)
    )


def load_combined_embedding_backgrounds(
    historical_dir: Path,
    supplemental_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    backgrounds = three_space.load_embedding_backgrounds(historical_dir)
    sources = {key: "historical" for key in backgrounds}
    if supplemental_dir is None:
        return backgrounds, sources
    for dataset_key, case_values in SUPPLEMENTAL_DATASET_TO_CASE.items():
        path = supplemental_dir / f"{dataset_key}__all_panel_test_points.csv"
        if not path.exists():
            continue
        background = _clean_embedding_background(read_csv(path))
        if background.empty:
            continue
        current_key = case_key(*case_values)
        backgrounds[current_key] = background
        sources[current_key] = "generated_local"
    return backgrounds, sources


def _bin_colors() -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("turbo", BIN_COUNT)
    return {label: cmap(index) for index, label in enumerate(BIN_LABELS)}


def _plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    label_jump_ids: bool,
    marker_policy: str,
) -> None:
    colors = _bin_colors()
    for label in BIN_LABELS:
        subset = data[data["tail_bin_5pct"].eq(label)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=18,
            color=colors[label],
            alpha=0.72,
            linewidths=0,
        )
    drivers = select_marker_rows(data, marker_policy)
    if not drivers.empty:
        ax.scatter(
            drivers[x_col],
            drivers[y_col],
            s=58,
            facecolors="none",
            edgecolors="black",
            linewidths=1.15,
            marker="o",
            label='Removable Wx jump driver' if marker_policy == 'drop_only' else 'Wx jump driver',
        )
        if label_jump_ids and "ID" in drivers.columns:
            for _, row in drivers.iterrows():
                if pd.notna(row.get("ID")) and math.isfinite(float(row[x_col])) and math.isfinite(float(row[y_col])):
                    ax.annotate(str(int(row["ID"])), (row[x_col], row[y_col]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(color="#e8e8e8", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles(marker_policy: str) -> list[Line2D]:
    colors = _bin_colors()
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[label], markeredgecolor="none", markersize=6, label=label)
        for label in BIN_LABELS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker='o',
            linestyle='',
            markerfacecolor='none',
            markeredgecolor='black',
            markersize=7,
            label='Removable Wx jump driver' if marker_policy == 'drop_only' else 'Wx jump driver',
        )
    )
    return handles


def plot_task_figures(
    task: pd.DataFrame,
    output_root: Path,
    formats: Iterable[str],
    dpi: int,
    label_jump_ids: bool = False,
    marker_policy: str = 'all_detected',
) -> None:
    current_key = str(task["case_key"].iloc[0])
    title = f"{task['alloy_family'].iloc[0]} / {task['property'].iloc[0]}"
    legend_handles = _legend_handles(marker_policy)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _plot_panel(ax, task, "x_tsne_1", "x_tsne_2", f"{title} - Wx / X-space t-SNE", label_jump_ids, marker_policy)
    ax.legend(handles=legend_handles, fontsize=6.5, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    save_figure(fig, output_root / "figures" / "xspace_tsne_by_task" / current_key, formats, dpi)

    z_ok = task["z_status"].eq("ok") & task["z_x"].notna() & task["z_y"].notna()
    if z_ok.any():
        z_task = task[z_ok].copy()
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        _plot_panel(ax, z_task, "z_x", "z_y", f"{title} - Wz / Z-space embedding", label_jump_ids, marker_policy)
        ax.legend(handles=legend_handles, fontsize=6.5, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
        save_figure(fig, output_root / "figures" / "zspace_embedding_by_task" / current_key, formats, dpi)

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        _plot_panel(axes[0], task, "x_tsne_1", "x_tsne_2", "Wx / X-space t-SNE", label_jump_ids, marker_policy)
        _plot_panel(axes[1], z_task, "z_x", "z_y", "Wz / Z-space embedding", label_jump_ids, marker_policy)
        fig.suptitle(title)
        fig.legend(handles=legend_handles, fontsize=6.5, ncol=4, loc="lower center", frameon=False)
        fig.subplots_adjust(bottom=0.22, wspace=0.28)
        save_figure(fig, output_root / "figures" / "combined_wx_wz_by_task" / current_key, formats, dpi)


def build_task_distribution_frame(
    spec,
    frame: pd.DataFrame,
    jump_drivers: pd.DataFrame,
    backgrounds: dict[str, pd.DataFrame],
    random_state: int,
    embedding_sources: dict[str, str] | None = None,
    marker_policy: str = 'all_detected',
) -> tuple[pd.DataFrame, dict[str, object]]:
    current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
    work = frame.copy()
    if "__source_index__" not in work.columns:
        work["__source_index__"] = np.arange(len(work), dtype=int)
    binned = assign_tail_bins(work, spec.target_col)
    x_projection, feature_names = build_xspace_tsne_embedding(binned, spec.target_col, random_state=random_state)
    sample = binned.merge(x_projection[["__source_index__", "x_tsne_1", "x_tsne_2"]], on="__source_index__", how="left")
    sample.insert(0, "case_key", current_key)
    sample.insert(0, "property", spec.property_name)
    sample.insert(0, "dataset_name", spec.dataset_name)
    sample.insert(0, "alloy_family", spec.alloy_family)
    sample["target_col"] = spec.target_col
    sample["target_value"] = pd.to_numeric(sample[spec.target_col], errors="coerce")
    sample = merge_jump_driver_flags(sample, jump_drivers)
    sample, z_status = merge_zspace_coordinates(sample, backgrounds.get(current_key))
    z_embedding_source = (embedding_sources or {}).get(current_key, "") if z_status == "ok" else ""
    sample["z_embedding_source"] = z_embedding_source
    manifest = {
        "case_key": current_key,
        "alloy_family": spec.alloy_family,
        "dataset_name": spec.dataset_name,
        "property": spec.property_name,
        "target_col": spec.target_col,
        "sample_count": int(len(sample)),
        "x_feature_count": int(len(feature_names)),
        "x_feature_columns": ";".join(feature_names),
        "z_status": z_status,
        "z_embedding_source": z_embedding_source,
        "jump_driver_count": int(sample['is_detected_jump_entrant'].map(bool).sum()),
        'detected_entrant_count': int(sample['is_detected_jump_entrant'].map(bool).sum()),
        'drop_jump_driver_count': int(sample['is_drop_jump_driver'].map(bool).sum()),
        'report_only_entrant_count': int(
            (sample['is_detected_jump_entrant'].map(bool) & ~sample['is_drop_jump_driver'].map(bool)).sum()
        ),
        'marker_policy': marker_policy,
    }
    return sample, manifest


def main() -> None:
    args = parse_args()
    config = load_analysis_config(Path(args.config))
    include_matbench = bool((config.get("defaults") or {}).get("include_matbench", True)) and not bool(args.exclude_matbench)
    manifests = discover_hybrid_manifests(
        Path(args.splits_root),
        case_contains=args.case_contains,
        limit=args.limit,
        include_matbench=include_matbench,
    )
    payloads = [build_cluster_frames_from_hybrid_manifest(path) for path in manifests]
    specs = [task_spec_from_payload(payload) for payload in payloads]
    if args.focus_only:
        specs = [spec for spec in specs if case_key(spec.alloy_family, spec.dataset_name, spec.property_name) in FOCUS_CASES]
    if not specs:
        raise SystemExit("No tasks matched the requested filters.")

    output_root = Path(args.output_root)
    for subdir in ["xspace_tsne_by_task", "zspace_embedding_by_task", "combined_wx_wz_by_task"]:
        clear_existing_figures(output_root / "figures" / subdir, args.formats)

    jump_drivers = read_csv(Path(args.jump_driver_csv)) if Path(args.jump_driver_csv).exists() else pd.DataFrame()
    supplemental_dir = (
        Path(args.supplemental_embedding_data_dir)
        if args.supplemental_embedding_data_dir
        else output_root / "zspace_embedding_data"
    )
    backgrounds, embedding_sources = load_combined_embedding_backgrounds(
        Path(args.embedding_data_dir),
        supplemental_dir,
    )
    sample_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for spec in specs:
        frame = load_task_dataframe(spec)
        sample, manifest = build_task_distribution_frame(
            spec=spec,
            frame=frame,
            jump_drivers=jump_drivers,
            backgrounds=backgrounds,
            random_state=int(args.random_state),
            embedding_sources=embedding_sources,
            marker_policy=args.marker_policy,
        )
        sample_parts.append(sample)
        manifest_rows.append(manifest)
        plot_task_figures(
            sample,
            output_root,
            args.formats,
            int(args.dpi),
            bool(args.label_jump_ids),
            args.marker_policy,
        )

    csv_dir = output_root / "csv"
    combined = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    manifest_frame = pd.DataFrame(manifest_rows)
    missing_frame = manifest_frame.loc[manifest_frame["z_status"].ne("ok")].copy()
    write_csv(combined, csv_dir / "wx_wz_tail_bin_embedding_by_sample.csv")
    write_csv(manifest_frame, csv_dir / "wx_wz_tail_bin_manifest.csv")
    write_csv(missing_frame, csv_dir / "missing_zspace_cases.csv")
    print(f"Tasks: {len(specs)}")
    print(f"Rows: {len(combined)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
