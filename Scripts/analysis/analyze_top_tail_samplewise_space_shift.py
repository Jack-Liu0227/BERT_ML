from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.analysis.analyze_top_tail_loco_proximity import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_SPLITS_ROOT,
    TaskSpec,
    build_cluster_frames_from_hybrid_manifest,
    case_key,
    clear_existing_figures,
    compute_space_shift_metrics,
    discover_hybrid_manifests,
    fallback_feature_columns,
    finite_float,
    load_analysis_config,
    load_task_dataframe,
    read_csv,
    save_figure,
    task_spec_from_payload,
    threshold_count,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / "xspace_samplewise_sensitivity"
DEFAULT_START_FRACTION = 0.05
DEFAULT_STOP_FRACTION = 0.95
BASELINE_POLICY = "baseline"
DROP_POLICY = "drop_samplewise_jump_drivers"
POLICY_LABELS = {
    BASELINE_POLICY: "baseline",
    DROP_POLICY: "drop samplewise jump drivers",
}
POLICY_COLORS = {
    BASELINE_POLICY: "#222222",
    DROP_POLICY: "#4c78a8",
}


def sample_count_bounds(
    total_rows: int,
    start_fraction: float = DEFAULT_START_FRACTION,
    stop_fraction: float = DEFAULT_STOP_FRACTION,
) -> tuple[int, int]:
    if int(total_rows) < 2:
        raise ValueError("at least two finite target rows are required")
    if not 0 < float(start_fraction) < float(stop_fraction) < 1:
        raise ValueError("samplewise fractions must satisfy 0 < start < stop < 1")
    lower = threshold_count(int(total_rows), float(start_fraction))
    upper = threshold_count(int(total_rows), float(stop_fraction))
    if lower > upper:
        raise ValueError("samplewise count range is empty")
    return lower, upper


def prepare_ranked_frame(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in frame.columns:
        raise ValueError(f"target column not found: {target_col}")
    work = frame.copy()
    work["__input_order__"] = np.arange(len(work), dtype=int)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()
    if len(work) < 2:
        raise ValueError("at least two rows with finite target values are required")

    if "__source_index__" not in work.columns:
        work["__source_index__"] = work["__input_order__"]
    source = pd.to_numeric(work["__source_index__"], errors="coerce")
    work["__rank_source__"] = source.fillna(work["__input_order__"]).astype(float)
    if "ID" in work.columns:
        ids = pd.to_numeric(work["ID"], errors="coerce")
        work["__rank_id__"] = ids.fillna(work["__input_order__"]).astype(float)
    else:
        work["__rank_id__"] = work["__input_order__"].astype(float)

    work = work.sort_values(
        [target_col, "__rank_source__", "__rank_id__", "__input_order__"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["target_rank"] = np.arange(1, len(work) + 1, dtype=int)
    grouped = work.groupby(target_col, sort=False, dropna=False)
    work["target_tie_size"] = grouped[target_col].transform("size").astype(int)
    work["tie_position"] = grouped.cumcount().astype(int) + 1
    work["tie_group_start_rank"] = grouped["target_rank"].transform("min").astype(int)
    work["tie_group_end_rank"] = grouped["target_rank"].transform("max").astype(int)
    work["tie_group_id"] = pd.factorize(work[target_col], sort=False)[0] + 1
    work["is_tied_target"] = work["target_tie_size"].gt(1)
    return work.drop(columns=["__input_order__", "__rank_source__", "__rank_id__"])


def _entering_metadata(row: pd.Series, target_col: str) -> dict[str, Any]:
    sample_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
    source_index = pd.to_numeric(pd.Series([row.get("__source_index__")]), errors="coerce").iloc[0]
    return {
        "entering_ID": int(sample_id) if pd.notna(sample_id) else float("nan"),
        "entering_source_index": int(source_index) if pd.notna(source_index) else float("nan"),
        "entering_target_value": finite_float(row.get(target_col)),
        "target_rank": int(row["target_rank"]),
        "target_tie_size": int(row["target_tie_size"]),
        "tie_position": int(row["tie_position"]),
        "tie_group_id": int(row["tie_group_id"]),
        "tie_group_start_rank": int(row["tie_group_start_rank"]),
        "tie_group_end_rank": int(row["tie_group_end_rank"]),
        "is_tied_target": bool(row["is_tied_target"]),
        "is_tie_order_sensitive": bool(row["is_tied_target"]),
    }


def compute_samplewise_dynamic_curve(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    start_fraction: float,
    stop_fraction: float,
    random_state: int,
    sliced_projections: int,
    metadata: dict[str, Any] | None = None,
    policy: str = BASELINE_POLICY,
) -> pd.DataFrame:
    ranked = prepare_ranked_frame(frame, target_col)
    total_rows = int(len(ranked))
    lower, upper = sample_count_bounds(total_rows, start_fraction, stop_fraction)
    first_computed = max(1, lower - 1)
    rows: list[dict[str, Any]] = []
    previous_wx = float("nan")
    previous_wy = float("nan")
    base_metadata = dict(metadata or {})

    for top_size in range(first_computed, upper + 1):
        top_tail = ranked.iloc[:top_size].copy()
        background = ranked.iloc[top_size:].copy()
        metrics = compute_space_shift_metrics(
            top_tail=top_tail,
            background=background,
            feature_columns=feature_columns,
            target_col=target_col,
            metadata={},
            sliced_projections=sliced_projections,
            random_state=random_state,
        )
        wx = finite_float(metrics["Wx"])
        wy = finite_float(metrics["Wy"])
        delta_wx = wx - previous_wx if math.isfinite(previous_wx) else float("nan")
        delta_wy = wy - previous_wy if math.isfinite(previous_wy) else float("nan")
        relative_delta = (
            delta_wx / previous_wx
            if math.isfinite(delta_wx) and math.isfinite(previous_wx) and abs(previous_wx) > 1e-12
            else float("nan")
        )
        entering = ranked.iloc[top_size - 1]
        rows.append(
            {
                **base_metadata,
                "policy": str(policy),
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "target_col": target_col,
                "total_valid_samples": total_rows,
                "top_tail_size": int(top_size),
                "previous_top_tail_size": int(top_size - 1),
                "background_size": int(total_rows - top_size),
                "tail_fraction": float(top_size / total_rows),
                "tail_percent": float((top_size / total_rows) * 100.0),
                "feature_count": int(len(feature_columns)),
                "Wx": wx,
                "Wy": wy,
                "delta_Wx": finite_float(delta_wx),
                "relative_delta_Wx": finite_float(relative_delta),
                "delta_Wy": finite_float(delta_wy),
                "x_distance_method": metrics["x_distance_method"],
                "x_sliced_projections": int(metrics["x_sliced_projections"]),
                "y_distance_method": metrics["y_distance_method"],
                **_entering_metadata(entering, target_col),
            }
        )
        previous_wx = wx
        previous_wy = wy

    result = pd.DataFrame(rows)
    return result[result["top_tail_size"].ge(lower)].reset_index(drop=True)


def detect_samplewise_jumps(
    curve: pd.DataFrame,
    min_abs_delta_wx: float,
    min_rel_delta_wx: float,
    detection_stage: str,
    driver_action: str,
) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    work = curve.sort_values("top_tail_size", kind="stable")
    candidates = work[
        pd.to_numeric(work["delta_Wx"], errors="coerce").ge(float(min_abs_delta_wx))
        & pd.to_numeric(work["relative_delta_Wx"], errors="coerce").ge(float(min_rel_delta_wx))
    ]
    rows: list[dict[str, Any]] = []
    passthrough = [
        "alloy_family",
        "dataset_name",
        "property",
        "target_col",
        "case_key",
        "policy",
        "total_valid_samples",
    ]
    for row in candidates.itertuples(index=False):
        payload = row._asdict()
        upper_size = int(payload["top_tail_size"])
        upper_wx = finite_float(payload["Wx"])
        upper_wy = finite_float(payload["Wy"])
        delta_wx = finite_float(payload["delta_Wx"])
        delta_wy = finite_float(payload["delta_Wy"])
        total_rows = int(
            payload.get("total_valid_samples")
            or (int(payload["top_tail_size"]) + int(payload["background_size"]))
        )
        rows.append(
            {
                **{column: payload.get(column) for column in passthrough if column in payload},
                "detection_stage": str(detection_stage),
                "jump_status": "removable" if str(driver_action) == "drop" else "residual_report_only",
                "driver_action": str(driver_action),
                "jump_reason": "samplewise_wx_delta_threshold",
                "lower_top_tail_size": int(upper_size - 1),
                "upper_top_tail_size": upper_size,
                "lower_background_size": int(payload["background_size"] + 1),
                "upper_background_size": int(payload["background_size"]),
                "lower_tail_fraction": float((upper_size - 1) / total_rows),
                "upper_tail_fraction": float(payload["tail_fraction"]),
                "lower_Wx": finite_float(upper_wx - delta_wx),
                "upper_Wx": upper_wx,
                "delta_Wx": delta_wx,
                "relative_delta_Wx": finite_float(payload["relative_delta_Wx"]),
                "lower_Wy": finite_float(upper_wy - delta_wy),
                "upper_Wy": upper_wy,
                "delta_Wy": delta_wy,
                "ID": payload.get("entering_ID"),
                "entering_source_index": payload.get("entering_source_index"),
                "entering_target_value": payload.get("entering_target_value"),
                "target_rank": payload.get("target_rank"),
                "target_tie_size": payload.get("target_tie_size"),
                "tie_position": payload.get("tie_position"),
                "tie_group_id": payload.get("tie_group_id"),
                "tie_group_start_rank": payload.get("tie_group_start_rank"),
                "tie_group_end_rank": payload.get("tie_group_end_rank"),
                "is_tied_target": bool(payload.get("is_tied_target", False)),
                "is_tie_order_sensitive": bool(payload.get("is_tie_order_sensitive", False)),
            }
        )
    return pd.DataFrame(rows)


def summarize_tie_group_contributions(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in [
            "alloy_family",
            "dataset_name",
            "property",
            "case_key",
            "policy",
            "tie_group_start_rank",
            "tie_group_end_rank",
            "entering_target_value",
            "target_tie_size",
        ]
        if column in curve.columns
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in curve.groupby(group_columns, sort=False, dropna=False):
        values = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        ordered = group.sort_values("top_tail_size", kind="stable")
        group_start = int(values["tie_group_start_rank"])
        group_end = int(values["tie_group_end_rank"])
        visible_start = int(ordered["top_tail_size"].min())
        visible_end = int(ordered["top_tail_size"].max())
        ids = pd.to_numeric(ordered.get("entering_ID"), errors="coerce").dropna().astype(int).tolist() if "entering_ID" in ordered else []
        rows.append(
            {
                **values,
                "visible_group_start_size": visible_start,
                "visible_group_end_size": visible_end,
                "visible_sample_count": int(len(ordered)),
                "tie_group_is_complete": bool(visible_start <= group_start and visible_end >= group_end),
                "entering_IDs": ";".join(str(value) for value in ids),
                "delta_Wx_group": finite_float(pd.to_numeric(ordered["delta_Wx"], errors="coerce").sum()),
                "delta_Wy_group": finite_float(pd.to_numeric(ordered["delta_Wy"], errors="coerce").sum()),
                "max_abs_delta_Wx_within_group": finite_float(pd.to_numeric(ordered["delta_Wx"], errors="coerce").abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _driver_ids(jumps: pd.DataFrame) -> list[int]:
    if jumps.empty or "ID" not in jumps.columns:
        return []
    return sorted(pd.to_numeric(jumps["ID"], errors="coerce").dropna().astype(int).unique().tolist())


def _removed_sample_rows(
    frame: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    ids = _driver_ids(baseline_jumps)
    if not ids or "ID" not in frame.columns:
        return pd.DataFrame()
    source = frame[pd.to_numeric(frame["ID"], errors="coerce").isin(ids)].copy()
    jump_columns = [
        column
        for column in [
            "ID",
            "lower_top_tail_size",
            "upper_top_tail_size",
            "delta_Wx",
            "relative_delta_Wx",
            "delta_Wy",
            "entering_target_value",
            "target_rank",
            "target_tie_size",
            "tie_position",
            "is_tie_order_sensitive",
            "jump_reason",
        ]
        if column in baseline_jumps.columns
    ]
    jump_info = baseline_jumps[jump_columns].copy()
    jump_info["ID"] = pd.to_numeric(jump_info["ID"], errors="coerce").astype("Int64")
    source["ID"] = pd.to_numeric(source["ID"], errors="coerce").astype("Int64")
    result = source.merge(jump_info, on="ID", how="left", suffixes=("", "_jump"))
    for key, value in metadata.items():
        result[key] = value
    result["removal_policy"] = DROP_POLICY
    return result


def run_samplewise_sensitivity_task(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    metadata: dict[str, Any],
    start_fraction: float,
    stop_fraction: float,
    min_abs_delta_wx: float,
    min_rel_delta_wx: float,
    random_state: int,
    sliced_projections: int,
) -> dict[str, pd.DataFrame]:
    baseline = compute_samplewise_dynamic_curve(
        frame=frame,
        target_col=target_col,
        feature_columns=feature_columns,
        start_fraction=start_fraction,
        stop_fraction=stop_fraction,
        random_state=random_state,
        sliced_projections=sliced_projections,
        metadata=metadata,
        policy=BASELINE_POLICY,
    )
    baseline_jumps = detect_samplewise_jumps(
        baseline,
        min_abs_delta_wx=min_abs_delta_wx,
        min_rel_delta_wx=min_rel_delta_wx,
        detection_stage=BASELINE_POLICY,
        driver_action="drop",
    )
    ids = _driver_ids(baseline_jumps)
    if ids and "ID" in frame.columns:
        filtered = frame[~pd.to_numeric(frame["ID"], errors="coerce").isin(ids)].reset_index(drop=True)
        after = compute_samplewise_dynamic_curve(
            frame=filtered,
            target_col=target_col,
            feature_columns=feature_columns,
            start_fraction=start_fraction,
            stop_fraction=stop_fraction,
            random_state=random_state,
            sliced_projections=sliced_projections,
            metadata=metadata,
            policy=DROP_POLICY,
        )
    else:
        after = baseline.copy()
        after["policy"] = DROP_POLICY
        after["policy_label"] = POLICY_LABELS[DROP_POLICY]
    after["removed_sample_count"] = int(len(ids))
    baseline["removed_sample_count"] = 0
    residual_jumps = detect_samplewise_jumps(
        after,
        min_abs_delta_wx=min_abs_delta_wx,
        min_rel_delta_wx=min_rel_delta_wx,
        detection_stage="post_drop",
        driver_action="report_only",
    )
    curves = pd.concat([baseline, after], ignore_index=True)
    tie_groups = summarize_tie_group_contributions(curves)
    removed_samples = _removed_sample_rows(frame, baseline_jumps, metadata)
    return {
        "curves": curves,
        "baseline_jumps": baseline_jumps,
        "residual_jumps": residual_jumps,
        "removed_samples": removed_samples,
        "tie_groups": tie_groups,
    }


def _case_title(group: pd.DataFrame) -> str:
    row = group.iloc[0]
    return f"{row.get('alloy_family', '')} / {row.get('property', '')}"


def _case_filename(group: pd.DataFrame) -> str:
    if "case_key" in group.columns and pd.notna(group.iloc[0].get("case_key")):
        return str(group.iloc[0]["case_key"])
    row = group.iloc[0]
    return case_key(row.get("alloy_family", ""), row.get("dataset_name", ""), row.get("property", ""))


def _sample_count_ticks(minimum: int, maximum: int, count: int = 10) -> np.ndarray:
    if maximum <= minimum:
        return np.asarray([minimum], dtype=int)
    raw = np.linspace(minimum, maximum, num=min(count, maximum - minimum + 1))
    ticks = np.unique(np.rint(raw).astype(int))
    return ticks


def _jump_sizes(jumps: pd.DataFrame, current_key: str) -> np.ndarray:
    if jumps.empty or "case_key" not in jumps.columns:
        return np.asarray([], dtype=int)
    values = pd.to_numeric(
        jumps.loc[jumps["case_key"].eq(current_key), "upper_top_tail_size"],
        errors="coerce",
    )
    return values.dropna().astype(int).to_numpy()


def _scatter_jump_points(
    ax: plt.Axes,
    work: pd.DataFrame,
    sizes: np.ndarray,
    value_column: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    if sizes.size == 0:
        return
    points = work[work["top_tail_size"].isin(sizes)]
    if points.empty:
        return
    ax.scatter(
        points["top_tail_size"],
        points[value_column],
        s=34,
        facecolors="none" if marker == "o" else color,
        edgecolors=color,
        marker=marker,
        linewidths=1.2,
        zorder=6,
        label=label,
    )


def low_value_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    minimum = float(np.min(finite))
    q05, median, q90 = [float(value) for value in np.quantile(finite, [0.05, 0.50, 0.90])]
    robust_upper = median + (1.5 * max(median - q05, 0.0))
    upper = min(q90, robust_upper) if robust_upper > minimum else q90
    if not math.isfinite(upper) or upper <= minimum:
        upper = minimum + max(abs(minimum) * 0.15, 0.1)
    padding = max((upper - minimum) * 0.08, 0.01)
    return max(0.0, minimum - padding), upper + padding


def plot_space_shift_by_task(
    curves: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    residual_jumps: pd.DataFrame,
    output_root: Path,
    formats: list[str],
    dpi: int,
) -> None:
    if curves.empty:
        return
    figure_dir = output_root / "figures" / "space_shift_by_task"
    clear_existing_figures(figure_dir, formats)
    for current_key, group in curves.groupby("case_key", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex="col", gridspec_kw={"hspace": 0.10})
        wy_limits = low_value_limits(group["Wy"])
        wx_limits = low_value_limits(group["Wx"])
        for policy in [BASELINE_POLICY, DROP_POLICY]:
            work = group[group["policy"].eq(policy)].sort_values("top_tail_size", kind="stable")
            if work.empty:
                continue
            for row_index, (column, limits) in enumerate([("Wy", wy_limits), ("Wx", wx_limits)]):
                axes[row_index, 0].plot(
                    work["top_tail_size"],
                    work[column],
                    color=POLICY_COLORS[policy],
                    linewidth=1.15,
                    label=POLICY_LABELS[policy],
                )
                axes[row_index, 1].plot(
                    work["top_tail_size"],
                    clip_values_to_limits(work[column], *limits),
                    color=POLICY_COLORS[policy],
                    linewidth=1.15,
                    label=POLICY_LABELS[policy],
                )
        baseline = group[group["policy"].eq(BASELINE_POLICY)]
        after = group[group["policy"].eq(DROP_POLICY)]
        baseline_sizes = _jump_sizes(baseline_jumps, str(current_key))
        residual_sizes = _jump_sizes(residual_jumps, str(current_key))
        for row_index, column in enumerate(["Wy", "Wx"]):
            for column_index in range(2):
                ax = axes[row_index, column_index]
                _scatter_jump_points(ax, baseline, baseline_sizes, column, "#d62728", "o", "baseline jump driver")
                _scatter_jump_points(ax, after, residual_sizes, column, "#f28e2b", "^", "post-drop residual jump")
        axes[0, 1].set_ylim(*wy_limits)
        axes[1, 1].set_ylim(*wx_limits)
        for ax in axes.flat:
            ax.axhline(0.0, color="#bdbdbd", linewidth=0.7, zorder=0)
            ax.grid(axis="y", color="#e6e6e6")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0, 0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper left")
        axes[0, 0].set_title("Full range", fontsize=10)
        axes[0, 1].set_title("Low-value zoom", fontsize=10)
        axes[0, 0].set_ylabel("Y-space Wasserstein distance")
        axes[1, 0].set_ylabel("X-space sliced Wasserstein distance")
        min_count = int(group["top_tail_size"].min())
        max_count = int(group["top_tail_size"].max())
        ticks = _sample_count_ticks(min_count, max_count)
        for ax in axes.flat:
            ax.set_xlim(min_count, max_count)
        for ax in axes[1, :]:
            ax.set_xticks(ticks)
            ax.set_xlabel("Top-tail sample count")
        fig.suptitle(_case_title(group), fontsize=12, fontweight="bold")
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.91, wspace=0.18, hspace=0.10)
        save_figure(fig, figure_dir / _case_filename(group), formats, dpi)


def _zoom_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return -1.0, 1.0
    lower, upper = np.quantile(finite, [0.05, 0.95])
    lower = min(float(lower), 0.0)
    upper = max(float(upper), 0.0)
    if upper - lower <= 1e-12:
        pad = max(abs(upper) * 0.15, 0.1)
    else:
        pad = (upper - lower) * 0.12
    return lower - pad, upper + pad


def clip_values_to_limits(values: pd.Series, lower: float, upper: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(numeric) & (numeric >= lower) & (numeric <= upper), numeric, np.nan)


def plot_per_sample_step_change_by_task(
    curves: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    residual_jumps: pd.DataFrame,
    output_root: Path,
    formats: list[str],
    dpi: int,
) -> None:
    if curves.empty:
        return
    figure_dir = output_root / "figures" / "per_sample_step_change_by_task"
    clear_existing_figures(figure_dir, formats)
    for current_key, group in curves.groupby("case_key", sort=True):
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True, gridspec_kw={"hspace": 0.10})
        zoom_lower, zoom_upper = _zoom_limits(group["delta_Wx"])
        for policy in [BASELINE_POLICY, DROP_POLICY]:
            work = group[group["policy"].eq(policy)].sort_values("top_tail_size", kind="stable")
            if work.empty:
                continue
            color = POLICY_COLORS[policy]
            for ax, column in [(axes[0], "delta_Wy"), (axes[1], "delta_Wx")]:
                ax.plot(
                    work["top_tail_size"],
                    work[column],
                    color=color,
                    linewidth=0.9,
                    alpha=0.9,
                    label=POLICY_LABELS[policy],
                )
                tied = work[work["is_tied_target"].astype(bool)]
                if not tied.empty:
                    ax.scatter(
                        tied["top_tail_size"],
                        tied[column],
                        s=10,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.45,
                        alpha=0.7,
                        zorder=4,
                    )
            zoomed_values = clip_values_to_limits(work["delta_Wx"], zoom_lower, zoom_upper)
            axes[2].plot(
                work["top_tail_size"],
                zoomed_values,
                color=color,
                linewidth=0.9,
                alpha=0.9,
                label=POLICY_LABELS[policy],
            )
            tied_mask = work["is_tied_target"].astype(bool).to_numpy() & np.isfinite(zoomed_values)
            if tied_mask.any():
                axes[2].scatter(
                    work.loc[tied_mask, "top_tail_size"],
                    zoomed_values[tied_mask],
                    s=10,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.45,
                    alpha=0.7,
                    zorder=4,
                )
        baseline = group[group["policy"].eq(BASELINE_POLICY)]
        after = group[group["policy"].eq(DROP_POLICY)]
        baseline_sizes = _jump_sizes(baseline_jumps, str(current_key))
        residual_sizes = _jump_sizes(residual_jumps, str(current_key))
        for ax, column in [(axes[0], "delta_Wy"), (axes[1], "delta_Wx"), (axes[2], "delta_Wx")]:
            _scatter_jump_points(ax, baseline, baseline_sizes, column, "#d62728", "o", "baseline jump driver")
            _scatter_jump_points(ax, after, residual_sizes, column, "#f28e2b", "^", "post-drop residual jump")
            ax.axhline(0.0, color="#8c8c8c", linewidth=0.8)
            ax.grid(axis="y", color="#e6e6e6")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Observed Delta Wy")
        axes[1].set_ylabel("Observed Delta Wx\n(full range)")
        axes[2].set_ylabel("Observed Delta Wx\n(zoomed range)")
        axes[2].set_ylim(zoom_lower, zoom_upper)
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper left")
        min_count = int(group["top_tail_size"].min())
        max_count = int(group["top_tail_size"].max())
        axes[2].set_xlim(min_count, max_count)
        axes[2].set_xticks(_sample_count_ticks(min_count, max_count))
        axes[2].set_xlabel("Top-tail sample count")
        fig.suptitle(f"{_case_title(group)}: one-sample observed step change", fontsize=12, fontweight="bold")
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.08, top=0.94, hspace=0.10)
        save_figure(fig, figure_dir / _case_filename(group), formats, dpi)


def samplewise_settings_from_config(config: dict[str, Any]) -> dict[str, float]:
    section = dict(config.get("samplewise_sensitivity") or {})
    jump_rules = dict(config.get("jump_detection") or {})
    defaults = dict(config.get("defaults") or {})
    return {
        "start_fraction": float(section.get("start", DEFAULT_START_FRACTION)),
        "stop_fraction": float(section.get("stop", DEFAULT_STOP_FRACTION)),
        "step_samples": float(section.get("step_samples", 1)),
        "min_abs_delta_wx": float(
            section.get("min_abs_delta_wx", jump_rules.get("min_abs_delta_wx", 0.15))
        ),
        "min_rel_delta_wx": float(
            section.get("min_rel_delta_wx", jump_rules.get("min_rel_delta_wx", 0.30))
        ),
        "sliced_projections": float(
            section.get("sliced_projections", defaults.get("sliced_projections", 128))
        ),
        "removal_passes": float(section.get("removal_passes", 1)),
    }


def discover_task_specs(
    splits_root: Path,
    include_matbench: bool,
    case_contains: str | None,
) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    needle = str(case_contains or "").lower()
    for manifest_path in discover_hybrid_manifests(splits_root, include_matbench=include_matbench):
        payload = build_cluster_frames_from_hybrid_manifest(manifest_path)
        spec = task_spec_from_payload(payload)
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        if needle and needle not in current_key.lower():
            continue
        specs.append(spec)
    return specs


def build_legacy_coarse_validation(
    curves: pd.DataFrame,
    legacy_csv: Path,
) -> pd.DataFrame:
    columns = [
        "case_key",
        "legacy_threshold_pct",
        "top_tail_size",
        "legacy_Wx",
        "samplewise_Wx",
        "abs_Wx_difference",
        "legacy_Wy",
        "samplewise_Wy",
        "abs_Wy_difference",
        "legacy_sliced_projections",
        "samplewise_sliced_projections",
        "same_projection_count",
        "is_tied_boundary",
        "metric_match",
        "validation_note",
    ]
    if not legacy_csv.exists() or curves.empty:
        return pd.DataFrame(columns=columns)
    legacy = read_csv(legacy_csv)
    required = {"case_key", "threshold_pct", "top_tail_size", "Wx", "Wy"}
    if not required.issubset(legacy.columns):
        return pd.DataFrame(columns=columns)
    baseline = curves[curves["policy"].eq(BASELINE_POLICY)].copy()
    legacy = legacy[legacy["case_key"].isin(baseline["case_key"].dropna().unique())].copy()
    if legacy.empty:
        return pd.DataFrame(columns=columns)
    if "x_sliced_projections" not in legacy.columns:
        legacy["x_sliced_projections"] = float("nan")
    sample_columns = [
        "case_key",
        "top_tail_size",
        "Wx",
        "Wy",
        "is_tied_target",
        "tie_group_start_rank",
        "tie_group_end_rank",
        "x_sliced_projections",
    ]
    merged = legacy[[*list(required), "x_sliced_projections"]].merge(
        baseline[sample_columns],
        on=["case_key", "top_tail_size"],
        how="left",
        suffixes=("_legacy", "_samplewise"),
    )
    merged["abs_Wx_difference"] = (merged["Wx_legacy"] - merged["Wx_samplewise"]).abs()
    merged["abs_Wy_difference"] = (merged["Wy_legacy"] - merged["Wy_samplewise"]).abs()
    merged["same_projection_count"] = merged["x_sliced_projections_legacy"].eq(
        merged["x_sliced_projections_samplewise"]
    )
    merged["metric_match"] = (
        merged["same_projection_count"]
        & merged["abs_Wx_difference"].le(1e-10)
        & merged["abs_Wy_difference"].le(1e-10)
    )
    merged["is_tied_boundary"] = merged["is_tied_target"].eq(True)
    merged["validation_note"] = np.where(
        ~merged["same_projection_count"],
        "sliced projection count differs",
        np.where(
            merged["metric_match"],
            "exact metric match",
            np.where(
                merged["is_tied_boundary"] & merged["threshold_pct"].eq(20),
                "legacy 20% hybrid split selected a different equal-target tie member",
                "metric mismatch requires review",
            ),
        ),
    )
    return merged.rename(
        columns={
            "threshold_pct": "legacy_threshold_pct",
            "Wx_legacy": "legacy_Wx",
            "Wx_samplewise": "samplewise_Wx",
            "Wy_legacy": "legacy_Wy",
            "Wy_samplewise": "samplewise_Wy",
            "x_sliced_projections_legacy": "legacy_sliced_projections",
            "x_sliced_projections_samplewise": "samplewise_sliced_projections",
        }
    )[columns]


def build_analysis_manifest(
    specs: list[TaskSpec],
    curves: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    residual_jumps: pd.DataFrame,
    runtimes: dict[str, float],
    settings: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec_by_key = {
        case_key(spec.alloy_family, spec.dataset_name, spec.property_name): spec
        for spec in specs
    }
    for current_key, group in curves.groupby("case_key", sort=True):
        spec = spec_by_key[str(current_key)]
        baseline = group[group["policy"].eq(BASELINE_POLICY)]
        after = group[group["policy"].eq(DROP_POLICY)]
        current_jumps = baseline_jumps[baseline_jumps.get("case_key", pd.Series(dtype=str)).eq(current_key)] if not baseline_jumps.empty else pd.DataFrame()
        current_residuals = residual_jumps[residual_jumps.get("case_key", pd.Series(dtype=str)).eq(current_key)] if not residual_jumps.empty else pd.DataFrame()
        rows.append(
            {
                "alloy_family": spec.alloy_family,
                "dataset_name": spec.dataset_name,
                "property": spec.property_name,
                "case_key": current_key,
                "target_col": spec.target_col,
                "data_file": str(spec.data_file),
                "manifest_path": str(spec.manifest_path or ""),
                "baseline_valid_samples": int(baseline["total_valid_samples"].iloc[0]),
                "baseline_min_top_tail_size": int(baseline["top_tail_size"].min()),
                "baseline_max_top_tail_size": int(baseline["top_tail_size"].max()),
                "baseline_point_count": int(len(baseline)),
                "post_drop_valid_samples": int(after["total_valid_samples"].iloc[0]),
                "post_drop_min_top_tail_size": int(after["top_tail_size"].min()),
                "post_drop_max_top_tail_size": int(after["top_tail_size"].max()),
                "post_drop_point_count": int(len(after)),
                "feature_count": int(baseline["feature_count"].iloc[0]),
                "baseline_jump_count": int(len(current_jumps)),
                "removed_unique_ID_count": int(pd.to_numeric(current_jumps.get("ID"), errors="coerce").nunique()) if not current_jumps.empty else 0,
                "tie_sensitive_baseline_jump_count": int(current_jumps.get("is_tie_order_sensitive", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not current_jumps.empty else 0,
                "post_drop_residual_jump_count": int(len(current_residuals)),
                "runtime_seconds": float(runtimes.get(str(current_key), float("nan"))),
                "start_fraction": float(settings["start_fraction"]),
                "stop_fraction": float(settings["stop_fraction"]),
                "step_samples": int(settings["step_samples"]),
                "min_abs_delta_wx": float(settings["min_abs_delta_wx"]),
                "min_rel_delta_wx": float(settings["min_rel_delta_wx"]),
                "sliced_projections": int(settings["sliced_projections"]),
                "standardization_mode": "dynamic_background",
                "removal_passes": int(settings["removal_passes"]),
            }
        )
    return pd.DataFrame(rows)


def write_samplewise_report(
    output_root: Path,
    manifest: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    residual_jumps: pd.DataFrame,
) -> None:
    lines = [
        "# X-space Samplewise Top-tail Sensitivity Report",
        "",
        "## Method",
        "",
        "Samples are ranked by descending target value. The top-tail size increases by one sample at each step from 5% to 95% of the valid task rows. At every step, the X-space imputer and scaler and the Y-space standardization are refitted on the current background. Therefore, observed step changes combine the entering sample effect with the change in background scaling and are not pure causal influence estimates.",
        "",
        "A baseline jump is detected when delta Wx is at least 0.15 and its relative increase is at least 30%. All baseline jump entrants in the plotted 5%-95% range are removed once in memory, after which the complete curve is recomputed. Residual post-drop jumps are reported but not removed again.",
        "",
        "## Task Summary",
        "",
        "| Task | Baseline n | Count range | Points | Removed IDs | Residual jumps | Tie-sensitive jumps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest.itertuples():
        lines.append(
            f"| {row.case_key} | {row.baseline_valid_samples} | {row.baseline_min_top_tail_size}-{row.baseline_max_top_tail_size} | {row.baseline_point_count} | {row.removed_unique_ID_count} | {row.post_drop_residual_jump_count} | {row.tie_sensitive_baseline_jump_count} |"
        )
    lines.extend(["", "## Largest Baseline Jumps", ""])
    if baseline_jumps.empty:
        lines.append("No baseline jumps met the configured detection rule.")
    else:
        lines.extend(
            [
                "| Task | Step | ID | Delta Wx | Relative delta | Tied target |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        ordered = baseline_jumps.sort_values("delta_Wx", ascending=False, kind="stable").head(30)
        for row in ordered.itertuples():
            lines.append(
                f"| {row.case_key} | {int(row.lower_top_tail_size)}->{int(row.upper_top_tail_size)} | {int(row.ID)} | {row.delta_Wx:.6g} | {row.relative_delta_Wx:.3g} | {bool(row.is_tie_order_sensitive)} |"
            )
    lines.extend(["", "## Residual Jumps", ""])
    lines.append(
        f"The post-drop curves contain {len(residual_jumps)} residual jumps under the same rule. These are diagnostic only and do not trigger a second removal pass."
    )
    report_path = output_root / "report" / "samplewise_sensitivity_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samplewise_outputs(
    output_root: Path,
    curves: pd.DataFrame,
    baseline_jumps: pd.DataFrame,
    residual_jumps: pd.DataFrame,
    removed_samples: pd.DataFrame,
    tie_groups: pd.DataFrame,
    manifest: pd.DataFrame,
    legacy_validation: pd.DataFrame,
    resolved_config: dict[str, Any],
) -> None:
    csv_dir = output_root / "csv"
    write_csv(curves, csv_dir / "samplewise_space_shift.csv")
    write_csv(baseline_jumps, csv_dir / "samplewise_jump_drivers.csv")
    write_csv(residual_jumps, csv_dir / "samplewise_residual_jumps.csv")
    write_csv(removed_samples, csv_dir / "removed_jump_driver_samples.csv")
    write_csv(tie_groups, csv_dir / "tie_group_contributions.csv")
    write_csv(manifest, csv_dir / "analysis_manifest.csv")
    write_csv(legacy_validation, csv_dir / "legacy_coarse_grid_validation.csv")
    write_json(resolved_config, csv_dir / "analysis_config_resolved.json")
    write_samplewise_report(output_root, manifest, baseline_jumps, residual_jumps)


def run_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    config_path = Path(args.config).resolve() if args.config else None
    config = load_analysis_config(config_path)
    settings = samplewise_settings_from_config(config)
    if int(settings["step_samples"]) != 1:
        raise ValueError("samplewise_sensitivity.step_samples must be 1")
    if int(settings["removal_passes"]) != 1:
        raise ValueError("samplewise_sensitivity.removal_passes must be 1")
    projections = int(args.sliced_projections or settings["sliced_projections"])
    settings["sliced_projections"] = float(projections)
    output_root = Path(args.output_root)
    specs = discover_task_specs(
        Path(args.splits_root),
        include_matbench=not bool(args.exclude_matbench),
        case_contains=args.case_contains,
    )
    if not specs:
        raise RuntimeError("No hybrid top-tail tasks matched the samplewise analysis filters.")

    curve_parts: list[pd.DataFrame] = []
    jump_parts: list[pd.DataFrame] = []
    residual_parts: list[pd.DataFrame] = []
    removed_parts: list[pd.DataFrame] = []
    tie_parts: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    for task_index, spec in enumerate(specs, start=1):
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        started = time.perf_counter()
        frame = load_task_dataframe(spec)
        if "__source_index__" not in frame.columns:
            frame["__source_index__"] = np.arange(len(frame), dtype=int)
        feature_columns = fallback_feature_columns(frame, spec.target_col, spec.feature_columns)
        metadata = {
            "alloy_family": spec.alloy_family,
            "dataset_name": spec.dataset_name,
            "property": spec.property_name,
            "case_key": current_key,
            "split_id": spec.split_id,
            "manifest_path": str(spec.manifest_path or ""),
        }
        result = run_samplewise_sensitivity_task(
            frame=frame,
            target_col=spec.target_col,
            feature_columns=feature_columns,
            metadata=metadata,
            start_fraction=float(settings["start_fraction"]),
            stop_fraction=float(settings["stop_fraction"]),
            min_abs_delta_wx=float(settings["min_abs_delta_wx"]),
            min_rel_delta_wx=float(settings["min_rel_delta_wx"]),
            random_state=int(args.random_state),
            sliced_projections=projections,
        )
        curve_parts.append(result["curves"])
        if not result["baseline_jumps"].empty:
            jump_parts.append(result["baseline_jumps"])
        if not result["residual_jumps"].empty:
            residual_parts.append(result["residual_jumps"])
        if not result["removed_samples"].empty:
            removed_parts.append(result["removed_samples"])
        if not result["tie_groups"].empty:
            tie_parts.append(result["tie_groups"])
        runtimes[current_key] = float(time.perf_counter() - started)
        print(
            f"[{task_index}/{len(specs)}] {current_key}: "
            f"baseline_points={len(result['curves'][result['curves']['policy'].eq(BASELINE_POLICY)])}, "
            f"drivers={len(_driver_ids(result['baseline_jumps']))}, "
            f"seconds={runtimes[current_key]:.1f}",
            flush=True,
        )

    curves = pd.concat(curve_parts, ignore_index=True)
    baseline_jumps = pd.concat(jump_parts, ignore_index=True) if jump_parts else pd.DataFrame()
    residual_jumps = pd.concat(residual_parts, ignore_index=True) if residual_parts else pd.DataFrame()
    removed_samples = pd.concat(removed_parts, ignore_index=True, sort=False) if removed_parts else pd.DataFrame()
    tie_groups = pd.concat(tie_parts, ignore_index=True) if tie_parts else pd.DataFrame()
    manifest = build_analysis_manifest(specs, curves, baseline_jumps, residual_jumps, runtimes, settings)
    legacy_validation = build_legacy_coarse_validation(
        curves,
        DEFAULT_OUTPUT_PARENT / "xspace" / "csv" / "threshold_sweep_space_shift.csv",
    )
    resolved_config = {
        "config_path": str(config_path or ""),
        "output_root": str(output_root),
        "task_count": int(len(specs)),
        "include_matbench": not bool(args.exclude_matbench),
        "random_state": int(args.random_state),
        "standardization_mode": "dynamic_background",
        "removal_scope": "all detected baseline jumps from 5% through 95%",
        "removal_strategy": "single baseline pass; post-drop residuals report only",
        "tie_breaker": "target descending, source index ascending, ID ascending",
        "settings": settings,
    }
    write_samplewise_outputs(
        output_root,
        curves,
        baseline_jumps,
        residual_jumps,
        removed_samples,
        tie_groups,
        manifest,
        legacy_validation,
        resolved_config,
    )
    if not args.no_figures:
        plot_space_shift_by_task(curves, baseline_jumps, residual_jumps, output_root, list(args.formats), int(args.dpi))
        plot_per_sample_step_change_by_task(curves, baseline_jumps, residual_jumps, output_root, list(args.formats), int(args.dpi))
    return {
        "curves": curves,
        "baseline_jumps": baseline_jumps,
        "residual_jumps": residual_jumps,
        "removed_samples": removed_samples,
        "tie_groups": tie_groups,
        "manifest": manifest,
        "legacy_validation": legacy_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sample-by-sample dynamic Wx/Wy top-tail sensitivity curves.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--exclude-matbench", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    baseline_rows = int(result["curves"]["policy"].eq(BASELINE_POLICY).sum())
    print(f"Tasks: {result['manifest']['case_key'].nunique()}")
    print(f"Baseline samplewise rows: {baseline_rows}")
    print(f"Baseline task-specific jump drivers: {len(result['baseline_jumps'])}")
    print(f"Post-drop residual jumps: {len(result['residual_jumps'])}")
    print(f"Output root: {Path(args.output_root)}")


if __name__ == "__main__":
    main()
