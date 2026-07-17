from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
    add_adjacent_shift_columns,
    build_cluster_frames_from_hybrid_manifest,
    case_key,
    clear_existing_figures,
    compute_space_shift_curve,
    discover_hybrid_manifests,
    fine_comparison_settings_from_config,
    fine_comparison_thresholds_from_config,
    load_analysis_config,
    load_task_dataframe,
    read_csv,
    save_figure,
    task_spec_from_payload,
    write_csv,
)


DEFAULT_INPUT_CSV = (
    DEFAULT_OUTPUT_PARENT
    / "sensitivity_comparison_1pct"
    / "csv"
    / "jump_driver_removal_wx_1pct.csv"
)
DEFAULT_JUMP_CSV = (
    DEFAULT_OUTPUT_PARENT
    / "sensitivity_comparison_1pct"
    / "csv"
    / "detected_wx_jumps_1pct.csv"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / "sensitivity_comparison_id350"
FOCUS_CASES = {
    "HEA__hea__UTS_MPa": "HEA / UTS(MPa)",
    "HEA__hea__YS_MPa": "HEA / YS(MPa)",
}
POLICY_ORDER = ["none", "drop_id_350", "drop_auto_detected_jump_drivers"]
POLICY_LABELS = {
    "none": "baseline",
    "drop_id_350": "drop ID 350 only",
    "drop_auto_detected_jump_drivers": "drop all five 1% drivers",
}
POLICY_COLORS = {
    "none": "#222222",
    "drop_id_350": "#4c78a8",
    "drop_auto_detected_jump_drivers": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HEA baseline, ID-350-only, and all-driver Wx curves.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--jump-csv", default=str(DEFAULT_JUMP_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def compute_drop_id350_curves(
    config: dict,
    splits_root: Path,
    random_state: int,
    sliced_projections: int,
) -> pd.DataFrame:
    thresholds = fine_comparison_thresholds_from_config(config)
    parts: list[pd.DataFrame] = []
    for manifest_path in discover_hybrid_manifests(splits_root, include_matbench=True):
        payload = build_cluster_frames_from_hybrid_manifest(manifest_path)
        spec = task_spec_from_payload(payload)
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        if current_key not in FOCUS_CASES:
            continue
        frame = load_task_dataframe(spec)
        filtered = frame[pd.to_numeric(frame["ID"], errors="coerce").ne(350)].reset_index(drop=True)
        curve = compute_space_shift_curve(
            spec=spec,
            frame=filtered,
            thresholds=thresholds,
            random_state=random_state,
            sliced_projections=sliced_projections,
            mode="drop_id_350",
        )
        curve["comparison_policy"] = "drop_id_350"
        curve["comparison_label"] = POLICY_LABELS["drop_id_350"]
        curve["policy"] = "drop_id_350"
        curve["x_threshold_pct"] = curve["threshold_pct"]
        parts.append(curve)
    if not parts:
        raise RuntimeError("No HEA UTS/YS tasks were found for the ID 350 comparison.")
    return pd.concat(parts, ignore_index=True)


def plot_case(
    case_data: pd.DataFrame,
    jumps: pd.DataFrame,
    output_root: Path,
    formats: list[str],
    dpi: int,
) -> None:
    current_key = str(case_data["case_key"].iloc[0])
    fig, (ax_full, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.08},
    )
    finite_low: list[float] = []
    payloads: list[tuple[str, np.ndarray, np.ndarray]] = []
    for policy in POLICY_ORDER:
        work = case_data[case_data["comparison_policy"].eq(policy)].sort_values("threshold_pct")
        if work.empty:
            continue
        x = work["threshold_pct"].to_numpy(dtype=float)
        y = work["Wx"].to_numpy(dtype=float)
        finite_low.extend([float(value) for value in y if math.isfinite(float(value)) and value <= 3.0])
        payloads.append((policy, x, y))

    zoom_max = max(1.0, (max(finite_low) if finite_low else 1.0) * 1.18)
    zoom_max = min(max(zoom_max, 1.15), 3.0)
    clipped = False
    for policy, x, y in payloads:
        color = POLICY_COLORS[policy]
        label = POLICY_LABELS[policy]
        ax_full.plot(x, y, marker="o", markersize=2.4, linewidth=1.15, color=color, label=label)
        y_zoom = np.where(np.isfinite(y) & (y <= zoom_max), y, np.nan)
        ax_zoom.plot(x, y_zoom, marker="o", markersize=2.4, linewidth=1.2, color=color)
        major = np.isclose(np.mod(x, 5), 0.0)
        for ax, plotted_y in ((ax_full, y), (ax_zoom, y_zoom)):
            visible = major & np.isfinite(plotted_y)
            ax.scatter(
                x[visible],
                plotted_y[visible],
                s=24,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
        above = np.isfinite(y) & (y > zoom_max)
        if above.any():
            clipped = True
            ax_zoom.scatter(
                x[above],
                [zoom_max * 0.985] * int(above.sum()),
                marker="^",
                s=28,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
            )

    removable = jumps[
        jumps["case_key"].eq(current_key) & jumps["jump_status"].eq("removable")
    ]
    intervals: list[str] = []
    for jump in removable.itertuples():
        lower = int(jump.lower_threshold_pct)
        upper = int(jump.upper_threshold_pct)
        if lower <= 0:
            continue
        intervals.append(f"{lower}-{upper}%")
        for ax in (ax_full, ax_zoom):
            ax.axvspan(lower, upper, color="#d62728", alpha=0.10)
            ax.axvline(lower, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axvline(upper, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_full.text(
        0.02,
        0.98,
        "Baseline jump interval: " + ", ".join(intervals),
        transform=ax_full.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#555555",
    )
    if clipped:
        ax_zoom.text(
            0.02,
            0.96,
            "Triangles mark values above the zoom range; see full range panel.",
            transform=ax_zoom.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#666666",
        )

    major_ticks = np.arange(5, 96, 5)
    ax_zoom.set_xticks(major_ticks)
    ax_zoom.set_xticklabels([f"{value}%" for value in major_ticks])
    ax_zoom.set_xticks(np.arange(5, 96), minor=True)
    ax_zoom.set_xlim(4.5, 95.5)
    ax_zoom.set_ylim(0.0, zoom_max)
    ax_zoom.set_xlabel("Top-tail threshold (%)")
    ax_full.set_ylabel("Wx full range")
    ax_zoom.set_ylabel("Wx zoomed low range")
    ax_full.set_title(FOCUS_CASES[current_key])
    ax_full.legend(frameon=False, fontsize=8.5)
    for ax in (ax_full, ax_zoom):
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save_figure(fig, output_root / "figures" / current_key, formats, dpi)


def main() -> None:
    args = parse_args()
    config = load_analysis_config(Path(args.config))
    settings = fine_comparison_settings_from_config(config)
    sliced_projections = int(
        args.sliced_projections
        if args.sliced_projections is not None
        else settings["sliced_projections"]
    )
    existing = read_csv(Path(args.input_csv))
    existing = existing[
        existing["case_key"].isin(FOCUS_CASES)
        & existing["comparison_policy"].isin(["none", "drop_auto_detected_jump_drivers"])
    ].copy()
    id350 = compute_drop_id350_curves(
        config,
        Path(args.splits_root),
        int(args.random_state),
        sliced_projections,
    )
    combined = pd.concat([existing, id350], ignore_index=True, sort=False)
    combined = add_adjacent_shift_columns(combined)
    baseline = combined[combined["comparison_policy"].eq("none")][
        ["case_key", "threshold_pct", "Wx"]
    ].rename(columns={"Wx": "baseline_Wx"})
    combined = combined.merge(baseline, on=["case_key", "threshold_pct"], how="left")
    combined["difference_from_baseline"] = combined["Wx"] - combined["baseline_Wx"]

    output_root = Path(args.output_root)
    write_csv(combined, output_root / "csv" / "hea_id350_comparison_1pct.csv")
    jumps = read_csv(Path(args.jump_csv))
    clear_existing_figures(output_root / "figures", args.formats)
    for current_key in FOCUS_CASES:
        plot_case(
            combined[combined["case_key"].eq(current_key)],
            jumps,
            output_root,
            list(args.formats),
            int(args.dpi),
        )
    print(f"Rows: {len(combined)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
