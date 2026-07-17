from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.analysis.analyze_top_tail_loco_proximity import (  # noqa: E402
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_SPLITS_ROOT,
    case_key,
    fallback_feature_columns,
    load_analysis_config,
    load_task_dataframe,
)
from Scripts.analysis.analyze_top_tail_samplewise_space_shift import discover_task_specs  # noqa: E402
from src.data_processing.strength_ood_common import canonical_row_hash, json_ready  # noqa: E402
from src.data_processing.w_jump_dataset_builder import (  # noqa: E402
    SAMPLE_KEY_COLUMN,
    STANDARDIZATION_MODE,
    compute_fixed_distance,
    compute_samplewise_fixed_curve,
    detect_jump_drivers,
    fit_fixed_w_reference,
    prepare_jump_dataset,
)


DEFAULT_CONFIG_PATH = Path("configs") / "analysis" / "w_jump_training_data.yaml"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / "w_jump_training_data_full_detect"
DEFAULT_SETTINGS: dict[str, Any] = {
    "include_matbench": True,
    "test_fraction": 0.20,
    "random_state": 42,
    "target_strata": 20,
    "start_fraction": 0.05,
    "stop_fraction": 0.95,
    "step_samples": 1,
    "min_abs_delta_wx": 0.15,
    "min_rel_delta_wx": 0.30,
    "sliced_projections": 128,
    "standardization_mode": STANDARDIZATION_MODE,
    "tie_policy": "group_leave_one_out",
    "removal_passes": 1,
}


def resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    configured = dict(config.get("defaults") or config.get("w_jump_training_data") or {})
    settings = dict(DEFAULT_SETTINGS)
    settings.update({key: value for key, value in configured.items() if key in settings})
    settings["include_matbench"] = bool(settings["include_matbench"])
    settings["test_fraction"] = float(settings["test_fraction"])
    settings["random_state"] = int(settings["random_state"])
    settings["target_strata"] = int(settings["target_strata"])
    settings["start_fraction"] = float(settings["start_fraction"])
    settings["stop_fraction"] = float(settings["stop_fraction"])
    settings["step_samples"] = int(settings["step_samples"])
    settings["min_abs_delta_wx"] = float(settings["min_abs_delta_wx"])
    settings["min_rel_delta_wx"] = float(settings["min_rel_delta_wx"])
    settings["sliced_projections"] = int(settings["sliced_projections"])
    settings["removal_passes"] = int(settings["removal_passes"])
    settings["standardization_mode"] = str(settings["standardization_mode"])
    settings["tie_policy"] = str(settings["tie_policy"])
    if settings["step_samples"] != 1:
        raise ValueError("step_samples must be 1")
    if settings["removal_passes"] != 1:
        raise ValueError("removal_passes must be 1")
    if settings["standardization_mode"] != STANDARDIZATION_MODE:
        raise ValueError(f"standardization_mode must be {STANDARDIZATION_MODE}")
    if settings["tie_policy"] != "group_leave_one_out":
        raise ValueError("tie_policy must be group_leave_one_out")
    return settings


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_with_columns(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _driver_ids(drivers: pd.DataFrame) -> list[Any]:
    if drivers.empty or "ID" not in drivers.columns:
        return []
    return drivers["ID"].dropna().tolist()


def _curve_pair(
    baseline: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    target_col: str,
    reference: Any,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    baseline_policy: str,
    cleaned_policy: str,
) -> pd.DataFrame:
    baseline_curve = baseline.copy()
    baseline_curve["policy"] = baseline_policy
    cleaned_curve = compute_samplewise_fixed_curve(
        cleaned_frame,
        target_col=target_col,
        reference=reference,
        start_fraction=float(settings["start_fraction"]),
        stop_fraction=float(settings["stop_fraction"]),
        metadata=metadata,
        policy=cleaned_policy,
    )
    return pd.concat([baseline_curve, cleaned_curve], ignore_index=True, sort=False)


def _coverage_table(
    train_baseline: pd.DataFrame,
    train_cleaned: pd.DataFrame,
    test_fixed: pd.DataFrame,
    target_col: str,
    reference: Any,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, train in [
        ("train_baseline", train_baseline),
        ("train_drop_jump_drivers", train_cleaned),
    ]:
        metrics = compute_fixed_distance(train, test_fixed, target_col, reference)
        rows.append(
            {
                **metadata,
                "policy": policy,
                "train_size": int(len(train)),
                "test_size": int(len(test_fixed)),
                "Wx_train_to_test": float(metrics["Wx"]),
                "Wy_train_to_test": float(metrics["Wy"]),
                "reference_hash": reference.reference_hash,
                "standardization_mode": STANDARDIZATION_MODE,
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["policy"].eq("train_baseline")].iloc[0]
    result["delta_Wx_vs_baseline"] = result["Wx_train_to_test"] - float(baseline["Wx_train_to_test"])
    result["delta_Wy_vs_baseline"] = result["Wy_train_to_test"] - float(baseline["Wy_train_to_test"])
    return result


def curves_are_identical(curves: pd.DataFrame, tolerance: float = 1e-12) -> bool:
    policies = list(curves["policy"].dropna().astype(str).unique())
    if len(policies) != 2:
        return False
    left = curves[curves["policy"].astype(str).eq(policies[0])][
        ["top_tail_size", "Wx", "Wy"]
    ].sort_values("top_tail_size", kind="stable")
    right = curves[curves["policy"].astype(str).eq(policies[1])][
        ["top_tail_size", "Wx", "Wy"]
    ].sort_values("top_tail_size", kind="stable")
    if len(left) != len(right):
        return False
    merged = left.merge(right, on="top_tail_size", how="inner", suffixes=("_left", "_right"))
    if len(merged) != len(left):
        return False
    return bool(
        np.allclose(merged["Wx_left"], merged["Wx_right"], atol=tolerance, rtol=0.0)
        and np.allclose(merged["Wy_left"], merged["Wy_right"], atol=tolerance, rtol=0.0)
    )


def _max_delta_summary(curves: pd.DataFrame, policy: str, prefix: str) -> dict[str, Any]:
    work = curves[curves["policy"].astype(str).eq(str(policy))].copy()
    values = pd.to_numeric(work.get("delta_Wx"), errors="coerce")
    finite = work[values.notna()].copy()
    if finite.empty:
        return {
            f"{prefix}_max_delta_Wx": float("nan"),
            f"{prefix}_max_relative_delta_Wx": float("nan"),
            f"{prefix}_max_delta_top_tail_size": None,
            f"{prefix}_max_delta_sample_key": None,
            f"{prefix}_max_delta_ID": None,
        }
    index = pd.to_numeric(finite["delta_Wx"], errors="coerce").idxmax()
    row = finite.loc[index]
    return {
        f"{prefix}_max_delta_Wx": float(row["delta_Wx"]),
        f"{prefix}_max_relative_delta_Wx": float(row["relative_delta_Wx"]),
        f"{prefix}_max_delta_top_tail_size": int(row["top_tail_size"]),
        f"{prefix}_max_delta_sample_key": str(row.get(SAMPLE_KEY_COLUMN, "")),
        f"{prefix}_max_delta_ID": row.get("ID"),
    }


def _plot_curve_pair(
    curves: pd.DataFrame,
    title: str,
    output_path: Path,
    policy_labels: dict[str, str],
    drivers: pd.DataFrame | None,
    formats: list[str],
    dpi: int,
) -> None:
    colors = ["#222222", "#4c78a8", "#d95f02"]
    line_styles = ["-", "--", ":"]
    identical = curves_are_identical(curves)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    for color, line_style, (policy, group) in zip(
        colors, line_styles, curves.groupby("policy", sort=False)
    ):
        work = group.sort_values("top_tail_size", kind="stable")
        axes[0].plot(work["top_tail_size"], work["Wy"], color=color, linestyle=line_style, linewidth=1.1, label=policy_labels.get(str(policy), str(policy)))
        axes[1].plot(work["top_tail_size"], work["Wx"], color=color, linestyle=line_style, linewidth=1.1, label=policy_labels.get(str(policy), str(policy)))
    if drivers is not None and not drivers.empty and "upper_top_tail_size" in drivers.columns:
        baseline_policy = str(curves["policy"].iloc[0])
        baseline = curves[curves["policy"].eq(baseline_policy)]
        sizes = pd.to_numeric(drivers["upper_top_tail_size"], errors="coerce").dropna().astype(int)
        points = baseline[baseline["top_tail_size"].isin(sizes)]
        if not points.empty:
            axes[1].scatter(
                points["top_tail_size"],
                points["Wx"],
                s=32,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.2,
                label="confirmed Wx jump driver",
                zorder=5,
            )
    if identical:
        axes[1].text(
            0.50,
            0.96,
            "No confirmed drivers; before/after curves are identical.",
            transform=axes[1].transAxes,
            fontsize=8,
            color="#555555",
            ha="center",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2.0},
        )
    axes[0].set_ylabel("Wy (fixed full-data scale)")
    axes[1].set_ylabel("Wx (fixed full-data scale)")
    axes[1].set_xlabel("Top-tail sample count")
    for ax in axes:
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_path.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_task(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    metadata: dict[str, Any],
    settings: dict[str, Any],
    case_dir: Path,
    formats: list[str],
    dpi: int,
    write_figures: bool = True,
    figure_root: Path | None = None,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    data_file_text = str(metadata.get("data_file", "")).strip()
    data_file = Path(data_file_text) if data_file_text else None
    if data_file is not None and not data_file.is_file():
        raise FileNotFoundError(f"declared source data file does not exist: {data_file}")
    source_hash_before = _file_sha256(data_file) if data_file is not None else ""
    reference = fit_fixed_w_reference(
        frame,
        target_col=target_col,
        feature_columns=feature_columns,
        sliced_projections=int(settings["sliced_projections"]),
        random_state=int(settings["random_state"]),
    )
    detection = detect_jump_drivers(
        frame,
        target_col=target_col,
        reference=reference,
        start_fraction=float(settings["start_fraction"]),
        stop_fraction=float(settings["stop_fraction"]),
        min_abs_delta_wx=float(settings["min_abs_delta_wx"]),
        min_rel_delta_wx=float(settings["min_rel_delta_wx"]),
        metadata=metadata,
    )
    driver_keys = (
        detection.drivers[SAMPLE_KEY_COLUMN].astype(str).unique().tolist()
        if not detection.drivers.empty
        else []
    )
    prepared = prepare_jump_dataset(
        frame,
        target_col=target_col,
        driver_sample_keys=driver_keys,
        test_fraction=float(settings["test_fraction"]),
        random_state=int(settings["random_state"]),
        target_strata=int(settings["target_strata"]),
    )
    cleaned_full = prepared.full_data[
        ~prepared.full_data[SAMPLE_KEY_COLUMN].isin(driver_keys)
    ].reset_index(drop=True)
    full_curves = _curve_pair(
        detection.curve,
        cleaned_full,
        target_col,
        reference,
        metadata,
        settings,
        "full_data_baseline",
        "full_data_drop_jump_drivers",
    )
    baseline_train_curve = compute_samplewise_fixed_curve(
        prepared.train_baseline,
        target_col=target_col,
        reference=reference,
        start_fraction=float(settings["start_fraction"]),
        stop_fraction=float(settings["stop_fraction"]),
        metadata=metadata,
        policy="train_baseline",
    )
    training_curves = _curve_pair(
        baseline_train_curve,
        prepared.train_drop_jump_drivers,
        target_col,
        reference,
        metadata,
        settings,
        "train_baseline",
        "train_drop_jump_drivers",
    )
    coverage = _coverage_table(
        prepared.train_baseline,
        prepared.train_drop_jump_drivers,
        prepared.test_fixed,
        target_col,
        reference,
        metadata,
    )

    candidate_columns = [
        SAMPLE_KEY_COLUMN,
        "ID",
        "candidate_type",
        "lower_top_tail_size",
        "upper_top_tail_size",
        "delta_Wx",
        "relative_delta_Wx",
        "loo_reduction_Wx",
        "loo_relative_reduction_Wx",
        "is_confirmed_driver",
        "driver_action",
    ]
    candidates = detection.candidates if not detection.candidates.empty else _empty_with_columns(candidate_columns)
    drivers = detection.drivers if not detection.drivers.empty else _empty_with_columns(candidate_columns)
    tie_columns = [
        "tie_group_id",
        "target_value",
        "member_count",
        "member_sample_keys",
        "delta_Wx",
        "relative_delta_Wx",
        "confirmed_driver_count",
        "tie_status",
    ]
    tie_groups = detection.tie_groups if not detection.tie_groups.empty else _empty_with_columns(tie_columns)

    paths = {
        "train_baseline": case_dir / "train_baseline.csv",
        "train_drop": case_dir / "train_drop_jump_drivers.csv",
        "test_fixed": case_dir / "test_fixed.csv",
        "jump_candidates": case_dir / "jump_candidates.csv",
        "jump_drivers": case_dir / "jump_drivers.csv",
        "jump_tie_groups": case_dir / "jump_tie_groups.csv",
        "full_curves": case_dir / "full_data_w_before_after.csv",
        "training_curves": case_dir / "training_w_before_after.csv",
        "coverage": case_dir / "train_test_coverage.csv",
        "assignments": case_dir / "split_assignments.csv",
        "reference": case_dir / "w_reference.npz",
        "manifest": case_dir / "manifest.json",
    }
    _write_csv(prepared.train_baseline, paths["train_baseline"])
    _write_csv(prepared.train_drop_jump_drivers, paths["train_drop"])
    _write_csv(prepared.test_fixed, paths["test_fixed"])
    _write_csv(candidates, paths["jump_candidates"])
    _write_csv(drivers, paths["jump_drivers"])
    _write_csv(tie_groups, paths["jump_tie_groups"])
    _write_csv(full_curves, paths["full_curves"])
    _write_csv(training_curves, paths["training_curves"])
    _write_csv(coverage, paths["coverage"])
    _write_csv(prepared.split_assignments, paths["assignments"])
    reference.save(paths["reference"])

    source_hash_after = _file_sha256(data_file) if data_file is not None else ""
    if source_hash_before and source_hash_before != source_hash_after:
        raise RuntimeError(f"source data changed during analysis: {data_file}")
    driver_ids = _driver_ids(drivers)
    test_driver_overlap = set(prepared.test_fixed[SAMPLE_KEY_COLUMN]).intersection(driver_keys)
    baseline_coverage = coverage.loc[coverage["policy"].eq("train_baseline")].iloc[0]
    cleaned_coverage = coverage.loc[coverage["policy"].eq("train_drop_jump_drivers")].iloc[0]
    manifest = {
        **metadata,
        "target_col": target_col,
        "feature_columns": list(reference.feature_columns),
        "feature_count": len(reference.feature_columns),
        "full_valid_count": len(prepared.full_data),
        "train_baseline_count": len(prepared.train_baseline),
        "train_drop_count": len(prepared.train_drop_jump_drivers),
        "test_fixed_count": len(prepared.test_fixed),
        "jump_candidate_member_count": len(candidates),
        "confirmed_driver_count": len(driver_keys),
        "confirmed_driver_ids": driver_ids,
        "confirmed_driver_sample_keys": driver_keys,
        "test_contains_driver": bool(test_driver_overlap),
        "test_driver_overlap": sorted(test_driver_overlap),
        "test_fraction_basis": "full_valid_data",
        "requested_test_fraction": float(settings["test_fraction"]),
        "actual_test_fraction": float(len(prepared.test_fixed) / len(prepared.full_data)),
        "target_strata": int(settings["target_strata"]),
        "random_state": int(settings["random_state"]),
        "standardization_mode": STANDARDIZATION_MODE,
        "reference_hash": reference.reference_hash,
        "sliced_projections": int(len(reference.projections)),
        "x_mean": reference.x_mean.tolist(),
        "x_scale": reference.x_scale.tolist(),
        "y_mean": reference.y_mean,
        "y_scale": reference.y_scale,
        "train_baseline_hash": canonical_row_hash(prepared.train_baseline),
        "train_drop_hash": canonical_row_hash(prepared.train_drop_jump_drivers),
        "test_hash": canonical_row_hash(prepared.test_fixed),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": bool(data_file is not None and source_hash_before == source_hash_after),
        "exploratory_non_blind_detection": True,
        "fixed_test_domain": "non_driver_samples_only",
        **_max_delta_summary(full_curves, "full_data_baseline", "baseline_full"),
        **_max_delta_summary(full_curves, "full_data_drop_jump_drivers", "cleaned_full"),
        "baseline_train_test_Wx": float(baseline_coverage["Wx_train_to_test"]),
        "cleaned_train_test_Wx": float(cleaned_coverage["Wx_train_to_test"]),
        "delta_train_test_Wx": float(cleaned_coverage["delta_Wx_vs_baseline"]),
        "baseline_train_test_Wy": float(baseline_coverage["Wy_train_to_test"]),
        "cleaned_train_test_Wy": float(cleaned_coverage["Wy_train_to_test"]),
        "delta_train_test_Wy": float(cleaned_coverage["delta_Wy_vs_baseline"]),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    _write_json(manifest, paths["manifest"])

    if write_figures:
        root = Path(figure_root) if figure_root is not None else case_dir / "figures"
        case_name = str(metadata.get("case_key") or "task")
        _plot_curve_pair(
            full_curves,
            f"{case_name}: full-data fixed-W jump detection",
            root / "full_data_jump_detection" / case_name,
            {
                "full_data_baseline": "full data baseline",
                "full_data_drop_jump_drivers": "full data after driver removal",
            },
            drivers,
            formats,
            dpi,
        )
        _plot_curve_pair(
            training_curves,
            f"{case_name}: training data before/after driver removal",
            root / "training_before_after" / case_name,
            {
                "train_baseline": "baseline training data",
                "train_drop_jump_drivers": "training data after driver removal",
            },
            None,
            formats,
            dpi,
        )
    return {
        "manifest": manifest,
        "registry_rows": [
            {
                **metadata,
                "variant": "baseline",
                "train_file": str(paths["train_baseline"]),
                "test_file": str(paths["test_fixed"]),
                "train_count": len(prepared.train_baseline),
                "test_count": len(prepared.test_fixed),
                "train_hash": manifest["train_baseline_hash"],
                "test_hash": manifest["test_hash"],
            },
            {
                **metadata,
                "variant": "drop_jump_drivers",
                "train_file": str(paths["train_drop"]),
                "test_file": str(paths["test_fixed"]),
                "train_count": len(prepared.train_drop_jump_drivers),
                "test_count": len(prepared.test_fixed),
                "train_hash": manifest["train_drop_hash"],
                "test_hash": manifest["test_hash"],
            },
        ],
        "drivers": drivers,
        "full_curves": full_curves,
        "training_curves": training_curves,
        "coverage": coverage,
    }


def run_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    config_path = Path(args.config).resolve()
    settings = resolve_settings(load_analysis_config(config_path))
    if args.random_state is not None:
        settings["random_state"] = int(args.random_state)
    if args.sliced_projections is not None:
        settings["sliced_projections"] = int(args.sliced_projections)
    include_matbench = bool(settings["include_matbench"]) and not bool(args.exclude_matbench)
    specs = discover_task_specs(
        Path(args.splits_root),
        include_matbench=include_matbench,
        case_contains=args.case_contains,
    )
    if args.limit is not None:
        specs = specs[: int(args.limit)]
    if not specs:
        raise RuntimeError("No matching strength tasks were discovered")

    output_root = Path(args.output_root)
    manifest_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    driver_parts: list[pd.DataFrame] = []
    for index, spec in enumerate(specs, start=1):
        frame = load_task_dataframe(spec)
        if "__source_index__" not in frame.columns:
            frame["__source_index__"] = np.arange(len(frame), dtype=int)
        features = fallback_feature_columns(frame, spec.target_col, spec.feature_columns)
        current_key = case_key(spec.alloy_family, spec.dataset_name, spec.property_name)
        metadata = {
            "alloy_family": spec.alloy_family,
            "dataset_name": spec.dataset_name,
            "property": spec.property_name,
            "case_key": current_key,
            "data_file": str(spec.data_file),
            "source_manifest": str(spec.manifest_path or ""),
        }
        started = time.perf_counter()
        result = process_task(
            frame=frame,
            target_col=spec.target_col,
            feature_columns=features,
            metadata=metadata,
            settings=settings,
            case_dir=output_root / "cases" / current_key,
            formats=list(args.formats),
            dpi=int(args.dpi),
            write_figures=not bool(args.no_figures),
            figure_root=output_root / "figures",
        )
        manifest = dict(result["manifest"])
        manifest["runtime_seconds"] = float(time.perf_counter() - started)
        manifest_rows.append(manifest)
        registry_rows.extend(result["registry_rows"])
        drivers = result["drivers"].copy()
        if not drivers.empty:
            driver_parts.append(drivers)
        print(
            f"[{index}/{len(specs)}] {current_key}: "
            f"n={manifest['full_valid_count']}, drivers={manifest['confirmed_driver_count']}, "
            f"test={manifest['test_fixed_count']}, seconds={manifest['runtime_seconds']:.1f}",
            flush=True,
        )

    task_manifest = pd.DataFrame(manifest_rows)
    split_registry = pd.DataFrame(registry_rows)
    all_drivers = (
        pd.concat(driver_parts, ignore_index=True, sort=False)
        if driver_parts
        else _empty_with_columns(
            [
                "alloy_family",
                "dataset_name",
                "property",
                "case_key",
                SAMPLE_KEY_COLUMN,
                "ID",
                "candidate_type",
                "delta_Wx",
                "relative_delta_Wx",
                "loo_reduction_Wx",
                "loo_relative_reduction_Wx",
                "driver_action",
            ]
        )
    )
    csv_root = output_root / "csv"
    _write_csv(task_manifest, csv_root / "task_manifest.csv")
    _write_csv(split_registry, csv_root / "split_registry.csv")
    _write_csv(all_drivers, csv_root / "confirmed_jump_drivers.csv")
    _write_json(
        {
            "config_path": str(config_path),
            "output_root": str(output_root),
            "task_count": len(specs),
            "settings": settings,
            "detection_scope": "full valid data before fixed-test split",
            "test_pool": "confirmed non-driver samples only",
            "model_training_performed": False,
        },
        csv_root / "analysis_config_resolved.json",
    )
    return {
        "task_manifest": task_manifest,
        "split_registry": split_registry,
        "drivers": all_drivers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify fixed-coordinate full-data Wx jump drivers and prepare shared random test datasets."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-contains", default=None)
    parser.add_argument("--exclude-matbench", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--sliced-projections", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args)
    print(f"Tasks: {len(result['task_manifest'])}")
    print(f"Confirmed drivers: {len(result['drivers'])}")
    print(f"Output root: {Path(args.output_root)}")


if __name__ == "__main__":
    main()
