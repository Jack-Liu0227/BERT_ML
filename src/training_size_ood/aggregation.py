"""Aggregation and coverage reporting for Figure 4 training-size runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import validate_completed_artifacts
from .core import (
    FAILURE_AXIS_LOCO,
    FAILURE_AXIS_TOP20,
    LOCO_TEST_NAME,
    TOP20_TEST_NAME,
    validate_run_metrics_payload,
)


RUN_KEY_COLUMNS = [
    "task_id",
    "model_family",
    "model",
    "training_fraction",
    "repeat_id",
]

RUN_IDENTITY_COLUMNS = RUN_KEY_COLUMNS + [
    "frozen_parameter_task_id",
    "sampling_seed",
    "snapshot_id",
    "parameter_hash",
    "subset_hash",
    "random_cv_test_hash",
    "top20_test_hash",
    "loco_test_hash",
    "training_pool_hash",
]

RESULT_COLUMNS = [
    "task_id",
    "frozen_parameter_task_id",
    "model_family",
    "model",
    "training_fraction",
    "repeat_id",
    "sampling_seed",
    "snapshot_id",
    "parameter_hash",
    "subset_hash",
    "random_cv_test_hash",
    "top20_test_hash",
    "loco_test_hash",
    "training_pool_hash",
    "predictions_sha256",
    "metrics_sha256",
    "failure_axis",
    "random_cv_mae",
    "ood_mae",
    "degradation_ratio",
    "run_dir",
]


def collect_run_results(output_root: str | Path) -> pd.DataFrame:
    root = Path(output_root)
    inventory_path = root / "csv" / "run_inventory.csv"
    valid_run_dirs: set[str] | None = None
    if inventory_path.exists():
        status = scan_run_status(root)
        valid_run_dirs = set(status.loc[status["status"].eq("complete"), "run_dir"].astype(str))
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").rglob("run_manifest.json")):
        if valid_run_dirs is not None and str(manifest_path.parent) not in valid_run_dirs:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete":
            continue
        if not _completion_artifacts_valid(manifest_path.parent, manifest):
            continue
        metrics_path = manifest_path.parent / "run_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_run_metrics_payload(metrics)
        random_cv_mae = float(metrics["random_cv_mae"])
        if not np.isfinite(random_cv_mae) or random_cv_mae <= 0:
            raise ValueError(f"Invalid Random-CV MAE in {metrics_path}: {random_cv_mae}")
        test_sets = dict(metrics.get("test_set_metrics", {}))
        for test_name, failure_axis in [
            (TOP20_TEST_NAME, FAILURE_AXIS_TOP20),
            (LOCO_TEST_NAME, FAILURE_AXIS_LOCO),
        ]:
            if test_name not in test_sets:
                raise ValueError(f"Missing {test_name} metrics in {metrics_path}")
            ood_mae = float(test_sets[test_name]["mae"])
            rows.append(
                {
                    **{
                        key: manifest.get(key)
                        for key in [
                            "task_id",
                            "frozen_parameter_task_id",
                            "model_family",
                            "model",
                            "training_fraction",
                            "repeat_id",
                            "sampling_seed",
                            "snapshot_id",
                            "parameter_hash",
                            "subset_hash",
                            "random_cv_test_hash",
                            "top20_test_hash",
                            "loco_test_hash",
                            "training_pool_hash",
                            "predictions_sha256",
                            "metrics_sha256",
                        ]
                    },
                    "failure_axis": failure_axis,
                    "random_cv_mae": random_cv_mae,
                    "ood_mae": ood_mae,
                    "degradation_ratio": float(ood_mae / random_cv_mae),
                    "run_dir": str(manifest_path.parent),
                }
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def degradation_ratio_summary(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task_id",
        "model_family",
        "model",
        "failure_axis",
        "training_fraction",
        "n_repeats",
        "mean_degradation_ratio",
        "sd_degradation_ratio",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        data.groupby(
            ["task_id", "model_family", "model", "failure_axis", "training_fraction"],
            dropna=False,
        )["degradation_ratio"]
        .agg(n_repeats="size", mean_degradation_ratio="mean", sd_degradation_ratio="std")
        .reset_index()
    )
    return summary[columns]


def family_curve_summary(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    columns = [
        "model_family",
        "failure_axis",
        "training_fraction",
        "task_count",
        "model_count",
        "mean_degradation_ratio",
        "ci95_low",
        "ci95_high",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    group_columns = ["model_family", "failure_axis", "training_fraction"]
    for keys, group in data.groupby(group_columns, dropna=False):
        point = _equal_task_mean(group)
        draws = np.asarray(
            [_hierarchical_mean_draw(group, rng) for _ in range(max(0, bootstrap_iterations))],
            dtype=float,
        )
        low, high = _percentile_interval(draws, point)
        rows.append(
            {
                "model_family": keys[0],
                "failure_axis": keys[1],
                "training_fraction": float(keys[2]),
                "task_count": int(group["task_id"].nunique()),
                "model_count": int(group[["task_id", "model"]].drop_duplicates().shape[0]),
                "mean_degradation_ratio": float(point),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def sensitivity_summary(
    data: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
) -> pd.DataFrame:
    columns = [
        "model_family",
        "failure_axis",
        "task_count",
        "model_count",
        "sensitivity",
        "ci95_low",
        "ci95_high",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(["model_family", "failure_axis"], dropna=False):
        point = _equal_task_sensitivity(group)
        draws = np.asarray(
            [_hierarchical_sensitivity_draw(group, rng) for _ in range(max(0, bootstrap_iterations))],
            dtype=float,
        )
        low, high = _percentile_interval(draws, point)
        rows.append(
            {
                "model_family": keys[0],
                "failure_axis": keys[1],
                "task_count": int(group["task_id"].nunique()),
                "model_count": int(group[["task_id", "model"]].drop_duplicates().shape[0]),
                "sensitivity": float(point),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def scan_run_status(output_root: str | Path) -> pd.DataFrame:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "runs").rglob("run_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                **payload,
                "run_dir": str(manifest_path.parent),
                "_completion_artifacts_valid": _completion_artifacts_valid(
                    manifest_path.parent,
                    payload,
                ),
            }
        )
    actual = pd.DataFrame(rows)
    inventory_path = root / "csv" / "run_inventory.csv"
    if not inventory_path.exists():
        if not actual.empty:
            invalid = actual["status"].eq("complete") & ~actual["_completion_artifacts_valid"]
            actual.loc[invalid, "status"] = "incomplete"
            actual = actual.drop(columns=["_completion_artifacts_valid"])
        return actual
    inventory = pd.read_csv(inventory_path)
    if actual.empty:
        inventory["status"] = "missing"
        inventory["run_dir"] = ""
        return inventory
    if actual.duplicated(RUN_KEY_COLUMNS).any():
        raise ValueError("Multiple run manifests share the same task/model/fraction/repeat key")
    actual_lookup = {
        _run_key(record): record for record in actual.to_dict(orient="records")
    }
    merged_rows: list[dict[str, Any]] = []
    for expected in inventory.to_dict(orient="records"):
        observed = actual_lookup.get(_run_key(expected))
        row = dict(expected)
        if observed is None:
            row.update({"status": "missing", "run_dir": ""})
        else:
            identity_matches = all(
                _values_equal(expected.get(column), observed.get(column))
                for column in RUN_IDENTITY_COLUMNS
            )
            observed_status = str(observed.get("status", "missing"))
            if not identity_matches:
                status = "stale"
            elif observed_status == "complete" and not bool(observed["_completion_artifacts_valid"]):
                status = "incomplete"
            else:
                status = observed_status
            row.update(
                {
                    "status": status,
                    "run_dir": str(observed.get("run_dir", "")),
                }
            )
            if observed.get("error"):
                row["error"] = observed["error"]
            for column in ["predictions_sha256", "metrics_sha256"]:
                if observed.get(column):
                    row[column] = observed[column]
        merged_rows.append(row)
    return pd.DataFrame(merged_rows)


def aggregate_experiment(
    output_root: str | Path,
    *,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 20260707,
    require_complete: bool = True,
) -> dict[str, pd.DataFrame]:
    root = Path(output_root)
    csv_dir = root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    status = scan_run_status(root)
    expected = int(len(status))
    complete = int(status["status"].eq("complete").sum()) if not status.empty else 0
    failed = int(status["status"].eq("failed").sum()) if not status.empty else 0
    stale = int(status["status"].eq("stale").sum()) if not status.empty else 0
    incomplete = int(status["status"].eq("incomplete").sum()) if not status.empty else 0
    missing = max(0, expected - complete - failed)
    coverage = pd.DataFrame(
        [
            {
                "expected_run_count": expected,
                "complete_run_count": complete,
                "failed_run_count": failed,
                "missing_run_count": missing,
                "stale_run_count": stale,
                "incomplete_run_count": incomplete,
                "coverage_fraction": float(complete / expected) if expected else 0.0,
            }
        ]
    )
    status.to_csv(csv_dir / "run_status.csv", index=False)
    coverage.to_csv(csv_dir / "coverage_report.csv", index=False)
    if require_complete and complete != expected:
        raise ValueError(f"Training-size coverage is incomplete: complete={complete}, expected={expected}")
    data = collect_run_results(root)
    if data.duplicated(RUN_KEY_COLUMNS + ["failure_axis"]).any():
        raise ValueError("Duplicate degradation-ratio records were collected")
    if len(data) != complete * 2:
        raise ValueError(
            f"Expected two failure-axis rows per complete run, found rows={len(data)}, complete={complete}"
        )
    model_summary = degradation_ratio_summary(data)
    family_summary = family_curve_summary(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    sensitivity = sensitivity_summary(
        data,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    result_dir = csv_dir if require_complete else csv_dir / "partial"
    result_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(result_dir / "degradation_ratio_by_repeat.csv", index=False)
    model_summary.to_csv(result_dir / "degradation_ratio_summary.csv", index=False)
    family_summary.to_csv(result_dir / "family_curve_summary.csv", index=False)
    sensitivity.to_csv(result_dir / "sensitivity_summary.csv", index=False)
    return {
        "by_repeat": data,
        "model_summary": model_summary,
        "family_summary": family_summary,
        "sensitivity": sensitivity,
        "status": status,
        "coverage": coverage,
    }


def _equal_task_mean(group: pd.DataFrame) -> float:
    model_means = (
        group.groupby(["task_id", "model"], dropna=False)["degradation_ratio"].mean().reset_index()
    )
    task_means = model_means.groupby("task_id", dropna=False)["degradation_ratio"].mean()
    return float(task_means.mean())


def _hierarchical_mean_draw(group: pd.DataFrame, rng: np.random.Generator) -> float:
    task_values: list[float] = []
    tasks = group["task_id"].drop_duplicates().to_numpy()
    for task_id in rng.choice(tasks, size=len(tasks), replace=True):
        task_group = group[group["task_id"].eq(task_id)]
        models = task_group["model"].drop_duplicates().to_numpy()
        model_values: list[float] = []
        for model in rng.choice(models, size=len(models), replace=True):
            values = task_group[task_group["model"].eq(model)]["degradation_ratio"].to_numpy(dtype=float)
            model_values.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        task_values.append(float(np.mean(model_values)))
    return float(np.mean(task_values))


def _equal_task_sensitivity(group: pd.DataFrame) -> float:
    model_values: list[dict[str, Any]] = []
    for keys, model_group in group.groupby(["task_id", "model"], dropna=False):
        sensitivity = _model_sensitivity(model_group)
        if np.isfinite(sensitivity):
            model_values.append({"task_id": keys[0], "sensitivity": sensitivity})
    frame = pd.DataFrame(model_values)
    if frame.empty:
        return float("nan")
    return float(frame.groupby("task_id")["sensitivity"].mean().mean())


def _hierarchical_sensitivity_draw(group: pd.DataFrame, rng: np.random.Generator) -> float:
    tasks = group["task_id"].drop_duplicates().to_numpy()
    task_values: list[float] = []
    for task_id in rng.choice(tasks, size=len(tasks), replace=True):
        task_group = group[group["task_id"].eq(task_id)]
        models = task_group["model"].drop_duplicates().to_numpy()
        model_values: list[float] = []
        for model in rng.choice(models, size=len(models), replace=True):
            model_group = task_group[task_group["model"].eq(model)]
            fraction_rows: list[dict[str, float]] = []
            for fraction, fraction_group in model_group.groupby("training_fraction"):
                values = fraction_group["degradation_ratio"].to_numpy(dtype=float)
                fraction_rows.append(
                    {
                        "training_fraction": float(fraction),
                        "degradation_ratio": float(rng.choice(values, size=len(values), replace=True).mean()),
                    }
                )
            value = _model_sensitivity(pd.DataFrame(fraction_rows))
            if np.isfinite(value):
                model_values.append(value)
        if model_values:
            task_values.append(float(np.mean(model_values)))
    return float(np.mean(task_values)) if task_values else float("nan")


def _model_sensitivity(group: pd.DataFrame) -> float:
    means = group.groupby("training_fraction", dropna=False)["degradation_ratio"].mean().reset_index()
    means = means[
        np.isfinite(means["training_fraction"])
        & np.isfinite(means["degradation_ratio"])
        & means["training_fraction"].gt(0)
    ]
    if len(means) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(means["training_fraction"].to_numpy(dtype=float)), means["degradation_ratio"], 1)
    return float(-slope)


def _percentile_interval(draws: np.ndarray, fallback: float) -> tuple[float, float]:
    valid = draws[np.isfinite(draws)]
    if valid.size == 0:
        return float(fallback), float(fallback)
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high)


def _run_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(column) for column in RUN_KEY_COLUMNS)


def _values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        return float(left) == float(right)
    return str(left) == str(right)


def _completion_artifacts_valid(run_dir: Path, manifest: dict[str, Any]) -> bool:
    metrics_path = run_dir / "run_metrics.json"
    marker_path = run_dir / "completion_marker.json"
    if not metrics_path.exists() or not marker_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("status") != "complete":
        return False
    if any(not _values_equal(manifest.get(column), marker.get(column)) for column in RUN_IDENTITY_COLUMNS):
        return False
    try:
        validate_run_metrics_payload(metrics)
    except ValueError:
        return False
    return validate_completed_artifacts(run_dir, manifest, marker)
