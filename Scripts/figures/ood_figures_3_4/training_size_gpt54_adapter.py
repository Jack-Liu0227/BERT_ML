from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SOURCE = "inner_random"
OOD_SOURCE_TO_AXIS = {
    "test_extrapolation_high20": "top20_property_extreme",
    "loco_farthest_cluster": "top_tail_distant_loco",
}
EXPECTED_ROW_COLUMNS = {
    RANDOM_SOURCE: "random_cv_fewshot_rows",
    "test_extrapolation_high20": "ood_top20_fewshot_rows",
    "loco_farthest_cluster": "ood_loco_farthest_fewshot_rows",
}
NORMALIZED_COLUMNS = [
    "task_id",
    "frozen_parameter_task_id",
    "model_family",
    "model",
    "display_method",
    "training_fraction",
    "repeat_id",
    "snapshot_id",
    "subset_hash",
    "predictions_sha256",
    "metrics_sha256",
    "failure_axis",
    "random_cv_mae",
    "ood_mae",
    "degradation_ratio",
    "run_dir",
    "source_name",
    "source_file",
]
AUDIT_COLUMNS = [
    "task_id",
    "training_fraction",
    "repeat_id",
    "model",
    "status",
    "include_in_figures",
    "reason",
    "random_cv_prediction_rows",
    "random_cv_expected_rows",
    "top20_prediction_rows",
    "top20_expected_rows",
    "loco_prediction_rows",
    "loco_expected_rows",
]


def load_gpt54_training_size_rows(
    root: str | Path,
    *,
    allow_partial: bool = False,
    model_name: str = "gpt-5.4",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize few-shot GPT training-size results into Figure 4 metric rows."""
    output_root = Path(root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"GPT training-size output root is missing: {output_root}")
    expected = _load_expected_rows(output_root)
    runs = _discover_runs(output_root, model_name=model_name)
    grouped: dict[tuple[str, float, int, str], dict[str, dict[str, Any]]] = {}
    for run in runs:
        key = (
            str(run["task_id"]),
            float(run["training_fraction"]),
            int(run["repeat_id"]),
            str(run["model"]),
        )
        grouped.setdefault(key, {})[str(run["source_name"])] = run

    normalized_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        task_id, fraction, repeat_id, model = key
        source_runs = grouped[key]
        expected_counts = expected.get((task_id, fraction, repeat_id), {})
        required_sources = [RANDOM_SOURCE, *OOD_SOURCE_TO_AXIS]
        missing_sources = [name for name in required_sources if name not in source_runs]
        reasons: list[str] = []
        if missing_sources:
            reasons.append(f"missing test-set results: {', '.join(missing_sources)}")

        partial_sources: list[str] = []
        for source_name in required_sources:
            run = source_runs.get(source_name)
            if run is None:
                continue
            if run["status"] != "complete":
                reasons.append(f"{source_name} status={run['status']}")
            mae = float(run["mae"])
            if not np.isfinite(mae) or (source_name == RANDOM_SOURCE and mae <= 0):
                reasons.append(f"{source_name} invalid MAE")
            # The split summary is immutable for an individual GPT evaluation.
            # Prefer it over aggregate CSVs, which may be regenerated or stale.
            expected_rows = run.get("expected_prediction_rows")
            if expected_rows is None:
                expected_rows = expected_counts.get(source_name)
            prediction_rows = run.get("prediction_rows")
            if expected_rows is None:
                partial_sources.append(f"{source_name} expected rows unavailable")
            elif prediction_rows != expected_rows:
                partial_sources.append(
                    f"{source_name} prediction_rows={prediction_rows}, expected={expected_rows}"
                )
        if partial_sources:
            reasons.extend(partial_sources)

        complete_sources = not missing_sources and all(
            source_runs[name]["status"] == "complete"
            and np.isfinite(float(source_runs[name]["mae"]))
            and (name != RANDOM_SOURCE or float(source_runs[name]["mae"]) > 0)
            for name in required_sources
        )
        full_test = complete_sources and not partial_sources
        include = complete_sources and (full_test or allow_partial)
        status = "complete" if full_test else "partial" if complete_sources else "incomplete"
        audit_rows.append(
            {
                "task_id": task_id,
                "training_fraction": fraction,
                "repeat_id": repeat_id,
                "model": model,
                "status": status,
                "include_in_figures": include,
                "reason": "; ".join(reasons),
                "random_cv_prediction_rows": _value(source_runs, RANDOM_SOURCE, "prediction_rows"),
                "random_cv_expected_rows": expected_counts.get(RANDOM_SOURCE),
                "top20_prediction_rows": _value(
                    source_runs, "test_extrapolation_high20", "prediction_rows"
                ),
                "top20_expected_rows": expected_counts.get("test_extrapolation_high20"),
                "loco_prediction_rows": _value(
                    source_runs, "loco_farthest_cluster", "prediction_rows"
                ),
                "loco_expected_rows": expected_counts.get("loco_farthest_cluster"),
            }
        )
        if not include:
            continue
        random_run = source_runs[RANDOM_SOURCE]
        random_cv_mae = float(random_run["mae"])
        for source_name, failure_axis in OOD_SOURCE_TO_AXIS.items():
            ood_run = source_runs[source_name]
            ood_mae = float(ood_run["mae"])
            normalized_rows.append(
                {
                    "task_id": task_id,
                    "frozen_parameter_task_id": task_id,
                    "model_family": "GPT",
                    "model": model,
                    "display_method": "GPT-5.4",
                    "training_fraction": fraction,
                    "repeat_id": repeat_id,
                    "snapshot_id": random_run.get("snapshot_id", ""),
                    "subset_hash": random_run.get("subset_hash", ""),
                    "predictions_sha256": ood_run.get("predictions_sha256", ""),
                    "metrics_sha256": ood_run.get("metrics_sha256", ""),
                    "failure_axis": failure_axis,
                    "random_cv_mae": random_cv_mae,
                    "ood_mae": ood_mae,
                    "degradation_ratio": ood_mae / random_cv_mae,
                    "run_dir": str(ood_run["run_dir"]),
                    "source_name": "gpt54_training_size_ood",
                    "source_file": str(ood_run["metrics_path"]),
                }
            )
    rows = pd.DataFrame(normalized_rows, columns=NORMALIZED_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return rows, audit


def _discover_runs(root: Path, *, model_name: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("training_size_ood_split_summary.json")):
        run_dir = summary_path.parent
        source_name = summary_path.parents[2].name
        if source_name not in {RANDOM_SOURCE, *OOD_SOURCE_TO_AXIS}:
            continue
        summary = _read_json(summary_path)
        if str(summary.get("model", "")) != model_name:
            continue
        marker_path = run_dir / "completion_marker.json"
        metrics_path = run_dir / "metrics.json"
        marker = _read_json(marker_path) if marker_path.exists() else {}
        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        runs.append(
            {
                "task_id": summary.get("task_id", ""),
                "training_fraction": float(summary.get("training_fraction", np.nan)),
                "repeat_id": int(summary.get("repeat_id", -1)),
                "snapshot_id": summary.get("snapshot_id", ""),
                "subset_hash": summary.get("subset_hash", ""),
                "model": summary.get("model", model_name),
                "source_name": source_name,
                "status": marker.get("status", "missing"),
                "prediction_rows": _optional_int(marker.get("prediction_rows")),
                "expected_prediction_rows": _optional_int(
                    summary.get("expected_prediction_rows")
                ),
                "predictions_sha256": marker.get("predictions_sha256", ""),
                "metrics_sha256": marker.get("metrics_sha256", ""),
                "mae": _extract_mae(metrics),
                "run_dir": run_dir,
                "metrics_path": metrics_path,
            }
        )
    return runs


def _load_expected_rows(root: Path) -> dict[tuple[str, float, int], dict[str, int]]:
    result: dict[tuple[str, float, int], dict[str, int]] = {}
    # `run_status.csv` is emitted together with each GPT prediction and retains
    # every task even when split_audit.csv is regenerated incrementally.
    status_path = root / "csv" / "run_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path)
        required = {
            "task_id",
            "training_fraction",
            "repeat_id",
            "test_set",
            "expected_prediction_rows",
        }
        missing = required.difference(status.columns)
        if missing:
            raise ValueError(f"GPT run status is missing columns: {sorted(missing)}")
        for row in status.to_dict(orient="records"):
            source_name = str(row["test_set"])
            if source_name not in EXPECTED_ROW_COLUMNS or pd.isna(row["expected_prediction_rows"]):
                continue
            key = (str(row["task_id"]), float(row["training_fraction"]), int(row["repeat_id"]))
            result.setdefault(key, {})[source_name] = int(row["expected_prediction_rows"])

    path = root / "csv" / "split_audit.csv"
    if not path.exists():
        return result
    audit = pd.read_csv(path)
    required = {"task_id", "training_fraction", "repeat_id", *EXPECTED_ROW_COLUMNS.values()}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"GPT split audit is missing columns: {sorted(missing)}")
    for row in audit.to_dict(orient="records"):
        key = (str(row["task_id"]), float(row["training_fraction"]), int(row["repeat_id"]))
        expected = result.setdefault(key, {})
        for source_name, column in EXPECTED_ROW_COLUMNS.items():
            expected.setdefault(source_name, int(row[column]))
    return result


def _extract_mae(metrics: dict[str, Any]) -> float:
    for value in metrics.values():
        if isinstance(value, dict) and "mae" in value:
            mae = float(value["mae"])
            if np.isfinite(mae) and mae >= 0:
                return mae
    return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _value(
    source_runs: dict[str, dict[str, Any]], source_name: str, key: str
) -> Any:
    run = source_runs.get(source_name)
    return None if run is None else run.get(key)
