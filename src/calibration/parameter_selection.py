from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .schemas import CalibrationConfig


FOLD_COUNT = 5
EXPECTED_TASK_MODEL_ENTRIES = 110
TASK_MODEL_KEY = ["task_id", "model_family", "model"]
FREEZABLE_STATUSES = {"reusable", "not_applicable"}
EXCLUDED_MODELS = {"TabPFN-V2-Numeric"}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_parameter_hash(parameters: dict[str, Any]) -> str:
    canonical = json.dumps(
        _canonicalize(parameters),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _model_kind(model_family: object, model: object) -> str:
    family = str(model_family).lower()
    name = str(model).lower()
    if "tabpfn" in family or "tabpfn" in name:
        return "tabpfn"
    if "llmprop" in family or "llm-prop" in name or "llmprop" in name:
        return "llmprop"
    if "bert" in family or "bert" in name:
        return "bert"
    return "traditional"


def _normalize_task_id(value: object) -> str:
    text = str(value)
    if text.startswith("MatbenchSteels__"):
        return "Matbench_Steel__" + text.split("__", 1)[1]
    return text


def _fold_number(value: object) -> int:
    text = str(value)
    if text.lower().startswith("fold_"):
        return int(text.split("_")[-1])
    raise ValueError(f"RandomCV split must use fold_N identifiers, found: {text}")


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _missing_score_record(
    base: dict[str, Any],
    fold: int,
    reason: str,
    model_dir: str,
    *,
    message: str = "",
) -> dict[str, Any]:
    return {
        **{name: base[name] for name in TASK_MODEL_KEY + ["parameter_hash"]},
        "fold_id": f"fold_{fold}",
        "missing_reason": reason,
        "model_dir": model_dir,
        "candidate_trial_number": base.get("candidate_trial_number"),
        "search_parameters_json": base.get("search_parameters_json", "{}"),
        "effective_parameters_json": base.get("effective_parameters_json", "{}"),
        "candidate_source_path": base.get("candidate_source_path", ""),
        "evaluation_output_dir": str(_evaluation_output_dir(base, fold)),
        "rerun_status": "pending",
        "message": message,
    }


def _evaluation_output_dir(base: dict[str, Any], fold: int) -> Path:
    return (
        Path("output")
        / "calibration"
        / "parameter_evaluations"
        / _safe_component(base["task_id"])
        / _safe_component(base["model"])
        / str(base["parameter_hash"])
        / f"fold_{fold}"
    )


def _json_dict(value: object) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    text = str(value).strip()
    if not text:
        return {}
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def _native_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
    return value


def _read_history_trials(model_dir: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "llmprop":
        summary_path = model_dir / "optuna_trials" / "llmprop_optuna_summary.json"
        if not summary_path.exists():
            return []
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(payload.get("selection_metric", "")) != "inner_valid_mae":
            raise ValueError(f"LLM-Prop Optuna summary is not based on inner_valid_mae: {summary_path}")
        records = []
        for item in payload.get("trials", []):
            if not isinstance(item, dict) or item.get("failed"):
                continue
            params = item.get("params", {})
            if not isinstance(params, dict):
                continue
            trial_number = int(item["trial_number"])
            manifest_path = (
                summary_path.parent / f"trial_{trial_number:03d}" / "llmprop_manifest.json"
            )
            if not manifest_path.exists():
                raise FileNotFoundError(f"LLM-Prop trial manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_mae = (manifest.get("metrics", {}) or {}).get("best_valid_mae")
            reported_mae = item.get("value")
            if manifest_mae is None or reported_mae is None or not np.isclose(
                float(manifest_mae),
                float(reported_mae),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "LLM-Prop trial best_valid_mae does not match its Optuna summary: "
                    f"{manifest_path}"
                )
            records.append(
                {
                    "trial_number": trial_number,
                    "search_parameters": _canonicalize(params),
                    "parameter_hash": build_parameter_hash(params),
                    "model_config": {},
                    "validation_mae": float(reported_mae),
                    "score_source_path": str(summary_path),
                }
            )
        return records

    history_path = model_dir / "optuna_study_history.csv"
    if not history_path.exists():
        return []
    columns = pd.read_csv(history_path, nrows=0).columns.tolist()
    parameter_columns = sorted(name for name in columns if name.startswith("params_"))
    use_columns = [name for name in ("number", "state", "user_attrs_model_config") if name in columns]
    use_columns.extend(parameter_columns)
    history = pd.read_csv(history_path, usecols=use_columns)
    if "state" in history.columns:
        history = history[history["state"].astype(str).str.upper().eq("COMPLETE")]
    records = []
    for _, row in history.iterrows():
        params = {
            name.removeprefix("params_"): _native_value(row[name])
            for name in parameter_columns
            if _native_value(row[name]) is not None
        }
        model_config: dict[str, Any] = {}
        text = str(row.get("user_attrs_model_config", "") or "").strip()
        if text:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    model_config = parsed
            except (SyntaxError, ValueError):
                model_config = {}
        records.append(
            {
                "trial_number": int(row["number"]),
                "search_parameters": _canonicalize(params),
                "parameter_hash": build_parameter_hash(params),
                "model_config": _canonicalize(model_config),
            }
        )
    return records


def _effective_parameters(
    template: dict[str, Any],
    kind: str,
    search_parameters: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    effective = deepcopy(template)
    if kind == "traditional":
        model_params = effective.get("model_params", {})
        if not isinstance(model_params, dict):
            model_params = {}
        model_params.update(search_parameters)
        effective["model_params"] = model_params
    elif kind == "bert":
        if model_config:
            effective["model_config"] = model_config
        effective["optuna_parameters"] = search_parameters
    elif kind == "llmprop":
        effective.update(search_parameters)
    return _canonicalize(effective)


def _actual_prediction_columns(frame: pd.DataFrame) -> tuple[str, str]:
    actual = next((name for name in frame.columns if str(name).lower().endswith(("_actual", "_true"))), None)
    predicted = next(
        (name for name in frame.columns if str(name).lower().endswith(("_predicted", "_prediction", "_pred"))),
        None,
    )
    if actual is None or predicted is None:
        raise ValueError(f"Prediction file lacks actual/predicted columns: {frame.columns.tolist()}")
    return str(actual), str(predicted)


def score_validation_prediction_files(trial_dir: str | Path) -> dict[str, Any]:
    root = Path(trial_dir)
    candidate_paths = sorted(root.glob("fold_*/all_predictions.csv"))
    direct_path = root / "predictions" / "validation_predictions.csv"
    if direct_path.exists():
        candidate_paths.append(direct_path)
    actual_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    used_paths: list[str] = []
    for path in candidate_paths:
        frame = pd.read_csv(path)
        dataset_column = next(
            (name for name in frame.columns if str(name).lower() in {"dataset", "split", "set"}),
            None,
        )
        if dataset_column is not None:
            mask = frame[dataset_column].astype(str).str.lower().isin({"validation", "valid", "val"})
            frame = frame[mask]
        elif path != direct_path:
            raise ValueError(f"Prediction file has no dataset label for validation filtering: {path}")
        if frame.empty:
            continue
        actual_column, predicted_column = _actual_prediction_columns(frame)
        actual_values.append(pd.to_numeric(frame[actual_column], errors="coerce").to_numpy(dtype=float))
        predicted_values.append(pd.to_numeric(frame[predicted_column], errors="coerce").to_numpy(dtype=float))
        used_paths.append(str(path))
    if not actual_values:
        raise FileNotFoundError(f"No validation predictions found below: {root}")
    actual = np.concatenate(actual_values)
    predicted = np.concatenate(predicted_values)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        raise ValueError(f"Validation predictions contain no finite values: {root}")
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_validation": int(len(actual)),
        "source_paths": used_paths,
    }


def _trial_dir(model_dir: Path, kind: str, trial_number: int) -> Path:
    if kind == "llmprop":
        return model_dir / "optuna_trials" / f"trial_{trial_number:03d}"
    return model_dir / "predictions" / "optuna_trials" / f"trial_{trial_number}"


def _deduplicate_randomcv_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    group_columns = TASK_MODEL_KEY + ["fold_number"]
    for _, group in frame.sort_values("model_dir").groupby(group_columns, sort=True):
        payloads = {
            json.dumps(_json_dict(value), sort_keys=True, separators=(",", ":"))
            for value in group["parameter_payload_json"]
        }
        if len(payloads) > 1:
            key = tuple(str(group.iloc[0][name]) for name in group_columns)
            raise ValueError(f"Duplicate RandomCV sources disagree for key: {key}")
        rows.append(group.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_randomcv_parameter_tables(
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "task_id",
        "protocol",
        "split_id",
        "model_family",
        "model",
        "status",
        "model_dir",
        "parameter_payload_json",
    }
    missing_columns = sorted(required - set(audit.columns))
    if missing_columns:
        raise ValueError(f"Parameter audit is missing columns: {missing_columns}")
    frame = audit[
        audit["protocol"].astype(str).str.lower().eq("randomcv")
        & audit["status"].astype(str).isin(FREEZABLE_STATUSES)
        & ~audit["model"].astype(str).isin(EXCLUDED_MODELS)
    ].copy()
    frame["task_id"] = frame["task_id"].map(_normalize_task_id)
    frame["fold_number"] = frame["split_id"].map(_fold_number)
    frame = _deduplicate_randomcv_rows(frame)

    trial_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def trials_for(model_dir: Path, kind: str) -> list[dict[str, Any]]:
        key = (str(model_dir.resolve()), kind)
        if key not in trial_cache:
            trial_cache[key] = _read_history_trials(model_dir, kind)
        return trial_cache[key]

    candidate_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(TASK_MODEL_KEY, sort=True):
        task_id, model_family, model = (str(value) for value in key)
        kind = _model_kind(model_family, model)
        by_fold = {int(row["fold_number"]): row for _, row in group.iterrows()}
        if 0 not in by_fold:
            missing_rows.append(
                {
                    "task_id": task_id,
                    "model_family": model_family,
                    "model": model,
                    "parameter_hash": "",
                    "fold_id": "fold_0",
                    "missing_reason": "missing_candidate_catalog_fold",
                    "model_dir": "",
                }
            )
            continue
        fold_zero = by_fold[0]
        template = _json_dict(fold_zero["parameter_payload_json"])
        if kind == "tabpfn":
            parameter_hash = build_parameter_hash(template)
            fixed_rows.append(
                {
                    "task_id": task_id,
                    "model_family": model_family,
                    "model": model,
                    "parameter_hash": parameter_hash,
                    "effective_parameters_json": json.dumps(template, ensure_ascii=True, sort_keys=True),
                    "selection_status": "fixed_runtime",
                    "source_paths_json": json.dumps(
                        [str(by_fold[fold]["model_dir"]) for fold in sorted(by_fold)],
                        ensure_ascii=True,
                    ),
                }
            )
            continue

        fold_zero_dir = Path(str(fold_zero["model_dir"]))
        candidates_by_hash: dict[str, dict[str, Any]] = {}
        for item in sorted(trials_for(fold_zero_dir, kind), key=lambda value: int(value["trial_number"])):
            candidates_by_hash.setdefault(str(item["parameter_hash"]), item)
        candidates = list(candidates_by_hash.values())
        for candidate in candidates:
            effective = _effective_parameters(
                template,
                kind,
                candidate["search_parameters"],
                candidate["model_config"],
            )
            base = {
                "task_id": task_id,
                "model_family": model_family,
                "model": model,
                "parameter_hash": candidate["parameter_hash"],
                "candidate_trial_number": int(candidate["trial_number"]),
                "search_parameters_json": json.dumps(
                    candidate["search_parameters"], ensure_ascii=True, sort_keys=True
                ),
                "effective_parameters_json": json.dumps(effective, ensure_ascii=True, sort_keys=True),
                "candidate_source_path": str(fold_zero_dir),
            }
            candidate_rows.append(base.copy())
            matrix = base.copy()
            score_sources: dict[str, Any] = {}
            for fold in range(FOLD_COUNT):
                matrix[f"mae_fold_{fold}"] = np.nan
                matrix[f"rmse_fold_{fold}"] = np.nan
                matrix[f"r2_fold_{fold}"] = np.nan
                row = by_fold.get(fold)
                if row is None:
                    missing_rows.append(_missing_score_record(base, fold, "missing_randomcv_fold", ""))
                    continue
                model_dir = Path(str(row["model_dir"]))
                evaluation_dir = _evaluation_output_dir(base, fold)
                if evaluation_dir.exists():
                    try:
                        score = score_validation_prediction_files(evaluation_dir)
                    except (FileNotFoundError, ValueError):
                        score = None
                    if score is not None:
                        matrix[f"mae_fold_{fold}"] = score["mae"]
                        matrix[f"rmse_fold_{fold}"] = score["rmse"]
                        matrix[f"r2_fold_{fold}"] = score["r2"]
                        score_sources[f"fold_{fold}"] = score["source_paths"]
                        continue
                matches = [
                    item
                    for item in trials_for(model_dir, kind)
                    if item["parameter_hash"] == candidate["parameter_hash"]
                ]
                if not matches:
                    missing_rows.append(
                        _missing_score_record(
                            base,
                            fold,
                            "parameter_not_evaluated_in_fold",
                            str(model_dir),
                        )
                    )
                    continue
                trial_number = min(int(item["trial_number"]) for item in matches)
                try:
                    score = score_validation_prediction_files(_trial_dir(model_dir, kind, trial_number))
                except (FileNotFoundError, ValueError) as exc:
                    missing_rows.append(
                        _missing_score_record(
                            base,
                            fold,
                            "missing_validation_predictions",
                            str(model_dir),
                            message=str(exc),
                        )
                    )
                    continue
                matrix[f"mae_fold_{fold}"] = score["mae"]
                matrix[f"rmse_fold_{fold}"] = score["rmse"]
                matrix[f"r2_fold_{fold}"] = score["r2"]
                score_sources[f"fold_{fold}"] = score["source_paths"]
            matrix["source_paths_json"] = json.dumps(score_sources, ensure_ascii=True, sort_keys=True)
            matrix_rows.append(matrix)

    candidates_frame = pd.DataFrame(candidate_rows)
    matrix_frame = pd.DataFrame(matrix_rows)
    if not matrix_frame.empty:
        _, matrix_frame = select_best_parameter_rows(matrix_frame)
    missing_frame = pd.DataFrame(missing_rows)
    fixed_frame = pd.DataFrame(fixed_rows)
    return candidates_frame, matrix_frame, missing_frame, fixed_frame


def build_existing_randomcv_parameter_tables(
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "task_id",
        "protocol",
        "split_id",
        "model_family",
        "model",
        "status",
        "model_dir",
        "parameter_payload_json",
    }
    missing_columns = sorted(required - set(audit.columns))
    if missing_columns:
        raise ValueError(f"Parameter audit is missing columns: {missing_columns}")
    frame = audit[
        audit["protocol"].astype(str).str.lower().eq("randomcv")
        & audit["status"].astype(str).isin(FREEZABLE_STATUSES)
        & ~audit["model"].astype(str).isin(EXCLUDED_MODELS)
    ].copy()
    frame["task_id"] = frame["task_id"].map(_normalize_task_id)
    frame["fold_number"] = frame["split_id"].map(_fold_number)
    frame = _deduplicate_randomcv_rows(frame)

    trial_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(TASK_MODEL_KEY, sort=True):
        task_id, model_family, model = (str(value) for value in key)
        kind = _model_kind(model_family, model)
        group = group.sort_values("fold_number", kind="stable")
        if kind == "tabpfn":
            representative = group.iloc[0]
            template = _json_dict(representative["parameter_payload_json"])
            fixed_rows.append(
                {
                    "task_id": task_id,
                    "model_family": model_family,
                    "model": model,
                    "parameter_hash": build_parameter_hash(template),
                    "effective_parameters_json": json.dumps(template, ensure_ascii=True, sort_keys=True),
                    "selection_status": "fixed_runtime",
                    "source_paths_json": json.dumps(group["model_dir"].astype(str).tolist(), ensure_ascii=True),
                }
            )
            continue

        for _, source in group.iterrows():
            outer_fold_id = str(source["split_id"])
            model_dir = Path(str(source["model_dir"]))
            template = _json_dict(source["parameter_payload_json"])
            trials = _read_history_trials(model_dir, kind)
            if not trials:
                missing_rows.append(
                    {
                        "task_id": task_id,
                        "model_family": model_family,
                        "model": model,
                        "outer_fold_id": outer_fold_id,
                        "missing_reason": "missing_trial_history",
                        "source_path": str(model_dir),
                    }
                )
                continue
            for trial in trials:
                if kind == "llmprop":
                    validation_mae = float(trial.get("validation_mae", np.nan))
                    validation_rmse = np.nan
                    validation_r2 = np.nan
                    score_source = str(trial.get("score_source_path", model_dir))
                    if not np.isfinite(validation_mae):
                        missing_rows.append(
                            {
                                "task_id": task_id,
                                "model_family": model_family,
                                "model": model,
                                "outer_fold_id": outer_fold_id,
                                "candidate_trial_number": int(trial["trial_number"]),
                                "missing_reason": "missing_reported_validation_mae",
                                "source_path": score_source,
                            }
                        )
                        continue
                else:
                    trial_path = _trial_dir(model_dir, kind, int(trial["trial_number"]))
                    try:
                        score = score_validation_prediction_files(trial_path)
                    except (FileNotFoundError, ValueError) as exc:
                        missing_rows.append(
                            {
                                "task_id": task_id,
                                "model_family": model_family,
                                "model": model,
                                "outer_fold_id": outer_fold_id,
                                "candidate_trial_number": int(trial["trial_number"]),
                                "parameter_hash": str(trial["parameter_hash"]),
                                "missing_reason": "missing_existing_validation_predictions",
                                "source_path": str(trial_path),
                                "message": str(exc),
                            }
                        )
                        continue
                    validation_mae = float(score["mae"])
                    validation_rmse = float(score["rmse"])
                    validation_r2 = float(score["r2"])
                    score_source = json.dumps(score["source_paths"], ensure_ascii=True)

                effective = _effective_parameters(
                    template,
                    kind,
                    trial["search_parameters"],
                    trial["model_config"],
                )
                trial_rows.append(
                    {
                        "task_id": task_id,
                        "model_family": model_family,
                        "model": model,
                        "outer_fold_id": outer_fold_id,
                        "candidate_trial_number": int(trial["trial_number"]),
                        "parameter_hash": str(trial["parameter_hash"]),
                        "validation_mae": validation_mae,
                        "validation_rmse": validation_rmse,
                        "validation_r2": validation_r2,
                        "search_parameters_json": json.dumps(
                            trial["search_parameters"], ensure_ascii=True, sort_keys=True
                        ),
                        "effective_parameters_json": json.dumps(effective, ensure_ascii=True, sort_keys=True),
                        "score_source_path": score_source,
                        "model_dir": str(model_dir),
                    }
                )

    trial_frame = pd.DataFrame(trial_rows)
    selected, summary = aggregate_existing_parameter_scores(trial_frame) if not trial_frame.empty else (pd.DataFrame(), pd.DataFrame())
    return trial_frame, summary, pd.DataFrame(missing_rows), pd.DataFrame(fixed_rows)


def build_task_model_frozen_entries(
    selected: pd.DataFrame,
    fixed_runtime: pd.DataFrame,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        entry = {
            "task_id": str(row["task_id"]),
            "model_family": str(row["model_family"]),
            "model": str(row["model"]),
            "parameter_hash": str(row["parameter_hash"]),
            "search_parameters": _json_dict(row.get("search_parameters_json", "{}")),
            "effective_parameters": _json_dict(row["effective_parameters_json"]),
            "selection_method": str(row.get("selection_method", "five_fold_randomcv_validation")),
            "selection_metric": str(row.get("selection_metric", "mean_randomcv_mae")),
            "mean_mae": float(row["mean_mae"]),
            "sd_mae": float(row["sd_mae"]),
            "mean_r2": float(row["mean_r2"]),
            "source_paths": _json_dict(row.get("source_paths_json", "{}")),
        }
        if pd.notna(row.get("observed_fold_count", np.nan)):
            entry["observed_fold_count"] = int(row["observed_fold_count"])
            entry["observed_fold_ids"] = json.loads(str(row.get("observed_fold_ids_json", "[]")))
            entry["representative_outer_fold_id"] = str(row.get("representative_outer_fold_id", ""))
            entry["representative_trial_number"] = int(row.get("representative_trial_number", -1))
        entries.append(entry)
    for _, row in fixed_runtime.iterrows():
        sources_value = row.get("source_paths_json", "[]")
        try:
            sources = json.loads(str(sources_value)) if str(sources_value).strip() else []
        except json.JSONDecodeError:
            sources = []
        entries.append(
            {
                "task_id": str(row["task_id"]),
                "model_family": str(row["model_family"]),
                "model": str(row["model"]),
                "parameter_hash": str(row["parameter_hash"]),
                "effective_parameters": _json_dict(row["effective_parameters_json"]),
                "selection_method": "fixed_runtime",
                "selection_metric": "not_applicable",
                "source_paths": sources,
            }
        )
    return sorted(entries, key=lambda entry: tuple(str(entry[name]) for name in TASK_MODEL_KEY))


def _is_randomcv_source_path(value: object) -> bool:
    normalized = str(value).replace("\\", "/").lower()
    return "/random_cv_baseline/" in normalized or "/randomcv/" in normalized


def _preserve_missing_fixed_runtime_rows(
    audit: pd.DataFrame,
    fixed_runtime: pd.DataFrame,
    snapshot_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    if not snapshot_path.exists():
        return fixed_runtime, []

    eligible = audit[
        audit["protocol"].astype(str).str.lower().eq("randomcv")
        & ~audit["model"].astype(str).isin(EXCLUDED_MODELS)
    ].copy()
    eligible = eligible[
        eligible.apply(lambda row: _model_kind(row["model_family"], row["model"]) == "tabpfn", axis=1)
    ]
    eligible_keys = {
        (_normalize_task_id(row["task_id"]), str(row["model_family"]), str(row["model"]))
        for _, row in eligible.iterrows()
    }
    present_keys = {
        (str(row["task_id"]), str(row["model_family"]), str(row["model"]))
        for _, row in fixed_runtime.iterrows()
    }
    if not eligible_keys - present_keys:
        return fixed_runtime, []

    existing = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    if str(existing.get("version", "")) != "v1":
        raise ValueError(f"Existing fixed-runtime snapshot must declare version v1: {snapshot_path}")
    preserved_rows: list[dict[str, Any]] = []
    preserved_keys: list[str] = []
    for entry in existing.get("entries", []):
        if not isinstance(entry, dict) or str(entry.get("selection_method")) != "fixed_runtime":
            continue
        key = tuple(str(entry.get(name, "")) for name in TASK_MODEL_KEY)
        if key not in eligible_keys or key in present_keys:
            continue
        source_paths = entry.get("source_paths", [])
        if not isinstance(source_paths, list) or not source_paths:
            raise ValueError(f"Existing fixed-runtime entry has no RandomCV source paths: {key}")
        if not all(_is_randomcv_source_path(path) for path in source_paths):
            raise ValueError(f"Existing fixed-runtime entry contains a non-RandomCV source path: {key}")
        effective = entry.get("effective_parameters", {})
        if not isinstance(effective, dict) or not effective:
            raise ValueError(f"Existing fixed-runtime entry has no effective parameters: {key}")
        preserved_rows.append(
            {
                "task_id": key[0],
                "model_family": key[1],
                "model": key[2],
                "parameter_hash": str(entry.get("parameter_hash", "")),
                "effective_parameters_json": json.dumps(effective, ensure_ascii=True, sort_keys=True),
                "selection_status": "fixed_runtime",
                "source_paths_json": json.dumps(source_paths, ensure_ascii=True),
            }
        )
        preserved_keys.append("||".join(key))

    if not preserved_rows:
        return fixed_runtime, []
    combined = pd.concat([fixed_runtime, pd.DataFrame(preserved_rows)], ignore_index=True)
    combined = combined.sort_values(TASK_MODEL_KEY, kind="stable").drop_duplicates(TASK_MODEL_KEY, keep="first")
    return combined.reset_index(drop=True), sorted(preserved_keys)


def _snapshot_entry_map(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        normalized = deepcopy(entry)
        normalized.pop("parameter_snapshot_id", None)
        key = "||".join(str(normalized.get(name, "")) for name in TASK_MODEL_KEY)
        mapped[key] = _canonicalize(normalized)
    return mapped


def _snapshot_diff_report(
    existing: dict[str, Any],
    candidate_entries: list[dict[str, Any]],
    candidate_snapshot_id: str,
) -> dict[str, Any]:
    existing_map = _snapshot_entry_map(
        [entry for entry in existing.get("entries", []) if isinstance(entry, dict)]
    )
    candidate_map = _snapshot_entry_map(candidate_entries)
    existing_keys = set(existing_map)
    candidate_keys = set(candidate_map)
    changed = sorted(
        key for key in existing_keys & candidate_keys if existing_map[key] != candidate_map[key]
    )
    return {
        "schema_version": 1,
        "existing_snapshot_id": str(existing.get("snapshot_id", "")),
        "candidate_snapshot_id": candidate_snapshot_id,
        "added_keys": sorted(candidate_keys - existing_keys),
        "removed_keys": sorted(existing_keys - candidate_keys),
        "changed_keys": changed,
    }


def _approved_tabpfn_api25_snapshot_migration(
    cfg: CalibrationConfig,
    candidate_entries: list[dict[str, Any]],
    candidate_snapshot_id: str,
) -> bool:
    repair_root = cfg.project_root / "output" / "ood_repair" / "tabpfn_v1_refresh"
    manifest_path = repair_root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("verification_complete"):
            return False
        if str(manifest.get("snapshot_id", "")) != candidate_snapshot_id:
            return False
        candidate_path = Path(
            str(
                manifest.get(
                    "snapshot_path",
                    repair_root / "frozen_model_params_v1_api25_candidate.yaml",
                )
            )
        )
        if not candidate_path.is_absolute():
            candidate_path = cfg.project_root / candidate_path
        payload = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return False
    if str(payload.get("snapshot_id", "")) != candidate_snapshot_id:
        return False
    locked_entries = [
        entry for entry in payload.get("entries", []) if isinstance(entry, dict)
    ]
    return _snapshot_entry_map(locked_entries) == _snapshot_entry_map(candidate_entries)


def select_best_parameter_rows(score_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(TASK_MODEL_KEY + ["parameter_hash", "effective_parameters_json"])
    required.update(f"mae_fold_{fold}" for fold in range(FOLD_COUNT))
    required.update(f"r2_fold_{fold}" for fold in range(FOLD_COUNT))
    missing = sorted(required - set(score_matrix.columns))
    if missing:
        raise ValueError(f"Parameter score matrix is missing columns: {missing}")

    scored = score_matrix.copy()
    mae_columns = [f"mae_fold_{fold}" for fold in range(FOLD_COUNT)]
    r2_columns = [f"r2_fold_{fold}" for fold in range(FOLD_COUNT)]
    scored["completed_fold_count"] = scored[mae_columns].notna().sum(axis=1)
    scored["mean_mae"] = scored[mae_columns].mean(axis=1)
    scored["sd_mae"] = scored[mae_columns].std(axis=1, ddof=1)
    scored["mean_r2"] = scored[r2_columns].mean(axis=1)
    scored["selection_status"] = np.where(
        scored["completed_fold_count"].eq(FOLD_COUNT),
        "eligible",
        "incomplete_folds",
    )

    eligible = scored[scored["selection_status"].eq("eligible")].copy()
    if eligible.empty:
        return eligible, scored
    eligible = eligible.sort_values(
        TASK_MODEL_KEY + ["mean_mae", "sd_mae", "mean_r2", "parameter_hash"],
        ascending=[True, True, True, True, True, False, True],
        kind="stable",
    )
    selected = eligible.groupby(TASK_MODEL_KEY, as_index=False, sort=False).head(1).copy()
    selected["selection_status"] = "selected"
    selected_keys = {
        tuple(str(row[name]) for name in TASK_MODEL_KEY + ["parameter_hash"])
        for _, row in selected.iterrows()
    }
    eligible_mask = scored["selection_status"].eq("eligible")
    identities = scored.apply(
        lambda row: tuple(str(row[name]) for name in TASK_MODEL_KEY + ["parameter_hash"]),
        axis=1,
    )
    scored.loc[eligible_mask, "selection_status"] = "not_selected"
    scored.loc[eligible_mask & identities.isin(selected_keys), "selection_status"] = "selected"
    return selected.reset_index(drop=True), scored.reset_index(drop=True)


def aggregate_existing_parameter_scores(trial_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(
        TASK_MODEL_KEY
        + [
            "parameter_hash",
            "outer_fold_id",
            "candidate_trial_number",
            "validation_mae",
            "validation_r2",
            "effective_parameters_json",
            "search_parameters_json",
            "score_source_path",
        ]
    )
    missing = sorted(required - set(trial_scores.columns))
    if missing:
        raise ValueError(f"Existing RandomCV trial scores are missing columns: {missing}")
    if trial_scores.empty:
        return pd.DataFrame(), pd.DataFrame()

    scores = trial_scores.copy()
    scores["validation_mae"] = pd.to_numeric(scores["validation_mae"], errors="coerce")
    scores["validation_r2"] = pd.to_numeric(scores["validation_r2"], errors="coerce")
    scores = scores[scores["validation_mae"].notna()].copy()
    scores = scores.sort_values(
        TASK_MODEL_KEY + ["parameter_hash", "outer_fold_id", "validation_mae", "candidate_trial_number"],
        kind="stable",
    )
    # Repeated trials with the same parameters in one outer fold count once.
    per_fold = scores.drop_duplicates(
        TASK_MODEL_KEY + ["parameter_hash", "outer_fold_id"],
        keep="first",
    )

    summary_rows: list[dict[str, Any]] = []
    group_columns = TASK_MODEL_KEY + ["parameter_hash"]
    for key, group in per_fold.groupby(group_columns, sort=True):
        representative = group.sort_values(
            ["validation_mae", "outer_fold_id", "candidate_trial_number"], kind="stable"
        ).iloc[0]
        fold_mae = group["validation_mae"].to_numpy(dtype=float)
        fold_r2 = group["validation_r2"].to_numpy(dtype=float)
        source_paths = {
            str(row["outer_fold_id"]): str(row["score_source_path"])
            for _, row in group.sort_values("outer_fold_id", kind="stable").iterrows()
        }
        row = {
            **{name: str(value) for name, value in zip(group_columns, key)},
            "observed_fold_count": int(group["outer_fold_id"].nunique()),
            "observed_fold_ids_json": json.dumps(
                sorted(group["outer_fold_id"].astype(str).unique().tolist()), ensure_ascii=True
            ),
            "mean_mae": float(np.mean(fold_mae)),
            "sd_mae": float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 else np.nan,
            "mean_r2": float(np.nanmean(fold_r2)) if np.isfinite(fold_r2).any() else np.nan,
            "representative_outer_fold_id": str(representative["outer_fold_id"]),
            "representative_trial_number": int(representative["candidate_trial_number"]),
            "effective_parameters_json": str(representative["effective_parameters_json"]),
            "search_parameters_json": str(representative["search_parameters_json"]),
            "source_paths_json": json.dumps(source_paths, ensure_ascii=True, sort_keys=True),
            "selection_status": "eligible_existing",
            "selection_method": "existing_randomcv_validation_only",
            "selection_metric": "mean_available_fold_validation_mae",
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["sd_mae_sort"] = summary["sd_mae"].fillna(np.inf)
    summary["mean_r2_sort"] = summary["mean_r2"].fillna(-np.inf)
    ranked = summary.sort_values(
        TASK_MODEL_KEY + ["mean_mae", "sd_mae_sort", "mean_r2_sort", "observed_fold_count", "parameter_hash"],
        ascending=[True, True, True, True, True, False, False, True],
        kind="stable",
    )
    selected = ranked.groupby(TASK_MODEL_KEY, as_index=False, sort=False).head(1).copy()
    selected["selection_status"] = "selected"
    selected_keys = {
        tuple(str(row[name]) for name in TASK_MODEL_KEY + ["parameter_hash"])
        for _, row in selected.iterrows()
    }
    identities = summary.apply(
        lambda row: tuple(str(row[name]) for name in TASK_MODEL_KEY + ["parameter_hash"]), axis=1
    )
    summary.loc[identities.isin(selected_keys), "selection_status"] = "selected"
    return (
        selected.drop(columns=["sd_mae_sort", "mean_r2_sort"], errors="ignore").reset_index(drop=True),
        summary.drop(columns=["sd_mae_sort", "mean_r2_sort"], errors="ignore").reset_index(drop=True),
    )


def validate_frozen_task_model_entries(
    entries: list[dict[str, Any]],
    *,
    expected_entry_count: int = EXPECTED_TASK_MODEL_ENTRIES,
) -> None:
    if len(entries) != expected_entry_count:
        raise ValueError(f"Frozen task/model registry must contain {expected_entry_count} entries, found {len(entries)}")
    keys = [tuple(str(entry.get(name, "")) for name in TASK_MODEL_KEY) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen task/model registry contains duplicate task/model keys.")
    if any(not str(entry.get("parameter_hash", "")) for entry in entries):
        raise ValueError("Every frozen task/model entry must contain a parameter_hash.")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def _snapshot_id(entries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        _canonicalize(entries),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_randomcv_parameter_selection(
    cfg: CalibrationConfig,
    audit: pd.DataFrame,
    *,
    expected_entry_count: int = EXPECTED_TASK_MODEL_ENTRIES,
) -> dict[str, Any]:
    output_dir = cfg.output_root / "parameter_freeze"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, matrix, missing, fixed = build_randomcv_parameter_tables(audit)
    selected = (
        matrix[matrix["selection_status"].astype(str).eq("selected")].copy()
        if not matrix.empty
        else pd.DataFrame()
    )
    entries = build_task_model_frozen_entries(selected, fixed)

    _atomic_write_csv(output_dir / "randomcv_parameter_candidates.csv", candidates)
    _atomic_write_csv(output_dir / "parameter_fold_score_matrix.csv", matrix)
    _atomic_write_csv(output_dir / "missing_parameter_fold_scores.csv", missing)

    summary_columns = [
        "task_id",
        "model_family",
        "model",
        "parameter_hash",
        "selection_status",
        "mean_mae",
        "sd_mae",
        "mean_r2",
        "effective_parameters_json",
        "source_paths_json",
    ]
    selected_summary = selected.copy()
    fixed_summary = fixed.copy()
    for frame in (selected_summary, fixed_summary):
        for column in summary_columns:
            if column not in frame.columns:
                frame[column] = np.nan
    summary = pd.concat(
        [selected_summary[summary_columns], fixed_summary[summary_columns]],
        ignore_index=True,
    )
    if not summary.empty:
        summary = summary.sort_values(TASK_MODEL_KEY, kind="stable")
    _atomic_write_csv(output_dir / "best_model_parameter_summary.csv", summary)

    generated_at = datetime.now(timezone.utc).isoformat()
    complete = len(entries) == expected_entry_count and missing.empty
    snapshot_path = output_dir / "frozen_model_params_v1.yaml"
    snapshot_id = ""
    if complete:
        validate_frozen_task_model_entries(entries, expected_entry_count=expected_entry_count)
        snapshot_id = _snapshot_id(entries)
        for entry in entries:
            entry["parameter_snapshot_id"] = snapshot_id
        payload = {
            "schema_version": 2,
            "version": "v1",
            "snapshot_id": snapshot_id,
            "generated_at": generated_at,
            "source_config": str(cfg.path),
            "key_fields": TASK_MODEL_KEY,
            "selection_metric": "mean_randomcv_mae",
            "entry_count": len(entries),
            "entries": entries,
        }
        if snapshot_path.exists():
            existing = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
            existing_id = str(existing.get("snapshot_id", "unknown"))
            if existing_id != snapshot_id:
                history_dir = output_dir / "history"
                history_dir.mkdir(parents=True, exist_ok=True)
                history_path = history_dir / f"frozen_model_params_v1_{existing_id}_legacy.yaml"
                if not history_path.exists():
                    shutil.copy2(snapshot_path, history_path)
        _atomic_write_text(snapshot_path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))

    manifest = {
        "schema_version": 2,
        "stage": "randomcv-parameter-selection",
        "generated_at": generated_at,
        "config": str(cfg.path),
        "selection_status": "complete" if complete else "incomplete",
        "expected_entry_count": expected_entry_count,
        "selected_tuned_entry_count": int(len(selected)),
        "fixed_runtime_entry_count": int(len(fixed)),
        "candidate_count": int(len(candidates)),
        "missing_parameter_fold_score_count": int(len(missing)),
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_path) if complete else "",
        "outputs": {
            "candidates": str(output_dir / "randomcv_parameter_candidates.csv"),
            "score_matrix": str(output_dir / "parameter_fold_score_matrix.csv"),
            "summary": str(output_dir / "best_model_parameter_summary.csv"),
            "missing": str(output_dir / "missing_parameter_fold_scores.csv"),
        },
    }
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
    )
    return manifest


def write_existing_randomcv_parameter_selection(
    cfg: CalibrationConfig,
    audit: pd.DataFrame,
    *,
    expected_entry_count: int = EXPECTED_TASK_MODEL_ENTRIES,
) -> dict[str, Any]:
    output_dir = cfg.output_root / "parameter_freeze"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "frozen_model_params_v1.yaml"
    trial_scores, parameter_summary, missing, fixed = build_existing_randomcv_parameter_tables(audit)
    selected = (
        parameter_summary[parameter_summary["selection_status"].astype(str).eq("selected")].copy()
        if not parameter_summary.empty
        else pd.DataFrame()
    )
    fixed, preserved_fixed_runtime_keys = _preserve_missing_fixed_runtime_rows(
        audit,
        fixed,
        snapshot_path,
    )
    entries = build_task_model_frozen_entries(selected, fixed)

    _atomic_write_csv(output_dir / "existing_randomcv_trial_scores.csv", trial_scores)
    _atomic_write_csv(output_dir / "existing_parameter_hash_summary.csv", parameter_summary)
    _atomic_write_csv(output_dir / "missing_existing_parameter_scores.csv", missing)

    summary_columns = [
        "task_id",
        "model_family",
        "model",
        "parameter_hash",
        "selection_status",
        "mean_mae",
        "sd_mae",
        "mean_r2",
        "observed_fold_count",
        "observed_fold_ids_json",
        "representative_outer_fold_id",
        "representative_trial_number",
        "effective_parameters_json",
        "source_paths_json",
    ]
    selected_summary = selected.copy()
    fixed_summary = fixed.copy()
    for frame in (selected_summary, fixed_summary):
        for column in summary_columns:
            if column not in frame.columns:
                frame[column] = np.nan
    best_summary = pd.concat(
        [selected_summary[summary_columns], fixed_summary[summary_columns]], ignore_index=True
    )
    if not best_summary.empty:
        best_summary = best_summary.sort_values(TASK_MODEL_KEY, kind="stable")
    _atomic_write_csv(output_dir / "best_model_parameter_summary.csv", best_summary)

    generated_at = datetime.now(timezone.utc).isoformat()
    complete = len(entries) == expected_entry_count
    snapshot_id = ""
    approved_tabpfn_api25_migration = False
    if complete:
        validate_frozen_task_model_entries(entries, expected_entry_count=expected_entry_count)
        snapshot_id = _snapshot_id(entries)
        for entry in entries:
            entry["parameter_snapshot_id"] = snapshot_id
        payload = {
            "schema_version": 2,
            "version": "v1",
            "snapshot_id": snapshot_id,
            "generated_at": generated_at,
            "source_config": str(cfg.path),
            "key_fields": TASK_MODEL_KEY,
            "selection_mode": "existing_results_only",
            "selection_metric": "mean_available_fold_validation_mae",
            "entry_count": len(entries),
            "entries": entries,
        }
        if snapshot_path.exists():
            existing = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
            existing_id = str(existing.get("snapshot_id", "unknown"))
            if existing_id != snapshot_id:
                diff_path = output_dir / "frozen_model_params_v1_diff.json"
                diff = _snapshot_diff_report(existing, entries, snapshot_id)
                _atomic_write_text(diff_path, json.dumps(diff, indent=2, ensure_ascii=True) + "\n")
                approved_tabpfn_api25_migration = _approved_tabpfn_api25_snapshot_migration(
                    cfg,
                    entries,
                    snapshot_id,
                )
                if not approved_tabpfn_api25_migration:
                    raise ValueError(
                        "Refusing to overwrite frozen v1 with changed parameter content: "
                        f"existing={existing_id}, candidate={snapshot_id}, diff={diff_path}"
                    )
                history_dir = output_dir / "history"
                history_dir.mkdir(parents=True, exist_ok=True)
                history_path = history_dir / f"frozen_model_params_v1_{existing_id}_pre_api25.yaml"
                if not history_path.exists():
                    shutil.copy2(snapshot_path, history_path)
        _atomic_write_text(snapshot_path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))

    manifest = {
        "schema_version": 2,
        "stage": "existing-randomcv-parameter-selection",
        "generated_at": generated_at,
        "config": str(cfg.path),
        "selection_status": "complete" if complete else "incomplete",
        "selection_mode": "existing_results_only",
        "expected_entry_count": expected_entry_count,
        "selected_tuned_entry_count": int(len(selected)),
        "fixed_runtime_entry_count": int(len(fixed)),
        "preserved_fixed_runtime_entry_count": len(preserved_fixed_runtime_keys),
        "preserved_fixed_runtime_keys": preserved_fixed_runtime_keys,
        "approved_tabpfn_api25_migration": approved_tabpfn_api25_migration,
        "existing_trial_score_count": int(len(trial_scores)),
        "parameter_hash_summary_count": int(len(parameter_summary)),
        "missing_existing_trial_score_count": int(len(missing)),
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_path) if complete else "",
        "outputs": {
            "trial_scores": str(output_dir / "existing_randomcv_trial_scores.csv"),
            "parameter_summary": str(output_dir / "existing_parameter_hash_summary.csv"),
            "summary": str(output_dir / "best_model_parameter_summary.csv"),
            "missing": str(output_dir / "missing_existing_parameter_scores.csv"),
        },
    }
    _atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    return manifest
