from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .confidence import TIER_ORDER
from .metrics import mean_ci95
from .parameter_registry import FrozenParameterRegistry
from .schemas import STANDARD_SAMPLE_COLUMNS, CalibrationConfig


TI_TASKS = ("Ti__titanium__El", "Ti__titanium__UTS_MPa")
TI_MODELS = (
    "CatBoost",
    "LightGBM",
    "MLP",
    "RF",
    "XGB",
    "MatSciBERT",
    "SciBERT",
    "SteelBERT",
    "LLM-Prop",
    "TabPFN-2.5-Plus-Numeric",
    "TabPFN-2.5-Plus-Text",
)
TI_PROTOCOL_ORDER = (
    "Extrapolation",
    "LOCO",
    "RandomCV",
    "SparseXsingle",
    "SparseXcluster",
    "SparseYsingle",
    "SparseYcluster",
)

SOURCE_INDEX_COLUMNS = [
    "parameter_snapshot_id",
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "frozen_entry_key",
    "train_split_hash",
    "test_split_hash",
    "uncertainty_method",
    "selected_trial_id",
    "member_prediction_files",
    "checkpoint_path",
    "train_split_path",
    "test_split_path",
    "source_validation_status",
    "source_validation_message",
]

POOLED_GROUPS = ["task_id", "protocol", "model_family", "model"]
MEMBER_COLUMNS = [
    "task_id",
    "protocol",
    "split_id",
    "model_family",
    "model",
    "sample_id",
    "member_id",
    "y_true",
    "y_pred",
    "source_path",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(str(path))
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return bool(np.isclose(float(left), float(right), rtol=1e-9, atol=1e-12, equal_nan=True))
    return _canonical(left) == _canonical(right)


def _effective_source_parameters(entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    family = str(entry["model_family"]).lower()
    model = str(entry["model"]).lower()
    if family == "traditional":
        params = payload.get("model_params")
        return {"model_type": payload.get("model_type"), "model_params": params} if isinstance(params, dict) else {}
    if "bert" in family or "bert" in model:
        return payload
    if "llmprop" in family or "llm-prop" in model:
        training = payload.get("training_hyperparameters", {})
        if not isinstance(training, dict):
            training = {}
        excluded = {"result_dir", "train_csv", "test_csv", "split_manifest"}
        effective = {key: value for key, value in training.items() if key not in excluded}
        for key in ("base_model_path", "tokenizer_path", "normalizer"):
            if key in payload:
                effective[key] = payload[key]
        return effective
    if "tabpfn" in family or "tabpfn" in model:
        keys = (
            "model_name",
            "model_dirname",
            "feature_mode",
            "requested_backend",
            "resolved_backend",
            "random_state",
            "feature_cols",
            "model_info",
        )
        return {key: payload[key] for key in keys if key in payload}
    return payload


def csv_identity_hash(path: str | Path) -> str:
    source = Path(path)
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    id_column = next(
        (name for name in ("ID", "id", "sample_id", "__row_id__", "__source_index__") if name in columns),
        None,
    )
    if id_column is None:
        raise ValueError(f"No stable sample ID column in split file: {source}")
    values = pd.read_csv(source, usecols=[id_column])[id_column].astype(str)
    canonical = "\n".join(values.tolist()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def select_ti_entries(registry: FrozenParameterRegistry) -> list[dict[str, Any]]:
    task_ids = set(TI_TASKS)
    models = set(TI_MODELS)
    return [
        entry
        for entry in registry.iter_run_entries()
        if str(entry.get("task_id")) in task_ids and str(entry.get("model")) in models
    ]


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    sample_col = next((name for name in ("ID", "id", "sample_id", "__row_id__") if name in frame.columns), None)
    true_col = next(
        (str(name) for name in frame.columns if str(name).lower() in {"y_true", "actual", "actual_value"}),
        None,
    )
    pred_col = next(
        (str(name) for name in frame.columns if str(name).lower() in {"y_pred", "prediction", "predicted", "predicted_value"}),
        None,
    )
    if true_col is None:
        true_col = next((str(name) for name in frame.columns if str(name).lower().endswith("_actual")), None)
    if pred_col is None:
        pred_col = next((str(name) for name in frame.columns if str(name).lower().endswith("_predicted")), None)
    if sample_col is None or true_col is None or pred_col is None:
        raise ValueError("Prediction file requires sample ID, actual, and predicted columns.")
    return sample_col, true_col, pred_col


def _evidence_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("best_params")
    return nested if isinstance(nested, dict) else payload


def match_optuna_trial(history_path: str | Path, evidence_parameters: dict[str, Any]) -> int:
    available_columns = pd.read_csv(history_path, nrows=0).columns.tolist()
    required_columns = [
        name
        for name in available_columns
        if name in {"number", "state"} or name.startswith("params_")
    ]
    history = pd.read_csv(history_path, usecols=required_columns)
    if "state" in history.columns:
        history = history[history["state"].astype(str).str.upper() == "COMPLETE"].copy()
    if history.empty:
        raise ValueError(f"No COMPLETE Optuna trials found: {history_path}")

    evidence = _evidence_parameters(evidence_parameters)
    if not evidence:
        raise ValueError("No parameter evidence is available for Optuna trial matching.")
    mask = pd.Series(True, index=history.index)
    for name, expected in evidence.items():
        column = f"params_{name}"
        if column not in history.columns:
            raise ValueError(f"Optuna history is missing evidence column: {column}")
        current = history[column]
        if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
            numeric = pd.to_numeric(current, errors="coerce")
            mask &= np.isclose(
                numeric.to_numpy(dtype=float),
                float(expected),
                rtol=1e-9,
                atol=1e-12,
                equal_nan=False,
            )
        elif isinstance(expected, (list, tuple, dict)):
            def parse_structured(value: Any) -> Any:
                if not isinstance(value, str):
                    return value
                try:
                    return ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return value

            mask &= current.map(lambda value: _canonical(parse_structured(value)) == _canonical(expected))
        else:
            mask &= current.astype(str) == str(expected)

    matches = history.loc[mask]
    if matches.empty:
        raise ValueError("No COMPLETE Optuna trial matches the frozen parameter evidence.")
    if len(matches) != 1:
        numbers = matches.get("number", matches.index).tolist()
        raise ValueError(f"Frozen parameters do not identify a unique Optuna trial: {numbers}")
    return int(matches.iloc[0]["number"] if "number" in matches.columns else matches.index[0])


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.parent.name if path.name == "all_predictions.csv" else path.name)
    return (int(match.group(1)) if match else 10**9, str(path))


def load_trial_member_predictions(entry: dict[str, Any], trial_dir: str | Path) -> pd.DataFrame:
    directory = Path(trial_dir)
    files = sorted(directory.glob("fold_*/all_predictions.csv"), key=_numeric_sort_key)
    if len(files) < 2:
        raise ValueError(f"At least two fold prediction files are required: {directory}")

    frames: list[pd.DataFrame] = []
    reference: pd.DataFrame | None = None
    for path in files:
        raw = pd.read_csv(path)
        if "Dataset" in raw.columns:
            raw = raw[raw["Dataset"].astype(str).str.lower() == "test"].copy()
        if raw.empty:
            raise ValueError(f"No Test predictions found: {path}")
        sample_col, true_col, pred_col = _prediction_columns(raw)
        member = pd.DataFrame(
            {
                "sample_id": raw[sample_col],
                "y_true": pd.to_numeric(raw[true_col], errors="coerce"),
                "y_pred": pd.to_numeric(raw[pred_col], errors="coerce"),
            }
        )
        if member[["sample_id", "y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"Non-numeric or missing Test prediction values: {path}")
        if member["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate Test sample IDs in member prediction file: {path}")
        member = member.sort_values("sample_id", key=lambda values: values.astype(str)).reset_index(drop=True)
        if reference is None:
            reference = member[["sample_id", "y_true"]].copy()
        else:
            if member["sample_id"].astype(str).tolist() != reference["sample_id"].astype(str).tolist():
                raise ValueError("Ensemble member Test IDs do not match.")
            if not np.allclose(member["y_true"], reference["y_true"], rtol=1e-10, atol=1e-12):
                raise ValueError("Ensemble member y_true values do not match.")
        member["member_id"] = path.parent.name
        member["source_path"] = str(path.resolve())
        for name in ("task_id", "protocol", "split_id", "model_family", "model"):
            member[name] = str(entry[name])
        frames.append(member[MEMBER_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def aggregate_member_predictions(
    members: pd.DataFrame,
    parameter_snapshot_id: str,
    uncertainty_method: str = "member_prediction_sd_ddof1",
) -> pd.DataFrame:
    group_columns = ["task_id", "protocol", "split_id", "model_family", "model", "sample_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in members.groupby(group_columns, dropna=False):
        truth = pd.to_numeric(group["y_true"], errors="coerce")
        predictions = pd.to_numeric(group["y_pred"], errors="coerce")
        if truth.isna().any() or predictions.isna().any():
            raise ValueError("Member predictions contain missing numeric values.")
        if not np.allclose(truth, truth.iloc[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Ensemble member y_true values do not match.")
        if len(predictions) < 2:
            raise ValueError("At least two member predictions are required for sample SD.")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "y_true": float(truth.iloc[0]),
                "y_pred": float(predictions.mean()),
                "uncertainty_score": float(predictions.std(ddof=1)),
                "member_count": int(len(predictions)),
                "uncertainty_method": uncertainty_method,
                "parameter_snapshot_id": parameter_snapshot_id,
                "source_path": ";".join(sorted(set(group["source_path"].astype(str)))),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["abs_error"] = (result["y_pred"] - result["y_true"]).abs()
    result["confidence_tier"] = pd.NA
    extra = ["member_count", "uncertainty_method"]
    return result[STANDARD_SAMPLE_COLUMNS + extra]


def assign_ranked_tertiles(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["confidence_tier"] = pd.NA
    output["_stable_index"] = np.arange(len(output))
    for _, group in output.groupby(POOLED_GROUPS, dropna=False, sort=False):
        valid = group[pd.to_numeric(group["uncertainty_score"], errors="coerce").notna()].copy()
        if valid.empty:
            continue
        valid["_uncertainty_numeric"] = pd.to_numeric(valid["uncertainty_score"], errors="coerce")
        valid["_split_sort"] = valid["split_id"].astype(str)
        valid["_sample_sort"] = valid["sample_id"].astype(str)
        ordered = valid.sort_values(
            ["_uncertainty_numeric", "_split_sort", "_sample_sort", "_stable_index"],
            kind="mergesort",
        )
        for tier, positions in zip(TIER_ORDER, np.array_split(np.arange(len(ordered)), 3)):
            output.loc[ordered.iloc[positions].index, "confidence_tier"] = tier
    output = output.drop(columns=["_stable_index"])
    output["confidence_tier"] = pd.Categorical(output["confidence_tier"], categories=TIER_ORDER, ordered=True)
    return output


def enable_mc_dropout(model: Any) -> None:
    import torch.nn as nn

    model.eval()
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


def _quantile_arrays(raw: Any, expected_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(raw, (list, tuple)):
        arrays = [np.asarray(item, dtype=float).reshape(-1) for item in raw]
        if len(arrays) != 3:
            raise ValueError("TabPFN quantile output must contain q10, q50, and q90.")
        q10, q50, q90 = arrays
    else:
        array = np.asarray(raw, dtype=float)
        if array.shape == (3, expected_n):
            q10, q50, q90 = array
        elif array.shape == (expected_n, 3):
            q10, q50, q90 = array.T
        else:
            raise ValueError(f"Unexpected TabPFN quantile output shape: {array.shape}")
    if any(len(item) != expected_n for item in (q10, q50, q90)):
        raise ValueError("TabPFN quantile output length does not match the test set.")
    return q10, q50, q90


def run_tabpfn_quantile_prediction(model: Any, X_train: Any, y_train: Any, X_test: Any) -> tuple[np.ndarray, np.ndarray]:
    model.fit(X_train, y_train)
    mean = np.asarray(model.predict(X_test, output_type="mean"), dtype=float).reshape(-1)
    q10, _, q90 = _quantile_arrays(
        model.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9]),
        expected_n=len(mean),
    )
    width = q90 - q10
    if np.any(width < -1e-12):
        raise ValueError("TabPFN q90 is below q10 for at least one sample.")
    return mean, np.maximum(width, 0.0)


def _split_paths(entry: dict[str, Any]) -> tuple[Path | None, Path | None, dict[str, Any]]:
    source = entry.get("source", {}) or {}
    config_path = _safe_path(source.get("parameter_config"))
    payload = _json_load(config_path)
    family = str(entry["model_family"]).lower()
    model = str(entry["model"]).lower()
    if "llmprop" in family or "llm-prop" in model:
        return _safe_path(payload.get("train_csv")), _safe_path(payload.get("test_csv")), payload
    if "tabpfn" in family or "tabpfn" in model:
        artifacts = payload.get("split_artifacts", {}) if isinstance(payload.get("split_artifacts"), dict) else {}
        return _safe_path(artifacts.get("train_file")), _safe_path(artifacts.get("test_file")), payload

    model_dir = _safe_path(source.get("model_dir"))
    manifest = _json_load(model_dir / "ood_manifest.json" if model_dir else None)
    return _safe_path(manifest.get("train_feature_file")), _safe_path(manifest.get("test_feature_file")), payload


def _source_hashes(
    entry: dict[str, Any],
    train_path: Path | None,
    test_path: Path | None,
    payload: dict[str, Any],
) -> tuple[str, str]:
    family = str(entry["model_family"]).lower()
    model = str(entry["model"]).lower()
    if "llmprop" in family or "llm-prop" in model:
        return str(payload.get("train_split_hash", "") or ""), str(payload.get("test_split_hash", "") or "")
    if train_path is None or test_path is None or not train_path.exists() or not test_path.exists():
        return "", ""
    return csv_identity_hash(train_path), csv_identity_hash(test_path)


def build_source_record(
    entry: dict[str, Any],
    parameter_snapshot_id: str,
    *,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> dict[str, Any]:
    source = entry.get("source", {}) or {}
    model_dir = _safe_path(source.get("model_dir"))
    frozen_key = FrozenParameterRegistry.entry_key(entry)
    record: dict[str, Any] = {
        "parameter_snapshot_id": parameter_snapshot_id,
        "task_id": str(entry["task_id"]),
        "protocol": str(entry["protocol"]),
        "split_id": str(entry["split_id"]),
        "model_family": str(entry["model_family"]),
        "model": str(entry["model"]),
        "frozen_entry_key": frozen_key,
        "train_split_hash": str(entry.get("train_split_hash", "") or ""),
        "test_split_hash": str(entry.get("test_split_hash", "") or ""),
        "uncertainty_method": "",
        "selected_trial_id": "",
        "member_prediction_files": "[]",
        "checkpoint_path": "",
        "train_split_path": "",
        "test_split_path": "",
        "source_validation_status": "invalid",
        "source_validation_message": "",
    }
    errors: list[str] = []
    if model_dir is None or not model_dir.exists():
        errors.append("Frozen model directory does not exist.")

    parameter_path = _safe_path(source.get("parameter_config"))
    parameter_payload = _json_load(parameter_path)
    source_effective = _effective_source_parameters(entry, parameter_payload)
    if not source_effective:
        errors.append("Parameter source is missing or unreadable.")
    elif not _values_match(source_effective, entry.get("effective_parameters", {})):
        errors.append("Source parameters do not match the frozen YAML entry.")

    resolved_train, resolved_test, split_payload = _split_paths(entry)
    resolved_train = Path(train_path) if train_path is not None else resolved_train
    resolved_test = Path(test_path) if test_path is not None else resolved_test
    record["train_split_path"] = str(resolved_train or "")
    record["test_split_path"] = str(resolved_test or "")
    if resolved_train is None or not resolved_train.exists():
        errors.append("Training split file is missing.")
    if resolved_test is None or not resolved_test.exists():
        errors.append("Test split file is missing.")
    actual_train_hash, actual_test_hash = _source_hashes(entry, resolved_train, resolved_test, split_payload)
    if actual_train_hash != record["train_split_hash"]:
        errors.append("Training split hash does not match the frozen YAML entry.")
    if actual_test_hash != record["test_split_hash"]:
        errors.append("Test split hash does not match the frozen YAML entry.")

    family = str(entry["model_family"]).lower()
    model = str(entry["model"]).lower()
    try:
        if family == "traditional" or "bert" in family or ("bert" in model and "tabpfn" not in model):
            evidence_path = _safe_path(source.get("parameter_evidence"))
            evidence = _json_load(evidence_path)
            history_path = model_dir / "optuna_study_history.csv" if model_dir else Path("missing")
            trial_number = match_optuna_trial(history_path, evidence)
            trial_id = f"trial_{trial_number}"
            member_files = sorted(
                (model_dir / "predictions" / "optuna_trials" / trial_id).glob("fold_*/all_predictions.csv"),
                key=_numeric_sort_key,
            ) if model_dir else []
            if len(member_files) != 9:
                errors.append(f"Expected 9 trial Test prediction members, found {len(member_files)}.")
            record["uncertainty_method"] = "inner_cv_member_sd_ddof1"
            record["selected_trial_id"] = trial_id
            record["member_prediction_files"] = json.dumps([str(path.resolve()) for path in member_files], ensure_ascii=True)
        elif "llmprop" in family or "llm-prop" in model:
            checkpoint = _safe_path(parameter_payload.get("best_model_path"))
            if checkpoint is None or not checkpoint.exists():
                errors.append("LLMProp checkpoint is missing.")
            record["uncertainty_method"] = "mc_dropout_sd_ddof1_30_passes"
            record["checkpoint_path"] = str(checkpoint or "")
        elif "tabpfn" in family or "tabpfn" in model:
            effective = entry.get("effective_parameters", {}) or {}
            model_info = effective.get("model_info", {}) if isinstance(effective.get("model_info"), dict) else {}
            if str(effective.get("resolved_backend", "")).lower() != "api":
                errors.append("Ti pilot only accepts the frozen TabPFN API backend.")
            if str(model_info.get("effective_model_version", "")) != "api_2.5_plus":
                errors.append("TabPFN API model version does not match api_2.5_plus.")
            if str(entry["model"]) not in {"TabPFN-2.5-Plus-Numeric", "TabPFN-2.5-Plus-Text"}:
                errors.append("Local or unsupported TabPFN model is outside the Ti pilot scope.")
            record["uncertainty_method"] = "predictive_q90_minus_q10"
        else:
            errors.append("Unsupported Ti pilot model family.")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        errors.append(str(exc))

    record["source_validation_status"] = "valid" if not errors else "invalid"
    record["source_validation_message"] = " ".join(errors)
    return record


def build_uncertainty_source_index(
    registry: FrozenParameterRegistry,
    pilot_root: str | Path,
) -> pd.DataFrame:
    root = Path(pilot_root)
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    entries = select_ti_entries(registry)
    rows = [build_source_record(entry, registry.snapshot_id) for entry in entries]
    index = pd.DataFrame(rows, columns=SOURCE_INDEX_COLUMNS)
    index.to_csv(audit_dir / "uncertainty_source_index.csv", index=False)
    invalid = index[index["source_validation_status"] != "valid"].copy()
    invalid.to_csv(audit_dir / "missing_or_invalid_sources.csv", index=False)
    coverage = build_runtime_coverage(index, set(), set())
    coverage.to_csv(audit_dir / "coverage_report.csv", index=False)
    return index


def source_index_is_current(index: pd.DataFrame, registry: FrozenParameterRegistry) -> bool:
    required = {"parameter_snapshot_id", "frozen_entry_key"}
    if index.empty or not required.issubset(index.columns):
        return False
    if set(index["parameter_snapshot_id"].astype(str)) != {registry.snapshot_id}:
        return False
    expected_keys = {FrozenParameterRegistry.entry_key(entry) for entry in select_ti_entries(registry)}
    actual_keys = set(index["frozen_entry_key"].astype(str))
    return actual_keys == expected_keys and len(index) == len(expected_keys)


def build_runtime_coverage(
    index: pd.DataFrame,
    completed_keys: set[str],
    runtime_error_keys: set[str],
) -> pd.DataFrame:
    working = index.copy()
    working["is_source_valid"] = working["source_validation_status"].eq("valid").astype(int)
    working["is_completed"] = working["frozen_entry_key"].astype(str).isin(completed_keys).astype(int)
    working["is_runtime_error"] = working["frozen_entry_key"].astype(str).isin(runtime_error_keys).astype(int)
    coverage = (
        working.groupby(["task_id", "protocol", "model_family", "model"], dropna=False)
        .agg(
            entry_count=("frozen_entry_key", "size"),
            valid_source_entry_count=("is_source_valid", "sum"),
            completed_entry_count=("is_completed", "sum"),
            runtime_error_count=("is_runtime_error", "sum"),
        )
        .reset_index()
    )
    coverage["invalid_source_entry_count"] = coverage["entry_count"] - coverage["valid_source_entry_count"]
    coverage["pending_entry_count"] = (
        coverage["entry_count"]
        - coverage["completed_entry_count"]
        - coverage["runtime_error_count"]
        - coverage["invalid_source_entry_count"]
    ).clip(lower=0)
    return coverage


def entry_cache_dir(pilot_root: str | Path, entry: dict[str, Any]) -> Path:
    root = Path(pilot_root)
    digest = hashlib.sha256(FrozenParameterRegistry.entry_key(entry).encode("utf-8")).hexdigest()[:12]
    name = "__".join(
        _safe_component(entry[key]) for key in ("task_id", "protocol", "split_id", "model")
    )
    return root / "members" / f"{name}__{digest}"


def collect_cached_samples(
    pilot_root: str | Path,
    entries: Sequence[dict[str, Any]],
    parameter_snapshot_id: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in entries:
        path = entry_cache_dir(pilot_root, entry) / "sample_uncertainty.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        snapshot_ids = set(frame.get("parameter_snapshot_id", pd.Series(dtype=str)).astype(str))
        if snapshot_ids != {parameter_snapshot_id}:
            raise ValueError(f"Cached Ti sample uses a different frozen snapshot: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(
        columns=STANDARD_SAMPLE_COLUMNS + ["member_count", "uncertainty_method"]
    )


def _sample_from_tabpfn(
    entry: dict[str, Any],
    snapshot_id: str,
    sample_ids: Sequence[Any],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    uncertainty: Sequence[float],
    source_path: str,
) -> pd.DataFrame:
    sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "uncertainty_score": np.asarray(uncertainty, dtype=float),
        }
    )
    for name in ("task_id", "protocol", "split_id", "model_family", "model"):
        sample[name] = str(entry[name])
    sample["abs_error"] = (sample["y_pred"] - sample["y_true"]).abs()
    sample["confidence_tier"] = pd.NA
    sample["parameter_snapshot_id"] = snapshot_id
    sample["source_path"] = source_path
    sample["member_count"] = 1
    sample["uncertainty_method"] = "predictive_q90_minus_q10"
    return sample[STANDARD_SAMPLE_COLUMNS + ["member_count", "uncertainty_method"]]


def _checkpoint_config_matches(frozen: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        return
    keys = ("target_column", "base_model_path", "tokenizer_path", "batch_size", "learning_rate", "max_len", "dropout", "pooling")
    for key in keys:
        if key in frozen and key in checkpoint_config and not _values_match(frozen[key], checkpoint_config[key]):
            raise ValueError(f"LLMProp checkpoint config differs from frozen YAML: {key}")


def _run_llmprop_entry(
    entry: dict[str, Any],
    snapshot_id: str,
    passes: int,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    from src.LLMProp.dataset import create_dataloader
    from src.LLMProp.trainer import LLMPropTrainingConfig, _load_model, _load_tokenizer

    source = entry.get("source", {}) or {}
    manifest = _json_load(_safe_path(source.get("parameter_config")))
    frozen = entry.get("effective_parameters", {}) or {}
    checkpoint_path = _safe_path(manifest.get("best_model_path"))
    test_path = _safe_path(manifest.get("test_csv"))
    if checkpoint_path is None or test_path is None:
        raise ValueError("LLMProp manifest is missing checkpoint or test data path.")
    if passes != len(seeds):
        raise ValueError("LLMProp MC-dropout pass count must match the registered seed count.")

    config = LLMPropTrainingConfig(
        result_dir=str(_safe_path(source.get("model_dir")) or ""),
        train_csv=str(_safe_path(manifest.get("train_csv")) or ""),
        test_csv=str(test_path),
        target_column=str(frozen["target_column"]),
        base_model_path=str(frozen["base_model_path"]),
        tokenizer_path=str(frozen["tokenizer_path"]),
        epochs=int(frozen.get("epochs", 50)),
        batch_size=int(frozen["batch_size"]),
        learning_rate=float(frozen["learning_rate"]),
        max_len=int(frozen["max_len"]),
        dropout=float(frozen["dropout"]),
        pooling=str(frozen["pooling"]),
        valid_ratio=float(frozen.get("valid_ratio", 0.2)),
        random_state=int(frozen.get("random_state", 42)),
        device=frozen.get("device"),
    )
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, _, _ = _load_tokenizer(config)
    model = _load_model(config, len(tokenizer)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _checkpoint_config_matches(frozen, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    normalizer = frozen.get("normalizer", {}) if isinstance(frozen.get("normalizer"), dict) else {}
    labels_mean = float(checkpoint.get("labels_mean", normalizer.get("mean", 0.0)))
    labels_std = float(checkpoint.get("labels_std", normalizer.get("std", 1.0)))
    if "mean" in normalizer and not _values_match(labels_mean, normalizer["mean"]):
        raise ValueError("LLMProp checkpoint label mean differs from frozen YAML.")
    if "std" in normalizer and not _values_match(labels_std, normalizer["std"]):
        raise ValueError("LLMProp checkpoint label std differs from frozen YAML.")

    test_df = pd.read_csv(test_path)
    dataloader = create_dataloader(
        tokenizer,
        test_df,
        config.max_len,
        config.batch_size,
        config.target_column,
        pooling=config.pooling,
        shuffle=False,
    )
    sample_ids = test_df["ID"].to_numpy() if "ID" in test_df.columns else np.arange(len(test_df))
    y_true = pd.to_numeric(test_df[config.target_column], errors="raise").to_numpy(dtype=float)
    std = labels_std if abs(labels_std) > 1e-12 else 1.0
    member_frames: list[pd.DataFrame] = []
    with torch.no_grad():
        for member_number, seed in enumerate(seeds):
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            enable_mc_dropout(model)
            predictions: list[float] = []
            for batch in dataloader:
                input_ids, attention_masks = (tensor.to(device) for tensor in batch[:2])
                _, outputs = model(input_ids, attention_masks)
                denormalized = outputs.detach().cpu().numpy().reshape(-1) * std + labels_mean
                predictions.extend(denormalized.tolist())
            member = pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "member_id": f"mc_{member_number:02d}_seed_{seed}",
                    "y_true": y_true,
                    "y_pred": np.asarray(predictions, dtype=float),
                    "source_path": str(checkpoint_path),
                }
            )
            for name in ("task_id", "protocol", "split_id", "model_family", "model"):
                member[name] = str(entry[name])
            member_frames.append(member[MEMBER_COLUMNS])
    members = pd.concat(member_frames, ignore_index=True)
    sample = aggregate_member_predictions(
        members,
        parameter_snapshot_id=snapshot_id,
        uncertainty_method="mc_dropout_sd_ddof1_30_passes",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return members, sample


def _run_tabpfn_entry(entry: dict[str, Any], snapshot_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import StandardScaler

    from src.TabPFN.model_factory import create_tabpfn_regressor

    frozen = entry.get("effective_parameters", {}) or {}
    if str(frozen.get("resolved_backend", "")).lower() != "api":
        raise ValueError("Only the frozen TabPFN API backend is allowed for this pilot.")
    train_path, test_path, manifest = _split_paths(entry)
    if train_path is None or test_path is None:
        raise ValueError("TabPFN split paths are missing from the source manifest.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    features = [str(name) for name in frozen.get("feature_cols", [])]
    missing = [name for name in features if name not in train.columns or name not in test.columns]
    if missing:
        raise ValueError(f"TabPFN frozen feature columns are missing: {missing}")
    target = str(manifest.get("target_col", ""))
    if target not in train.columns or target not in test.columns:
        raise ValueError("TabPFN target column is missing from a frozen split.")

    X_train = train[features].copy()
    X_test = test[features].copy()
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(X_train[name])]
    text = [name for name in features if name not in numeric]
    if numeric:
        X_train.loc[:, numeric] = X_train[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test.loc[:, numeric] = X_test[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
        scaler = StandardScaler()
        X_train.loc[:, numeric] = scaler.fit_transform(X_train[numeric])
        X_test.loc[:, numeric] = scaler.transform(X_test[numeric])
    if text:
        X_train.loc[:, text] = X_train[text].fillna("").astype(str)
        X_test.loc[:, text] = X_test[text].fillna("").astype(str)

    random_state = int(frozen.get("random_state", 42))
    feature_mode = str(frozen.get("feature_mode", "")).strip().lower()
    if feature_mode not in {"numeric", "text"}:
        feature_mode = "numeric" if str(entry.get("model", "")).endswith("-Numeric") else "text"
    model, model_info = create_tabpfn_regressor(
        base_path=Path.cwd(),
        backend="api",
        preferred_model_version="v2.5",
        feature_mode=feature_mode,
        random_state=random_state,
    )
    if model_info.get("requested_model_path") != "v2.5_default":
        raise ValueError("The Ti TabPFN pilot is not pinned to API 2.5 Plus.")
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(dtype=float)
    y_true = pd.to_numeric(test[target], errors="raise").to_numpy(dtype=float)
    y_pred, uncertainty = run_tabpfn_quantile_prediction(model, X_train, y_train, X_test)
    sample_ids = test["ID"].to_numpy() if "ID" in test.columns else np.arange(len(test))
    sample = _sample_from_tabpfn(
        entry,
        snapshot_id,
        sample_ids,
        y_true,
        y_pred,
        uncertainty,
        source_path=f"{train_path};{test_path}",
    )
    members = sample[[
        "task_id", "protocol", "split_id", "model_family", "model", "sample_id", "y_true", "y_pred", "source_path"
    ]].copy()
    members["member_id"] = "predictive_mean"
    members = members[MEMBER_COLUMNS]
    members["q90_minus_q10"] = uncertainty
    return members, sample


def _adapter_name(entry: dict[str, Any]) -> str:
    family = str(entry["model_family"]).lower()
    if family in {"traditional", "bert"}:
        return "reuse"
    if "llmprop" in family:
        return "llmprop"
    if "tabpfn" in family:
        return "tabpfn"
    return "unsupported"


def is_fatal_adapter_error(adapter: str, exc: Exception) -> bool:
    message = str(exc).lower()
    if adapter != "tabpfn":
        return False
    return any(
        phrase in message
        for phrase in (
            "invalid authentication credentials",
            "no tabpfn api key",
            "tabpfn-client is required",
        )
    )


def run_ti_uncertainty(
    cfg: CalibrationConfig,
    snapshot_path: str | Path,
    *,
    resume: bool = True,
    adapters: Iterable[str] = ("reuse", "llmprop", "tabpfn"),
    mc_passes: int = 30,
) -> pd.DataFrame:
    registry = FrozenParameterRegistry(snapshot_path)
    pilot_root = cfg.output_root / "pilots" / "ti"
    pilot_root.mkdir(parents=True, exist_ok=True)
    index_path = pilot_root / "audit" / "uncertainty_source_index.csv"
    cached_index = pd.read_csv(index_path) if resume and index_path.exists() else pd.DataFrame()
    index = cached_index if source_index_is_current(cached_index, registry) else build_uncertainty_source_index(registry, pilot_root)
    selected_adapters = {str(name).strip().lower() for name in adapters}
    settings = ((cfg.raw.get("pilots", {}) or {}).get("ti", {}) or {})
    configured_seeds = settings.get("llmprop_mc_seeds", list(range(42, 72)))
    seeds = [int(value) for value in configured_seeds][:mc_passes]
    if len(seeds) != mc_passes:
        raise ValueError("The Ti pilot config does not provide enough LLMProp MC-dropout seeds.")

    entries_by_key = {FrozenParameterRegistry.entry_key(entry): entry for entry in select_ti_entries(registry)}
    runtime_errors: list[dict[str, Any]] = []
    blocked_adapters: dict[str, str] = {}
    for _, source_row in index.iterrows():
        frozen_key = str(source_row["frozen_entry_key"])
        entry = entries_by_key[frozen_key]
        adapter = _adapter_name(entry)
        if adapter not in selected_adapters:
            continue
        if adapter in blocked_adapters:
            runtime_errors.append(
                {
                    **source_row.to_dict(),
                    "source_validation_status": "runtime_error",
                    "source_validation_message": blocked_adapters[adapter],
                    "runtime_stage": adapter,
                }
            )
            continue
        if source_row["source_validation_status"] != "valid":
            runtime_errors.append({**source_row.to_dict(), "runtime_stage": adapter})
            continue

        cache_dir = entry_cache_dir(pilot_root, entry)
        member_path = cache_dir / "member_predictions.csv"
        sample_path = cache_dir / "sample_uncertainty.csv"
        if resume and sample_path.exists():
            cached = pd.read_csv(sample_path)
            if set(cached.get("parameter_snapshot_id", [])) == {registry.snapshot_id}:
                continue
        try:
            if adapter == "reuse":
                model_dir = _safe_path(entry.get("source", {}).get("model_dir"))
                trial_dir = model_dir / "predictions" / "optuna_trials" / str(source_row["selected_trial_id"])
                members = load_trial_member_predictions(entry, trial_dir)
                sample = aggregate_member_predictions(members, registry.snapshot_id)
            elif adapter == "llmprop":
                members, sample = _run_llmprop_entry(entry, registry.snapshot_id, mc_passes, seeds)
            elif adapter == "tabpfn":
                members, sample = _run_tabpfn_entry(entry, registry.snapshot_id)
            else:
                raise ValueError(f"Unsupported Ti pilot adapter: {adapter}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            members.to_csv(member_path, index=False)
            sample.to_csv(sample_path, index=False)
        except Exception as exc:  # Keep the 330-entry pilot resumable and auditable.
            if is_fatal_adapter_error(adapter, exc):
                blocked_adapters[adapter] = str(exc)
            runtime_errors.append(
                {
                    **source_row.to_dict(),
                    "source_validation_status": "runtime_error",
                    "source_validation_message": str(exc),
                    "runtime_stage": adapter,
                }
            )

    combined = collect_cached_samples(pilot_root, list(entries_by_key.values()), registry.snapshot_id)
    if not combined.empty and set(combined["parameter_snapshot_id"].astype(str)) != {registry.snapshot_id}:
        raise ValueError("Ti pilot sample outputs contain mixed frozen parameter snapshot IDs.")
    combined.to_csv(pilot_root / "sample_level_uncertainty.csv", index=False)

    invalid_path = pilot_root / "audit" / "missing_or_invalid_sources.csv"
    existing_invalid = pd.read_csv(invalid_path) if invalid_path.exists() else pd.DataFrame()
    if not existing_invalid.empty and "runtime_stage" in existing_invalid.columns:
        retained_invalid = existing_invalid[
            ~existing_invalid["runtime_stage"].fillna("").astype(str).isin(selected_adapters)
        ].copy()
    else:
        retained_invalid = existing_invalid
    source_invalid = index[index["source_validation_status"] != "valid"].copy()
    runtime_frame = pd.DataFrame(runtime_errors)
    all_invalid = pd.concat([source_invalid, retained_invalid, runtime_frame], ignore_index=True, sort=False)
    if not all_invalid.empty:
        dedupe_columns = [
            column
            for column in ("frozen_entry_key", "source_validation_status", "source_validation_message", "runtime_stage")
            if column in all_invalid.columns
        ]
        all_invalid = all_invalid.drop_duplicates(subset=dedupe_columns, keep="last")
    all_invalid.to_csv(invalid_path, index=False)
    completed_keys = {
        FrozenParameterRegistry.entry_key(entry)
        for entry in entries_by_key.values()
        if (entry_cache_dir(pilot_root, entry) / "sample_uncertainty.csv").exists()
    }
    runtime_error_keys = (
        set(
            all_invalid.loc[
                all_invalid["source_validation_status"].astype(str).eq("runtime_error"),
                "frozen_entry_key",
            ].astype(str)
        )
        if not all_invalid.empty
        and {"source_validation_status", "frozen_entry_key"}.issubset(all_invalid.columns)
        else set()
    )
    coverage = build_runtime_coverage(index, completed_keys, runtime_error_keys)
    coverage.to_csv(pilot_root / "audit" / "coverage_report.csv", index=False)
    existing_manifest = _json_load(pilot_root / "manifest.json")
    executed_adapters = sorted(set(existing_manifest.get("executed_adapters", [])) | selected_adapters)
    _update_manifest(
        pilot_root,
        {
            "parameter_snapshot_id": registry.snapshot_id,
            "snapshot_path": str(Path(snapshot_path).resolve()),
            "tasks": list(TI_TASKS),
            "models": list(TI_MODELS),
            "source_entry_count": int(len(index)),
            "valid_source_entry_count": int(index["source_validation_status"].eq("valid").sum()),
            "executed_adapters": executed_adapters,
            "completed_entry_count": len(completed_keys),
            "sample_count": int(len(combined)),
            "runtime_error_count": len(runtime_error_keys),
            "sample_level_uncertainty": str(pilot_root / "sample_level_uncertainty.csv"),
        },
    )
    return combined


def summarize_ti_tier_error(sample: pd.DataFrame) -> pd.DataFrame:
    working = sample.dropna(subset=["confidence_tier", "abs_error"]).copy()
    if working.empty:
        return pd.DataFrame()
    overall = working.groupby(POOLED_GROUPS, dropna=False)["abs_error"].mean().rename("overall_mae")
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(POOLED_GROUPS + ["confidence_tier"], observed=True, dropna=False):
        mean, sd, se, low, high = mean_ci95(group["abs_error"])
        denominator = float(overall.loc[keys[:-1]])
        rows.append(
            {
                **dict(zip(POOLED_GROUPS, keys[:-1])),
                "confidence_tier": str(keys[-1]),
                "n": int(len(group)),
                "mae": mean,
                "sd": sd,
                "se": se,
                "ci95_low": low,
                "ci95_high": high,
                "overall_mae": denominator,
                "normalized_mae": mean / denominator if denominator else np.nan,
                "normalized_ci95_low": low / denominator if denominator else np.nan,
                "normalized_ci95_high": high / denominator if denominator else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["confidence_tier"] = pd.Categorical(result["confidence_tier"], TIER_ORDER, ordered=True)
    result = result.sort_values(POOLED_GROUPS + ["confidence_tier"])
    result["confidence_tier"] = result["confidence_tier"].astype("object")
    return result


def summarize_ti_uncertainty_error(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in sample.groupby(POOLED_GROUPS, dropna=False):
        valid = group[["uncertainty_score", "abs_error"]].apply(pd.to_numeric, errors="coerce").dropna()
        correlation = float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) > 1 else np.nan
        rows.append(
            {
                **dict(zip(POOLED_GROUPS, keys)),
                "sample_n": int(len(valid)),
                "spearman_uncertainty_error": correlation,
            }
        )
    return pd.DataFrame(rows)


def _plot_ti_by_model(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    if summary.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_paths: list[str] = []
    tier_positions = np.arange(len(TIER_ORDER))
    protocol_rank = {name: index for index, name in enumerate(TI_PROTOCOL_ORDER)}
    for (task_id, model), model_frame in summary.groupby(["task_id", "model"], dropna=False):
        protocols = sorted(model_frame["protocol"].astype(str).unique(), key=lambda name: protocol_rank.get(name, 999))
        columns = min(4, max(1, len(protocols)))
        rows = int(math.ceil(len(protocols) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.15 * columns, 2.8 * rows), squeeze=False, sharey=True)
        for axis, protocol in zip(axes.flat, protocols):
            panel = model_frame[model_frame["protocol"].astype(str) == protocol].copy()
            panel["confidence_tier"] = pd.Categorical(panel["confidence_tier"], TIER_ORDER, ordered=True)
            panel = panel.sort_values("confidence_tier")
            positions = [TIER_ORDER.index(str(value)) for value in panel["confidence_tier"]]
            y = panel["normalized_mae"].to_numpy(dtype=float)
            low = panel["normalized_ci95_low"].to_numpy(dtype=float)
            high = panel["normalized_ci95_high"].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(y - low, 0), np.maximum(high - y, 0)])
            axis.errorbar(positions, y, yerr=yerr, color="#1f5a8a", marker="o", linewidth=1.5, capsize=3)
            axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
            axis.set_title(protocol, fontsize=10)
            axis.set_xticks(tier_positions, TIER_ORDER)
            axis.grid(axis="y", alpha=0.2)
        for axis in axes.flat[len(protocols):]:
            axis.set_visible(False)
        for axis in axes[:, 0]:
            axis.set_ylabel("Normalized MAE")
        for axis in axes[-1, :]:
            if axis.get_visible():
                axis.set_xlabel("Confidence tier")
        fig.suptitle(f"{model} | {task_id}", fontsize=12)
        fig.tight_layout()
        model_dir = output_dir / _safe_component(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{_safe_component(task_id)}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(str(path))
    return output_paths


def _update_manifest(root: Path, updates: dict[str, Any]) -> None:
    path = root / "manifest.json"
    payload = _json_load(path)
    payload.update({"schema_version": 1, "pilot": "ti", "generated_at": _utc_now(), **updates})
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def run_ti_calibration(cfg: CalibrationConfig) -> pd.DataFrame:
    pilot_root = cfg.output_root / "pilots" / "ti"
    uncertainty_path = pilot_root / "sample_level_uncertainty.csv"
    if not uncertainty_path.exists():
        raise FileNotFoundError(f"Ti uncertainty table not found: {uncertainty_path}")
    sample = pd.read_csv(uncertainty_path)
    if not sample.empty:
        sample["abs_error"] = (
            pd.to_numeric(sample["y_pred"], errors="coerce") - pd.to_numeric(sample["y_true"], errors="coerce")
        ).abs()
        sample = assign_ranked_tertiles(sample)
    sample.to_csv(pilot_root / "sample_level_calibration.csv", index=False)
    tier_summary = summarize_ti_tier_error(sample)
    tier_summary.to_csv(pilot_root / "tier_error_summary.csv", index=False)
    metrics = summarize_ti_uncertainty_error(sample)
    metrics.to_csv(pilot_root / "uncertainty_error_metrics.csv", index=False)
    figure_paths = _plot_ti_by_model(tier_summary, pilot_root / "figures" / "by_model")
    tier_counts = (
        sample.groupby(POOLED_GROUPS + ["confidence_tier"], observed=True).size().rename("n").reset_index()
        if not sample.empty
        else pd.DataFrame()
    )
    max_tier_difference = 0
    if not tier_counts.empty:
        differences = tier_counts.groupby(POOLED_GROUPS)["n"].agg(lambda values: int(values.max() - values.min()))
        max_tier_difference = int(differences.max()) if not differences.empty else 0
    _update_manifest(
        pilot_root,
        {
            "calibration_sample_count": int(len(sample)),
            "tier_summary_count": int(len(tier_summary)),
            "uncertainty_metric_count": int(len(metrics)),
            "max_tier_count_difference": max_tier_difference,
            "figure_count": len(figure_paths),
            "figure_paths": figure_paths,
            "sample_level_calibration": str(pilot_root / "sample_level_calibration.csv"),
            "tier_error_summary": str(pilot_root / "tier_error_summary.csv"),
            "uncertainty_error_metrics": str(pilot_root / "uncertainty_error_metrics.csv"),
        },
    )
    return sample


def run_ti_pilot(
    cfg: CalibrationConfig,
    snapshot_path: str | Path,
    *,
    resume: bool = True,
    adapters: Iterable[str] = ("reuse", "llmprop", "tabpfn"),
    mc_passes: int = 30,
) -> pd.DataFrame:
    run_ti_uncertainty(
        cfg,
        snapshot_path,
        resume=resume,
        adapters=adapters,
        mc_passes=mc_passes,
    )
    return run_ti_calibration(cfg)
