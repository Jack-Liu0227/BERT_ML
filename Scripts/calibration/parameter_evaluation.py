from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .schemas import CalibrationConfig


TRADITIONAL_MODEL_TYPES = {
    "RF": "rf",
    "XGB": "xgb",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "MLP": "ann",
}


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object for effective parameters.")
    return payload


def _resolve_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_traditional_candidate(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    inner_folds: int = 9,
    random_seed: int = 42,
) -> dict[str, Any]:
    from src.models.trainers.ml_trainer import MLTrainer

    project_root = Path(project_root).resolve()
    model_dir = _resolve_path(row["model_dir"], project_root).resolve()
    output_dir = _resolve_path(row["evaluation_output_dir"], project_root).resolve()
    manifest_path = model_dir / "ood_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"OOD manifest not found: {manifest_path}")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = _resolve_path(source_manifest["train_feature_file"], project_root).resolve()
    target_column = str(source_manifest["target_column"])
    if not train_path.exists():
        raise FileNotFoundError(f"Outer-train feature file not found: {train_path}")

    frame = pd.read_csv(train_path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is missing from {train_path}")
    feature_columns = [name for name in frame.columns if name not in {"ID", target_column}]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {train_path}")
    if len(frame) < inner_folds:
        raise ValueError(f"Need at least {inner_folds} rows for inner CV, found {len(frame)}")

    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=float).reshape(-1, 1)
    ids = frame["ID"].to_numpy() if "ID" in frame.columns else np.arange(len(frame))
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    effective = _json_mapping(row.get("effective_parameters_json", "{}"))
    model_name = str(row["model"])
    model_type = str(effective.get("model_type") or TRADITIONAL_MODEL_TYPES.get(model_name, "")).lower()
    if model_type not in set(TRADITIONAL_MODEL_TYPES.values()):
        raise ValueError(f"Unsupported Traditional model: {model_name} ({model_type})")
    model_parameters = effective.get("model_params", effective)
    if not isinstance(model_parameters, dict):
        raise ValueError("Traditional effective parameters must contain a model_params object.")

    validation_parts: list[pd.DataFrame] = []
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    for inner_fold, (train_indices, validation_indices) in enumerate(splitter.split(X_scaled), start=1):
        trainer = MLTrainer(
            model_type=model_type,
            result_dir=str(output_dir),
            model_name=f"fixed_parameter_inner_fold_{inner_fold}",
            model_params=dict(model_parameters),
            is_cv=True,
        )
        trainer.target_names = [target_column]
        model = trainer._create_model()
        train_y = y_scaled[train_indices].ravel()
        model.fit(X_scaled[train_indices], train_y)
        predicted_scaled = np.asarray(model.predict(X_scaled[validation_indices])).reshape(-1, 1)
        predicted = scaler_y.inverse_transform(predicted_scaled).ravel()
        actual = y[validation_indices].ravel()
        part = pd.DataFrame(
            {
                "_row_position": validation_indices,
                "ID": ids[validation_indices],
                "Dataset": "Validation",
                "inner_fold": inner_fold,
                f"{target_column}_Actual": actual,
                f"{target_column}_Predicted": predicted,
            }
        )
        validation_parts.append(part)

    predictions = (
        pd.concat(validation_parts, ignore_index=True)
        .sort_values("_row_position", kind="stable")
        .drop(columns="_row_position")
        .reset_index(drop=True)
    )
    prediction_path = output_dir / "predictions" / "validation_predictions.csv"
    _atomic_write_csv(prediction_path, predictions)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": str(row["task_id"]),
        "model_family": str(row["model_family"]),
        "model": model_name,
        "parameter_hash": str(row["parameter_hash"]),
        "outer_fold_id": str(row["fold_id"]),
        "inner_fold_count": inner_folds,
        "random_seed": random_seed,
        "model_type": model_type,
        "model_parameters": model_parameters,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_feature_file": str(train_path),
        "test_feature_file_used": False,
        "validation_prediction_file": str(prediction_path),
        "validation_row_count": int(len(predictions)),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def evaluate_bert_candidate(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    inner_folds: int = 9,
    random_seed: int = 42,
) -> dict[str, Any]:
    import torch

    from src.feature_engineering.utils import set_seed
    from src.models.nn_train_eval_pipeline.pipeline import TrainingPipeline as NNTrainingPipeline
    from src.models.trainers import TrainerFactory

    project_root = Path(project_root).resolve()
    model_dir = _resolve_path(row["model_dir"], project_root).resolve()
    output_dir = _resolve_path(row["evaluation_output_dir"], project_root).resolve()
    manifest_path = model_dir / "ood_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"OOD manifest not found: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = _resolve_path(source_manifest["train_feature_file"], project_root).resolve()
    target_column = str(source_manifest["target_column"])
    frame = pd.read_csv(train_path)
    feature_columns = [name for name in frame.columns if name not in {"ID", target_column}]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {train_path}")
    if len(frame) < inner_folds:
        raise ValueError(f"Need at least {inner_folds} rows for inner CV, found {len(frame)}")

    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    y = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1)
    ids = frame["ID"].to_numpy() if "ID" in frame.columns else np.arange(len(frame))
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_scaled = scaler_X.transform(X).astype(np.float32)
    y_scaled = scaler_y.transform(y).astype(np.float32)

    effective = _json_mapping(row.get("effective_parameters_json", "{}"))
    args_payload = effective.get("args", {})
    model_config = effective.get("model_config", {})
    if not isinstance(args_payload, dict) or not isinstance(model_config, dict):
        raise ValueError("BERT effective parameters require args and model_config objects.")
    args_payload = dict(args_payload)
    args_payload.update(
        {
            "model_type": "nn",
            "target_column": target_column,
            "target_columns": [target_column],
            "result_dir": str(output_dir),
            "random_state": random_seed,
            "use_multi_gpu": False,
        }
    )
    if str(args_payload.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
        args_payload["device"] = "cpu"
    defaults = {
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "patience": 30,
        "use_lr_scheduler": False,
        "lr_scheduler_patience": 10,
        "lr_scheduler_factor": 0.5,
        "device": "cpu",
    }
    for name, value in defaults.items():
        args_payload.setdefault(name, value)
    args = SimpleNamespace(**args_payload)
    pipeline = NNTrainingPipeline(args)
    pipeline.feature_names = feature_columns
    training_parameters = pipeline._get_training_params()
    training_parameters["save_config_on_init"] = False
    training_parameters["save_checkpoints"] = False

    set_seed(random_seed)
    validation_parts: list[pd.DataFrame] = []
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    for inner_fold, (train_indices, validation_indices) in enumerate(splitter.split(X_scaled), start=1):
        model, _ = pipeline._create_model(config=model_config)
        fold_train = {"X": X_scaled[train_indices], "y": y_scaled[train_indices]}
        fold_validation = {"X": X_scaled[validation_indices], "y": y_scaled[validation_indices]}
        trainer = TrainerFactory.create_trainer(
            model_type="nn",
            model=model,
            result_dir=str(output_dir),
            model_name=f"fixed_parameter_inner_fold_{inner_fold}",
            target_names=[target_column],
            train_data=fold_train,
            val_data=fold_validation,
            training_params=dict(training_parameters),
        )
        _, _, state_dict = trainer.train(num_epochs=int(args.epochs))
        if state_dict is None:
            raise RuntimeError(f"BERT inner fold {inner_fold} did not return a trained state dict.")
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device(str(args.device))
        model.to(device)
        with torch.no_grad():
            tensor = torch.tensor(X_scaled[validation_indices], dtype=torch.float32, device=device)
            predicted_scaled = model(tensor).detach().cpu().numpy().reshape(-1, 1)
        predicted = scaler_y.inverse_transform(predicted_scaled).ravel()
        validation_parts.append(
            pd.DataFrame(
                {
                    "_row_position": validation_indices,
                    "ID": ids[validation_indices],
                    "Dataset": "Validation",
                    "inner_fold": inner_fold,
                    f"{target_column}_Actual": y[validation_indices].ravel(),
                    f"{target_column}_Predicted": predicted,
                }
            )
        )

    predictions = (
        pd.concat(validation_parts, ignore_index=True)
        .sort_values("_row_position", kind="stable")
        .drop(columns="_row_position")
        .reset_index(drop=True)
    )
    prediction_path = output_dir / "predictions" / "validation_predictions.csv"
    _atomic_write_csv(prediction_path, predictions)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": str(row["task_id"]),
        "model_family": str(row["model_family"]),
        "model": str(row["model"]),
        "parameter_hash": str(row["parameter_hash"]),
        "outer_fold_id": str(row["fold_id"]),
        "inner_fold_count": inner_folds,
        "random_seed": random_seed,
        "model_config": model_config,
        "training_parameters": training_parameters,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "train_feature_file": str(train_path),
        "test_feature_file_used": False,
        "validation_prediction_file": str(prediction_path),
        "validation_row_count": int(len(predictions)),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def evaluate_llmprop_candidate(
    row: Mapping[str, Any],
    *,
    project_root: str | Path,
    inner_folds: int = 9,
    random_seed: int = 42,
    training_runner: Any = None,
) -> dict[str, Any]:
    import torch

    from src.LLMProp.trainer import LLMPropTrainingConfig, run_llmprop_ood_training

    if training_runner is None:
        training_runner = run_llmprop_ood_training
    project_root = Path(project_root).resolve()
    model_dir = _resolve_path(row["model_dir"], project_root).resolve()
    output_dir = _resolve_path(row["evaluation_output_dir"], project_root).resolve()
    manifest_path = model_dir / "llmprop_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LLMProp manifest not found: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = _resolve_path(source_manifest["train_csv"], project_root).resolve()
    target_column = str(source_manifest["target_column"])
    frame = pd.read_csv(train_path)
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is missing from {train_path}")
    if len(frame) < inner_folds:
        raise ValueError(f"Need at least {inner_folds} rows for inner CV, found {len(frame)}")

    effective = _json_mapping(row.get("effective_parameters_json", "{}"))
    config_fields = {item.name for item in fields(LLMPropTrainingConfig)}
    reusable = {name: value for name, value in effective.items() if name in config_fields}
    for path_field in ("base_model_path", "tokenizer_path"):
        value = reusable.get(path_field)
        if value:
            candidate = _resolve_path(value, project_root)
            if candidate.exists():
                reusable[path_field] = str(candidate.resolve())
    reusable["use_optuna"] = False
    reusable["target_column"] = target_column
    reusable["random_state"] = random_seed
    reusable["device"] = reusable.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    reusable["test_set_csvs"] = {}

    data_dir = output_dir / "inner_fold_data"
    validation_parts: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    for inner_fold, (train_indices, validation_indices) in enumerate(splitter.split(frame), start=1):
        fold_data_dir = data_dir / f"fold_{inner_fold}"
        train_fold_path = fold_data_dir / "train.csv"
        validation_fold_path = fold_data_dir / "validation.csv"
        _atomic_write_csv(train_fold_path, frame.iloc[train_indices].reset_index(drop=True))
        _atomic_write_csv(validation_fold_path, frame.iloc[validation_indices].reset_index(drop=True))
        fold_output = output_dir / "llmprop_inner_folds" / f"fold_{inner_fold}"
        config = LLMPropTrainingConfig(
            **{
                **reusable,
                "result_dir": str(fold_output),
                "train_csv": str(train_fold_path),
                "test_csv": str(validation_fold_path),
                "fold_index": inner_fold - 1,
                "ood_method": "calibration_inner_validation",
            }
        )
        result = training_runner(config)
        prediction_path = fold_output / "predictions" / "test_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(f"LLMProp validation predictions were not written: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        if len(predictions) != len(validation_indices):
            raise ValueError(
                f"LLMProp inner fold {inner_fold} returned {len(predictions)} predictions for "
                f"{len(validation_indices)} validation rows."
            )
        predictions["Dataset"] = "Validation"
        predictions["inner_fold"] = inner_fold
        predictions["_row_position"] = validation_indices
        validation_parts.append(predictions)
        checkpoint_dir = fold_output / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        fold_manifests.append(
            {
                "inner_fold": inner_fold,
                "train_csv": str(train_fold_path),
                "validation_csv": str(validation_fold_path),
                "result_dir": str(fold_output),
                "best_valid_mae": result.get("metrics", {}).get("best_valid_mae"),
            }
        )

    predictions = (
        pd.concat(validation_parts, ignore_index=True)
        .sort_values("_row_position", kind="stable")
        .drop(columns="_row_position")
        .reset_index(drop=True)
    )
    prediction_path = output_dir / "predictions" / "validation_predictions.csv"
    _atomic_write_csv(prediction_path, predictions)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": str(row["task_id"]),
        "model_family": str(row["model_family"]),
        "model": str(row["model"]),
        "parameter_hash": str(row["parameter_hash"]),
        "outer_fold_id": str(row["fold_id"]),
        "inner_fold_count": inner_folds,
        "random_seed": random_seed,
        "effective_parameters": effective,
        "target_column": target_column,
        "train_feature_file": str(train_path),
        "ood_test_file_used": False,
        "validation_prediction_file": str(prediction_path),
        "validation_row_count": int(len(predictions)),
        "inner_folds": fold_manifests,
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _completed_manifest(row: Mapping[str, Any], project_root: Path) -> dict[str, Any] | None:
    output_dir = _resolve_path(row["evaluation_output_dir"], project_root)
    path = output_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("status") != "complete":
        return None
    if str(payload.get("parameter_hash", "")) != str(row["parameter_hash"]):
        return None
    return payload


def run_parameter_evaluation_queue(
    cfg: CalibrationConfig,
    *,
    families: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    evaluator_map: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    queue_path = cfg.output_root / "parameter_freeze" / "missing_parameter_fold_scores.csv"
    if not queue_path.exists():
        raise FileNotFoundError(f"Missing parameter evaluation queue: {queue_path}")
    queue = pd.read_csv(queue_path)
    required = {
        "task_id",
        "model_family",
        "model",
        "parameter_hash",
        "fold_id",
        "model_dir",
        "effective_parameters_json",
        "evaluation_output_dir",
    }
    missing_columns = sorted(required - set(queue.columns))
    if missing_columns:
        raise ValueError(f"Parameter evaluation queue is missing columns: {missing_columns}")

    evaluators = {
        "Traditional": evaluate_traditional_candidate,
        "BERT": evaluate_bert_candidate,
        "LLMProp": evaluate_llmprop_candidate,
    }
    if evaluator_map is not None:
        evaluators.update(evaluator_map)
    selected_families = set(families or evaluators)
    queue = queue[queue["model_family"].astype(str).isin(selected_families)].copy()
    queue = queue.sort_values(
        ["model_family", "task_id", "model", "parameter_hash", "fold_id"],
        kind="stable",
    )

    selection_cfg = cfg.raw.get("parameter_selection", {})
    inner_folds = int(selection_cfg.get("inner_cv_folds", 9))
    random_seed = int(selection_cfg.get("random_seed", 42))
    status_path = cfg.output_root / "parameter_evaluations" / "evaluation_status.csv"
    status_keys = ["task_id", "model_family", "model", "parameter_hash", "fold_id"]
    existing_status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()

    def merged_status(current: pd.DataFrame) -> pd.DataFrame:
        if existing_status.empty:
            return current
        combined = pd.concat([existing_status, current], ignore_index=True)
        return combined.drop_duplicates(status_keys, keep="last").sort_values(status_keys, kind="stable")

    status_rows: list[dict[str, Any]] = []
    attempted = 0
    for _, series in queue.iterrows():
        row = series.to_dict()
        base = {
            "task_id": str(row["task_id"]),
            "model_family": str(row["model_family"]),
            "model": str(row["model"]),
            "parameter_hash": str(row["parameter_hash"]),
            "fold_id": str(row["fold_id"]),
            "evaluation_output_dir": str(row["evaluation_output_dir"]),
        }
        existing = _completed_manifest(row, cfg.project_root) if resume else None
        if existing is not None:
            status_rows.append(
                {
                    **base,
                    "evaluation_status": "complete",
                    "execution_mode": "resumed",
                    "message": "",
                }
            )
            continue
        if limit is not None and attempted >= limit:
            status_rows.append(
                {
                    **base,
                    "evaluation_status": "pending",
                    "execution_mode": "not_started",
                    "message": "Evaluation limit reached.",
                }
            )
            continue
        evaluator = evaluators.get(str(row["model_family"]))
        if evaluator is None:
            status_rows.append(
                {
                    **base,
                    "evaluation_status": "unsupported",
                    "execution_mode": "not_started",
                    "message": f"No evaluator registered for {row['model_family']}",
                }
            )
            continue

        attempted += 1
        try:
            result = evaluator(
                row,
                project_root=cfg.project_root,
                inner_folds=inner_folds,
                random_seed=random_seed,
            )
            status = str(result.get("status", "complete"))
            message = ""
        except Exception as exc:  # Keep the queue resumable after individual model failures.
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
        status_rows.append(
            {
                **base,
                "evaluation_status": status,
                "execution_mode": "executed",
                "message": message,
            }
        )
        _atomic_write_csv(status_path, merged_status(pd.DataFrame(status_rows)))

    status_frame = pd.DataFrame(status_rows)
    _atomic_write_csv(status_path, merged_status(status_frame))
    return status_frame
