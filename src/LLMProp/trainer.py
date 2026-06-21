from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from src.LLMProp.dataset import create_dataloader
from src.LLMProp.model import T5Predictor
from src.data_processing.strength_ood_common import canonical_row_hash, save_json, stable_hash


@dataclass
class LLMPropTrainingConfig:
    result_dir: str
    train_csv: str
    test_csv: str
    target_column: str
    test_set_csvs: Dict[str, str] = field(default_factory=dict)
    base_model_path: str = "models/llmprop/google_t5_v1_1_small"
    tokenizer_path: str = "models/llmprop/tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge"
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_len: int = 888
    dropout: float = 0.2
    pooling: str = "cls"
    valid_ratio: float = 0.2
    random_state: int = 42
    device: str | None = None
    split_manifest: Dict[str, Any] | None = None
    fold_index: int | None = None
    ood_method: str | None = None
    warmup_fraction: float = 0.3
    normalizer: str = "z_norm"
    use_optuna: bool = False
    n_trials: int = 20


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_existing_or_hf(path_or_id: str, fallback_hf_id: str | None = None) -> str:
    path = Path(path_or_id)
    if path.exists():
        return str(path)
    return fallback_hf_id or path_or_id


def _load_tokenizer(config: LLMPropTrainingConfig):
    tokenizer_source = _resolve_existing_or_hf(config.tokenizer_path, "t5-small")
    tokenizer_path = Path(tokenizer_source) if isinstance(tokenizer_source, str) else None
    if tokenizer_path is not None and tokenizer_path.exists() and not (tokenizer_path / "spiece.model").exists():
        fallback_path = Path(config.base_model_path)
        if fallback_path.exists() and (fallback_path / "spiece.model").exists():
            print(
                "[LLM-Prop] Modified tokenizer lacks spiece.model for slow T5 loading; "
                f"falling back to base tokenizer: {fallback_path}"
            )
            tokenizer_source = str(fallback_path)
        else:
            print(
                "[LLM-Prop] Modified tokenizer lacks spiece.model; "
                "falling back to Hugging Face t5-small tokenizer."
            )
            tokenizer_source = "t5-small"
    # Use the slow SentencePiece tokenizer to avoid optional fast-tokenizer
    # conversion dependencies (tiktoken/blobfile) in newer transformers.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    added = 0
    if config.pooling == "cls":
        added += tokenizer.add_tokens(["[CLS]"])
    return tokenizer, tokenizer_source, added


def _load_model(config: LLMPropTrainingConfig, tokenizer_size: int) -> T5Predictor:
    base_source = _resolve_existing_or_hf(config.base_model_path, "google/t5-v1_1-small")
    try:
        base_model = T5EncoderModel.from_pretrained(base_source)
    except Exception:
        if base_source == config.base_model_path:
            raise
        # transformers versions newer than LLM-Prop's original environment can
        # hit optional torch/torchvision import issues when resolving T5 classes
        # through AutoModel. Importing the concrete modeling class lazily keeps a
        # second path available without changing the public behavior.
        from transformers.models.t5.modeling_t5 import T5EncoderModel as ConcreteT5EncoderModel

        base_model = ConcreteT5EncoderModel.from_pretrained(base_source)
    base_model.resize_token_embeddings(tokenizer_size)
    hidden_size = int(getattr(base_model.config, "d_model", 512))
    return T5Predictor(
        base_model=base_model,
        base_model_output_size=hidden_size,
        n_classes=1,
        drop_rate=config.dropout,
        pooling=config.pooling,
    )


def _predict(model: nn.Module, dataloader, device: torch.device, labels_mean: float, labels_std: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    std = labels_std if abs(labels_std) > 1e-12 else 1.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch[:3])
            _, outputs = model(input_ids, attention_masks)
            denorm = outputs.detach().cpu().numpy().reshape(-1) * std + labels_mean
            predictions.extend(denorm.tolist())
            targets.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
    return np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, target_column: str, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_{target_column}_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_{target_column}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_{target_column}_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def _plain_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def _save_predictions(
    result_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    target_column: str,
) -> None:
    predictions_dir = result_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    train_out = pd.DataFrame()
    test_out = pd.DataFrame()
    if "ID" in train_df.columns:
        train_out["ID"] = train_df["ID"].to_numpy()
    if "ID" in test_df.columns:
        test_out["ID"] = test_df["ID"].to_numpy()
    train_out["Dataset"] = "Train"
    test_out["Dataset"] = "OODTest"
    train_out[f"{target_column}_Actual"] = y_train_true
    train_out[f"{target_column}_Predicted"] = y_train_pred
    test_out[f"{target_column}_Actual"] = y_test_true
    test_out[f"{target_column}_Predicted"] = y_test_pred

    train_out.to_csv(predictions_dir / "train_predictions.csv", index=False)
    test_out.to_csv(predictions_dir / "test_predictions.csv", index=False)
    pd.concat([train_out, test_out], ignore_index=True).to_csv(predictions_dir / "all_predictions.csv", index=False)


def _save_test_set_predictions(
    result_dir: Path,
    test_set_name: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_column: str,
) -> None:
    predictions_dir = result_dir / "predictions" / "test_sets"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame()
    if "ID" in test_df.columns:
        out["ID"] = test_df["ID"].to_numpy()
    out["Dataset"] = test_set_name
    out[f"{target_column}_Actual"] = y_true
    out[f"{target_column}_Predicted"] = y_pred
    out.to_csv(predictions_dir / f"{test_set_name}_predictions.csv", index=False)


def _candidate_values(values: list[Any], current: Any) -> list[Any]:
    result: list[Any] = []
    for value in [current, *values]:
        if value not in result:
            result.append(value)
    return result


def _run_llmprop_optuna_search(config: LLMPropTrainingConfig) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("LLM-Prop Optuna search requires optuna. Install optuna or run without --use_optuna.") from exc

    result_dir = Path(config.result_dir)
    optuna_dir = result_dir / "optuna_trials"
    optuna_dir.mkdir(parents=True, exist_ok=True)

    lr_choices = _candidate_values([1e-4, 3e-4, 1e-3], float(config.learning_rate))
    # Keep the default sweep GPU-safe for T5-small. Users can still force a
    # larger fixed batch/max_len for a non-Optuna run, but Optuna should not
    # repeatedly sample combinations that can crash CUDA and abort the study.
    batch_choices = _candidate_values([8, 16, 32], min(int(config.batch_size), 32))
    max_len_choices = _candidate_values([256, 512], min(int(config.max_len), 512))
    dropout_choices = _candidate_values([0.1, 0.2, 0.3], float(config.dropout))
    pooling_choices = _candidate_values(["cls", "mean"], str(config.pooling))

    trial_records: list[Dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        trial_config = replace(
            config,
            use_optuna=False,
            result_dir=str(optuna_dir / f"trial_{trial.number:03d}"),
            learning_rate=float(trial.suggest_categorical("learning_rate", lr_choices)),
            batch_size=int(trial.suggest_categorical("batch_size", batch_choices)),
            max_len=int(trial.suggest_categorical("max_len", max_len_choices)),
            dropout=float(trial.suggest_categorical("dropout", dropout_choices)),
            pooling=str(trial.suggest_categorical("pooling", pooling_choices)),
            random_state=int(config.random_state + trial.number),
        )
        try:
            manifest = run_llmprop_ood_training(trial_config)
        except RuntimeError as exc:
            message = str(exc)
            if "CUDA" in message or "CUBLAS" in message or "out of memory" in message.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                record = {
                    "trial_number": int(trial.number),
                    "value": float("inf"),
                    "params": dict(trial.params),
                    "result_dir": trial_config.result_dir,
                    "best_epoch": None,
                    "failed": True,
                    "error": message[:500],
                }
                trial_records.append(record)
                pd.DataFrame(trial_records).to_csv(optuna_dir / "llmprop_optuna_trials.csv", index=False)
                raise optuna.exceptions.TrialPruned(message)
            raise
        score = float(manifest["metrics"]["best_valid_mae"])
        record = {
            "trial_number": int(trial.number),
            "value": score,
            "params": dict(trial.params),
            "result_dir": trial_config.result_dir,
            "best_epoch": manifest["metrics"].get("best_epoch"),
        }
        trial_records.append(record)
        pd.DataFrame(trial_records).to_csv(optuna_dir / "llmprop_optuna_trials.csv", index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="llmprop_hyperparameter_search")
    study.optimize(objective, n_trials=max(1, int(config.n_trials)), catch=(RuntimeError,))

    completed = [trial for trial in study.trials if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("All LLM-Prop Optuna trials failed or were pruned; reduce batch size/max_len and retry.")

    best_params = dict(study.best_trial.params)
    final_config = replace(
        config,
        use_optuna=False,
        learning_rate=float(best_params.get("learning_rate", config.learning_rate)),
        batch_size=int(best_params.get("batch_size", config.batch_size)),
        max_len=int(best_params.get("max_len", config.max_len)),
        dropout=float(best_params.get("dropout", config.dropout)),
        pooling=str(best_params.get("pooling", config.pooling)),
    )
    final_manifest = run_llmprop_ood_training(final_config)

    optuna_summary = {
        "enabled": True,
        "n_trials": int(config.n_trials),
        "best_trial_number": int(study.best_trial.number),
        "best_value_valid_mae": float(study.best_value),
        "best_params": best_params,
        "selection_metric": "inner_valid_mae",
        "test_usage": "OOD test is evaluated only after training and is not used for trial selection.",
        "trials": trial_records,
    }
    save_json(optuna_dir / "llmprop_optuna_summary.json", optuna_summary)

    manifest_path = result_dir / "llmprop_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["optuna"] = optuna_summary
        save_json(manifest_path, payload)
        final_manifest = payload
    else:
        final_manifest["optuna"] = optuna_summary
    return final_manifest


def run_llmprop_ood_training(config: LLMPropTrainingConfig) -> Dict[str, Any]:
    if config.use_optuna:
        return _run_llmprop_optuna_search(config)

    _set_seed(config.random_state)
    result_dir = Path(config.result_dir)
    checkpoint_dir = result_dir / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_full = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    test_set_frames = {
        str(name): pd.read_csv(path).reset_index(drop=True)
        for name, path in (config.test_set_csvs or {}).items()
    }
    if len(train_full) < 3:
        # Very small smoke-test splits cannot support a separate inner
        # validation set. Keep the OOD test untouched and reuse outer train for
        # checkpoint selection only in this degenerate case.
        train_df = train_full.copy()
        valid_df = train_full.copy()
    else:
        train_df, valid_df = train_test_split(
            train_full,
            test_size=config.valid_ratio,
            random_state=config.random_state,
            shuffle=True,
        )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    labels = pd.to_numeric(train_df[config.target_column], errors="coerce").to_numpy(dtype=np.float32)
    labels_mean = float(np.mean(labels))
    labels_std = float(np.std(labels, ddof=1)) if len(labels) > 1 else 1.0
    if abs(labels_std) <= 1e-12:
        labels_std = 1.0

    tokenizer, tokenizer_source, _ = _load_tokenizer(config)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(config, tokenizer_size=len(tokenizer)).to(device)

    train_loader = create_dataloader(
        tokenizer,
        train_df,
        config.max_len,
        config.batch_size,
        config.target_column,
        pooling=config.pooling,
        labels_mean=labels_mean,
        labels_std=labels_std,
        shuffle=True,
    )
    valid_loader = create_dataloader(
        tokenizer,
        valid_df,
        config.max_len,
        config.batch_size,
        config.target_column,
        pooling=config.pooling,
        shuffle=False,
    )
    outer_train_loader = create_dataloader(
        tokenizer,
        train_full,
        config.max_len,
        config.batch_size,
        config.target_column,
        pooling=config.pooling,
        shuffle=False,
    )
    test_loader = create_dataloader(
        tokenizer,
        test_df,
        config.max_len,
        config.batch_size,
        config.target_column,
        pooling=config.pooling,
        shuffle=False,
    )
    test_set_loaders = {
        name: create_dataloader(
            tokenizer,
            frame,
            config.max_len,
            config.batch_size,
            config.target_column,
            pooling=config.pooling,
            shuffle=False,
        )
        for name, frame in test_set_frames.items()
    }

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = max(1, len(train_loader) * max(1, config.epochs))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=max(1, config.epochs),
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=config.warmup_fraction,
    )
    mae_loss = nn.L1Loss()
    best_val_mae = float("inf")
    best_epoch = 0
    history: list[Dict[str, float]] = []
    best_model_path = checkpoint_dir / "best_model.pt"

    for epoch in range(max(1, config.epochs)):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"LLM-Prop epoch {epoch + 1}/{config.epochs}", leave=False):
            input_ids, attention_masks, _labels, normalized_labels = tuple(t.to(device) for t in batch)
            _, outputs = model(input_ids, attention_masks)
            loss = mae_loss(outputs, normalized_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach().cpu().item())

        valid_true, valid_pred = _predict(model, valid_loader, device, labels_mean, labels_std)
        valid_mae = float(mean_absolute_error(valid_true, valid_pred))
        train_loss = total_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "train_normalized_mae": train_loss, "valid_mae": valid_mae})
        if valid_mae <= best_val_mae:
            best_val_mae = valid_mae
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "labels_mean": labels_mean,
                    "labels_std": labels_std,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                },
                best_model_path,
            )

    loaded = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model_state_dict"])
    train_true, train_pred = _predict(model, outer_train_loader, device, labels_mean, labels_std)
    test_true, test_pred = _predict(model, test_loader, device, labels_mean, labels_std)
    test_set_metrics: Dict[str, Any] = {}
    for test_set_name, loader in test_set_loaders.items():
        test_set_true, test_set_pred = _predict(model, loader, device, labels_mean, labels_std)
        test_set_metrics[test_set_name] = _plain_metrics(test_set_true, test_set_pred)
        _save_test_set_predictions(
            result_dir=result_dir,
            test_set_name=test_set_name,
            test_df=test_set_frames[test_set_name],
            y_true=test_set_true,
            y_pred=test_set_pred,
            target_column=config.target_column,
        )

    metrics: Dict[str, Any] = {}
    metrics.update(_metrics(train_true, train_pred, config.target_column, "train"))
    metrics.update(_metrics(test_true, test_pred, config.target_column, "test"))
    if test_set_metrics:
        metrics["test_set_metrics"] = test_set_metrics
    metrics["best_epoch"] = best_epoch
    metrics["best_valid_mae"] = best_val_mae
    (result_dir / "final_evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    _save_predictions(result_dir, train_full, test_df, train_true, train_pred, test_true, test_pred, config.target_column)

    manifest = {
        "base_model_path": config.base_model_path,
        "base_model_resolved": _resolve_existing_or_hf(config.base_model_path, "google/t5-v1_1-small"),
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_resolved": tokenizer_source,
        "target_column": config.target_column,
        "ood_method": config.ood_method,
        "fold_index": config.fold_index,
        "train_csv": config.train_csv,
        "test_csv": config.test_csv,
        "test_set_csvs": dict(config.test_set_csvs or {}),
        "train_split_hash": canonical_row_hash(pd.read_csv(config.train_csv)),
        "test_split_hash": canonical_row_hash(pd.read_csv(config.test_csv)),
        "test_set_split_hashes": {
            name: canonical_row_hash(frame)
            for name, frame in test_set_frames.items()
        },
        "valid_split_hash": canonical_row_hash(valid_df),
        "normalizer": {
            "type": config.normalizer,
            "mean": labels_mean,
            "std": labels_std,
            "fit_scope": "inner_train_only",
        },
        "training_hyperparameters": asdict(config),
        "split_manifest_hash": stable_hash(config.split_manifest or {}),
        "best_model_path": str(best_model_path),
        "metrics": metrics,
    }
    save_json(result_dir / "llmprop_manifest.json", manifest)
    return manifest
