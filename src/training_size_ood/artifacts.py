"""Canonical prediction and artifact handling for training-size runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_processing.strength_ood_common import PreparedSplit

from .core import LOCO_TEST_NAME, TOP20_TEST_NAME


RANDOM_CV_TEST_NAME = "random_cv"
PREDICTION_COLUMNS = ["ID", "test_set", "y_true", "y_pred", "abs_error"]
FORBIDDEN_MODEL_SUFFIXES = {
    ".bin",
    ".bst",
    ".cbm",
    ".ckpt",
    ".joblib",
    ".model",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".ubj",
}
FORBIDDEN_MODEL_DIR_NAMES = {"checkpoint", "checkpoints", "models", "saved_model", "saved_models"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_predictions(
    run_dir: str | Path,
    prepared_split: PreparedSplit,
    *,
    random_cv_mae: float,
    test_set_mae: Mapping[str, float],
) -> Path:
    root = Path(run_dir)
    expected = {
        RANDOM_CV_TEST_NAME: prepared_split.test_df,
        TOP20_TEST_NAME: prepared_split.test_sets[TOP20_TEST_NAME],
        LOCO_TEST_NAME: prepared_split.test_sets[LOCO_TEST_NAME],
    }
    expected_mae = {
        RANDOM_CV_TEST_NAME: float(random_cv_mae),
        TOP20_TEST_NAME: float(test_set_mae[TOP20_TEST_NAME]),
        LOCO_TEST_NAME: float(test_set_mae[LOCO_TEST_NAME]),
    }
    parts = [
        _load_prediction_set(
            root,
            test_name=test_name,
            target_col=prepared_split.target_col,
            expected_frame=frame,
        )
        for test_name, frame in expected.items()
    ]
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(["test_set", "ID"]).any():
        raise ValueError("Canonical run predictions contain duplicate test-set IDs")
    for test_name, group in predictions.groupby("test_set", sort=False):
        observed = float(group["abs_error"].mean())
        reference = expected_mae[str(test_name)]
        if not np.isclose(observed, reference, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Prediction MAE mismatch for {test_name}: predictions={observed}, metrics={reference}"
            )
    output_path = root / "predictions.csv"
    _atomic_csv_write(output_path, predictions[PREDICTION_COLUMNS])
    return output_path


def cleanup_training_artifacts(run_dir: str | Path) -> list[str]:
    root = Path(run_dir).resolve()
    removed: list[str] = []
    directories = sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_dir() and path.name.lower() in FORBIDDEN_MODEL_DIR_NAMES
        ],
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        _assert_within(path, root)
        removed.append(str(path.relative_to(root)))
        shutil.rmtree(path)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FORBIDDEN_MODEL_SUFFIXES:
            continue
        _assert_within(path, root)
        removed.append(str(path.relative_to(root)))
        path.unlink()
    remaining = find_training_artifacts(root)
    if remaining:
        raise ValueError(f"Serialized model artifacts remain after cleanup: {remaining[:5]}")
    return removed


def find_training_artifacts(run_dir: str | Path) -> list[str]:
    root = Path(run_dir)
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() in FORBIDDEN_MODEL_DIR_NAMES:
            found.append(str(path.relative_to(root)))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES:
            found.append(str(path.relative_to(root)))
    return sorted(set(found))


def validate_completed_artifacts(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> bool:
    root = Path(run_dir)
    metrics_path = root / "run_metrics.json"
    predictions_path = root / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return False
    expected_predictions_hash = str(marker.get("predictions_sha256", ""))
    expected_metrics_hash = str(marker.get("metrics_sha256", ""))
    if not expected_predictions_hash or not expected_metrics_hash:
        return False
    if str(manifest.get("predictions_sha256", "")) != expected_predictions_hash:
        return False
    if str(manifest.get("metrics_sha256", "")) != expected_metrics_hash:
        return False
    if sha256_file(predictions_path) != expected_predictions_hash:
        return False
    if sha256_file(metrics_path) != expected_metrics_hash:
        return False
    try:
        predictions = pd.read_csv(predictions_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    if list(predictions.columns) != PREDICTION_COLUMNS or predictions.empty:
        return False
    if set(predictions["test_set"].astype(str)) != {
        RANDOM_CV_TEST_NAME,
        TOP20_TEST_NAME,
        LOCO_TEST_NAME,
    }:
        return False
    if predictions.duplicated(["test_set", "ID"]).any():
        return False
    numeric = predictions[["y_true", "y_pred", "abs_error"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return False
    if not np.allclose(
        numeric["abs_error"].to_numpy(dtype=float),
        np.abs(numeric["y_true"].to_numpy(dtype=float) - numeric["y_pred"].to_numpy(dtype=float)),
    ):
        return False
    return not find_training_artifacts(root)


def _load_prediction_set(
    run_dir: Path,
    *,
    test_name: str,
    target_col: str,
    expected_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "ID" not in expected_frame.columns or target_col not in expected_frame.columns:
        raise ValueError(f"Expected {test_name} frame is missing ID or target column")
    candidates = _prediction_candidates(run_dir, test_name)
    errors: list[str] = []
    for path in candidates:
        try:
            return _normalize_prediction_file(
                path,
                test_name=test_name,
                target_col=target_col,
                expected_frame=expected_frame,
            )
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
    detail = "; ".join(errors[:4]) if errors else "no source prediction files found"
    raise ValueError(f"Could not load {test_name} predictions: {detail}")


def _prediction_candidates(run_dir: Path, test_name: str) -> list[Path]:
    predictions_dir = run_dir / "predictions"
    if test_name == RANDOM_CV_TEST_NAME:
        preferred = [predictions_dir / "test_predictions.csv"]
        discovered = [
            *run_dir.rglob("test_predictions.csv"),
            *run_dir.rglob("*all_predictions.csv"),
        ]
    else:
        preferred = [predictions_dir / "test_sets" / f"{test_name}_predictions.csv"]
        discovered = list(run_dir.rglob(f"{test_name}_predictions.csv"))
    ordered = sorted(
        [path for path in [*preferred, *discovered] if path.exists()],
        key=lambda path: (len(path.relative_to(run_dir).parts), str(path)),
    )
    return list(dict.fromkeys(ordered))


def _normalize_prediction_file(
    path: Path,
    *,
    test_name: str,
    target_col: str,
    expected_frame: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "ID" not in data.columns:
        raise ValueError("prediction file has no ID column")
    data = _filter_prediction_rows(data, path, test_name)
    actual_col, predicted_col = _prediction_value_columns(data, target_col)
    if data.empty:
        raise ValueError("prediction file contains no matching test rows")
    if data["ID"].map(_normalized_id).duplicated().any():
        raise ValueError("prediction file contains duplicate IDs")
    expected = expected_frame[["ID", target_col]].copy()
    expected["_id_key"] = expected["ID"].map(_normalized_id)
    observed = data[["ID", actual_col, predicted_col]].copy()
    observed["_id_key"] = observed["ID"].map(_normalized_id)
    if set(observed["_id_key"]) != set(expected["_id_key"]):
        raise ValueError("prediction IDs do not match the fixed test split")
    observed = observed.set_index("_id_key").loc[expected["_id_key"]].reset_index(drop=True)
    actual = pd.to_numeric(observed[actual_col], errors="coerce")
    predicted = pd.to_numeric(observed[predicted_col], errors="coerce")
    expected_actual = pd.to_numeric(expected[target_col], errors="coerce")
    if not np.isfinite(actual.to_numpy(dtype=float)).all() or not np.isfinite(
        predicted.to_numpy(dtype=float)
    ).all():
        raise ValueError("prediction values contain NaN or infinity")
    if not np.allclose(actual.to_numpy(dtype=float), expected_actual.to_numpy(dtype=float)):
        raise ValueError("prediction targets do not match the fixed test split")
    result = pd.DataFrame(
        {
            "ID": expected["ID"].to_numpy(),
            "test_set": test_name,
            "y_true": actual.to_numpy(dtype=float),
            "y_pred": predicted.to_numpy(dtype=float),
        }
    )
    result["abs_error"] = np.abs(result["y_true"] - result["y_pred"])
    return result


def _filter_prediction_rows(data: pd.DataFrame, path: Path, test_name: str) -> pd.DataFrame:
    if test_name != RANDOM_CV_TEST_NAME or path.name == "test_predictions.csv":
        return data.reset_index(drop=True)
    if "Dataset" in data.columns:
        labels = data["Dataset"].astype(str).str.lower()
        return data[~labels.isin({"train", "training", "val", "validation"})].reset_index(drop=True)
    if "set" in data.columns:
        labels = data["set"].astype(str).str.lower()
        return data[labels.isin({"test", "oodtest", "ood_test", "random_cv"})].reset_index(drop=True)
    return data.reset_index(drop=True)


def _prediction_value_columns(data: pd.DataFrame, target_col: str) -> tuple[str, str]:
    exact = (f"{target_col}_Actual", f"{target_col}_Predicted")
    if exact[0] in data.columns and exact[1] in data.columns:
        return exact
    actual = [column for column in data.columns if str(column).endswith("_Actual")]
    predicted = [column for column in data.columns if str(column).endswith("_Predicted")]
    if len(actual) == 1 and len(predicted) == 1:
        return str(actual[0]), str(predicted[0])
    actual = [column for column in data.columns if str(column).startswith("True_")]
    predicted = [column for column in data.columns if str(column).startswith("Pred_")]
    if len(actual) == 1 and len(predicted) == 1:
        return str(actual[0]), str(predicted[0])
    raise ValueError("prediction file does not contain one actual/predicted target pair")


def _normalized_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_within(path: Path, root: Path) -> None:
    path.resolve().relative_to(root)


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
