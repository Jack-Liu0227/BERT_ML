from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedFold,
    PreparedSplit,
    extract_split_identity,
    save_json,
    save_prepared_split,
    split_identity_matches,
    stable_hash,
)


SPLIT_MANIFEST_NAME = "split_manifest.json"


def _dataset_name_from_path(data_file: str) -> str:
    dataset_name = Path(data_file).stem
    for suffix in ["_Processed_cleaned", "_with_ID", "_withID", "_cleaned", "_processed", "_Processed"]:
        dataset_name = dataset_name.replace(suffix, "")
    return dataset_name


def _safe_path_part(value: Any) -> str:
    text = str(value)
    for ch in '<>:"/\\|?*':
        text = text.replace(ch, "_")
    text = text.strip().strip(".")
    return text or "unnamed"


def build_method_param_hash(method_params: Dict[str, Any], random_state: int) -> str:
    return stable_hash(
        {
            "method_params": method_params,
            "random_state": int(random_state),
            "schema_version": 1,
        },
        length=16,
    )


def build_split_cache_dir(
    split_cache_root: str | Path,
    alloy_type: str | None,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
) -> Path:
    dataset_name = _dataset_name_from_path(data_file)
    param_hash = build_method_param_hash(method_params, random_state)
    return (
        Path(split_cache_root)
        / _safe_path_part(alloy_type or dataset_name)
        / _safe_path_part(dataset_name)
        / _safe_path_part(target_column)
        / _safe_path_part(method_name)
        / param_hash
    )


def _load_manifest(cache_dir: Path) -> Dict[str, Any] | None:
    manifest_path = cache_dir / SPLIT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    import json

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _frame_from_artifact(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _single_entry_from_artifacts(artifacts: Dict[str, str], split: PreparedSplit) -> Dict[str, Any]:
    train_df = _frame_from_artifact(artifacts["train_file"])
    test_df = _frame_from_artifact(artifacts["test_file"])
    return {
        "train_file": artifacts["train_file"],
        "test_file": artifacts["test_file"],
        "summary_file": artifacts["summary_file"],
        "trace_dir": artifacts.get("trace_dir"),
        "trace_manifest": artifacts.get("trace_manifest"),
        "train_label": split.train_label,
        "test_label": split.test_label,
        "split_summary": split.summary,
        "train_identity": extract_split_identity(train_df),
        "test_identity": extract_split_identity(test_df),
    }


def _validate_manifest_header(
    manifest: Dict[str, Any],
    *,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
) -> None:
    expected = {
        "data_file": str(data_file),
        "target_column": str(target_column),
        "split_strategy": str(method_name),
        "method_params": method_params,
        "random_state": int(random_state),
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Canonical OOD split cache mismatch for {key}: "
                f"existing={manifest.get(key)!r}, requested={expected_value!r}"
            )


def _validate_split_files(entry: Dict[str, Any]) -> None:
    train_file = Path(entry["train_file"])
    test_file = Path(entry["test_file"])
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Cached split files are missing: {train_file}, {test_file}")

    train_identity = extract_split_identity(pd.read_csv(train_file))
    test_identity = extract_split_identity(pd.read_csv(test_file))
    if not split_identity_matches(train_identity, entry.get("train_identity", {})):
        raise ValueError(f"Cached train split identity mismatch: {train_file}")
    if not split_identity_matches(test_identity, entry.get("test_identity", {})):
        raise ValueError(f"Cached test split identity mismatch: {test_file}")


def validate_cached_manifest(
    manifest: Dict[str, Any],
    *,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
) -> None:
    _validate_manifest_header(
        manifest,
        data_file=data_file,
        target_column=target_column,
        method_name=method_name,
        method_params=method_params,
        random_state=random_state,
    )
    if manifest.get("is_multi_fold"):
        for fold in manifest.get("folds", []):
            _validate_split_files(fold)
    else:
        _validate_split_files(manifest["split"])


def load_cached_split_manifest(
    cache_dir: str | Path,
    *,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
) -> Dict[str, Any] | None:
    cache_path = Path(cache_dir)
    manifest = _load_manifest(cache_path)
    if manifest is None:
        return None
    validate_cached_manifest(
        manifest,
        data_file=data_file,
        target_column=target_column,
        method_name=method_name,
        method_params=method_params,
        random_state=random_state,
    )
    return manifest


def save_single_split_cache(
    cache_dir: str | Path,
    prepared_split: PreparedSplit,
    *,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
    alloy_type: str | None = None,
) -> Dict[str, Any]:
    cache_path = Path(cache_dir)
    if cache_path.exists():
        existing = _load_manifest(cache_path)
        if existing is not None:
            validate_cached_manifest(
                existing,
                data_file=data_file,
                target_column=target_column,
                method_name=method_name,
                method_params=method_params,
                random_state=random_state,
            )
            return existing
        shutil.rmtree(cache_path)

    split_dir = cache_path / "split_data"
    artifacts = save_prepared_split(prepared_split, split_dir)
    manifest = {
        "schema_version": 1,
        "cache_dir": str(cache_path),
        "alloy_type": alloy_type,
        "data_file": str(data_file),
        "target_column": str(target_column),
        "split_strategy": str(method_name),
        "method_params": method_params,
        "random_state": int(random_state),
        "is_multi_fold": False,
        "split": _single_entry_from_artifacts(artifacts, prepared_split),
    }
    save_json(cache_path / SPLIT_MANIFEST_NAME, manifest)
    return manifest


def save_multi_fold_split_cache(
    cache_dir: str | Path,
    folds: List[PreparedFold],
    *,
    data_file: str,
    target_column: str,
    method_name: str,
    method_params: Dict[str, Any],
    random_state: int,
    alloy_type: str | None = None,
) -> Dict[str, Any]:
    cache_path = Path(cache_dir)
    if cache_path.exists():
        existing = _load_manifest(cache_path)
        if existing is not None:
            validate_cached_manifest(
                existing,
                data_file=data_file,
                target_column=target_column,
                method_name=method_name,
                method_params=method_params,
                random_state=random_state,
            )
            return existing
        shutil.rmtree(cache_path)

    fold_entries: List[Dict[str, Any]] = []
    for prepared_fold in folds:
        split_dir = cache_path / "folds" / f"fold_{prepared_fold.fold_index}" / "split_data"
        artifacts = save_prepared_split(prepared_fold.split, split_dir)
        entry = {
            "fold_index": int(prepared_fold.fold_index),
            "held_out_cluster_id": int(prepared_fold.held_out_cluster_id),
            "metadata": prepared_fold.metadata,
            **_single_entry_from_artifacts(artifacts, prepared_fold.split),
        }
        fold_entries.append(entry)

    manifest = {
        "schema_version": 1,
        "cache_dir": str(cache_path),
        "alloy_type": alloy_type,
        "data_file": str(data_file),
        "target_column": str(target_column),
        "split_strategy": str(method_name),
        "method_params": method_params,
        "random_state": int(random_state),
        "is_multi_fold": True,
        "fold_count": len(fold_entries),
        "folds": fold_entries,
    }
    save_json(cache_path / SPLIT_MANIFEST_NAME, manifest)
    return manifest
