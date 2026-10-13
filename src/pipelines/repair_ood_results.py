from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.calibration.parameter_registry import FrozenParameterRegistry
from src.calibration.parameter_selection import (
    _snapshot_id as _calibration_snapshot_id,
    build_parameter_hash,
)
from src.data_processing.canonical_split_registry import (
    CanonicalSplitRegistry,
    CanonicalSplitEntry,
    sample_identity,
    sample_identity_hash,
)
from src.data_processing.strength_ood_common import save_prepared_split
from src.TabPFN.train_tabpfn_ood import TabPFNOODTrainer, safe_mape
from src.feature_engineering.utils import set_seed
from src.pipelines.batch_configs_ood import ALLOY_CONFIGS_OOD, OOD_METHODS
from src.pipelines.ood_pipeline import run_prepared_split


BERT_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "model_family": "BERT",
        "model": "SciBERT",
        "model_dirname": "scibert",
        "experiment_prefix": "experiment2a_all_nn_scibert",
    },
    {
        "model_family": "BERT",
        "model": "SteelBERT",
        "model_dirname": "steelbert",
        "experiment_prefix": "experiment2b_all_nn_steelbert",
    },
    {
        "model_family": "BERT",
        "model": "MatSciBERT",
        "model_dirname": "matscibert",
        "experiment_prefix": "experiment2c_all_nn_matscibert",
    },
)

TABPFN_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "model_family": "TabPFN",
        "model": "TabPFN-2.5-Plus-Numeric",
        "output_dirname": "ood_results_TabPFN-2.5-Plus-Numeric",
        "feature_mode": "numeric",
    },
    {
        "model_family": "TabPFN",
        "model": "TabPFN-2.5-Plus-Text",
        "output_dirname": "ood_results_TabPFN-2.5-Plus-Text",
        "feature_mode": "text",
    },
)

EXPECTED_REPAIR_COUNTS = {
    "SciBERT": 63,
    "SteelBERT": 63,
    "MatSciBERT": 70,
    "TabPFN-2.5-Plus-Numeric": 6,
    "TabPFN-2.5-Plus-Text": 6,
}
EXPECTED_FAKE_K5_COUNTS = {"MatSciBERT": 36, "SteelBERT": 1}
REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_PYTHON = Path(r"C:\Users\HK\miniconda3\envs\llm\python.exe")
CANONICAL_LAYOUT_PROTOCOLS = {
    "loco",
    "sparse_x_single",
    "sparse_x_cluster",
    "sparse_y_single",
    "sparse_y_cluster",
}


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(_lexical_absolute(path))))


def _is_directory_redirect(path: str | Path) -> bool:
    candidate = _lexical_absolute(path)
    return candidate.exists() and _path_key(candidate.resolve()) != _path_key(candidate)


def _reported_cluster_count(entry: CanonicalSplitEntry, summary: dict[str, Any]) -> int | None:
    if entry.protocol == "loco":
        value = summary.get("loco_cluster_count_requested")
    elif entry.protocol.startswith("sparse_"):
        value = summary.get("sparse_cluster_count_requested")
    else:
        return None
    return int(value) if value is not None else None


def audit_result_split(entry: CanonicalSplitEntry, run_root: str | Path) -> dict[str, Any]:
    run_path = Path(run_root).resolve()
    split_dir = run_path / "split_data"
    train_file = split_dir / entry.train_file.name
    test_file = split_dir / entry.test_file.name
    summary_file = split_dir / "split_summary.json"
    base = {
        "task_id": entry.task_id,
        "protocol": entry.protocol,
        "split_id": entry.split_id,
        "run_root": str(run_path),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "canonical_train_hash": entry.identity_record()["train_hash"],
        "canonical_test_hash": entry.identity_record()["test_hash"],
        "reported_cluster_count": None,
    }
    if not train_file.exists() or not test_file.exists() or not summary_file.exists():
        return {**base, "status": "rerun_required", "reason": "missing_split_artifacts"}

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**base, "status": "rerun_required", "reason": "invalid_split_summary"}
    reported_count = _reported_cluster_count(entry, summary)
    base["reported_cluster_count"] = reported_count
    if entry.protocol == "loco":
        expected_count = entry.method_params.get("loco_cluster_count")
    elif entry.protocol.startswith("sparse_"):
        expected_count = entry.method_params.get("sparse_cluster_count")
    else:
        expected_count = None
    if expected_count is not None and reported_count != int(expected_count):
        return {**base, "status": "rerun_required", "reason": "invalid_k5_parameters"}

    train = entry.load_train()
    test = entry.load_test()
    result_train = pd.read_csv(train_file)
    result_test = pd.read_csv(test_file)
    base["result_train_hash"] = sample_identity_hash(result_train, entry.target_column)
    base["result_test_hash"] = sample_identity_hash(result_test, entry.target_column)
    exact = (
        sample_identity(train, entry.target_column)
        == sample_identity(result_train, entry.target_column)
        and sample_identity(test, entry.target_column)
        == sample_identity(result_test, entry.target_column)
    )
    if exact:
        return {**base, "status": "reuse_exact", "reason": "canonical_match"}
    same_sets = (
        sorted(sample_identity(train, entry.target_column))
        == sorted(sample_identity(result_train, entry.target_column))
        and sorted(sample_identity(test, entry.target_column))
        == sorted(sample_identity(result_test, entry.target_column))
    )
    if same_sets:
        return {**base, "status": "reuse_reordered", "reason": "canonical_set_order_differs"}
    return {**base, "status": "rerun_required", "reason": "split_identity_mismatch"}


def _bert_result_paths(
    entry: CanonicalSplitEntry,
    model_spec: dict[str, str],
    workspace_root: Path,
) -> tuple[Path, Path]:
    method_meta = OOD_METHODS[entry.protocol]
    experiment_root = (
        workspace_root
        / "output"
        / "ood_results"
        / f"{model_spec['experiment_prefix']}_{method_meta['config_suffix']}"
    )
    alloy_dir = entry.alloy_type
    if entry.protocol == "target_extrapolation" and entry.alloy_type == "HEA":
        alloy_dir = "HEA_half"
    task_model_root = (
        experiment_root
        / alloy_dir
        / entry.dataset_name
        / entry.target_column
        / method_meta["result_dir_suffix"]
        / model_spec["model_dirname"]
    )
    run_root = task_model_root
    if entry.split_id != "main":
        run_root = task_model_root / "folds" / entry.split_id
    return _lexical_absolute(task_model_root), _lexical_absolute(run_root)


def _tabpfn_result_paths(
    entry: CanonicalSplitEntry,
    model_spec: dict[str, str],
    workspace_root: Path,
) -> tuple[Path, Path]:
    method_dir = OOD_METHODS[entry.protocol]["result_dir_suffix"]
    official_root = workspace_root / "output" / model_spec["output_dirname"]
    task_root = (
        official_root
        / method_dir
        / entry.alloy_type
        / entry.dataset_name
        / entry.target_column
    )
    run_root = task_root
    if entry.split_id != "main":
        run_root = task_root / "folds" / entry.split_id
    return task_root.resolve(), run_root.resolve()


def _traditional_result_paths(
    entry: CanonicalSplitEntry,
    workspace_root: Path,
) -> tuple[Path, Path]:
    method_meta = OOD_METHODS[entry.protocol]
    alloy_dir = entry.alloy_type
    if entry.protocol == "target_extrapolation" and entry.alloy_type == "HEA":
        alloy_dir = "HEA_half"
    task_root = (
        Path(workspace_root)
        / "output"
        / "ood_results"
        / f"experiment1_all_ml_models_{method_meta['config_suffix']}"
        / alloy_dir
        / entry.dataset_name
        / entry.target_column
        / method_meta["result_dir_suffix"]
        / "tradition"
    )
    run_root = task_root
    if entry.split_id != "main":
        run_root = task_root / "folds" / entry.split_id
    return _lexical_absolute(task_root), _lexical_absolute(run_root)


def _frame_content_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    payload = normalized.to_json(
        orient="split",
        index=False,
        date_format="iso",
        double_precision=15,
        force_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_test_split_to_canonical(
    entry: CanonicalSplitEntry,
    result_test_path: str | Path,
) -> dict[str, Any]:
    path = Path(result_test_path)
    canonical = entry.load_test().reset_index(drop=True)
    base = {
        "canonical_test_path": str(entry.test_file.resolve()),
        "result_test_path": str(path.resolve()),
        "canonical_count": len(canonical),
        "result_count": 0,
        "canonical_identity_hash": sample_identity_hash(canonical, entry.target_column),
        "result_identity_hash": "",
        "canonical_full_hash": _frame_content_hash(canonical),
        "result_full_hash": "",
        "identity_match": False,
        "full_frame_match": False,
    }
    if not path.exists():
        return {**base, "comparison_status": "missing"}

    result = pd.read_csv(path).reset_index(drop=True)
    base["result_count"] = len(result)
    try:
        base["result_identity_hash"] = sample_identity_hash(result, entry.target_column)
        identity_match = sample_identity(canonical, entry.target_column) == sample_identity(
            result,
            entry.target_column,
        )
    except (KeyError, TypeError, ValueError):
        identity_match = False
    base["identity_match"] = identity_match
    base["result_full_hash"] = _frame_content_hash(result)

    full_match = False
    if list(canonical.columns) == list(result.columns):
        try:
            pd.testing.assert_frame_equal(
                canonical,
                result,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-10,
            )
            full_match = True
        except AssertionError:
            pass
    base["full_frame_match"] = full_match
    if full_match:
        status = "exact"
    elif identity_match:
        status = "schema_mismatch"
    else:
        status = "identity_mismatch"
    return {**base, "comparison_status": status}


def build_cross_family_test_split_audit(
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workspace = Path(workspace_root).resolve()
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        _, traditional_run = _traditional_result_paths(entry, workspace)
        copies: list[tuple[str, str, Path]] = [
            ("Traditional", "Traditional-Shared", traditional_run),
        ]
        for spec in BERT_MODEL_SPECS:
            _, run_root = _bert_result_paths(entry, spec, workspace)
            copies.append(("BERT", spec["model"], run_root))
        for spec in TABPFN_MODEL_SPECS:
            _, run_root = _tabpfn_result_paths(entry, spec, workspace)
            copies.append(("TabPFN", spec["model"], run_root))

        for model_family, model, run_root in copies:
            test_path = run_root / "split_data" / entry.test_file.name
            comparison = compare_test_split_to_canonical(entry, test_path)
            rows.append(
                {
                    "task_id": entry.task_id,
                    "protocol": entry.protocol,
                    "split_id": entry.split_id,
                    "model_family": model_family,
                    "model": model,
                    "run_root": str(run_root),
                    **comparison,
                }
            )

    audit = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    grouping = ["task_id", "protocol", "split_id"]
    for key, group in audit.groupby(grouping, sort=True, dropna=False):
        exact_count = int(group["comparison_status"].astype(str).eq("exact").sum())
        identity_count = int(group["identity_match"].astype(bool).sum())
        expected = 6
        summary_rows.append(
            {
                "task_id": str(key[0]),
                "protocol": str(key[1]),
                "split_id": str(key[2]),
                "expected_copy_count": expected,
                "observed_copy_count": len(group),
                "exact_copy_count": exact_count,
                "identity_match_count": identity_count,
                "traditional_exact_count": int(
                    group.loc[group["model_family"].eq("Traditional"), "comparison_status"]
                    .astype(str)
                    .eq("exact")
                    .sum()
                ),
                "bert_exact_count": int(
                    group.loc[group["model_family"].eq("BERT"), "comparison_status"]
                    .astype(str)
                    .eq("exact")
                    .sum()
                ),
                "tabpfn_exact_count": int(
                    group.loc[group["model_family"].eq("TabPFN"), "comparison_status"]
                    .astype(str)
                    .eq("exact")
                    .sum()
                ),
                "comparison_status": (
                    "passed" if len(group) == expected and exact_count == expected else "failed"
                ),
            }
        )
    return audit, pd.DataFrame(summary_rows)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_tabpfn_prediction(
    entry: CanonicalSplitEntry,
    run_root: Path,
) -> dict[str, Any]:
    prediction_path = run_root / "predictions" / "all_predictions.csv"
    if not prediction_path.exists():
        return {
            "prediction_path": str(prediction_path),
            "prediction_status": "missing",
            "prediction_count": 0,
        }
    try:
        predictions, true_column, _ = _test_prediction_frame(prediction_path)
        canonical = entry.load_test().reset_index(drop=True)
        predicted_ids = pd.to_numeric(predictions["ID"], errors="raise").astype("int64")
        canonical_ids = pd.to_numeric(canonical["ID"], errors="raise").astype("int64")
        predicted_true = pd.to_numeric(predictions[true_column], errors="raise").to_numpy(float)
        canonical_true = pd.to_numeric(
            canonical[entry.target_column],
            errors="raise",
        ).to_numpy(float)
        exact = predicted_ids.tolist() == canonical_ids.tolist() and np.allclose(
            predicted_true,
            canonical_true,
            rtol=1e-6,
            atol=1e-4,
            equal_nan=False,
        )
        if exact:
            status = "exact"
        elif (
            not predicted_ids.duplicated().any()
            and not canonical_ids.duplicated().any()
            and set(predicted_ids.tolist()) == set(canonical_ids.tolist())
        ):
            by_id = pd.Series(predicted_true, index=predicted_ids.to_numpy())
            aligned = by_id.loc[canonical_ids.to_numpy()].to_numpy(float)
            status = (
                "reorder_required"
                if np.allclose(
                    aligned,
                    canonical_true,
                    rtol=1e-6,
                    atol=1e-4,
                    equal_nan=False,
                )
                else "identity_mismatch"
            )
        else:
            status = "identity_mismatch"
        return {
            "prediction_path": str(prediction_path),
            "prediction_status": status,
            "prediction_count": len(predictions),
        }
    except Exception as exc:
        return {
            "prediction_path": str(prediction_path),
            "prediction_status": "invalid",
            "prediction_count": 0,
            "prediction_error": f"{type(exc).__name__}: {exc}",
        }


def _runtime_payload_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "feature_cols",
        "feature_mode",
        "model_dirname",
        "model_name",
        "random_state",
        "requested_backend",
        "resolved_backend",
        "model_info",
    )
    return {key: payload.get(key) for key in keys}


def _runtime_diff_paths(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        paths: list[str] = []
        for key in sorted(set(actual) | set(expected)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_runtime_diff_paths(actual.get(key), expected.get(key), child))
        return paths
    return [] if actual == expected else [prefix]


TABPFN_ALLOWED_RUNTIME_METADATA_DIFFS = {
    "requested_backend",
    "model_info.requested_backend",
    "model_info.requested_feature_mode",
    "model_info.requested_model_version",
}


def canonicalize_tabpfn_api25_parameters(
    effective_parameters: dict[str, Any],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(effective_parameters))
    feature_mode = str(normalized.get("feature_mode", "")).lower()
    if feature_mode not in {"numeric", "text"}:
        raise ValueError(f"Unsupported TabPFN feature mode: {feature_mode}")
    normalized["requested_backend"] = "api"
    normalized["resolved_backend"] = "api"
    model_info = dict(normalized.get("model_info", {}) or {})
    model_info.update(
        {
            "backend": "api",
            "requested_backend": "api",
            "resolved_backend": "api",
            "requested_feature_mode": feature_mode,
            "feature_mode": feature_mode,
            "requested_model_version": "v2.5",
            "requested_model_path": "v2.5_default",
            "preferred_model_version": "api_2.5_plus",
            "effective_model_version": "api_2.5_plus",
        }
    )
    normalized["model_info"] = model_info
    return normalized


def build_tabpfn_api25_snapshot_candidate(
    parameter_registry: FrozenParameterRegistry,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    approved_models = {spec["model"] for spec in TABPFN_MODEL_SPECS}
    for source_entry in parameter_registry.iter_entries():
        entry = json.loads(json.dumps(source_entry))
        entry.pop("parameter_snapshot_id", None)
        if (
            str(entry.get("model_family")) == "TabPFN"
            and str(entry.get("model")) in approved_models
        ):
            effective = canonicalize_tabpfn_api25_parameters(
                dict(entry.get("effective_parameters", {}) or {})
            )
            entry["effective_parameters"] = effective
            entry["parameter_hash"] = build_parameter_hash(effective)
        entries.append(entry)
    entries.sort(
        key=lambda entry: (
            str(entry.get("task_id", "")),
            str(entry.get("model_family", "")),
            str(entry.get("model", "")),
        )
    )
    snapshot_id = _calibration_snapshot_id(entries)
    for entry in entries:
        entry["parameter_snapshot_id"] = snapshot_id
    payload = json.loads(json.dumps(parameter_registry.payload))
    payload.update(
        {
            "snapshot_id": snapshot_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": entries,
            "migration": {
                "kind": "tabpfn_api_2.5_plus_runtime_normalization",
                "source_snapshot_id": parameter_registry.snapshot_id,
                "allowed_models": sorted(approved_models),
            },
        }
    )
    return payload


def _audit_tabpfn_runtime(
    frozen_entry: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "runtime_status": "missing",
            "runtime_diff_paths": "",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = _runtime_payload_from_manifest(manifest)
        expected = _runtime_payload_from_manifest(
            canonicalize_tabpfn_api25_parameters(
                dict(frozen_entry.get("effective_parameters", {}) or {})
            )
        )
        differences = _runtime_diff_paths(actual, expected)
    except Exception as exc:
        return {
            "runtime_status": "invalid",
            "runtime_diff_paths": "",
            "runtime_error": f"{type(exc).__name__}: {exc}",
        }
    if not differences:
        status = "exact"
    elif set(differences).issubset(TABPFN_ALLOWED_RUNTIME_METADATA_DIFFS):
        status = "normalize_required"
    else:
        status = "invalid"
    return {
        "runtime_status": status,
        "runtime_diff_paths": json.dumps(differences, ensure_ascii=True),
    }


def verify_tabpfn_api25_quantile_artifact(
    entry: CanonicalSplitEntry,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(run_root)
    quantile_path = root / "predictions" / "predictive_quantiles.csv"
    manifest_path = root / "pipeline_manifest.json"
    reasons: list[str] = []
    count = 0
    try:
        expected_columns = [
            "ID",
            "y_true",
            "y_pred_mean",
            "q10",
            "q50",
            "q90",
            "uncertainty_score",
        ]
        quantiles = pd.read_csv(quantile_path)
        count = len(quantiles)
        if quantiles.columns.tolist() != expected_columns:
            reasons.append("unexpected_quantile_schema")
        canonical = entry.load_test()
        if quantiles["ID"].astype(str).tolist() != canonical["ID"].astype(str).tolist():
            reasons.append("quantile_id_order_mismatch")
        y_true = pd.to_numeric(quantiles["y_true"], errors="raise").to_numpy(float)
        canonical_true = pd.to_numeric(
            canonical[entry.target_column],
            errors="raise",
        ).to_numpy(float)
        if not np.allclose(y_true, canonical_true, rtol=1e-6, atol=1e-4):
            reasons.append("quantile_y_true_mismatch")
        numeric = quantiles[expected_columns[1:]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric.to_numpy(float)).all():
            reasons.append("quantile_nonfinite")
        if (numeric["q10"] > numeric["q50"]).any() or (
            numeric["q50"] > numeric["q90"]
        ).any():
            reasons.append("quantile_order_invalid")
        if not np.allclose(
            numeric["uncertainty_score"],
            numeric["q90"] - numeric["q10"],
            rtol=1e-8,
            atol=1e-10,
        ):
            reasons.append("quantile_uncertainty_mismatch")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_info = dict(manifest.get("model_info", {}) or {})
        requested_backend = str(
            manifest.get("requested_backend")
            or model_info.get("requested_backend")
            or ""
        )
        resolved_backend = str(
            manifest.get("resolved_backend")
            or model_info.get("resolved_backend")
            or ""
        )
        if requested_backend != "api" or resolved_backend != "api":
            reasons.append("runtime_backend_not_api")
        if str(model_info.get("requested_model_path", "")) != "v2.5_default":
            reasons.append("runtime_model_path_not_v2_5_default")
        if str(model_info.get("effective_model_version", "")) != "api_2.5_plus":
            reasons.append("runtime_version_not_api_2_5_plus")
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}:{exc}")
    return {
        "verification_status": "failed" if reasons else "passed",
        "verification_reason": ";".join(dict.fromkeys(reasons)),
        "prediction_count": count,
        "quantile_path": str(quantile_path),
        "manifest_path": str(manifest_path),
    }


def _verify_tabpfn_quantile_mean_matches_ood_prediction(
    run_root: str | Path,
) -> dict[str, str]:
    root = Path(run_root)
    reasons: list[str] = []
    try:
        predictions, _, prediction_column = _test_prediction_frame(
            root / "predictions" / "all_predictions.csv"
        )
        quantiles = pd.read_csv(root / "predictions" / "predictive_quantiles.csv")
        if predictions["ID"].astype(str).tolist() != quantiles["ID"].astype(str).tolist():
            reasons.append("quantile_prediction_id_mismatch")
        prediction_values = pd.to_numeric(
            predictions[prediction_column],
            errors="raise",
        ).to_numpy(float)
        quantile_means = pd.to_numeric(
            quantiles["y_pred_mean"],
            errors="raise",
        ).to_numpy(float)
        if len(prediction_values) != len(quantile_means) or not np.allclose(
            prediction_values,
            quantile_means,
            rtol=1e-10,
            atol=1e-12,
        ):
            reasons.append("quantile_prediction_mean_mismatch")
    except Exception as exc:
        reasons.append(f"{type(exc).__name__}:{exc}")
    return {
        "verification_status": "failed" if reasons else "passed",
        "verification_reason": ";".join(dict.fromkeys(reasons)),
    }


def build_tabpfn_v1_refresh_inventory(
    registry: CanonicalSplitRegistry,
    parameter_registry: FrozenParameterRegistry,
    workspace_root: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        for spec in TABPFN_MODEL_SPECS:
            parent_root, run_root = _tabpfn_result_paths(entry, spec, workspace)
            frozen_entry = parameter_registry.get(
                entry.task_id,
                entry.protocol,
                entry.split_id,
                "TabPFN",
                spec["model"],
            )
            required = {
                "pipeline_manifest": run_root / "pipeline_manifest.json",
                "prediction": run_root / "predictions" / "all_predictions.csv",
                "metrics": run_root / "metrics" / "metrics_summary.json",
            }
            complete = all(path.exists() for path in required.values())
            split_comparison = compare_test_split_to_canonical(
                entry,
                run_root / "split_data" / entry.test_file.name,
            )
            prediction = _audit_tabpfn_prediction(entry, run_root)
            runtime = _audit_tabpfn_runtime(
                frozen_entry,
                required["pipeline_manifest"],
            )
            rows.append(
                {
                    "repair_key": _repair_key(spec["model"], entry),
                    "task_id": entry.task_id,
                    "alloy_type": entry.alloy_type,
                    "dataset_name": entry.dataset_name,
                    "target_column": entry.target_column,
                    "protocol": entry.protocol,
                    "split_id": entry.split_id,
                    "model_family": "TabPFN",
                    "model": spec["model"],
                    "feature_mode": spec["feature_mode"],
                    "official_output_root": str(
                        (workspace / "output" / spec["output_dirname"]).resolve()
                    ),
                    "parent_result_root": str(parent_root),
                    "run_root": str(run_root),
                    "result_complete": complete,
                    "refresh_action": "canonicalize" if complete else "rerun",
                    "parameter_snapshot_id": parameter_registry.snapshot_id,
                    "parameter_hash": str(frozen_entry.get("parameter_hash", "")),
                    **split_comparison,
                    **prediction,
                    **runtime,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["task_id", "model", "protocol", "split_id"],
        kind="stable",
    ).reset_index(drop=True)


def _tabpfn_refresh_status(inventory: pd.DataFrame) -> pd.DataFrame:
    status = inventory[
        [
            "repair_key",
            "task_id",
            "protocol",
            "split_id",
            "model",
            "refresh_action",
            "run_root",
        ]
    ].copy()
    status["archive_status"] = "pending"
    status["canonicalize_status"] = np.where(
        status["refresh_action"].eq("canonicalize"),
        "pending",
        "not_required",
    )
    status["rerun_status"] = np.where(
        status["refresh_action"].eq("rerun"),
        "pending",
        "not_required",
    )
    status["verification_status"] = "pending"
    status["attempt_count"] = 0
    status["last_error"] = ""
    return status


def _initial_tabpfn_api25_all_status(inventory: pd.DataFrame) -> pd.DataFrame:
    required = [
        "repair_key",
        "task_id",
        "protocol",
        "split_id",
        "model",
        "run_root",
    ]
    missing = sorted(set(required).difference(inventory.columns))
    if missing:
        raise ValueError(f"TabPFN API 2.5 inventory is missing columns: {missing}")
    status = inventory[required].copy()
    if not status["repair_key"].astype(str).is_unique:
        raise ValueError("TabPFN API 2.5 inventory contains duplicate repair keys")
    status["run_status"] = "pending"
    status["verification_status"] = "pending"
    status["attempt_count"] = 0
    status["started_at"] = ""
    status["finished_at"] = ""
    status["failed_attempt_path"] = ""
    status["last_error"] = ""
    return status


def _select_tabpfn_api25_all_rows(
    inventory: pd.DataFrame,
    repair_key: str | None,
) -> pd.DataFrame:
    selected = inventory.copy()
    if repair_key is not None:
        selected = selected[selected["repair_key"].astype(str).eq(str(repair_key))].copy()
        if len(selected) != 1:
            raise ValueError(
                f"TabPFN API 2.5 repair key is not unique in the 300-unit inventory: {repair_key}"
            )
    return selected.reset_index(drop=True)


def run_prepare_tabpfn_v1_refresh_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    archive_root: str | Path,
    snapshot_path: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    split_registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    source_snapshot = _resolve_against(snapshot_path, workspace)
    source_parameter_registry = FrozenParameterRegistry(source_snapshot)
    candidate_payload = build_tabpfn_api25_snapshot_candidate(source_parameter_registry)
    candidate_path = output / "frozen_model_params_v1_api25_candidate.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate_payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    parameter_registry = FrozenParameterRegistry(candidate_path)

    cross_audit, cross_summary = build_cross_family_test_split_audit(
        split_registry,
        workspace,
    )
    traditional = cross_audit[cross_audit["model_family"].eq("Traditional")]
    bert = cross_audit[cross_audit["model_family"].eq("BERT")]
    tabpfn = cross_audit[cross_audit["model_family"].eq("TabPFN")]
    traditional_exact = int(traditional["comparison_status"].eq("exact").sum())
    bert_exact = int(bert["comparison_status"].eq("exact").sum())
    tabpfn_identity = int(tabpfn["identity_match"].astype(bool).sum())
    if len(split_registry) != 150:
        raise ValueError(f"Expected 150 canonical split units, found {len(split_registry)}")
    if len(cross_audit) != 900:
        raise ValueError(f"Expected 900 cross-family split copies, found {len(cross_audit)}")
    if traditional_exact != 150 or bert_exact != 450:
        raise ValueError(
            "Traditional/BERT reference splits are not canonical: "
            f"Traditional={traditional_exact}/150, BERT={bert_exact}/450"
        )
    if tabpfn_identity != 300:
        raise ValueError(f"TabPFN test identities are not canonical: {tabpfn_identity}/300")

    inventory = build_tabpfn_v1_refresh_inventory(
        split_registry,
        parameter_registry,
        workspace,
    )
    canonicalize = inventory[inventory["refresh_action"].eq("canonicalize")].copy()
    rerun = inventory[inventory["refresh_action"].eq("rerun")].copy()
    reorder = inventory[inventory["prediction_status"].eq("reorder_required")].copy()
    metadata = inventory[inventory["runtime_status"].eq("normalize_required")].copy()
    invalid_predictions = inventory[
        inventory["prediction_status"].isin(["invalid", "identity_mismatch"])
    ]
    invalid_runtime = inventory[inventory["runtime_status"].eq("invalid")]
    if not invalid_predictions.empty or not invalid_runtime.empty:
        raise ValueError(
            "Unexpected completed TabPFN corruption is outside the approved refresh matrix: "
            f"predictions={len(invalid_predictions)}, runtime={len(invalid_runtime)}"
        )
    counts = (len(inventory), len(canonicalize), len(rerun), len(reorder), len(metadata))
    if counts != (300, 288, 12, 258, 288):
        raise ValueError(
            "TabPFN refresh matrix changed: "
            f"inventory/canonicalize/rerun/reorder/metadata={counts}"
        )

    repair_payload = inventory[
        ["repair_key", "refresh_action", "parameter_hash", "run_root"]
    ].to_json(orient="records", force_ascii=True)
    repair_id = hashlib.sha256(
        f"{parameter_registry.snapshot_id}|{repair_payload}".encode("utf-8")
    ).hexdigest()[:20]
    snapshot = parameter_registry.path.resolve()
    protected = cross_audit[cross_audit["model_family"].isin(["Traditional", "BERT"])].copy()
    protected["protected_file_sha256"] = protected["result_test_path"].map(_file_sha256)

    cross_audit.to_csv(output / "cross_family_test_split_audit.csv", index=False)
    cross_summary.to_csv(output / "cross_family_test_split_summary.csv", index=False)
    protected.to_csv(output / "protected_reference_test_splits.csv", index=False)
    inventory.to_csv(output / "tabpfn_v1_inventory.csv", index=False)
    canonicalize.to_csv(output / "tabpfn_canonicalize_queue.csv", index=False)
    reorder.to_csv(output / "tabpfn_prediction_reorder_queue.csv", index=False)
    metadata.to_csv(output / "tabpfn_manifest_normalization_queue.csv", index=False)
    rerun.to_csv(output / "tabpfn_api_rerun_queue.csv", index=False)
    _tabpfn_refresh_status(inventory).to_csv(
        output / "tabpfn_v1_refresh_status.csv",
        index=False,
    )
    archive_plan = build_tabpfn_v1_archive_plan(
        inventory,
        workspace_root=workspace,
        archive_root=archive_root,
        repair_id=repair_id,
    )
    archive_plan.to_csv(output / "tabpfn_archive_plan.csv", index=False)
    manifest = {
        "schema_version": 1,
        "stage": "prepare-tabpfn-v1-refresh",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repair_id": repair_id,
        "workspace_root": str(workspace),
        "split_root": str(_resolve_against(split_root, workspace)),
        "output_root": str(output),
        "archive_root": str(_resolve_against(archive_root, workspace)),
        "snapshot_path": str(snapshot),
        "source_snapshot_path": str(source_snapshot),
        "source_snapshot_file_sha256": _file_sha256(source_snapshot),
        "source_snapshot_id": source_parameter_registry.snapshot_id,
        "candidate_snapshot_file_sha256": _file_sha256(snapshot),
        "snapshot_id": parameter_registry.snapshot_id,
        "snapshot_entry_count": len(parameter_registry.iter_entries()),
        "inventory_count": len(inventory),
        "canonicalize_count": len(canonicalize),
        "rerun_count": len(rerun),
        "prediction_reorder_count": len(reorder),
        "metadata_normalization_count": len(metadata),
        "traditional_exact_count": traditional_exact,
        "bert_exact_count": bert_exact,
        "tabpfn_identity_count": tabpfn_identity,
        "archive_plan_count": len(archive_plan),
        "archive_executed": False,
        "canonicalize_complete": False,
        "rerun_complete": False,
        "verification_complete": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def _artifact_tree_hash(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return _file_sha256(root)
    if not root.is_dir():
        return ""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_tabpfn_refresh_path(
    path: str | Path,
    official_root: str | Path,
) -> tuple[Path, Path]:
    candidate = _lexical_absolute(path)
    official = _lexical_absolute(official_root)
    approved_names = {spec["output_dirname"] for spec in TABPFN_MODEL_SPECS}
    if official.name not in approved_names:
        raise ValueError(f"Unapproved TabPFN output root: {official}")
    try:
        candidate.relative_to(official)
    except ValueError as exc:
        raise ValueError(
            f"Refresh source is outside its approved TabPFN root: {candidate}"
        ) from exc
    if candidate.exists() and _is_directory_redirect(candidate):
        raise ValueError(f"Refresh source is a directory redirect: {candidate}")
    return candidate, official


def _validate_tabpfn_api25_all_inventory(
    inventory: pd.DataFrame,
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> None:
    if len(inventory) != 300:
        raise ValueError(
            f"TabPFN API 2.5 all-rerun inventory must contain 300 rows, found {len(inventory)}"
        )
    if not inventory["repair_key"].astype(str).is_unique:
        raise ValueError("TabPFN API 2.5 all-rerun inventory contains duplicate repair keys")
    expected_models = {
        "TabPFN-2.5-Plus-Numeric": 150,
        "TabPFN-2.5-Plus-Text": 150,
    }
    model_counts = inventory["model"].astype(str).value_counts().to_dict()
    if model_counts != expected_models:
        raise ValueError(
            "TabPFN API 2.5 all-rerun model counts changed: "
            f"actual={model_counts}, expected={expected_models}"
        )

    workspace = Path(workspace_root).resolve()
    specs = {str(spec["model"]): spec for spec in TABPFN_MODEL_SPECS}
    seen_units: set[tuple[str, str, str, str]] = set()
    for _, row in inventory.iterrows():
        model = str(row["model"])
        entry = registry.require(
            str(row["task_id"]),
            str(row["protocol"]),
            str(row["split_id"]),
        )
        unit_key = (entry.task_id, entry.protocol, entry.split_id, model)
        if unit_key in seen_units:
            raise ValueError(f"Duplicate TabPFN API 2.5 unit: {unit_key}")
        seen_units.add(unit_key)
        expected_parent, expected_run = _tabpfn_result_paths(entry, specs[model], workspace)
        expected_official = (workspace / "output" / specs[model]["output_dirname"]).resolve()
        actual_run, actual_official = _validate_tabpfn_refresh_path(
            str(row["run_root"]),
            str(row["official_output_root"]),
        )
        if _path_key(actual_run) != _path_key(expected_run):
            raise ValueError(
                f"TabPFN API 2.5 run path differs from the canonical path: {actual_run} != {expected_run}"
            )
        if _path_key(actual_official) != _path_key(expected_official):
            raise ValueError(
                "TabPFN API 2.5 official root differs from the approved root: "
                f"{actual_official} != {expected_official}"
            )
        if _path_key(str(row["parent_result_root"])) != _path_key(expected_parent):
            raise ValueError(
                "TabPFN API 2.5 parent result path differs from the canonical path: "
                f"{row['parent_result_root']} != {expected_parent}"
            )


def build_tabpfn_v1_archive_plan(
    inventory: pd.DataFrame,
    *,
    workspace_root: str | Path,
    archive_root: str | Path,
    repair_id: str,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    archive_base = (
        _resolve_against(archive_root, workspace)
        / f"tabpfn_v1_refresh_{repair_id}"
    )
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def add_record(
        *,
        source: Path,
        official: Path,
        action: str,
        repair_key: str,
    ) -> None:
        locked_source, _ = _validate_tabpfn_refresh_path(source, official)
        try:
            relative = locked_source.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Refresh source is outside workspace: {locked_source}") from exc
        destination = archive_base / relative
        key = (action, _path_key(locked_source))
        if key in records:
            existing = records[key]
            keys = set(str(existing["repair_keys"]).split(";"))
            keys.add(repair_key)
            existing["repair_keys"] = ";".join(sorted(value for value in keys if value))
            return
        records[key] = {
            "archive_action": action,
            "source_path": str(locked_source),
            "destination_path": str(destination),
            "official_output_root": str(official),
            "repair_keys": repair_key,
            "source_exists": locked_source.exists(),
            "source_kind": (
                "directory" if locked_source.is_dir() else "file" if locked_source.is_file() else "missing"
            ),
            "source_hash": _artifact_tree_hash(locked_source),
        }

    for _, row in inventory.iterrows():
        run_root = _lexical_absolute(str(row["run_root"]))
        official = _lexical_absolute(str(row["official_output_root"]))
        repair_key = str(row["repair_key"])
        _validate_tabpfn_refresh_path(run_root, official)
        if str(row["refresh_action"]) == "rerun":
            add_record(
                source=run_root,
                official=official,
                action="move_directory",
                repair_key=repair_key,
            )
        else:
            add_record(
                source=run_root / "split_data",
                official=official,
                action="copy_directory",
                repair_key=repair_key,
            )
            for relative in (
                Path("predictions") / "all_predictions.csv",
                Path("metrics") / "metrics_summary.json",
                Path("pipeline_manifest.json"),
            ):
                add_record(
                    source=run_root / relative,
                    official=official,
                    action="copy_file",
                    repair_key=repair_key,
                )

        protocol = str(row["protocol"])
        if protocol in {"loco", "random_cv_baseline"}:
            manifest_name = (
                "loco_manifest.json"
                if protocol == "loco"
                else "random_cv_baseline_manifest.json"
            )
            parent_manifest = _lexical_absolute(str(row["parent_result_root"])) / manifest_name
            if parent_manifest.exists():
                add_record(
                    source=parent_manifest,
                    official=official,
                    action="copy_file",
                    repair_key=repair_key,
                )

    return pd.DataFrame(records.values()).sort_values(
        ["archive_action", "source_path"],
        kind="stable",
    ).reset_index(drop=True)


def execute_tabpfn_v1_archive_plan(
    plan: pd.DataFrame,
    *,
    execute: bool,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        record = dict(row)
        source = _lexical_absolute(str(row["source_path"]))
        destination = _lexical_absolute(str(row["destination_path"]))
        official = _lexical_absolute(str(row["official_output_root"]))
        action = str(row["archive_action"])
        expected_hash = str(row.get("source_hash", ""))
        try:
            _validate_tabpfn_refresh_path(source, official)
            if not execute:
                status = "planned"
            elif action in {"copy_file", "copy_directory"}:
                if not source.exists():
                    raise FileNotFoundError(f"Archive source is missing: {source}")
                if destination.exists():
                    if _artifact_tree_hash(destination) != expected_hash:
                        raise FileExistsError(
                            f"Archive destination exists with a different hash: {destination}"
                        )
                    status = "already_copied"
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if action == "copy_file":
                        shutil.copy2(source, destination)
                    else:
                        shutil.copytree(source, destination)
                    if _artifact_tree_hash(destination) != expected_hash:
                        raise ValueError(f"Archive hash mismatch after copy: {destination}")
                    status = "copied"
            elif action == "move_directory":
                if source.exists() and destination.exists():
                    raise FileExistsError(
                        f"Both archive source and destination exist: {source} / {destination}"
                    )
                if destination.exists():
                    if _artifact_tree_hash(destination) != expected_hash:
                        raise FileExistsError(
                            f"Moved archive destination has a different hash: {destination}"
                        )
                    status = "already_moved"
                elif not source.exists():
                    raise FileNotFoundError(f"Archive source is missing: {source}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(destination)
                    if _artifact_tree_hash(destination) != expected_hash:
                        raise ValueError(f"Archive hash mismatch after move: {destination}")
                    status = "moved"
            else:
                raise ValueError(f"Unsupported TabPFN archive action: {action}")
            record["archive_status"] = status
            record["archive_error"] = ""
        except Exception as exc:
            record["archive_status"] = "failed"
            record["archive_error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
    return pd.DataFrame(results)


def replace_tabpfn_split_data_with_canonical(
    entry: CanonicalSplitEntry,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(run_root)
    split_dir = root / "split_data"
    temporary = root / "split_data.__canonical_tmp__"
    previous = root / "split_data.__previous__"
    for path in (temporary, previous):
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Unexpected non-directory split staging path: {path}")
            shutil.rmtree(path)

    artifacts = save_prepared_split(entry.load_prepared_split(), temporary)
    temporary_test = temporary / entry.test_file.name
    comparison = compare_test_split_to_canonical(entry, temporary_test)
    if comparison["comparison_status"] != "exact":
        shutil.rmtree(temporary)
        raise ValueError(
            f"Canonical split staging failed for {entry.task_id}/{entry.protocol}/{entry.split_id}"
        )

    if split_dir.exists():
        split_dir.replace(previous)
    try:
        temporary.replace(split_dir)
    except Exception:
        if previous.exists() and not split_dir.exists():
            previous.replace(split_dir)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    final = compare_test_split_to_canonical(entry, split_dir / entry.test_file.name)
    return {**final, "split_artifacts": artifacts}


def reorder_tabpfn_prediction_rows(
    entry: CanonicalSplitEntry,
    prediction_path: str | Path,
) -> dict[str, Any]:
    path = Path(prediction_path)
    frame = pd.read_csv(path)
    if "ID" not in frame.columns:
        raise ValueError(f"Prediction CSV is missing ID: {path}")
    true_column, _, set_column = _prediction_columns(frame)
    if set_column is None:
        raise ValueError(f"Prediction CSV is missing Dataset/Set labels: {path}")
    labels = frame[set_column].astype(str).str.strip().str.lower()
    test_mask = labels.isin({"oodtest", "ood_test", "test", "testing"})
    test = frame.loc[test_mask].copy()
    retained = frame.loc[~test_mask].copy()
    canonical = entry.load_test().reset_index(drop=True)
    test_ids = pd.to_numeric(test["ID"], errors="raise").astype("int64")
    canonical_ids = pd.to_numeric(canonical["ID"], errors="raise").astype("int64")
    if test_ids.duplicated().any() or canonical_ids.duplicated().any():
        raise ValueError(f"Prediction or canonical test IDs are duplicated: {path}")
    if set(test_ids.tolist()) != set(canonical_ids.tolist()):
        raise ValueError(f"Prediction test IDs do not match the canonical test set: {path}")
    indexed = test.assign(__canonical_id=test_ids.to_numpy()).set_index("__canonical_id")
    ordered = indexed.loc[canonical_ids.to_numpy()].reset_index(drop=True)
    predicted_true = pd.to_numeric(ordered[true_column], errors="raise").to_numpy(float)
    canonical_true = pd.to_numeric(
        canonical[entry.target_column],
        errors="raise",
    ).to_numpy(float)
    if not np.allclose(
        predicted_true,
        canonical_true,
        rtol=1e-6,
        atol=1e-4,
        equal_nan=False,
    ):
        raise ValueError(f"Prediction y_true does not match the canonical test set: {path}")
    changed = test_ids.tolist() != canonical_ids.tolist()
    if changed:
        refreshed = pd.concat([retained, ordered[frame.columns]], ignore_index=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        refreshed.to_csv(temporary, index=False)
        temporary.replace(path)
    return {
        "prediction_path": str(path),
        "changed": changed,
        "test_count": len(ordered),
    }


def _plain_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(safe_mape(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _request_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    model_info = payload.get("model_info", {})
    if not isinstance(model_info, dict):
        model_info = {}
    return {
        "requested_backend": payload.get("requested_backend"),
        "model_info": {
            "requested_backend": model_info.get("requested_backend"),
            "requested_feature_mode": model_info.get("requested_feature_mode"),
            "requested_model_version": model_info.get("requested_model_version"),
        },
    }


def _current_split_artifacts(entry: CanonicalSplitEntry, run_root: Path) -> dict[str, Any]:
    split_dir = run_root / "split_data"
    artifacts: dict[str, Any] = {
        "train_file": str((split_dir / entry.train_file.name).resolve()),
        "test_file": str((split_dir / entry.test_file.name).resolve()),
        "combined_test_file": str((split_dir / entry.test_file.name).resolve()),
        "summary_file": str((split_dir / "split_summary.json").resolve()),
    }
    trace_manifest = split_dir / "trace" / "split_manifest.json"
    if trace_manifest.exists():
        artifacts["trace_dir"] = str(trace_manifest.parent.resolve())
        artifacts["trace_manifest"] = str(trace_manifest.resolve())
    return artifacts


def refresh_tabpfn_metrics_and_manifest(
    entry: CanonicalSplitEntry,
    run_root: str | Path,
    *,
    frozen_entry: dict[str, Any],
    snapshot_id: str,
    snapshot_path: str | Path,
    repair_key: str,
    repair_id: str,
) -> dict[str, Any]:
    root = Path(run_root)
    prediction_path = root / "predictions" / "all_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    true_column, pred_column, set_column = _prediction_columns(predictions)
    if set_column is None:
        raise ValueError(f"Prediction CSV is missing Dataset/Set labels: {prediction_path}")
    labels = predictions[set_column].astype(str).str.strip().str.lower()
    test_mask = labels.isin({"oodtest", "ood_test", "test", "testing"})
    train_mask = labels.isin({"train", "training"})
    if not test_mask.any() or not train_mask.any():
        raise ValueError(f"Prediction CSV is missing train or OOD test rows: {prediction_path}")

    train_true = pd.to_numeric(predictions.loc[train_mask, true_column], errors="raise").to_numpy(float)
    train_pred = pd.to_numeric(predictions.loc[train_mask, pred_column], errors="raise").to_numpy(float)
    test_true = pd.to_numeric(predictions.loc[test_mask, true_column], errors="raise").to_numpy(float)
    test_pred = pd.to_numeric(predictions.loc[test_mask, pred_column], errors="raise").to_numpy(float)
    if not np.isfinite(train_pred).all() or not np.isfinite(test_pred).all():
        raise ValueError(f"Prediction CSV contains non-finite predictions: {prediction_path}")
    train_metrics = _plain_regression_metrics(train_true, train_pred)
    test_metrics = _plain_regression_metrics(test_true, test_pred)

    effective = canonicalize_tabpfn_api25_parameters(
        dict(frozen_entry.get("effective_parameters", {}) or {})
    )
    runtime_payload = _runtime_payload_from_manifest(effective)
    metrics_path = root / "metrics" / "metrics_summary.json"
    metrics_payload = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    manifest_path = root / "pipeline_manifest.json"
    manifest_payload = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    original_request_metadata = _request_metadata(manifest_payload)
    audit_payload = {
        "repair_id": repair_id,
        "repair_key": repair_key,
        "canonicalized_at": datetime.now(timezone.utc).isoformat(),
        "original_request_metadata": original_request_metadata,
        "runtime_policy": "explicit_api_2.5_plus",
    }

    metrics_payload.update(runtime_payload)
    metrics_payload["train"] = train_metrics
    metrics_payload["ood_test"] = test_metrics
    metrics_payload["repair_audit"] = audit_payload
    _atomic_json_write(metrics_path, metrics_payload)

    manifest_payload.update(runtime_payload)
    manifest_payload.update(
        {
            "split_artifacts": _current_split_artifacts(entry, root),
            "split_summary": entry.split_summary,
            "canonical_split_identity": entry.identity_record(),
            "canonical_train_full_hash": _frame_content_hash(entry.load_train()),
            "canonical_test_full_hash": _frame_content_hash(entry.load_test()),
            "parameter_snapshot_id": snapshot_id,
            "parameter_hash": str(frozen_entry.get("parameter_hash", "")),
            "frozen_parameter_file": str(Path(snapshot_path).resolve()),
            "repair_key": repair_key,
            "repair_audit": audit_payload,
        }
    )
    _atomic_json_write(manifest_path, manifest_payload)
    return {"train": train_metrics, "ood_test": test_metrics}


def _repair_key(model: str, entry: CanonicalSplitEntry) -> str:
    return "||".join((model, entry.task_id, entry.protocol, entry.split_id))


def _bert_model_spec(model: str) -> dict[str, str]:
    for spec in BERT_MODEL_SPECS:
        if spec["model"] == str(model):
            return spec
    raise ValueError(f"Unsupported BERT model: {model}")


def _standard_task_row(
    entry: CanonicalSplitEntry,
    model_spec: dict[str, str],
    workspace: Path,
) -> dict[str, Any]:
    method_meta = OOD_METHODS[entry.protocol]
    output_root, _ = _bert_result_paths(entry, model_spec, workspace)
    config_name = f"{model_spec['experiment_prefix']}_{method_meta['config_suffix']}"
    queue_id = "||".join((config_name, entry.alloy_type, entry.target_column))
    return {
        "queue_id": queue_id,
        "config_name": config_name,
        "alloy_type": entry.alloy_type,
        "dataset_name": entry.dataset_name,
        "target_column": entry.target_column,
        "task_id": entry.task_id,
        "protocol": entry.protocol,
        "model_family": "BERT",
        "model": model_spec["model"],
        "embedding_type": model_spec["model_dirname"],
        "batch_task_key": f"{entry.alloy_type}_{entry.target_column}",
        "output_root": str(output_root),
        "expected_output_root": str(output_root),
    }


def build_standard_bert_task_inventory(
    *,
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> pd.DataFrame:
    """Return one row for every standard BERT task/protocol/model run."""

    workspace = Path(workspace_root).resolve()
    representative: dict[tuple[str, str], CanonicalSplitEntry] = {}
    for entry in registry.entries():
        representative.setdefault((entry.task_id, entry.protocol), entry)
    rows = [
        _standard_task_row(entry, spec, workspace)
        for entry in representative.values()
        for spec in BERT_MODEL_SPECS
    ]
    return pd.DataFrame(rows).sort_values(
        ["config_name", "task_id", "model"], kind="stable"
    ).reset_index(drop=True)


def build_standard_bert_rerun_queue(
    repair_queue: pd.DataFrame,
    *,
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> pd.DataFrame:
    """Collapse split-level BERT errors into atomic standard batch tasks."""

    required = {"repair_key", "task_id", "protocol", "split_id", "model_family", "model"}
    missing = required - set(repair_queue.columns)
    if missing:
        raise ValueError(f"Repair queue is missing required columns: {sorted(missing)}")
    selected = repair_queue[
        repair_queue["model_family"].astype(str).eq("BERT")
        & repair_queue["protocol"].astype(str).isin(CANONICAL_LAYOUT_PROTOCOLS)
    ].copy()
    workspace = Path(workspace_root).resolve()
    rows: list[dict[str, Any]] = []
    grouping = ["task_id", "protocol", "model"]
    for (task_id, protocol, model), group in selected.groupby(grouping, sort=True):
        first_split = sorted(group["split_id"].astype(str))[0]
        entry = registry.require(str(task_id), str(protocol), first_split)
        row = _standard_task_row(entry, _bert_model_spec(str(model)), workspace)
        row.update(
            {
                "source_error_unit_count": len(group),
                "rerun_split_unit_count": 5 if protocol == "loco" else 1,
                "source_repair_keys": ";".join(sorted(group["repair_key"].astype(str))),
            }
        )
        rows.append(row)
    queue = pd.DataFrame(rows).sort_values(
        ["config_name", "task_id", "model"], kind="stable"
    ).reset_index(drop=True)
    if queue["queue_id"].duplicated().any():
        duplicates = queue.loc[queue["queue_id"].duplicated(False), "queue_id"].tolist()
        raise ValueError(f"Standard BERT rerun queue contains duplicate tasks: {duplicates}")
    return queue


def build_protected_correct_tasks(
    inventory: pd.DataFrame,
    rerun_queue: pd.DataFrame,
) -> pd.DataFrame:
    rerun_ids = set(rerun_queue["queue_id"].astype(str))
    protected = inventory[~inventory["queue_id"].astype(str).isin(rerun_ids)].copy()
    return protected.sort_values(
        ["config_name", "task_id", "model"], kind="stable"
    ).reset_index(drop=True)


def _key_artifact_hash(root: Path) -> str:
    if not root.exists():
        return "missing"
    selected: list[Path] = []
    pruned_dirs = {"features", "checkpoints", "plots", "loss", "logs", "optuna_trials"}
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name.lower() not in pruned_dirs]
        current_path = Path(current)
        for filename in files:
            path = current_path / filename
            relative = path.relative_to(root)
            name = filename.lower()
            is_final_prediction = name.endswith("all_predictions.csv")
            if (
                name.endswith("manifest.json")
                or name == "final_evaluation_metrics.json"
                or is_final_prediction
                or "split_data" in {part.lower() for part in relative.parts}
            ):
                selected.append(path)
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda value: str(value.relative_to(root)).lower()):
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def add_task_artifact_fingerprints(tasks: pd.DataFrame) -> pd.DataFrame:
    enriched = tasks.copy()
    roots = [Path(value) for value in enriched["output_root"].astype(str)]
    enriched["exists"] = [root.exists() for root in roots]
    enriched["artifact_hash"] = [_key_artifact_hash(root) for root in roots]
    return enriched


def _directory_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def execute_standard_rerun_deletion(
    queue: pd.DataFrame,
    *,
    workspace_root: str | Path,
    execute: bool,
) -> pd.DataFrame:
    """Delete only queue-approved BERT task roots below output/ood_results."""

    workspace = Path(workspace_root).resolve()
    official_root = (workspace / "output" / "ood_results").resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in queue.iterrows():
        target = _lexical_absolute(str(row["output_root"]))
        expected = _lexical_absolute(str(row["expected_output_root"]))
        if _path_key(target) != _path_key(expected):
            raise ValueError(f"Deletion path does not match the standard batch path: {target} != {expected}")
        try:
            target.relative_to(official_root)
        except ValueError as exc:
            raise ValueError(f"Deletion target is outside output/ood_results: {target}") from exc
        if _path_key(target) == _path_key(official_root):
            raise ValueError("Refusing to delete the output/ood_results root")
        resolved = target.resolve() if target.exists() else target
        try:
            resolved.relative_to(official_root)
        except ValueError as exc:
            raise ValueError(f"Deletion target resolves outside output/ood_results: {target} -> {resolved}") from exc
        key = _path_key(target)
        if key in seen:
            raise ValueError(f"Duplicate deletion target: {target}")
        seen.add(key)
        exists_before = target.exists()
        file_count, total_bytes = _directory_stats(target)
        if not exists_before:
            status = "already_missing"
        elif not execute:
            status = "planned"
        else:
            shutil.rmtree(target)
            status = "deleted"
        records.append(
            {
                **row.to_dict(),
                "exists_before": exists_before,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "deletion_status": status,
                "exists_after": target.exists(),
            }
        )
    return pd.DataFrame(records)


def _validate_standard_rerun_counts(
    queue: pd.DataFrame,
    protected: pd.DataFrame,
) -> None:
    expected_models = {"MatSciBERT": 43, "SciBERT": 36, "SteelBERT": 36}
    if len(queue) != 115:
        raise ValueError(f"Standard BERT rerun queue must contain 115 tasks, found {len(queue)}")
    if len(protected) != 95:
        raise ValueError(f"Protected BERT task inventory must contain 95 tasks, found {len(protected)}")
    if int(queue["protocol"].astype(str).eq("loco").sum()) != 21:
        raise ValueError("Standard BERT rerun queue must contain 21 LOCO tasks")
    if int(queue["protocol"].astype(str).str.startswith("sparse_").sum()) != 94:
        raise ValueError("Standard BERT rerun queue must contain 94 Sparse tasks")
    model_counts = queue.groupby("model").size().to_dict()
    if model_counts != expected_models:
        raise ValueError(f"Standard BERT rerun model counts are invalid: {model_counts}")
    if int(queue["rerun_split_unit_count"].astype(int).sum()) != 199:
        raise ValueError("Standard BERT rerun queue must expand to 199 split units")
    if set(queue["queue_id"].astype(str)) & set(protected["queue_id"].astype(str)):
        raise ValueError("Rerun and protected task inventories overlap")


def _assert_protected_tasks_unchanged(protected: pd.DataFrame) -> None:
    current = add_task_artifact_fingerprints(protected.drop(columns=["exists", "artifact_hash"], errors="ignore"))
    expected = protected.set_index("queue_id")
    actual = current.set_index("queue_id")
    changed: list[str] = []
    for queue_id in expected.index:
        if queue_id not in actual.index:
            changed.append(str(queue_id))
            continue
        if bool(expected.at[queue_id, "exists"]) != bool(actual.at[queue_id, "exists"]):
            changed.append(str(queue_id))
            continue
        if str(expected.at[queue_id, "artifact_hash"]) != str(actual.at[queue_id, "artifact_hash"]):
            changed.append(str(queue_id))
    if changed:
        raise ValueError(f"Protected BERT tasks changed: {changed[:20]}")


def run_prepare_standard_rerun_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output, manifest, locked_queue, _ = _load_locked_state(workspace, output_root)
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    bert_errors = locked_queue[locked_queue["model_family"].astype(str).eq("BERT")].copy()
    if len(bert_errors) != 196:
        raise ValueError(f"Locked repair queue must contain 196 BERT split units, found {len(bert_errors)}")
    queue = build_standard_bert_rerun_queue(
        bert_errors,
        registry=registry,
        workspace_root=workspace,
    )
    inventory = build_standard_bert_task_inventory(registry=registry, workspace_root=workspace)
    protected = add_task_artifact_fingerprints(build_protected_correct_tasks(inventory, queue))
    _validate_standard_rerun_counts(queue, protected)
    if not protected["exists"].astype(bool).all():
        missing = protected.loc[~protected["exists"].astype(bool), "output_root"].tolist()
        raise FileNotFoundError(f"Protected correct BERT task outputs are missing: {missing[:20]}")
    deletion = execute_standard_rerun_deletion(
        queue,
        workspace_root=workspace,
        execute=False,
    )
    queue.to_csv(output / "bert_standard_rerun_queue.csv", index=False)
    protected.to_csv(output / "protected_correct_tasks.csv", index=False)
    deletion.to_csv(output / "deletion_manifest.csv", index=False)
    manifest.update(
        {
            "standard_rerun_prepared_at": datetime.now(timezone.utc).isoformat(),
            "standard_rerun_task_count": len(queue),
            "standard_rerun_split_unit_count": int(queue["rerun_split_unit_count"].sum()),
            "protected_correct_task_count": len(protected),
            "standard_rerun_queue_hash": _stable_frame_hash(queue),
            "standard_rerun_deleted": False,
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return {
        "queue_count": len(queue),
        "protected_count": len(protected),
        "rerun_split_units": int(queue["rerun_split_unit_count"].sum()),
        "deletion_statuses": dict(Counter(deletion["deletion_status"].astype(str))),
    }


def run_delete_standard_rerun_stage(
    *,
    workspace_root: str | Path,
    output_root: str | Path,
    execute: bool,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    queue_path = output / "bert_standard_rerun_queue.csv"
    protected_path = output / "protected_correct_tasks.csv"
    manifest_path = output / "manifest.json"
    for path in (queue_path, protected_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Run prepare-standard-rerun first; missing: {path}")
    queue = pd.read_csv(queue_path, keep_default_na=False)
    protected = pd.read_csv(protected_path, keep_default_na=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _stable_frame_hash(queue) != str(manifest.get("standard_rerun_queue_hash", "")):
        raise ValueError("bert_standard_rerun_queue.csv does not match the locked manifest hash")
    _validate_standard_rerun_counts(queue, protected)
    _assert_protected_tasks_unchanged(protected)
    result = execute_standard_rerun_deletion(
        queue,
        workspace_root=workspace,
        execute=execute,
    )
    result.to_csv(output / "deletion_manifest.csv", index=False)
    _assert_protected_tasks_unchanged(protected)
    if execute:
        remaining = [path for path in queue["output_root"].astype(str) if Path(path).exists()]
        if remaining:
            raise ValueError(f"Queued BERT outputs remain after deletion: {remaining[:20]}")
        manifest["standard_rerun_deleted_at"] = datetime.now(timezone.utc).isoformat()
        manifest["standard_rerun_deleted"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return result


def build_existing_result_audit(
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        for spec in BERT_MODEL_SPECS:
            parent_root, run_root = _bert_result_paths(entry, spec, workspace)
            audit = audit_result_split(entry, run_root)
            rows.append(
                {
                    **audit,
                    "repair_key": _repair_key(spec["model"], entry),
                    "model_family": spec["model_family"],
                    "model": spec["model"],
                    "model_dirname": spec["model_dirname"],
                    "feature_mode": "",
                    "parent_result_root": str(parent_root),
                    "archive_source": str(run_root),
                    "archive_scope": "fold" if entry.split_id != "main" else "model_run",
                    "official_output_root": str((workspace / "output" / "ood_results").resolve()),
                }
            )
        for spec in TABPFN_MODEL_SPECS:
            parent_root, run_root = _tabpfn_result_paths(entry, spec, workspace)
            audit = audit_result_split(entry, run_root)
            official_root = workspace / "output" / spec["output_dirname"]
            rows.append(
                {
                    **audit,
                    "repair_key": _repair_key(spec["model"], entry),
                    "model_family": spec["model_family"],
                    "model": spec["model"],
                    "model_dirname": "",
                    "feature_mode": spec["feature_mode"],
                    "parent_result_root": str(parent_root),
                    "archive_source": str(run_root),
                    "archive_scope": "fold" if entry.split_id != "main" else "task_run",
                    "official_output_root": str(official_root.resolve()),
                }
            )
    columns = [
        "repair_key",
        "task_id",
        "protocol",
        "split_id",
        "model_family",
        "model",
        "model_dirname",
        "feature_mode",
        "status",
        "reason",
        "reported_cluster_count",
        "run_root",
        "parent_result_root",
        "archive_source",
        "archive_scope",
        "official_output_root",
        "train_file",
        "test_file",
        "canonical_train_hash",
        "canonical_test_hash",
        "result_train_hash",
        "result_test_hash",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].sort_values(
        ["model_family", "model", "task_id", "protocol", "split_id"],
        kind="stable",
    ).reset_index(drop=True)


def _legacy_bert_model_root(
    entry: CanonicalSplitEntry,
    model_spec: dict[str, str],
    workspace_root: Path,
) -> Path:
    method_meta = OOD_METHODS[entry.protocol]
    alloy_dir = "HEA_half" if entry.alloy_type == "HEA" else entry.alloy_type
    legacy_method_dir = str(method_meta["result_dir_suffix"]).removesuffix("_k5")
    return _lexical_absolute(
        workspace_root
        / "output"
        / "ood_results"
        / f"{model_spec['experiment_prefix']}_{method_meta['config_suffix']}"
        / alloy_dir
        / entry.dataset_name
        / entry.target_column
        / legacy_method_dir
        / model_spec["model_dirname"]
    )


def _layout_cluster_and_test_counts(
    model_root: Path,
    entries: list[CanonicalSplitEntry],
) -> tuple[list[int], list[int]]:
    cluster_counts: set[int] = set()
    test_counts: set[int] = set()
    for entry in entries:
        run_root = model_root
        if entry.split_id != "main":
            run_root = model_root / "folds" / entry.split_id
        summary_path = run_root / "split_data" / "split_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            reported = _reported_cluster_count(entry, summary)
            if reported is not None:
                cluster_counts.add(int(reported))
        test_path = run_root / "split_data" / entry.test_file.name
        if test_path.exists():
            try:
                test_counts.add(len(pd.read_csv(test_path)))
            except Exception:
                pass
    return sorted(cluster_counts), sorted(test_counts)


def build_bert_legacy_layout_audit(
    registry: CanonicalSplitRegistry,
    workspace_root: str | Path,
) -> pd.DataFrame:
    workspace = _lexical_absolute(workspace_root)
    grouped: dict[tuple[str, str], list[CanonicalSplitEntry]] = {}
    for entry in registry.entries():
        if entry.protocol not in CANONICAL_LAYOUT_PROTOCOLS:
            continue
        grouped.setdefault((entry.task_id, entry.protocol), []).append(entry)

    rows: list[dict[str, Any]] = []
    for entries in grouped.values():
        entries = sorted(entries, key=lambda item: item.split_id)
        representative = entries[0]
        for spec in BERT_MODEL_SPECS:
            canonical_root, _ = _bert_result_paths(representative, spec, workspace)
            legacy_root = _legacy_bert_model_root(representative, spec, workspace)
            unit_audits = []
            for entry in entries:
                run_root = canonical_root
                if entry.split_id != "main":
                    run_root = canonical_root / "folds" / entry.split_id
                unit_audits.append(audit_result_split(entry, run_root))

            canonical_exists = canonical_root.exists()
            legacy_exists = legacy_root.exists()
            canonical_resolved = canonical_root.resolve() if canonical_exists else canonical_root
            legacy_resolved = legacy_root.resolve() if legacy_exists else legacy_root
            same_path = _path_key(canonical_root) == _path_key(legacy_root)
            same_target = (
                canonical_exists
                and legacy_exists
                and _path_key(canonical_resolved) == _path_key(legacy_resolved)
            )
            canonical_redirect = _is_directory_redirect(canonical_root)
            legacy_redirect = _is_directory_redirect(legacy_root)
            exact_count = sum(audit["status"] == "reuse_exact" for audit in unit_audits)

            if same_path:
                layout_status = "single_path"
            elif not canonical_exists:
                layout_status = "missing_canonical"
            elif exact_count != len(entries):
                layout_status = "invalid_canonical"
            elif not legacy_exists:
                layout_status = "canonical_only"
            elif same_target and canonical_redirect and not legacy_redirect:
                layout_status = "canonical_redirects_to_legacy"
            elif same_target and legacy_redirect and not canonical_redirect:
                layout_status = "legacy_redirects_to_canonical"
            elif same_target:
                layout_status = "shared_redirect_target"
            else:
                layout_status = "duplicate_distinct"

            canonical_clusters, canonical_tests = _layout_cluster_and_test_counts(
                canonical_root,
                entries,
            )
            legacy_clusters, legacy_tests = _layout_cluster_and_test_counts(
                legacy_root,
                entries,
            )
            experiment_root = workspace / "output" / "ood_results" / canonical_root.relative_to(
                workspace / "output" / "ood_results"
            ).parts[0]
            rows.append(
                {
                    "layout_key": "||".join(
                        (spec["model"], representative.task_id, representative.protocol)
                    ),
                    "task_id": representative.task_id,
                    "protocol": representative.protocol,
                    "model": spec["model"],
                    "canonical_path": str(canonical_root),
                    "canonical_resolved_path": str(canonical_resolved),
                    "canonical_exists": canonical_exists,
                    "canonical_is_redirect": canonical_redirect,
                    "canonical_unit_count": len(entries),
                    "canonical_reuse_exact_count": exact_count,
                    "canonical_cluster_counts": ";".join(map(str, canonical_clusters)),
                    "canonical_test_counts": ";".join(map(str, canonical_tests)),
                    "legacy_path": str(legacy_root),
                    "legacy_resolved_path": str(legacy_resolved),
                    "legacy_exists": legacy_exists,
                    "legacy_is_redirect": legacy_redirect,
                    "legacy_cluster_counts": ";".join(map(str, legacy_clusters)),
                    "legacy_test_counts": ";".join(map(str, legacy_tests)),
                    "experiment_root": str(experiment_root),
                    "layout_status": layout_status,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "task_id", "protocol"],
        kind="stable",
    ).reset_index(drop=True)


def build_legacy_layout_plan(
    layout_audit: pd.DataFrame,
    *,
    workspace_root: str | Path,
    archive_root: str | Path,
    repair_id: str,
) -> pd.DataFrame:
    workspace = _lexical_absolute(workspace_root)
    official_root = workspace / "output" / "ood_results"
    archive = _lexical_absolute(archive_root)
    rows: list[dict[str, Any]] = []
    for _, source_row in layout_audit.iterrows():
        canonical = _lexical_absolute(source_row["canonical_path"])
        legacy = _lexical_absolute(source_row["legacy_path"])
        if _path_key(canonical) == _path_key(legacy) or not legacy.exists():
            continue
        for candidate in (canonical, legacy):
            try:
                candidate.relative_to(official_root)
            except ValueError as exc:
                raise ValueError(
                    f"Layout path is outside the official BERT result root: {candidate}"
                ) from exc
        expected_units = int(source_row.get("canonical_unit_count", 0))
        exact_units = int(source_row.get("canonical_reuse_exact_count", 0))
        if expected_units <= 0 or exact_units != expected_units:
            raise ValueError(
                f"Canonical result is not fully verified for layout migration: "
                f"{source_row.get('layout_key', canonical)}"
            )
        if not canonical.exists():
            raise FileNotFoundError(f"Canonical result path does not exist: {canonical}")

        canonical_resolved = canonical.resolve()
        legacy_resolved = legacy.resolve()
        if (
            _path_key(canonical_resolved) == _path_key(legacy_resolved)
            and _is_directory_redirect(canonical)
            and not _is_directory_redirect(legacy)
        ):
            action = "promote_legacy"
            archive_path = ""
        elif _path_key(canonical_resolved) != _path_key(legacy_resolved):
            action = "archive_legacy"
            relative_legacy = legacy.relative_to(workspace)
            archive_path = str(
                archive
                / repair_id
                / "legacy_layout_cleanup"
                / relative_legacy
            )
        elif _is_directory_redirect(legacy):
            continue
        else:
            raise ValueError(
                f"Ambiguous canonical/legacy layout relationship: {canonical} <-> {legacy}"
            )
        rows.append(
            {
                **dict(source_row),
                "action": action,
                "archive_path": archive_path,
                "official_output_root": str(official_root),
            }
        )
    return pd.DataFrame(rows)


def _prune_empty_layout_parents(start: Path, stop: Path) -> None:
    current = start
    while _path_key(current) != _path_key(stop):
        try:
            current.relative_to(stop)
        except ValueError:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _create_directory_redirect(link: Path, target: Path) -> None:
    if not target.exists() or _is_directory_redirect(target):
        raise ValueError(f"Redirect target must be a physical result directory: {target}")
    if link.exists():
        if _is_directory_redirect(link) and _path_key(link.resolve()) == _path_key(target):
            return
        raise FileExistsError(f"Compatibility path already exists with another target: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link, target_is_directory=True)


def execute_legacy_layout_plan(plan: pd.DataFrame, *, execute: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        record = dict(row)
        canonical = _lexical_absolute(row["canonical_path"])
        legacy = _lexical_absolute(row["legacy_path"])
        official_root = _lexical_absolute(row["official_output_root"])
        experiment_root = _lexical_absolute(
            row.get("experiment_root", official_root / canonical.relative_to(official_root).parts[0])
        )
        for candidate in (canonical, legacy):
            try:
                candidate.relative_to(official_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe layout migration path: {candidate}") from exc

        action = str(row["action"])
        if not execute:
            status = "planned"
        elif action == "archive_legacy":
            archive_path = _lexical_absolute(row["archive_path"])
            if (
                legacy.exists()
                and _is_directory_redirect(legacy)
                and _path_key(legacy.resolve()) == _path_key(canonical)
                and archive_path.exists()
            ):
                status = "already_archived_and_linked"
            elif legacy.exists() and archive_path.exists():
                raise FileExistsError(
                    f"Both legacy source and archive destination exist: {legacy} -> {archive_path}"
                )
            if not legacy.exists() and archive_path.exists():
                _create_directory_redirect(legacy, canonical)
                status = "linked_archived_legacy"
            elif not legacy.exists():
                raise FileNotFoundError(f"Legacy source disappeared before archive: {legacy}")
            elif not _is_directory_redirect(legacy):
                if not canonical.exists() or _is_directory_redirect(canonical):
                    raise ValueError(f"Canonical path is not a physical result directory: {canonical}")
                if _path_key(canonical.resolve()) == _path_key(legacy.resolve()):
                    raise ValueError(f"Refusing to archive the canonical result target: {legacy}")
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                legacy.rename(archive_path)
                _prune_empty_layout_parents(legacy.parent, experiment_root)
                _create_directory_redirect(legacy, canonical)
                status = "archived_legacy"
        elif action == "promote_legacy":
            if (
                canonical.exists()
                and not _is_directory_redirect(canonical)
                and legacy.exists()
                and _is_directory_redirect(legacy)
                and _path_key(legacy.resolve()) == _path_key(canonical)
            ):
                status = "already_promoted_and_linked"
            elif canonical.exists() and not _is_directory_redirect(canonical) and not legacy.exists():
                _create_directory_redirect(legacy, canonical)
                status = "linked_promoted_legacy"
            else:
                if canonical.exists():
                    if not _is_directory_redirect(canonical):
                        raise ValueError(f"Canonical path is unexpectedly physical: {canonical}")
                    if not legacy.exists() or _path_key(canonical.resolve()) != _path_key(legacy.resolve()):
                        raise ValueError(
                            f"Canonical redirect does not target the planned legacy result: {canonical}"
                        )
                    canonical.rmdir()
                if not legacy.exists():
                    raise FileNotFoundError(f"Legacy canonical source is missing: {legacy}")
                canonical.parent.mkdir(parents=True, exist_ok=True)
                legacy.rename(canonical)
                _prune_empty_layout_parents(legacy.parent, experiment_root)
                _create_directory_redirect(legacy, canonical)
                status = "promoted_legacy"
        else:
            raise ValueError(f"Unsupported layout migration action: {action}")

        record["migration_status"] = status
        record["canonical_exists_after"] = canonical.exists()
        record["canonical_is_redirect_after"] = _is_directory_redirect(canonical)
        record["legacy_exists_after"] = legacy.exists()
        record["legacy_is_redirect_after"] = _is_directory_redirect(legacy)
        archive_value = str(row.get("archive_path", "")).strip()
        record["archive_exists_after"] = bool(
            archive_value and _lexical_absolute(archive_value).exists()
        )
        records.append(record)
    return pd.DataFrame(records)


def build_repair_queue(existing_result_audit: pd.DataFrame) -> pd.DataFrame:
    queue = existing_result_audit[
        existing_result_audit["status"].astype(str).eq("rerun_required")
    ].copy()
    if queue["repair_key"].duplicated().any():
        duplicates = sorted(queue.loc[queue["repair_key"].duplicated(False), "repair_key"].unique())
        raise ValueError(f"Duplicate repair keys: {duplicates}")
    if queue["archive_source"].duplicated().any():
        duplicates = sorted(queue.loc[queue["archive_source"].duplicated(False), "archive_source"].unique())
        raise ValueError(f"Multiple repair rows point to the same archive source: {duplicates}")
    return queue.reset_index(drop=True)


def _resolve_against(path_value: object, base: Path) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def build_archive_plan(
    repair_queue: pd.DataFrame,
    *,
    workspace_root: str | Path,
    archive_root: str | Path,
    repair_id: str,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    archive = _resolve_against(archive_root, workspace)
    if not repair_id.strip():
        raise ValueError("repair_id must not be empty")

    grouped: dict[Path, dict[str, Any]] = {}
    for _, row in repair_queue.iterrows():
        source = _resolve_against(row["archive_source"], workspace)
        official_root = _resolve_against(row["official_output_root"], workspace)
        try:
            source.relative_to(official_root)
        except ValueError as exc:
            raise ValueError(
                f"Archive source is outside its official output root: {source}"
            ) from exc
        try:
            relative_source = source.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Archive source is outside the workspace: {source}") from exc
        destination = (archive / repair_id / relative_source).resolve()
        try:
            destination.relative_to((archive / repair_id).resolve())
        except ValueError as exc:
            raise ValueError(f"Archive destination escaped the repair root: {destination}") from exc
        record = grouped.setdefault(
            source,
            {
                "source_path": str(source),
                "destination_path": str(destination),
                "source_exists": source.exists(),
                "destination_exists": destination.exists(),
                "repair_keys": [],
            },
        )
        record["repair_keys"].append(str(row["repair_key"]))

    rows = []
    for source in sorted(grouped, key=str):
        record = grouped[source]
        record["repair_keys"] = ";".join(sorted(set(record["repair_keys"])))
        rows.append(record)
    return pd.DataFrame(rows)


def execute_archive_plan(plan: pd.DataFrame, *, execute: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        source = Path(str(row["source_path"])).resolve()
        destination = Path(str(row["destination_path"])).resolve()
        record = dict(row)
        source_exists = source.exists()
        destination_exists = destination.exists()
        if source_exists and destination_exists:
            raise FileExistsError(
                f"Both archive source and destination exist; refusing to overwrite: {source} -> {destination}"
            )
        if not source_exists and destination_exists:
            record["archive_status"] = "already_moved"
        elif not source_exists:
            record["archive_status"] = "missing_source"
        elif not execute:
            record["archive_status"] = "planned"
        else:
            if source.anchor.lower() != destination.anchor.lower():
                raise ValueError(f"Archive move must stay on the same volume: {source} -> {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            record["archive_status"] = "moved"
        record["source_exists_after"] = source.exists()
        record["destination_exists_after"] = destination.exists()
        records.append(record)
    return pd.DataFrame(records)


def _canonical_audit(registry: CanonicalSplitRegistry) -> pd.DataFrame:
    rows = []
    for entry in registry.entries():
        identity = entry.identity_record()
        expected_k = None
        reported_k = None
        if entry.protocol == "loco":
            expected_k = entry.method_params.get("loco_cluster_count")
            reported_k = entry.split_summary.get("loco_cluster_count_requested")
        elif entry.protocol.startswith("sparse_"):
            expected_k = entry.method_params.get("sparse_cluster_count")
            reported_k = entry.split_summary.get("sparse_cluster_count_requested")
        rows.append(
            {
                **identity,
                "alloy_type": entry.alloy_type,
                "dataset_name": entry.dataset_name,
                "random_state": entry.random_state,
                "method_params_json": json.dumps(
                    entry.method_params,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "expected_cluster_count": expected_k,
                "reported_cluster_count": reported_k,
                "canonical_status": (
                    "valid"
                    if expected_k is None or int(expected_k) == int(reported_k)
                    else "invalid"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["task_id", "protocol", "split_id"],
        kind="stable",
    ).reset_index(drop=True)


def _stable_frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.fillna("").astype(str)
    payload = normalized.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_expected_repair_matrix(
    canonical_audit: pd.DataFrame,
    existing_audit: pd.DataFrame,
    repair_queue: pd.DataFrame,
) -> None:
    if len(canonical_audit) != 150:
        raise ValueError(f"Canonical split count must be 150, found {len(canonical_audit)}")
    if set(canonical_audit["canonical_status"].astype(str)) != {"valid"}:
        raise ValueError("Canonical split audit contains invalid entries")
    if len(existing_audit) != 750:
        raise ValueError(f"Existing BERT/TabPFN audit count must be 750, found {len(existing_audit)}")
    if len(repair_queue) != 208:
        raise ValueError(f"Repair queue must contain exactly 208 rows, found {len(repair_queue)}")
    model_counts = Counter(repair_queue["model"].astype(str))
    if model_counts != Counter(EXPECTED_REPAIR_COUNTS):
        raise ValueError(f"Repair model counts do not match the approved matrix: {model_counts}")
    fake_k5 = Counter(
        repair_queue.loc[
            repair_queue["reason"].astype(str).eq("invalid_k5_parameters"),
            "model",
        ].astype(str)
    )
    if fake_k5 != Counter(EXPECTED_FAKE_K5_COUNTS):
        raise ValueError(f"Fake-K5 counts do not match the approved matrix: {fake_k5}")


def _initial_repair_status(queue: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "repair_key",
        "task_id",
        "protocol",
        "split_id",
        "model_family",
        "model",
        "run_root",
    ]
    status = queue[columns].copy()
    status["archive_status"] = "pending"
    status["run_status"] = "pending"
    status["verification_status"] = "pending"
    status["attempt_count"] = 0
    status["started_at"] = ""
    status["finished_at"] = ""
    status["last_error"] = ""
    return status


def apply_legacy_layout_path_map(
    frame: pd.DataFrame,
    layout_plan: pd.DataFrame,
) -> pd.DataFrame:
    mapped = frame.copy()
    path_columns = ("run_root", "parent_result_root", "train_file", "test_file")
    for _, layout_row in layout_plan.iterrows():
        mask = (
            mapped["model"].astype(str).eq(str(layout_row["model"]))
            & mapped["task_id"].astype(str).eq(str(layout_row["task_id"]))
            & mapped["protocol"].astype(str).eq(str(layout_row["protocol"]))
        )
        if not mask.any():
            continue
        legacy = _lexical_absolute(layout_row["legacy_path"])
        canonical = _lexical_absolute(layout_row["canonical_path"])
        for column in path_columns:
            if column not in mapped.columns:
                continue
            for index in mapped.index[mask]:
                value = str(mapped.at[index, column]).strip()
                if not value:
                    continue
                candidate = _lexical_absolute(value)
                try:
                    relative = candidate.relative_to(legacy)
                except ValueError:
                    continue
                mapped.at[index, column] = str(canonical / relative)
    return mapped


def run_audit_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    output.mkdir(parents=True, exist_ok=True)
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    canonical = _canonical_audit(registry)
    existing = build_existing_result_audit(registry, workspace)
    queue = build_repair_queue(existing)
    validate_expected_repair_matrix(canonical, existing, queue)

    queue_hash = _stable_frame_hash(queue)
    manifest_path = output / "manifest.json"
    prior_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prior_manifest.get("repair_queue_hash") == queue_hash:
        repair_id = str(prior_manifest["repair_id"])
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        repair_id = f"canonical_ood_{stamp}_{queue_hash[:12]}"

    canonical.to_csv(output / "canonical_audit.csv", index=False)
    existing.to_csv(output / "existing_result_audit.csv", index=False)
    queue.to_csv(output / "repair_queue.csv", index=False)
    status_path = output / "repair_status.csv"
    if not status_path.exists() or prior_manifest.get("repair_queue_hash") != queue_hash:
        _initial_repair_status(queue).to_csv(status_path, index=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "stage": "audit",
        "generated_at": generated_at,
        "workspace_root": str(workspace),
        "canonical_split_root": str(_resolve_against(split_root, workspace)),
        "output_root": str(output),
        "repair_id": repair_id,
        "repair_queue_hash": queue_hash,
        "canonical_split_count": len(canonical),
        "existing_result_audit_count": len(existing),
        "repair_count": len(queue),
        "repair_counts_by_model": dict(Counter(queue["model"].astype(str))),
        "fake_k5_counts_by_model": dict(
            Counter(
                queue.loc[
                    queue["reason"].astype(str).eq("invalid_k5_parameters"),
                    "model",
                ].astype(str)
            )
        ),
        "archive_executed": bool(prior_manifest.get("archive_executed", False)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def _load_locked_state(
    workspace_root: str | Path,
    output_root: str | Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    manifest_path = output / "manifest.json"
    queue_path = output / "repair_queue.csv"
    status_path = output / "repair_status.csv"
    for path in (manifest_path, queue_path, status_path):
        if not path.exists():
            raise FileNotFoundError(f"Run the audit stage first; missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = pd.read_csv(queue_path, keep_default_na=False)
    if _stable_frame_hash(queue) != str(manifest.get("repair_queue_hash", "")):
        raise ValueError("repair_queue.csv no longer matches the locked manifest hash")
    status = pd.read_csv(status_path, keep_default_na=False)
    if set(status["repair_key"].astype(str)) != set(queue["repair_key"].astype(str)):
        raise ValueError("repair_status.csv keys do not match the locked repair queue")
    layout_plan_path = output / "legacy_layout_plan.csv"
    if manifest.get("legacy_layout_complete") and layout_plan_path.exists():
        layout_plan = pd.read_csv(layout_plan_path, keep_default_na=False)
        queue = apply_legacy_layout_path_map(queue, layout_plan)
        status = apply_legacy_layout_path_map(status, layout_plan)
    return output, manifest, queue, status


def _write_status(status: pd.DataFrame, output_root: Path) -> None:
    status.sort_values(
        ["model_family", "model", "task_id", "protocol", "split_id"],
        kind="stable",
    ).to_csv(output_root / "repair_status.csv", index=False)


def run_archive_stage(
    *,
    workspace_root: str | Path,
    output_root: str | Path,
    archive_root: str | Path,
    execute: bool,
) -> pd.DataFrame:
    output, manifest, queue, status = _load_locked_state(workspace_root, output_root)
    plan = build_archive_plan(
        queue,
        workspace_root=workspace_root,
        archive_root=archive_root,
        repair_id=str(manifest["repair_id"]),
    )
    result = execute_archive_plan(plan, execute=execute)
    result.to_csv(output / "archive_manifest.csv", index=False)

    status_by_key = status.set_index("repair_key", drop=False)
    for _, archive_row in result.iterrows():
        for repair_key in str(archive_row["repair_keys"]).split(";"):
            if repair_key in status_by_key.index:
                status_by_key.loc[repair_key, "archive_status"] = str(
                    archive_row["archive_status"]
                )
    status = status_by_key.reset_index(drop=True)
    _write_status(status, output)

    if execute:
        invalid = result[
            ~result["archive_status"].astype(str).isin({"moved", "already_moved"})
        ]
        if not invalid.empty:
            raise ValueError(
                "Archive execution did not complete for all repair paths: "
                + ", ".join(invalid["source_path"].astype(str).tolist())
            )
        manifest["archive_executed"] = True
        manifest["archive_completed_at"] = datetime.now(timezone.utc).isoformat()
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    return result


def run_canonicalize_layout_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    archive_root: str | Path,
    execute: bool,
) -> pd.DataFrame:
    workspace = _lexical_absolute(workspace_root)
    output = _resolve_against(output_root, workspace)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repair_id = str(manifest.get("repair_id", "")).strip()
    if not repair_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        repair_id = f"canonical_ood_layout_{stamp}"

    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    audit = build_bert_legacy_layout_audit(registry, workspace)
    audit.to_csv(output / "legacy_layout_audit.csv", index=False)
    invalid = audit[
        audit["canonical_reuse_exact_count"].astype(int)
        != audit["canonical_unit_count"].astype(int)
    ]
    if not invalid.empty:
        raise ValueError(
            f"Canonical BERT layout contains {len(invalid)} unverified task/model roots"
        )

    fresh_plan = build_legacy_layout_plan(
        audit,
        workspace_root=workspace,
        archive_root=archive_root,
        repair_id=repair_id,
    )
    plan_path = output / "legacy_layout_plan.csv"
    if execute and plan_path.exists():
        prior_plan = pd.read_csv(plan_path, keep_default_na=False)
        plan = prior_plan if not prior_plan.empty else fresh_plan
    else:
        plan = fresh_plan
    plan.to_csv(plan_path, index=False)
    result = execute_legacy_layout_plan(plan, execute=execute)
    result.to_csv(output / "legacy_layout_manifest.csv", index=False)

    manifest["legacy_layout_plan_count"] = len(plan)
    manifest["legacy_layout_plan_actions"] = (
        dict(Counter(plan["action"].astype(str))) if not plan.empty else {}
    )
    manifest["legacy_layout_last_run_at"] = datetime.now(timezone.utc).isoformat()
    manifest["legacy_layout_execute"] = execute
    if execute:
        post_audit = build_bert_legacy_layout_audit(registry, workspace)
        post_audit.to_csv(output / "post_layout_audit.csv", index=False)
        remaining_redirects = int(post_audit["canonical_is_redirect"].astype(bool).sum())
        compatibility_links = int(
            post_audit["layout_status"].astype(str).eq("legacy_redirects_to_canonical").sum()
        )
        bad_legacy = int(
            ~post_audit["layout_status"].astype(str).isin(
                {"canonical_only", "single_path", "legacy_redirects_to_canonical"}
            )
        ).sum()
        invalid_post = int(
            (
                post_audit["canonical_reuse_exact_count"].astype(int)
                != post_audit["canonical_unit_count"].astype(int)
            ).sum()
        )
        manifest["legacy_layout_remaining_canonical_redirects"] = remaining_redirects
        manifest["legacy_layout_compatibility_links"] = compatibility_links
        manifest["legacy_layout_remaining_legacy_paths"] = bad_legacy
        manifest["legacy_layout_invalid_canonical_count"] = invalid_post
        manifest["legacy_layout_complete"] = (
            remaining_redirects == 0 and bad_legacy == 0 and invalid_post == 0
        )
        if not manifest["legacy_layout_complete"]:
            raise ValueError(
                "BERT legacy layout migration did not produce a clean canonical-only result tree"
            )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return result


def create_repair_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit, archive, rerun, and verify canonical BERT/TabPFN OOD repairs."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "audit",
            "archive",
            "canonicalize-layout",
            "prepare-standard-rerun",
            "delete-standard-rerun",
            "verify-standard-rerun",
            "prepare-tabpfn-v1-refresh",
            "archive-tabpfn-v1-refresh",
            "canonicalize-tabpfn-v1",
            "rerun-tabpfn-v1",
            "rerun-tabpfn-api25-all",
            "verify-tabpfn-v1",
            "rerun",
            "verify",
            "_run-item",
        ],
    )
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split-root", type=Path, default=Path("output/ood_splits"))
    parser.add_argument("--output-root", type=Path, default=Path("output/ood_repair"))
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("output/ood_repair_archive"),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("output/calibration/parameter_freeze/frozen_model_params_v1.yaml"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tabpfn", "scibert", "steelbert", "matscibert"],
        default=["tabpfn", "scibert", "steelbert", "matscibert"],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-key", default=None)
    return parser


def build_bert_repair_args(
    row: pd.Series,
    entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
) -> argparse.Namespace:
    effective = frozen_entry.get("effective_parameters", {}) or {}
    frozen_args = effective.get("args", {}) or {}
    alloy_config = ALLOY_CONFIGS_OOD.get(entry.alloy_type, {})
    method_params = dict(entry.method_params)
    processing_cols = frozen_args.get("other_features_name")
    if processing_cols is None:
        processing_cols = alloy_config.get("processing_cols", [])
    return argparse.Namespace(
        data_file=str(frozen_args.get("data_file", alloy_config.get("raw_data", ""))),
        result_dir=str(row["run_root"]),
        target_column=entry.target_column,
        alloy_type=entry.alloy_type,
        processing_cols=list(processing_cols or []),
        processing_text_column=str(
            alloy_config.get("processing_text_column", "Processing_Description")
        ),
        use_composition_feature=bool(frozen_args.get("use_composition_feature", False)),
        use_element_embedding=bool(frozen_args.get("use_element_embedding", True)),
        use_process_embedding=bool(frozen_args.get("use_process_embedding", True)),
        use_temperature=bool(frozen_args.get("use_temperature", False)),
        embedding_type=str(row["model_dirname"]),
        use_llmprop=False,
        models=None,
        use_nn=True,
        cross_validate=False,
        num_folds=int(frozen_args.get("num_folds", 9)),
        test_size=float(method_params.get("test_size", 0.2)),
        outer_test_size=float(method_params.get("outer_test_size", 0.2)),
        random_state=int(entry.random_state),
        epochs=int(frozen_args.get("epochs", 200)),
        patience=int(frozen_args.get("patience", 30)),
        batch_size=int(frozen_args.get("batch_size", 256)),
        use_optuna=False,
        n_trials=0,
        mlp_max_iter=300,
        evaluate_after_train=True,
        run_shap_analysis=False,
        split_strategy=entry.protocol,
        extrapolation_side=str(method_params.get("extrapolation_side", "low_to_high")),
        sparse_candidate_pool_size=int(method_params.get("sparse_candidate_pool_size", 500)),
        sparse_cluster_count=int(method_params.get("sparse_cluster_count", 5)),
        sparse_samples_per_cluster=int(method_params.get("sparse_samples_per_cluster", 1)),
        sparse_kde_bandwidth=method_params.get("sparse_kde_bandwidth"),
        sparse_neighbors_per_seed=int(method_params.get("sparse_neighbors_per_seed", 5)),
        loco_cluster_count=int(method_params.get("loco_cluster_count", 5)),
        baseline_num_folds=int(method_params.get("baseline_num_folds", 5)),
    )


def run_bert_repair_item(
    row: pd.Series,
    split_registry: CanonicalSplitRegistry,
    parameter_registry: FrozenParameterRegistry,
) -> dict[str, Any]:
    if str(row["model_family"]) != "BERT":
        raise ValueError("run_bert_repair_item only accepts BERT repair rows")
    entry = split_registry.require(
        str(row["task_id"]),
        str(row["protocol"]),
        str(row["split_id"]),
    )
    frozen_entry = parameter_registry.get(
        entry.task_id,
        entry.protocol,
        entry.split_id,
        "BERT",
        str(row["model"]),
    )
    effective = frozen_entry.get("effective_parameters", {}) or {}
    if not effective.get("model_config"):
        raise ValueError(f"Frozen BERT model_config is missing for {row['repair_key']}")
    set_seed(entry.random_state)
    args = build_bert_repair_args(row, entry, frozen_entry)
    return run_prepared_split(
        args=args,
        prepared_split=entry.load_prepared_split(),
        result_dir=str(row["run_root"]),
        frozen_parameters=effective,
        extra_manifest={
            "canonical_split_identity": entry.identity_record(),
            "parameter_snapshot_id": parameter_registry.snapshot_id,
            "parameter_hash": str(frozen_entry.get("parameter_hash", "")),
            "frozen_parameter_file": str(parameter_registry.path.resolve()),
            "repair_key": str(row["repair_key"]),
        },
    )


def run_tabpfn_repair_item(
    row: pd.Series,
    split_registry: CanonicalSplitRegistry,
    parameter_registry: FrozenParameterRegistry,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    if str(row["model_family"]) != "TabPFN":
        raise ValueError("run_tabpfn_repair_item only accepts TabPFN repair rows")
    model = str(row["model"])
    if model not in {spec["model"] for spec in TABPFN_MODEL_SPECS}:
        raise ValueError(f"Unsupported TabPFN repair model: {model}")
    entry = split_registry.require(
        str(row["task_id"]),
        str(row["protocol"]),
        str(row["split_id"]),
    )
    frozen_entry = parameter_registry.get(
        entry.task_id,
        entry.protocol,
        entry.split_id,
        "TabPFN",
        model,
    )
    effective = frozen_entry.get("effective_parameters", {}) or {}
    model_info = effective.get("model_info", {}) or {}
    feature_mode = str(effective.get("feature_mode", row.get("feature_mode", ""))).lower()
    if feature_mode not in {"numeric", "text"}:
        raise ValueError(f"Unsupported frozen TabPFN feature mode: {feature_mode}")
    if str(effective.get("resolved_backend", model_info.get("resolved_backend", ""))) != "api":
        raise ValueError("Canonical repair only permits the TabPFN API backend")
    if str(model_info.get("effective_model_version", "")) != "api_2.5_plus":
        raise ValueError("Canonical repair only permits TabPFN API 2.5 Plus")
    feature_cols = [str(column) for column in effective.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError(f"Frozen TabPFN feature columns are missing for {row['repair_key']}")
    method_params = dict(entry.method_params)
    trainer = TabPFNOODTrainer(
        alloy_type=entry.alloy_type,
        target_col=entry.target_column,
        base_path=str(Path(workspace_root).resolve()),
        output_root=None,
        test_size=float(method_params.get("test_size", 0.2)),
        outer_test_size=float(method_params.get("outer_test_size", 0.2)),
        random_state=int(effective.get("random_state", entry.random_state)),
        split_strategy=entry.protocol,
        extrapolation_side=str(method_params.get("extrapolation_side", "low_to_high")),
        sparse_candidate_pool_size=int(method_params.get("sparse_candidate_pool_size", 500)),
        sparse_cluster_count=int(method_params.get("sparse_cluster_count", 5)),
        sparse_samples_per_cluster=int(method_params.get("sparse_samples_per_cluster", 1)),
        sparse_kde_bandwidth=method_params.get("sparse_kde_bandwidth"),
        sparse_neighbors_per_seed=int(method_params.get("sparse_neighbors_per_seed", 5)),
        loco_cluster_count=int(method_params.get("loco_cluster_count", 5)),
        baseline_num_folds=int(method_params.get("baseline_num_folds", 5)),
        backend="api",
        model_version="v2.5",
        feature_mode=feature_mode,
    )
    actual_runtime = getattr(trainer, "runtime_info", None)
    if isinstance(actual_runtime, dict):
        if str(actual_runtime.get("resolved_backend", "")) != "api":
            raise ValueError("TabPFN repair runtime did not resolve to the API backend")
        if str(actual_runtime.get("effective_model_version", "")) != "api_2.5_plus":
            raise ValueError("TabPFN repair runtime did not resolve to API 2.5 Plus")
        if str(actual_runtime.get("feature_mode", "")).lower() != feature_mode:
            raise ValueError(
                "TabPFN repair runtime feature mode does not match the frozen parameters"
            )
    fold_metadata = None
    if entry.split_id != "main":
        fold_metadata = {
            "fold_index": int(entry.split_id.removeprefix("fold_")),
            "held_out_cluster_id": int(entry.split_summary.get("held_out_cluster_id", -1)),
        }
    result = trainer.run_canonical_split(
        entry.load_prepared_split(),
        result_dir=str(row["run_root"]),
        feature_cols=feature_cols,
        align_predictions=False,
        fold_metadata=fold_metadata,
    )
    manifest_path = Path(str(row["run_root"])) / "pipeline_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "canonical_split_identity": entry.identity_record(),
                "parameter_snapshot_id": parameter_registry.snapshot_id,
                "parameter_hash": str(frozen_entry.get("parameter_hash", "")),
                "frozen_parameter_file": str(parameter_registry.path.resolve()),
                "repair_key": str(row["repair_key"]),
            }
        )
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    return result


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str | None]:
    true_columns = [
        column
        for column in frame.columns
        if str(column).lower().startswith("true_")
        or str(column).lower().endswith("_actual")
    ]
    pred_columns = [
        column
        for column in frame.columns
        if str(column).lower().startswith("pred_")
        or str(column).lower().endswith("_predicted")
    ]
    if len(true_columns) != 1 or len(pred_columns) != 1:
        raise ValueError(
            f"Prediction columns are ambiguous: true={true_columns}, predicted={pred_columns}"
        )
    set_column = next(
        (column for column in frame.columns if str(column).lower() in {"set", "dataset"}),
        None,
    )
    return str(true_columns[0]), str(pred_columns[0]), str(set_column) if set_column else None


def _test_prediction_frame(path: Path) -> tuple[pd.DataFrame, str, str]:
    frame = pd.read_csv(path)
    if "ID" not in frame.columns:
        raise ValueError("Prediction CSV is missing ID")
    true_column, pred_column, set_column = _prediction_columns(frame)
    if set_column is not None:
        labels = frame[set_column].astype(str).str.strip().str.lower()
        test_mask = labels.isin({"oodtest", "ood_test", "test", "testing"})
        frame = frame[test_mask].copy()
    return frame.reset_index(drop=True), true_column, pred_column


def _prediction_path(row: pd.Series) -> Path:
    run_root = Path(str(row["run_root"]))
    if str(row["model_family"]) == "TabPFN":
        return run_root / "predictions" / "all_predictions.csv"
    return run_root / "predictions" / "best_model_all_predictions.csv"


def _recorded_metrics(row: pd.Series) -> dict[str, float]:
    run_root = Path(str(row["run_root"]))
    if str(row["model_family"]) == "TabPFN":
        path = run_root / "metrics" / "metrics_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("ood_test") or payload.get("test") or {}
        return {
            name: float(metrics[name])
            for name in ("mae", "rmse", "r2")
            if name in metrics
        }

    candidates = [
        run_root / "final_evaluation_metrics.json",
        *sorted(run_root.glob("*evaluation_summary.json")),
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("combined_evaluation_summary", payload)
        metrics = summary.get("test_set_metrics") if isinstance(summary, dict) else None
        if not isinstance(metrics, dict):
            continue
        extracted: dict[str, float] = {}
        for metric_name in ("mae", "rmse", "r2"):
            matches = [
                value
                for key, value in metrics.items()
                if str(key).lower().endswith(f"_{metric_name}")
            ]
            if len(matches) == 1:
                extracted[metric_name] = float(matches[0])
        if extracted:
            return extracted
    return {}


def verify_repair_item(
    row: pd.Series,
    entry: CanonicalSplitEntry,
    *,
    expected_snapshot_id: str,
    expected_parameter_hash: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    split_audit = audit_result_split(entry, row["run_root"])
    if split_audit["status"] != "reuse_exact":
        reasons.append(f"split_{split_audit['reason']}")

    manifest_path = Path(str(row["run_root"])) / "pipeline_manifest.json"
    if not manifest_path.exists():
        reasons.append("missing_pipeline_manifest")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("parameter_snapshot_id", "")) != str(expected_snapshot_id):
            reasons.append("parameter_snapshot_mismatch")
        if str(manifest.get("parameter_hash", "")) != str(expected_parameter_hash):
            reasons.append("parameter_hash_mismatch")

    prediction_path = _prediction_path(row)
    recomputed = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    prediction_count = 0
    if not prediction_path.exists():
        reasons.append("missing_prediction_csv")
    else:
        try:
            predictions, true_column, pred_column = _test_prediction_frame(prediction_path)
            canonical_test = entry.load_test()
            predicted_ids = pd.to_numeric(predictions["ID"], errors="raise").astype("int64").tolist()
            canonical_ids = (
                pd.to_numeric(canonical_test["ID"], errors="raise").astype("int64").tolist()
            )
            predicted_true = pd.to_numeric(predictions[true_column], errors="raise").to_numpy(float)
            canonical_true = pd.to_numeric(
                canonical_test[entry.target_column],
                errors="raise",
            ).to_numpy(float)
            if predicted_ids != canonical_ids or not np.allclose(
                predicted_true,
                canonical_true,
                rtol=1e-6,
                atol=1e-4,
                equal_nan=False,
            ):
                reasons.append("prediction_test_identity_mismatch")
            predicted = pd.to_numeric(predictions[pred_column], errors="raise").to_numpy(float)
            prediction_count = len(predicted)
            if len(predicted) != len(canonical_true):
                reasons.append("prediction_test_count_mismatch")
            elif not np.isfinite(predicted).all():
                reasons.append("prediction_contains_nonfinite_values")
            else:
                recomputed = {
                    "mae": float(mean_absolute_error(canonical_true, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(canonical_true, predicted))),
                    "r2": float(r2_score(canonical_true, predicted)),
                }
                recorded = _recorded_metrics(row)
                if not recorded:
                    reasons.append("missing_recorded_metrics")
                else:
                    for metric_name, value in recomputed.items():
                        if metric_name not in recorded or not np.isclose(
                            value,
                            recorded[metric_name],
                            rtol=1e-5,
                            atol=1e-5,
                        ):
                            reasons.append(f"recorded_{metric_name}_mismatch")
        except Exception as exc:
            reasons.append(f"prediction_verification_error:{type(exc).__name__}:{exc}")

    return {
        "repair_key": str(row["repair_key"]),
        "task_id": entry.task_id,
        "protocol": entry.protocol,
        "split_id": entry.split_id,
        "model_family": str(row["model_family"]),
        "model": str(row["model"]),
        "run_root": str(row["run_root"]),
        "verification_status": "failed" if reasons else "passed",
        "verification_reason": ";".join(reasons),
        "prediction_count": prediction_count,
        "recomputed_mae": recomputed["mae"],
        "recomputed_rmse": recomputed["rmse"],
        "recomputed_r2": recomputed["r2"],
        "prediction_file": str(prediction_path),
        "parameter_snapshot_id": expected_snapshot_id,
        "parameter_hash": expected_parameter_hash,
    }


def verify_standard_batch_split(
    queue_row: pd.Series,
    entry: CanonicalSplitEntry,
    run_root: str | Path,
) -> dict[str, Any]:
    reasons: list[str] = []
    run_path = Path(run_root)
    split_audit = audit_result_split(entry, run_path)
    if split_audit["status"] != "reuse_exact":
        reasons.append(f"split_{split_audit['reason']}")

    manifest_path = run_path / "pipeline_manifest.json"
    if not manifest_path.exists():
        reasons.append("missing_pipeline_manifest")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        execution = manifest.get("execution") or {}
        if execution.get("entrypoint") != "src.pipelines.run_batch_ood":
            reasons.append("nonstandard_pipeline_entrypoint")
        if str(execution.get("batch_config_name", "")) != str(queue_row["config_name"]):
            reasons.append("batch_config_mismatch")
        if not bool(execution.get("require_existing_splits", False)):
            reasons.append("canonical_split_mode_not_recorded")
        if not bool(execution.get("use_optuna", False)):
            reasons.append("optuna_not_recorded")
        if int(execution.get("n_trials", -1)) != 30:
            reasons.append("optuna_trial_count_mismatch")
        split_artifacts = manifest.get("split_artifacts") or {}
        expected_paths = {
            "canonical_train_file": entry.train_file,
            "canonical_test_file": entry.test_file,
        }
        for key, expected in expected_paths.items():
            raw = split_artifacts.get(key)
            if not raw or Path(str(raw)).resolve() != expected.resolve():
                reasons.append(f"{key}_mismatch")
        for section_name in ("feature_files", "split_artifacts"):
            section = manifest.get(section_name) or {}
            for name, raw in section.items():
                if not isinstance(raw, str):
                    continue
                artifact_path = _lexical_absolute(raw)
                if not artifact_path.exists():
                    reasons.append(f"stale_{section_name}_{name}")
                    continue
                if not str(name).startswith("canonical_"):
                    try:
                        artifact_path.relative_to(_lexical_absolute(run_path))
                    except ValueError:
                        reasons.append(f"external_{section_name}_{name}")

    verification_row = pd.Series(
        {
            "run_root": str(run_path),
            "model_family": "BERT",
            "model": str(queue_row["model"]),
        }
    )
    prediction_path = _prediction_path(verification_row)
    recomputed = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    prediction_count = 0
    if not prediction_path.exists():
        reasons.append("missing_prediction_csv")
    else:
        try:
            predictions, true_column, pred_column = _test_prediction_frame(prediction_path)
            canonical_test = entry.load_test()
            predicted_ids = pd.to_numeric(predictions["ID"], errors="raise").astype("int64").tolist()
            canonical_ids = pd.to_numeric(canonical_test["ID"], errors="raise").astype("int64").tolist()
            predicted_true = pd.to_numeric(predictions[true_column], errors="raise").to_numpy(float)
            canonical_true = pd.to_numeric(
                canonical_test[entry.target_column], errors="raise"
            ).to_numpy(float)
            if predicted_ids != canonical_ids or not np.allclose(
                predicted_true,
                canonical_true,
                rtol=1e-6,
                atol=1e-4,
                equal_nan=False,
            ):
                reasons.append("prediction_test_identity_mismatch")
            predicted = pd.to_numeric(predictions[pred_column], errors="raise").to_numpy(float)
            prediction_count = len(predicted)
            if len(predicted) != len(canonical_true):
                reasons.append("prediction_test_count_mismatch")
            elif not np.isfinite(predicted).all():
                reasons.append("prediction_contains_nonfinite_values")
            else:
                recomputed = {
                    "mae": float(mean_absolute_error(canonical_true, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(canonical_true, predicted))),
                    "r2": float(r2_score(canonical_true, predicted)),
                }
                recorded = _recorded_metrics(verification_row)
                if not recorded:
                    reasons.append("missing_recorded_metrics")
                else:
                    for metric_name, value in recomputed.items():
                        if metric_name not in recorded or not np.isclose(
                            value,
                            recorded[metric_name],
                            rtol=1e-5,
                            atol=1e-5,
                        ):
                            reasons.append(f"recorded_{metric_name}_mismatch")
        except Exception as exc:
            reasons.append(f"prediction_verification_error:{type(exc).__name__}:{exc}")

    return {
        "queue_id": str(queue_row["queue_id"]),
        "task_id": entry.task_id,
        "protocol": entry.protocol,
        "split_id": entry.split_id,
        "model_family": "BERT",
        "model": str(queue_row["model"]),
        "run_root": str(run_path),
        "verification_status": "failed" if reasons else "passed",
        "verification_reason": ";".join(dict.fromkeys(reasons)),
        "prediction_count": prediction_count,
        "recomputed_mae": recomputed["mae"],
        "recomputed_rmse": recomputed["rmse"],
        "recomputed_r2": recomputed["r2"],
        "prediction_file": str(prediction_path),
    }


def _mean_metric_payload(payloads: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for payload in payloads
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: float(np.mean([float(payload[key]) for payload in payloads if key in payload]))
        for key in keys
    }


def rebuild_parent_manifest(
    row: pd.Series,
    split_registry: CanonicalSplitRegistry,
) -> Path | None:
    protocol = str(row["protocol"])
    if protocol not in {"loco", "random_cv_baseline"}:
        return None
    task_id = str(row["task_id"])
    parent_root = Path(str(row["parent_result_root"])).resolve()
    parent_root.mkdir(parents=True, exist_ok=True)
    entries = [
        entry
        for entry in split_registry.entries()
        if entry.task_id == task_id and entry.protocol == protocol
    ]
    entries.sort(key=lambda entry: int(entry.split_id.removeprefix("fold_")))
    if not entries:
        raise ValueError(f"No canonical folds found for {task_id} / {protocol}")

    if str(row["model_family"]) == "TabPFN":
        folds: list[dict[str, Any]] = []
        train_metrics: list[dict[str, Any]] = []
        test_metrics: list[dict[str, Any]] = []
        for entry in entries:
            fold_index = int(entry.split_id.removeprefix("fold_"))
            fold_root = parent_root / "folds" / entry.split_id
            metrics_path = fold_root / "metrics" / "metrics_summary.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            train = dict(metrics.get("train") or {})
            test = dict(metrics.get("ood_test") or metrics.get("test") or {})
            train_metrics.append(train)
            test_metrics.append(test)
            folds.append(
                {
                    "fold_index": fold_index,
                    "held_out_cluster_id": int(entry.split_summary.get("held_out_cluster_id", -1)),
                    "split_summary": entry.split_summary,
                    "train": train,
                    "ood_test": test,
                }
            )
        payload = {
            "alloy_type": entries[0].alloy_type,
            "target_col": entries[0].target_column,
            "split_strategy": protocol,
            "fold_count": len(folds),
            "expected_fold_count": len(entries),
            "repair_incomplete": len(folds) != len(entries),
            "folds": folds,
            "aggregated_metrics": {
                "train": _mean_metric_payload(train_metrics) if train_metrics else {},
                "ood_test": _mean_metric_payload(test_metrics) if test_metrics else {},
            },
        }
        manifest_name = "loco_manifest.json" if protocol == "loco" else f"{protocol}_manifest.json"
    else:
        folds = []
        for entry in entries:
            fold_index = int(entry.split_id.removeprefix("fold_"))
            fold_root = parent_root / "folds" / entry.split_id
            pipeline_path = fold_root / "pipeline_manifest.json"
            if not pipeline_path.exists():
                continue
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            folds.append(
                {
                    "fold_index": fold_index,
                    "held_out_cluster_id": int(entry.split_summary.get("held_out_cluster_id", -1)),
                    "metadata": {
                        "fold_index": fold_index,
                        "held_out_cluster_id": int(
                            entry.split_summary.get("held_out_cluster_id", -1)
                        ),
                        "cluster_size": int(len(entry.load_test())),
                    },
                    "run_root": str(fold_root),
                    "split_artifacts": pipeline.get("split_artifacts", {}),
                    "feature_files": pipeline.get("feature_files", {}),
                    "split_summary": entry.split_summary,
                }
            )
        payload = {
            "data_file": str(
                next(
                    (
                        json.loads(
                            (parent_root / "folds" / entry.split_id / "pipeline_manifest.json").read_text(
                                encoding="utf-8"
                            )
                        ).get("data_file", "")
                        for entry in entries
                        if (parent_root / "folds" / entry.split_id / "pipeline_manifest.json").exists()
                    ),
                    "",
                )
            ),
            "target_column": entries[0].target_column,
            "method_params": entries[0].method_params,
            "fold_count": len(folds),
            "expected_fold_count": len(entries),
            "repair_incomplete": len(folds) != len(entries),
            "folds": folds,
        }
        manifest_name = "loco_manifest.json" if protocol == "loco" else f"{protocol}_manifest.json"

    manifest_path = parent_root / manifest_name
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest_path


MODEL_GROUPS = {
    "tabpfn": {"TabPFN-2.5-Plus-Numeric", "TabPFN-2.5-Plus-Text"},
    "scibert": {"SciBERT"},
    "steelbert": {"SteelBERT"},
    "matscibert": {"MatSciBERT"},
}


def select_repair_rows(queue: pd.DataFrame, model_groups: list[str]) -> pd.DataFrame:
    models: set[str] = set()
    for group in model_groups:
        if group not in MODEL_GROUPS:
            raise ValueError(f"Unsupported repair model group: {group}")
        models.update(MODEL_GROUPS[group])
    return queue[queue["model"].astype(str).isin(models)].copy().reset_index(drop=True)


def _frozen_entry_for_row(
    parameter_registry: FrozenParameterRegistry,
    row: pd.Series,
) -> dict[str, Any]:
    return parameter_registry.get(
        str(row["task_id"]),
        str(row["protocol"]),
        str(row["split_id"]),
        str(row["model_family"]),
        str(row["model"]),
    )


def _set_status_values(
    status: pd.DataFrame,
    repair_key: str,
    **values: Any,
) -> pd.DataFrame:
    mask = status["repair_key"].astype(str).eq(repair_key)
    if int(mask.sum()) != 1:
        raise ValueError(f"Repair status row is not unique: {repair_key}")
    for name, value in values.items():
        status.loc[mask, name] = value
    return status


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "item"


def _move_failed_attempt(run_root: Path, output_root: Path, repair_key: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output_root / "failed_attempts" / stamp / _safe_name(repair_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_root.replace(destination)
    return destination


def _cleanup_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_repair_item_command(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    snapshot_path: str | Path,
    repair_key: str,
) -> list[str]:
    workspace = Path(workspace_root).resolve()
    return [
        str(LLM_PYTHON),
        "-m",
        "src.pipelines.repair_ood_results",
        "--stage",
        "_run-item",
        "--workspace-root",
        str(workspace),
        "--split-root",
        str(_resolve_against(split_root, workspace)),
        "--output-root",
        str(_resolve_against(output_root, workspace)),
        "--snapshot",
        str(_resolve_against(snapshot_path, workspace)),
        "--repair-key",
        repair_key,
    ]


def run_single_repair_item_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    snapshot_path: str | Path,
    repair_key: str,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    _, _, queue, _ = _load_locked_state(workspace, output_root)
    selected = queue[queue["repair_key"].astype(str).eq(repair_key)]
    if len(selected) != 1:
        raise ValueError(f"Repair key is not unique in the locked queue: {repair_key}")
    row = selected.iloc[0]
    split_registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    parameter_registry = FrozenParameterRegistry(_resolve_against(snapshot_path, workspace))
    if str(row["model_family"]) == "BERT":
        run_bert_repair_item(row, split_registry, parameter_registry)
    else:
        run_tabpfn_repair_item(
            row,
            split_registry,
            parameter_registry,
            workspace_root=workspace,
        )
    entry = split_registry.require(
        str(row["task_id"]),
        str(row["protocol"]),
        str(row["split_id"]),
    )
    frozen_entry = _frozen_entry_for_row(parameter_registry, row)
    verification = verify_repair_item(
        row,
        entry,
        expected_snapshot_id=parameter_registry.snapshot_id,
        expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
    )
    if verification["verification_status"] != "passed":
        raise ValueError(str(verification["verification_reason"]))
    return verification


def _run_bert_item_subprocess(
    row: pd.Series,
    *,
    workspace_root: Path,
    split_root: str | Path,
    output_root: Path,
    snapshot_path: str | Path,
) -> None:
    repair_key = str(row["repair_key"])
    command = build_repair_item_command(
        workspace_root=workspace_root,
        split_root=split_root,
        output_root=output_root,
        snapshot_path=snapshot_path,
        repair_key=repair_key,
    )
    log_path = output_root / "logs" / f"{_safe_name(repair_key)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"Isolated BERT repair failed with exit code {completed.returncode}. "
            f"Log: {log_path}\n" + "\n".join(tail)
        )


def run_rerun_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    snapshot_path: str | Path,
    model_groups: list[str],
    resume: bool,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output, manifest, queue, status = _load_locked_state(workspace, output_root)
    if not manifest.get("archive_executed"):
        raise ValueError("Archive must be executed successfully before rerunning models")
    split_registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    parameter_registry = FrozenParameterRegistry(_resolve_against(snapshot_path, workspace))
    selected = select_repair_rows(queue, model_groups)
    if selected.empty:
        raise ValueError("No repair rows matched the requested model groups")
    for _, row in selected.iterrows():
        _frozen_entry_for_row(parameter_registry, row)

    succeeded = 0
    skipped = 0
    failed = 0
    for index, row in selected.iterrows():
        repair_key = str(row["repair_key"])
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        expected_hash = str(frozen_entry.get("parameter_hash", ""))
        entry = split_registry.require(
            str(row["task_id"]),
            str(row["protocol"]),
            str(row["split_id"]),
        )
        status_row = status[status["repair_key"].astype(str).eq(repair_key)].iloc[0]
        if str(status_row["archive_status"]) not in {"moved", "already_moved"}:
            raise ValueError(f"Repair row was not archived successfully: {repair_key}")

        run_root = Path(str(row["run_root"]))
        if resume and run_root.exists():
            verification = verify_repair_item(
                row,
                entry,
                expected_snapshot_id=parameter_registry.snapshot_id,
                expected_parameter_hash=expected_hash,
            )
            if verification["verification_status"] == "passed":
                status = _set_status_values(
                    status,
                    repair_key,
                    run_status="success",
                    verification_status="passed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    last_error="",
                )
                _write_status(status, output)
                skipped += 1
                print(f"[SKIP {index + 1}/{len(selected)}] {repair_key} already verified", flush=True)
                continue
            failed_attempt = _move_failed_attempt(run_root, output, repair_key)
            print(f"[RETRY] moved failed attempt to {failed_attempt}", flush=True)
        elif run_root.exists():
            raise FileExistsError(
                f"Repair output already exists; use --resume after inspecting it: {run_root}"
            )

        attempts = int(status_row.get("attempt_count", 0) or 0) + 1
        status = _set_status_values(
            status,
            repair_key,
            run_status="running",
            verification_status="pending",
            attempt_count=attempts,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at="",
            last_error="",
        )
        _write_status(status, output)
        print(f"[RUN {index + 1}/{len(selected)}] {repair_key}", flush=True)
        try:
            if str(row["model_family"]) == "BERT":
                _run_bert_item_subprocess(
                    row,
                    workspace_root=workspace,
                    split_root=split_root,
                    output_root=output,
                    snapshot_path=snapshot_path,
                )
            else:
                run_tabpfn_repair_item(
                    row,
                    split_registry,
                    parameter_registry,
                    workspace_root=workspace,
                )
            if str(row["split_id"]) != "main":
                rebuild_parent_manifest(row, split_registry)
            verification = verify_repair_item(
                row,
                entry,
                expected_snapshot_id=parameter_registry.snapshot_id,
                expected_parameter_hash=expected_hash,
            )
            if verification["verification_status"] != "passed":
                raise ValueError(str(verification["verification_reason"]))
            status = _set_status_values(
                status,
                repair_key,
                run_status="success",
                verification_status="passed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_error="",
            )
            succeeded += 1
        except Exception as exc:
            status = _set_status_values(
                status,
                repair_key,
                run_status="failed",
                verification_status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_error=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
            print(f"[FAILED] {repair_key}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            _write_status(status, output)
            _cleanup_accelerator_memory()

    manifest["last_rerun_at"] = datetime.now(timezone.utc).isoformat()
    manifest["last_rerun_model_groups"] = list(model_groups)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return {
        "selected": len(selected),
        "succeeded": succeeded,
        "skipped_verified": skipped,
        "failed": failed,
    }


def _rebuild_all_affected_parent_manifests(
    queue: pd.DataFrame,
    split_registry: CanonicalSplitRegistry,
) -> list[str]:
    multi = queue[queue["split_id"].astype(str).ne("main")].copy()
    paths: list[str] = []
    grouping = ["task_id", "protocol", "model_family", "model", "parent_result_root"]
    for _, group in multi.groupby(grouping, sort=True, dropna=False):
        path = rebuild_parent_manifest(group.iloc[0], split_registry)
        if path is not None:
            paths.append(str(path))
    return paths


def run_verify_standard_rerun_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    queue_path = output / "bert_standard_rerun_queue.csv"
    protected_path = output / "protected_correct_tasks.csv"
    manifest_path = output / "manifest.json"
    for path in (queue_path, protected_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Standard rerun metadata is missing: {path}")
    queue = pd.read_csv(queue_path, keep_default_na=False)
    protected = pd.read_csv(protected_path, keep_default_na=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_standard_rerun_counts(queue, protected)
    _assert_protected_tasks_unchanged(protected)
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    grouped_entries: dict[tuple[str, str], list[CanonicalSplitEntry]] = {}
    for entry in registry.entries():
        grouped_entries.setdefault((entry.task_id, entry.protocol), []).append(entry)

    rows: list[dict[str, Any]] = []
    for _, queue_row in queue.iterrows():
        key = (str(queue_row["task_id"]), str(queue_row["protocol"]))
        entries = sorted(grouped_entries.get(key, []), key=lambda value: value.split_id)
        if len(entries) != int(queue_row["rerun_split_unit_count"]):
            raise ValueError(
                f"Canonical split count mismatch for {queue_row['queue_id']}: "
                f"expected={queue_row['rerun_split_unit_count']}, found={len(entries)}"
            )
        task_root = Path(str(queue_row["output_root"]))
        for entry in entries:
            run_root = task_root if entry.split_id == "main" else task_root / "folds" / entry.split_id
            rows.append(verify_standard_batch_split(queue_row, entry, run_root))
    report = pd.DataFrame(rows)
    report.to_csv(output / "standard_rerun_verification.csv", index=False)

    full_audit = build_existing_result_audit(registry, workspace)
    bert_audit = full_audit[full_audit["model_family"].astype(str).eq("BERT")].copy()
    bert_audit.to_csv(output / "post_standard_rerun_bert_split_audit.csv", index=False)
    _assert_protected_tasks_unchanged(protected)
    passed = int(report["verification_status"].astype(str).eq("passed").sum())
    bert_exact = int(bert_audit["status"].astype(str).eq("reuse_exact").sum())
    complete = len(report) == 199 and passed == 199 and len(bert_audit) == 450 and bert_exact == 450
    manifest.update(
        {
            "standard_rerun_verified_at": datetime.now(timezone.utc).isoformat(),
            "standard_rerun_verification_count": len(report),
            "standard_rerun_verification_passed": passed,
            "bert_canonical_split_unit_count": len(bert_audit),
            "bert_canonical_exact_count": bert_exact,
            "standard_rerun_verification_complete": complete,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return report


def _load_tabpfn_v1_refresh_state(
    workspace_root: str | Path,
    output_root: str | Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, FrozenParameterRegistry]:
    workspace = Path(workspace_root).resolve()
    output = _resolve_against(output_root, workspace)
    required = {
        "manifest": output / "manifest.json",
        "inventory": output / "tabpfn_v1_inventory.csv",
        "status": output / "tabpfn_v1_refresh_status.csv",
        "candidate": output / "frozen_model_params_v1_api25_candidate.yaml",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(f"Run prepare-tabpfn-v1-refresh first; missing: {path}")
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    inventory = pd.read_csv(required["inventory"], keep_default_na=False)
    status = pd.read_csv(required["status"], keep_default_na=False)
    parameter_registry = FrozenParameterRegistry(required["candidate"])
    if parameter_registry.snapshot_id != str(manifest.get("snapshot_id", "")):
        raise ValueError("Locked API 2.5 Plus candidate snapshot ID changed")
    if _file_sha256(required["candidate"]) != str(
        manifest.get("candidate_snapshot_file_sha256", "")
    ):
        raise ValueError("Locked API 2.5 Plus candidate snapshot file changed")
    return output, manifest, inventory, status, parameter_registry


def _write_tabpfn_refresh_status(status: pd.DataFrame, output: Path) -> None:
    temporary = output / "tabpfn_v1_refresh_status.csv.tmp"
    status.to_csv(temporary, index=False)
    temporary.replace(output / "tabpfn_v1_refresh_status.csv")


def _write_tabpfn_api25_all_status(status: pd.DataFrame, output: Path) -> None:
    temporary = output / "tabpfn_api25_all_status.csv.tmp"
    status.to_csv(temporary, index=False)
    temporary.replace(output / "tabpfn_api25_all_status.csv")


def _load_or_create_tabpfn_api25_all_status(
    inventory: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    path = output / "tabpfn_api25_all_status.csv"
    expected_keys = inventory["repair_key"].astype(str).tolist()
    if path.exists():
        status = pd.read_csv(path, keep_default_na=False)
        actual_keys = status["repair_key"].astype(str).tolist()
        if actual_keys != expected_keys:
            raise ValueError("Locked TabPFN API 2.5 all-rerun status inventory changed")
        return status
    status = _initial_tabpfn_api25_all_status(inventory)
    _write_tabpfn_api25_all_status(status, output)
    return status


def _verify_tabpfn_api25_all_unit(
    row: pd.Series,
    entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    base = verify_repair_item(
        row,
        entry,
        expected_snapshot_id=snapshot_id,
        expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
    )
    if base["verification_status"] != "passed":
        reasons.append(str(base["verification_reason"]))
    run_root = Path(str(row["run_root"]))
    split = compare_test_split_to_canonical(
        entry,
        run_root / "split_data" / entry.test_file.name,
    )
    if split["comparison_status"] != "exact":
        reasons.append(f"split_full_{split['comparison_status']}")
    runtime = _audit_tabpfn_runtime(
        frozen_entry,
        run_root / "pipeline_manifest.json",
    )
    if runtime["runtime_status"] != "exact":
        reasons.append(f"runtime_{runtime['runtime_status']}")
    quantile = verify_tabpfn_api25_quantile_artifact(entry, run_root)
    if quantile["verification_status"] != "passed":
        reasons.append(str(quantile["verification_reason"]))
    mean_match = _verify_tabpfn_quantile_mean_matches_ood_prediction(run_root)
    if mean_match["verification_status"] != "passed":
        reasons.append(str(mean_match["verification_reason"]))
    return {
        **base,
        "verification_status": "failed" if reasons else "passed",
        "verification_reason": ";".join(filter(None, dict.fromkeys(reasons))),
        "split_full_status": split["comparison_status"],
        "runtime_status": runtime["runtime_status"],
        "quantile_status": quantile["verification_status"],
        "quantile_prediction_status": mean_match["verification_status"],
        "quantile_path": quantile["quantile_path"],
    }


def run_archive_tabpfn_v1_refresh_stage(
    *,
    workspace_root: str | Path,
    output_root: str | Path,
    archive_root: str | Path,
    execute: bool,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    output, manifest, inventory, status, _ = _load_tabpfn_v1_refresh_state(
        workspace,
        output_root,
    )
    plan_path = output / "tabpfn_archive_plan.csv"
    if plan_path.exists():
        plan = pd.read_csv(plan_path, keep_default_na=False)
    else:
        plan = build_tabpfn_v1_archive_plan(
            inventory,
            workspace_root=workspace,
            archive_root=archive_root,
            repair_id=str(manifest["repair_id"]),
        )
    result = execute_tabpfn_v1_archive_plan(plan, execute=execute)
    result.to_csv(plan_path, index=False)
    if execute:
        failed = result[result["archive_status"].astype(str).eq("failed")]
        for repair_key in status["repair_key"].astype(str):
            rows = result[
                result["repair_keys"]
                .astype(str)
                .str.split(";")
                .map(lambda values: repair_key in values)
            ]
            successful = (
                not rows.empty
                and rows["archive_status"]
                .astype(str)
                .isin({"copied", "already_copied", "moved", "already_moved"})
                .all()
            )
            status = _set_status_values(
                status,
                repair_key,
                archive_status="success" if successful else "failed",
                last_error=("" if successful else "Archive action failed or is missing"),
            )
        _write_tabpfn_refresh_status(status, output)
        manifest["archive_executed"] = failed.empty
        manifest["archive_completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["archive_status_counts"] = dict(
            Counter(result["archive_status"].astype(str))
        )
        _atomic_json_write(output / "manifest.json", manifest)
    return result


def canonicalize_tabpfn_existing_item(
    row: pd.Series,
    entry: CanonicalSplitEntry,
    frozen_entry: dict[str, Any],
    *,
    snapshot_id: str,
    snapshot_path: str | Path,
    repair_id: str,
) -> dict[str, Any]:
    run_root = Path(str(row["run_root"]))
    _validate_tabpfn_refresh_path(run_root, str(row["official_output_root"]))
    split = replace_tabpfn_split_data_with_canonical(entry, run_root)
    prediction = reorder_tabpfn_prediction_rows(
        entry,
        run_root / "predictions" / "all_predictions.csv",
    )
    metrics = refresh_tabpfn_metrics_and_manifest(
        entry,
        run_root,
        frozen_entry=frozen_entry,
        snapshot_id=snapshot_id,
        snapshot_path=snapshot_path,
        repair_key=str(row["repair_key"]),
        repair_id=repair_id,
    )
    verification = verify_repair_item(
        row,
        entry,
        expected_snapshot_id=snapshot_id,
        expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
    )
    runtime = _audit_tabpfn_runtime(
        frozen_entry,
        run_root / "pipeline_manifest.json",
    )
    if split["comparison_status"] != "exact":
        raise ValueError("Canonical TabPFN split replacement did not verify")
    if runtime["runtime_status"] != "exact":
        raise ValueError(f"TabPFN runtime metadata did not normalize: {runtime}")
    if verification["verification_status"] != "passed":
        raise ValueError(str(verification["verification_reason"]))
    return {
        "split": split,
        "prediction": prediction,
        "metrics": metrics,
        "verification": verification,
    }


def run_canonicalize_tabpfn_v1_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    execute: bool,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output, manifest, inventory, status, parameter_registry = _load_tabpfn_v1_refresh_state(
        workspace,
        output_root,
    )
    if not manifest.get("archive_executed"):
        raise ValueError("TabPFN archive must complete before canonicalization")
    selected = inventory[inventory["refresh_action"].astype(str).eq("canonicalize")].copy()
    if not execute:
        return {"selected": len(selected), "succeeded": 0, "failed": 0, "executed": False}
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    succeeded = 0
    skipped = 0
    failed = 0
    for _, row in selected.iterrows():
        repair_key = str(row["repair_key"])
        status_row = status[status["repair_key"].astype(str).eq(repair_key)].iloc[0]
        if str(status_row["archive_status"]) != "success":
            raise ValueError(f"TabPFN archive is incomplete for {repair_key}")
        if str(status_row["canonicalize_status"]) == "success":
            skipped += 1
            continue
        entry = registry.require(str(row["task_id"]), str(row["protocol"]), str(row["split_id"]))
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        try:
            canonicalize_tabpfn_existing_item(
                row,
                entry,
                frozen_entry,
                snapshot_id=parameter_registry.snapshot_id,
                snapshot_path=parameter_registry.path,
                repair_id=str(manifest["repair_id"]),
            )
            status = _set_status_values(
                status,
                repair_key,
                canonicalize_status="success",
                verification_status="passed",
                last_error="",
            )
            succeeded += 1
        except Exception as exc:
            status = _set_status_values(
                status,
                repair_key,
                canonicalize_status="failed",
                verification_status="failed",
                last_error=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
        finally:
            _write_tabpfn_refresh_status(status, output)
    _rebuild_all_affected_parent_manifests(selected, registry)
    complete = int(status["canonicalize_status"].astype(str).eq("success").sum()) == 288
    manifest["canonicalize_complete"] = complete
    manifest["canonicalize_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["canonicalize_status_counts"] = dict(
        Counter(status["canonicalize_status"].astype(str))
    )
    _atomic_json_write(output / "manifest.json", manifest)
    return {
        "selected": len(selected),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "executed": True,
        "complete": complete,
    }


def run_rerun_tabpfn_v1_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    execute: bool,
    resume: bool,
    repair_key: str | None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output, manifest, inventory, status, parameter_registry = _load_tabpfn_v1_refresh_state(
        workspace,
        output_root,
    )
    if not manifest.get("archive_executed") or not manifest.get("canonicalize_complete"):
        raise ValueError("Archive and canonicalization must complete before TabPFN API reruns")
    selected = inventory[inventory["refresh_action"].astype(str).eq("rerun")].copy()
    if repair_key:
        selected = selected[selected["repair_key"].astype(str).eq(repair_key)].copy()
        if len(selected) != 1:
            raise ValueError(f"TabPFN repair key is not unique in the rerun queue: {repair_key}")
    if not execute:
        return {"selected": len(selected), "succeeded": 0, "failed": 0, "executed": False}
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    succeeded = 0
    skipped = 0
    failed = 0
    for _, row in selected.iterrows():
        key = str(row["repair_key"])
        status_row = status[status["repair_key"].astype(str).eq(key)].iloc[0]
        entry = registry.require(str(row["task_id"]), str(row["protocol"]), str(row["split_id"]))
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        run_root = Path(str(row["run_root"]))
        if resume and str(status_row["rerun_status"]) == "success":
            skipped += 1
            continue
        if run_root.exists():
            if not resume:
                raise FileExistsError(f"TabPFN rerun output already exists: {run_root}")
            failed_attempt = _move_failed_attempt(run_root, output, key)
            print(f"[RETRY] moved failed TabPFN attempt to {failed_attempt}", flush=True)
        attempts = int(status_row.get("attempt_count", 0) or 0) + 1
        status = _set_status_values(
            status,
            key,
            rerun_status="running",
            verification_status="pending",
            attempt_count=attempts,
            last_error="",
        )
        _write_tabpfn_refresh_status(status, output)
        try:
            run_tabpfn_repair_item(
                row,
                registry,
                parameter_registry,
                workspace_root=workspace,
            )
            reorder_tabpfn_prediction_rows(
                entry,
                run_root / "predictions" / "all_predictions.csv",
            )
            refresh_tabpfn_metrics_and_manifest(
                entry,
                run_root,
                frozen_entry=frozen_entry,
                snapshot_id=parameter_registry.snapshot_id,
                snapshot_path=parameter_registry.path,
                repair_key=key,
                repair_id=str(manifest["repair_id"]),
            )
            split = compare_test_split_to_canonical(
                entry,
                run_root / "split_data" / entry.test_file.name,
            )
            runtime = _audit_tabpfn_runtime(
                frozen_entry,
                run_root / "pipeline_manifest.json",
            )
            verification = verify_repair_item(
                row,
                entry,
                expected_snapshot_id=parameter_registry.snapshot_id,
                expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
            )
            if split["comparison_status"] != "exact":
                raise ValueError("Rerun TabPFN split is not canonical")
            if runtime["runtime_status"] != "exact":
                raise ValueError(f"Rerun TabPFN runtime is not explicit API 2.5 Plus: {runtime}")
            if verification["verification_status"] != "passed":
                raise ValueError(str(verification["verification_reason"]))
            if str(row["split_id"]) != "main":
                rebuild_parent_manifest(row, registry)
            status = _set_status_values(
                status,
                key,
                rerun_status="success",
                verification_status="passed",
                last_error="",
            )
            succeeded += 1
        except Exception as exc:
            status = _set_status_values(
                status,
                key,
                rerun_status="failed",
                verification_status="failed",
                last_error=f"{type(exc).__name__}: {exc}",
            )
            failed += 1
        finally:
            _write_tabpfn_refresh_status(status, output)
            _cleanup_accelerator_memory()
    complete = int(status["rerun_status"].astype(str).eq("success").sum()) == 12
    manifest["rerun_complete"] = complete
    manifest["rerun_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["rerun_status_counts"] = dict(Counter(status["rerun_status"].astype(str)))
    _atomic_json_write(output / "manifest.json", manifest)
    return {
        "selected": len(selected),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "executed": True,
        "complete": complete,
    }


def run_rerun_tabpfn_api25_all_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    execute: bool,
    resume: bool,
    repair_key: str | None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output, manifest, inventory, _, parameter_registry = _load_tabpfn_v1_refresh_state(
        workspace,
        output_root,
    )
    if not manifest.get("archive_executed"):
        raise ValueError("TabPFN archive must complete before the 300-unit API 2.5 rerun")
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    _validate_tabpfn_api25_all_inventory(inventory, registry, workspace)

    inventory_hash = _stable_frame_hash(inventory)
    locked_hash = str(manifest.get("api25_all_inventory_hash", ""))
    if locked_hash and locked_hash != inventory_hash:
        raise ValueError("Locked TabPFN API 2.5 all-rerun inventory changed")
    if not locked_hash:
        manifest["api25_all_inventory_hash"] = inventory_hash
        manifest["api25_all_inventory_count"] = len(inventory)
        manifest["api25_all_status_file"] = str(output / "tabpfn_api25_all_status.csv")
        _atomic_json_write(output / "manifest.json", manifest)

    status = _load_or_create_tabpfn_api25_all_status(inventory, output)
    selected = _select_tabpfn_api25_all_rows(inventory, repair_key)
    if not execute:
        return {
            "selected": len(selected),
            "succeeded": 0,
            "skipped": 0,
            "failed": 0,
            "executed": False,
            "complete": int(status["run_status"].astype(str).eq("success").sum()) == 300,
        }

    succeeded = 0
    skipped = 0
    failed = 0
    for _, row in selected.iterrows():
        key = str(row["repair_key"])
        entry = registry.require(
            str(row["task_id"]),
            str(row["protocol"]),
            str(row["split_id"]),
        )
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        run_root, _ = _validate_tabpfn_refresh_path(
            str(row["run_root"]),
            str(row["official_output_root"]),
        )
        status_row = status[status["repair_key"].astype(str).eq(key)].iloc[0]

        if resume and run_root.exists():
            verification = _verify_tabpfn_api25_all_unit(
                row,
                entry,
                frozen_entry,
                snapshot_id=parameter_registry.snapshot_id,
            )
            if verification["verification_status"] == "passed":
                status = _set_status_values(
                    status,
                    key,
                    run_status="success",
                    verification_status="passed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    failed_attempt_path="",
                    last_error="",
                )
                _write_tabpfn_api25_all_status(status, output)
                skipped += 1
                continue

        failed_attempt_path = ""
        if run_root.exists():
            failed_attempt = _move_failed_attempt(run_root, output, f"api25_all__{key}")
            failed_attempt_path = str(failed_attempt)
            print(
                f"[REPLACE] moved prior TabPFN result to {failed_attempt}",
                flush=True,
            )
        attempts = int(status_row.get("attempt_count", 0) or 0) + 1
        status = _set_status_values(
            status,
            key,
            run_status="running",
            verification_status="pending",
            attempt_count=attempts,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at="",
            failed_attempt_path=failed_attempt_path,
            last_error="",
        )
        _write_tabpfn_api25_all_status(status, output)
        try:
            run_tabpfn_repair_item(
                row,
                registry,
                parameter_registry,
                workspace_root=workspace,
            )
            reorder_tabpfn_prediction_rows(
                entry,
                run_root / "predictions" / "all_predictions.csv",
            )
            refresh_tabpfn_metrics_and_manifest(
                entry,
                run_root,
                frozen_entry=frozen_entry,
                snapshot_id=parameter_registry.snapshot_id,
                snapshot_path=parameter_registry.path,
                repair_key=key,
                repair_id=str(manifest["repair_id"]),
            )
            verification = _verify_tabpfn_api25_all_unit(
                row,
                entry,
                frozen_entry,
                snapshot_id=parameter_registry.snapshot_id,
            )
            if verification["verification_status"] != "passed":
                raise ValueError(str(verification["verification_reason"]))
            if str(row["split_id"]) != "main":
                rebuild_parent_manifest(row, registry)
            status = _set_status_values(
                status,
                key,
                run_status="success",
                verification_status="passed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                last_error="",
            )
            succeeded += 1
        except Exception as exc:
            partial_attempt = ""
            if run_root.exists():
                _validate_tabpfn_refresh_path(
                    run_root,
                    str(row["official_output_root"]),
                )
                partial_attempt = str(
                    _move_failed_attempt(run_root, output, f"api25_all_failed__{key}")
                )
            message = f"{type(exc).__name__}: {exc}"
            if partial_attempt:
                message = f"{message}; failed_attempt={partial_attempt}"
            status = _set_status_values(
                status,
                key,
                run_status="failed",
                verification_status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                failed_attempt_path=partial_attempt or failed_attempt_path,
                last_error=message,
            )
            failed += 1
        finally:
            _write_tabpfn_api25_all_status(status, output)
            _cleanup_accelerator_memory()

    complete = int(status["run_status"].astype(str).eq("success").sum()) == 300
    if complete:
        _rebuild_all_affected_parent_manifests(inventory, registry)
    manifest["api25_all_rerun_complete"] = complete
    manifest["api25_all_rerun_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["api25_all_rerun_status_counts"] = dict(
        Counter(status["run_status"].astype(str))
    )
    manifest["verification_complete"] = False
    _atomic_json_write(output / "manifest.json", manifest)
    return {
        "selected": len(selected),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "executed": True,
        "complete": complete,
    }


def _manifest_contains_legacy_tabpfn_path(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_manifest_contains_legacy_tabpfn_path(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_manifest_contains_legacy_tabpfn_path(value) for value in payload)
    if not isinstance(payload, str) or ("/" not in payload and "\\" not in payload):
        return False
    normalized = payload.replace("\\", "/").lower()
    if "ood_results_tabpfn-v2-numeric" in normalized:
        return True
    result_roots = (
        "ood_results_tabpfn-2.5-plus-numeric",
        "ood_results_tabpfn-2.5-plus-text",
    )
    if not any(root in normalized for root in result_roots):
        return False
    legacy_protocol_dirs = (
        "/loco/",
        "/sparse_x_single/",
        "/sparse_x_cluster/",
        "/sparse_y_single/",
        "/sparse_y_cluster/",
    )
    return any(fragment in normalized for fragment in legacy_protocol_dirs)


def run_verify_tabpfn_v1_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    output, manifest, inventory, status, parameter_registry = _load_tabpfn_v1_refresh_state(
        workspace,
        output_root,
    )
    api25_all_complete = bool(manifest.get("api25_all_rerun_complete"))
    legacy_refresh_complete = bool(
        manifest.get("canonicalize_complete") and manifest.get("rerun_complete")
    )
    if not api25_all_complete and not legacy_refresh_complete:
        raise ValueError(
            "Complete either the 300-unit API 2.5 rerun or the legacy refresh before verification"
        )
    registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    _rebuild_all_affected_parent_manifests(inventory, registry)
    rows: list[dict[str, Any]] = []
    for _, row in inventory.iterrows():
        entry = registry.require(str(row["task_id"]), str(row["protocol"]), str(row["split_id"]))
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        base = verify_repair_item(
            row,
            entry,
            expected_snapshot_id=parameter_registry.snapshot_id,
            expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
        )
        reasons = [] if base["verification_status"] == "passed" else [base["verification_reason"]]
        run_root = Path(str(row["run_root"]))
        split = compare_test_split_to_canonical(
            entry,
            run_root / "split_data" / entry.test_file.name,
        )
        prediction = _audit_tabpfn_prediction(entry, run_root)
        runtime = _audit_tabpfn_runtime(frozen_entry, run_root / "pipeline_manifest.json")
        quantile = verify_tabpfn_api25_quantile_artifact(entry, run_root)
        mean_match = _verify_tabpfn_quantile_mean_matches_ood_prediction(run_root)
        if split["comparison_status"] != "exact":
            reasons.append(f"split_full_{split['comparison_status']}")
        if prediction["prediction_status"] != "exact":
            reasons.append(f"prediction_{prediction['prediction_status']}")
        if runtime["runtime_status"] != "exact":
            reasons.append(f"runtime_{runtime['runtime_status']}")
        if quantile["verification_status"] != "passed":
            reasons.append(str(quantile["verification_reason"]))
        if mean_match["verification_status"] != "passed":
            reasons.append(str(mean_match["verification_reason"]))
        pipeline = json.loads((run_root / "pipeline_manifest.json").read_text(encoding="utf-8"))
        if _manifest_contains_legacy_tabpfn_path(pipeline):
            reasons.append("legacy_manifest_path")
        verification_status = "failed" if reasons else "passed"
        rows.append(
            {
                **base,
                "verification_status": verification_status,
                "verification_reason": ";".join(filter(None, dict.fromkeys(reasons))),
                "split_full_status": split["comparison_status"],
                "prediction_order_status": prediction["prediction_status"],
                "runtime_status": runtime["runtime_status"],
                "quantile_status": quantile["verification_status"],
                "quantile_prediction_status": mean_match["verification_status"],
                "quantile_path": quantile["quantile_path"],
            }
        )
        status = _set_status_values(
            status,
            str(row["repair_key"]),
            verification_status=verification_status,
            last_error=("" if verification_status == "passed" else ";".join(reasons)),
        )
    report = pd.DataFrame(rows)
    report.to_csv(output / "verification_report.csv", index=False)
    _write_tabpfn_refresh_status(status, output)
    api25_status_path = output / "tabpfn_api25_all_status.csv"
    api25_status_success = not api25_all_complete
    if api25_status_path.exists():
        api25_status = pd.read_csv(api25_status_path, keep_default_na=False)
        for _, verification_row in report.iterrows():
            api25_status = _set_status_values(
                api25_status,
                str(verification_row["repair_key"]),
                run_status=(
                    "success"
                    if verification_row["verification_status"] == "passed"
                    else "failed"
                ),
                verification_status=str(verification_row["verification_status"]),
                last_error=str(verification_row["verification_reason"]),
            )
        _write_tabpfn_api25_all_status(api25_status, output)
        api25_status_success = bool(
            len(api25_status) == 300
            and api25_status["run_status"].astype(str).eq("success").all()
            and api25_status["verification_status"].astype(str).eq("passed").all()
        )

    cross_audit, cross_summary = build_cross_family_test_split_audit(registry, workspace)
    cross_audit.to_csv(output / "cross_family_test_split_audit.csv", index=False)
    cross_summary.to_csv(output / "cross_family_test_split_summary.csv", index=False)
    protected = pd.read_csv(output / "protected_reference_test_splits.csv", keep_default_na=False)
    protected_unchanged = True
    for _, row in protected.iterrows():
        path = Path(str(row["result_test_path"]))
        if not path.exists() or _file_sha256(path) != str(row["protected_file_sha256"]):
            protected_unchanged = False
            break

    parent_rows: list[dict[str, Any]] = []
    multi = inventory[inventory["protocol"].astype(str).isin(["loco", "random_cv_baseline"])]
    for _, group in multi.groupby(
        ["task_id", "protocol", "model", "parent_result_root"],
        sort=True,
    ):
        protocol = str(group.iloc[0]["protocol"])
        name = "loco_manifest.json" if protocol == "loco" else "random_cv_baseline_manifest.json"
        path = Path(str(group.iloc[0]["parent_result_root"])) / name
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        parent_rows.append(
            {
                "task_id": str(group.iloc[0]["task_id"]),
                "protocol": protocol,
                "model": str(group.iloc[0]["model"]),
                "manifest_path": str(path),
                "fold_count": int(payload.get("fold_count", 0) or 0),
                "expected_fold_count": int(payload.get("expected_fold_count", 0) or 0),
                "repair_incomplete": bool(payload.get("repair_incomplete", True)),
            }
        )
    parent_report = pd.DataFrame(parent_rows)
    parent_report.to_csv(output / "tabpfn_parent_manifest_verification.csv", index=False)
    parent_passed = bool(
        len(parent_report) == 40
        and parent_report["fold_count"].eq(5).all()
        and parent_report["expected_fold_count"].eq(5).all()
        and ~parent_report["repair_incomplete"].astype(bool).any()
    )
    complete = bool(
        len(report) == 300
        and report["verification_status"].eq("passed").all()
        and len(cross_audit) == 900
        and cross_audit["comparison_status"].eq("exact").all()
        and len(cross_summary) == 150
        and cross_summary["comparison_status"].eq("passed").all()
        and protected_unchanged
        and parent_passed
        and api25_status_success
    )
    manifest.update(
        {
            "verification_complete": complete,
            "verification_completed_at": datetime.now(timezone.utc).isoformat(),
            "verification_passed_count": int(report["verification_status"].eq("passed").sum()),
            "cross_family_exact_count": int(cross_audit["comparison_status"].eq("exact").sum()),
            "protected_reference_hashes_unchanged": protected_unchanged,
            "parent_manifest_count": len(parent_report),
            "parent_manifests_passed": parent_passed,
            "api25_all_status_passed": api25_status_success,
        }
    )
    _atomic_json_write(output / "manifest.json", manifest)
    return report


def run_verify_stage(
    *,
    workspace_root: str | Path,
    split_root: str | Path,
    output_root: str | Path,
    snapshot_path: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace_root).resolve()
    output, manifest, queue, status = _load_locked_state(workspace, output_root)
    split_registry = CanonicalSplitRegistry.from_root(_resolve_against(split_root, workspace))
    parameter_registry = FrozenParameterRegistry(_resolve_against(snapshot_path, workspace))
    _rebuild_all_affected_parent_manifests(queue, split_registry)

    rows = []
    for _, row in queue.iterrows():
        entry = split_registry.require(
            str(row["task_id"]),
            str(row["protocol"]),
            str(row["split_id"]),
        )
        frozen_entry = _frozen_entry_for_row(parameter_registry, row)
        verification = verify_repair_item(
            row,
            entry,
            expected_snapshot_id=parameter_registry.snapshot_id,
            expected_parameter_hash=str(frozen_entry.get("parameter_hash", "")),
        )
        rows.append(verification)
        status = _set_status_values(
            status,
            str(row["repair_key"]),
            verification_status=verification["verification_status"],
            last_error=(
                ""
                if verification["verification_status"] == "passed"
                else verification["verification_reason"]
            ),
        )
    report = pd.DataFrame(rows)
    report.to_csv(output / "verification_report.csv", index=False)
    _write_status(status, output)

    post_audit = build_existing_result_audit(split_registry, workspace)
    post_audit.to_csv(output / "post_repair_result_audit.csv", index=False)
    remaining = post_audit[post_audit["status"].astype(str).ne("reuse_exact")]
    all_passed = bool(
        len(report) == len(queue)
        and report["verification_status"].astype(str).eq("passed").all()
        and remaining.empty
    )
    manifest["verification_completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["verification_passed_count"] = int(
        report["verification_status"].astype(str).eq("passed").sum()
    )
    manifest["remaining_canonical_mismatch_count"] = len(remaining)
    manifest["verification_complete"] = all_passed
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return report


def require_llm_python() -> None:
    current = Path(sys.executable).resolve()
    if current != LLM_PYTHON.resolve():
        raise RuntimeError(
            "OOD repair commands must use the llm environment: "
            f"{LLM_PYTHON}; current interpreter is {current}"
        )


def main() -> int:
    args = create_repair_argument_parser().parse_args()
    require_llm_python()
    workspace = args.workspace_root.resolve()
    if args.stage == "audit":
        result: Any = run_audit_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
        )
    elif args.stage == "archive":
        archive = run_archive_stage(
            workspace_root=workspace,
            output_root=args.output_root,
            archive_root=args.archive_root,
            execute=args.execute,
        )
        result = {
            "count": len(archive),
            "statuses": dict(Counter(archive["archive_status"].astype(str))),
            "executed": args.execute,
        }
    elif args.stage == "canonicalize-layout":
        layout = run_canonicalize_layout_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            archive_root=args.archive_root,
            execute=args.execute,
        )
        result = {
            "count": len(layout),
            "statuses": (
                dict(Counter(layout["migration_status"].astype(str)))
                if not layout.empty
                else {}
            ),
            "executed": args.execute,
        }
    elif args.stage == "prepare-standard-rerun":
        result = run_prepare_standard_rerun_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
        )
    elif args.stage == "delete-standard-rerun":
        deletion = run_delete_standard_rerun_stage(
            workspace_root=workspace,
            output_root=args.output_root,
            execute=args.execute,
        )
        result = {
            "count": len(deletion),
            "statuses": dict(Counter(deletion["deletion_status"].astype(str))),
            "executed": args.execute,
        }
    elif args.stage == "verify-standard-rerun":
        report = run_verify_standard_rerun_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
        )
        result = {
            "count": len(report),
            "statuses": dict(Counter(report["verification_status"].astype(str))),
        }
    elif args.stage == "prepare-tabpfn-v1-refresh":
        result = run_prepare_tabpfn_v1_refresh_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            archive_root=args.archive_root,
            snapshot_path=args.snapshot,
        )
    elif args.stage == "archive-tabpfn-v1-refresh":
        archive = run_archive_tabpfn_v1_refresh_stage(
            workspace_root=workspace,
            output_root=args.output_root,
            archive_root=args.archive_root,
            execute=args.execute,
        )
        result = {
            "count": len(archive),
            "statuses": dict(Counter(archive["archive_status"].astype(str))),
            "executed": args.execute,
        }
    elif args.stage == "canonicalize-tabpfn-v1":
        result = run_canonicalize_tabpfn_v1_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            execute=args.execute,
        )
    elif args.stage == "rerun-tabpfn-v1":
        result = run_rerun_tabpfn_v1_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            execute=args.execute,
            resume=args.resume,
            repair_key=args.repair_key,
        )
    elif args.stage == "rerun-tabpfn-api25-all":
        result = run_rerun_tabpfn_api25_all_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            execute=args.execute,
            resume=args.resume,
            repair_key=args.repair_key,
        )
    elif args.stage == "verify-tabpfn-v1":
        report = run_verify_tabpfn_v1_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
        )
        result = {
            "count": len(report),
            "statuses": dict(Counter(report["verification_status"].astype(str))),
        }
    elif args.stage == "rerun":
        result = run_rerun_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            snapshot_path=args.snapshot,
            model_groups=args.models,
            resume=args.resume,
        )
    elif args.stage == "_run-item":
        if not args.repair_key:
            raise ValueError("--repair-key is required for the internal _run-item stage")
        result = run_single_repair_item_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            snapshot_path=args.snapshot,
            repair_key=args.repair_key,
        )
    elif args.stage == "verify":
        report = run_verify_stage(
            workspace_root=workspace,
            split_root=args.split_root,
            output_root=args.output_root,
            snapshot_path=args.snapshot,
        )
        result = {
            "count": len(report),
            "statuses": dict(Counter(report["verification_status"].astype(str))),
        }
    else:  # pragma: no cover - argparse restricts --stage to the handled values
        raise ValueError(f"Unsupported repair stage: {args.stage}")
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    if args.stage == "rerun" and int(result.get("failed", 0)) > 0:
        return 1
    if args.stage in {
        "canonicalize-tabpfn-v1",
        "rerun-tabpfn-v1",
        "rerun-tabpfn-api25-all",
    } and int(
        result.get("failed", 0)
    ) > 0:
        return 1
    if args.stage in {"verify", "verify-standard-rerun", "verify-tabpfn-v1"} and result.get("statuses", {}).get("failed", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
