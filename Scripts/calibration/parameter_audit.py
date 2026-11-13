from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import PARAMETER_AUDIT_COLUMNS, CalibrationConfig
from .sources import ParameterCandidate, discover_parameter_candidates


FINAL_CONFIG_NAMES = (
    "final_best_model_config.json",
    "best_model_config.json",
    "llmprop_manifest.json",
    "pipeline_manifest.json",
)
OPTUNA_EVIDENCE_NAMES = (
    "optuna_best_params.json",
    "best_optuna_params.json",
    "llmprop_optuna_summary.json",
)


@lru_cache(maxsize=None)
def _json_load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_local_file(model_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = model_dir / name
        if direct.exists():
            return direct
    for name in names:
        nested = model_dir / "optuna_trials" / name
        if nested.exists():
            return nested
    return None


@lru_cache(maxsize=None)
def _find_upwards(model_dir: Path, relative: str, max_levels: int = 9) -> Path | None:
    current = model_dir
    for _ in range(max_levels + 1):
        candidate = current / relative
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_manifest_path(model_dir: Path, value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), model_dir, *model_dir.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _find_pipeline_manifest(model_dir: Path) -> Path | None:
    return _find_upwards(model_dir, "pipeline_manifest.json")


def _find_split_manifest(model_dir: Path, pipeline_payload: dict[str, Any]) -> Path | None:
    artifacts = pipeline_payload.get("split_artifacts", {})
    if isinstance(artifacts, dict):
        trace_manifest = _resolve_manifest_path(model_dir, artifacts.get("trace_manifest"))
        if trace_manifest is not None and trace_manifest.exists():
            return trace_manifest
    return _find_upwards(model_dir, "split_data/trace/split_manifest.json")


def _hash_values(values: pd.Series) -> str:
    canonical = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=None)
def _csv_identity_hash(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        id_column = next((name for name in ("ID", "id", "sample_id", "__row_id__", "__source_index__") if name in columns), None)
        if id_column is not None:
            values = pd.read_csv(path, usecols=[id_column])[id_column]
            return _hash_values(values)
    except (OSError, ValueError, pd.errors.ParserError):
        return ""
    return ""


def _identity_from_payload(payload: dict[str, Any], key: str) -> str:
    identity = payload.get(key, {})
    if isinstance(identity, dict):
        return str(identity.get("row_hash", "") or "")
    return ""


def _split_hashes(
    model_dir: Path,
    config_payload: dict[str, Any],
    pipeline_payload: dict[str, Any],
    split_payload: dict[str, Any],
) -> tuple[str, str]:
    train_hash = str(config_payload.get("train_split_hash", "") or "")
    test_hash = str(config_payload.get("test_split_hash", "") or "")
    train_hash = train_hash or _identity_from_payload(split_payload, "train_identity")
    test_hash = test_hash or _identity_from_payload(split_payload, "test_identity")

    artifacts = pipeline_payload.get("split_artifacts", {})
    if isinstance(artifacts, dict):
        train_file = _resolve_manifest_path(model_dir, artifacts.get("train_file"))
        test_file = _resolve_manifest_path(model_dir, artifacts.get("test_file"))
        train_hash = train_hash or _csv_identity_hash(train_file)
        test_hash = test_hash or _csv_identity_hash(test_file)

    if not train_hash or not test_hash:
        ood_manifest_path = _find_upwards(model_dir, "ood_manifest.json", max_levels=3)
        ood_payload = _json_load(ood_manifest_path)
        train_file = _resolve_manifest_path(model_dir, ood_payload.get("train_feature_file"))
        test_file = _resolve_manifest_path(model_dir, ood_payload.get("test_feature_file"))
        train_hash = train_hash or _csv_identity_hash(train_file)
        test_hash = test_hash or _csv_identity_hash(test_file)
    return train_hash, test_hash


def _model_kind(candidate: ParameterCandidate) -> str:
    family = candidate.model_family.lower()
    model = candidate.model.lower()
    if "tabpfn" in family or "tabpfn" in model:
        return "tabpfn"
    if "llmprop" in family or "llm-prop" in model or "llmprop" in model:
        return "llmprop"
    if "bert" in family or "bert" in model:
        return "bert"
    if family in {"traditional", "ml", "machine learning"}:
        return "traditional"
    return "generic"


def _llmprop_effective_config(payload: dict[str, Any]) -> dict[str, Any]:
    training = payload.get("training_hyperparameters", {})
    if not isinstance(training, dict):
        training = {}
    excluded = {"result_dir", "train_csv", "test_csv", "split_manifest"}
    effective = {key: value for key, value in training.items() if key not in excluded}
    for key in ("base_model_path", "tokenizer_path", "normalizer"):
        if key in payload:
            effective[key] = payload[key]
    return effective


def _tabpfn_effective_config(payload: dict[str, Any]) -> dict[str, Any]:
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


def _effective_config(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "traditional":
        params = payload.get("model_params")
        if isinstance(params, dict):
            return {"model_type": payload.get("model_type"), "model_params": params}
    if kind == "llmprop":
        return _llmprop_effective_config(payload)
    if kind == "tabpfn":
        return _tabpfn_effective_config(payload)
    return payload


def _selection_basis(kind: str, config_payload: dict[str, Any], evidence_payload: dict[str, Any]) -> str:
    if kind == "tabpfn":
        return "fixed_runtime"
    if kind == "llmprop":
        metrics = config_payload.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("best_valid_mae") is not None:
            return "inner_validation_mae"
        if evidence_payload.get("best_params"):
            return "inner_validation_optuna"
        return "unknown"
    if evidence_payload:
        return "inner_cv_optuna"
    return "unknown"


def audit_candidate(candidate: ParameterCandidate) -> dict[str, Any]:
    model_dir = candidate.model_dir
    reporting_uses_ood = "oodtest" in candidate.reporting_selection_mode.lower()
    base = {
        "task_id": candidate.task_id,
        "protocol": candidate.protocol,
        "split_id": candidate.split_id,
        "model_family": candidate.model_family,
        "model": candidate.model,
        "source_root": candidate.source_root,
        "model_dir": str(model_dir),
        "reporting_selection_mode": candidate.reporting_selection_mode,
        "reporting_uses_ood_test": reporting_uses_ood,
    }
    if not model_dir.exists():
        return {
            **base,
            "status": "missing",
            "parameter_config_path": "",
            "parameter_evidence_path": "",
            "split_manifest_path": "",
            "train_split_hash": "",
            "test_split_hash": "",
            "parameter_selection_basis": "unknown",
            "parameter_payload_json": "",
            "message": "Model result directory does not exist.",
        }

    kind = _model_kind(candidate)
    config_path = _find_local_file(model_dir, FINAL_CONFIG_NAMES)
    if kind == "tabpfn":
        config_path = _find_pipeline_manifest(model_dir) or config_path
    evidence_path = _find_local_file(model_dir, OPTUNA_EVIDENCE_NAMES)
    config_payload = _json_load(config_path)
    evidence_payload = _json_load(evidence_path)
    pipeline_path = _find_pipeline_manifest(model_dir)
    pipeline_payload = _json_load(pipeline_path)
    split_path = _find_split_manifest(model_dir, pipeline_payload)
    split_payload = _json_load(split_path)
    train_hash, test_hash = _split_hashes(model_dir, config_payload, pipeline_payload, split_payload)
    effective = _effective_config(kind, config_payload)
    basis = _selection_basis(kind, config_payload, evidence_payload)

    if not effective:
        status = "missing"
        message = "No effective model configuration was found."
    elif kind == "tabpfn":
        required = {"resolved_backend", "feature_mode", "model_info"}
        if required.issubset(effective):
            status = "not_applicable"
            message = "TabPFN uses a fixed runtime configuration; Optuna is not applicable."
        else:
            status = "incomplete"
            message = "TabPFN runtime configuration is incomplete."
    elif basis == "unknown":
        status = "incomplete"
        message = "Effective parameters exist, but their validation-based selection evidence is missing."
    elif not train_hash or not test_hash:
        status = "incomplete"
        message = "Effective parameters exist, but train/test split identity is incomplete."
    elif "oodtest" in basis.lower():
        status = "unsafe_selection"
        message = "Parameter selection appears to use OOD test performance."
    else:
        status = "reusable"
        message = "Existing parameters are reusable."

    if reporting_uses_ood:
        message += " OOD-test representative selection is recorded for reporting only and is not used here."

    return {
        **base,
        "status": status,
        "parameter_config_path": str(config_path or ""),
        "parameter_evidence_path": str(evidence_path or ""),
        "split_manifest_path": str(split_path or ""),
        "train_split_hash": train_hash,
        "test_split_hash": test_hash,
        "parameter_selection_basis": basis,
        "parameter_payload_json": json.dumps(effective, ensure_ascii=True, sort_keys=True),
        "message": message,
    }


def build_parameter_audit(cfg: CalibrationConfig) -> pd.DataFrame:
    rows = [audit_candidate(candidate) for candidate in discover_parameter_candidates(cfg)]
    return pd.DataFrame(rows, columns=PARAMETER_AUDIT_COLUMNS)


def write_parameter_audit(cfg: CalibrationConfig) -> pd.DataFrame:
    output_dir = cfg.output_root / "parameter_freeze"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_parameter_audit(cfg)
    audit_path = output_dir / "parameter_source_audit.csv"
    audit.to_csv(audit_path, index=False)
    missing = audit[audit["status"].isin(["missing", "incomplete", "unsafe_selection"])].copy()
    missing.to_csv(output_dir / "missing_or_unsafe_parameters.csv", index=False)
    status_counts = {str(key): int(value) for key, value in audit["status"].value_counts().items()}
    manifest = {
        "schema_version": 1,
        "stage": "audit-parameters",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg.path),
        "row_count": int(len(audit)),
        "status_counts": status_counts,
        "audit_csv": str(audit_path),
    }
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    return audit
