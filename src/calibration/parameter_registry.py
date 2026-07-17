from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .parameter_audit import write_parameter_audit
from .schemas import CalibrationConfig


FREEZABLE_STATUSES = {"reusable", "not_applicable"}
MATBENCH_CANONICAL_TASK_ID = "Matbench_Steel__matbench_steels_ood__yield_strength"
MATBENCH_EXPECTED_MODELS = {
    ("Traditional", "CatBoost"),
    ("Traditional", "LightGBM"),
    ("Traditional", "MLP"),
    ("Traditional", "RF"),
    ("Traditional", "XGB"),
    ("BERT", "MatSciBERT"),
    ("BERT", "SciBERT"),
    ("BERT", "SteelBERT"),
    ("LLMProp", "LLM-Prop"),
    ("TabPFN", "TabPFN-2.5-Plus-Numeric"),
    ("TabPFN", "TabPFN-2.5-Plus-Text"),
}
TI_PILOT_EXCLUDED_MODELS = {"TabPFN-V2-Numeric"}


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(
        str(entry[name])
        for name in ("task_id", "protocol", "split_id", "model_family", "model")
    )


def count_ti_pilot_entries(entries: list[dict[str, Any]]) -> int:
    return sum(
        str(entry.get("task_id", "")).startswith("Ti__")
        and str(entry.get("model", "")) not in TI_PILOT_EXCLUDED_MODELS
        for entry in entries
    )


def assert_scoped_entries_unchanged(
    old_entries: list[dict[str, Any]],
    new_entries: list[dict[str, Any]],
    *,
    task_prefix: str,
) -> None:
    old_scope = {_entry_key(entry): entry for entry in old_entries if str(entry.get("task_id", "")).startswith(task_prefix)}
    new_scope = {_entry_key(entry): entry for entry in new_entries if str(entry.get("task_id", "")).startswith(task_prefix)}
    if old_scope != new_scope:
        old_keys = set(old_scope)
        new_keys = set(new_scope)
        changed = sorted(key for key in old_keys & new_keys if old_scope[key] != new_scope[key])
        raise ValueError(
            "Ti frozen entries changed during snapshot repair: "
            f"missing={sorted(old_keys - new_keys)}, added={sorted(new_keys - old_keys)}, changed={changed}"
        )


def _normalize_missing_source_paths(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(entry))
    source = normalized.get("source", {})
    if isinstance(source, dict):
        for name in ("parameter_config", "parameter_evidence", "split_manifest"):
            value = str(source.get(name, "") or "").strip()
            if value.lower() == "nan":
                source[name] = ""
    return normalized


def preserve_scoped_entries(
    old_entries: list[dict[str, Any]],
    discovered_entries: list[dict[str, Any]],
    *,
    task_prefix: str,
) -> list[dict[str, Any]]:
    old_scope = {_entry_key(entry): entry for entry in old_entries if str(entry.get("task_id", "")).startswith(task_prefix)}
    discovered_scope = {
        _entry_key(entry): entry
        for entry in discovered_entries
        if str(entry.get("task_id", "")).startswith(task_prefix)
    }
    if set(old_scope) != set(discovered_scope):
        raise ValueError("Scoped frozen entry keys changed during snapshot repair.")
    changed = [
        key
        for key in sorted(old_scope)
        if _normalize_missing_source_paths(old_scope[key])
        != _normalize_missing_source_paths(discovered_scope[key])
    ]
    if changed:
        raise ValueError(f"Scoped frozen entry semantic content changed during snapshot repair: {changed}")
    merged = [entry for entry in discovered_entries if not str(entry.get("task_id", "")).startswith(task_prefix)]
    merged.extend(old_scope[key] for key in sorted(old_scope))
    return sorted(merged, key=_entry_key)


def _snapshot_id(entries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(entries, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_frozen_payload(
    entries: list[dict[str, Any]],
    *,
    version: str,
    source_config: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": version,
        "snapshot_id": _snapshot_id(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_config": source_config,
        "entry_count": len(entries),
        "entries": entries,
    }


def validate_matbench_canonical_snapshot(
    entries: list[dict[str, Any]],
    *,
    expected_total_entries: int,
) -> None:
    if len(entries) != expected_total_entries:
        raise ValueError(f"Frozen entry count mismatch: expected {expected_total_entries}, found {len(entries)}")
    noncanonical = [
        str(entry.get("task_id", ""))
        for entry in entries
        if "matbench" in str(entry.get("task_id", "")).lower()
        and str(entry.get("task_id")) != MATBENCH_CANONICAL_TASK_ID
    ]
    if noncanonical:
        raise ValueError(f"Frozen snapshot contains non-canonical Matbench task IDs: {sorted(set(noncanonical))}")
    matbench = [entry for entry in entries if str(entry.get("task_id")) == MATBENCH_CANONICAL_TASK_ID]
    if len(matbench) != 165:
        raise ValueError(f"Canonical Matbench task must contain 165 entries, found {len(matbench)}")
    keys = [_entry_key(entry) for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen snapshot contains duplicate entry keys.")
    models = {(str(entry["model_family"]), str(entry["model"])) for entry in matbench}
    if models != MATBENCH_EXPECTED_MODELS:
        raise ValueError(f"Canonical Matbench model coverage mismatch: {sorted(models)}")
    protocol_splits = {(str(entry["protocol"]), str(entry["split_id"])) for entry in matbench}
    if len(protocol_splits) != 15:
        raise ValueError(f"Canonical Matbench protocol/split coverage must contain 15 keys, found {len(protocol_splits)}")


def _entry_from_row(row: pd.Series) -> dict[str, Any]:
    payload_text = str(row.get("parameter_payload_json", "") or "")
    payload = json.loads(payload_text) if payload_text else {}
    return {
        "task_id": str(row["task_id"]),
        "protocol": str(row["protocol"]),
        "split_id": str(row["split_id"]),
        "model_family": str(row["model_family"]),
        "model": str(row["model"]),
        "audit_status": str(row["status"]),
        "selection_basis": str(row["parameter_selection_basis"]),
        "effective_parameters": payload,
        "train_split_hash": str(row.get("train_split_hash", "") or ""),
        "test_split_hash": str(row.get("test_split_hash", "") or ""),
        "source": {
            "model_dir": str(row["model_dir"]),
            "parameter_config": str(row.get("parameter_config_path", "") or ""),
            "parameter_evidence": str(row.get("parameter_evidence_path", "") or ""),
            "split_manifest": str(row.get("split_manifest_path", "") or ""),
        },
    }


def build_entries_from_audit(audit: pd.DataFrame) -> list[dict[str, Any]]:
    freezable = audit[audit["status"].isin(FREEZABLE_STATUSES)].copy()
    return [
        _entry_from_row(row)
        for _, row in freezable.sort_values(
            ["task_id", "protocol", "split_id", "model_family", "model"]
        ).iterrows()
    ]


def freeze_parameters(
    cfg: CalibrationConfig,
    audit: pd.DataFrame | None = None,
    version: str = "v1",
) -> Path:
    if audit is None:
        audit_path = cfg.output_root / "parameter_freeze" / "parameter_source_audit.csv"
        audit = pd.read_csv(audit_path) if audit_path.exists() else write_parameter_audit(cfg)
    entries = build_entries_from_audit(audit)
    output_dir = cfg.output_root / "parameter_freeze"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frozen_model_params_{version}.yaml"
    payload = build_frozen_payload(entries, version=version, source_config=str(cfg.path))
    snapshot_id = str(payload["snapshot_id"])

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
        if existing.get("snapshot_id") != snapshot_id:
            raise FileExistsError(
                f"Frozen parameter snapshot already exists with different content: {output_path}. "
                "Use a new version such as v2."
            )
        return output_path

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "stage": "freeze-parameters",
                "generated_at": payload["generated_at"],
                "snapshot_id": snapshot_id,
                "snapshot_path": str(output_path),
                "entry_count": len(entries),
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )
    return output_path


class FrozenParameterRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.payload = yaml.safe_load(handle) or {}
        self.snapshot_id = str(self.payload.get("snapshot_id", ""))
        self.key_fields = tuple(
            str(value)
            for value in self.payload.get(
                "key_fields",
                ("task_id", "protocol", "split_id", "model_family", "model"),
            )
        )
        self._entries = {
            self._registry_key(entry): entry
            for entry in self.payload.get("entries", [])
            if isinstance(entry, dict)
        }

    @property
    def is_task_model_registry(self) -> bool:
        return self.key_fields == ("task_id", "model_family", "model")

    def _registry_key(self, entry: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(entry[name]) for name in self.key_fields)

    @staticmethod
    def entry_key(entry: dict[str, Any]) -> str:
        fields = (
            ("task_id", "protocol", "split_id", "model_family", "model")
            if "protocol" in entry and "split_id" in entry
            else ("task_id", "model_family", "model")
        )
        return "||".join(str(entry[name]) for name in fields)

    def iter_entries(self) -> list[dict[str, Any]]:
        return [self._entries[key] for key in sorted(self._entries)]

    def iter_run_entries(self, audit_path: str | Path | None = None) -> list[dict[str, Any]]:
        if self.key_fields != ("task_id", "model_family", "model"):
            return self.iter_entries()
        source_path = Path(audit_path) if audit_path is not None else self.path.parent / "parameter_source_audit.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Parameter source audit is required to expand task/model entries: {source_path}")
        audit = pd.read_csv(source_path)
        audit = audit[
            audit["status"].astype(str).isin(FREEZABLE_STATUSES)
            & ~audit["model"].astype(str).eq("TabPFN-V2-Numeric")
        ].copy()
        audit["task_id"] = audit["task_id"].astype(str).str.replace(
            r"^MatbenchSteels__",
            "Matbench_Steel__",
            regex=True,
        )
        run_key = ["task_id", "protocol", "split_id", "model_family", "model"]
        audit = audit.sort_values(run_key + ["model_dir"], kind="stable").drop_duplicates(run_key, keep="first")
        entries: list[dict[str, Any]] = []
        for _, row in audit.iterrows():
            task_id = str(row["task_id"])
            model_family = str(row["model_family"])
            model = str(row["model"])
            try:
                frozen = self.get(task_id, "", "", model_family, model)
            except KeyError:
                continue
            entry = deepcopy(frozen)
            entry.update(
                {
                    "protocol": str(row["protocol"]),
                    "split_id": str(row["split_id"]),
                    "train_split_hash": str(row.get("train_split_hash", "") or ""),
                    "test_split_hash": str(row.get("test_split_hash", "") or ""),
                    "source": {
                        "model_dir": str(row.get("model_dir", "") or ""),
                        "parameter_config": str(row.get("parameter_config_path", "") or ""),
                        "parameter_evidence": str(row.get("parameter_evidence_path", "") or ""),
                        "split_manifest": str(row.get("split_manifest_path", "") or ""),
                    },
                }
            )
            entries.append(entry)
        return sorted(entries, key=lambda entry: tuple(str(entry[name]) for name in run_key))

    def get(self, task_id: str, protocol: str, split_id: str, model_family: str, model: str) -> dict[str, Any]:
        if self.key_fields == ("task_id", "model_family", "model"):
            key = (task_id, model_family, model)
        else:
            key = (task_id, protocol, split_id, model_family, model)
        if key not in self._entries:
            raise KeyError(f"No frozen parameters for key: {key}")
        return self._entries[key]

    def validate_split_hashes(
        self,
        entry: dict[str, Any],
        train_split_hash: str,
        test_split_hash: str,
    ) -> None:
        expected_train = str(entry.get("train_split_hash", "") or "")
        expected_test = str(entry.get("test_split_hash", "") or "")
        if expected_train and expected_train != train_split_hash:
            raise ValueError("Training split hash does not match the frozen parameter snapshot.")
        if expected_test and expected_test != test_split_hash:
            raise ValueError("Test split hash does not match the frozen parameter snapshot.")
