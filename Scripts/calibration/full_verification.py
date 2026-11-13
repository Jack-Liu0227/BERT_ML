from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing.canonical_split_registry import CanonicalSplitRegistry

from .full_pipeline import calibration_unit_is_complete
from .parameter_registry import FrozenParameterRegistry


STRICT_EXPECTATIONS = {
    "ti": {
        "split_count": 30,
        "unit_count": 330,
        "family_counts": {
            "Traditional": 150,
            "BERT": 90,
            "LLMProp": 30,
            "TabPFN": 60,
        },
        "tier_summary_count": 462,
        "uncertainty_metric_count": 154,
    },
    "all": {
        "split_count": 150,
        "unit_count": 1650,
        "family_counts": {
            "Traditional": 750,
            "BERT": 450,
            "LLMProp": 150,
            "TabPFN": 300,
        },
        "tier_summary_count": 2310,
        "uncertainty_metric_count": 770,
    },
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_protected_hashes(output_root: Path) -> list[str]:
    path = output_root / "audit" / "protected_file_hashes.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    path_column = next(
        (name for name in ("path", "file_path", "protected_path") if name in frame.columns),
        None,
    )
    hash_column = next(
        (name for name in ("sha256", "file_sha256", "protected_sha256") if name in frame.columns),
        None,
    )
    if path_column is None or hash_column is None:
        return ["protected_file_hashes.csv has no path/SHA256 columns"]
    errors: list[str] = []
    for _, row in frame.iterrows():
        candidate = Path(str(row[path_column]))
        if not candidate.exists():
            errors.append(f"protected file is missing: {candidate}")
        elif _sha256(candidate) != str(row[hash_column]):
            errors.append(f"protected file hash changed: {candidate}")
    return errors


def _scoped_split_count(inventory: pd.DataFrame) -> int:
    required = ["task_id", "canonical_protocol", "split_id"]
    missing = sorted(set(required).difference(inventory.columns))
    if missing:
        raise ValueError(f"Calibration inventory is missing split columns: {missing}")
    return int(inventory[required].drop_duplicates().shape[0])


def verify_full_results(
    *,
    output_root: str | Path,
    split_root: str | Path,
    snapshot_path: str | Path,
    scope: str,
    strict_counts: bool = True,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    splits = Path(split_root).resolve()
    snapshot = Path(snapshot_path).resolve()
    manifest_path = output / "manifest.json"
    inventory_path = output / "audit" / "calibration_run_inventory.csv"
    status_path = output / "status" / "run_status.csv"
    for required in (manifest_path, inventory_path, status_path, snapshot):
        if not required.exists():
            raise FileNotFoundError(required)

    errors: list[str] = []
    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = FrozenParameterRegistry(snapshot)
    split_registry = CanonicalSplitRegistry.from_root(splits)
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    status = pd.read_csv(status_path, keep_default_na=False)
    if str(root_manifest.get("scope")) != str(scope):
        errors.append("root manifest scope mismatch")
    if str(root_manifest.get("snapshot_id")) != registry.snapshot_id:
        errors.append("root manifest snapshot mismatch")
    if str(root_manifest.get("frozen_parameter_file_sha256")) != _sha256(snapshot):
        errors.append("frozen v1 SHA256 changed")
    if Path(str(root_manifest.get("split_root", ""))).resolve() != splits:
        errors.append("root manifest split root mismatch")

    unit_rows: list[dict[str, Any]] = []
    for _, row in inventory.iterrows():
        reasons: list[str] = []
        try:
            split_entry = split_registry.require(
                str(row["task_id"]),
                str(row["canonical_protocol"]),
                str(row["split_id"]),
            )
            frozen = registry.get(
                str(row["task_id"]),
                "",
                "",
                str(row["model_family"]),
                str(row["model"]),
            )
            if not calibration_unit_is_complete(
                row,
                split_entry,
                frozen,
                registry.snapshot_id,
            ):
                reasons.append("unit files or manifest failed fresh verification")
            result_manifest_path = Path(str(row["result_dir"])) / "manifest.json"
            if result_manifest_path.exists() and str(row["model_family"]) == "TabPFN":
                payload = json.loads(result_manifest_path.read_text(encoding="utf-8"))
                metadata = dict(payload.get("adapter_metadata") or {})
                if metadata.get("effective_model_version") != "api_2.5_plus":
                    reasons.append("TabPFN runtime is not api_2.5_plus")
                if metadata.get("requested_model_path") != "v2.5_default":
                    reasons.append("TabPFN model path is not v2.5_default")
                if not (Path(str(row["result_dir"])) / "predictive_quantiles.csv").exists():
                    reasons.append("TabPFN predictive_quantiles.csv is missing")
        except Exception as exc:
            reasons.append(f"{type(exc).__name__}: {exc}")
        unit_rows.append(
            {
                "run_key": str(row["run_key"]),
                "task_id": str(row["task_id"]),
                "protocol": str(row["protocol"]),
                "split_id": str(row["split_id"]),
                "model_family": str(row["model_family"]),
                "model": str(row["model"]),
                "verification_status": "failed" if reasons else "passed",
                "verification_reason": "; ".join(reasons),
            }
        )
    units = pd.DataFrame(unit_rows)
    failed_units = units[units["verification_status"].eq("failed")]
    if not failed_units.empty:
        errors.append(f"{len(failed_units)} calibration units failed fresh verification")
    status_by_key = status.set_index("run_key")["run_status"].astype(str).to_dict()
    non_success_status = [
        key for key in inventory["run_key"].astype(str) if status_by_key.get(key) != "success"
    ]
    if non_success_status:
        errors.append(f"{len(non_success_status)} run-status rows are not success")
    errors.extend(_verify_protected_hashes(output))

    aggregate_root = output / "aggregate"
    tier_path = aggregate_root / "all_tier_error_summary.csv"
    metric_path = aggregate_root / "all_uncertainty_error_metrics.csv"
    coverage_path = aggregate_root / "all_coverage_summary.csv"
    for required in (tier_path, metric_path, coverage_path):
        if not required.exists():
            errors.append(f"aggregate output is missing: {required.name}")
    tier_count = len(pd.read_csv(tier_path)) if tier_path.exists() else 0
    metric_count = len(pd.read_csv(metric_path)) if metric_path.exists() else 0
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    if not coverage.empty and int(pd.to_numeric(coverage["missing_units"]).sum()) != 0:
        errors.append("aggregate coverage reports missing units")

    verified = units[units["verification_status"].eq("passed")]
    family_counts = verified["model_family"].value_counts().to_dict()
    if strict_counts:
        if scope not in STRICT_EXPECTATIONS:
            raise ValueError(f"Unsupported strict verification scope: {scope}")
        expected = STRICT_EXPECTATIONS[scope]
        scoped_split_count = _scoped_split_count(inventory)
        if scoped_split_count != expected["split_count"]:
            errors.append(
                "canonical split count mismatch: "
                f"{scoped_split_count} != {expected['split_count']}"
            )
        if len(inventory) != expected["unit_count"]:
            errors.append(f"unit count mismatch: {len(inventory)} != {expected['unit_count']}")
        if family_counts != expected["family_counts"]:
            errors.append(
                f"verified family counts mismatch: {family_counts} != {expected['family_counts']}"
            )
        if tier_count != expected["tier_summary_count"]:
            errors.append(
                f"tier summary count mismatch: {tier_count} != {expected['tier_summary_count']}"
            )
        if metric_count != expected["uncertainty_metric_count"]:
            errors.append(
                "uncertainty metric count mismatch: "
                f"{metric_count} != {expected['uncertainty_metric_count']}"
            )

    audit_root = output / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    units.to_csv(audit_root / "full_verification_units.csv", index=False)
    report = {
        "schema_version": 1,
        "stage": "full-verify",
        "scope": scope,
        "verification_status": "failed" if errors else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": registry.snapshot_id,
        "inventory_unit_count": int(len(inventory)),
        "verified_unit_count": int(len(verified)),
        "failed_unit_count": int(len(failed_units)),
        "verified_family_counts": family_counts,
        "tier_summary_count": int(tier_count),
        "uncertainty_metric_count": int(metric_count),
        "protected_hash_error_count": int(len(_verify_protected_hashes(output))),
        "errors": errors,
    }
    _atomic_json_write(audit_root / "full_verification_report.json", report)
    if errors:
        raise ValueError("Full calibration verification failed: " + "; ".join(errors))
    return report
