from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from .parameter_audit import build_parameter_audit
from .parameter_registry import (
    assert_scoped_entries_unchanged,
    build_entries_from_audit,
    build_frozen_payload,
    count_ti_pilot_entries,
    preserve_scoped_entries,
    validate_matbench_canonical_snapshot,
)
from .schemas import CalibrationConfig


SNAPSHOT_PATTERN = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def migrate_ti_snapshot_references(
    pilot_root: str | Path,
    old_snapshot_id: str,
    new_snapshot_id: str,
) -> dict[str, Any]:
    root = Path(pilot_root)
    if not root.exists():
        raise FileNotFoundError(f"Ti pilot root does not exist: {root}")
    csv_updates: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.csv")):
        text = path.read_text(encoding="utf-8")
        snapshot_ids = set(SNAPSHOT_PATTERN.findall(text))
        if not snapshot_ids:
            continue
        unexpected = snapshot_ids - {old_snapshot_id, new_snapshot_id}
        if unexpected:
            raise ValueError(f"Ti CSV contains an unexpected snapshot ID: {path}: {sorted(unexpected)}")
        if old_snapshot_id in snapshot_ids:
            csv_updates.append((path, text.replace(old_snapshot_id, new_snapshot_id)))

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ti pilot manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_snapshot = str(manifest.get("parameter_snapshot_id", ""))
    if manifest_snapshot not in {old_snapshot_id, new_snapshot_id}:
        raise ValueError(f"Ti manifest contains an unexpected snapshot ID: {manifest_snapshot}")
    for path, text in csv_updates:
        _atomic_write_text(path, text)
    manifest["parameter_snapshot_id"] = new_snapshot_id
    manifest["snapshot_migration"] = {
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "old_snapshot_id": old_snapshot_id,
        "new_snapshot_id": new_snapshot_id,
        "reason": "Matbench Steel task-ID canonicalization in frozen v1; Ti frozen entries unchanged.",
        "model_inference_rerun": False,
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    return {
        "old_snapshot_id": old_snapshot_id,
        "new_snapshot_id": new_snapshot_id,
        "updated_csv_count": len(csv_updates),
        "manifest": str(manifest_path),
    }


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    temporary.replace(path)


def repair_v1_matbench_snapshot(cfg: CalibrationConfig) -> dict[str, Any]:
    output_dir = cfg.output_root / "parameter_freeze"
    snapshot_path = output_dir / "frozen_model_params_v1.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Frozen v1 snapshot does not exist: {snapshot_path}")
    old_payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    old_entries = [entry for entry in old_payload.get("entries", []) if isinstance(entry, dict)]
    old_snapshot_id = str(old_payload.get("snapshot_id", ""))

    audit = build_parameter_audit(cfg)
    entries = build_entries_from_audit(audit)
    entries = preserve_scoped_entries(old_entries, entries, task_prefix="Ti__")
    validate_matbench_canonical_snapshot(entries, expected_total_entries=1690)
    if any("gpt" in str(entry.get("model", "")).lower() for entry in entries):
        raise ValueError("Frozen v1 repair unexpectedly includes GPT entries.")
    old_ti_total_count = sum(str(entry.get("task_id", "")).startswith("Ti__") for entry in old_entries)
    new_ti_total_count = sum(str(entry.get("task_id", "")).startswith("Ti__") for entry in entries)
    if old_ti_total_count != 350 or new_ti_total_count != 350:
        raise ValueError(
            "Total Ti frozen entry count must remain 350: "
            f"old={old_ti_total_count}, new={new_ti_total_count}"
        )
    old_ti_pilot_count = count_ti_pilot_entries(old_entries)
    new_ti_pilot_count = count_ti_pilot_entries(entries)
    if old_ti_pilot_count != 330 or new_ti_pilot_count != 330:
        raise ValueError(
            "Ti pilot entry count must remain 330: "
            f"old={old_ti_pilot_count}, new={new_ti_pilot_count}"
        )
    assert_scoped_entries_unchanged(old_entries, entries, task_prefix="Ti__")

    new_payload = build_frozen_payload(entries, version="v1", source_config=str(cfg.path))
    new_snapshot_id = str(new_payload["snapshot_id"])
    if new_snapshot_id == old_snapshot_id:
        raise ValueError("Matbench v1 repair did not change the snapshot content.")

    status_counts = {str(key): int(value) for key, value in audit["status"].value_counts().items()}
    missing = audit[audit["status"].isin(["missing", "incomplete", "unsafe_selection"])].copy()
    generated_at = str(new_payload["generated_at"])
    audit_manifest = {
        "schema_version": 1,
        "stage": "audit-parameters",
        "generated_at": generated_at,
        "config": str(cfg.path),
        "row_count": int(len(audit)),
        "status_counts": status_counts,
        "audit_csv": str(output_dir / "parameter_source_audit.csv"),
        "canonicalization": {
            "MatbenchSteels": "Matbench Steel",
            "Matbench_Steel": "Matbench Steel",
        },
    }
    freeze_manifest = {
        "schema_version": 1,
        "stage": "repair-v1-matbench",
        "generated_at": generated_at,
        "version": "v1",
        "snapshot_id": new_snapshot_id,
        "snapshot_path": str(snapshot_path),
        "entry_count": len(entries),
        "repair": {
            "old_snapshot_id": old_snapshot_id,
            "new_snapshot_id": new_snapshot_id,
            "old_entry_count": len(old_entries),
            "new_entry_count": len(entries),
            "canonical_task_id": "Matbench_Steel__matbench_steels_ood__yield_strength",
            "removed_duplicate_entry_count": len(old_entries) - len(entries),
            "ti_total_entry_count": new_ti_total_count,
            "ti_pilot_entry_count": new_ti_pilot_count,
            "ti_entries_unchanged": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_dir / "parameter_source_audit.csv", audit)
    _atomic_write_csv(output_dir / "missing_or_unsafe_parameters.csv", missing)
    _atomic_write_text(output_dir / "audit_manifest.json", json.dumps(audit_manifest, indent=2, ensure_ascii=True) + "\n")
    _atomic_write_yaml(snapshot_path, new_payload)
    _atomic_write_text(output_dir / "manifest.json", json.dumps(freeze_manifest, indent=2, ensure_ascii=True) + "\n")
    ti_migration = migrate_ti_snapshot_references(
        cfg.output_root / "pilots" / "ti",
        old_snapshot_id,
        new_snapshot_id,
    )
    return {
        "old_snapshot_id": old_snapshot_id,
        "new_snapshot_id": new_snapshot_id,
        "old_entry_count": len(old_entries),
        "new_entry_count": len(entries),
        "matbench_entry_count": 165,
        "ti_total_entry_count": new_ti_total_count,
        "ti_pilot_entry_count": new_ti_pilot_count,
        "ti_migration": ti_migration,
    }
