from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import StageResult, coalesce_column, ensure_dirs
from .config import FigureConfig, resolve_path
from .manifest import write_manifest


DISCOVERY_COLUMNS = [
    "source_name",
    "source_type",
    "path",
    "exists",
    "usable_for",
    "missing_reason",
    "task_id",
    "alloy_family",
    "dataset_name",
    "property",
    "protocol",
    "failure_axis",
    "model",
    "model_family",
    "confidence_source",
]


def _roots(cfg: FigureConfig) -> dict[str, Any]:
    return cfg.raw.get("discovery_roots", {}) or {}


def _protocol_to_axis(protocol: object) -> str:
    value = str(protocol or "").lower()
    if "extrapolation" in value or value in {"extra", "extra."}:
        return "R_y_extreme"
    if "loco" in value:
        return "R_z_extreme"
    return "other"


def _normalize_protocol(protocol: object) -> str:
    value = str(protocol or "")
    lowered = value.lower()
    if "extrapolation" in lowered:
        return "Extrapolation"
    if "random" in lowered:
        return "RandomCV"
    if "loco" in lowered:
        return "LOCO"
    return value


def _task_metadata(row: pd.Series) -> dict[str, Any]:
    task_id = row.get("task_id")
    if task_id is None or pd.isna(task_id):
        task_id = row.get("task_key")
    parts = str(task_id).split("__") if task_id is not None and not pd.isna(task_id) else []
    return {
        "task_id": task_id,
        "alloy_family": row.get("alloy_family") if "alloy_family" in row else (parts[0] if len(parts) > 0 else pd.NA),
        "dataset_name": row.get("dataset_name") if "dataset_name" in row else (parts[1] if len(parts) > 1 else pd.NA),
        "property": row.get("property") if "property" in row else (parts[2] if len(parts) > 2 else pd.NA),
    }


def _gpt_rows_from_inventory(inventory_path: Path) -> list[dict[str, Any]]:
    if not inventory_path.exists():
        return [
            {
                "source_name": "gpt_semantic",
                "source_type": "prediction_inventory",
                "path": str(inventory_path),
                "exists": False,
                "usable_for": "figure3",
                "missing_reason": "prediction_source_inventory.csv not found",
            }
        ]
    inventory = pd.read_csv(inventory_path)
    rows: list[dict[str, Any]] = []
    for _, row in inventory[inventory.get("model", pd.Series(dtype=object)).astype(str).str.contains("GPT-5.4", case=False, na=False)].iterrows():
        prediction_path = resolve_path(row.get("prediction_file", ""))
        task_metadata = _task_metadata(row)
        missing_reason = ""
        if not prediction_path.exists():
            missing_reason = "prediction file not found"
        else:
            try:
                cols = pd.read_csv(prediction_path, nrows=0).columns
                if "confidence" not in cols and "confidence_tier" not in cols and "semantic_confidence" not in cols:
                    missing_reason = "confidence column not found"
            except Exception as exc:  # pragma: no cover - defensive provenance reporting
                missing_reason = f"could not inspect prediction file: {exc}"
        rows.append(
            {
                "source_name": "gpt_semantic",
                "source_type": "prediction_file",
                "path": str(prediction_path),
                "exists": prediction_path.exists(),
                "usable_for": "figure3",
                "missing_reason": missing_reason,
                "task_id": task_metadata["task_id"],
                "alloy_family": task_metadata["alloy_family"],
                "dataset_name": task_metadata["dataset_name"],
                "property": task_metadata["property"],
                "protocol": _normalize_protocol(row.get("method")),
                "failure_axis": _protocol_to_axis(row.get("method")),
                "model": "GPT-5.4",
                "model_family": "GPT",
                "confidence_source": "semantic",
            }
        )
    return rows


def _combined_summary_row(combined_path: Path) -> dict[str, Any]:
    return {
        "source_name": "combined_summary",
        "source_type": "combined_summary",
        "path": str(combined_path),
        "exists": combined_path.exists(),
        "usable_for": "figure4_full_size",
        "missing_reason": "" if combined_path.exists() else "combined summary not found",
    }


def _uncertainty_rows(root_path: Path) -> list[dict[str, Any]]:
    if not root_path.exists():
        return [
            {
                "source_name": "non_gpt_uncertainty",
                "source_type": "uncertainty_root",
                "path": str(root_path),
                "exists": False,
                "usable_for": "figure3",
                "missing_reason": "uncertainty root not found",
            }
        ]
    rows: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for pattern in ["*uncert*.csv", "*sigma*.csv", "*std*.csv", "*predictive*.csv"]:
        candidate_paths.extend(root_path.rglob(pattern))
    seen: set[Path] = set()
    unique_candidates = []
    for path in candidate_paths:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)
    for csv_path in unique_candidates[:200]:
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        if any(col in cols for col in ["uncertainty_score", "sigma", "std", "predictive_std", "confidence_tier"]):
            rows.append(
                {
                    "source_name": "non_gpt_uncertainty",
                    "source_type": "uncertainty_file",
                    "path": str(csv_path),
                    "exists": True,
                    "usable_for": "figure3",
                    "missing_reason": "",
                }
            )
    if not rows:
        rows.append(
            {
                "source_name": "non_gpt_uncertainty",
                "source_type": "uncertainty_file",
                "path": str(root_path),
                "exists": True,
                "usable_for": "figure3",
                "missing_reason": "no CSV with uncertainty_score/sigma/std/predictive_std/confidence_tier found",
            }
        )
    return rows


def _uncertainty_scan_disabled_row(root_path: Path) -> dict[str, Any]:
    return {
        "source_name": "non_gpt_uncertainty",
        "source_type": "uncertainty_root",
        "path": str(root_path),
        "exists": root_path.exists(),
        "usable_for": "figure3",
        "missing_reason": "uncertainty scan disabled; set discovery_roots.enable_uncertainty_scan=true to search this root",
    }


def run_discovery(cfg: FigureConfig) -> StageResult:
    audit_dir = cfg.output_root / "audit"
    ensure_dirs(audit_dir)
    roots = _roots(cfg)
    inventory_path = resolve_path(roots.get("prediction_inventory", "output/w_error_relationship_full/OOD/csv/prediction_source_inventory.csv"))
    combined_path = resolve_path(roots.get("combined_summary", "output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv"))
    uncertainty_root = resolve_path(roots.get("uncertainty_root", "output/new_results_withuncertainty"))

    rows = []
    rows.extend(_gpt_rows_from_inventory(inventory_path))
    rows.append(_combined_summary_row(combined_path))
    if bool(roots.get("enable_uncertainty_scan", False)):
        rows.extend(_uncertainty_rows(uncertainty_root))
    else:
        rows.append(_uncertainty_scan_disabled_row(uncertainty_root))
    discovered = pd.DataFrame(rows)
    for column in DISCOVERY_COLUMNS:
        if column not in discovered.columns:
            discovered[column] = pd.NA
    discovered = discovered[DISCOVERY_COLUMNS]
    discovered.to_csv(audit_dir / "discovered_sources.csv", index=False)
    write_manifest(
        audit_dir / "discovery_manifest.json",
        stage="discover",
        config_path=cfg.config_path,
        output_root=cfg.output_root,
        payload={"row_count": int(len(discovered)), "usable_count": int(discovered["missing_reason"].fillna("").eq("").sum())},
        source_files=[path for path in [inventory_path, combined_path] if path.exists()],
    )
    missing_count = int(discovered["missing_reason"].fillna("").ne("").sum())
    return StageResult(stage="discover", output_dir=audit_dir, row_count=len(discovered), missing_count=missing_count)
