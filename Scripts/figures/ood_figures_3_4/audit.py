from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import StageResult, ensure_dirs
from .config import FigureConfig, configured_required_names, list_section_sources, resolve_path
from .manifest import write_json, write_manifest


def _audit_entry(section: str, source_type: str, entry: dict[str, Any], required_names: set[str]) -> dict[str, Any]:
    name = str(entry.get("name") or entry.get("path") or source_type)
    required = bool(entry.get("required", name in required_names))
    path = resolve_path(entry.get("path", ""))
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "section": section,
        "source_type": source_type,
        "source_name": name,
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": size,
        "status": "ok" if exists else ("missing_required" if required else "missing_optional"),
    }


def _discovered_ok_names(cfg: FigureConfig, usable_prefix: str) -> set[str]:
    path = cfg.output_root / "audit" / "discovered_sources.csv"
    if not path.exists():
        return set()
    discovered = pd.read_csv(path)
    if "source_name" not in discovered.columns or "missing_reason" not in discovered.columns:
        return set()
    usable = discovered["usable_for"].astype(str).str.startswith(usable_prefix, na=False)
    ok = discovered["missing_reason"].fillna("").eq("")
    return set(discovered.loc[usable & ok, "source_name"].astype(str))


def run_audit(cfg: FigureConfig) -> StageResult:
    audit_dir = cfg.output_root / "audit"
    ensure_dirs(audit_dir)
    rows: list[dict[str, Any]] = []

    f3_required = configured_required_names(cfg, "figure3")
    f3_sources = list_section_sources(cfg, "figure3", "sample_level_sources")
    rows.extend(_audit_entry("figure3", "sample_level", entry, f3_required) for entry in f3_sources)
    configured_f3 = {str(entry.get("name")) for entry in f3_sources if entry.get("name")}
    configured_f3.update(_discovered_ok_names(cfg, "figure3"))
    for name in sorted(f3_required - configured_f3):
        rows.append(
            {
                "section": "figure3",
                "source_type": "sample_level",
                "source_name": name,
                "path": "",
                "required": True,
                "exists": False,
                "size_bytes": 0,
                "status": "missing_required",
            }
        )

    f4_required = configured_required_names(cfg, "figure4")
    f4_sources = list_section_sources(cfg, "figure4", "degradation_sources")
    rows.extend(_audit_entry("figure4", "degradation", entry, f4_required) for entry in f4_sources)
    configured_f4 = {str(entry.get("name")) for entry in f4_sources if entry.get("name")}
    for name in sorted(f4_required - configured_f4):
        rows.append(
            {
                "section": "figure4",
                "source_type": "degradation",
                "source_name": name,
                "path": "",
                "required": True,
                "exists": False,
                "size_bytes": 0,
                "status": "missing_required",
            }
        )

    report = pd.DataFrame(rows)
    missing = report[report["status"].eq("missing_required")].copy()
    report.to_csv(audit_dir / "data_availability_report.csv", index=False)
    missing.to_csv(audit_dir / "missing_required_inputs.csv", index=False)
    write_json(
        audit_dir / "data_availability_report.json",
        {
            "summary": {
                "source_count": int(len(report)),
                "missing_required_count": int(len(missing)),
            },
            "records": report.to_dict(orient="records"),
        },
    )
    source_files = [Path(path) for path in report.loc[report["exists"], "path"].tolist()]
    write_manifest(
        audit_dir / "manifest.json",
        stage="audit",
        config_path=cfg.config_path,
        output_root=cfg.output_root,
        payload={"row_count": int(len(report)), "missing_required_count": int(len(missing))},
        source_files=source_files,
    )
    return StageResult(stage="audit", output_dir=audit_dir, row_count=len(report), missing_count=len(missing))
