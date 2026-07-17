from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat() if stat else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    stage: str,
    config_path: Path,
    output_root: Path,
    payload: dict[str, Any] | None = None,
    source_files: list[Path] | None = None,
) -> None:
    body: dict[str, Any] = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": file_record(config_path),
        "output_root": str(output_root),
        "source_files": [file_record(source) for source in source_files or []],
    }
    if payload:
        body.update(payload)
    write_json(path, body)
