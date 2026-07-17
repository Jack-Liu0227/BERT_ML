from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from .schemas import CalibrationConfig


CLEANUP_MANIFEST_NAME = "pre_rebuild_cleanup_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_reparse_points(root: Path) -> None:
    stack = [root]
    while stack:
        current = stack.pop()
        if _is_reparse_point(current):
            raise ValueError(f"Cleanup tree contains a reparse point: {current}")
        if not current.is_dir():
            continue
        with os.scandir(current) as entries:
            for entry in entries:
                child = Path(entry.path)
                if _is_reparse_point(child):
                    raise ValueError(f"Cleanup tree contains a reparse point: {child}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(child)


def _freeze_hashes(freeze_root: Path) -> dict[str, str]:
    _assert_no_reparse_points(freeze_root)
    hashes: dict[str, str] = {}
    for path in sorted(freeze_root.rglob("*")):
        if not path.is_file() or path.name == CLEANUP_MANIFEST_NAME:
            continue
        if _is_reparse_point(path):
            raise ValueError(f"parameter_freeze contains a reparse point: {path}")
        hashes[path.relative_to(freeze_root).as_posix()] = _file_sha256(path)
    return hashes


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reset_calibration_output(
    cfg: CalibrationConfig,
    *,
    execute: bool = False,
) -> dict[str, Any]:
    root = cfg.output_root.resolve()
    expected_root = (cfg.project_root / "output" / "calibration").resolve()
    if root != expected_root:
        raise ValueError(
            "Refusing to reset a non-canonical calibration output root: "
            f"actual={root}, expected={expected_root}"
        )
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Calibration output root does not exist: {root}")
    if _is_reparse_point(root):
        raise ValueError(f"Calibration output root is a reparse point: {root}")

    freeze_root = root / "parameter_freeze"
    if not freeze_root.exists() or not freeze_root.is_dir():
        raise FileNotFoundError(f"Protected parameter_freeze directory is missing: {freeze_root}")
    if _is_reparse_point(freeze_root):
        raise ValueError(f"Protected parameter_freeze is a reparse point: {freeze_root}")

    before_hashes = _freeze_hashes(freeze_root)
    candidates = sorted(
        (path for path in root.iterdir() if path.name != "parameter_freeze"),
        key=lambda path: path.name.lower(),
    )
    for candidate in candidates:
        if candidate.parent.resolve() != root:
            raise ValueError(f"Cleanup candidate is not a direct child of calibration root: {candidate}")
        if _is_reparse_point(candidate):
            raise ValueError(f"Cleanup candidate is a reparse point: {candidate}")
        _assert_no_reparse_points(candidate)

    manifest_path = freeze_root / CLEANUP_MANIFEST_NAME
    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "reset-output",
        "generated_at": _utc_now(),
        "calibration_root": str(root),
        "protected_root": str(freeze_root),
        "execute": execute,
        "status": "planned",
        "planned_count": len(candidates),
        "planned_paths": [str(path) for path in candidates],
        "parameter_freeze_hashes_before": before_hashes,
        "parameter_freeze_hashes_unchanged": True,
    }
    _atomic_json_write(manifest_path, payload)

    deleted_count = 0
    if execute:
        for candidate in candidates:
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            deleted_count += 1
        after_hashes = _freeze_hashes(freeze_root)
        unchanged = after_hashes == before_hashes
        payload.update(
            {
                "completed_at": _utc_now(),
                "status": "complete" if unchanged else "failed",
                "deleted_count": deleted_count,
                "parameter_freeze_hashes_after": after_hashes,
                "parameter_freeze_hashes_unchanged": unchanged,
            }
        )
        _atomic_json_write(manifest_path, payload)
        if not unchanged:
            raise ValueError("parameter_freeze hashes changed during calibration reset")
    else:
        payload["deleted_count"] = 0
        _atomic_json_write(manifest_path, payload)

    return {
        "executed": execute,
        "planned_count": len(candidates),
        "deleted_count": deleted_count,
        "manifest_path": str(manifest_path),
        "parameter_freeze_hashes_unchanged": True,
    }
