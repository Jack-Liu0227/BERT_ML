from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from .schemas import CalibrationConfig


ALLOY_FAMILY_ALIASES = {
    "matbench steel": "Matbench Steel",
    "matbenchsteels": "Matbench Steel",
    "matbench_steel": "Matbench Steel",
}


@dataclass(frozen=True)
class ParameterCandidate:
    task_id: str
    protocol: str
    split_id: str
    model_family: str
    model: str
    source_root: str
    model_dir: Path
    reporting_selection_mode: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.task_id, self.protocol, self.split_id, self.model_family, self.model)


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def normalize_alloy_family(value: object) -> str:
    text = str(value or "unknown").strip()
    return ALLOY_FAMILY_ALIASES.get(text.lower(), text)


def build_task_id(row: pd.Series) -> str:
    return "__".join(
        [
            _safe_component(normalize_alloy_family(row.get("alloy_family", "unknown"))),
            _safe_component(row.get("dataset_name", "unknown")),
            _safe_component(row.get("property", "unknown")),
        ]
    )


def normalize_protocol(value: object, aliases: dict[str, str] | None = None) -> str:
    text = str(value or "Unknown").strip()
    lookup = {str(key).lower(): str(result) for key, result in (aliases or {}).items()}
    return lookup.get(text.lower(), text)


def _parse_json_list(value: object) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _resolve_model_dir(cfg: CalibrationConfig, value: object) -> Path | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else cfg.project_root / path


def _split_id_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"fold_(\d+)", part, flags=re.IGNORECASE)
        if match:
            return f"fold_{int(match.group(1))}"
    return "main"


def _source_root_name(cfg: CalibrationConfig, path: Path) -> str:
    roots = cfg.raw.get("source_roots", []) or []
    resolved = path.resolve()
    for root_value in roots:
        root = cfg.resolve(root_value).resolve()
        try:
            resolved.relative_to(root)
            return str(root_value)
        except ValueError:
            continue
    return "unregistered"


def _details_model_dirs(cfg: CalibrationConfig, row: pd.Series) -> list[Path]:
    directories: list[Path] = []
    for column in ("loco_outer_fold_best_details_json", "tabpfn_loco_fold_details_json"):
        for item in _parse_json_list(row.get(column)):
            path = _resolve_model_dir(cfg, item.get("model_dir"))
            if path is not None:
                directories.append(path)
    return directories


def _expand_fold_siblings(path: Path) -> list[Path]:
    parts = list(path.parts)
    try:
        folds_index = next(index for index, part in enumerate(parts) if part.lower() == "folds")
    except StopIteration:
        return [path]
    if folds_index + 1 >= len(parts) or not re.fullmatch(r"fold_\d+", parts[folds_index + 1], re.IGNORECASE):
        return [path]

    folds_root = Path(*parts[: folds_index + 1])
    suffix = parts[folds_index + 2 :]
    if not folds_root.exists():
        return [path]
    siblings = sorted(
        (item for item in folds_root.iterdir() if item.is_dir() and re.fullmatch(r"fold_\d+", item.name, re.IGNORECASE)),
        key=lambda item: int(item.name.split("_")[-1]),
    )
    expanded = [item.joinpath(*suffix) for item in siblings]
    return expanded or [path]


def _candidate_dirs(cfg: CalibrationConfig, row: pd.Series) -> list[Path]:
    directories = _details_model_dirs(cfg, row)
    model_dir = _resolve_model_dir(cfg, row.get("model_dir"))
    if model_dir is not None and not directories:
        directories.append(model_dir)

    expanded: list[Path] = []
    for directory in directories:
        expanded.extend(_expand_fold_siblings(directory))
    unique: dict[str, Path] = {}
    for directory in expanded:
        unique[str(directory.resolve()).lower()] = directory
    return list(unique.values())


def _direct_tabpfn_candidates(cfg: CalibrationConfig) -> list[ParameterCandidate]:
    aliases = cfg.raw.get("protocol_aliases", {}) or {}
    candidates: list[ParameterCandidate] = []
    for root_value in cfg.raw.get("source_roots", []) or []:
        if "tabpfn" not in str(root_value).lower():
            continue
        root = cfg.resolve(root_value)
        if not root.exists():
            continue
        for manifest_path in root.rglob("pipeline_manifest.json"):
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or "model_info" not in payload:
                continue
            relative_parts = manifest_path.relative_to(root).parts
            dataset_name = relative_parts[2] if len(relative_parts) > 2 else Path(str(payload.get("raw_data", "dataset"))).stem
            row = pd.Series(
                {
                    "alloy_family": payload.get("alloy_type", "unknown"),
                    "dataset_name": dataset_name,
                    "property": payload.get("target_col", "unknown"),
                }
            )
            model_info = payload.get("model_info", {}) if isinstance(payload.get("model_info"), dict) else {}
            model = str(model_info.get("model_run_dirname") or payload.get("model_name") or root.name)
            candidates.append(
                ParameterCandidate(
                    task_id=build_task_id(row),
                    protocol=normalize_protocol(payload.get("split_strategy", relative_parts[0]), aliases),
                    split_id=_split_id_from_path(manifest_path.parent),
                    model_family="TabPFN",
                    model=model,
                    source_root=str(root_value),
                    model_dir=manifest_path.parent.resolve(),
                    reporting_selection_mode="fixed_runtime",
                )
            )
    return candidates


def discover_parameter_candidates(cfg: CalibrationConfig) -> list[ParameterCandidate]:
    summary_value = cfg.raw.get(
        "combined_summary",
        "output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv",
    )
    summary_path = cfg.resolve(summary_value)
    if not summary_path.exists():
        raise FileNotFoundError(f"Combined OOD summary not found: {summary_path}")

    frame = pd.read_csv(summary_path)
    aliases = cfg.raw.get("protocol_aliases", {}) or {}
    candidates: dict[tuple[str, str, str, str, str], ParameterCandidate] = {}
    for _, row in frame.iterrows():
        model = str(row.get("model", "Unknown"))
        model_family = str(row.get("model_family", "Unknown"))
        if "gpt" in model.lower() or model_family.lower() in {"llm", "gpt"}:
            continue
        task_id = build_task_id(row)
        protocol = normalize_protocol(row.get("ood_method"), aliases)
        selection_mode = str(row.get("representative_selection_mode", "") or "")
        for model_dir in _candidate_dirs(cfg, row):
            candidate = ParameterCandidate(
                task_id=task_id,
                protocol=protocol,
                split_id=_split_id_from_path(model_dir),
                model_family=model_family,
                model=model,
                source_root=_source_root_name(cfg, model_dir),
                model_dir=model_dir.resolve(),
                reporting_selection_mode=selection_mode,
            )
            candidates[candidate.key] = candidate
    for candidate in _direct_tabpfn_candidates(cfg):
        candidates[candidate.key] = candidate
    return sorted(candidates.values(), key=lambda item: item.key)


def iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.exists():
            yield path
