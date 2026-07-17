from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import StageResult, ensure_dirs, read_source_csv, source_path
from .config import FigureConfig, list_section_sources
from .manifest import write_manifest


OUTPUT_COLUMNS = [
    "task_id",
    "frozen_parameter_task_id",
    "model_family",
    "model",
    "failure_axis",
    "training_fraction",
    "repeat_id",
    "random_cv_mae",
    "ood_mae",
    "degradation_ratio",
    "source_name",
    "source_file",
]

REQUIRED_COLUMNS = [
    "task_id",
    "model_family",
    "model",
    "failure_axis",
    "training_fraction",
    "repeat_id",
    "random_cv_mae",
    "ood_mae",
]


def full_size_degradation_from_combined(cfg: FigureConfig) -> pd.DataFrame:
    """Legacy compatibility hook: old full-size summaries are intentionally ignored."""

    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def load_training_size_results(cfg: FigureConfig) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    sources: list[Path] = []
    for entry in list_section_sources(cfg, "figure4", "degradation_sources"):
        path = source_path(entry)
        if not path.exists():
            continue
        frame = read_source_csv(entry)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Figure 4 sweep source {entry.get('name')} missing columns: {missing}")
        if "degradation_ratio" not in frame.columns:
            frame["degradation_ratio"] = frame["ood_mae"] / frame["random_cv_mae"]
        frames.append(frame)
        sources.append(path)
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), sources
    return pd.concat(frames, ignore_index=True), sources


def write_missing_fraction_report(cfg: FigureConfig, available: pd.DataFrame) -> int:
    audit_dir = cfg.output_root / "audit"
    ensure_dirs(audit_dir)
    path = audit_dir / "missing_required_inputs.csv"
    columns = [
        "section",
        "source_type",
        "source_name",
        "path",
        "required",
        "exists",
        "size_bytes",
        "status",
        "message",
    ]
    expected = [float(value) for value in cfg.raw.get("figure4", {}).get("training_fractions", [1.0])]
    present = set(pd.to_numeric(available.get("training_fraction"), errors="coerce").dropna().tolist())
    rows = [
        {
            "section": "figure4",
            "source_type": "training_size_sweep",
            "source_name": f"figure4_{int(round(fraction * 100))}pct",
            "path": "",
            "required": True,
            "exists": False,
            "size_bytes": 0,
            "status": "missing_training_size_sweep",
            "message": (
                "Strict Figure 4 requires retraining on the fixed split; no legacy full-size "
                "summary is substituted."
            ),
        }
        for fraction in expected
        if fraction not in present
    ]
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    existing = existing[
        ~(
            existing.get("section", pd.Series(dtype=str)).astype(str).eq("figure4")
            & existing.get("source_type", pd.Series(dtype=str)).astype(str).eq("training_size_sweep")
        )
    ]
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(path, index=False)
    return len(rows)


def run_figure4_sweep(cfg: FigureConfig) -> StageResult:
    figure_cfg = dict(cfg.raw.get("figure4") or {})
    root_value = figure_cfg.get("training_output_root", "output/training_size_ood")
    stage_dir = Path(str(root_value))
    if not stage_dir.is_absolute():
        stage_dir = (Path.cwd() / stage_dir).resolve()
    csv_dir = stage_dir / "csv"
    ensure_dirs(csv_dir)
    by_repeat, sources = load_training_size_results(cfg)
    missing_count = write_missing_fraction_report(cfg, by_repeat)
    by_repeat.to_csv(csv_dir / "degradation_ratio_by_repeat.csv", index=False)
    write_manifest(
        stage_dir / "sweep_manifest.json",
        stage="figure4-sweep",
        config_path=cfg.config_path,
        output_root=stage_dir,
        payload={"row_count": int(len(by_repeat)), "missing_training_fraction_count": missing_count},
        source_files=sources,
    )
    return StageResult(
        stage="figure4-sweep",
        output_dir=stage_dir,
        row_count=len(by_repeat),
        missing_count=missing_count,
    )
