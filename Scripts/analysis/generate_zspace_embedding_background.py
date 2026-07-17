from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.analysis.analyze_top_tail_loco_proximity import DEFAULT_OUTPUT_PARENT, read_csv, write_csv  # noqa: E402


DEFAULT_STEEL_YS_SPLIT_ROOT = (
    Path("output")
    / "ood_splits"
    / "Steel"
    / "steel"
    / "YS(MPa)"
    / "hybrid_extrapolation_loco"
)
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_PARENT
    / "wx_wz_tail_bin_distribution"
    / "zspace_embedding_data"
    / "Steel__steel__YSMPa__all_panel_test_points.csv"
)
SAMPLE_FILE_NAMES = (
    "train.csv",
    "test.csv",
    "test_hybrid_combined.csv",
    "test_extrapolation_high20.csv",
    "test_inner_ood.csv",
)
REQUIRED_PROJECTION_COLUMNS = ["__source_index__", "projection_x", "projection_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable Z-space embedding background from OOD split projections.")
    parser.add_argument("--split-root", default=str(DEFAULT_STEEL_YS_SPLIT_ROOT))
    parser.add_argument("--target-column", default="YS(MPa)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--diagnostics-output", default=None)
    return parser.parse_args()


def collect_projection_files(split_root: Path) -> list[Path]:
    return sorted(split_root.rglob("trace/projection_2d.csv"))


def validate_projection_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PROJECTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing projection columns: {missing}")
    projection = frame.copy()
    for column in REQUIRED_PROJECTION_COLUMNS:
        projection[column] = pd.to_numeric(projection[column], errors="coerce")
    projection = projection.dropna(subset=REQUIRED_PROJECTION_COLUMNS).copy()
    if projection.empty:
        raise ValueError(f"{path} has no finite projection rows")
    projection["__source_index__"] = projection["__source_index__"].astype(int)
    return projection


def load_stable_projection(split_root: Path) -> tuple[pd.DataFrame, int]:
    files = collect_projection_files(split_root)
    if not files:
        raise FileNotFoundError(f"No trace/projection_2d.csv files found under {split_root}")
    frames: list[pd.DataFrame] = []
    for path in files:
        projection = validate_projection_frame(read_csv(path), path)
        projection["__projection_file__"] = str(path)
        frames.append(projection)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    rounded = combined.assign(
        __x_round__=combined["projection_x"].round(8),
        __y_round__=combined["projection_y"].round(8),
    )
    stability = rounded.groupby("__source_index__", dropna=False).agg(
        x_values=("__x_round__", "nunique"),
        y_values=("__y_round__", "nunique"),
    )
    inconsistent = stability[stability["x_values"].gt(1) | stability["y_values"].gt(1)]
    if not inconsistent.empty:
        examples = ", ".join(str(int(value)) for value in inconsistent.index[:10])
        raise ValueError(f"Inconsistent projection coordinates for __source_index__: {examples}")
    stable = (
        combined.sort_values(["__source_index__", "__projection_file__"], kind="stable")
        .drop_duplicates("__source_index__", keep="first")
        .drop(columns="__projection_file__")
        .sort_values("__source_index__", kind="stable")
        .reset_index(drop=True)
    )
    return stable, len(files)


def iter_sample_files(split_root: Path) -> Iterable[Path]:
    for filename in SAMPLE_FILE_NAMES:
        yield from sorted(split_root.rglob(filename))


def load_sample_rows(split_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in iter_sample_files(split_root):
        frame = read_csv(path)
        if "ID" not in frame.columns:
            continue
        current = frame.copy()
        current["__sample_file__"] = str(path)
        current["ID"] = pd.to_numeric(current["ID"], errors="coerce")
        current = current.dropna(subset=["ID"]).copy()
        current["ID"] = current["ID"].astype(int)
        if "__source_index__" in current.columns:
            source_index = pd.to_numeric(current["__source_index__"], errors="coerce")
        else:
            source_index = pd.Series(np.nan, index=current.index, dtype=float)
        current["__source_index__"] = source_index.fillna(current["ID"] - 1).astype(int)
        frames.append(current)
    if not frames:
        raise FileNotFoundError(f"No split sample CSVs with ID were found under {split_root}")
    samples = pd.concat(frames, ignore_index=True, sort=False)
    mapping_counts = samples.groupby("ID", dropna=False)["__source_index__"].nunique()
    inconsistent = mapping_counts[mapping_counts.gt(1)]
    if not inconsistent.empty:
        examples = ", ".join(str(int(value)) for value in inconsistent.index[:10])
        raise ValueError(f"Inconsistent ID to __source_index__ mapping: {examples}")
    return (
        samples.sort_values(["ID", "__sample_file__"], kind="stable")
        .drop_duplicates("ID", keep="first")
        .drop(columns="__sample_file__")
        .sort_values("ID", kind="stable")
        .reset_index(drop=True)
    )


def build_embedding_background(split_root: Path, target_col: str) -> tuple[pd.DataFrame, dict[str, object]]:
    projection, projection_file_count = load_stable_projection(Path(split_root))
    samples = load_sample_rows(Path(split_root))
    projection_keys = set(projection["__source_index__"].astype(int))
    sample_keys = set(samples["__source_index__"].astype(int))
    missing_keys = sorted(projection_keys - sample_keys)
    if missing_keys:
        examples = ", ".join(str(value) for value in missing_keys[:10])
        raise ValueError(f"Missing sample mapping for projection source indices: {examples}")
    projection_columns = [
        column
        for column in ["__source_index__", "__row_id__", target_col, "projection_x", "projection_y"]
        if column in projection.columns
    ]
    background = samples.merge(
        projection[projection_columns],
        on="__source_index__",
        how="inner",
        suffixes=("", "_projection"),
        sort=False,
    )
    if f"{target_col}_projection" in background.columns:
        background = background.drop(columns=f"{target_col}_projection")
    background["x"] = pd.to_numeric(background["projection_x"], errors="coerce")
    background["y"] = pd.to_numeric(background["projection_y"], errors="coerce")
    background = background.dropna(subset=["ID", "__source_index__", "x", "y"]).copy()
    background["ID"] = background["ID"].astype(int)
    background["__source_index__"] = background["__source_index__"].astype(int)
    front_columns = [
        column
        for column in ["ID", "__row_id__", "__source_index__", target_col, "projection_x", "projection_y", "x", "y"]
        if column in background.columns
    ]
    remaining = [column for column in background.columns if column not in front_columns]
    background = background[front_columns + remaining].sort_values("ID", kind="stable").reset_index(drop=True)
    diagnostics = {
        "split_root": str(split_root),
        "target_col": target_col,
        "projection_files": projection_file_count,
        "projection_rows": int(len(projection)),
        "sample_ids": int(samples["ID"].nunique()),
        "output_rows": int(len(background)),
        "id_min": int(background["ID"].min()) if not background.empty else None,
        "id_max": int(background["ID"].max()) if not background.empty else None,
        "finite_x": int(np.isfinite(background["x"]).sum()),
        "finite_y": int(np.isfinite(background["y"]).sum()),
    }
    return background, diagnostics


def main() -> None:
    args = parse_args()
    background, diagnostics = build_embedding_background(Path(args.split_root), str(args.target_column))
    output = Path(args.output)
    write_csv(background, output)
    diagnostics_path = Path(args.diagnostics_output) if args.diagnostics_output else output.with_suffix(".diagnostics.json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(background)} rows to {output}")
    print(f"Wrote diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
