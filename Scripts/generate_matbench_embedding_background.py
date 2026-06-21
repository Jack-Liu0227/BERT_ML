from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Scripts"))

from Scripts import export_three_space_w_ood_report as three_space

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SPLIT_ROOT = (
    Path("output")
    / "ood_splits"
    / "MatbenchSteels"
    / "matbench_steels_ood"
    / "yield strength"
)
DEFAULT_OUTPUT_FILE = (
    three_space.DEFAULT_EMBEDDING_DATA_DIR
    / "Matbench_Steel__matbench_steels_ood__yieldstrength__all_panel_test_points.csv"
)
TARGET_COLUMN = "yield strength"
REQUIRED_PROJECTION_COLUMNS = ["__source_index__", "projection_x", "projection_y"]
SAMPLE_FILE_NAMES = (
    "train.csv",
    "test_hybrid_combined.csv",
    "test.csv",
    "test_extrapolation_high20.csv",
    "test_inner_ood.csv",
)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def collect_projection_files(split_root: Path) -> list[Path]:
    return sorted(split_root.rglob("trace/projection_2d.csv"))


def validate_projection_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PROJECTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing projection columns: {missing}")
    projection = frame.copy()
    projection["__source_index__"] = pd.to_numeric(projection["__source_index__"], errors="coerce")
    projection["projection_x"] = pd.to_numeric(projection["projection_x"], errors="coerce")
    projection["projection_y"] = pd.to_numeric(projection["projection_y"], errors="coerce")
    projection = projection.dropna(subset=["__source_index__", "projection_x", "projection_y"]).copy()
    projection["__source_index__"] = projection["__source_index__"].astype(int)
    if projection.empty:
        raise ValueError(f"{path} has no finite projection rows")
    return projection


def load_stable_projection(split_root: Path) -> tuple[pd.DataFrame, int]:
    projection_files = collect_projection_files(split_root)
    if not projection_files:
        raise FileNotFoundError(f"No trace/projection_2d.csv files found under {split_root}")

    frames: list[pd.DataFrame] = []
    for path in projection_files:
        projection = validate_projection_frame(read_csv(path), path)
        projection["__projection_file__"] = str(path)
        frames.append(projection)

    all_projection = pd.concat(frames, ignore_index=True, sort=False)
    rounded = all_projection.assign(
        __x_round__=all_projection["projection_x"].round(8),
        __y_round__=all_projection["projection_y"].round(8),
    )
    stability = rounded.groupby("__source_index__", dropna=False).agg(
        x_values=("__x_round__", "nunique"),
        y_values=("__y_round__", "nunique"),
    )
    inconsistent = stability[stability["x_values"].gt(1) | stability["y_values"].gt(1)]
    if not inconsistent.empty:
        examples = ", ".join(str(int(value)) for value in inconsistent.index[:10])
        raise ValueError(f"Inconsistent projection coordinates for __source_index__: {examples}")

    projection = (
        all_projection.sort_values(["__source_index__", "__projection_file__"])
        .drop_duplicates("__source_index__", keep="first")
        .drop(columns=["__projection_file__"], errors="ignore")
        .sort_values("__source_index__")
        .reset_index(drop=True)
    )
    return projection, len(projection_files)


def iter_sample_files(split_root: Path) -> Iterable[Path]:
    for name in SAMPLE_FILE_NAMES:
        yield from sorted(split_root.rglob(name))


def load_sample_rows(split_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in iter_sample_files(split_root):
        frame = read_csv(path)
        if "ID" not in frame.columns:
            continue
        frame = frame.copy()
        frame["__sample_file__"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No split sample CSVs with ID were found under {split_root}")

    samples = pd.concat(frames, ignore_index=True, sort=False)
    samples["ID"] = pd.to_numeric(samples["ID"], errors="coerce")
    samples = samples.dropna(subset=["ID"]).copy()
    samples["ID"] = samples["ID"].astype(int)
    samples["__source_index__"] = samples["ID"] - 1

    value_columns = [column for column in samples.columns if column != "__sample_file__"]
    samples = (
        samples.sort_values(["ID", "__sample_file__"])
        .drop_duplicates("ID", keep="first")[value_columns]
        .sort_values("ID")
        .reset_index(drop=True)
    )
    return samples


def build_embedding_background(split_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    split_root = Path(split_root)
    projection, projection_file_count = load_stable_projection(split_root)
    samples = load_sample_rows(split_root)

    projection_keys = set(projection["__source_index__"].astype(int).tolist())
    sample_keys = set(samples["__source_index__"].astype(int).tolist())
    missing_ids = sorted(projection_keys - sample_keys)
    if missing_ids:
        examples = ", ".join(str(int(value + 1)) for value in missing_ids[:10])
        raise ValueError(f"Missing ID mapping for projection rows. Example IDs: {examples}")

    projection_cols = [
        column
        for column in [
            "__source_index__",
            "__row_id__",
            TARGET_COLUMN,
            "projection_x",
            "projection_y",
            "x_density",
            "y_density",
            "selection_density",
            "selection_space",
        ]
        if column in projection.columns
    ]
    background = samples.merge(
        projection[projection_cols],
        on="__source_index__",
        how="inner",
        suffixes=("", "_projection"),
        sort=False,
    )
    if TARGET_COLUMN in background.columns and f"{TARGET_COLUMN}_projection" in background.columns:
        background = background.drop(columns=[f"{TARGET_COLUMN}_projection"])
    elif f"{TARGET_COLUMN}_projection" in background.columns:
        background = background.rename(columns={f"{TARGET_COLUMN}_projection": TARGET_COLUMN})

    background["x"] = pd.to_numeric(background["projection_x"], errors="coerce")
    background["y"] = pd.to_numeric(background["projection_y"], errors="coerce")
    background = background.dropna(subset=["ID", "__source_index__", "x", "y"]).copy()
    background["ID"] = background["ID"].astype(int)
    background["__source_index__"] = background["__source_index__"].astype(int)

    front_columns = [
        column
        for column in [
            "ID",
            "__row_id__",
            "__source_index__",
            TARGET_COLUMN,
            "projection_x",
            "projection_y",
            "x",
            "y",
            "x_density",
            "y_density",
            "selection_density",
            "selection_space",
            "composition",
        ]
        if column in background.columns
    ]
    remaining_columns = [column for column in background.columns if column not in front_columns]
    background = background[front_columns + remaining_columns].sort_values("ID").reset_index(drop=True)

    diagnostics = {
        "split_root": str(split_root),
        "projection_files": projection_file_count,
        "projection_rows": int(len(projection)),
        "sample_ids": int(samples["ID"].nunique()),
        "output_rows": int(len(background)),
        "id_min": int(background["ID"].min()) if not background.empty else None,
        "id_max": int(background["ID"].max()) if not background.empty else None,
        "finite_x": int(np.isfinite(pd.to_numeric(background["x"], errors="coerce")).sum()),
        "finite_y": int(np.isfinite(pd.to_numeric(background["y"], errors="coerce")).sum()),
    }
    return background, diagnostics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the Matbench Steel-YS embedding background CSV used by Z-space W calculations."
    )
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--diagnostics-output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    background, diagnostics = build_embedding_background(Path(args.split_root))
    output = Path(args.output)
    write_csv(background, output)
    diagnostics_output = Path(args.diagnostics_output) if args.diagnostics_output else output.with_suffix(".diagnostics.json")
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(background)} rows to {output}")
    print(f"Wrote diagnostics to {diagnostics_output}")


if __name__ == "__main__":
    main()
