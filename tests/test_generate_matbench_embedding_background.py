from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Scripts" / "generate_matbench_embedding_background.py"


def load_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("generate_matbench_embedding_background", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_split(
    split_root: Path,
    rel: str,
    projection: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    split_data = split_root / rel / "split_data"
    trace_dir = split_data / "trace"
    trace_dir.mkdir(parents=True)
    projection.to_csv(trace_dir / "projection_2d.csv", index=False)
    train.to_csv(split_data / "train.csv", index=False)
    test.to_csv(split_data / "test_hybrid_combined.csv", index=False)


def test_build_embedding_background_uses_stable_projection_and_id_mapping(tmp_path: Path) -> None:
    mod = load_module()
    projection = pd.DataFrame(
        {
            "__row_id__": [0, 1, 2],
            "__source_index__": [0, 1, 2],
            "yield strength": [100.0, 200.0, 300.0],
            "projection_x": [1.0, 2.0, 3.0],
            "projection_y": [4.0, 5.0, 6.0],
            "x_density": [0.1, 0.2, 0.3],
        }
    )
    train = pd.DataFrame(
        {
            "ID": [1, 2],
            "composition": ["Fe1", "Fe2"],
            "Fe(at%)": [99.0, 98.0],
            "yield strength": [100.0, 200.0],
        }
    )
    test = pd.DataFrame(
        {
            "ID": [3],
            "composition": ["Fe3"],
            "Fe(at%)": [97.0],
            "yield strength": [300.0],
        }
    )
    write_split(tmp_path, "hybrid_extrapolation_loco/run/folds/fold_0", projection, train, test)
    write_split(tmp_path, "hybrid_extrapolation_random_cv/run/folds/fold_0", projection, train, test)

    background, diagnostics = mod.build_embedding_background(tmp_path)

    assert diagnostics["projection_files"] == 2
    assert diagnostics["output_rows"] == 3
    assert background["ID"].tolist() == [1, 2, 3]
    assert background["__source_index__"].tolist() == [0, 1, 2]
    assert background["x"].tolist() == [1.0, 2.0, 3.0]
    assert background["y"].tolist() == [4.0, 5.0, 6.0]
    assert background["x"].equals(background["projection_x"])
    assert background["y"].equals(background["projection_y"])
    assert "composition" in background.columns


def test_build_embedding_background_rejects_inconsistent_projection_coordinates(tmp_path: Path) -> None:
    mod = load_module()
    projection_a = pd.DataFrame(
        {
            "__source_index__": [0, 1],
            "projection_x": [1.0, 2.0],
            "projection_y": [3.0, 4.0],
        }
    )
    projection_b = pd.DataFrame(
        {
            "__source_index__": [0, 1],
            "projection_x": [1.0, 20.0],
            "projection_y": [3.0, 4.0],
        }
    )
    rows = pd.DataFrame({"ID": [1, 2], "yield strength": [10.0, 20.0]})
    write_split(tmp_path, "a", projection_a, rows.iloc[[0]], rows.iloc[[1]])
    write_split(tmp_path, "b", projection_b, rows.iloc[[0]], rows.iloc[[1]])

    with pytest.raises(ValueError, match="Inconsistent projection coordinates"):
        mod.build_embedding_background(tmp_path)


def test_build_embedding_background_requires_id_coverage(tmp_path: Path) -> None:
    mod = load_module()
    projection = pd.DataFrame(
        {
            "__source_index__": [0, 1, 2],
            "projection_x": [1.0, 2.0, 3.0],
            "projection_y": [4.0, 5.0, 6.0],
        }
    )
    train = pd.DataFrame({"ID": [1], "yield strength": [10.0]})
    test = pd.DataFrame({"ID": [2], "yield strength": [20.0]})
    write_split(tmp_path, "a", projection, train, test)

    with pytest.raises(ValueError, match="Missing ID mapping"):
        mod.build_embedding_background(tmp_path)
