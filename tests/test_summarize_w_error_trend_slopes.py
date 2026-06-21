from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Scripts" / "summarize_w_error_trend_slopes.py"


def load_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("summarize_w_error_trend_slopes", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_slope_uses_only_finite_distinct_points() -> None:
    mod = load_module()

    result = mod.fit_slope(
        pd.Series([0.0, 1.0, 2.0, math.nan, 3.0]),
        pd.Series([1.0, 3.0, 5.0, 7.0, math.inf]),
    )

    assert result["slope_status"] == "ok"
    assert result["n_points"] == 3
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)
    assert result["x_min"] == 0.0
    assert result["x_max"] == 2.0


def test_fit_slope_marks_insufficient_or_constant_w_as_nan() -> None:
    mod = load_module()

    too_few = mod.fit_slope(pd.Series([1.0]), pd.Series([2.0]))
    constant_w = mod.fit_slope(pd.Series([1.0, 1.0, 1.0]), pd.Series([2.0, 3.0, 4.0]))

    assert too_few["slope_status"] == "insufficient_points"
    assert math.isnan(too_few["slope"])
    assert constant_w["slope_status"] == "constant_w"
    assert math.isnan(constant_w["slope"])


def test_filter_relative_error_keeps_none_boundary_and_nan() -> None:
    mod = load_module()
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "relative_error_pct": [199.9, 200.0, 200.1, math.nan],
        }
    )

    unfiltered = mod.filter_by_relative_error(frame, None)
    filtered = mod.filter_by_relative_error(frame, 200.0)

    assert unfiltered["ID"].tolist() == [1, 2, 3, 4]
    assert filtered["ID"].tolist() == [1, 2, 4]


def test_build_slope_table_outputs_spaces_and_metrics() -> None:
    mod = load_module()
    rows = []
    for space, offset in [("X-space", 0.0), ("Y-space", 10.0), ("Z-space", 20.0)]:
        for w in [0.0, 1.0, 2.0]:
            rows.append(
                {
                    "scope": "ood",
                    "task_key": "Al__aluminum__UTS_MPa",
                    "task_id": "Al-UTS",
                    "method_short": "LOCO",
                    "model": "GPT-5.4",
                    "space": space,
                    "sample_w_contribution": w,
                    "abs_error": 1.0 + offset + 2.0 * w,
                    "relative_error_pct": 5.0 + offset + 3.0 * w,
                }
            )
    frame = pd.DataFrame(rows)

    slopes = mod.build_slope_table(frame, subset="standard")

    assert set(slopes["space"]) == {"X-space", "Y-space", "Z-space"}
    assert set(slopes["error_metric"]) == {"abs_error", "relative_error_pct"}
    assert len(slopes) == 6
    assert slopes.loc[slopes["error_metric"].eq("abs_error"), "slope"].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert slopes.loc[slopes["error_metric"].eq("relative_error_pct"), "slope"].tolist() == pytest.approx([3.0, 3.0, 3.0])
    assert set(slopes["slope_status"]) == {"ok"}


def test_plot_slope_comparisons_write_both_comparison_groups(tmp_path: Path) -> None:
    mod = load_module()
    slopes = pd.DataFrame(
        {
            "scope": ["ood"] * 12,
            "subset": ["standard"] * 12,
            "task_key": ["Al__aluminum__UTS_MPa"] * 12,
            "task_id": ["Al-UTS"] * 12,
            "method_short": ["LOCO"] * 6 + ["RandCV"] * 6,
            "model": ["GPT-5.4", "GPT-5.4", "GPT-5.4", "GPT-5.4", "GPT-5.4", "GPT-5.4"] * 2,
            "space": ["X-space", "Y-space", "Z-space", "X-space", "Y-space", "Z-space"] * 2,
            "error_metric": ["abs_error", "abs_error", "abs_error", "relative_error_pct", "relative_error_pct", "relative_error_pct"] * 2,
            "slope": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "intercept": [0.0] * 12,
            "n_points": [3] * 12,
            "x_min": [0.0] * 12,
            "x_max": [1.0] * 12,
            "y_min": [0.0] * 12,
            "y_max": [1.0] * 12,
            "slope_status": ["ok"] * 12,
        }
    )

    manifest = mod.plot_slope_comparisons(slopes, tmp_path, ["png"], dpi=50)

    assert set(manifest["figure_group"]) == {"slope_by_task_method", "slope_by_task_model"}
    assert all(Path(path).exists() for path in manifest["figure"])
    assert (tmp_path / "figures" / "slope_comparison" / "by_task_method" / "Al-UTS" / "LOCO_slope_by_model.png").exists()
    assert (tmp_path / "figures" / "slope_comparison" / "by_task_model" / "Al-UTS" / "GPT-5.4_slope_by_method.png").exists()


def test_run_one_output_writes_slope_csvs_and_manifest(tmp_path: Path) -> None:
    mod = load_module()
    rows = []
    for method in ["LOCO", "RandCV"]:
        for model in ["GPT-5.4", "RF"]:
            for space in ["X-space", "Y-space", "Z-space"]:
                for w in [0.0, 1.0, 2.0]:
                    rows.append(
                        {
                            "scope": "ood",
                            "task_key": "Al__aluminum__UTS_MPa",
                            "task_id": "Al-UTS",
                            "method_short": method,
                            "model": model,
                            "space": space,
                            "sample_w_contribution": w,
                            "abs_error": 1.0 + w,
                            "relative_error_pct": 10.0 + 2.0 * w,
                        }
                    )
    frame = pd.DataFrame(rows)

    mod.run_one_output(frame, tmp_path, subset="standard", formats=["png"], dpi=50, max_relative_error_pct=200.0)

    slopes = pd.read_csv(tmp_path / "csv" / "w_error_trend_slopes_long.csv")
    by_task_method = pd.read_csv(tmp_path / "csv" / "w_error_trend_slopes_by_task_method.csv")
    by_task_model = pd.read_csv(tmp_path / "csv" / "w_error_trend_slopes_by_task_model.csv")
    manifest = pd.read_csv(tmp_path / "csv" / "slope_figure_manifest.csv")

    assert not slopes.empty
    assert set(slopes["space"]) == {"X-space", "Y-space", "Z-space"}
    assert set(slopes["error_metric"]) == {"abs_error", "relative_error_pct"}
    assert list(by_task_method.columns) == mod.BY_TASK_METHOD_COLUMNS
    assert list(by_task_model.columns) == mod.BY_TASK_MODEL_COLUMNS
    assert set(manifest["figure_group"]) == {"slope_by_task_method", "slope_by_task_model"}
