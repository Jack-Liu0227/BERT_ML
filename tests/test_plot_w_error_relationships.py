from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Scripts" / "plot_w_error_relationships.py"


def load_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("plot_w_error_relationships", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_methods_covers_standard_and_hybrid_names() -> None:
    mod = load_module()

    assert mod.normalize_standard_method("random_cv_baseline") == "RandCV"
    assert mod.normalize_standard_method("target_extrapolation") == "Extra."
    assert mod.normalize_standard_method("sparse_x_cluster") == "SX-cls"
    assert mod.normalize_hybrid_method("hybrid_extrapolation_sparse_y_single") == "SY-sgl"
    assert mod.normalize_hybrid_method("HybridHigh20+LOCO") == "LOCO"
    assert mod.normalize_hybrid_method("HybridHigh20+RandCV") == "RandCV"


def test_parse_fold_detail_prediction_sources_accepts_tabpfn_keys(tmp_path: Path) -> None:
    mod = load_module()
    row = pd.Series(
        {
            "tabpfn_loco_fold_details_json": (
                '[{"fold_index": 2, "model_dir": "models/fold_2", '
                '"predictions_file": "predictions/fold_2.csv"}]'
            )
        }
    )

    details = mod.parse_fold_detail_prediction_sources(row, "tabpfn_loco_fold_details_json")

    assert len(details) == 1
    assert details[0]["fold_id"] == "fold_2"
    assert details[0]["prediction_file"] == REPO_ROOT / "predictions" / "fold_2.csv"
    assert details[0]["model_dir"] == REPO_ROOT / "models" / "fold_2"


def test_parse_tabpfn_fold_details_prefers_artifact_strategy() -> None:
    mod = load_module()
    row = pd.Series(
        {
            "artifact_predictions_file": (
                r"output\ood_results_TabPFN-2.5-Plus-Numeric\loco_k5\Al\aluminum"
                r"\UTS(MPa)\folds\fold_0\predictions\all_predictions.csv"
            ),
            "tabpfn_loco_fold_details_json": (
                r'[{"fold_index": 0, "predictions_file": '
                r'"output\\ood_results_TabPFN-2.5-Plus-Numeric\\loco\\Al\\folds\\fold_0\\predictions\\all_predictions.csv"},'
                r'{"fold_index": 0, "predictions_file": '
                r'"output\\ood_results_TabPFN-2.5-Plus-Numeric\\loco_k5\\Al\\folds\\fold_0\\predictions\\all_predictions.csv"}]'
            ),
        }
    )

    details = mod.parse_fold_detail_prediction_sources(row, "tabpfn_loco_fold_details_json")

    assert len(details) == 1
    assert "loco_k5" in str(details[0]["prediction_file"])


def test_compute_error_columns_keeps_mae_when_true_value_is_zero() -> None:
    mod = load_module()
    frame = pd.DataFrame({"true_value": [0.0, 10.0], "predicted_value": [2.5, 7.0]})

    result = mod.compute_error_columns(frame, true_col="true_value", pred_col="predicted_value")

    assert result["abs_error"].tolist() == [2.5, 3.0]
    assert math.isnan(result.loc[0, "relative_error_pct"])
    assert result.loc[1, "relative_error_pct"] == 30.0


def test_parse_max_relative_error_pct_accepts_numbers_and_none() -> None:
    mod = load_module()

    assert mod.parse_max_relative_error_pct("200") == 200.0
    assert mod.parse_max_relative_error_pct("none") is None
    assert mod.parse_max_relative_error_pct("NULL") is None
    with pytest.raises(SystemExit):
        mod.parse_args(["--max-relative-error-pct", "-1"])


def test_filter_errors_by_relative_error_threshold_keeps_none_boundary_and_nan() -> None:
    mod = load_module()
    errors = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "relative_error_pct": [199.9, 200.0, 200.1, math.nan],
            "abs_error": [10.0, 20.0, 30.0, 40.0],
        }
    )

    unfiltered = mod.filter_errors_by_relative_error(errors, None)
    filtered = mod.filter_errors_by_relative_error(errors, 200.0)

    assert unfiltered["ID"].tolist() == [1, 2, 3, 4]
    assert filtered["ID"].tolist() == [1, 2, 4]
    assert filtered.loc[filtered["ID"].eq(4), "abs_error"].iloc[0] == 40.0


def test_gpt_true_predicted_columns_are_read_and_resolved(tmp_path: Path) -> None:
    mod = load_module()
    prediction_file = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "ID": [101, 102],
            "UTS(MPa)": [410.0, 520.0],
            "UTS(MPa)_true": [400.0, 500.0],
            "UTS(MPa)_predicted": [390.0, 530.0],
            "confidence": ["high", "medium"],
        }
    ).to_csv(prediction_file, index=False)

    frame = mod.read_prediction_csv(prediction_file)
    dataset_col, actual_col, pred_col = mod.resolve_prediction_columns(frame, "UTS(MPa)")

    assert dataset_col is None
    assert actual_col == "UTS(MPa)_true"
    assert pred_col == "UTS(MPa)_predicted"


def test_join_w_and_errors_prefers_fold_id_and_id() -> None:
    mod = load_module()
    w_values = pd.DataFrame(
        {
            "scope": ["ood", "ood"],
            "task_key": ["Al__aluminum__UTS_MPa", "Al__aluminum__UTS_MPa"],
            "method_short": ["LOCO", "LOCO"],
            "fold_id": ["fold_0", "fold_1"],
            "ID": [7, 7],
            "space": ["X-space", "X-space"],
            "sample_w_contribution": [0.1, 0.9],
        }
    )
    errors = pd.DataFrame(
        {
            "scope": ["ood"],
            "task_key": ["Al__aluminum__UTS_MPa"],
            "method_short": ["LOCO"],
            "fold_id": ["fold_1"],
            "ID": [7],
            "model": ["RF"],
            "abs_error": [12.0],
            "relative_error_pct": [3.0],
        }
    )

    joined, coverage = mod.join_w_and_errors(w_values, errors)

    assert len(joined) == 1
    assert joined.iloc[0]["sample_w_contribution"] == 0.9
    assert joined.iloc[0]["model"] == "RF"
    assert coverage.iloc[0]["matched_rows"] == 1
    assert coverage.iloc[0]["unmatched_w_rows"] == 1


def test_confidence_prediction_sources_fall_back_to_gpt_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_module()
    final_root = tmp_path / "final"
    table_dir = final_root / "OOD" / "confidence_ood_relationship" / "csv"
    table_dir.mkdir(parents=True)
    gpt_root = tmp_path / "fewshot-guided" / "output" / "ood" / "k5"
    actual_prediction = (
        gpt_root
        / "strength_ood_extrapolation_no_analysis"
        / "al_uts_target_extrapolation"
        / "kilig"
        / "gpt-5.4"
        / "predictions.csv"
    )
    actual_prediction.parent.mkdir(parents=True)
    actual_prediction.write_text("ID,UTS(MPa)_true,UTS(MPa)_predicted\n1,400,410\n", encoding="utf-8")
    stale_prediction = (
        final_root
        / "OOD"
        / "confidence_ood_k5_inputs"
        / "strength_ood_extrapolation_no_analysis"
        / "al_uts_target_extrapolation"
        / "kilig"
        / "gpt-5.4"
        / "predictions.csv"
    )
    pd.DataFrame(
        [
            {
                "task_id": "Al-UTS",
                "method": "Extra.",
                "ood_method": "Extrapolation",
                "root": "strength_ood_extrapolation_no_analysis",
                "task_dir": "al_uts_target_extrapolation",
                "fold": "",
                "prediction_path": str(stale_prediction),
            }
        ]
    ).to_csv(table_dir / "confidence_by_prediction_file.csv", index=False)
    monkeypatch.setattr(mod, "final_results_root", lambda: final_root)
    monkeypatch.setattr(mod, "GPT_PREDICTION_ROOT", gpt_root, raising=False)

    sources = mod.collect_confidence_prediction_sources("ood", case_contains="Al__aluminum__UTS_MPa")

    assert len(sources) == 1
    assert sources[0].prediction_file == actual_prediction


def test_hybrid_gpt_confidence_sources_use_ood_split_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_module()
    final_root = tmp_path / "final"
    table_dir = final_root / "OOD HYBIRD" / "confidence_ood_relationship" / "csv"
    table_dir.mkdir(parents=True)
    gpt_root = tmp_path / "fewshot-guided" / "output" / "ood" / "k5"
    prediction_file = (
        gpt_root
        / "strength_ood_hybrid_extrapolation_loco_no_analysis"
        / "al_uts_hybrid_extrapolation_loco"
        / "fold_0"
        / "openai"
        / "gpt-5.4"
        / "predictions.csv"
    )
    prediction_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "UTS(MPa)_true": [400.0, 500.0, 600.0],
            "UTS(MPa)_predicted": [410.0, 480.0, 630.0],
        }
    ).to_csv(prediction_file, index=False)
    split_data = (
        tmp_path
        / "output"
        / "ood_splits"
        / "Al"
        / "aluminum"
        / "UTS(MPa)"
        / "hybrid_extrapolation_loco"
        / "abc123"
        / "folds"
        / "fold_0"
        / "split_data"
    )
    split_data.mkdir(parents=True)
    (split_data / "split_summary.json").write_text(
        json.dumps({"outer_test_size": 1, "inner_test_size": 1}),
        encoding="utf-8",
    )
    pd.DataFrame({"ID": [1, 2, 3]}).to_csv(split_data / "test_hybrid_combined.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_id": "Al-UTS",
                "method": "LOCO",
                "ood_method": "HybridHigh20+LOCO",
                "root": "strength_ood_hybrid_extrapolation_loco_no_analysis",
                "task_dir": "al_uts_hybrid_extrapolation_loco",
                "fold": "fold_0",
                "prediction_path": str(prediction_file),
            }
        ]
    ).to_csv(table_dir / "confidence_by_prediction_file.csv", index=False)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "final_results_root", lambda: final_root)
    monkeypatch.setattr(mod, "GPT_PREDICTION_ROOT", gpt_root, raising=False)

    sources = mod.collect_confidence_prediction_sources("hybrid", case_contains="Al__aluminum__UTS_MPa")
    errors = mod.prediction_errors_from_source(sources[0], hybrid_subset="test_inner_ood")

    assert sources[0].expected_split_file == split_data / "test_hybrid_combined.csv"
    assert errors["ID"].tolist() == [2]


def test_filter_hybrid_subset_selects_combined_and_child_sets() -> None:
    mod = load_module()
    frame = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "test_set": ["test_hybrid_combined", "test_extrapolation_high20", "test_inner_ood"],
        }
    )

    assert mod.filter_hybrid_subset(frame, "combined")["ID"].tolist() == [1, 2, 3]
    assert mod.filter_hybrid_subset(frame, "test_extrapolation_high20")["ID"].tolist() == [2]
    assert mod.filter_hybrid_subset(frame, "test_inner_ood")["ID"].tolist() == [3]


def test_case_filter_matches_hybrid_task_key_not_only_path_text() -> None:
    mod = load_module()

    assert mod.case_filter_matches(
        "Al__aluminum__UTS_MPa",
        r"D:\repo\output\ood_splits\Al\aluminum\UTS(MPa)\hybrid_extrapolation_loco",
        "Al__aluminum__UTS_MPa",
    )


def test_hybrid_cache_roots_separate_full_and_filtered_runs(tmp_path: Path) -> None:
    mod = load_module()

    assert mod.hybrid_cache_run_root(tmp_path, None) == tmp_path / "full"
    filtered = mod.hybrid_cache_run_root(tmp_path, "Al__aluminum__UTS_MPa")
    assert filtered.parent == tmp_path / "case_filters"
    assert filtered != tmp_path / "full"


def test_fit_linear_trend_line_uses_only_finite_distinct_points() -> None:
    mod = load_module()

    line = mod.fit_linear_trend_line(
        pd.Series([0.0, 1.0, 2.0, math.nan, 3.0]),
        pd.Series([1.0, 3.0, 5.0, 7.0, math.inf]),
    )

    assert line is not None
    line_x, line_y = line
    assert line_x.tolist() == [0.0, 2.0]
    assert line_y.tolist() == pytest.approx([1.0, 5.0])
    assert mod.fit_linear_trend_line(pd.Series([0.1]), pd.Series([1.0])) is None
    assert mod.fit_linear_trend_line(pd.Series([0.1, 0.1]), pd.Series([1.0, 2.0])) is None


def test_empty_output_tables_keep_expected_headers() -> None:
    mod = load_module()

    joined, coverage = mod.join_w_and_errors(pd.DataFrame(), pd.DataFrame())
    correlations = mod.build_correlations(joined)
    figures = mod.plot_task_model_figures(joined, Path("."), ["png"], 100)

    assert list(joined.columns) == mod.JOINED_OUTPUT_COLUMNS + ["__w_row_id"]
    assert list(coverage.columns) == mod.COVERAGE_COLUMNS
    assert list(correlations.columns) == mod.CORRELATION_COLUMNS
    assert list(figures.columns) == mod.FIGURE_MANIFEST_COLUMNS


def test_build_correlations_handles_constant_inputs_without_warnings() -> None:
    mod = load_module()
    joined = pd.DataFrame(
        {
            "scope": ["ood", "ood"],
            "task_key": ["Al__aluminum__UTS_MPa", "Al__aluminum__UTS_MPa"],
            "task_id": ["Al-UTS", "Al-UTS"],
            "model": ["GPT-5.4", "GPT-5.4"],
            "method_short": ["LOCO", "LOCO"],
            "space": ["X-space", "X-space"],
            "sample_w_contribution": [0.1, 0.2],
            "abs_error": [10.0, 10.0],
            "relative_error_pct": [5.0, 5.0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlations = mod.build_correlations(joined)

    assert caught == []
    assert correlations["pearson_r"].isna().all()
    assert correlations["spearman_r"].isna().all()


def test_plot_filenames_do_not_collide_when_model_names_contain_dots(tmp_path: Path) -> None:
    mod = load_module()
    joined = pd.DataFrame(
        {
            "task_id": ["Al-UTS", "Al-UTS"],
            "model": ["TabPFN-2.5-Plus-Numeric", "TabPFN-2.5-Plus-Text"],
            "space": ["X-space", "X-space"],
            "method_short": ["LOCO", "LOCO"],
            "sample_w_contribution": [0.1, 0.2],
            "abs_error": [1.0, 2.0],
            "relative_error_pct": [3.0, 4.0],
        }
    )

    manifest = mod.plot_task_model_figures(joined, tmp_path, ["png"], dpi=50)

    assert len(manifest) == 2
    assert set(manifest["figure_group"]) == {"by_task_model"}
    assert manifest["method_short"].isna().all()
    assert manifest["figure"].nunique() == 2
    assert all(Path(path).name.endswith("_w_error_scatter.png") for path in manifest["figure"])
    assert all(Path(path).exists() for path in manifest["figure"])


def test_plot_method_figures_groups_by_method_task_and_model(tmp_path: Path) -> None:
    mod = load_module()
    joined = pd.DataFrame(
        {
            "task_id": ["Al-UTS", "Al-UTS", "Al-UTS", "Al-UTS"],
            "model": ["GPT-5.4", "GPT-5.4", "GPT-5.4", "GPT-5.4"],
            "space": ["X-space", "X-space", "X-space", "X-space"],
            "method_short": ["LOCO", "LOCO", "RandCV", "RandCV"],
            "sample_w_contribution": [0.1, 0.2, 0.3, 0.4],
            "abs_error": [1.0, 2.0, 2.0, 4.0],
            "relative_error_pct": [10.0, 20.0, 30.0, 40.0],
        }
    )

    manifest = mod.plot_method_figures(joined, tmp_path, ["png"], dpi=50)

    assert len(manifest) == 2
    assert set(manifest["figure_group"]) == {"by_method"}
    assert set(manifest["method_short"]) == {"LOCO", "RandCV"}
    assert all("\\by_method\\" in str(path) or "/by_method/" in str(path) for path in manifest["figure"])
    assert all(Path(path).exists() for path in manifest["figure"])


def test_run_one_output_writes_task_model_and_method_manifest(tmp_path: Path) -> None:
    mod = load_module()
    w_table = pd.DataFrame(
        {
            "scope": ["ood", "ood"],
            "task_key": ["Al__aluminum__UTS_MPa", "Al__aluminum__UTS_MPa"],
            "task_id": ["Al-UTS", "Al-UTS"],
            "alloy_family": ["Al", "Al"],
            "dataset_name": ["aluminum", "aluminum"],
            "property": ["UTS(MPa)", "UTS(MPa)"],
            "method_short": ["LOCO", "LOCO"],
            "method": ["loco", "loco"],
            "fold_id": ["fold_0", "fold_0"],
            "ID": [1, 2],
            "sample_order": [0, 1],
            "test_set": ["test", "test"],
            "space": ["X-space", "X-space"],
            "sample_w_contribution": [0.1, 0.2],
            "sample_w_mass_contribution": [0.1, 0.2],
            "split_w": [0.3, 0.3],
            "target_value": [100.0, 120.0],
            "source_split_dir": ["split", "split"],
        }
    )
    errors = pd.DataFrame(
        {
            "scope": ["ood", "ood"],
            "task_key": ["Al__aluminum__UTS_MPa", "Al__aluminum__UTS_MPa"],
            "method_short": ["LOCO", "LOCO"],
            "method": ["loco", "loco"],
            "model": ["GPT-5.4", "GPT-5.4"],
            "model_family": ["GPT", "GPT"],
            "fold_id": ["fold_0", "fold_0"],
            "ID": [1, 2],
            "sample_order": [0, 1],
            "test_set": ["test", "test"],
            "true_value": [100.0, 120.0],
            "predicted_value": [110.0, 132.0],
            "signed_error": [10.0, 12.0],
            "abs_error": [10.0, 12.0],
            "relative_error_pct": [10.0, 10.0],
            "prediction_file": ["predictions.csv", "predictions.csv"],
            "source_split_dir": ["split", "split"],
        }
    )

    mod.run_one_output(
        w_table,
        errors,
        pd.DataFrame(columns=mod.PREDICTION_INVENTORY_COLUMNS),
        tmp_path,
        ["png"],
        dpi=50,
        max_relative_error_pct=200.0,
    )

    manifest = pd.read_csv(tmp_path / "csv" / "figure_manifest.csv")
    assert set(manifest["figure_group"]) == {"by_task_model", "by_method"}
    assert (tmp_path / "figures" / "by_task_model").exists()
    assert (tmp_path / "figures" / "by_method").exists()
    assert "- `figures/by_method/`" in (tmp_path / "analysis_report.md").read_text(encoding="utf-8")


def test_build_phase_diagram_samples_wide_keeps_sample_level_points() -> None:
    mod = load_module()
    rows = []
    for sample_id, order in [(1, 0), (2, 1)]:
        for space, value in [("X-space", 0.1 + order), ("Y-space", 0.2 + order), ("Z-space", 0.3 + order)]:
            rows.append(
                {
                    "scope": "ood",
                    "task_key": "Al__aluminum__UTS_MPa",
                    "task_id": "Al-UTS",
                    "alloy_family": "Al",
                    "dataset_name": "aluminum",
                    "property": "UTS(MPa)",
                    "method_short": "LOCO",
                    "method": "loco",
                    "model": "GPT-5.4",
                    "model_family": "LLM",
                    "fold_id": "fold_0",
                    "ID": sample_id,
                    "sample_order": order,
                    "test_set": "ood_test",
                    "space": space,
                    "sample_w_contribution": value,
                    "sample_w_mass_contribution": value / 10.0,
                    "split_w": 0.5,
                    "target_value": 100.0 + order,
                    "true_value": 100.0 + order,
                    "predicted_value": 110.0 + order,
                    "signed_error": 10.0,
                    "abs_error": 10.0,
                    "relative_error_pct": 10.0 + order,
                    "prediction_file": "predictions.csv",
                    "source_split_dir": "split",
                    "join_mode": "fold_id_id",
                }
            )

    wide = mod.build_phase_diagram_samples_wide(pd.DataFrame(rows))

    assert len(wide) == 2
    assert wide["ID"].tolist() == [1, 2]
    assert wide["X_space_w"].tolist() == [0.1, 1.1]
    assert wide["Y_space_w"].tolist() == [0.2, 1.2]
    assert wide["Z_space_w"].tolist() == [0.3, 1.3]
    assert wide["relative_error_pct"].tolist() == [10.0, 11.0]


def test_phase_diagram_points_drop_missing_target_or_rep_space() -> None:
    mod = load_module()
    wide = pd.DataFrame(
        {
            "task_id": ["Al-UTS", "Al-UTS", "Al-UTS"],
            "method_short": ["LOCO", "LOCO", "LOCO"],
            "model": ["GPT-5.4", "GPT-5.4", "GPT-5.4"],
            "model_family": ["LLM", "LLM", "LLM"],
            "X_space_w": [0.1, 0.2, 0.3],
            "Y_space_w": [0.4, None, 0.6],
            "Z_space_w": [0.7, 0.8, None],
            "relative_error_pct": [5.0, 50.0, 500.0],
        }
    )

    points = mod.phase_diagram_points(wide, "Z-space")

    assert len(points) == 1
    assert points.iloc[0]["Z_space_w"] == 0.7
    assert points.iloc[0]["Y_space_w"] == 0.4


def test_phase_color_limit_uses_percentile_and_ignores_nan() -> None:
    mod = load_module()
    values = pd.Series([1.0, 2.0, 3.0, float("nan"), 1000.0])

    limit = mod.phase_color_limit(values, percentile=75)

    assert limit == pytest.approx(3.0)


def test_plot_phase_diagram_figures_write_zy_and_xy_outputs(tmp_path: Path) -> None:
    mod = load_module()
    wide = pd.DataFrame(
        {
            "task_id": ["Al-UTS", "Al-UTS", "Al-UTS", "Al-UTS"],
            "method_short": ["LOCO", "LOCO", "LOCO", "LOCO"],
            "model": ["GPT-5.4", "GPT-5.4", "LLM-Prop", "LLM-Prop"],
            "model_family": ["LLM", "LLM", "LLMProp", "LLMProp"],
            "X_space_w": [0.1, 0.2, 0.3, 0.4],
            "Y_space_w": [0.5, 0.6, 0.7, 0.8],
            "Z_space_w": [0.9, 1.0, 1.1, 1.2],
            "relative_error_pct": [5.0, 10.0, 15.0, 20.0],
        }
    )

    manifest = mod.plot_phase_diagram_figures(
        wide,
        tmp_path,
        ["png"],
        dpi=50,
        spaces=["zy", "xy"],
        color_percentile=95,
    )

    assert len(manifest) == 2
    assert set(manifest["figure_group"]) == {"phase_diagram"}
    assert set(manifest["method_short"]) == {"LOCO"}
    assert set(manifest["rep_space"]) == {"Z-space", "X-space"}
    assert all(Path(path).exists() for path in manifest["figure"])
    assert any(Path(path).name == "zy_phase_diagram.png" for path in manifest["figure"])
    assert any(Path(path).name == "xy_phase_diagram.png" for path in manifest["figure"])
