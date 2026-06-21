from __future__ import annotations

import math

import pandas as pd

from Scripts.export_wx_v2_sample_w_reports import (
    build_v2_sample_w_table,
    combine_v2_x_with_cached_yz,
    summarize_v2_sample_w,
)


def test_build_v2_sample_w_table_maps_wx_v2_scores_to_legacy_columns() -> None:
    samples = pd.DataFrame(
        {
            "alloy_family": ["Al", "Al"],
            "dataset_name": ["aluminum", "aluminum"],
            "property": ["UTS(MPa)", "UTS(MPa)"],
            "method": ["loco", "loco"],
            "split_id": ["abc", "abc"],
            "fold_id": ["fold_3", "fold_3"],
            "source_split_dir": ["split", "split"],
            "ID": [2, 1],
            "__row_id__": [20, 10],
            "__source_index__": [200, 100],
            "target_col": ["UTS(MPa)", "UTS(MPa)"],
            "target_value": [300.0, 250.0],
            "wx_v2_sample_score": [0.2, 0.8],
            "wx_v2_mass_contribution": [0.1, 0.4],
            "wx_v2_rank_desc": [2, 1],
        }
    )

    table = build_v2_sample_w_table(samples)

    assert table["sample_w_contribution"].tolist() == [0.8, 0.2]
    assert table["sample_w_mass_contribution"].tolist() == [0.4, 0.1]
    assert table["sample_w_rank_desc"].tolist() == [1, 2]
    assert table["case_label"].tolist() == ["Al / aluminum / UTS(MPa)", "Al / aluminum / UTS(MPa)"]
    assert table["method"].tolist() == ["loco", "loco"]


def test_summarize_v2_sample_w_computes_method_level_statistics() -> None:
    table = pd.DataFrame(
        {
            "alloy_family": ["Al", "Al", "Al"],
            "dataset_name": ["aluminum", "aluminum", "aluminum"],
            "property": ["UTS(MPa)", "UTS(MPa)", "UTS(MPa)"],
            "method": ["random_cv_baseline", "random_cv_baseline", "random_cv_baseline"],
            "fold_id": ["fold_0", "fold_0", "fold_1"],
            "sample_w_contribution": [0.1, 0.3, 0.5],
            "sample_w_mass_contribution": [0.05, 0.15, 0.25],
            "ood_score": [0.1, 0.2, 0.3],
            "ood_percentile_vs_train": [10.0, 20.0, 30.0],
        }
    )

    summary = summarize_v2_sample_w(table)
    row = summary.iloc[0]

    assert row["sample_count"] == 3
    assert row["fold_count"] == 2
    assert math.isclose(row["sample_w_mean"], 0.3)
    assert math.isclose(row["sample_w_median"], 0.3)
    assert math.isclose(row["sample_w_mass_sum"], 0.45)


def test_combine_v2_x_with_cached_yz_replaces_only_x_space_rows() -> None:
    v2_x = pd.DataFrame(
        {
            "case_key": ["Al__aluminum__UTS_MPa"],
            "case_label": ["Al / aluminum / UTS(MPa)"],
            "alloy_family": ["Al"],
            "dataset_name": ["aluminum"],
            "property": ["UTS(MPa)"],
            "method": ["loco"],
            "space": ["X-space"],
            "sample_w_contribution": [0.4],
            "sample_w_mass_contribution": [0.2],
            "split_w": [0.4],
        }
    )
    cached = pd.DataFrame(
        {
            "case_key": ["old_x", "old_y", "old_z"],
            "case_label": ["old", "old", "old"],
            "alloy_family": ["Al", "Al", "Al"],
            "dataset_name": ["aluminum", "aluminum", "aluminum"],
            "property": ["UTS(MPa)", "UTS(MPa)", "UTS(MPa)"],
            "method": ["loco", "loco", "loco"],
            "space": ["X-space", "Y-space", "Z-space"],
            "sample_w_contribution": [9.0, 1.0, 2.0],
            "sample_w_mass_contribution": [9.0, 1.0, 2.0],
            "split_w": [9.0, 1.0, 2.0],
        }
    )

    combined = combine_v2_x_with_cached_yz(v2_x, cached)

    assert combined["space"].tolist() == ["X-space", "Y-space", "Z-space"]
    assert combined.loc[combined["space"] == "X-space", "sample_w_contribution"].iloc[0] == 0.4
    assert combined.loc[combined["space"] == "Y-space", "sample_w_contribution"].iloc[0] == 1.0
    assert combined.loc[combined["space"] == "Z-space", "sample_w_contribution"].iloc[0] == 2.0
