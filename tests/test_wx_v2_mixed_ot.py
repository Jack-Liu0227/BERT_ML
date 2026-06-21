from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data_processing.compute_wx_v2_mixed_ot import (
    build_mixed_distance_components,
    compute_wx_v2_for_frames,
    positive_train_scale,
)


def test_zero_and_presence_distance_rules_are_bounded() -> None:
    train = pd.DataFrame(
        {
            "all_zero": [0.0, 0.0],
            "positive": [2.0, 4.0],
            "single_positive": [0.0, 5.0],
        }
    )
    test = pd.DataFrame(
        {
            "all_zero": [0.0, 7.0],
            "positive": [2.0, 10.0],
            "single_positive": [0.0, 20.0],
        }
    )

    components = build_mixed_distance_components(train, test, list(train.columns))

    all_zero = components.feature_distances["all_zero"]
    assert np.allclose(all_zero[:, 0], [0.0, 0.0])
    assert np.allclose(all_zero[:, 1], [1.0, 1.0])

    positive = components.feature_distances["positive"]
    assert np.allclose(positive[:, 0], [0.0, 1.0])
    assert np.allclose(positive[:, 1], [1.0, 1.0])

    single_positive = components.feature_distances["single_positive"]
    assert np.allclose(single_positive[:, 0], [0.0, 1.0])
    assert np.allclose(single_positive[:, 1], [1.0, 1.0])

    assert np.nanmax(components.cost_matrix) <= 1.0
    assert np.nanmin(components.cost_matrix) >= 0.0


def test_positive_train_scale_fallbacks_are_finite() -> None:
    assert positive_train_scale(pd.Series([0.0, 0.0, np.nan])) == 1.0
    assert positive_train_scale(pd.Series([0.0, 5.0, 5.0])) == 5.0
    assert positive_train_scale(pd.Series([1.0, 3.0, 5.0])) == 2.0


def test_wx_v2_sample_and_feature_contributions_are_conservative() -> None:
    train = pd.DataFrame({"feature": [0.0, 1.0]})
    test = pd.DataFrame({"feature": [1.0]})

    result = compute_wx_v2_for_frames(
        train_df=train,
        test_df=test,
        feature_columns=["feature"],
        metadata={"method": "unit", "fold_id": "fold_0"},
        target_col="target",
    )

    assert math.isclose(result.split_row["wx_v2"], 0.5, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(result.sample_scores["wx_v2_sample_score"].mean(), result.split_row["wx_v2"])
    assert math.isclose(result.sample_scores["wx_v2_mass_contribution"].sum(), result.split_row["wx_v2"])

    feature_sum = result.feature_contributions.groupby("__row_id__")["wx_v2_feature_score"].sum()
    sample_scores = result.sample_scores.set_index("__row_id__")["wx_v2_sample_score"]
    assert np.allclose(feature_sum.loc[sample_scores.index].to_numpy(), sample_scores.to_numpy())


def test_wx_v2_preserves_test_row_order_and_identifiers() -> None:
    train = pd.DataFrame({"ID": [1, 2], "__row_id__": [10, 11], "__source_index__": [100, 101], "target": [0, 1], "feature": [0, 1]})
    test = pd.DataFrame({"ID": [8, 7], "__row_id__": [20, 21], "__source_index__": [200, 201], "target": [2, 3], "feature": [1, 0]})

    result = compute_wx_v2_for_frames(
        train_df=train,
        test_df=test,
        feature_columns=["feature"],
        metadata={"method": "unit", "fold_id": "fold_0"},
        target_col="target",
    )

    assert result.sample_scores["ID"].tolist() == [8, 7]
    assert result.sample_scores["__row_id__"].tolist() == [20, 21]
    assert result.sample_scores["__source_index__"].tolist() == [200, 201]
    assert result.sample_scores["target_value"].tolist() == [2, 3]
    assert result.sample_scores["wx_v2_rank_desc"].notna().all()
