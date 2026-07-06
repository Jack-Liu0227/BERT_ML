from __future__ import annotations

import pandas as pd

from src.data_processing.compute_wx_mixed_ot import compute_wx_for_frames

from .common import SPLIT_SOURCE, WX_MODE, infer_x_feature_columns
from .sources import ResultSource
from .splits import configured_feature_columns, load_train_and_test


def compute_source_wx(source: ResultSource, *, hybrid_subset: str = "combined"):
    train, test, summary, test_set = load_train_and_test(source, hybrid_subset=hybrid_subset)
    target_col = str(summary.get("split_target_col") or source.property_name)
    feature_columns = infer_x_feature_columns(
        train,
        test,
        target_col=target_col,
        configured=configured_feature_columns(summary),
    )
    metadata = {
        "alloy_family": source.family,
        "dataset_name": source.dataset,
        "property": source.property_name,
        "method": source.method,
        "split_strategy": summary.get("split_strategy", source.method_dir),
        "split_id": source.method_dir,
        "fold_id": source.fold_id,
        "source_split_dir": str(source.split_dir),
        "train_file": str(source.train_file),
        "test_file": str(source.combined_test_file),
    }
    result = compute_wx_for_frames(
        train_df=train,
        test_df=test,
        feature_columns=feature_columns,
        metadata=metadata,
        target_col=target_col,
        configured_roles=summary.get("x_space_feature_roles") or {},
    )
    return result, train, test, summary, feature_columns, test_set


def w_inventory_row(source: ResultSource, wx_result, train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], test_set: str) -> dict[str, object]:
    samples = wx_result.sample_scores
    return {
        "ood_scope": source.scope,
        "task_key": source.task_key,
        "model": source.model,
        "method": source.method,
        "fold_id": source.fold_id,
        "test_set": test_set,
        "wx_mode": WX_MODE,
        "split_source": SPLIT_SOURCE,
        "source_split_dir": str(source.split_dir),
        "train_file": str(source.train_file),
        "test_file": str(source.combined_test_file),
        "split_summary_file": str(source.split_summary_file),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": ";".join(feature_columns),
        "wx": float(wx_result.split_row["wx"]),
        "mean_wx_sample_score": float(pd.to_numeric(samples["wx_sample_score"], errors="coerce").mean()),
        "sum_wx_mass_contribution": float(pd.to_numeric(samples["wx_mass_contribution"], errors="coerce").sum()),
    }
