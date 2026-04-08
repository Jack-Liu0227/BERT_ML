from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedSplit,
    StrengthOODProcessorBase,
    prepare_sparse_single_split,
)


class StrengthSparseYSingleDataProcessor(StrengthOODProcessorBase):
    split_strategy = "sparse_y_single"

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> PreparedSplit:
        return prepare_sparse_single_split(
            df=df,
            target_col=target_col,
            split_strategy=self.split_strategy,
            density_space="y",
            test_ratio=float(kwargs.get("test_ratio", kwargs.get("test_size", 0.2))),
            candidate_pool_size=int(kwargs.get("sparse_candidate_pool_size", 500)),
            cluster_count=int(kwargs.get("sparse_cluster_count", 50)),
            samples_per_cluster=int(kwargs.get("sparse_samples_per_cluster", 1)),
            random_state=self.random_state,
            kde_bandwidth=kwargs.get("sparse_kde_bandwidth"),
        )
