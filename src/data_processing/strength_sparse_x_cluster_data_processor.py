from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedSplit,
    StrengthOODProcessorBase,
    prepare_sparse_cluster_split,
)


class StrengthSparseXClusterDataProcessor(StrengthOODProcessorBase):
    split_strategy = "sparse_x_cluster"

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> PreparedSplit:
        return prepare_sparse_cluster_split(
            df=df,
            target_col=target_col,
            split_strategy=self.split_strategy,
            density_space="x",
            test_ratio=float(kwargs.get("test_ratio", kwargs.get("test_size", 0.2))),
            candidate_pool_size=int(kwargs.get("sparse_candidate_pool_size", 500)),
            cluster_count=int(kwargs.get("sparse_cluster_count", 50)),
            neighbors_per_seed=int(kwargs.get("sparse_neighbors_per_seed", 5)),
            random_state=self.random_state,
        )
