from __future__ import annotations

from typing import Any, List

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedFold,
    PreparedSplit,
    StrengthOODProcessorBase,
    inner_strategy_from_hybrid,
    prepare_hybrid_extrapolation_split,
)


class StrengthHybridExtrapolationDataProcessor(StrengthOODProcessorBase):
    split_strategy = ""

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> PreparedSplit | List[PreparedFold]:
        inner_strategy = inner_strategy_from_hybrid(self.split_strategy)
        if inner_strategy == "loco":
            inner_cluster_count = int(kwargs.get("loco_cluster_count", 5))
        elif inner_strategy == "random_cv":
            inner_cluster_count = int(kwargs.get("baseline_num_folds", 5))
        else:
            inner_cluster_count = int(kwargs.get("sparse_cluster_count", 5))
        return prepare_hybrid_extrapolation_split(
            df=df,
            target_col=target_col,
            split_strategy=self.split_strategy,
            inner_strategy=inner_strategy,
            outer_test_ratio=float(kwargs.get("outer_test_size", 0.2)),
            inner_test_ratio=float(kwargs.get("test_ratio", kwargs.get("test_size", 0.2))),
            random_state=self.random_state,
            candidate_pool_size=int(kwargs.get("sparse_candidate_pool_size", 500)),
            cluster_count=inner_cluster_count,
            samples_per_cluster=int(kwargs.get("sparse_samples_per_cluster", 1)),
            neighbors_per_seed=int(kwargs.get("sparse_neighbors_per_seed", 5)),
            kde_bandwidth=kwargs.get("sparse_kde_bandwidth"),
            processing_cols=kwargs.get("processing_cols", self.processing_cols),
        )


class StrengthHybridExtrapolationSparseXSingleDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_sparse_x_single"


class StrengthHybridExtrapolationSparseYSingleDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_sparse_y_single"


class StrengthHybridExtrapolationSparseXClusterDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_sparse_x_cluster"


class StrengthHybridExtrapolationSparseYClusterDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_sparse_y_cluster"


class StrengthHybridExtrapolationLocoDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_loco"


class StrengthHybridExtrapolationRandomCVDataProcessor(StrengthHybridExtrapolationDataProcessor):
    split_strategy = "hybrid_extrapolation_random_cv"
