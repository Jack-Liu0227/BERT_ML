from __future__ import annotations

from typing import Any, List

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedFold,
    StrengthOODProcessorBase,
    prepare_loco_folds,
)


class StrengthLocoDataProcessor(StrengthOODProcessorBase):
    split_strategy = "loco"

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> List[PreparedFold]:
        return prepare_loco_folds(
            df=df,
            target_col=target_col,
            cluster_count=int(kwargs.get("loco_cluster_count", 5)),
            random_state=self.random_state,
        )
