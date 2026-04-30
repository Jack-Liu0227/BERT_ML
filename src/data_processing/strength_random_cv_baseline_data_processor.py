from __future__ import annotations

from typing import Any, List

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedFold,
    StrengthOODProcessorBase,
    prepare_random_cv_baseline_folds,
)


class StrengthRandomCVBaselineDataProcessor(StrengthOODProcessorBase):
    split_strategy = "random_cv_baseline"

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> List[PreparedFold]:
        return prepare_random_cv_baseline_folds(
            df=df,
            target_col=target_col,
            num_folds=int(kwargs.get("baseline_num_folds", 5)),
            random_state=self.random_state,
        )
