from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_processing.strength_ood_common import (
    PreparedSplit,
    StrengthOODProcessorBase,
    prepare_target_extrapolation_split,
)


class StrengthExtrapolationDataProcessor(StrengthOODProcessorBase):
    """Dedicated processor for target-based extrapolation only."""

    split_strategy = "target_extrapolation"

    def prepare(self, df: pd.DataFrame, target_col: str, **kwargs: Any) -> PreparedSplit:
        return prepare_target_extrapolation_split(
            df=df,
            target_col=target_col,
            test_ratio=float(kwargs.get("test_ratio", kwargs.get("test_size", 0.2))),
            extrapolation_side=str(kwargs.get("extrapolation_side", "low_to_high")),
            random_state=self.random_state,
        )

    def split_low_high(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_ratio: float = 0.2,
        extrapolation_side: str = "low_to_high",
    ) -> PreparedSplit:
        return self.prepare(
            df=df,
            target_col=target_col,
            test_ratio=test_ratio,
            extrapolation_side=extrapolation_side,
        )
