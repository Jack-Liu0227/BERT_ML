from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass
class ExtrapolationSplitSummary:
    split_target_col: str
    extrapolation_side: str
    test_ratio: float
    train_size: int
    test_size: int
    total_size: int
    train_target_min: float
    train_target_max: float
    test_target_min: float
    test_target_max: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class StrengthExtrapolationDataProcessor:
    """Split a dataset into low-target train and high-target extrapolation test sets."""

    def __init__(self, input_file: str, random_state: int = 42) -> None:
        self.input_file = input_file
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.input_file)

    def split_low_high(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_ratio: float = 0.2,
        extrapolation_side: str = "low_to_high",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, ExtrapolationSplitSummary]:
        if target_col not in df.columns:
            raise ValueError(f"split target column '{target_col}' not found in dataset")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if extrapolation_side not in {"low_to_high", "high_to_low"}:
            raise ValueError("extrapolation_side must be one of ['low_to_high', 'high_to_low']")

        work_df = df.copy()
        work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
        work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(work_df) < 2:
            raise ValueError("at least two valid rows are required for extrapolation split")

        ascending = extrapolation_side == "low_to_high"
        work_df = work_df.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)

        train_count = int(round(len(work_df) * (1 - test_ratio)))
        train_count = max(1, min(train_count, len(work_df) - 1))

        train_df = work_df.iloc[:train_count].copy().reset_index(drop=True)
        test_df = work_df.iloc[train_count:].copy().reset_index(drop=True)

        train_target = pd.to_numeric(train_df[target_col], errors="coerce")
        test_target = pd.to_numeric(test_df[target_col], errors="coerce")

        summary = ExtrapolationSplitSummary(
            split_target_col=target_col,
            extrapolation_side=extrapolation_side,
            test_ratio=test_ratio,
            train_size=len(train_df),
            test_size=len(test_df),
            total_size=len(work_df),
            train_target_min=float(train_target.min()),
            train_target_max=float(train_target.max()),
            test_target_min=float(test_target.min()),
            test_target_max=float(test_target.max()),
        )
        return train_df, test_df, summary

    def save_split_artifacts(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        summary: ExtrapolationSplitSummary,
        output_dir: str,
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train_low.csv"
        test_path = output_path / "test_high.csv"
        summary_path = output_path / "split_summary.json"

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "train_file": str(train_path),
            "test_file": str(test_path),
            "summary_file": str(summary_path),
        }
