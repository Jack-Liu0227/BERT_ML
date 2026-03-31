"""
Data preparation utilities for single-target TabPFN extrapolation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ExtrapolationSplitSummary:
    target_col: str
    extrapolation_side: str
    test_size: float
    train_size: int
    test_size_rows: int
    total_size: int
    train_target_min: float
    train_target_max: float
    test_target_min: float
    test_target_max: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TabPFNExtrapolationDataProcessor:
    """Prepare one target at a time for low-to-high extrapolation."""

    def __init__(self, config: Dict, base_path: str = "."):
        self.config = config
        self.base_path = Path(base_path)
        self.scaler = StandardScaler()
        self.data: pd.DataFrame | None = None
        self.available_feature_cols: list[str] = []
        self.numeric_feature_cols: list[str] = []
        self.non_numeric_feature_cols: list[str] = []

    def load_data(self) -> pd.DataFrame:
        data_path = self.base_path / self.config["raw_data"]
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = pd.read_csv(data_path)
        return self.data

    def _set_feature_type_metadata(self, X: pd.DataFrame) -> None:
        self.numeric_feature_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.non_numeric_feature_cols = [
            col for col in X.columns if col not in self.numeric_feature_cols
        ]

    def _sanitize_non_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        if self.non_numeric_feature_cols:
            X_clean.loc[:, self.non_numeric_feature_cols] = (
                X_clean[self.non_numeric_feature_cols].fillna("").astype(str)
            )
        return X_clean

    def _fill_missing_feature_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()
        self._set_feature_type_metadata(X_filled)
        if self.numeric_feature_cols:
            X_filled.loc[:, self.numeric_feature_cols] = X_filled[self.numeric_feature_cols].fillna(0)
        if self.non_numeric_feature_cols:
            X_filled.loc[:, self.non_numeric_feature_cols] = (
                X_filled[self.non_numeric_feature_cols].fillna("").astype(str)
            )
        return X_filled

    def prepare_feature_frame(
        self,
        target_col: str,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        feature_cols = self.config.get("feature_cols", [])
        available_features = [col for col in feature_cols if col in self.data.columns]
        if not available_features:
            raise ValueError("No configured feature columns are present in the dataset")
        self.available_feature_cols = available_features

        frame = self.data[available_features].copy()
        frame[target_col] = pd.to_numeric(self.data[target_col], errors="coerce")
        if "ID" in self.data.columns:
            frame["ID"] = self.data["ID"].copy()
        else:
            frame["ID"] = pd.Series(range(1, len(self.data) + 1), name="ID")

        if drop_na:
            frame = frame.dropna(subset=available_features + [target_col]).copy()
        else:
            frame[available_features] = self._fill_missing_feature_values(frame[available_features])
            frame[target_col] = frame[target_col].fillna(frame[target_col].mean())

        if frame.empty:
            raise ValueError("No rows remain after preparing extrapolation data")

        frame = frame.sort_values("ID").reset_index(drop=True)
        self._set_feature_type_metadata(frame[self.available_feature_cols])
        frame.loc[:, self.available_feature_cols] = self._sanitize_non_numeric_columns(
            frame[self.available_feature_cols]
        )
        return frame

    def split_low_high(
        self,
        feature_frame: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        extrapolation_side: str = "low_to_high",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, ExtrapolationSplitSummary]:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if extrapolation_side != "low_to_high":
            raise ValueError("Only 'low_to_high' extrapolation is supported")
        if len(feature_frame) < 2:
            raise ValueError("At least two valid rows are required for extrapolation split")

        ordered = feature_frame.sort_values(target_col, ascending=True).reset_index(drop=True)
        train_count = int(round(len(ordered) * (1 - test_size)))
        train_count = max(1, min(train_count, len(ordered) - 1))

        train_df = ordered.iloc[:train_count].copy().reset_index(drop=True)
        test_df = ordered.iloc[train_count:].copy().reset_index(drop=True)

        summary = ExtrapolationSplitSummary(
            target_col=target_col,
            extrapolation_side=extrapolation_side,
            test_size=test_size,
            train_size=len(train_df),
            test_size_rows=len(test_df),
            total_size=len(ordered),
            train_target_min=float(train_df[target_col].min()),
            train_target_max=float(train_df[target_col].max()),
            test_target_min=float(test_df[target_col].min()),
            test_target_max=float(test_df[target_col].max()),
        )
        return train_df, test_df, summary

    def build_model_inputs(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        scale: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.available_feature_cols:
            raise ValueError("Feature columns are not prepared")

        X_train = train_df[self.available_feature_cols].copy()
        X_test = test_df[self.available_feature_cols].copy()
        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()
        ids_train = train_df["ID"].to_numpy()
        ids_test = test_df["ID"].to_numpy()

        self._set_feature_type_metadata(X_train)
        X_train = self._sanitize_non_numeric_columns(X_train)
        X_test = self._sanitize_non_numeric_columns(X_test)

        if scale and self.numeric_feature_cols:
            X_train_numeric = pd.DataFrame(
                self.scaler.fit_transform(X_train[self.numeric_feature_cols]),
                columns=self.numeric_feature_cols,
                index=X_train.index,
            )
            X_test_numeric = pd.DataFrame(
                self.scaler.transform(X_test[self.numeric_feature_cols]),
                columns=self.numeric_feature_cols,
                index=X_test.index,
            )
            X_train.loc[:, self.numeric_feature_cols] = X_train_numeric
            X_test.loc[:, self.numeric_feature_cols] = X_test_numeric

        return X_train, X_test, y_train, y_test, ids_train, ids_test

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

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "train_file": str(train_path),
            "test_file": str(test_path),
            "summary_file": str(summary_path),
        }
