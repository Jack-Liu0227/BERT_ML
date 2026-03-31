"""
Data preparation utilities for TabPFN training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabPFNDataProcessor:
    """Load alloy data and prepare mixed-type feature frames for TabPFN."""

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.data: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.numeric_feature_names: list[str] = []
        self.non_numeric_feature_names: list[str] = []

    def load_data(self, base_path: str = ".") -> pd.DataFrame:
        data_path = Path(base_path) / self.config["raw_data"]
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading data from: {data_path}")
        self.data = pd.read_csv(data_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def _set_feature_type_metadata(self, X: pd.DataFrame) -> None:
        self.numeric_feature_names = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.non_numeric_feature_names = [
            col for col in X.columns if col not in self.numeric_feature_names
        ]

    def _sanitize_non_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()
        if self.non_numeric_feature_names:
            X_clean.loc[:, self.non_numeric_feature_names] = (
                X_clean[self.non_numeric_feature_names].fillna("").astype(str)
            )
        return X_clean

    def _fill_missing_feature_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()
        self._set_feature_type_metadata(X_filled)
        if self.numeric_feature_names:
            X_filled.loc[:, self.numeric_feature_names] = X_filled[self.numeric_feature_names].fillna(0)
        if self.non_numeric_feature_names:
            X_filled.loc[:, self.non_numeric_feature_names] = (
                X_filled[self.non_numeric_feature_names].fillna("").astype(str)
            )
        return X_filled

    def prepare_features_and_targets(
        self,
        target_col: str,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        feature_cols = self.config["feature_cols"]
        available_features = [col for col in feature_cols if col in self.data.columns]
        missing_features = [col for col in feature_cols if col not in self.data.columns]

        if missing_features:
            print(
                f"Warning: {len(missing_features)} feature columns not found: "
                f"{missing_features[:5]}..."
            )

        print(f"Using {len(available_features)} features")
        self.feature_names = available_features

        X = self.data[available_features].copy()
        y = self.data[target_col].copy()
        if "ID" in self.data.columns:
            ids = self.data["ID"].copy()
        else:
            ids = pd.Series(range(1, len(self.data) + 1), name="ID")

        temp_df = X.copy()
        temp_df["_target"] = y
        temp_df["_id"] = ids

        if drop_na:
            n_before = len(temp_df)
            temp_df = temp_df.dropna(subset=available_features + ["_target"]).copy()
            n_after = len(temp_df)
            if n_before > n_after:
                print(f"Dropped {n_before - n_after} rows with missing values")
        else:
            temp_df[available_features] = self._fill_missing_feature_values(
                temp_df[available_features]
            )
            temp_df["_target"] = temp_df["_target"].fillna(temp_df["_target"].mean())

        temp_df = temp_df.sort_values("_id").reset_index(drop=True)

        X = temp_df[available_features].copy()
        y = temp_df["_target"].copy()
        ids = temp_df["_id"].copy()
        self._set_feature_type_metadata(X)
        X = self._sanitize_non_numeric_columns(X)

        print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print("Data sorted by ID for consistent train/test split")
        return X, y, ids

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ids: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        test_size = test_size or self.config.get("test_size", 0.3)
        random_state = random_state or self.config.get("random_state", 42)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X,
            y,
            ids,
            test_size=test_size,
            random_state=random_state,
        )

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train/Test split: {(1 - test_size) * 100:.0f}% / {test_size * 100:.0f}%")
        return X_train, X_test, y_train, y_test, ids_train, ids_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._set_feature_type_metadata(X_train)

        X_train_scaled = self._sanitize_non_numeric_columns(X_train)
        X_test_scaled = self._sanitize_non_numeric_columns(X_test)

        if self.numeric_feature_names:
            X_train_numeric = pd.DataFrame(
                self.scaler.fit_transform(X_train[self.numeric_feature_names]),
                columns=self.numeric_feature_names,
                index=X_train.index,
            )
            X_test_numeric = pd.DataFrame(
                self.scaler.transform(X_test[self.numeric_feature_names]),
                columns=self.numeric_feature_names,
                index=X_test.index,
            )
            X_train_scaled.loc[:, self.numeric_feature_names] = X_train_numeric
            X_test_scaled.loc[:, self.numeric_feature_names] = X_test_numeric

        return X_train_scaled, X_test_scaled

    def get_full_pipeline(
        self,
        target_col: str,
        base_path: str = ".",
        scale: bool = True,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print(f"\n{'=' * 60}")
        print(f"Processing {self.config['description']}")
        print(f"Target: {target_col}")
        print(f"{'=' * 60}")

        self.load_data(base_path)
        X, y, ids = self.prepare_features_and_targets(target_col, drop_na=drop_na)
        X_train, X_test, y_train, y_test, ids_train, ids_test = self.split_data(X, y, ids)

        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
            if self.non_numeric_feature_names:
                print(
                    "Scaled numeric features with StandardScaler and kept "
                    "text/categorical columns unchanged"
                )
            else:
                print("Features scaled using StandardScaler")
        else:
            self._set_feature_type_metadata(X_train)
            X_train = self._sanitize_non_numeric_columns(X_train)
            X_test = self._sanitize_non_numeric_columns(X_test)
            print("Features kept in DataFrame format without scaling")

        return (
            X_train,
            X_test,
            y_train.to_numpy(),
            y_test.to_numpy(),
            ids_train.to_numpy(),
            ids_test.to_numpy(),
        )


def create_data_processor(alloy_type: str, config_dict: Dict) -> TabPFNDataProcessor:
    config = config_dict.get(alloy_type)
    if config is None:
        raise ValueError(f"Unknown alloy type: {alloy_type}")
    return TabPFNDataProcessor(config)
