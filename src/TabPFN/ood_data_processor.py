"""
Data preparation utilities for single-target TabPFN OOD runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_processing.strength_ood_common import PreparedFold, PreparedSplit, save_prepared_split
from src.data_processing.strength_ood_registry import create_ood_processor


class TabPFNOODDataProcessor:
    """Prepare one target at a time for supported OOD split strategies."""

    def __init__(self, config: Dict, base_path: str = "."):
        self.config = config
        self.base_path = Path(base_path)
        self.scaler = StandardScaler()
        self.data: pd.DataFrame | None = None
        self.prepared_result: PreparedSplit | List[PreparedFold] | None = None
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
            raise ValueError("No rows remain after preparing OOD data")

        frame = frame.sort_values("ID").reset_index(drop=True)
        self._set_feature_type_metadata(frame[self.available_feature_cols])
        frame.loc[:, self.available_feature_cols] = self._sanitize_non_numeric_columns(
            frame[self.available_feature_cols]
        )
        return frame

    def prepare_ood_result(
        self,
        feature_frame: pd.DataFrame,
        target_col: str,
        split_strategy: str,
        test_size: float = 0.2,
        extrapolation_side: str = "low_to_high",
        sparse_candidate_pool_size: int = 500,
        sparse_cluster_count: int = 50,
        sparse_samples_per_cluster: int = 1,
        sparse_kde_bandwidth: float | None = None,
        sparse_neighbors_per_seed: int = 5,
        loco_cluster_count: int = 5,
    ) -> PreparedSplit | List[PreparedFold]:
        processor = create_ood_processor(
            split_strategy=split_strategy,
            input_file=str(self.base_path / self.config["raw_data"]),
            random_state=int(self.config.get("random_state", 42)),
        )
        prepared_result = processor.prepare(
            df=feature_frame,
            target_col=target_col,
            test_ratio=test_size,
            extrapolation_side=extrapolation_side,
            sparse_candidate_pool_size=sparse_candidate_pool_size,
            sparse_cluster_count=sparse_cluster_count,
            sparse_samples_per_cluster=sparse_samples_per_cluster,
            sparse_kde_bandwidth=sparse_kde_bandwidth,
            sparse_neighbors_per_seed=sparse_neighbors_per_seed,
            loco_cluster_count=loco_cluster_count,
        )
        self.prepared_result = prepared_result
        return prepared_result

    def prepare_ood_split(
        self,
        feature_frame: pd.DataFrame,
        target_col: str,
        split_strategy: str,
        test_size: float = 0.2,
        extrapolation_side: str = "low_to_high",
        sparse_candidate_pool_size: int = 500,
        sparse_cluster_count: int = 50,
        sparse_samples_per_cluster: int = 1,
        sparse_kde_bandwidth: float | None = None,
        sparse_neighbors_per_seed: int = 5,
        loco_cluster_count: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        prepared_result = self.prepare_ood_result(
            feature_frame=feature_frame,
            target_col=target_col,
            split_strategy=split_strategy,
            test_size=test_size,
            extrapolation_side=extrapolation_side,
            sparse_candidate_pool_size=sparse_candidate_pool_size,
            sparse_cluster_count=sparse_cluster_count,
            sparse_samples_per_cluster=sparse_samples_per_cluster,
            sparse_kde_bandwidth=sparse_kde_bandwidth,
            sparse_neighbors_per_seed=sparse_neighbors_per_seed,
            loco_cluster_count=loco_cluster_count,
        )
        if isinstance(prepared_result, list):
            raise ValueError(
                f"TabPFN OOD trainer currently supports only single-split strategies, got '{split_strategy}'"
            )

        train_df = prepared_result.train_df.copy().reset_index(drop=True)
        test_df = prepared_result.test_df.copy().reset_index(drop=True)
        return train_df, test_df, dict(prepared_result.summary)

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
        summary: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, str]:
        if self.prepared_result is None:
            raise ValueError("No prepared OOD split is available to save")
        if isinstance(self.prepared_result, list):
            raise ValueError("save_split_artifacts only supports single prepared splits")
        if self.prepared_result.summary != summary:
            raise ValueError("Provided summary does not match prepared OOD split summary")

        return save_prepared_split(self.prepared_result, output_dir)
