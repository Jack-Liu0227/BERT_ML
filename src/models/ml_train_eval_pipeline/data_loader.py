"""
Data loading functionality for the training pipeline.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
try:
    # 尝试相对导入（当作为模块运行时）
    from .utils import now
except ImportError:
    # 尝试直接导入（当直接运行时）
    try:
        from utils import now
    except ImportError:
        # 使用完整路径导入
        from src.models.ml_train_eval_pipeline.utils import now

def load_data(data_file: str,
              target_columns: List[str],
              test_size: float = 0.2,
              random_state: int = 42,
              processing_cols: Optional[List[str]] = None,
              use_composition_feature: bool = False,
              use_temperature: bool = False,
              other_features_name: Optional[List[str]] = None,
              result_dir: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load data from processed feature file and split into train and test sets, with flexible feature selection.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    print(f"[{now()}] Loading data from: {data_file}")

    data = pd.read_csv(data_file)

    # Extract IDs if available
    ids_series = None
    if 'ID' in data.columns:
        ids_series = data['ID']
        print(f"[{now()}] Found ID column with {len(ids_series)} records")
    else:
        print(f"[{now()}] No ID column found, will use row indices as IDs")
    # Fill NaN values with 0 for all columns (自动将所有NaN值填充为0)
    data = data.fillna(0)

    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    feature_cols = [col for col in data.columns if col not in target_columns]
    print(f"[{now()}] Found {len(feature_cols)} feature columns and {len(target_columns)} target columns")

    composition_cols = [col for col in feature_cols if '(wt%)' in col or '(at%)' in col]
    temperature_cols = [col for col in feature_cols if 'temperature' in col.lower()]
    other_cols = []
    if other_features_name and other_features_name != ['None']:
        for name in other_features_name:
            name = name.strip()
            if name:
                other_cols.extend([col for col in feature_cols if name in col])

    selected_cols = []
    if use_composition_feature: selected_cols.extend(composition_cols)
    if other_cols: selected_cols.extend(other_cols)
    if processing_cols:
        # Validate that processing_cols exist in the data
        valid_processing_cols = [col for col in processing_cols if col in feature_cols]
        invalid_processing_cols = [col for col in processing_cols if col not in feature_cols]

        if invalid_processing_cols:
            print(f"[{now()}] Warning: The following processing columns were not found in data: {invalid_processing_cols}")
            print(f"[{now()}] Available feature columns: {feature_cols}")

        selected_cols.extend(valid_processing_cols)
    if use_temperature:
        selected_cols.extend([col for col in feature_cols if 'temperature' in col.lower()])

    selected_cols = list(dict.fromkeys(selected_cols))

    if not selected_cols:
        raise ValueError('No features selected! Please check your feature selection parameters.')
    print(f"[{now()}] Selected {len(selected_cols)} features for X: {selected_cols}")
    
    X = data[selected_cols].values.astype('float32')
    y = data[target_columns].values.astype('float32')

    # Split data and IDs together if IDs are available
    if ids_series is not None:
        X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
            X, y, ids_series.values, test_size=test_size, random_state=random_state
        )
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    if result_dir is not None:
        # Save raw (un-normalized) features
        pass # This will be handled by the pipeline class

    train_val_data = {'X': X_train_val, 'y': y_train_val, 'feature_names': selected_cols}
    test_data = {'X': X_test, 'y': y_test, 'feature_names': selected_cols}

    # Add IDs if available
    if ids_series is not None:
        train_val_data['ids'] = ids_train_val
        test_data['ids'] = ids_test

    return train_val_data, test_data, selected_cols