"""
Data loading functionality for the CNN training pipeline.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from .utils import now

def load_data(data_file: str, 
              target_columns: List[str], 
              test_size: float = 0.2, 
              random_state: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load data from a CSV file, automatically selecting all feature columns.
    The CNN model will internally select which features to use based on configuration.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    print(f"[{now()}] Loading data from: {data_file}")
    
    all_data = pd.read_csv(data_file)
    
    # Check for missing target columns
    missing_targets = [col for col in target_columns if col not in all_data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")
    
    # All columns that are not targets are considered features
    feature_names = [col for col in all_data.columns if col not in target_columns]
    
    if not feature_names:
        raise ValueError("No feature columns found. Ensure your CSV file contains feature columns.")

    print(f"[{now()}] Found {len(feature_names)} feature columns and {len(target_columns)} target columns.")
    
    X = all_data[feature_names].values.astype('float32')
    y = all_data[target_columns].values.astype('float32')
    
    # Split data into training/validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_val_data = {'X': X_train_val, 'y': y_train_val}
    test_data = {'X': X_test, 'y': y_test}
    
    return train_val_data, test_data, feature_names 