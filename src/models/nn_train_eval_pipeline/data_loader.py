"""
Data loading functionality for the training pipeline.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from .utils import now

def load_data(data_file: str, 
              target_columns: List[str], 
              test_size: float = 0.2, 
              random_state: int = 42,
              use_process_embedding: bool = False,
              use_joint_composition_process_embedding: bool = False,
              use_element_embedding: bool = False,
              use_composition_feature: bool = False,
              use_feature1: bool = False,
              use_feature2: bool = False,
              other_features_name: Optional[List[str]] = None,
              use_temperature: bool = False,
              result_dir: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load data from processed feature file and split into train and test sets, with flexible feature selection.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    print(f"[{now()}] Loading data from: {data_file}")
    
    data = pd.read_csv(data_file)
    
    is_target_only = all(col in target_columns for col in data.columns)
    
    if is_target_only:
        features_path = os.path.normpath(os.path.join(os.path.dirname(data_file), 'features_with_id.csv'))
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        print(f"[{now()}] Loading features from: {features_path}")
        features_data = pd.read_csv(features_path)
        
        if len(features_data) != len(data):
            raise ValueError(f"Features file has {len(features_data)} rows but target file has {len(data)} rows")
        
        for col in data.columns:
            features_data[col] = data[col].values
        
        data = features_data

    # 优先从 features_with_id.csv 读取ID（首列），确保与特征/目标完全对齐
    ids_series = None
    try:
        f_with_id_path = os.path.normpath(os.path.join(os.path.dirname(data_file), 'features_with_id.csv'))
        if os.path.exists(f_with_id_path):
            print(f"[{now()}] Loading IDs from: {f_with_id_path}")
            f_with_id_df = pd.read_csv(f_with_id_path)
            if f_with_id_df.shape[1] >= 1:
                ids_series = f_with_id_df.iloc[:, 0]
            if ids_series is not None and len(ids_series) != len(data):
                raise ValueError(f"IDs length ({len(ids_series)}) does not match data length ({len(data)})")
    except Exception as e:
        print(f"[{now()}] WARNING: Failed to load features_with_id.csv for IDs: {e}")
    
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")
    
    feature_cols = [col for col in data.columns if col not in target_columns]
    print(f"[{now()}] Found {len(feature_cols)} feature columns and {len(target_columns)} target columns")
    
    ele_emb_cols = [col for col in feature_cols if 'ele_emb' in col]
    proc_emb_cols = [col for col in feature_cols if 'proc_emb' in col]
    joint_emb_cols = [col for col in feature_cols if 'joint_emb' in col]
    composition_cols = [col for col in feature_cols if '(wt%)' in col or '(at%)' in col]
    feature1_cols = [col for col in feature_cols if 'feature1' in col.lower()]
    feature2_cols = [col for col in feature_cols if 'feature2' in col.lower()]
    temperature_cols = [col for col in feature_cols if 'temperature' in col.lower()]
    other_cols = []
    if other_features_name and other_features_name != ['None']:
        for name in other_features_name:
            name = name.strip()
            if name:
                other_cols.extend([col for col in feature_cols if name in col])

    selected_cols = []
    if use_element_embedding: selected_cols.extend(ele_emb_cols)
    if use_process_embedding: selected_cols.extend(proc_emb_cols)
    if use_joint_composition_process_embedding: selected_cols.extend(joint_emb_cols)
    if use_composition_feature: selected_cols.extend(composition_cols)
    if use_feature1: selected_cols.extend(feature1_cols)
    if use_feature2: selected_cols.extend(feature2_cols)
    if use_temperature: selected_cols.extend(temperature_cols)
    if other_cols: selected_cols.extend(other_cols)

    selected_cols = list(dict.fromkeys(selected_cols))

    if not selected_cols:
        raise ValueError('No features selected! Please check your feature selection parameters.')
    print(f"[{now()}] Selected {len(selected_cols)} features for X: {selected_cols}")
    
    X = data[selected_cols].values.astype('float32')
    y = data[target_columns].values.astype('float32')

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

    # 加入ID（若存在）
    if ids_series is not None:
        train_val_data['ids'] = ids_train_val
        test_data['ids'] = ids_test
    
    return train_val_data, test_data, selected_cols 