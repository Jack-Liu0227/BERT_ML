"""
Unified pipeline for feature engineering, custom data splitting, training, and evaluation.

This script combines feature generation, a specific data splitting strategy based on
temperature-based labels and outlier detection, neural network training, and evaluation
into a single workflow.

The data splitting logic is as follows:
1.  Data is partitioned by a 'label' column (e.g., temperature zones 0, 1, 2, 3).
2.  For each partition, outliers are identified using the 1.5 * IQR rule on temperature.
    - Low-temperature outliers are points below Q1 - 1.5 * IQR.
    - High-temperature outliers are points above Q3 + 1.5 * IQR.
    - "Whiskers" data are points within these bounds.
3.  The training set is built by:
    - Sampling 2,000 data points from the "whiskers" data of each label.
4.  Two test sets are created:
    - 'low_T_tests': Contains all low-temperature outliers from all labels. This set is
      designed to test the model's performance on a critical, targeted subset.
    - 'all_tests': A broader test set. For each label, it includes all high-temperature
      outliers and all "whiskers" data that was not used for training.
5.  This ensures the training and test sets are strictly mutually exclusive.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
from collections import OrderedDict
import json

# Add src to Python path to allow for absolute imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Assuming these are the correct import paths for your project structure
from src.feature_engineering.feature_processor import FeatureProcessor
from src.models.base.alloys_nn import AlloyNN
from src.models.trainers import TrainerFactory
from src.models.evaluators import EvaluatorFactory
from src.models.visualization.plot_utils import plot_compare_scatter, plot_loss_r2_curves


def now():
    """Return current time string for logging."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def generate_features(
    data_path: str,
    feature_dir: str,
    target_columns: List[str],
    args: argparse.Namespace
) -> pd.DataFrame:
    """
    Runs the feature processing pipeline. If features already exist, it loads them
    unless forced to regenerate.

    Args:
        data_path (str): Path to the raw data file.
        feature_dir (str): Directory to save feature-related outputs.
        target_columns (List[str]): List of target column names.
        args (argparse.Namespace): Parsed arguments from the command line.

    Returns:
        pd.DataFrame: A dataframe containing the original data plus the generated features.
    """
    features_path = os.path.join(feature_dir, "features.csv")

    if os.path.exists(features_path) and not args.force_regenerate:
        print(f"[{now()}] Found existing features at {features_path}. Loading them.")
        features_df = pd.read_csv(features_path).drop(columns=target_columns)
        print(f"[{now()}] Loaded {features_df.shape[1]} features.")
    else:
        print(f"[{now()}] Starting feature generation... (Forced: {args.force_regenerate})")
        log_dir = os.path.join(feature_dir, "logs")
        model_path = "./SteelBERTmodel"  # Make sure this path is correct

        processor = FeatureProcessor(
            data_path=data_path,
            use_process_embedding=args.use_process_embedding,
            use_composition_embedding=args.use_composition_embedding,
            use_feature1=args.use_feature1,
            use_feature2=args.use_feature2,
            use_composition_feature=args.use_composition_feature,
            use_temperature=args.use_temperature,
            standardize_features=False, # Standardization will be handled later
            feature_dir=feature_dir,
            log_dir=log_dir,
            target_columns=target_columns,
            model_path=model_path,
            other_features_name=args.other_features_name,
            use_joint_composition_process_embedding=args.use_joint_composition_process_embedding
        )

        features_df, _ = processor.process()
        print(f"[{now()}] Feature generation complete. {features_df.shape[1]} features created.")

    # The processor separates features and targets, we need to bring them back together
    # with the original data for the custom split logic.
    original_data = pd.read_csv(data_path)
    
    # Combine original data (which has labels and temp) with the new features
    # Assuming the processor maintains the original order
    full_data = pd.concat([original_data, features_df], axis=1)
    # print(f"[{now()}] Full data shape: {full_data.shape}")
    # Check for duplicate columns that might arise if features were already in original_data
    full_data = full_data.loc[:, ~full_data.columns.duplicated()]

    return full_data


def custom_data_split(
    data: pd.DataFrame,
    target_columns: List[str], 
    number_of_train_samples: int = 2000,
    number_of_test_samples: int = 10000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and test sets based on the new rules:
    - High-temperature outliers are discarded.
    - Low-temperature outliers form the test set.
    - All data within the whiskers forms the training set.

    Args:
        data (pd.DataFrame): The full dataset containing features and labels.
        target_columns (List[str]): The target column names. The first is used for outlier detection.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training dataframe and the test dataframe.
    """
    print(f"[{now()}] Starting custom data splitting with revised rules...")
    train_dfs = []
    all_test_dfs = []
    low_T_test_dfs = []
    low_outliers_dfs = []
    whiskers_low_T_test_dfs = []
    labels = sorted(data['label'].unique())
    print(f"[{now()}] Found labels: {labels}")

    # Use the first target column for the outlier splitting logic
    split_col = target_columns[0]
    print(f"[{now()}] Using column '{split_col}' for outlier detection logic.")

    for label in labels:
        label_data = data[data['label'] == label].copy()
        print(f"[{now()}] Processing label {label}: {len(label_data)} data points")

        if len(label_data) == 0:
            continue

        # 计算箱线图的关键统计量
        # Calculate key statistics for box plot
        
        # 计算第一四分位数(Q1)
        # Calculate first quartile (Q1)
        q1 = label_data[split_col].quantile(0.25)
        
        # 计算第三四分位数(Q3) 
        # Calculate third quartile (Q3)
        q3 = label_data[split_col].quantile(0.75)
        
        # 计算四分位距(IQR)
        # Calculate interquartile range (IQR)
        iqr = q3 - q1
        
        # 计算下界:Q1-1.5*IQR,用于识别低温异常值
        # Calculate lower bound: Q1-1.5*IQR, used to identify low temperature outliers
        lower_bound = q1 - 1.5 * iqr
        
        # 计算上界:Q3+1.5*IQR,用于识别高温异常值
        # Calculate upper bound: Q3+1.5*IQR, used to identify high temperature outliers  
        upper_bound = q3 + 1.5 * iqr

        low_outliers = label_data[label_data[split_col] < lower_bound]
        high_outliers = label_data[label_data[split_col] > upper_bound]
        # high_outliers are now discarded
        whiskers_data = label_data[
            (label_data[split_col] >= lower_bound) & (label_data[split_col] <= upper_bound)
        ]
        n_train_samples = min(number_of_train_samples, len(whiskers_data))
        n_test_samples = number_of_test_samples
        print(f"[{now()}] Label {label}: Low outliers (for test set)={len(low_outliers)}, Whiskers (for train set)={len(whiskers_data)}, High outliers (discarded)={len(high_outliers)}")
        print(f"[{now()}] Number of train samples: {n_train_samples}, Number of test samples: {n_test_samples}")

        # Low-temp outliers go to the test set
        if not low_outliers.empty:
            low_T_test_dfs.append(low_outliers)
            all_test_dfs.append(low_outliers)
        # Sample from whiskers data without replacement to ensure no overlap
        
        if not whiskers_data.empty:
            # Sample indices without replacement
            all_indices = whiskers_data.index.tolist()
            train_indices = np.random.RandomState(random_state).choice(
                all_indices, size=n_train_samples, replace=False
            )
            remaining_indices = list(set(all_indices) - set(train_indices))
            test_indices = np.random.RandomState(random_state).choice(
                remaining_indices, size=min(n_test_samples, len(remaining_indices)), replace=False
            )
            
            train_dfs.append(whiskers_data.loc[train_indices])
            all_test_dfs.append(whiskers_data.loc[test_indices])
            if label==0:
                low_T_test_dfs.append(whiskers_data.loc[test_indices])
                whiskers_low_T_test_dfs.append(whiskers_data.loc[test_indices])
                low_outliers_dfs.append(low_outliers)            
    # Combine data from all labels
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame(columns=data.columns)
    all_test_df = pd.concat(all_test_dfs) if all_test_dfs else pd.DataFrame(columns=data.columns)
    low_T_test_df = pd.concat(low_T_test_dfs) if low_T_test_dfs else pd.DataFrame(columns=data.columns)
    low_outliers_df = pd.concat(low_outliers_dfs) if low_outliers_dfs else pd.DataFrame(columns=data.columns)
    whiskers_low_T_test_df = pd.concat(whiskers_low_T_test_dfs) if whiskers_low_T_test_dfs else pd.DataFrame(columns=data.columns)
    # Shuffle and reset indices
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    all_test_df = all_test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    low_T_test_df = low_T_test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[{now()}] Splitting complete:")
    print(f"  - Total Training data points: {len(train_df)}")
    print(f"  - Total Test data points: {len(all_test_df)}")
    print(f"  - Total Low-T Test data points: {len(low_T_test_df)}")
    print(f"  - Total Whiskers Low-T Test data points: {len(whiskers_low_T_test_df)}")
    print(f"  - Total Low-outliers Test data points: {len(low_outliers_df)}")

    return train_df, all_test_df, low_T_test_df, low_outliers_df, whiskers_low_T_test_df


def select_features(
    data: pd.DataFrame,
    target_columns: List[str],
    use_process_embedding: bool = False,
    use_joint_composition_process_embedding: bool = False,
    use_composition_embedding: bool = False,
    use_composition_feature: bool = False,
    use_feature1: bool = False,
    use_feature2: bool = False,
    other_features_name: Optional[List[str]] = None,
    use_temperature: bool = False
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Selects feature columns based on boolean flags.

    Args:
        data (pd.DataFrame): The input dataframe.
        target_columns (List[str]): List of target column names.
        ... (feature selection flags) ...

    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: X, y, and selected feature names.
    """
    feature_cols = [col for col in data.columns if col not in target_columns and col != 'label']
    
    # Classify feature columns
    ele_emb_cols = [col for col in feature_cols if 'ele_emb' in col]
    proc_emb_cols = [col for col in feature_cols if 'proc_emb' in col]
    joint_emb_cols = [col for col in feature_cols if 'joint_emb' in col]
    composition_cols = [col for col in feature_cols if '(wt%)' in col or '(at%)' in col]
    feature1_cols = [col for col in feature_cols if 'feature1' in col.lower()]
    feature2_cols = [col for col in feature_cols if 'feature2' in col.lower()]
    temperature_cols = [col for col in feature_cols if 'temperature' in col.lower()]
    other_cols = []
    if other_features_name:
        for name in other_features_name:
            if name and name != 'None':
                other_cols.extend([col for col in feature_cols if name in col])

    # Build the list of selected columns
    selected_cols = []
    if use_composition_embedding:
        selected_cols.extend(ele_emb_cols)
    if use_process_embedding:
        selected_cols.extend(proc_emb_cols)
    if use_joint_composition_process_embedding:
        selected_cols.extend(joint_emb_cols)
    if use_composition_feature:
        selected_cols.extend(composition_cols)
    if use_feature1:
        selected_cols.extend(feature1_cols)
    if use_feature2:
        selected_cols.extend(feature2_cols)
    if use_temperature:
        selected_cols.extend(temperature_cols)
    if other_cols:
        selected_cols.extend(other_cols)
    
    # Remove duplicates while preserving order
    selected_cols = list(OrderedDict.fromkeys(selected_cols))

    if not selected_cols:
        raise ValueError("No features were selected. Check feature selection flags.")

    print(f"[{now()}] Selected {len(selected_cols)} features.")
    
    X = data[selected_cols]
    y = data[target_columns]
    
    return X, y, selected_cols


def prepare_data_for_model(
    train_df: pd.DataFrame,
    test_df_dict: Dict[str, pd.DataFrame],
    all_feature_names: List[str],
    target_names: List[str],
    result_dir: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], StandardScaler, StandardScaler, List[str]]:
    """
    Prepares train and test sets for the model: scaling, splitting, etc.
    It identifies which columns from the dataframe are actual features for the model.
    """
    # Exclude non-feature columns from the list of features to be used in the model
    model_feature_names = [
        col for col in all_feature_names
        if col not in target_names and 'label' not in col and 'temperature' not in col
    ]
    # This is a heuristic; a more robust way is to get the exact feature names from FeatureProcessor
    print(f"[{now()}] Identified {len(model_feature_names)} features for model training.")

    # Prepare training data
    X_train = train_df[model_feature_names].values.astype('float32')
    y_train = train_df[target_names].values.astype('float32')

    # 输出训练集特征和标签的shape
    print(f"[{now()}] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  # 打印训练特征和标签的shape

    # Normalize features based on the training set
    scaler_X = StandardScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    joblib.dump(scaler_X, os.path.join(result_dir, 'scaler_X.pkl'))
    print(f"[{now()}] Feature scaler (scaler_X) saved.")

    # Normalize targets based on the training set
    scaler_y = StandardScaler().fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    joblib.dump(scaler_y, os.path.join(result_dir, 'scaler_y.pkl'))
    print(f"[{now()}] Target scaler (scaler_y) saved.")

    train_data = {'X': X_train_scaled, 'y': y_train_scaled, 'feature_names': model_feature_names}

    # Prepare test data sets
    prepared_test_sets = {}
    for name, df in test_df_dict.items():
        if df.empty:
            print(f"[{now()}] Test set '{name}' is empty, skipping.")
            prepared_test_sets[name] = None
            continue
        
        X_test = df[model_feature_names].values.astype('float32')
        y_test = df[target_names].values.astype('float32')
        
        X_test_scaled = scaler_X.transform(X_test)
        # Note: y_test is NOT scaled here. The evaluator will use the scaler for predictions.
        
        prepared_test_sets[name] = {'X': X_test_scaled, 'y': y_test, 'feature_names': model_feature_names}
        print(f"[{now()}] Prepared test set '{name}' with {len(df)} samples.")
        
    return train_data, prepared_test_sets, scaler_X, scaler_y, model_feature_names


def main():
    parser = argparse.ArgumentParser(description='Unified pipeline for alloy property prediction.')
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the raw data file with labels and temperature.')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save all results')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--target_columns', type=str, nargs='+', required=True, help='List of target column names')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--number_of_train_samples', type=int, default=2000, help='Number of training samples')
    # Model and Training arguments
    parser.add_argument('--validation_size', type=float, default=0.2, help='Proportion of training set to use for validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='Hidden dimensions for prediction network')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10000, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training (e.g., "cpu", "cuda")')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use DataParallel for multi-GPU training')
    
    # Feature generation arguments (from feature_engineering/example.py)
    parser.add_argument('--force_regenerate', action='store_true', help='Force regeneration of features even if they exist.')
    parser.add_argument('--use_composition_embedding', action='store_true', help="Whether to use elemental composition embedding vectors")
    parser.add_argument('--use_process_embedding', action='store_true', help="Whether to use process description embedding vectors")
    parser.add_argument('--use_joint_composition_process_embedding', action='store_true', help="Whether to use joint composition+process BERT embedding")
    parser.add_argument('--use_composition_feature', action='store_true', help="Whether to use elemental composition features")
    parser.add_argument('--use_feature1', action='store_true', help="Whether to use Feature1 features")
    parser.add_argument('--use_feature2', action='store_true', help="Whether to use Feature2 features")
    parser.add_argument('--other_features_name', type=str, nargs='+', default=None, help="Custom feature column names")
    parser.add_argument('--use_temperature', action='store_true', help="Whether to use temperature as a feature")
    
    # Feature projection dimension arguments
    parser.add_argument('--emb_hidden_dim', type=int, default=0)
    parser.add_argument('--feature1_hidden_dim', type=int, default=0)
    parser.add_argument('--feature2_hidden_dim', type=int, default=0)
    parser.add_argument('--other_features_hidden_dim', type=int, default=0)
    
    args = parser.parse_args()
    
    os.makedirs(args.result_dir, exist_ok=True)
    feature_gen_dir = args.feature_dir
    os.makedirs(feature_gen_dir, exist_ok=True)

    # ================= 数据处理部分（前置，集中） =================
    print(f"[{now()}] 1. Generating features...")
    # 1. 生成特征
    full_data = generate_features(
        data_path=args.data_path,
        feature_dir=feature_gen_dir,
        target_columns=args.target_columns,
        args=args
    )

    print(f"[{now()}] 2. Custom data splitting...")
    # 2. 自定义数据划分
    train_df, all_test_df, low_T_test_df, low_outliers_df, whiskers_low_T_test_df = custom_data_split(
        data=full_data,
        number_of_train_samples=args.number_of_train_samples,
        target_columns=args.target_columns,
        random_state=args.random_state
    )
    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=args.validation_size,
        random_state=args.random_state
    )

    # # 输出各数据集划分 shape
    # print(f"[{now()}] Train set shape: {train_df.shape}")
    # print(f"[{now()}] Validation set shape: {val_df.shape}")
    # print(f"[{now()}] All test set shape: {all_test_df.shape}")
    # print(f"[{now()}] Low-T test set shape: {low_T_test_df.shape}")
    # print(f"[{now()}] Low-outliers test set shape: {low_outliers_df.shape}")
    # print(f"[{now()}] Whiskers Low-T test set shape: {whiskers_low_T_test_df.shape}")

    print(f"[{now()}] 3. Selecting features for the model...")
    # 3. 特征选择
    _, _, model_feature_names = select_features(
        data=train_df,
        target_columns=args.target_columns,
        use_process_embedding=args.use_process_embedding,
        use_joint_composition_process_embedding=args.use_joint_composition_process_embedding,
        use_composition_embedding=args.use_composition_embedding,
        use_composition_feature=args.use_composition_feature,
        use_feature1=args.use_feature1,
        use_feature2=args.use_feature2,
        other_features_name=args.other_features_name,
        use_temperature=args.use_temperature
    )

    # 4. 标准化与 shape 输出
    # 训练集
    X_train = train_df[model_feature_names].values.astype('float32')
    y_train = train_df[args.target_columns].values.astype('float32')
    print(f"[{now()}] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    scaler_X = StandardScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    joblib.dump(scaler_X, os.path.join(args.result_dir, 'scaler_X.pkl'))
    scaler_y = StandardScaler().fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    joblib.dump(scaler_y, os.path.join(args.result_dir, 'scaler_y.pkl'))
    train_data = {'X': X_train_scaled, 'y': y_train_scaled}

    # 验证集
    X_val = val_df[model_feature_names].values.astype('float32')
    y_val = val_df[args.target_columns].values.astype('float32')
    print(f"[{now()}] X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    val_data = {'X': X_val_scaled, 'y': y_val_scaled}

    # 测试集
    X_test = all_test_df[model_feature_names].values.astype('float32')
    y_test_unscaled = all_test_df[args.target_columns].values.astype('float32')
    print(f"[{now()}] X_test shape: {X_test.shape}, y_test shape: {y_test_unscaled.shape}")
    X_test_scaled = scaler_X.transform(X_test)
    test_data = {'X': X_test_scaled, 'y': y_test_unscaled, 'feature_names': model_feature_names}

    # 其它测试集 shape 输出（可选）
    for name, df in zip([
        'low_T_test', 'low_outliers', 'whiskers_low_T_test'],
        [low_T_test_df, low_outliers_df, whiskers_low_T_test_df]):
        if not df.empty:
            print(f"[{now()}] {name} X shape: {df[model_feature_names].shape}, y shape: {df[args.target_columns].shape}")

    # ================= 训练与评估部分 =================
    # 5. Build Model
    feature_type_config = {}
    feature_hidden_dims = {}
    if args.use_composition_embedding or args.use_joint_composition_process_embedding:
        feature_type_config['emb'] = ['emb']
        feature_hidden_dims['emb'] = args.emb_hidden_dim
    if args.use_feature1:
        feature_type_config['feature1'] = ['feature1']
        feature_hidden_dims['feature1'] = args.feature1_hidden_dim
    if args.use_feature2:
        feature_type_config['feature2'] = ['feature2']
        feature_hidden_dims['feature2'] = args.feature2_hidden_dim
    if args.other_features_name:
        feature_type_config['other_features'] = args.other_features_name
        feature_hidden_dims['other_features'] = args.other_features_hidden_dim

    model = AlloyNN(
        column_names=model_feature_names,
        output_dim=len(args.target_columns),
        feature_type_config=feature_type_config,
        feature_hidden_dims=feature_hidden_dims,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    )
    model_structure_path = os.path.join(args.result_dir, 'model_structure.txt')
    model.print_structure(file_path=model_structure_path)
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"[{now()}] Using {torch.cuda.device_count()} GPUs for DataParallel training.")
        model = torch.nn.DataParallel(model)

    # 6. Train Model
    print(f"[{now()}] Creating trainer...")
    training_params = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'early_stopping_patience': args.patience
    }
    trainer = TrainerFactory.create_trainer(
        trainer_type='nn',
        model=model,
        result_dir=args.result_dir,
        model_name='alloy_nn_unified',
        target_names=args.target_columns,
        train_data={'X': train_data['X'], 'y': train_data['y']},
        val_data=val_data, # Pass validation data to the trainer
        training_params=training_params
    )
    print(f"[{now()}] Starting training...")
    history = trainer.train(num_epochs=args.epochs)
    print(f"[{now()}] Training finished.")

    # 绘制训练/验证损失和R2曲线
    plot_dir = os.path.join(args.result_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "loss_r2_curve.png")
    plot_loss_r2_curves(
        history=history,
        save_path=plot_path,
        title="Training/Validation Loss & R2 Curves"
    )
    print(f"[{now()}] Loss and R2 curves saved to {plot_path}")

    # 7. Evaluate Model
    print(f"[{now()}] Evaluating model on test sets...")
    evaluator = EvaluatorFactory.create_evaluator(
        evaluator_type='alloys',
        result_dir=args.result_dir,
        model_name='alloy_nn_unified',
        target_names=args.target_columns,
        target_scaler=scaler_y
    )
    best_model_filename = f"{trainer.model_name}_best.pt"
    best_model_path = os.path.join(trainer.result_dir, "checkpoints", best_model_filename)
    if os.path.exists(best_model_path):
        print(f"[{now()}] Loading best model from {best_model_path}")
        loaded_object = torch.load(best_model_path, map_location=args.device)
        model_state_dict_to_load = loaded_object.get('model_state_dict', loaded_object)
        current_model_is_parallel = isinstance(model, torch.nn.DataParallel)
        saved_state_has_module_prefix = any(key.startswith('module.') for key in model_state_dict_to_load.keys())
        if current_model_is_parallel and not saved_state_has_module_prefix:
            model.module.load_state_dict(model_state_dict_to_load)
        elif not current_model_is_parallel and saved_state_has_module_prefix:
            final_state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in model_state_dict_to_load.items()])
            model.load_state_dict(final_state_dict)
        else:
            model.load_state_dict(model_state_dict_to_load)
    else:
        print(f"[{now()}] Best model checkpoint not found. Using model from last epoch.")
    model.to(args.device)
    model.eval()
    # 只用准备好的数据进行评估
    test_sets_to_evaluate = {
        "all_tests": all_test_df,
        "low_T_tests": low_T_test_df,
        "low_outliers": low_outliers_df,
        "whiskers_low_T_tests": whiskers_low_T_test_df
    }
    for name, df in test_sets_to_evaluate.items():
        if df.empty:
            print(f"--- Skipping evaluation for '{name}' as it is empty. ---")
            continue
        print(f"--- Evaluating on: {name} ---")
        X_test_current = df[model_feature_names].values.astype('float32')
        y_test_current_unscaled = df[args.target_columns].values.astype('float32')
        X_test_current_scaled = scaler_X.transform(X_test_current)
        current_test_data = {
            'X': X_test_current_scaled, 
            'y': y_test_current_unscaled, 
            'feature_names': model_feature_names
        }
        evaluator.evaluate_model(
            model=model,
            train_data=train_data, # Not re-evaluating on train data
            val_data=val_data,
            test_data=current_test_data,
            save_prefix=f'best_model_{name}_',
            feature_names=model_feature_names
        )
    print(f"[{now()}] Pipeline finished. Results are in {args.result_dir}")


if __name__ == '__main__':
    main()
"""
python src/pipelines/unified_pipeline.py \
    --data_path datasets/Ni_alloys/NiAlloy-Tm-CALPHAD-693K-descriptor_withlabel.csv \
    --feature_dir Features/Ni_alloys/all_features_withlabel \
    --result_dir output/results/Ni_alloys/run_samples_composition_feature/5 \
    --target_columns "Liquidus Temperature (℃)" \
    --number_of_train_samples 5 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_composition_feature \
    --hidden_dims 2048 1024 512 256 128 \
    --use_multi_gpu \
    --other_features_hidden_dim 0 \
    --patience 100

"""