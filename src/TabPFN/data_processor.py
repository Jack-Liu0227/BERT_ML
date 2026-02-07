"""
Data Processor for TabPFN
数据预处理模块

Handles data loading, cleaning, and preprocessing for TabPFN model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabPFNDataProcessor:
    """TabPFN 数据预处理器 / TabPFN Data Processor"""
    
    def __init__(self, config: Dict):
        """
        初始化数据处理器
        
        Args:
            config: 数据集配置字典
        """
        self.config = config
        self.scaler = StandardScaler()
        self.data = None
        self.feature_names = None
        
    def load_data(self, base_path: str = ".") -> pd.DataFrame:
        """
        加载数据集
        Load dataset
        
        Args:
            base_path: 项目根目录路径
            
        Returns:
            加载的数据框 / Loaded DataFrame
        """
        data_path = Path(base_path) / self.config["raw_data"]
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading data from: {data_path}")
        self.data = pd.read_csv(data_path)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        return self.data
    
    def prepare_features_and_targets(
        self, 
        target_col: str,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和目标变量
        Prepare features and targets
        
        Args:
            target_col: 目标列名
            drop_na: 是否删除缺失值
            
        Returns:
            (特征DataFrame, 目标Series)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # 检查目标列是否存在
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # 选择特征列
        feature_cols = self.config["feature_cols"]
        
        # 检查哪些特征列存在
        available_features = [col for col in feature_cols if col in self.data.columns]
        missing_features = [col for col in feature_cols if col not in self.data.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} feature columns not found: {missing_features[:5]}...")
        
        print(f"Using {len(available_features)} features")
        self.feature_names = available_features
        
        # 提取特征和目标
        X = self.data[available_features].copy()
        y = self.data[target_col].copy()
        
        # 保留原始 ID（如果存在 'ID' 列）
        if 'ID' in self.data.columns:
            ids = self.data['ID'].copy()
        else:
            # 如果没有 ID 列，使用索引+1作为 ID
            ids = pd.Series(range(1, len(self.data) + 1), name='ID')
        
        # 创建临时 DataFrame 以便一起处理
        temp_df = X.copy()
        temp_df['_target'] = y
        temp_df['_id'] = ids
        
        # 处理缺失值
        if drop_na:
            # 记录删除前的数据量
            n_before = len(temp_df)
            
            # 删除特征或目标中有缺失值的行
            temp_df = temp_df.dropna(subset=available_features + ['_target'])
            
            n_after = len(temp_df)
            if n_before > n_after:
                print(f"Dropped {n_before - n_after} rows with missing values")
        else:
            # 用0填充特征缺失值
            temp_df[available_features] = temp_df[available_features].fillna(0)
            # 用均值填充目标缺失值
            temp_df['_target'] = temp_df['_target'].fillna(temp_df['_target'].mean())
        
        # 按 ID 排序，确保不同运行时相同 random_state 产生相同的分割
        # Sort by ID to ensure same random_state produces same split across runs
        temp_df = temp_df.sort_values('_id').reset_index(drop=True)
        
        # 分离特征、目标和 ID
        X = temp_df[available_features].copy()
        y = temp_df['_target'].copy()
        ids = temp_df['_id'].copy()
        
        print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Data sorted by ID for consistent train/test split")
        
        return X, y, ids
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ids: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        分割训练集和测试集
        Split train and test sets
        
        Args:
            X: 特征
            y: 目标
            ids: 原始 ID
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            (X_train, X_test, y_train, y_test, ids_train, ids_test)
        """
        test_size = test_size or self.config.get("test_size", 0.3)
        random_state = random_state or self.config.get("random_state", 42)
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids,
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Train/Test split: {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
        
        return X_train, X_test, y_train, y_test, ids_train, ids_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        标准化特征
        Scale features using StandardScaler
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            
        Returns:
            (scaled_X_train, scaled_X_test)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def get_full_pipeline(
        self,
        target_col: str,
        base_path: str = ".",
        scale: bool = True,
        drop_na: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整的数据处理流程
        Full data processing pipeline
        
        Args:
            target_col: 目标列名
            base_path: 项目根目录
            scale: 是否标准化特征
            drop_na: 是否删除缺失值
            
        Returns:
            (X_train, X_test, y_train, y_test, ids_train, ids_test)
        """
        print(f"\n{'='*60}")
        print(f"Processing {self.config['description']}")
        print(f"Target: {target_col}")
        print(f"{'='*60}")
        
        # 加载数据
        self.load_data(base_path)
        
        # 准备特征和目标
        X, y, ids = self.prepare_features_and_targets(target_col, drop_na=drop_na)
        
        # 分割数据
        X_train, X_test, y_train, y_test, ids_train, ids_test = self.split_data(X, y, ids)
        
        # 标准化特征
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
            print("Features scaled using StandardScaler")
        else:
            X_train = X_train.values
            X_test = X_test.values
        
        y_train = y_train.values
        y_test = y_test.values
        ids_train = ids_train.values
        ids_test = ids_test.values
        
        return X_train, X_test, y_train, y_test, ids_train, ids_test


def create_data_processor(alloy_type: str, config_dict: Dict) -> TabPFNDataProcessor:
    """
    创建数据处理器
    Create data processor
    
    Args:
        alloy_type: 合金类型
        config_dict: 配置字典
        
    Returns:
        数据处理器实例
    """
    config = config_dict.get(alloy_type)
    if config is None:
        raise ValueError(f"Unknown alloy type: {alloy_type}")
    
    return TabPFNDataProcessor(config)
