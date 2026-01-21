import os
import logging
import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    """
    设置全局随机种子以保证结果可复现
    Set global random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch 种子设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性计算设置（会稍微降低性能，但保证结果一致）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 记录日志
    logging.info(f"已设置全局随机种子: {seed}")

def setup_logging(log_name=None, log_dir=None):
    """
    设置日志配置
    
    Args:
        log_name: 日志文件名
        log_dir: 日志目录
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name) if log_name else None
    else:
        log_path = None
    
    # 创建日志格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 创建文件处理器（如果指定了日志文件）
    handlers = [console_handler]
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)

def save_features(features, target, feature_dir, prefix="", ids=None, id_column_name="ID"):
    """
    保存特征和目标变量
    
    Args:
        features: 特征DataFrame
        target: 目标变量DataFrame
        feature_dir: 特征存储目录
        prefix: 文件名前缀
        ids: 可选，样本ID序列或DataFrame（与features按行对齐）
        id_column_name: ID列名
    """
    ensure_dir(feature_dir)
    
    # # 保存特征（仅特征，不包含目标列，避免与 target.csv 重复）
    # features_path = os.path.join(feature_dir, f"{prefix}features.csv")
    # features.to_csv(features_path, index=False, encoding='utf-8')
    
    # # 保存目标变量
    # target_path = os.path.join(feature_dir, f"{prefix}target.csv")
    # target.to_csv(target_path, index=False, encoding='utf-8')

    # 额外保存与特征对齐的ID，便于后续溯源（不参与训练特征）
    if ids is not None:
        # 将ids统一为DataFrame
        if not isinstance(ids, pd.DataFrame):
            ids_df = pd.DataFrame({id_column_name: ids})
        else:
            ids_df = ids.copy()
            # 若列名不包含指定id列名，则重命名为该列名
            if ids_df.shape[1] == 1 and ids_df.columns[0] != id_column_name:
                ids_df.columns = [id_column_name]

        ids_path = os.path.join(feature_dir, f"{prefix}ids.csv")
        ids_df.to_csv(ids_path, index=False, encoding='utf-8')

        # 便于排查问题：额外输出带ID的完整表
        features_with_id_path = os.path.join(feature_dir, f"{prefix}features_with_id.csv")
        features_with_id = pd.concat([ids_df.reset_index(drop=True),
                                      features.reset_index(drop=True),
                                      target.reset_index(drop=True)], axis=1)
        features_with_id.to_csv(features_with_id_path, index=False, encoding='utf-8')

        # 额外输出带ID的target，便于直接按ID定位目标
        target_with_id_path = os.path.join(feature_dir, f"{prefix}target_with_id.csv")
        target_with_id = pd.concat([ids_df.reset_index(drop=True),
                                    target.reset_index(drop=True)], axis=1)
        target_with_id.to_csv(target_with_id_path, index=False, encoding='utf-8')

def standardize_features(features, target_cols=None):
    """
    标准化特征
    
    Args:
        features: 特征DataFrame
        target_cols: 目标变量列名列表
    
    Returns:
        tuple: (标准化后的特征, 标准化器)
    """
    scaler = StandardScaler()
    
    # 确定需要标准化的列
    numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
    if target_cols:
        numeric_cols = [col for col in numeric_cols if col not in target_cols]
    
    if len(numeric_cols) > 0:
        features[numeric_cols] = scaler.fit_transform(features[numeric_cols])
    
    return features, scaler

def save_feature_names(features, feature_dir, feature_info=None):
    """
    保存特征名称
    
    Args:
        features: 特征DataFrame
        feature_dir: 特征存储目录
        feature_info: 特征信息字典
    """
    ensure_dir(feature_dir)
    feature_names_path = os.path.join(feature_dir, "feature_names.txt")
    
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        f.write("# === 使用的特征列表 ===\n\n")
        f.write(f"# 总特征数: {len(features.columns)}\n\n")
        
        if feature_info:
            f.write("# 特征类型使用情况:\n")
            for key, value in feature_info.items():
                f.write(f"# - {key}: {'是' if value else '否'}\n")
            f.write("\n")
        
        for col in features.columns:
            f.write(f"{col}\n") 