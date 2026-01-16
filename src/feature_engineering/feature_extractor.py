import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特征提取器，用于处理不同类型的特征提取"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.feature1_features = None
        self.feature2_features = None
        self.composition_features = None
        self.other_features_columns=None
    def extract_feature1_features(self, data):
        """
        提取ACTA元数据特征
        
        Args:
            data: 原始数据DataFrame
        
        Returns:
            DataFrame: Feature1特征
        """
        logger.info("正在提取Feature1特征...")
        
        # Feature1相关特征列名
        # feature_columns = [f"Feature1_{i}" for i in range(19)]
        
        # 检查哪些列存在于数据中
        available_columns = [col for col in  data.columns if col.startswith('Feature1_')]
        
        if not available_columns:
            logger.warning("警告: 未找到任何Feature1特征列")
            return pd.DataFrame()
        
        # 提取Feature1特征
        feature_features = data[available_columns].copy()
        
        for col in feature_features.columns:
            if not pd.api.types.is_numeric_dtype(feature_features[col]):
                try:
                    feature_features[col] = pd.to_numeric(feature_features[col], errors='coerce')
                except:
                    feature_features = feature_features.drop(columns=[col])
        
        # 填充缺失值
        if feature_features.isnull().any().any():
            feature_features = feature_features.fillna(feature_features.mean())
        
        logger.info(f"Feature1元数据特征提取完成，共{len(feature_features.columns)}个特征")
        return feature_features
    
    def extract_feature2_features(self, data):
        """
        提取ACTA元数据特征
        
        Args:
            data: 原始数据DataFrame
        
        Returns:
            DataFrame: Feature1特征
        """
        logger.info("正在提取Feature2特征...")
        
        # Feature1相关特征列名
        # feature_columns = [f"Feature1_{i}" for i in range(19)]
        
        # 检查哪些列存在于数据中
        available_columns = [col for col in  data.columns if col.startswith('Feature2_')]
        
        if not available_columns:
            logger.warning("警告: 未找到任何Feature1特征列")
            return pd.DataFrame()
        
        feature_features = data[available_columns].copy()
        
        for col in feature_features.columns:
            if not pd.api.types.is_numeric_dtype(feature_features[col]):
                try:
                    feature_features[col] = pd.to_numeric(feature_features[col], errors='coerce')
                except:
                    feature_features = feature_features.drop(columns=[col])
        
        # 填充缺失值
        if feature_features.isnull().any().any():
            feature_features = feature_features.fillna(feature_features.mean())
        
        logger.info(f"Feature2元数据特征提取完成，共{len(feature_features.columns)}个特征")
        return feature_features
    
    def extract_other_features(self, data, columns=None):
        """
        Extract user-specified features from the DataFrame.
        
        Args:
            data: 原始数据DataFrame
            columns: list, 指定要提取的特征列名
        
        Returns:
            DataFrame: 提取的自定义特征
        """
        logger.info("正在提取自定义特征列...")
        if columns is None:
            logger.warning("未指定自定义特征列名，返回空DataFrame")
            return pd.DataFrame()
        available_columns = [col for col in columns if col in data.columns]
        if not available_columns:
            logger.warning("警告: 未找到任何自定义特征列")
            return pd.DataFrame()
        feature_features = data[available_columns].copy()
        for col in feature_features.columns:
            if not pd.api.types.is_numeric_dtype(feature_features[col]):
                try:
                    feature_features[col] = pd.to_numeric(feature_features[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"列 {col} 转换为数值型失败，已删除: {e}")
                    feature_features = feature_features.drop(columns=[col])
        # Fill missing values with mean
        if feature_features.isnull().any().any():
            feature_features = feature_features.fillna(feature_features.mean())
        logger.info(f"自定义特征提取完成，共{len(feature_features.columns)}个特征")
        return feature_features

    def extract_composition_features(self, data):
        """
        提取元素成分特征，支持(at%)和(wt%)两种类型

        Args:
            data: 原始数据DataFrame

        Returns:
            DataFrame: 元素成分特征（包含所有以(at%)或(wt%)结尾的列）
        """
        # 日志：开始提取元素成分比例特征
        logger.info("正在提取元素成分比例特征 (支持 at% 和 wt%) ...")
        
        # 查找所有以(at%)或(wt%)结尾的列
        available_columns = [col for col in data.columns if col.endswith('(at%)') or col.endswith('(wt%)')]
        
        # 如果没有找到相关列，发出警告并返回空DataFrame
        if not available_columns:
            logger.warning("警告: 未找到任何元素成分特征列 (at% 或 wt%)")
            return pd.DataFrame()
        
        # 复制相关列
        feature_features = data[available_columns].copy()
        
        # 确保所有列为数值型，无法转换的列将被丢弃
        for col in feature_features.columns:
            if not pd.api.types.is_numeric_dtype(feature_features[col]):
                try:
                    feature_features[col] = pd.to_numeric(feature_features[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"列 {col} 转换为数值型失败，已删除: {e}")
                    feature_features = feature_features.drop(columns=[col])
        
        # 填充缺失值，使用均值填充
        if feature_features.isnull().any().any():
            feature_features = feature_features.fillna(feature_features.mean())
        
        # 日志：特征提取完成
        logger.info(f"元素成分比例特征提取完成，共{len(feature_features.columns)}个特征 (at%+wt%)")
        return feature_features
        
    def extract_temperature_features(self, data):
        """
        提取温度特征，支持任何包含Temperature的列
        
        Args:
            data: 原始数据DataFrame
        
        Returns:
            DataFrame: 温度特征
        """
        # 查找所有包含Temperature的列
        temp_columns = [col for col in data.columns if 'temperature'== col.split("(")[0].lower()]
        
        if temp_columns:
            logger.info(f"找到以下温度特征列: {temp_columns}")
            temp_features = data[temp_columns].copy()
            
            # 确保所有列为数值型
            for col in temp_features.columns:
                if not pd.api.types.is_numeric_dtype(temp_features[col]):
                    try:
                        temp_features[col] = pd.to_numeric(temp_features[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"列 {col} 转换为数值型失败，已删除: {e}")
                        temp_features = temp_features.drop(columns=[col])
            
            # 填充缺失值
            if temp_features.isnull().any().any():
                temp_features = temp_features.fillna(temp_features.mean())
                
            return temp_features
            
        logger.warning("未找到任何温度特征列")
        return pd.DataFrame()
    
    def create_formula(self, data):
        """
        从元素成分列创建化学式
        
        Args:
            data: 包含元素成分的DataFrame
        
        Returns:
            Series: 化学式列
        """
        # 找出所有元素列(以(at%)结尾的列)
        element_cols = [col for col in data.columns if col.endswith(('(at%)', '(wt%)'))]
        
        if not element_cols:
            raise ValueError("数据中未找到元素成分列(应以(at%)结尾)")
        
        # 创建化学式列
        formulas = []
        for idx, row in data.iterrows():
            try:
                formula_parts = []
                for el_col in element_cols:
                    at_percent = row[el_col]
                    if at_percent > 0:  # 只处理含量大于0的元素
                        element = el_col.split('(')[0]
                        formula_parts.append(f"{element}{at_percent}")
                formulas.append("".join(formula_parts))
            except Exception as e:
                logger.error(f"处理第 {idx} 行时出错: {str(e)}")
                formulas.append("")  # 出错时使用空字符串
        
        return pd.Series(formulas, index=data.index, name='Formula') 