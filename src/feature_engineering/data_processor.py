import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器，用于处理数据加载和预处理"""
    
    def __init__(self, data_path=None):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        """
        加载数据并进行预处理
        
        Returns:
            DataFrame: 处理后的数据
        """
        if not self.data_path:
            raise ValueError("未指定数据文件路径")
            
        logger.info(f"正在从指定路径加载数据: {self.data_path}...")
        
        # 检查文件是否存在
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 尝试不同的编码方式读取数据
        encodings = ['utf-8']
        for encoding in encodings:
            try:
                self.data = pd.read_csv(self.data_path, encoding=encoding)
                logger.info(f"成功使用 {encoding} 编码读取数据")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"使用 {encoding} 编码读取数据时发生错误: {str(e)}")
                continue
        else:
            raise ValueError("无法使用任何编码方式读取数据文件")
        
        logger.info(f"数据加载完成，共{len(self.data)}条记录")
        
        return self.data
    
    def validate_columns(self, required_cols):
        """
        验证必要列是否存在
        
        Args:
            required_cols: 必要列名列表
        
        Returns:
            bool: 验证是否通过
        """
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少必要的列: {missing_cols}")
        return True
    
    def handle_missing_values(self):
        """
        处理缺失值
        
        Returns:
            DataFrame: 处理后的数据
        """
        logger.info("处理缺失值...")
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            logger.info("发现缺失值:")
            logger.info(missing_values[missing_values > 0])
            # 对于数值列，使用中位数填充
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
            # 对于非数值列，使用众数填充
            non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
            self.data[non_numeric_cols] = self.data[non_numeric_cols].fillna(self.data[non_numeric_cols].mode().iloc[0])
        
        return self.data
    
    def validate_data_types(self):
        """
        验证数据类型
        
        Returns:
            DataFrame: 处理后的数据
        """
        logger.info("验证数据类型...")
        for col in self.data.columns:
            if col.endswith(('(at%)', '(wt%)')):
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    try:
                        self.data[col] = pd.to_numeric(self.data[col])
                    except:
                        raise ValueError(f"列 {col} 无法转换为数值类型")
        
        return self.data
    
    def process(self, required_cols=None):
        """
        执行完整的数据处理流程
        
        Args:
            required_cols: 必要列名列表
        
        Returns:
            DataFrame: 处理后的数据
        """
        # 加载数据
        self.load_data()
        
        # 验证必要列
        if required_cols:
            self.validate_columns(required_cols)
        
        # 处理缺失值
        self.handle_missing_values()
        
        # 验证数据类型
        self.validate_data_types()
        
        logger.info("数据基本信息:")
        logger.info(self.data.info())
        logger.info("数据前几行:")
        logger.info(self.data.head())
        
        return self.data 