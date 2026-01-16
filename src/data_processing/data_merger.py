#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data merger for combining alloy datasets
合金数据集合并器
"""

import pandas as pd
import logging
from .data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataMerger(DataProcessor):
    """
    Class for merging alloy datasets
    合金数据集合并器类
    """
    
    def __init__(self, root_dir=None):
        """
        Initialize the data merger
        初始化数据合并器
        
        Args:
            root_dir: Project root directory, defaults to current directory
                     项目根目录，默认为当前目录
        """
        super().__init__(root_dir)
    
    def merge_datasets(self, df1, df2, merge_columns, how='inner', source_columns=None):
        """
        Merge two datasets based on matching columns
        基于匹配列合并两个数据集
        
        Args:
            df1: First DataFrame
                第一个数据框
            df2: Second DataFrame
                第二个数据框
            merge_columns: List of columns to match for merging
                         用于匹配合并的列名列表
            how: Type of merge ('inner', 'outer', 'left', 'right')
                合并类型（'inner', 'outer', 'left', 'right'）
            source_columns: Dictionary mapping source names to column lists (optional)
                          源名称到列列表的映射字典（可选）
            
        Returns:
            pandas.DataFrame: Merged DataFrame
                            合并后的数据框
        """
        logger.info(f"Merging datasets using columns: {merge_columns}")
        logger.info(f"First dataset size: {len(df1)} rows")
        logger.info(f"Second dataset size: {len(df2)} rows")
        
        # Add source identifiers if specified
        if source_columns:
            for source, columns in source_columns.items():
                if source == 'df1':
                    for col in columns:
                        df1[col] = f"{source}_{df1[col]}"
                elif source == 'df2':
                    for col in columns:
                        df2[col] = f"{source}_{df2[col]}"
        
        # Perform merge
        merged_df = pd.merge(df1, df2, on=merge_columns, how=how)
        
        logger.info(f"Merged dataset size: {len(merged_df)} rows")
        return merged_df
    
    def merge_hea_datasets(self, df1, df2, merge_columns=None):
        """
        Merge two high entropy alloy datasets
        合并两个高熵合金数据集
        
        Args:
            df1: First HEA DataFrame
                第一个高熵合金数据框
            df2: Second HEA DataFrame
                第二个高熵合金数据框
            merge_columns: List of columns to match for merging (optional)
                         用于匹配合并的列名列表（可选）
            
        Returns:
            pandas.DataFrame: Merged DataFrame
                            合并后的数据框
        """
        # Default merge columns for HEA datasets
        if merge_columns is None:
            merge_columns = ['composition', 'processing_method']
        
        # Define source columns for HEA datasets
        source_columns = {
            'df1': ['composition', 'processing_method'],
            'df2': ['composition', 'processing_method']
        }
        
        return self.merge_datasets(df1, df2, merge_columns, source_columns=source_columns)
    
    def merge_al_datasets(self, df1, df2, merge_columns=None):
        """
        Merge two aluminum alloy datasets
        合并两个铝合金数据集
        
        Args:
            df1: First aluminum alloy DataFrame
                第一个铝合金数据框
            df2: Second aluminum alloy DataFrame
                第二个铝合金数据框
            merge_columns: List of columns to match for merging (optional)
                         用于匹配合并的列名列表（可选）
            
        Returns:
            pandas.DataFrame: Merged DataFrame
                            合并后的数据框
        """
        # Default merge columns for aluminum alloy datasets
        if merge_columns is None:
            merge_columns = ['alloy_composition', 'heat_treatment']
        
        # Define source columns for aluminum alloy datasets
        source_columns = {
            'df1': ['alloy_composition', 'heat_treatment'],
            'df2': ['alloy_composition', 'heat_treatment']
        }
        
        return self.merge_datasets(df1, df2, merge_columns, source_columns=source_columns) 