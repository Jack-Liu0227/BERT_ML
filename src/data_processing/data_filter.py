#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data filter for alloy data
合金数据过滤器
"""

import pandas as pd
import logging
from .data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFilter(DataProcessor):
    """
    Class for filtering alloy data
    合金数据过滤器类
    """
    
    def __init__(self, root_dir=None):
        """
        Initialize the data filter
        初始化数据过滤器
        
        Args:
            root_dir: Project root directory, defaults to current directory
                     项目根目录，默认为当前目录
        """
        super().__init__(root_dir)
    
    def filter_by_temperature(self, df, temp_column, min_temp=None, max_temp=None):
        """
        Filter data by temperature range
        按温度范围过滤数据
        
        Args:
            df: Input DataFrame
                输入数据框
            temp_column: Name of the temperature column
                        温度列名
            min_temp: Minimum temperature (optional)
                     最小温度（可选）
            max_temp: Maximum temperature (optional)
                     最大温度（可选）
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
                            过滤后的数据框
        """
        logger.info(f"Filtering data by temperature range: {min_temp}°C to {max_temp}°C")
        
        # Create a copy of the input DataFrame
        filtered_df = df.copy()
        
        # Apply temperature filters
        if min_temp is not None:
            filtered_df = filtered_df[filtered_df[temp_column] >= min_temp]
            logger.info(f"Applied minimum temperature filter: {min_temp}°C")
            
        if max_temp is not None:
            filtered_df = filtered_df[filtered_df[temp_column] <= max_temp]
            logger.info(f"Applied maximum temperature filter: {max_temp}°C")
        
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df
    
    def filter_by_composition(self, df, element_columns, min_composition=None, max_composition=None):
        """
        Filter data by element composition range
        按元素成分范围过滤数据
        
        Args:
            df: Input DataFrame
                输入数据框
            element_columns: List of element column names
                           元素列名列表
            min_composition: Minimum composition percentage (optional)
                           最小成分百分比（可选）
            max_composition: Maximum composition percentage (optional)
                           最大成分百分比（可选）
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
                            过滤后的数据框
        """
        logger.info("Filtering data by element composition")
        
        # Create a copy of the input DataFrame
        filtered_df = df.copy()
        
        # Apply composition filters for each element
        for element_col in element_columns:
            if min_composition is not None:
                filtered_df = filtered_df[filtered_df[element_col] >= min_composition]
                logger.info(f"Applied minimum composition filter for {element_col}: {min_composition}%")
                
            if max_composition is not None:
                filtered_df = filtered_df[filtered_df[element_col] <= max_composition]
                logger.info(f"Applied maximum composition filter for {element_col}: {max_composition}%")
        
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df
    
    def filter_by_property(self, df, property_column, min_value=None, max_value=None):
        """
        Filter data by property value range
        按性能值范围过滤数据
        
        Args:
            df: Input DataFrame
                输入数据框
            property_column: Name of the property column
                           性能列名
            min_value: Minimum property value (optional)
                      最小性能值（可选）
            max_value: Maximum property value (optional)
                      最大性能值（可选）
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
                            过滤后的数据框
        """
        logger.info(f"Filtering data by property: {property_column}")
        
        # Create a copy of the input DataFrame
        filtered_df = df.copy()
        
        # Apply property filters
        if min_value is not None:
            filtered_df = filtered_df[filtered_df[property_column] >= min_value]
            logger.info(f"Applied minimum value filter: {min_value}")
            
        if max_value is not None:
            filtered_df = filtered_df[filtered_df[property_column] <= max_value]
            logger.info(f"Applied maximum value filter: {max_value}")
        
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df 