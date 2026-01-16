#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for alloy data processing
合金数据处理工具函数
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_composition(df, element_columns):
    """
    Normalize element compositions to sum to 100%
    将元素成分归一化到100%
    
    Args:
        df: Input DataFrame
            输入数据框
        element_columns: List of element column names
                       元素列名列表
        
    Returns:
        pandas.DataFrame: DataFrame with normalized compositions
                        归一化成分后的数据框
    """
    logger.info("Normalizing element compositions")
    
    # Create a copy of the input DataFrame
    normalized_df = df.copy()
    
    # Calculate sum of compositions for each row
    composition_sum = normalized_df[element_columns].sum(axis=1)
    
    # Normalize each element column
    for col in element_columns:
        normalized_df[col] = (normalized_df[col] / composition_sum) * 100
    
    logger.info("Composition normalization completed")
    return normalized_df

def calculate_entropy(df, element_columns):
    """
    Calculate configurational entropy for high entropy alloys
    计算高熵合金的构型熵
    
    Args:
        df: Input DataFrame
            输入数据框
        element_columns: List of element column names
                       元素列名列表
        
    Returns:
        pandas.Series: Configurational entropy values
                     构型熵值
    """
    logger.info("Calculating configurational entropy")
    
    # Convert compositions to mole fractions
    compositions = df[element_columns].div(100)
    
    # Calculate entropy for each row
    entropy = -np.sum(compositions * np.log(compositions), axis=1)
    
    logger.info("Entropy calculation completed")
    return entropy

def validate_composition(df, element_columns, tolerance=0.1):
    """
    Validate that element compositions sum to 100%
    验证元素成分总和是否为100%
    
    Args:
        df: Input DataFrame
            输入数据框
        element_columns: List of element column names
                       元素列名列表
        tolerance: Allowed deviation from 100% (default: 0.1)
                  允许偏离100%的误差（默认：0.1）
        
    Returns:
        pandas.Series: Boolean mask of valid compositions
                     有效成分的布尔掩码
    """
    logger.info("Validating element compositions")
    
    # Calculate sum of compositions
    composition_sum = df[element_columns].sum(axis=1)
    
    # Check if sum is within tolerance of 100%
    is_valid = abs(composition_sum - 100) <= tolerance
    
    logger.info(f"Found {is_valid.sum()} valid compositions out of {len(df)} total")
    return is_valid

def format_temperature(temp, unit='C'):
    """
    Format temperature value with unit
    格式化温度值及其单位
    
    Args:
        temp: Temperature value
             温度值
        unit: Temperature unit ('C' or 'K')
             温度单位（'C'或'K'）
        
    Returns:
        str: Formatted temperature string
            格式化后的温度字符串
    """
    if unit == 'C':
        return f"{temp}°C"
    elif unit == 'K':
        return f"{temp}K"
    else:
        raise ValueError("Unit must be 'C' or 'K'")

def format_composition(composition_dict):
    """
    Format element composition dictionary
    格式化元素成分字典
    
    Args:
        composition_dict: Dictionary of element compositions
                        元素成分字典
        
    Returns:
        str: Formatted composition string
            格式化后的成分字符串
    """
    return ' '.join(f"{element}{composition}%" for element, composition in composition_dict.items()) 