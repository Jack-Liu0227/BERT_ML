#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base data processor class for handling alloy data
合金数据处理的基础类
"""

import pandas as pd
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Base class for processing alloy data
    合金数据处理的基础类
    """
    
    def __init__(self, root_dir=None):
        """
        Initialize the data processor
        初始化数据处理器
        
        Args:
            root_dir: Project root directory, defaults to current directory
                     项目根目录，默认为当前目录
        """
        if root_dir is None:
            self.root_dir = Path.cwd()
        else:
            self.root_dir = Path(root_dir)
            
        logger.info(f"Working directory: {Path().absolute()}")
        logger.info(f"Root directory: {self.root_dir}")
    
    def load_data(self, file_path):
        """
        Load data from file
        从文件加载数据
        
        Args:
            file_path: Path to the data file
                      数据文件路径
            
        Returns:
            pandas.DataFrame: Loaded dataframe
                            加载的数据框
        """
        logger.info(f"Loading data from: {file_path}")
        
        # Check file format
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Handle special format (first line is header, second line is column names)
        if first_line.startswith(',') and ('Property' in first_line or 'Chemical compositions' in first_line):
            logger.info("Detected special format: first line is header, second line is column names")
            df = pd.read_csv(file_path, header=1)
        else:
            df = pd.read_csv(file_path)
            
        logger.info(f"Successfully loaded {len(df)} rows")
        return df
    
    def save_data(self, df, output_path):
        """
        Save data to file
        保存数据到文件
        
        Args:
            df: DataFrame to save
                要保存的数据框
            output_path: Path to save the data
                        保存数据的路径
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
    
    def remove_missing_values(self, df, columns=None):
        """
        Remove rows with missing values
        删除含有缺失值的行
        
        Args:
            df: Input DataFrame
                输入数据框
            columns: List of columns to check for missing values
                    需要检查缺失值的列名列表
            
        Returns:
            pandas.DataFrame: DataFrame with missing values removed
                            删除缺失值后的数据框
        """
        original_rows = len(df)
        
        if columns is None:
            cleaned_df = df.dropna()
        else:
            cleaned_df = df.dropna(subset=columns)
        
        removed_rows = original_rows - len(cleaned_df)
        logger.info(f"Removed {removed_rows} rows with missing values (from {original_rows} to {len(cleaned_df)} rows)")
        
        return cleaned_df 