#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process description generator for alloy data
合金工艺描述生成器
"""

import pandas as pd
import logging
from .data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DescriptionGenerator(DataProcessor):
    """
    Class for generating process descriptions from alloy data
    合金工艺描述生成器类
    """
    
    def __init__(self, root_dir=None):
        """
        Initialize the description generator
        初始化描述生成器
        
        Args:
            root_dir: Project root directory, defaults to current directory
                     项目根目录，默认为当前目录
        """
        super().__init__(root_dir)
    
    def generate_processing_description(self, df, process_params):
        """
        Generate processing descriptions from parameters
        从参数生成工艺描述
        
        Args:
            df: Input DataFrame containing process parameters
                包含工艺参数的输入数据框
            process_params: Dictionary mapping parameter columns to descriptions
                          参数列到描述的映射字典
            
        Returns:
            pandas.DataFrame: DataFrame with added processing descriptions
                            添加了工艺描述的数据框
        """
        logger.info("Generating processing descriptions")
        
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Initialize description column
        result_df['processing_description'] = ''
        
        # Generate descriptions for each row
        for idx, row in result_df.iterrows():
            description_parts = []
            
            # Process each parameter
            for param_col, desc_template in process_params.items():
                if param_col in row and pd.notna(row[param_col]):
                    # Format the description with the parameter value
                    description = desc_template.format(value=row[param_col])
                    description_parts.append(description)
            
            # Join all description parts
            result_df.at[idx, 'processing_description'] = ' '.join(description_parts)
        
        logger.info(f"Generated descriptions for {len(result_df)} rows")
        return result_df
    
    def generate_al_description(self, df):
        """
        Generate descriptions specifically for aluminum alloys
        专门为铝合金生成描述
        
        Args:
            df: Input DataFrame containing aluminum alloy data
                包含铝合金数据的输入数据框
            
        Returns:
            pandas.DataFrame: DataFrame with added processing descriptions
                            添加了工艺描述的数据框
        """
        # Define process parameter mappings for aluminum alloys
        process_params = {
            'temperature': 'Processed at {value}°C',
            'time': 'for {value} hours',
            'cooling_rate': 'with cooling rate of {value}°C/min',
            'pressure': 'under {value} MPa pressure',
            'atmosphere': 'in {value} atmosphere'
        }
        
        return self.generate_processing_description(df, process_params)
    
    def generate_hea_description(self, df):
        """
        Generate descriptions specifically for high entropy alloys
        专门为高熵合金生成描述
        
        Args:
            df: Input DataFrame containing HEA data
                包含高熵合金数据的输入数据框
            
        Returns:
            pandas.DataFrame: DataFrame with added processing descriptions
                            添加了工艺描述的数据框
        """
        # Define process parameter mappings for HEAs
        process_params = {
            'processing_temperature': 'Processed at {value}°C',
            'processing_time': 'for {value} hours',
            'cooling_rate': 'with cooling rate of {value}°C/min',
            'pressure': 'under {value} GPa pressure',
            'atmosphere': 'in {value} atmosphere',
            'method': 'using {value} method'
        }
        
        return self.generate_processing_description(df, process_params) 