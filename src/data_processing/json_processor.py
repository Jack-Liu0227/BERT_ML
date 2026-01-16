#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON data processor for handling alloy data in JSON format
JSON格式合金数据处理器
"""

import json
import pandas as pd
from pathlib import Path
import logging
from .data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JsonProcessor(DataProcessor):
    """
    Class for processing JSON format alloy data
    JSON格式合金数据处理类
    """
    
    def __init__(self, root_dir=None):
        """
        Initialize the JSON processor
        初始化JSON处理器
        
        Args:
            root_dir: Project root directory, defaults to current directory
                     项目根目录，默认为当前目录
        """
        super().__init__(root_dir)
    
    def load_json(self, file_path):
        """
        Load JSON data from file
        从文件加载JSON数据
        
        Args:
            file_path: Path to the JSON file
                      JSON文件路径
            
        Returns:
            dict: Loaded JSON data
                 加载的JSON数据
        """
        logger.info(f"Loading JSON data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"Successfully loaded JSON data")
        return data
    
    def save_json(self, data, output_path):
        """
        Save data to JSON file
        保存数据到JSON文件
        
        Args:
            data: Data to save
                  要保存的数据
            output_path: Path to save the data
                        保存数据的路径
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Data saved to: {output_path}")
    
    def json_to_dataframe(self, json_data, material_type=None):
        """
        Convert JSON data to DataFrame
        将JSON数据转换为DataFrame
        
        Args:
            json_data: JSON data to convert
                      要转换的JSON数据
            material_type: Type of material to extract (optional)
                          要提取的材料类型（可选）
            
        Returns:
            pandas.DataFrame: Converted DataFrame
                            转换后的数据框
        """
        logger.info("Converting JSON data to DataFrame")
        
        # Extract data based on material type if specified
        if material_type:
            if material_type not in json_data:
                raise ValueError(f"Material type '{material_type}' not found in JSON data")
            data = json_data[material_type]
        else:
            data = json_data
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"Successfully converted {len(df)} rows to DataFrame")
        return df
    
    def dataframe_to_json(self, df, material_type=None):
        """
        Convert DataFrame to JSON format
        将DataFrame转换为JSON格式
        
        Args:
            df: DataFrame to convert
                要转换的数据框
            material_type: Type of material (optional)
                          材料类型（可选）
            
        Returns:
            dict: JSON formatted data
                 JSON格式的数据
        """
        logger.info("Converting DataFrame to JSON format")
        
        # Convert DataFrame to dict
        data = df.to_dict(orient='records')
        
        # Add material type wrapper if specified
        if material_type:
            data = {material_type: data}
            
        logger.info(f"Successfully converted {len(df)} rows to JSON format")
        return data 