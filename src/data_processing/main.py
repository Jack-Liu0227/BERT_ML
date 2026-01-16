#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for alloy data processing
合金数据处理主程序入口
"""

import argparse
import logging
from pathlib import Path
from .data_processor import DataProcessor
from .json_processor import JsonProcessor
from .description_generator import DescriptionGenerator
from .data_filter import DataFilter
from .data_merger import DataMerger
from . import utils
from . import constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_al_alloys(input_file, output_file, add_descriptions=True):
    """
    Process aluminum alloy data
    处理铝合金数据
    
    Args:
        input_file: Path to input file
                   输入文件路径
        output_file: Path to output file
                    输出文件路径
        add_descriptions: Whether to add processing descriptions
                        是否添加工艺描述
    """
    logger.info("Processing aluminum alloy data")
    
    # Initialize processors
    processor = DataProcessor()
    desc_generator = DescriptionGenerator()
    
    # Load data
    df = processor.load_data(input_file)
    
    # Add processing descriptions if requested
    if add_descriptions:
        df = desc_generator.generate_al_description(df)
    
    # Save processed data
    processor.save_data(df, output_file)
    
    logger.info("Aluminum alloy data processing completed")

def process_hea_data(input_file, output_file, add_descriptions=True):
    """
    Process high entropy alloy data
    处理高熵合金数据
    
    Args:
        input_file: Path to input file
                   输入文件路径
        output_file: Path to output file
                    输出文件路径
        add_descriptions: Whether to add processing descriptions
                        是否添加工艺描述
    """
    logger.info("Processing high entropy alloy data")
    
    # Initialize processors
    processor = DataProcessor()
    desc_generator = DescriptionGenerator()
    
    # Load data
    df = processor.load_data(input_file)
    
    # Add processing descriptions if requested
    if add_descriptions:
        df = desc_generator.generate_hea_description(df)
    
    # Save processed data
    processor.save_data(df, output_file)
    
    logger.info("High entropy alloy data processing completed")

def merge_datasets(input_files, output_file, dataset_type='hea'):
    """
    Merge multiple datasets
    合并多个数据集
    
    Args:
        input_files: List of input file paths
                    输入文件路径列表
        output_file: Path to output file
                    输出文件路径
        dataset_type: Type of datasets to merge ('hea' or 'al')
                     要合并的数据集类型（'hea'或'al'）
    """
    logger.info(f"Merging {dataset_type} datasets")
    
    # Initialize processor
    merger = DataMerger()
    
    # Load first dataset
    df1 = merger.load_data(input_files[0])
    
    # Merge with remaining datasets
    for input_file in input_files[1:]:
        df2 = merger.load_data(input_file)
        if dataset_type == 'hea':
            df1 = merger.merge_hea_datasets(df1, df2)
        else:
            df1 = merger.merge_al_datasets(df1, df2)
    
    # Save merged data
    merger.save_data(df1, output_file)
    
    logger.info("Dataset merging completed")

def main():
    """
    Main function to parse arguments and process data
    解析参数并处理数据的主函数
    """
    parser = argparse.ArgumentParser(description='Process alloy data')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for processing aluminum alloys
    al_parser = subparsers.add_parser('process-al', help='Process aluminum alloy data')
    al_parser.add_argument('input_file', help='Input file path')
    al_parser.add_argument('output_file', help='Output file path')
    al_parser.add_argument('--no-descriptions', action='store_true', help='Skip adding processing descriptions')
    
    # Parser for processing high entropy alloys
    hea_parser = subparsers.add_parser('process-hea', help='Process high entropy alloy data')
    hea_parser.add_argument('input_file', help='Input file path')
    hea_parser.add_argument('output_file', help='Output file path')
    hea_parser.add_argument('--no-descriptions', action='store_true', help='Skip adding processing descriptions')
    
    # Parser for merging datasets
    merge_parser = subparsers.add_parser('merge', help='Merge datasets')
    merge_parser.add_argument('input_files', nargs='+', help='Input file paths')
    merge_parser.add_argument('output_file', help='Output file path')
    merge_parser.add_argument('--type', choices=['hea', 'al'], default='hea', help='Type of datasets to merge')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == 'process-al':
        process_al_alloys(args.input_file, args.output_file, not args.no_descriptions)
    elif args.command == 'process-hea':
        process_hea_data(args.input_file, args.output_file, not args.no_descriptions)
    elif args.command == 'merge':
        merge_datasets(args.input_files, args.output_file, args.type)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 