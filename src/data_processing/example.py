#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the usage of the alloy data processing module.
This script shows how to:
1. Process aluminum alloy data
2. Process high entropy alloy data
3. Merge datasets
4. Filter data by various criteria
"""

import os
import sys
import pandas as pd

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Now import the modules
from src.data_processing.data_processor import DataProcessor
from src.data_processing.description_generator import DescriptionGenerator
from src.data_processing.data_filter import DataFilter
from src.data_processing.data_merger import DataMerger
from src.data_processing.constants import DEFAULT_TEMP_RANGES, DEFAULT_COLUMNS

def process_al_alloys():
    """Process aluminum alloy data example"""
    print("\n=== Processing Aluminum Alloys ===")
    
    # Initialize processors
    processor = DataProcessor()
    desc_generator = DescriptionGenerator()
    filter = DataFilter()
    
    # Load data
    input_file = 'datasets/Al_Alloys/Al_alloys.csv'
    print(f"Loading data from: {input_file}")
    df = processor.load_data(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Filter by composition (Al content between 90-95%)
    print("\nFiltering by Al composition (90-95%)...")
    df = filter.filter_by_composition(
        df,
        element_columns=['Al(at%)'],
        min_composition=90,
        max_composition=95
    )
    print(f"Data shape after composition filtering: {df.shape}")
    
    # Filter by property (tensile strength > 300 MPa)
    print("\nFiltering by tensile strength (>300 MPa)...")
    df = filter.filter_by_property(
        df=df,
        property_column='UTS(Mpa)',
        min_value=300
    )
    print(f"Data shape after property filtering: {df.shape}")
    
    # Generate processing descriptions
    print("\nGenerating processing descriptions...")
    df = desc_generator.generate_al_description(df)
    
    # Save processed data
    output_file = 'datasets/Al/processed_al_alloys.csv'
    processor.save_data(df, output_file)
    print(f"\nProcessed data saved to: {output_file}")

def process_hea_data():
    """Process high entropy alloy data example"""
    print("\n=== Processing High Entropy Alloys ===")
    
    # Initialize processors
    processor = DataProcessor()
    desc_generator = DescriptionGenerator()
    filter = DataFilter()
    
    # Load data
    input_file = 'datasets/HEA_data/merged_HEA_datasets.csv'
    print(f"Loading data from: {input_file}")
    df = processor.load_data(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Filter by temperature (room temperature)
    print("\nFiltering by room temperature...")
    room_temp_range = DEFAULT_TEMP_RANGES['room_temp']
    df = filter.filter_by_temperature(
        df=df,
        temp_column='Temperature(K)',
        min_temp=room_temp_range[0],
        max_temp=room_temp_range[1]
    )
    print(f"Data shape after temperature filtering: {df.shape}")
    
    # Filter by composition (all elements between 5-35%)
    print("\nFiltering by composition (5-35% for all elements)...")
    df = filter.filter_by_composition(
        df=df,
        element_columns=['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni'],
        min_composition=5,
        max_composition=35
    )
    print(f"Data shape after composition filtering: {df.shape}")
    
    # Generate processing descriptions
    print("\nGenerating processing descriptions...")
    df = desc_generator.generate_hea_description(df)
    
    # Save processed data
    output_file = 'processed_hea_data.csv'
    processor.save_data(df, output_file)
    print(f"\nProcessed data saved to: {output_file}")

def merge_datasets():
    """Merge datasets example"""
    print("\n=== Merging Datasets ===")
    
    # Initialize merger
    merger = DataMerger()
    
    # Load datasets
    dataset1 = 'datasets/HEA_data/dataset1.csv'
    dataset2 = 'datasets/HEA_data/dataset2.csv'
    print(f"Loading datasets:\n1. {dataset1}\n2. {dataset2}")
    
    df1 = merger.load_data(dataset1)
    df2 = merger.load_data(dataset2)
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = merger.merge_hea_datasets(df1, df2)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Save merged data
    output_file = 'datasets/HEA_data/merged_HEA_datasets.csv'
    merger.save_data(merged_df, output_file)
    print(f"\nMerged data saved to: {output_file}")

def main():
    """Main function to run all examples"""
    try:
        # Process aluminum alloys
        process_al_alloys()
        
        # Process high entropy alloys
        # process_hea_data()
        
        # Merge datasets
        # merge_datasets()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 