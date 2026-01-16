import pandas as pd
import os
import numpy as np

def create_processing_description(row):
    """
    Creates a detailed string describing the material processing sequence from a row of data,
    with ordered steps.
    根据数据行创建一个详细描述材料处理序列的字符串，并带有有序的步骤。

    Args:
        row (pd.Series): A row from the alloy data DataFrame.
                         来自合金数据DataFrame的一行。

    Returns:
        str: A string describing the processing sequence.
             一个描述处理序列的字符串。
    """
    # Use .get() to avoid KeyError if a column is missing
    # 使用.get()以避免在列丢失时出现KeyError
    def get(key):
        return row.get(key)

    steps = []

    # Step 1: Solution Treatment and Quenching
    # 步骤1：固溶处理和淬火
    solution_parts = []
    if pd.notna(get('Solution_Temp_C')) and get('Solution_Temp_C') != '':
        solution_parts.append(f"solution treated at {get('Solution_Temp_C')}°C")
        if pd.notna(get('Solution_Time_h')) and get('Solution_Time_h') != '':
            solution_parts.append(f"for {get('Solution_Time_h')}h")
    
    if pd.notna(get('Quench_Method')) and get('Quench_Method') != '':
        # Combine with solution treatment if available
        # 如果有固溶处理信息，则进行合并
        if solution_parts:
            solution_parts.append(f"and quenched in {get('Quench_Method')}")
            steps.append(" ".join(solution_parts))
        else:
            steps.append(f"quenched in {get('Quench_Method')}")
    elif solution_parts:
        steps.append(" ".join(solution_parts))

    # Step 2: Thermo-Mechanical Treatment (TMT)
    # 步骤2：热机械处理（TMT）
    tmt_parts = []
    if pd.notna(get('TMT_Temp_C')) and get('TMT_Temp_C') != '':
        tmt_parts.append(f"thermo-mechanically treated at {get('TMT_Temp_C')}°C")
    if pd.notna(get('TMT_Deformation_percent')) and get('TMT_Deformation_percent') != '':
        tmt_parts.append(f"with {get('TMT_Deformation_percent')}% deformation")
    
    if tmt_parts:
        steps.append(" ".join(tmt_parts))

    # Step 3: Aging / Retrogression
    # 步骤3：时效/回归处理
    aging_parts = []
    # First stage aging
    if pd.notna(get('Aging_Temp_1_C')) and get('Aging_Temp_1_C') != '' and pd.notna(get('Aging_Time_1_h')) and get('Aging_Time_1_h') != '':
        aging_parts.append(f"aged at {get('Aging_Temp_1_C')}°C for {get('Aging_Time_1_h')}h")

    # Retrogression
    if pd.notna(get('Retrogression_Temp_C')) and get('Retrogression_Temp_C') != '' and pd.notna(get('Retrogression_Time_min')) and get('Retrogression_Time_min') != '':
        aging_parts.append(f"retrogression at {get('Retrogression_Temp_C')}°C for {get('Retrogression_Time_min')}min")

    # Second stage aging (or re-aging after retrogression)
    if pd.notna(get('Aging_Temp_2_C')) and get('Aging_Temp_2_C') != '' and pd.notna(get('Aging_Time_2_h')) and get('Aging_Time_2_h') != '':
        aging_parts.append(f"then aged at {get('Aging_Temp_2_C')}°C for {get('Aging_Time_2_h')}h")

    if aging_parts:
        steps.append(", ".join(aging_parts))

    # Step 4: Final Cooling
    # 步骤4：最终冷却
    if pd.notna(get('Cooling_Method')) and get('Cooling_Method') != '':
        steps.append(f"cooled by {get('Cooling_Method')}")

    if not steps:
        return "No processing information available."

    # Create the final ordered string
    # 创建最终的有序字符串
    ordinals = ["first heat treatment", "second heat treatment", "third heat treatment", "fourth heat treatment", "fifth heat treatment"]
    numbered_description = []
    for i, step in enumerate(steps):
        if i < len(ordinals):
            prefix = ordinals[i]
        else:
            prefix = f"{i+1}th"
        numbered_description.append(f"{prefix}, {step}")
    
    return "; ".join(numbered_description) + "."

# File paths
# 文件路径
input_file = 'datasets/Al_Alloys/USTB/processed_acta_materialia_data_cleaned.csv'
output_file = 'datasets/Al_Alloys/USTB/processed_acta_materialia_data_with_description.csv'

# Check if the input file exists
# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"Input file not found: {input_file}")
else:
    # Read the CSV file
    # 读取CSV文件
    df = pd.read_csv(input_file)


    # Create the new descriptive column
    # 创建新的描述性列
    df['Processing_Description'] = df.apply(create_processing_description, axis=1)

    # Save the updated dataframe
    # 保存更新后的数据框
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Processing complete. New file saved to: {output_file}")
    print("\nPreview of the new 'Processing_Description' column:")
    print(df[['Processing_Type', 'Processing_Description']].head()) 