"""
快速验证和修复预测文件的Dataset列
"""
import pandas as pd
import os

# 文件路径
pred_file = r"output\new_results_withuncertainty\HEA_corrosion\Pitting_potential_data_xiongjie\tradition\model_comparison\mlp_results\closest_to_mean_evaluation\predictions\closest_to_mean_model_evaluation_all_predictions.csv"

if os.path.exists(pred_file):
    print(f"Reading file: {pred_file}")
    df = pd.read_csv(pred_file)
    
    print(f"Current columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head(10))
    
    print(f"\nDataset column unique values: {df['Dataset'].unique()}")
    print(f"Dataset column null count: {df['Dataset'].isna().sum()}")
    
    # 如果Dataset列为空，尝试修复
    if df['Dataset'].isna().all() or (df['Dataset'] == '').all():
        print("\nDataset column is empty! Attempting to infer...")
        
        # 根据行数推断（假设Train, Val, Test各占1/3或类似比例）
        total_rows = len(df)
        # 这需要根据实际情况调整
        print(f"Total rows: {total_rows}")
        print("Cannot auto-fix without knowing the split. Please re-run the pipeline.")
    else:
        print("\nDataset column looks OK!")
else:
    print(f"File not found: {pred_file}")
