import pandas as pd
from sklearn.metrics import r2_score

# 读取closest_to_mean模型的评估结果
file_path = r"output\new_results_withuncertainty\HEA_corrosion\Pitting_potential_data_xiongjie\tradition\model_comparison\mlp_results\closest_to_mean_evaluation\predictions\closest_to_mean_model_evaluation_all_predictions.csv"

df = pd.read_csv(file_path)

print(f"总行数: {len(df)}")
print(f"\nDataset列的唯一值: {df['Dataset'].unique()}")
print(f"各Dataset的行数:")
print(df['Dataset'].value_counts())

# 计算各个数据集的R²
for dataset in ['Train', 'Validation', 'Test']:
    data = df[df['Dataset'] == dataset]
    if len(data) > 0:
        r2 = r2_score(data['Ep(mV)_Actual'], data['Ep(mV)_Predicted'])
        print(f"\n{dataset} set:")
        print(f"  样本数: {len(data)}")
        print(f"  R²: {r2:.6f}")
