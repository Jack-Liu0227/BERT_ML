"""
检查select_best_model_and_plot.py找到了哪些文件
"""
from pathlib import Path

base_dir = r"output\new_results_withuncertainty\HEA_corrosion\Pitting_potential_data_xiongjie\tradition\model_comparison\mlp_results"
base_path = Path(base_dir)

# 查找所有all_predictions.csv文件
all_files = list(base_path.rglob("all_predictions.csv"))

print(f"找到 {len(all_files)} 个 all_predictions.csv 文件:\n")
for i, file in enumerate(all_files, 1):
    rel_path = file.relative_to(base_path)
    print(f"{i}. {rel_path}")
