from pathlib import Path

base_dir = r"output\new_results_withuncertainty\HEA_corrosion\Pitting_potential_data_xiongjie\tradition\model_comparison\mlp_results"
base_path = Path(base_dir)
all_files = list(base_path.rglob("all_predictions.csv"))

print(f"Total files: {len(all_files)}")

valid_files = []
for file_path in all_files:
    path_str = str(file_path)
    
    if 'optuna_trials' in path_str:
        continue
    
    is_fold = 'fold_' in path_str.lower()
    is_mean = 'closest_to_mean' in path_str.lower()
    
    if not (is_fold or is_mean):
        print(f"Skipped non-fold/mean: {path_str}")
        continue
        
    valid_files.append(file_path)
    print(f"Kept: {path_str}")

print(f"Valid files count: {len(valid_files)}")
