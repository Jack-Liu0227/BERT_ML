from .config import get_args
from .pipeline import CnnTrainingPipeline
from .utils import to_long_path

def main():
    """Main entry point for the training pipeline."""
    args = get_args()
    
    # Handle long paths on Windows
    args.result_dir = to_long_path(args.result_dir)
    
    pipeline = CnnTrainingPipeline(args)
    pipeline.run()

if __name__ == '__main__':
    main()

# --- Example Usage ---
"""
# 推荐：使用新的多模型目录结构
# 格式：Features/{model_name}/{dataset_path}/ 和 output/{model_name}/{experiment_path}/

# To run a standard cross-validation training (使用 SteelBERT 特征):
python -m src.models.cnn_train_eval_pipeline.run_pipeline `
    --data_file "Features/steelbert/Steel/USTB_steel/all_features/features.csv" `
    --result_dir "output/steelbert/Steel/USTB_steel/CNN_cv5_optuna" `
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" `
    --cross_validate `
    --num_folds 5 `
    --epochs 200 `
    --batch_size 1024 `
    --evaluate_after_train `
    --device "cuda:0" `
    --use_optuna `
    --n_trials 2 `
    --study_name "cnn_hyperparameter_optimization" `
    --study_storage "sqlite:///output/steelbert/Steel/USTB_steel/optuna.db" `
    --load_study_if_exists

# To run a standard training without cross-validation:
python -m src.models.cnn_train_eval_pipeline.run_pipeline \\
    --data_file "Features/HEA_data/SHU/corrosion/all_features/features.csv" \\
    --result_dir "output/results/HEA_data/SHU/corrosion/CNN_pipeline_test" \\
    --epochs 100 \\
    --batch_size 1024 \\
    --learning_rate 0.001 \\
    --weight_decay 1e-5 \\
    --cnn_1d_config '{"ele_emb": {"keyword": "ele_emb", "pre_cnn_fc_layers": [512], "out_channels": [16, 32], "fc_layers": [128]}, "proc_emb": {"keyword": "proc_emb", "pre_cnn_fc_layers": [512], "out_channels": [16, 32], "fc_layers": [128]}}' \\
    --cnn_2d_config '{"out_channels": [16, 32], "fc_layers": [128]}' \\
    --features_for_2d_cnn "ele_emb" "proc_emb" \\
    --other_feature_configs '{"temperature": {"keyword": "Temperature"}, "cl_concentration": {"keyword": "Cl Concentration"}, "ph": {"keyword": "PH"}}' \\
    --prediction_hidden_dims 256 128 \\
    --dropout_rate 0.0 \\
    --target_columns  "Ep(mV)" \\
    --patience 2000 \\
    --evaluate_after_train \\
    --device cuda

# To continue an existing Optuna study (使用 SteelBERT 特征):
python -m src.models.cnn_train_eval_pipeline.run_pipeline `
    --data_file "Features/steelbert/Steel/USTB_steel/all_features/features.csv" `
    --result_dir "output/steelbert/Steel/USTB_steel/CNN_cv5_optuna" `
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" `
    --cross_validate `
    --num_folds 5 `
    --epochs 200 `
    --batch_size 1024 `
    --evaluate_after_train `
    --device "cuda:0" `
    --use_optuna `
    --n_trials 10 `
    --study_name "cnn_hyperparameter_optimization" `
    --study_storage "sqlite:///output/steelbert/Steel/USTB_steel/optuna.db" `
    --load_study_if_exists

# 使用 MatSciBERT 特征进行对比实验:
python -m src.models.cnn_train_eval_pipeline.run_pipeline `
    --data_file "Features/matscibert/Steel/USTB_steel/all_features/features.csv" `
    --result_dir "output/matscibert/Steel/USTB_steel/CNN_cv5" `
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" `
    --cross_validate `
    --num_folds 5 `
    --epochs 200 `
    --batch_size 1024 `
    --evaluate_after_train `
    --device "cuda:0"
"""