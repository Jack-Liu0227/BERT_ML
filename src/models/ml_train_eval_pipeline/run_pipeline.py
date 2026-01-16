from .config import get_args
from .pipeline import MLTrainingPipeline
from .utils import to_long_path

def main():
    """Main entry point for the training pipeline."""
    args = get_args()
    
    # Handle long paths on Windows
    args.result_dir = to_long_path(args.result_dir)
    
    pipeline = MLTrainingPipeline(args)
    pipeline.run()

if __name__ == '__main__':
    
    main()

# --- Example Usage ---



# To run a standard cross-validation training:
# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" `
#     --result_dir "output/results/Ti_alloys/Xue/ID/opt_mlp" `
#     --target_columns "UTS(MPa)" "El(%)"`
#     --processing_cols "Solution Temperature(℃)" "Solution Time(h)" "Aging Temperature(℃)" "Aging Time(h)" "Thermo-Mechanical Treatment Temperature()" "Deformation(%)" `
#     --model_type "mlp" `
#     --mlp_max_iter 500 `
#     --cross_validate `
#     --num_folds 3 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature False `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50

# # Aluminum alloys example
# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets/Al_Alloys/USTB/USTB_Al_alloys_processed_split_withID.csv" `
#     --result_dir "output/results/Al_alloys/USTB_new/ID/opt_mlp" `
#     --target_columns "UTS(MPa)" `
#     --processing_cols "ST1" "TIME1" "ST2" "TIME2" "ST3" "TIME3" "Cold_Deformation_percent" "First_Aging_Temp_C" "First_Aging_Time_h" "Second_Aging_Temp_C" "Second_Aging_Time_h" "Third_Aging_Temp_C" "Third_Aging_Time_h" `
#     --model_type "mlp" `
#     --mlp_max_iter 500 `
#     --cross_validate `
#     --num_folds 9 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature False `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50

# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets\Nb_Alloys\Nb_cleandata\Nb_clean_with_processing_sequence_withID.csv" `
#     --result_dir "output/results/Nb_alloys/Nb_cleandata/ID/opt_mlp/withTem" `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
#     --processing_cols "Temperature((K))" "Anealing Temperature((K))" "Anealing times(h)" "Thermo-Mechanical Treatment Temperature((K))" "Deformation(%)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" `
#     --model_type "mlp" `
#     --cross_validate `
#     --mlp_max_iter 500 `
#     --num_folds 9 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature True `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50


# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets\Nb_Alloys\Nb_cleandata\Nb_clean_with_processing_sequence_withID.csv" `
#     --result_dir "output/results/Nb_alloys/Nb_cleandata/ID/opt_mlp/noTem" `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
#     --processing_cols "Anealing Temperature((K))" "Anealing times(h)" "Thermo-Mechanical Treatment Temperature((K))" "Deformation(%)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" "Anealing Temperature((K))2" "Anealing times(h)2" "Anealing Temperature((K))3" "Anealing times(h)3" "reduction(mm)" "Cold rolling((K))" "Cold rolling(h)" "Stress Relieved((K))" "Stress Relieved(h)" "Recrystallized((K))" "Recrystallized(h)" "Cold Worked ratio(%)" "warm work ratio(%)" "warm swaged((K))" "warm swaged(h)" `
#     --model_type "mlp" `
#     --cross_validate `
#     --num_folds 9 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature False `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50
	
# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets/HEA_data/RoomTemperature_HEA_train_with_ID.csv" `
#     --result_dir "output/results/HEA_data/yasir_data_half/ID//opt_mlp" `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" `
#     --processing_cols "Hom_Temp(K)" "CR(%)" "recrystalize temperature/K" "recrystalize time/mins" "Anneal_Temp(K)" "Anneal_Time(h)" "aging temperature/K" "aging time/hours" `
#     --model_type "mlp" `
#     --mlp_max_iter 500 `
#     --cross_validate `
#     --num_folds 9 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature False `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50

# python -m src.models.ml_train_eval_pipeline.run_pipeline `
#     --data_file "datasets\Nb_Alloys\Harbin\HTC_processed_withID.csv" `
#     --result_dir "output/results/Nb_alloys/Harbin/HTC/opt_mlp" `
#     --target_columns "high-temperature compressive strength(MPa)" `
#     --model_type "mlp" `
#     --mlp_max_iter 500 `
#     --cross_validate `
#     --num_folds 9 `
#     --test_size 0.2 `
#     --random_state 42  `
#     --evaluate_after_train `
#     --use_composition_feature True `
#     --use_temperature False `
#     --run_shap_analysis `
#     --use_optuna `
#     --n_trials 50

# python -m src.models.ml_train_eval_pipeline.run_pipeline \
#     --data_file "datasets\Nb_Alloys\Harbin\KQ_processed_withID.csv" \
#     --result_dir "output/results/Nb_alloys/Harbin/KQ/opt_mlp" \
#     --target_columns "KQ(MPa·m^(1/2))" \
#     --model_type "mlp" \
#     --mlp_max_iter 500 \
#     --cross_validate \
#     --num_folds 9 \
#     --test_size 0.2 \
#     --random_state 42  \
#     --evaluate_after_train \
#     --use_composition_feature True \
#     --use_temperature False \
#     --run_shap_analysis \
#     --use_optuna \
#     --n_trials 50