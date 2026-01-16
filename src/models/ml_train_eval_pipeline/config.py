"""
Configuration setup for the training pipeline.
"""
import argparse

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML Training and Evaluation Pipeline')
    
    # Data and Directories
    parser.add_argument('--data_file', type=str, required=True, help='Path to the processed feature file (features.csv or target.csv)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save training results')
    
    # Model Selection
    parser.add_argument('--model_type', type=str, default='xgboost', 
                        choices=['xgboost', 'lightgbm', 'sklearn_gpr', 'catboost', 'sklearn_rf', 'sklearn_svr', 'mlp'], 
                        help='Type of model to train')

    # Feature Selection
    parser.add_argument('--target_columns', type=str, nargs='+', required=True, help='List of target column names')
    parser.add_argument('--processing_cols', type=str, nargs='*', default=[], help='List of processing column names')
    parser.add_argument('--use_composition_feature', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--use_temperature', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--other_features_name', type=str, nargs='*', default=None)

    # Execution Control
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset for the test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--evaluate_after_train', action='store_true', help='Evaluate model after training')
    parser.add_argument('--run_shap_analysis', action='store_true', help='Run SHAP feature importance analysis after training')
    parser.add_argument('--cross_validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')

    # Optuna Hyperparameter Optimization
    parser.add_argument('--use_optuna', action='store_true', help='Enable Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='ml_hyperparameter_optimization', help='Name of the Optuna study')

    # MLP-specific parameters
    parser.add_argument('--mlp_max_iter', type=int, default=200, help='Maximum number of iterations for MLP training')

    return parser.parse_args()