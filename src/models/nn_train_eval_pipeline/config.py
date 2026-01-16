"""
Configuration setup for the training pipeline.
"""
import argparse

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NN Training and Evaluation Pipeline')
    
    # Data and Directories
    parser.add_argument('--data_file', type=str, required=True, help='Path to the processed feature file (features.csv or target.csv)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save training results')
    
    # Model Selection
    parser.add_argument('--model_type', type=str, default='nn', choices=['nn', 'bert', 'cnn'], help='Type of model to train')

    # Model Architecture
    parser.add_argument('--emb_hidden_dim', type=int, default=256, help='Hidden dimension for embedding features')
    parser.add_argument('--feature1_hidden_dim', type=int, default=256, help='Hidden dimension for feature1 features')
    parser.add_argument('--feature2_hidden_dim', type=int, default=256, help='Hidden dimension for feature2 features')
    parser.add_argument('--other_features_hidden_dim', type=int, default=0, help='Hidden dimension for other features')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], help='Hidden dimensions for prediction network')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for L2 regularization')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Enable learning rate scheduler')
    parser.add_argument('--lr_scheduler_patience', type=int, default=10, help='Patience for learning rate scheduler')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='Factor by which learning rate will be reduced')
    
    # Feature Selection
    parser.add_argument('--target_columns', type=str, nargs='+', required=True, help='List of target column names')
    parser.add_argument('--use_process_embedding', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_joint_composition_process_embedding', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_element_embedding', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_composition_feature', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_feature1', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_feature2', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--other_features_name', type=str, nargs='+', default=None)
    parser.add_argument('--use_temperature', type=lambda x: x.lower() == 'true', default=False)

    # Execution Control
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset for the test split')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--evaluate_after_train', action='store_true', help='Evaluate model after training')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training (e.g., "cuda", "cpu")')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs if available')
    parser.add_argument('--cross_validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')

    # Optuna Hyperparameter Optimization
    parser.add_argument('--use_optuna', action='store_true', help='Enable Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='nn_hyperparameter_optimization', help='Name of the Optuna study')
    
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of times to repeat the experiment (for statistical analysis)')
    
    return parser.parse_args() 