"""
Configuration setup for the CNN training pipeline.
"""
import argparse

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CNN Training and Evaluation Pipeline')
    
    # Data and Directories
    parser.add_argument('--data_file', type=str, required=True, help='Path to the processed feature file (features.csv)')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save training results')
    
    # Model Configuration (AlloyCnnV2 specific)
    parser.add_argument('--cnn_1d_config', type=str, 
                        default='{"ele_emb": {"keyword": "ele_emb", "pre_cnn_fc_layers": [512, 256], "out_channels": [16, 32], "fc_layers": [128]}, "proc_emb": {"keyword": "proc_emb", "pre_cnn_fc_layers": [512, 256], "out_channels": [16, 32], "fc_layers": [128]}}',
                        help='JSON string for 1D CNN configuration.')
    
    parser.add_argument('--cnn_2d_config', type=str, default='{"out_channels": [16, 32], "fc_layers": [128]}',
                        help='JSON string for 2D CNN configuration.')

    parser.add_argument('--features_for_2d_cnn', type=str, nargs='+', default=['ele_emb', 'proc_emb'],
                        help='List of feature types to be stacked for the 2D CNN.')

    parser.add_argument('--other_feature_configs', type=str, default='None',
                        help='JSON string for other features not processed by CNNs.')

    parser.add_argument('--prediction_hidden_dims', type=int, nargs='+', default=[256, 128],
                      help='Hidden dimensions for the final prediction network.')
    
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
    
    # Feature and Target Selection
    parser.add_argument('--target_columns', type=str, nargs='+', required=True, help='List of target column names')

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
    parser.add_argument('--study_name', type=str, default='cnn_hyperparameter_optimization', help='Name of the Optuna study')
    parser.add_argument('--study_storage', type=str, default=None, 
                        help='Storage URL for Optuna study (e.g., "sqlite:///optuna.db"). If None, study will not be persisted.')
    parser.add_argument('--load_study_if_exists', action='store_true', 
                        help='Load existing study if it exists in the storage')
    
    return parser.parse_args() 