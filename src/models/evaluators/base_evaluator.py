"""
Base evaluator for model evaluation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..utils.json_utils import save_metrics
from ..visualization.plot_utils import (
    plot_prediction_scatter,
    plot_feature_importance
)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class BaseEvaluator:
    """
    Base class for model evaluation
    
    Args:
        result_dir (str): Directory to save evaluation results
        model_name (str): Name of the model
        target_names (List[str]): List of target names
    """
    def __init__(self, 
                 result_dir: str,
                 model_name: str,
                 target_names: List[str]):
        self.result_dir = result_dir
        self.model_name = model_name
        self.target_names = target_names
        
        # Create directories
        self.plots_dir = os.path.join(result_dir, 'plots')
        self.predictions_dir = os.path.join(result_dir, 'predictions')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {}

    def print_metrics_summary(self, metrics_data: Dict[str, Dict[str, float]]):
        """
        Prints a formatted summary of metrics for train, validation, and test sets.

        Args:
            metrics_data (Dict[str, Dict[str, float]]): 
                A dictionary containing metrics for different data splits.
                Example: {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
        """
        print("\n[Evaluator] Metrics Summary:")
        for target in self.target_names:
            print(f"\n  Target: {target}")
            
            # Safely access metrics for each data split
            train_metrics = metrics_data.get('train', {})
            val_metrics = metrics_data.get('val', {})
            test_metrics = metrics_data.get('test', {})

            if train_metrics:
                print("    Training Set:")
                print(f"      RMSE: {train_metrics.get(f'train_{target}_rmse', float('nan')):.4f}")
                print(f"      MAE:  {train_metrics.get(f'train_{target}_mae', float('nan')):.4f}")
                print(f"      R2:   {train_metrics.get(f'train_{target}_r2', float('nan')):.4f}")

            if val_metrics:
                print("    Validation Set:")
                print(f"      RMSE: {val_metrics.get(f'val_{target}_rmse', float('nan')):.4f}")
                print(f"      MAE:  {val_metrics.get(f'val_{target}_mae', float('nan')):.4f}")
                print(f"      R2:   {val_metrics.get(f'val_{target}_r2', float('nan')):.4f}")

            if test_metrics:
                print("    Test Set:")
                print(f"      RMSE: {test_metrics.get(f'test_{target}_rmse', float('nan')):.4f}")
                print(f"      MAE:  {test_metrics.get(f'test_{target}_mae', float('nan')):.4f}")
                print(f"      R2:   {test_metrics.get(f'test_{target}_r2', float('nan')):.4f}")

    def compute_metrics(self, 
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       prefix: str = '') -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            prefix (str): Prefix for metric names
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # 保证y_true和y_pred都是2d
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        metrics = {}
        for i, target in enumerate(self.target_names):
            target_true = y_true[:, i]
            target_pred = y_pred[:, i]
            
            metrics[f'{prefix}{target}_rmse'] = np.sqrt(mean_squared_error(target_true, target_pred))
            metrics[f'{prefix}{target}_mae'] = mean_absolute_error(target_true, target_pred)
            metrics[f'{prefix}{target}_r2'] = r2_score(target_true, target_pred)
            
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], filepath: str):
        """
        Save metrics to JSON file
        
        Args:
            metrics (Dict[str, Any]): Metrics to save
            filepath (str): Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, cls=NumpyEncoder, indent=4)

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load metrics from JSON file
        
        Args:
            filepath (str): Path to metrics file
            
        Returns:
            Dict[str, Any]: Loaded metrics
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def plot_training_curves(
                           history: Dict[str, List[float]],
                           save_path: str,
                           title: str = "Training Curves"):
        """
        Plot training curves from history
        
        Args:
            history (Dict[str, List[float]]): Training history
            save_path (str): Path to save the plot
            title (str): Title of the plot
        """
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot loss curves
        if 'train_r2' in history and 'val_r2' in history:
            # If R2 scores are available, plot both loss and R2
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss', color='blue')
            if 'val_loss' in history: # Check if validation loss exists
                plt.plot(history['val_loss'], label='Validation Loss', color='red')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot R² curve
            plt.subplot(1, 2, 2)
            plt.plot(history['train_r2'], label='Training R²', color='blue')
            if 'val_r2' in history: # Check if validation R2 exists
                plt.plot(history['val_r2'], label='Validation R²', color='green')
            plt.title('Validation R² Score')
            plt.xlabel('Epoch')
            plt.ylabel('R² Score')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            # If R2 scores are not available, just plot loss
            plt.plot(history['train_loss'], label='Training Loss', color='blue')
            if 'val_loss' in history: # Check if validation loss exists
                plt.plot(history['val_loss'], label='Validation Loss', color='red')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_prediction_scatter(
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              save_dir: str,
                              target_names: List[str]):
        """
        Plot prediction scatter plots for each target
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            save_dir (str): Directory to save plots
            target_names (List[str]): List of target names
        """
        # 保证y_true和y_pred都是2d
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        num_targets = y_true.shape[1]
        for i in range(num_targets):
            target_name = target_names[i] if i < len(target_names) else f"Target {i+1}"
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, label='Predictions')
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Ideal Prediction')
            plt.xlabel(f'True {target_name}')
            plt.ylabel(f'Predicted {target_name}')
            plt.title(f'Prediction Scatter Plot for {target_name}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(save_dir, f'{target_name}_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Removes characters that are invalid for filenames."""
        return name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'percent')

    @staticmethod
    def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str, target_names: List[str], predictions_dir: str, dataset_name: str, ids: Optional[np.ndarray] = None):
        """
        Save predictions to a single CSV file.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            prefix (str): Prefix for the filename.
            target_names (List[str]): List of target names.
            predictions_dir (str): Directory to save predictions.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Create a single DataFrame for all targets
        all_preds = pd.DataFrame()
        # Prepend ID if provided
        if ids is not None:
            all_preds['ID'] = ids
        for i, target in enumerate(target_names):
            sanitized_target = BaseEvaluator._sanitize_filename(target)
            all_preds[f'True_{sanitized_target}'] = y_true[:, i]
            all_preds[f'Pred_{sanitized_target}'] = y_pred[:, i]
        
        all_preds['set'] = dataset_name
        
        # Define the file path
        filepath = os.path.join(predictions_dir, f'{prefix}all_predictions.csv')
        
        # Append to the file if it exists, otherwise create a new one
        if os.path.exists(filepath):
            all_preds.to_csv(filepath, mode='a', header=False, index=False)
        else:
            all_preds.to_csv(filepath, mode='w', header=True, index=False)


    def plot_model_comparison(self):
        """
        Plot comparison of different models' performance
        """
        if not self.metrics:
            print("No metrics available for comparison")
            return
            
        model_names = list(self.metrics.keys())
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Train RMSE': [self.metrics[m]['train_rmse'] for m in model_names],
            'Test RMSE': [self.metrics[m]['test_rmse'] for m in model_names],
            'Train R²': [self.metrics[m]['train_r2'] for m in model_names],
            'Test R²': [self.metrics[m]['test_r2'] for m in model_names]
        })
        
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(self.result_dir, 'model_comparison.csv'), index=False)
        
        # Save all metrics to JSON
        save_metrics(self.metrics, os.path.join(self.result_dir, 'evaluation_metrics.json'))
    
    def plot_feature_importance(self, model, model_name, feature_names):
        """
        Plot feature importance if supported by the model
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            feature_names (list): List of feature names
        """
        plot_feature_importance(model, model_name, feature_names, self.plots_dir) 