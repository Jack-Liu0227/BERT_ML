"""
Evaluator for machine learning models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Any, Union, Tuple
from .base_evaluator import BaseEvaluator
from ..visualization.plot_utils import (
    plot_all_sets_compare_scatter,
    plot_cv_error_boxplot
)

# 添加SHAP支持
try:
    import shap
except ImportError as e:
    print(f"[MLEvaluator] Warning: Could not import 'shap'. SHAP analysis will be unavailable. Error: {e}")
    shap = None
import torch




class MLEvaluator(BaseEvaluator):
    """
    Evaluator for traditional machine learning models.
    Provides methods for standard evaluation and cross-validation evaluation.

    Args:
        result_dir (str): Directory to save evaluation results.
        model_name (str): Name of the model.
        target_names (List[str]): List of target names.
    """
    def __init__(self,
                 result_dir: str,
                 model_name: str,
                 target_names: List[str]):
        super().__init__(result_dir, model_name, target_names)
        print(f"\n[MLEvaluator] Initialized for model '{self.model_name}'.")
        print(f"[MLEvaluator] Results will be saved in: {self.result_dir}")
        print(f"[MLEvaluator] Target names: {self.target_names}")

    def evaluate(self,
                 model: Any,
                 train_data: Dict[str, np.ndarray],
                 test_data: Dict[str, np.ndarray],
                 val_data: Optional[Dict[str, np.ndarray]] = None,
                 save_prefix: Optional[str] = None,
                 scaler_y: Optional[Any] = None,
                 save_predictions: bool = True) -> Dict[str, Any]:
        """
        Evaluate model performance on dedicated train, validation, and test sets.
        This is used for non-cross-validation training runs.

        Args:
            model (Any): The trained model object.
            train_data (Dict[str, np.ndarray]): Training data dictionary {'X':, 'y':}.
            test_data (Dict[str, np.ndarray]): Test data dictionary {'X':, 'y':}.
            val_data (Optional[Dict[str, np.ndarray]]): Validation data dictionary {'X':, 'y':}.
            save_prefix (Optional[str]): A prefix for saving results. If None, model_name is used.
            scaler_y (Optional[Any]): The scaler used for the target variable 'y'.
            save_predictions (bool): Whether to save prediction files.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        prefix = save_prefix if save_prefix is not None else self.model_name
        print(f"\n[MLEvaluator] Starting standard evaluation for model: {prefix}")

        # --- Get Predictions and ensure they are 2D ---
        train_pred = model.predict(train_data['X'])
        if train_pred.ndim == 1:
            train_pred = train_pred.reshape(-1, 1)

        test_pred = model.predict(test_data['X'])
        if test_pred.ndim == 1:
            test_pred = test_pred.reshape(-1, 1)

        val_pred = None
        if val_data:
            val_pred = model.predict(val_data['X'])
            if val_pred.ndim == 1:
                val_pred = val_pred.reshape(-1, 1)

        # --- Inverse transform predictions and true values if scaler_y is provided ---
        if scaler_y:
            print("[MLEvaluator] scaler_y provided, performing inverse transform on predictions and true values.")
            train_pred = scaler_y.inverse_transform(train_pred)
            train_true = scaler_y.inverse_transform(train_data['y'])
            test_pred = scaler_y.inverse_transform(test_pred)
            test_true = scaler_y.inverse_transform(test_data['y'])
            if val_data:
                val_pred = scaler_y.inverse_transform(val_pred)
                val_true = scaler_y.inverse_transform(val_data['y'])
        else:
            train_true = train_data['y']
            test_true = test_data['y']
            if val_data:
                val_true = val_data['y']


        # --- Compute, Print, and Save Metrics ---
        print("\n[MLEvaluator] Computing metrics...")
        
        # Compute with simple prefixes for printing
        train_metrics = self.compute_metrics(train_true, train_pred, prefix='train_')
        test_metrics = self.compute_metrics(test_true, test_pred, prefix='test_')
        val_metrics = {}
        if val_data:
            val_metrics = self.compute_metrics(val_true, val_pred, prefix='val_')

        # Print summary, which expects simple prefixes
        self.print_metrics_summary({
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        })
        
        # Create a new dictionary with the full prefix for saving and returning
        all_metrics = {}
        all_metrics.update({f"{prefix}_{k}": v for k, v in train_metrics.items()})
        all_metrics.update({f"{prefix}_{k}": v for k, v in test_metrics.items()})
        if val_data:
            all_metrics.update({f"{prefix}_{k}": v for k, v in val_metrics.items()})
        
        metrics_path = os.path.join(self.result_dir, f'{prefix}_metrics.json')
        self.save_metrics(all_metrics, metrics_path)

        # --- Save Predictions ---
        if save_predictions:
            self.save_predictions(train_true, train_pred, f'{prefix}_train', self.target_names, self.predictions_dir, dataset_name='Train')
            self.save_predictions(test_true, test_pred, f'{prefix}_test', self.target_names, self.predictions_dir, dataset_name='Test')
            if val_data:
                self.save_predictions(val_true, val_pred, f'{prefix}_val', self.target_names, self.predictions_dir, dataset_name='Validation')
        
        # --- Generate and Save Plots ---
        print("\n[MLEvaluator] Generating and saving plots...")
        plot_path = os.path.join(self.plots_dir, f'{prefix}_all_sets_comparison.png')
        # 输出传入数据的维度
        print(f"train_true.shape: {train_true.shape}")
        print(f"test_true.shape: {test_true.shape}")
        # print(f"val_true.shape: {val_true.shape}")
        print(f"train_pred.shape: {train_pred.shape}")
        print(f"test_pred.shape: {test_pred.shape}")
        # print(f"val_pred.shape: {val_pred.shape}")
        plot_all_sets_compare_scatter(
            train_data=(train_true, train_pred),
            test_data=(test_true, test_pred),
            target_names=self.target_names,
            save_path=plot_path,
            val_data=(val_true, val_pred) if val_data else None
        )
        print(f"[MLEvaluator] Comparison plot saved to {plot_path}")

        print(f"\n[MLEvaluator] Evaluation for model '{prefix}' completed successfully!")
        return all_metrics

    def evaluate_cross_validation(self,
                                  model: Any,
                                  cv_results: Dict[str, Any],
                                  train_data: Dict[str, np.ndarray],
                                  test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate the best model from a cross-validation run.
        This involves evaluating the best model on the full training and test sets,
        and generating a box plot of cross-validation errors.

        Args:
            model (Any): The best model object saved from the CV run.
            cv_results (Dict[str, Any]): The results dictionary from the trainer's CV run.
                                         Must contain 'predictions' with fold data.
            train_data (Dict[str, np.ndarray]): The full training data.
            test_data (Dict[str, np.ndarray]): The test data.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        print(f"\n[MLEvaluator] Starting cross-validation evaluation for model: {self.model_name}")

        # --- Standard evaluation on full data splits ---
        metrics = self.evaluate(
            model=model,
            train_data=train_data,
            test_data=test_data,
            val_data=None  # No dedicated validation set in this context
        )

        # --- CV-specific evaluation and plotting ---
        print("\n[MLEvaluator] Generating CV-specific plots...")
        fold_predictions = cv_results.get('predictions')
        
        cv_metrics = {}
        if fold_predictions:
            # --- CV Metrics Calculation ---
            all_fold_metrics = {metric: {target: [] for target in self.target_names} for metric in ['r2', 'mae', 'rmse']}

            for fold_pred_data in fold_predictions:
                true_vals = fold_pred_data['true']
                pred_vals = fold_pred_data['pred']
                
                # Ensure 2D for metric calculation
                if true_vals.ndim == 1: true_vals = true_vals.reshape(-1, 1)
                if pred_vals.ndim == 1: pred_vals = pred_vals.reshape(-1, 1)
                
                fold_metrics = self.compute_metrics(true_vals, pred_vals)
                
                for i, target_name in enumerate(self.target_names):
                    all_fold_metrics['r2'][target_name].append(fold_metrics.get(f'{target_name}_r2', np.nan))
                    all_fold_metrics['mae'][target_name].append(fold_metrics.get(f'{target_name}_mae', np.nan))
                    all_fold_metrics['rmse'][target_name].append(fold_metrics.get(f'{target_name}_rmse', np.nan))

            # --- Aggregate CV Metrics (Mean and Std) ---
            for metric, target_data in all_fold_metrics.items():
                for target, values in target_data.items():
                    cv_metrics[f'cv_{target}_{metric}_mean'] = np.mean(values)
                    cv_metrics[f'cv_{target}_{metric}_std'] = np.std(values)
            
            # --- CV Plotting ---
            # Generate and save the CV error box plot
            plot_cv_error_boxplot(
                model_name=self.model_name,
                fold_predictions=fold_predictions,
                target_names=self.target_names,
                save_dir=self.plots_dir
            )
        else:
            print("[MLEvaluator] WARNING: 'predictions' not found in cv_results. Skipping CV error box plot and metrics.")
        
        # Merge the CV metrics into the main metrics dictionary
        metrics.update(cv_metrics)
        # Re-save metrics file to include CV stats
        metrics_path = os.path.join(self.result_dir, f'{self.model_name}_metrics.json')
        self.save_metrics(metrics, metrics_path)
            
        print(f"\n[MLEvaluator] Cross-validation evaluation for model '{self.model_name}' completed successfully!")
        return metrics 