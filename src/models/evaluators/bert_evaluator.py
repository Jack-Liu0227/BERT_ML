"""
BERT model evaluator implementation
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import DataLoader
from .base_evaluator import BaseEvaluator
from ..visualization.plot_utils import plot_all_sets_compare_scatter,plot_compare_scatter
class BERTEvaluator(BaseEvaluator):
    """
    Evaluator for BERT-based alloy property prediction models
    
    Args:
        result_dir (str): Directory to save evaluation results
        model_name (str): Name of the model
        target_names (List[str]): List of target names
        device (str): Device to use for evaluation ('cuda' or 'cpu')
        batch_size (int): Batch size for evaluation
    """
    def __init__(self,
                 result_dir: str,
                 model_name: str,
                 target_names: List[str],
                 device: str = 'cuda',
                 batch_size: int = 32):
        super().__init__(result_dir, model_name, target_names)
        self.device = device
        self.batch_size = batch_size
        
    def evaluate_model(self,
                      model: torch.nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      test_loader: DataLoader,
                      save_prefix: str = '') -> Dict[str, Any]:
        """
        Evaluate model performance on train and test data
        
        Args:
            model (torch.nn.Module): Trained model
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            save_prefix (str): Prefix for saved files
            
        Returns:
            Dict[str, Any]: Evaluation metrics and results
        """
        model.to(self.device)
        model.eval()
        
        # Get predictions
        train_true, train_pred = self._get_predictions(model, train_loader)
        val_true, val_pred = self._get_predictions(model, val_loader)
        test_true, test_pred = self._get_predictions(model, test_loader)
        
        # Compute metrics
        train_metrics = self.compute_metrics(train_true, train_pred, prefix='train_')
        val_metrics = self.compute_metrics(val_true, val_pred, prefix='val_')
        test_metrics = self.compute_metrics(test_true, test_pred, prefix='test_')
        
        # Combine metrics
        metrics = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_name': self.model_name,
            'target_names': self.target_names
        }
        
        # Save metrics
        metrics_path = os.path.join(self.result_dir, f'{save_prefix}metrics.json')
        self.save_metrics(metrics, metrics_path)
        
        # Save predictions
        self.save_predictions(train_true, train_pred, f'{save_prefix}train')
        self.save_predictions(val_true, val_pred, f'{save_prefix}val')
        self.save_predictions(test_true, test_pred, f'{save_prefix}test')
        
        # # Plot results
        # self.plot_prediction_scatter(
        #     train_true, train_pred,
        #     os.path.join(self.plots_dir, f'{save_prefix}train')
        # )
        # self.plot_prediction_scatter(
        #     val_true, val_pred,
        #     os.path.join(self.plots_dir, f'{save_prefix}val')
        # )
        # self.plot_prediction_scatter(
        #     test_true, test_pred,
        #     os.path.join(self.plots_dir, f'{save_prefix}test')
        # )
        
        # Plot comparison
        plot_all_sets_compare_scatter(
            train_data=(train_true, train_pred),
            val_data=(val_true, val_pred),
            test_data=(test_true, test_pred),
            target_names=self.target_names,
            save_path=os.path.join(self.plots_dir, f'{save_prefix}comparison.png'),
            metrics=metrics
        )
        
        return metrics
    
    def evaluate_cross_validation(self,
                                model_class: torch.nn.Module,
                                cv_loaders: List[Dict[str, DataLoader]],
                                save_prefix: str = '') -> Dict[str, Any]:
        """
        Evaluate model performance using cross-validation
        
        Args:
            model_class (torch.nn.Module): Model class (not trained)
            cv_loaders (List[Dict[str, DataLoader]]): List of cross-validation data loaders
            save_prefix (str): Prefix for saved files
            
        Returns:
            Dict[str, Any]: Cross-validation metrics and results
        """
        cv_metrics = []
        cv_predictions = []
        
        for i, loaders in enumerate(cv_loaders):
            # Initialize and train model
            fold_model = model_class().to(self.device)
            # Note: Training should be done before calling this method
            fold_model.eval()
            
            # Get predictions
            train_true, train_pred = self._get_predictions(fold_model, loaders['train'])
            val_true, val_pred = self._get_predictions(fold_model, loaders['val'])
            
            # Compute metrics
            train_metrics = self.compute_metrics(
                train_true, train_pred, prefix=f'fold_{i}_train_'
            )
            val_metrics = self.compute_metrics(
                val_true, val_pred, prefix=f'fold_{i}_val_'
            )
            
            # Store metrics and predictions
            cv_metrics.append({
                'fold': i,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            cv_predictions.append({
                'fold': i,
                'train_true': train_true,
                'train_pred': train_pred,
                'val_true': val_true,
                'val_pred': val_pred
            })
            
            # Save fold predictions
            self.save_predictions(
                train_true, train_pred,
                f'{save_prefix}fold_{i}_train'
            )
            self.save_predictions(
                val_true, val_pred,
                f'{save_prefix}fold_{i}_val'
            )
        
        # Compute average metrics
        avg_metrics = {
            'train_metrics': {},
            'val_metrics': {}
        }
        
        for target in self.target_names:
            for metric in ['RMSE', 'MAE', 'R2']:
                train_values = [m['train_metrics'][f'fold_{i}_train_{target}_{metric}']
                              for i, m in enumerate(cv_metrics)]
                val_values = [m['val_metrics'][f'fold_{i}_val_{target}_{metric}']
                            for i, m in enumerate(cv_metrics)]
                
                avg_metrics['train_metrics'][f'{target}_{metric}'] = np.mean(train_values)
                avg_metrics['val_metrics'][f'{target}_{metric}'] = np.mean(val_values)
        
        # Save CV results
        cv_results = {
            'cv_metrics': cv_metrics,
            'avg_metrics': avg_metrics,
            'model_name': self.model_name,
            'target_names': self.target_names,
            'n_folds': len(cv_loaders)
        }
        
        metrics_path = os.path.join(self.result_dir, f'{save_prefix}cv_metrics.json')
        self.save_metrics(cv_results, metrics_path)
        
        # Plot CV results
        for i, pred in enumerate(cv_predictions):
            self.plot_prediction_scatter(
                pred['train_true'], pred['train_pred'],
                os.path.join(self.plots_dir, f'{save_prefix}fold_{i}_train')
            )
            self.plot_prediction_scatter(
                pred['val_true'], pred['val_pred'],
                os.path.join(self.plots_dir, f'{save_prefix}fold_{i}_val')
            )
        
        return cv_results
    
    def _get_predictions(self,
                        model: torch.nn.Module,
                        data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for a data loader
        
        Args:
            model (torch.nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader
            
        Returns:
            tuple[np.ndarray, np.ndarray]: True values and predictions
        """
        model.eval()
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get model predictions
                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels)
                predictions = outputs['logits']
                
                # Store predictions and true values
                all_true.append(labels.cpu().numpy())
                all_pred.append(predictions.cpu().numpy())
        
        # Concatenate all batches
        true_values = np.concatenate(all_true, axis=0)
        predictions = np.concatenate(all_pred, axis=0)
        
        return true_values, predictions 