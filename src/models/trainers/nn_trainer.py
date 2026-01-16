"""
Neural network model trainer implementation
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from torch.utils.data import DataLoader, TensorDataset
from .base_trainer import BaseTrainer
from sklearn.metrics import r2_score, mean_squared_error
from src.models.evaluators.base_evaluator import BaseEvaluator
import sys # Import sys module

class NNTrainer(BaseTrainer):
    """
    Trainer for neural network models
    
    Args:
        model (nn.Module): Neural network model
        result_dir (str): Directory to save training results
        model_name (str): Name of the model
        target_names (List[str]): List of target names
        train_data (Dict[str, np.ndarray]): Training data dictionary
        val_data (Dict[str, np.ndarray]): Validation data dictionary
        training_params (Dict[str, Any]): Dictionary of training parameters
    """
    def __init__(self,
                 model: nn.Module,
                 result_dir: str,
                 model_name: str,
                 target_names: List[str],
                 train_data: Dict[str, np.ndarray],
                 val_data: Dict[str, np.ndarray],
                 training_params: Dict[str, Any]):
        
        super().__init__(
            model=model,
            result_dir=result_dir,
            model_name=model_name,
            target_names=target_names,
            train_data=train_data,
            val_data=val_data,
            training_params=training_params
        )
        
        # Unpack training parameters
        self.batch_size = training_params.get('batch_size', 32)
        self.learning_rate = training_params.get('learning_rate', 1e-3)
        self.weight_decay = training_params.get('weight_decay', 1e-5)
        self.early_stopping_patience = training_params.get('patience', 200)
        self.early_stopping_delta = training_params.get('early_stopping_delta', 1e-4)
        self.save_checkpoints = training_params.get('save_checkpoints', True)
        device = training_params.get('device', None)
        
        # Set device (CPU or CUDA)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("\nCUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        
        print(f"\nUsing device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # 兼容DataParallel模型
        self.is_parallel = isinstance(self.model, torch.nn.DataParallel)
        
        # Create data loaders
        self.train_loader = self._create_data_loader(
            train_data['X'], train_data['y'], self.batch_size, shuffle=True
        )
        self.val_loader = self._create_data_loader(
            val_data['X'], val_data['y'], self.batch_size, shuffle=False
        )
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Set early stopping parameters
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Save training configuration
        self.config = training_params
        if training_params.get('save_config_on_init', True):
            self._save_config()
    
    def _create_data_loader(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          batch_size: int,
                          shuffle: bool) -> DataLoader:
        """
        Create PyTorch data loader
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            DataLoader: PyTorch data loader
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def _save_config(self):
        """Save training configuration"""
        config_path = os.path.join(self.result_dir, f'{self.model_name}_config.json')
        # 确保存放配置文件的目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save config to {config_path}. Error: {e}")
            print("Will continue training without saving config.")
    
    def train_epoch(self) -> float:
        """
        Trains the model for one epoch.
        
        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_train_loss += loss.item()
            
        return total_train_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Validates the model.
        
        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_loader)

    def _compute_metrics(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Computes loss, R², RMSE, and MAE for a given data loader.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        r2 = r2_score(all_targets, all_preds, multioutput='uniform_average')
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = np.mean(np.abs(all_targets - all_preds))

        return {
            'loss': total_loss / len(data_loader),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }

    def _get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predictions and targets from a data loader."""
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint if enabled."""
        if not self.save_checkpoints:
            return

        checkpoints_dir = os.path.join(self.result_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoints_dir, f'{self.model_name}_best.pt')
        if is_best:
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            }, checkpoint_path)
    
    def train(self, num_epochs: int, callbacks: Optional[List[Callable]] = None) -> Tuple[Dict[str, List[float]], float, Optional[Dict[str, torch.Tensor]]]:
        """
        Train the model
        
        Returns:
            Tuple containing:
            - Dict[str, List[float]]: Training history
            - float: Best validation loss
            - Optional[Dict[str, torch.Tensor]]: State dictionary of the best model
        """
        sys.stdout.reconfigure(encoding='utf-8') # Set stdout encoding to UTF-8
        print(f"\nStarting training for {num_epochs} epochs...")
        history = {
            'train_loss': [],
            'train_rmse': [],
            'train_mae': [],
            'train_r2': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_r2': []
        }
        
        best_state_dict = None

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Compute detailed metrics for logging
            train_metrics = self._compute_metrics(self.train_loader)
            val_metrics = self._compute_metrics(self.val_loader)

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['train_mae'].append(train_metrics['mae'])
            history['train_r2'].append(train_metrics['r2'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_r2'].append(val_metrics['r2'])
            
            # Print progress
            print(f"[{self.model_name}] {time.strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, R²: {train_metrics['r2']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, R²: {val_metrics['r2']:.4f}", flush=True)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                best_state_dict = {k: v.cpu() for k, v in (self.model.module.state_dict() if self.is_parallel else self.model.state_dict()).items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            # Execute callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics)
        
        return history, self.best_val_loss, best_state_dict
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint, compatible with DataParallel"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.is_parallel:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
    
    def train_cross_validation(self, num_epochs: int, num_folds: int, fold_data: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Trains a model using k-fold cross-validation.
        
        Args:
            num_epochs (int): The number of epochs to train for each fold.
            num_folds (int): The number of folds for cross-validation.
            fold_data (List[Dict[str, np.ndarray]]): A list of data dictionaries for each fold.
            
        Returns:
            Dict[str, Any]: A dictionary containing the cross-validation results.
        """
        all_fold_metrics = []
        best_overall_val_loss = float('inf')
        best_model_state = None
        
        print(f"\nStarting {num_folds}-fold cross-validation...")

        for i in range(num_folds):
            print(f"\n--- Fold {i+1}/{num_folds} ---")
            
            # Reset model weights for each fold
            self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.best_val_loss = float('inf')
            self.patience_counter = 0

            # Update data loaders for the current fold
            self.train_loader = self._create_data_loader(fold_data[i]['train_X'], fold_data[i]['train_y'], self.batch_size, True)
            self.val_loader = self._create_data_loader(fold_data[i]['val_X'], fold_data[i]['val_y'], self.batch_size, False)
            
            # Train the model and get the best state dict for the fold
            _, _, best_fold_state_dict = self.train(num_epochs)
            
            # Load the best model from the fold to compute metrics
            if best_fold_state_dict:
                self.model.load_state_dict(best_fold_state_dict)
                val_metrics = self._compute_metrics(self.val_loader)
                all_fold_metrics.append(val_metrics)

                if val_metrics['loss'] < best_overall_val_loss:
                    best_overall_val_loss = val_metrics['loss']
                    best_model_state = best_fold_state_dict
                    print(f"New best model found in fold {i+1} with validation loss: {best_overall_val_loss:.4f}")
            else:
                print(f"Warning: No best model found for fold {i+1}.")

        # Save the best model from all folds
        if best_model_state:
            checkpoints_dir = os.path.join(self.result_dir, 'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoints_dir, f'{self.model_name}_cv_best.pt')
            torch.save({'model_state_dict': best_model_state}, best_model_path)
            print(f"\nSaved best overall model from cross-validation to {best_model_path}")
        
        # Aggregate and return results
        aggregated_metrics = self._aggregate_cv_metrics(all_fold_metrics)
        aggregated_metrics['best_val_loss'] = best_overall_val_loss
        return aggregated_metrics

    def _aggregate_cv_metrics(self, all_fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregates metrics (mean and std) from all folds.
        """
        if not all_fold_metrics:
            return {}
        
        aggregated = {}
        metric_keys = ['loss', 'r2', 'rmse', 'mae']
        for key in metric_keys:
            values = [m[key] for m in all_fold_metrics]
            aggregated[f'cv_{key}_mean'] = np.mean(values)
            aggregated[f'cv_{key}_std'] = np.std(values)
            
        return aggregated
