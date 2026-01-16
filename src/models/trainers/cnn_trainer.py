"""
Trainer for the AlloyCnnV2 model.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import collections
from sklearn.metrics import r2_score

from src.models.evaluators.base_evaluator import BaseEvaluator

def now():
    """Return current time string for logging."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class CnnTrainer:
    """
    A trainer for CNN-based models for alloy property prediction.
    It handles training, validation, early stopping, and cross-validation.
    """
    def __init__(
        self,
        model: nn.Module,
        result_dir: str,
        model_name: str,
        training_params: Dict[str, Any],
        target_names: List[str]
    ):
        """
        Args:
            model (nn.Module): The neural network model to train.
            result_dir (str): Directory to save results and checkpoints.
            model_name (str): Name of the model, used for saving files.
            training_params (Dict[str, Any]): A dictionary of training hyperparameters.
            target_names (List[str]): A list of target variable names.
        """
        self.model = model
        self.result_dir = result_dir
        self.model_name = model_name
        self.params = training_params
        self.target_names = target_names
        
        # Modify device selection logic
        device_arg = self.params.get('device')
        if device_arg and 'cuda' in device_arg and not torch.cuda.is_available():
            print(f"[{now()}] Warning: CUDA device '{device_arg}' is not available. Falling back to CPU.")
            self.device = 'cpu'
        elif device_arg:
            self.device = device_arg
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[{now()}] Using device: {self.device}")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.params.get('learning_rate', 0.001), 
            weight_decay=self.params.get('weight_decay', 0)
        )
        self.loss_fn = nn.MSELoss()
        
        self._setup_lr_scheduler()
            
        self.checkpoints_dir = os.path.join(self.result_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _setup_lr_scheduler(self):
        """Initializes the learning rate scheduler."""
        self.lr_scheduler = None
        if self.params.get('use_lr_scheduler', False):
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                'min',
                patience=self.params.get('lr_scheduler_patience', 10),
                factor=self.params.get('lr_scheduler_factor', 0.5),
            )

    def _get_data_loader(self, data: Dict[str, np.ndarray], shuffle: bool) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.params.get('batch_size', 32), shuffle=shuffle)

    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch and calculate metrics."""
        self.model.train()
        total_loss = 0
        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []

        for X_batch, y_batch in data_loader:
            X_batch, y_batch_cpu = X_batch.to(self.device), y_batch
            y_batch_gpu = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred_batch = self.model(X_batch)
            loss = self.loss_fn(y_pred_batch, y_batch_gpu)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            
            all_y_true.append(y_batch_cpu.numpy())
            all_y_pred.append(y_pred_batch.detach().cpu().numpy())

        metrics: Dict[str, float] = {'loss': total_loss / len(data_loader.dataset)}
        
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        
        r2_scores = r2_score(y_true, y_pred, multioutput='raw_values')
        if y_true.ndim == 1:
            r2_scores = [r2_scores]
        
        for i, target_name in enumerate(self.target_names):
            metrics[f'r2_{target_name}'] = float(r2_scores[i])
        metrics['r2'] = float(np.mean(r2_scores))
        
        return metrics

    def _eval_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Run one evaluation epoch and calculate loss and R2 score."""
        self.model.eval()
        total_loss = 0
        all_y_true: List[np.ndarray] = []
        all_y_pred: List[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch_cpu = X_batch.to(self.device), y_batch
                y_pred_batch = self.model(X_batch)

                y_batch_gpu = y_batch.to(self.device)
                loss = self.loss_fn(y_pred_batch, y_batch_gpu)
                total_loss += loss.item() * X_batch.size(0)
                
                all_y_true.append(y_batch_cpu.numpy())
                all_y_pred.append(y_pred_batch.cpu().numpy())
        
        metrics: Dict[str, float] = {'loss': total_loss / len(data_loader.dataset)}
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        
        r2_scores = r2_score(y_true, y_pred, multioutput='raw_values')
        if y_true.ndim == 1:
            r2_scores = [r2_scores]
        
        for i, target_name in enumerate(self.target_names):
            metrics[f'r2_{target_name}'] = float(r2_scores[i])
        metrics['r2'] = float(np.mean(r2_scores))
        
        return metrics

    def train(self, num_epochs: int, train_data: Dict[str, np.ndarray], val_data: Optional[Dict[str, np.ndarray]] = None, model_save_name: Optional[str] = None, save_checkpoint: bool = True) -> Tuple[Dict[str, List[float]], float, Optional[dict]]:
        """
        Main training loop with validation and early stopping.
        
        Returns:
            A tuple containing:
            - history (dict): A dictionary of training and validation metrics for each epoch.
            - best_val_loss (float): The best validation loss achieved.
            - best_model_state_dict (dict): The state dictionary of the model with the best validation loss.
        """
        train_loader = self._get_data_loader(train_data, shuffle=True)
        val_loader = self._get_data_loader(val_data, shuffle=False) if val_data else None

        best_val_loss = float('inf')
        best_model_state_dict = None
        epochs_no_improve = 0
        patience = self.params.get('early_stopping_patience', 20)
        history: Dict[str, List[float]] = collections.defaultdict(list)

        if save_checkpoint:
            final_save_name = model_save_name or f"{self.model_name}_best.pt"
            best_model_path = os.path.join(self.checkpoints_dir, final_save_name)

        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_metrics = self._train_epoch(train_loader)
            for key, value in train_metrics.items():
                history[f'train_{key}'].append(value)

            val_loss_str = "N/A"
            val_r2_str = "N/A"
            if val_loader:
                val_metrics = self._eval_epoch(val_loader)
                for key, value in val_metrics.items():
                    history[f'val_{key}'].append(value)

                val_loss = val_metrics['loss']
                val_loss_str = f"{val_metrics['loss']:.6f}"
                val_r2_str = f"{val_metrics['r2']:.4f}"
                
                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    best_model_state_dict = deepcopy(model_to_save.state_dict())
                    if save_checkpoint:
                        torch.save({'model_state_dict': best_model_state_dict}, best_model_path)
                else:
                    epochs_no_improve += 1
            
            epoch_duration = time.time() - start_time
            print(f"[{now()}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_metrics['loss']:.6f} | Train R2: {train_metrics['r2']:.4f} | Val Loss: {val_loss_str} | Val R2: {val_r2_str} | Time: {epoch_duration:.2f}s")
            
            if val_loader and epochs_no_improve >= patience:
                print(f"[{now()}] Early stopping triggered after {epoch + 1} epochs.")
                break
        
        return history, best_val_loss, best_model_state_dict

    def train_cross_validation(self, num_epochs: int, num_folds: int, fold_data: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Perform cross-validation training.
        """
        all_fold_val_metrics: Dict[str, List[float]] = collections.defaultdict(list)
        all_fold_predictions = [] # To store predictions for each fold
        best_overall_val_loss = float('inf')
        
        initial_model_state = deepcopy(self.model.state_dict())

        for i, fold in enumerate(fold_data):
            print(f"[{now()}] --- Starting Fold {i+1}/{num_folds} ---")
            
            # Reset model to initial state and re-init optimizer for each fold
            self.model.load_state_dict(initial_model_state)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.get('learning_rate', 0.001), weight_decay=self.params.get('weight_decay', 0))
            self._setup_lr_scheduler()

            train_data_fold = {'X': fold['train_X'], 'y': fold['train_y']}
            val_data_fold = {'X': fold['val_X'], 'y': fold['val_y']}

            fold_model_save_name = f"{self.model_name}_fold{i+1}_best.pt"
            
            history, fold_best_val_loss, _ = self.train(num_epochs, train_data_fold, val_data_fold, model_save_name=fold_model_save_name)
            
            # Plot training curves for the current fold
            plots_dir = os.path.join(self.result_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'fold_{i+1}_training_curves.png')
            BaseEvaluator.plot_training_curves(history, save_path, title=f"Fold {i+1} Training Curves")
            print(f"[{now()}] Fold {i+1} training curves saved to {save_path}")
            
            # Load the best model for this fold to evaluate
            best_fold_model_path = os.path.join(self.checkpoints_dir, fold_model_save_name)
            if os.path.exists(best_fold_model_path):
                model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_to_load.load_state_dict(torch.load(best_fold_model_path, map_location=self.device)['model_state_dict'])
                
                val_loader = self._get_data_loader(val_data_fold, shuffle=False)
                fold_val_metrics = self._eval_epoch(val_loader)

                for key, value in fold_val_metrics.items():
                    all_fold_val_metrics[key].append(value)
                
                fold_val_loss = fold_val_metrics['loss']
                print(f"[{now()}] Fold {i+1} Best Validation Loss: {fold_val_loss:.6f}, R2: {fold_val_metrics['r2']:.4f}")

                # Collect predictions for analysis
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(val_data_fold['X'], dtype=torch.float32).to(self.device)
                    y_val_pred = self.model(X_val_tensor).cpu().numpy()
                all_fold_predictions.append({'true': val_data_fold['y'], 'pred': y_val_pred})

                # Check if this fold's model is the best overall
                if fold_val_loss < best_overall_val_loss:
                    best_overall_val_loss = fold_val_loss
                    best_cv_model_path = os.path.join(self.checkpoints_dir, f"{self.model_name}_cv_best.pt")
                    torch.save({'model_state_dict': model_to_load.state_dict()}, best_cv_model_path)
                    print(f"[{now()}] New best CV model saved to {best_cv_model_path}")
            else:
                print(f"[{now()}] Warning: Best model for fold {i+1} not found. Skipping metrics for this fold.")

        cv_results: Dict[str, Any] = {
            'predictions': all_fold_predictions
        }
        # Handle loss separately
        if 'loss' in all_fold_val_metrics:
            cv_results['cv_val_loss_mean'] = np.mean(all_fold_val_metrics['loss'])
            cv_results['cv_val_loss_std'] = np.std(all_fold_val_metrics['loss'])

        # Handle R2 for each target to match the format expected by build_summary_table
        for target_name in self.target_names:
            r2_key = f'r2_{target_name}'
            if r2_key in all_fold_val_metrics:
                r2_values = all_fold_val_metrics[r2_key]
                # Key format for build_summary_table: cv_{target}_{metric_name.lower()}_mean
                summary_key_mean = f'cv_{target_name}_r2_mean'
                summary_key_std = f'cv_{target_name}_r2_std'
                cv_results[summary_key_mean] = np.mean(r2_values)
                cv_results[summary_key_std] = np.std(r2_values)

        # Also add the average R2 for logging purposes
        if 'r2' in all_fold_val_metrics:
             cv_results['cv_val_r2_mean'] = np.mean(all_fold_val_metrics['r2'])
             cv_results['cv_val_r2_std'] = np.std(all_fold_val_metrics['r2'])

        mean_cv_loss = cv_results.get('cv_val_loss_mean', -1)
        std_cv_loss = cv_results.get('cv_val_loss_std', -1)
        print(f"[{now()}] Average CV Validation Loss: {mean_cv_loss:.6f} +/- {std_cv_loss:.6f}")
        
        # This structure now contains keys compatible with the test script's summary builder
        return cv_results 