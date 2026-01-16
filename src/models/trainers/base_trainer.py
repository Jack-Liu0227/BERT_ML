"""
Base trainer for model training
"""

import os
import torch
import json
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod

class BaseTrainer:
    """Base class for all trainers"""
    def __init__(
        self,
        model,
        result_dir: str,
        model_name: str,
        target_names: List[str],
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 1e-4,
        train_data: Optional[Dict[str, np.ndarray]] = None,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: The model to train.
            result_dir: Directory to save results.
            model_name: Name of the model.
            target_names: List of target column names.
            early_stopping_patience: Number of epochs to wait for improvement.
            early_stopping_delta: Minimum improvement required for early stopping.
            train_data: Dictionary containing training data ('X' and 'y').
            val_data: Dictionary containing validation data ('X' and 'y').
            training_params: Dictionary containing training parameters.
        """
        self.model = model
        self.result_dir = result_dir
        self.model_name = model_name
        self.target_names = target_names
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.train_data = train_data
        self.val_data = val_data
        
        # Default training parameters
        self._default_training_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'device': None,
            'early_stopping_patience': self.early_stopping_patience,
            'use_lr_scheduler': False,
            'lr_scheduler_patience': 5,
            'lr_scheduler_factor': 0.5
        }
        
        # Update with user-provided parameters
        self._default_training_params.update(training_params or {})
        self.training_params = self._default_training_params
        
        # Create checkpoints directory if it doesn't exist
        self.checkpoints_dir = os.path.join(self.result_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, is_best: bool = False, **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
            **kwargs: Additional items to save in the checkpoint.
        """
        # Ensure the checkpoint directory exists
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Base filename
        checkpoint_filename = f"{self.model_name}_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_filename)
        
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            **kwargs
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model so far, create a copy
        if is_best:
            best_filename = f"{self.model_name}_best.pt"
            best_path = os.path.join(self.checkpoints_dir, best_filename)
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            
        Returns:
            Dictionary containing loaded checkpoint data.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model. Must be implemented by subclasses.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Dictionary containing training history.
        """
        raise NotImplementedError("Subclasses must implement train method")

    def early_stopping(self, current_val_loss: float, patience: int = 10) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            current_val_loss: Current validation loss.
            patience: Number of epochs to wait for improvement.
            
        Returns:
            True if training should stop, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement early_stopping method")
        
    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data: Dictionary containing data ('X' and 'y').
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        raise NotImplementedError("Subclasses must implement evaluate method") 