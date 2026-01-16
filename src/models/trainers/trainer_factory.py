"""
Factory for creating model trainers
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Type
from .base_trainer import BaseTrainer
from .nn_trainer import NNTrainer
from .bert_trainer import BERTTrainer
from .ml_trainer import MLTrainer
from .cnn_trainer import CnnTrainer

class TrainerFactory:
    """
    Factory class for creating model trainers
    """
    
    @staticmethod
    def create_trainer(model_type: str,
                      model: Any = None,
                      result_dir: str = None,
                      model_name: str = None,
                      target_names: List[str] = None,
                      train_data: Dict[str, Any] = None,
                      val_data: Dict[str, Any] = None,
                      model_params: Dict[str, Any] = None,
                      training_params: Dict[str, Any] = None) -> BaseTrainer:
        """
        Create a trainer instance
        
        Args:
            model_type (str): Type of model ('nn', 'ml', 'bert', 'cnn')
            model: Any: Model object
            result_dir (str): Directory to save results
            model_name (str): Name of the model
            target_names (List[str]): List of target names
            train_data (Dict[str, Any]): Training data
            val_data (Dict[str, Any]): Validation data
            model_params (Dict[str, Any]): Model parameters
            training_params (Dict[str, Any]): Training parameters
            
        Returns:
            BaseTrainer: Trainer instance
        """
        trainer_map = {
            'nn': NNTrainer,
            'bert': BERTTrainer,
            'ml': MLTrainer,
            'cnn': CnnTrainer,
        }
        
        if model_type not in trainer_map:
            raise ValueError(f"Unsupported model_type for trainer creation: {model_type}")

        trainer_class = trainer_map[model_type]
        
        # Base arguments for all trainers
        trainer_args = {
            'result_dir': result_dir,
            'model_name': model_name,
            'target_names': target_names,
            'training_params': training_params or {}
        }
        
        # Add model-specific arguments
        if model_type in ['nn', 'bert', 'cnn']:
            trainer_args['model'] = model
            if model_type in ['nn', 'bert']: # Keep old behavior for nn, bert
                trainer_args['train_data'] = train_data
                trainer_args['val_data'] = val_data
        elif model_type == 'ml':
            trainer_args['model_type'] = model_name # In MLTrainer, model_type is the specific algorithm
            trainer_args['model_params'] = model_params or {}
            trainer_args['train_data'] = train_data
            trainer_args['val_data'] = val_data
            
        return trainer_class(**trainer_args)
    
    @staticmethod
    def get_default_params(trainer_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get default parameters for different model types
        
        Args:
            trainer_type (str): Type of trainer ('ml' or 'bert')
            
        Returns:
            Dict[str, Dict[str, Any]]: Default parameters for each model type
        """
        if trainer_type == 'ml':
            return {
                'xgb': {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0
                },
                'rf': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                'svr': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'epsilon': 0.1
                },
                'ridge': {
                    'alpha': 1.0
                },
                'lasso': {
                    'alpha': 1.0
                },
                'elastic': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5
                }
            }
        elif trainer_type == 'bert':
            return {
                'bert': {
                    'pretrained_model_name': 'bert-base-uncased',
                    'max_length': 128,
                    'dropout_rate': 0.1,
                    'use_features': True,
                    'use_lora': False,
                    'lora_config': {
                        'r': 8,
                        'lora_alpha': 32,
                        'lora_dropout': 0.1,
                        'bias': 'none'
                    }
                }
            }
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    @staticmethod
    def get_supported_trainers() -> List[str]:
        """
        Get list of supported trainer types
        
        Returns:
            List[str]: List of supported trainer types
        """
        return ['nn', 'ml', 'bert', 'cnn']
    
    @staticmethod
    def get_trainer_description(trainer_type: str) -> str:
        """
        Get description of a trainer type
        
        Args:
            trainer_type (str): Type of trainer
            
        Returns:
            str: Description of the trainer type
            
        Raises:
            ValueError: If trainer_type is not supported
        """
        descriptions = {
            'nn': 'Neural network trainer for traditional ML models',
            'ml': 'Machine learning trainer for traditional ML models',
            'bert': 'BERT trainer for text classification',
            'cnn': 'CNN trainer for models with 1D/2D convolutions'
        }
        
        if trainer_type not in descriptions:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
        
        return descriptions[trainer_type]
    
    @staticmethod
    def get_trainer_class(trainer_type: str) -> Type[BaseTrainer]:
        """
        Get trainer class for a given type
        
        Args:
            trainer_type (str): Type of trainer
            
        Returns:
            Type[BaseTrainer]: Trainer class
            
        Raises:
            ValueError: If trainer_type is not supported
        """
        trainer_classes = {
            'nn': NNTrainer,
            'ml': MLTrainer,
            'bert': BERTTrainer,
            'cnn': CnnTrainer
        }
        
        if trainer_type not in trainer_classes:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
        
        return trainer_classes[trainer_type] 