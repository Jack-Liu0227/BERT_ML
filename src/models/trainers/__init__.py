"""
Trainer module for model training

This module provides a set of trainer classes for training different types of models:
- BaseTrainer: Abstract base class defining the training interface
- NNTrainer: Trainer for neural network models
- BERTTrainer: Trainer for BERT-based models
- TrainerFactory: Factory class for creating different types of trainers
"""

from .base_trainer import BaseTrainer
from .nn_trainer import NNTrainer
from .bert_trainer import BERTTrainer
from .trainer_factory import TrainerFactory

__all__ = [
    'BaseTrainer',
    'NNTrainer',
    'BERTTrainer',
    'TrainerFactory'
] 