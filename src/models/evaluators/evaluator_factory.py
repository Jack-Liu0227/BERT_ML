"""
Factory class for creating model evaluators
"""

from typing import Dict, List, Optional, Union, Any
from .base_evaluator import BaseEvaluator
from .alloys_evaluator import AlloysEvaluator
from .bert_evaluator import BERTEvaluator
from .ml_evaluator import MLEvaluator
import inspect

class EvaluatorFactory:
    """
    Factory class for creating model evaluators
    
    This class provides a centralized way to create different types of evaluators
    based on the model type and evaluation requirements.
    """
    
    @staticmethod
    def create_evaluator(evaluator_type: str,
                        result_dir: str,
                        model_name: str,
                        target_names: List[str],
                        **kwargs) -> BaseEvaluator:
        """
        Create an evaluator instance
        
        Args:
            evaluator_type (str): Type of evaluator to create ('alloys' or 'bert' or 'ml')
            result_dir (str): Directory to save evaluation results
            model_name (str): Name of the model
            target_names (List[str]): List of target names
            **kwargs: Additional arguments for specific evaluator types
            
        Returns:
            BaseEvaluator: Created evaluator instance
            
        Raises:
            ValueError: If evaluator_type is not supported
        """
        evaluator_classes = {
            "base": BaseEvaluator,
            "alloys": AlloysEvaluator,
            "ml": MLEvaluator,
            "bert": BERTEvaluator,
        }

        evaluator_class = evaluator_classes.get(evaluator_type)
        if not evaluator_class:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")

        # Extract arguments relevant to the specific evaluator class
        # Get the constructor signature
        sig = inspect.signature(evaluator_class.__init__)
        valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Add 'self' to valid_args if it's not there, just in case.
        if 'self' not in valid_args:
            valid_args.pop('self', None) # to be safe

        try:
            return evaluator_class(result_dir=result_dir,
                                  model_name=model_name,
                                  target_names=target_names,
                                  **valid_args)
        except TypeError as e:
            print(f"Error creating evaluator '{evaluator_type}' with args {valid_args}.")
            raise e
    
    @staticmethod
    def get_supported_evaluators() -> List[str]:
        """
        Get list of supported evaluator types
        
        Returns:
            List[str]: List of supported evaluator types
        """
        return ['alloys', 'ml', 'bert']
    
    @staticmethod
    def get_evaluator_description(evaluator_type: str) -> str:
        """
        Get description of evaluator type
        
        Args:
            evaluator_type (str): Type of evaluator
            
        Returns:
            str: Description of evaluator type
            
        Raises:
            ValueError: If evaluator_type is not supported
        """
        descriptions = {
            'alloys': 'Evaluator for traditional machine learning and neural network models',
            'ml': 'Evaluator for traditional machine learning models',
            'bert': 'Evaluator for BERT-based models with PyTorch DataLoader support'
        }
        
        if evaluator_type.lower() not in descriptions:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
            
        return descriptions[evaluator_type.lower()] 