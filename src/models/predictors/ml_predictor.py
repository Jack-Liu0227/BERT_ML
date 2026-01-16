"""
Machine learning model predictor implementation
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
import joblib

class MLPredictor:
    """
    Predictor for machine learning models (XGBoost, RandomForest, etc.)
    
    Args:
        model_type (str): Type of model ('xgb', 'rf', 'svr', 'ridge', 'lasso', 'elastic')
        model_path (str): Path to the trained model file
    """
    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self) -> Any:
        """
        Load model based on model type
        
        Returns:
            Any: Loaded model instance
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"\nLoading model from: {self.model_path}")
        
        if self.model_type == 'xgb':
            model = xgb.XGBRegressor()
            model.load_model(self.model_path)
        elif self.model_type == 'rf':
            model = joblib.load(self.model_path)
        elif self.model_type == 'svr':
            model = joblib.load(self.model_path)
        elif self.model_type == 'ridge':
            model = joblib.load(self.model_path)
        elif self.model_type == 'lasso':
            model = joblib.load(self.model_path)
        elif self.model_type == 'elastic':
            model = joblib.load(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        return self.model.predict(X) 