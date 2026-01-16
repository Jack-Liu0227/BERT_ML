"""
Machine learning model trainer implementation
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from .base_trainer import BaseTrainer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold

def now():
    """Return current time string for logging."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 参数网格（中英注释）
PARAM_GRIDS = {
    'rf': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [3, 4, 5]
    },
    'gbt': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'svr': 
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.5]
        }
    ,
    'knn': {
        'n_neighbors': [2, 3, 4, 5]
    }
}

def get_base_model(model_type: str) -> Any:
    """
    根据模型类型返回基础模型实例
    Get base model instance by model_type
    """
    model_type = model_type.lower()
    model_map = {
        'rf': RandomForestRegressor,
        'gbt': GradientBoostingRegressor,
        'svr': SVR,
        'knn': KNeighborsRegressor,
        'ann': MLPRegressor,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic': ElasticNet,
        'krr': KernelRidge,
        'xgb': xgb.XGBRegressor,
        'lightgbm': lgb.LGBMRegressor,
        'catboost': cb.CatBoostRegressor
    }
    model_class = model_map.get(model_type)
    
    if model_type == 'xgb' and xgb is None:
        raise ImportError("XGBoost is not installed. Please install it using 'pip install xgboost'")
    if model_type == 'lightgbm' and lgb is None:
        raise ImportError("LightGBM is not installed. Please install it using 'pip install lightgbm'")
    if model_type == 'catboost' and cb is None:
        raise ImportError("CatBoost is not installed. Please install it using 'pip install catboost'")

    if model_class:
        return model_class()
    else:
        raise ValueError(f'Unknown model type for get_base_model: {model_type}')

class MLTrainer(BaseTrainer):
    """
    Trainer for machine learning models (XGBoost, RandomForest, etc.)
    
    Args:
        model_type (str): Type of model ('xgb', 'rf', 'svr', 'ridge', 'lasso', 'elastic')
        result_dir (str): Directory to save training results
        model_name (str): Name of the model
        model_params (Dict[str, Any]): Model parameters
        training_params (Dict[str, Any]): Training parameters
    """
    def __init__(self,
                 model_type: str,
                 result_dir: str,
                 model_name: str,
                 model_params: Optional[Dict[str, Any]] = None,
                 training_params: Optional[Dict[str, Any]] = None,
                 is_cv: bool = False):
        # 初始化父类
        super().__init__(
            model=None,  # Will be created in _create_model
            result_dir=result_dir,
            model_name=model_name,
            target_names=None, # Will be set in train method
            early_stopping_patience=training_params.get('early_stopping_patience', 1000) if training_params else 1000,
            early_stopping_delta=training_params.get('early_stopping_delta', 1e-4) if training_params else 1e-4
        )
        
        self.model_type = model_type
        # 合并默认参数和用户自定义参数，用户参数优先
        default_params = self._get_default_params()
        self.model_params = {**default_params, **(model_params or {})}
        self.training_params = training_params or {}
        
        # Create model
        self.model = None # Model will be created in train method
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Save training configuration
        self.config = {
            'model_type': model_type,
            'model_params': self.model_params,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_delta': self.early_stopping_delta
        }
        if not is_cv:
            self._save_config()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default training parameters for each model type
        获取每种模型类型的默认训练参数
        Returns:
            Dict[str, Any]: Default parameters
        """
        if self.model_type.lower() == 'xgb':
            # XGBoost 默认参数
            return {
                'n_estimators': 100,  # 基学习器数量 (number of trees)
                'max_depth': 6,       # 树最大深度 (maximum tree depth)
                'learning_rate': 0.1, # 学习率 (learning rate)
                'subsample': 0.8,     # 子样本比例 (subsample ratio)
                'colsample_bytree': 0.8, # 每棵树的特征采样比例 (feature subsample)
                'random_state': 42    # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'rf':
            # 随机森林默认参数（参考notebook参数网格）
            return {
                'n_estimators': 100,      # 树数量 (number of trees)
                'max_depth': 3,          # 最大深度 (maximum depth)
                'max_features': 'sqrt',  # 最大特征数 (max features)
                'min_samples_split': 3,  # 内部节点再划分所需最小样本数 (min samples to split)
                'min_samples_leaf': 4,   # 叶子节点最少样本数 (min samples per leaf)
                'random_state': 42       # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'gbt':
            # 梯度提升树默认参数
            return {
                'n_estimators': 100,  # 基学习器数量 (number of boosting stages)
                'max_depth': 6,       # 最大深度 (maximum depth)
                'learning_rate': 0.1, # 学习率 (learning rate)
                'subsample': 0.8,     # 子样本比例 (subsample ratio)
                'random_state': 42    # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'ann':
            # 多层感知机默认参数
            return {
                'hidden_layer_sizes': (256,128), # 隐藏层结构 (hidden layer sizes)
                'activation': 'relu',        # 激活函数 (activation function)
                'solver': 'adam',            # 优化器 (optimizer)
                'max_iter': 200,            # 最大迭代次数 (max iterations)
                'random_state': 42           # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'knn':
            # K近邻回归默认参数（参考notebook参数网格）
            return {
                'n_neighbors': 2,    # 邻居数 (number of neighbors)
                'weights': 'uniform', # 权重方式 (weight function)
                'algorithm': 'auto'  # 近邻算法 (algorithm)
            }
        elif self.model_type.lower() == 'svr':
            # 支持向量回归默认参数（参考notebook参数网格）
            return {
                'C': 1,              # 惩罚系数 (regularization parameter)
                'kernel': 'rbf',     # 核函数 (kernel type)
                'epsilon': 0.001     # epsilon-tube (epsilon)
            }
        elif self.model_type.lower() == 'ridge':
            # 岭回归默认参数
            return {
                'alpha': 1.0,       # 正则化强度 (regularization strength)
                'solver': 'auto',   # 求解器 (solver)
                'random_state': 42  # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'lasso':
            # Lasso回归默认参数
            return {
                'alpha': 1.0,       # 正则化强度 (regularization strength)
                'random_state': 42  # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'elastic':
            # 弹性网回归默认参数
            return {
                'alpha': 1.0,       # 正则化强度 (regularization strength)
                'l1_ratio': 0.5,    # L1与L2的权重比 (L1 ratio)
                'random_state': 42  # 随机种子 (random seed)
            }
        elif self.model_type.lower() == 'krr':
            # 核岭回归默认参数
            return {
                'alpha': 1.0,       # 正则化强度 (regularization strength)
                'kernel': 'rbf',    # 核函数 (kernel type)
                'gamma': None       # 核系数 (kernel coefficient)
            }
        elif self.model_type.lower() == 'lightgbm':
            # LightGBM 默认参数
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
        else:
            return {}
    
    def _create_model(self) -> Any:
        """
        Create model based on model type, correctly handling single vs. multi-target regression.
        
        Returns:
            Any: Model instance
        """
        is_multi_target = len(self.target_names) > 1
        model_type = self.model_type.lower()

        # Models that support multi-output natively or are handled specially
        if model_type == 'xgb':
            return xgb.XGBRegressor(**self.model_params)
        elif model_type == 'rf':
            return RandomForestRegressor(**self.model_params)
        elif model_type == 'lightgbm':
            # LightGBM doesn't support multi-output natively, use MultiOutputRegressor
            base_model = lgb.LGBMRegressor(**self.model_params)
            if is_multi_target:
                return MultiOutputRegressor(base_model)
            else:
                return base_model
        elif model_type == 'catboost':
            # CatBoost supports multi-output with loss_function='MultiRMSE'
            if is_multi_target:
                # Set multi-regression parameters for CatBoost
                catboost_params = self.model_params.copy()
                catboost_params['loss_function'] = 'MultiRMSE'
                return cb.CatBoostRegressor(**catboost_params)
            else:
                return cb.CatBoostRegressor(**self.model_params)

        # Base models that require a wrapper for multi-target regression
        base_model_map = {
            'gbt': GradientBoostingRegressor,
            'ann': MLPRegressor,
            'krr': KernelRidge,
            'knn': KNeighborsRegressor,
            'svr': SVR,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic': ElasticNet
        }

        if model_type in base_model_map:
            base_model_instance = base_model_map[model_type](**self.model_params)
            if is_multi_target:
                # Wrap the base model for multi-target regression
                return MultiOutputRegressor(base_model_instance)
            else:
                # Use the base model directly for single-target regression
                return base_model_instance
        
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _save_config(self):
        """Save training configuration"""
        config_path = os.path.join(self.result_dir, f'{self.model_name}_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
    
    def train(self, train_data: Dict[str, np.ndarray], val_data: Dict[str, np.ndarray], target_names: List[str], is_cv: bool = False) -> Tuple[Any, Dict[str, List[float]]]:
        """
        Train the model
        
        Args:
            train_data (Dict[str, np.ndarray]): Training data.
            val_data (Dict[str, np.ndarray]): Validation data.
            target_names (List[str]): List of target column names.
            
        Returns:
            Tuple[Any, Dict[str, List[float]]]: The trained model and training history.
        """
        self.target_names = target_names
        self.model = self._create_model()

        print(f"\nTraining {self.model_type} model...")
        start_time = time.time()
        
        # For single-target regression, sklearn expects y to be a 1D array
        train_y_to_fit = train_data['y']
        if train_y_to_fit.shape[1] == 1:
            train_y_to_fit = train_y_to_fit.ravel()

        # Train model
        self.model.fit(train_data['X'], train_y_to_fit)
        
        # Get predictions and ensure they are 2D
        train_pred = self.model.predict(train_data['X'])
        if train_pred.ndim == 1:
            train_pred = train_pred.reshape(-1, 1)
            
        val_pred = self.model.predict(val_data['X'])
        if val_pred.ndim == 1:
            val_pred = val_pred.reshape(-1, 1)
        
        # Compute metrics
        train_metrics = self.compute_metrics(train_data['y'], train_pred, prefix='train_')
        val_metrics = self.compute_metrics(val_data['y'], val_pred, prefix='val_')

        current_val_loss = val_metrics['val_rmse']
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.save_checkpoint(0, is_best=True, is_cv=is_cv)

        history = {key: [value] for key, value in {**train_metrics, **val_metrics}.items()}

        # For MLP, add detailed training history
        if self.model_type.lower() == 'ann':
            if isinstance(self.model, MultiOutputRegressor):
                # For multi-output, aggregate history from the first estimator as representative
                first_estimator = self.model.estimators_[0]
                history.update({
                    'loss_curve_': first_estimator.loss_curve_,
                    'best_loss_': first_estimator.best_loss_,
                    'n_iter_': first_estimator.n_iter_
                })
            else:
                # For single-output, access attributes directly
                history.update({
                    'loss_curve_': self.model.loss_curve_,
                    'best_loss_': self.model.best_loss_,
                    'n_iter_': self.model.n_iter_
                })

        train_time = time.time() - start_time
        print(f"[{self.model_name}] {time.strftime('%Y-%m-%d %H:%M:%S')} Training completed in {train_time:.2f} seconds")
        print(f"[{self.model_name}] Final validation RMSE: {history['val_rmse'][-1]:.4f}")
        
        return self.model, history
    
    def validate(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Validate the model
        
        Returns:
            Dict[str, Union[float, np.ndarray]]: 
                A dictionary containing validation metrics and data.
        """
        val_pred = self.predict(self.val_data['X'])
        val_true = self.val_data['y']
        # 保证val_pred和val_true都是2d
        if val_pred.ndim == 1:
            val_pred = val_pred.reshape(-1, 1)
        if val_true.ndim == 1:
            val_true = val_true.reshape(-1, 1)
        val_loss = mean_squared_error(val_true, val_pred)
        # Compute R² score for each target
        r2_scores = []
        for i in range(val_true.shape[1]):
            target_pred = val_pred[:, i]
            target_true = val_true[:, i]
            r2 = r2_score(target_true, target_pred)
            r2_scores.append(r2)
        return {
            'loss': val_loss,
            'r2': np.mean(r2_scores),
            'preds': val_pred,
            'targets': val_true
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.model_type == 'xgb':
            # 使用XGBRegressor的predict方法，无需DMatrix
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_cv: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number.
            is_best (bool): Whether this is the best model so far.
            is_cv (bool): Whether the save is for a cross-validation best model.
        """
        if not is_best:
            return

        # Do not save individual fold models during cross-validation
        if is_cv:
            return

        suffix = "_best"
        model_filename = f'{self.model_name}{suffix}'
        
        checkpoint_dir = os.path.join(self.result_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.model_type == 'xgb':
            model_path = os.path.join(checkpoint_dir, f'{model_filename}.json')
            self.model.save_model(model_path)
        else:
            model_path = os.path.join(checkpoint_dir, f'{model_filename}.pkl')
            import joblib
            joblib.dump(self.model, model_path)
        
        print(f"Saved best model to: {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        print(f"\nLoading model from: {checkpoint_path}")
        
        if self.model_type == 'xgb':
            # 使用XGBRegressor的load_model方法
            self.model = xgb.XGBRegressor()
            self.model.load_model(checkpoint_path)
        else:
            # Load other ML models
            import joblib
            self.model = joblib.load(checkpoint_path)
        
        # Validate loaded model
        val_metrics = self.validate()
        print(f"Loaded model validation loss: {val_metrics['loss']:.4f}")
        print(f"Loaded model validation R²: {val_metrics['r2']:.4f}")
    
    def train_cross_validation(self,
                             num_folds: int,
                             fold_data: List[Dict[str, np.ndarray]],
                             callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Train model using cross-validation
        
        Args:
            num_folds (int): Number of cross-validation folds
            fold_data (List[Dict[str, np.ndarray]]): List of fold data dictionaries
            callbacks (Optional[List[Callable]]): List of callback functions
            
        Returns:
            Dict[str, Any]: Cross-validation results, including fold predictions.
        """
        all_fold_preds = []
        all_fold_targets = []
        cv_results = {
            'fold_metrics': [],
            'best_val_loss': float('inf'),
            'best_fold': -1,
        }
        
        print(f"\n[{self.model_name}] {time.strftime('%Y-%m-%d %H:%M:%S')} Starting {num_folds}-fold cross-validation...")
        start_time = time.time()
        
        for fold in range(num_folds):
            # Create a fresh model for each fold, using the best params if found by GridSearchCV
            self.model = self._create_model() 

            # Update data for current fold
            train_X = fold_data[fold]['train_X']
            train_y = fold_data[fold]['train_y']
            val_X = fold_data[fold]['val_X']
            val_y = fold_data[fold]['val_y']

            # For single-target regression, sklearn expects y to be a 1D array
            if train_y.shape[1] == 1:
                train_y = train_y.ravel()

            # Train model for this fold
            self.model.fit(train_X, train_y)
            
            # Validate and store fold results
            val_pred = self.model.predict(val_X)
            if val_pred.ndim == 1:
                val_pred = val_pred.reshape(-1, 1)
            val_loss = mean_squared_error(val_y, val_pred)
            val_r2 = r2_score(val_y, val_pred)
            
            print(f"[{self.model_name}] Fold {fold+1}/{num_folds} | Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")

            cv_results['fold_metrics'].append({
                'fold': fold,
                'val_loss': val_loss,
                'val_r2': val_r2
            })
            
            # Store predictions from the fold for box plot analysis
            all_fold_preds.append(val_pred)
            all_fold_targets.append(val_y)
            
            # Update overall best model
            if val_loss < cv_results['best_val_loss']:
                cv_results['best_val_loss'] = val_loss
                cv_results['best_fold'] = fold
                # Save overall best model checkpoint
                self.save_checkpoint(epoch=fold, is_best=True, is_cv=True)
        
        cv_time = time.time() - start_time
        print(f"\n[{self.model_name}] Cross-validation completed in {cv_time:.2f} seconds.")
        print(f"[{self.model_name}] Best model from fold {cv_results['best_fold']+1} with validation loss: {cv_results['best_val_loss']:.4f}")

        # Structure predictions for evaluator
        cv_results['predictions'] = [{'true': target, 'pred': pred} for target, pred in zip(all_fold_targets, all_fold_preds)]
        
        return cv_results
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Compute metrics for predictions, allowing for multi-target cases.
        
        Args:
            y_true (np.ndarray): True values, shape (n_samples, n_targets)
            y_pred (np.ndarray): Predicted values, shape (n_samples, n_targets)
            prefix (str): Prefix to add to metric names (e.g., 'train_')
            
        Returns:
            Dict[str, float]: Dictionary of metrics for each target and overall.
        """
        metrics = {}
        y_true = np.atleast_2d(y_true)
        y_pred = np.atleast_2d(y_pred)
        
        # Overall metrics
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f'{prefix}mae'] = np.mean(np.abs(y_true - y_pred))
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
        
        # Per-target metrics
        for i, name in enumerate(self.target_names):
            target_true = y_true[:, i]
            target_pred = y_pred[:, i]
            metrics[f'{prefix}{name}_rmse'] = np.sqrt(mean_squared_error(target_true, target_pred))
            metrics[f'{prefix}{name}_mae'] = np.mean(np.abs(target_true - target_pred))
            metrics[f'{prefix}{name}_r2'] = r2_score(target_true, target_pred)
            
        return metrics

    @staticmethod
    def search_best_params(model_type, X, y, param_grid, num_folds=5, n_jobs=-1):
        """
        Performs grid search to find the best hyperparameters for a given model type.

        Args:
            model_type (str): The type of model to tune ('xgb', 'rf', etc.).
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            param_grid (Dict): The dictionary of hyperparameters to search.
            num_folds (int): The number of folds for cross-validation.
            n_jobs (int): The number of jobs to run in parallel for GridSearchCV.

        Returns:
            Tuple[Dict, float, pd.DataFrame]: Best parameters, best score, and CV results.
        """
        print(f"[{now()}] Starting GridSearchCV for {model_type.upper()}...")
        base_model = get_base_model(model_type)

        is_multi_output = y.ndim > 1 and y.shape[1] > 1
        
        # Models that do not natively support multi-output regression
        # Of the scikit-learn models we use, only RandomForestRegressor supports it natively.
        needs_multi_output_wrapper = model_type.lower() not in ['rf', 'xgb']

        if is_multi_output and needs_multi_output_wrapper:
            print(f"[{now()}] Using MultiOutputRegressor for multi-target prediction with {model_type.upper()}.")
            model_to_tune = MultiOutputRegressor(base_model)
            # Add 'estimator__' prefix to param_grid keys for GridSearchCV
            if isinstance(param_grid, list): # handle list of dicts for params
                param_grid_for_search = [{f'estimator__{k}': v for k, v in d.items()} for d in param_grid]
            else:
                param_grid_for_search = {f'estimator__{k}': v for k, v in param_grid.items()}
        else:
            model_to_tune = base_model
            param_grid_for_search = param_grid

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=model_to_tune,
            param_grid=param_grid_for_search,
            scoring='neg_mean_squared_error',
            cv=kf,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # For multi-output models not needing a wrapper, y is passed as is (n_samples, n_targets).
        # For wrapped models, y is also passed as is.
        # For single-output, y is squeezed to (n_samples,).
        y_to_fit = y.squeeze() if not is_multi_output else y
        grid_search.fit(X, y_to_fit)
        
        best_params = grid_search.best_params_
        # If the model was wrapped, remove the 'estimator__' prefix from the keys
        if is_multi_output and needs_multi_output_wrapper:
            best_params = {key.replace('estimator__', ''): value for key, value in best_params.items()}

        print(f"[{now()}] Best parameters found: {best_params}")
        print(f"[{now()}] Best score (neg_mean_squared_error): {grid_search.best_score_}")
        
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        
        return best_params, grid_search.best_score_, cv_results_df
