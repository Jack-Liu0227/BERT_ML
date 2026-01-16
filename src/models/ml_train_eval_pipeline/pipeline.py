"""
Main training and evaluation pipeline for ML models.
"""
import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    # 尝试相对导入（当作为模块运行时）
    from .utils import now
    from .data_loader import load_data
    from ..trainers.ml_trainer import MLTrainer
    from ..evaluators.ml_evaluator import MLEvaluator
    from ..evaluators.shap_analyzer import MLShapAnalyzer
except ImportError:
    # 尝试直接导入（当直接运行时）
    try:
        from utils import now
        from data_loader import load_data
        from src.models.trainers.ml_trainer import MLTrainer
        from src.models.evaluators.ml_evaluator import MLEvaluator
        from src.models.evaluators.shap_analyzer import MLShapAnalyzer
    except ImportError:
        # 使用完整路径导入
        from src.models.ml_train_eval_pipeline.utils import now
        from src.models.ml_train_eval_pipeline.data_loader import load_data
        from src.models.trainers.ml_trainer import MLTrainer
        from src.models.evaluators.ml_evaluator import MLEvaluator
        from src.models.evaluators.shap_analyzer import MLShapAnalyzer

# Map pipeline-facing model names to internal trainer model names
MODEL_TYPE_MAP = {
    'xgboost': 'xgb',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'sklearn_rf': 'rf',
    'sklearn_svr': 'svr',
    'mlp': 'ann'
}

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

class MLTrainingPipeline:
    def __init__(self, args):
        self.args = args
        self.trainer_model_type = MODEL_TYPE_MAP.get(args.model_type, args.model_type)
        self.train_val_data = None
        self.test_data = None
        self.feature_names = None
        self.scaler_X = None
        self.scaler_y = None
        self.best_model_path = None
        self.best_model = None

    def run(self):
        """Main entry point to run the pipeline."""
        self._prepare_data()
        
        if self.args.use_optuna:
            self._run_optuna()
        else:
            self._run_standard_training()

    def _prepare_data(self):
        """Load, preprocess, and save scalers."""
        print(f"[{now()}] Preparing data...")
        self.train_val_data, self.test_data, self.feature_names = load_data(
            data_file=self.args.data_file,
            target_columns=self.args.target_columns,
            test_size=self.args.test_size,
            random_state=self.args.random_state,
            processing_cols=self.args.processing_cols,
            use_composition_feature=self.args.use_composition_feature,
            other_features_name=self.args.other_features_name,
        )
        
        self.scaler_X = StandardScaler().fit(self.train_val_data['X'])
        self.train_val_data['X'] = self.scaler_X.transform(self.train_val_data['X']).astype(np.float32)
        self.test_data['X'] = self.scaler_X.transform(self.test_data['X']).astype(np.float32)

        self.scaler_y = StandardScaler().fit(self.train_val_data['y'])
        self.train_val_data['y'] = self.scaler_y.transform(self.train_val_data['y']).astype(np.float32)
        self.test_data['y'] = self.scaler_y.transform(self.test_data['y']).astype(np.float32)

        os.makedirs(self.args.result_dir, exist_ok=True)
        joblib.dump(self.scaler_X, os.path.join(self.args.result_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.args.result_dir, 'scaler_y.pkl'))
        print(f"[{now()}] Scaler for X and y saved to: {self.args.result_dir}")

    def _run_standard_training(self):
        """Run training with or without cross-validation."""
        if self.args.cross_validate:
            self._run_cross_validation()
        else:
            self._train_final_model()

        if self.args.evaluate_after_train and self.best_model:
            self._evaluate_best_model()

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        Performs k-fold cross-validation for a given set of hyperparameters.
        """
        print(f"[{now()}] Trial {trial.number} starting...")

        # Define hyperparameter search space based on model_type
        model_params = {}
        if self.trainer_model_type == 'xgb':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'random_state': self.args.random_state,
                'n_jobs': -1
            }
        elif self.trainer_model_type == 'lightgbm':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'random_state': self.args.random_state,
                'n_jobs': -1
            }
        elif self.trainer_model_type == 'catboost':
            model_params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_seed': self.args.random_state,
                'verbose': 0,
                'thread_count': -1
            }
        elif self.trainer_model_type == 'rf':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'random_state': self.args.random_state,
                'n_jobs': -1
            }
        elif self.trainer_model_type == 'svr':
            model_params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
            }
        elif self.trainer_model_type == 'ann':
            # For MLP, we might suggest architecture parameters
            # For simplicity, let's suggest some common ones
            model_params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': self.args.mlp_max_iter, # Use fixed max_iter from args
                'random_state': self.args.random_state,
            }
        
        print(f"[{now()}] Trial {trial.number} starting with params: {model_params}")
        
        scores = []
        trial_fold_metrics = [] # Initialize trial_fold_metrics
        
        km = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=self.args.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(km.split(self.train_val_data['X'])):
            fold_train_data = {'X': self.train_val_data['X'][train_idx], 'y': self.train_val_data['y'][train_idx]}
            fold_val_data = {'X': self.train_val_data['X'][val_idx], 'y': self.train_val_data['y'][val_idx]}
            
            trainer = MLTrainer(
                model_type=self.trainer_model_type,
                result_dir=self.args.result_dir,
                model_name=f"optuna_trial_{trial.number}_fold_{fold+1}",
                model_params=model_params,
                is_cv=True # Prevent saving config for each fold
            )
            
            model, _ = trainer.train(
                train_data=fold_train_data,
                val_data=fold_val_data,
                target_names=self.args.target_columns,
                is_cv=True # Inform the trainer it's a CV run
            )
            
            # Predict validation set
            y_val_pred = model.predict(fold_val_data['X'])
            
            # Ensure 2D for inverse transform
            if y_val_pred.ndim == 1:
                y_val_pred = y_val_pred.reshape(-1, 1)
            
            y_val_true_scaled = fold_val_data['y']
            if y_val_true_scaled.ndim == 1:
                y_val_true_scaled = y_val_true_scaled.reshape(-1, 1)

            # Inverse transform to original scale for metric calculation
            y_val_pred_orig = self.scaler_y.inverse_transform(y_val_pred)
            y_val_true_orig = self.scaler_y.inverse_transform(y_val_true_scaled)
            
            fold_metrics_for_trial = {}
            # Calculate R2, RMSE, MAE per target
            for i, target_name in enumerate(self.args.target_columns):
                y_true = y_val_true_orig[:, i]
                y_pred = y_val_pred_orig[:, i]
                
                fold_metrics_for_trial[f"{target_name}_r2"] = r2_score(y_true, y_pred)
                fold_metrics_for_trial[f"{target_name}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                fold_metrics_for_trial[f"{target_name}_mae"] = mean_absolute_error(y_true, y_pred)
            
            # --- Save Predictions for this Trial Fold ---
            trial_fold_dir = os.path.join(self.args.result_dir, "predictions", "optuna_trials", f"trial_{trial.number}", f"fold_{fold+1}")
            os.makedirs(trial_fold_dir, exist_ok=True)
            
            # Predict (Inverse transform)
            # Train (fold)
            y_train_pred = model.predict(fold_train_data['X'])
            if y_train_pred.ndim == 1: y_train_pred = y_train_pred.reshape(-1, 1)
            y_train_pred_orig = self.scaler_y.inverse_transform(y_train_pred)
            y_train_true_orig = self.scaler_y.inverse_transform(fold_train_data['y'].reshape(-1, 1) if fold_train_data['y'].ndim==1 else fold_train_data['y'])
            
            # Val (already computed as y_val_pred_orig, y_val_true_orig)
            
            # Test (global)
            y_test_pred = model.predict(self.test_data['X'])
            if y_test_pred.ndim == 1: y_test_pred = y_test_pred.reshape(-1, 1)
            y_test_pred_orig = self.scaler_y.inverse_transform(y_test_pred)
            y_test_true_orig = self.scaler_y.inverse_transform(self.test_data['y'].reshape(-1, 1) if self.test_data['y'].ndim==1 else self.test_data['y'])
            
            # Handle IDs if present
            train_ids, val_ids = None, None
            if 'ids' in self.train_val_data:
                # Assuming ids is a numpy array or pandas Series/DataFrame that supports indexing
                ids_data = self.train_val_data['ids']
                if hasattr(ids_data, 'iloc'):
                    train_ids = ids_data.iloc[train_idx]
                    val_ids = ids_data.iloc[val_idx]
                else:
                    # Numpy array or list
                    train_ids = np.array(ids_data)[train_idx]
                    val_ids = np.array(ids_data)[val_idx]

            # Construct DataFrames
            df_train = pd.DataFrame()
            if train_ids is not None: df_train['ID'] = train_ids
            df_train['Dataset'] = 'Train'
            
            df_val = pd.DataFrame()
            if val_ids is not None: df_val['ID'] = val_ids
            df_val['Dataset'] = 'Validation'
            
            df_test = pd.DataFrame()
            if 'ids' in self.test_data: df_test['ID'] = self.test_data['ids']
            df_test['Dataset'] = 'Test'
            
            for i, target in enumerate(self.args.target_columns):
                df_train[f"{target}_Actual"] = y_train_true_orig[:, i]
                df_train[f"{target}_Predicted"] = y_train_pred_orig[:, i]
                
                df_val[f"{target}_Actual"] = y_val_true_orig[:, i]
                df_val[f"{target}_Predicted"] = y_val_pred_orig[:, i]
                
                df_test[f"{target}_Actual"] = y_test_true_orig[:, i]
                df_test[f"{target}_Predicted"] = y_test_pred_orig[:, i]
            
            # Combine and Save
            df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
            df_all.to_csv(os.path.join(trial_fold_dir, "all_predictions.csv"), index=False)

            trial_fold_metrics.append(fold_metrics_for_trial) # Append metrics to trial_fold_metrics

            # Save trial metrics incrementally (so we have them even if pruned)
            trial_dir = os.path.join(self.args.result_dir, "predictions", "optuna_trials", f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            cv_csv_path = os.path.join(trial_dir, 'cross_validation_results.csv')
            pd.DataFrame(trial_fold_metrics).to_csv(cv_csv_path, index=False)
            
            # For Optuna, we typically optimize a single metric. Let's use average R2.
            avg_r2 = np.mean([fold_metrics_for_trial[f"{t}_r2"] for t in self.args.target_columns])
            scores.append(avg_r2)
            
            # Report intermediate value to Optuna
            trial.report(avg_r2, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # After the loop
        print(f"[{now()}] Saved detailed cross-validation results for trial {trial.number} to: {cv_csv_path}")

        # Return the average score across folds for Optuna to minimize/maximize
        # Optuna maximizes by default, so we return the mean R2
        mean_score = np.mean(scores)
        print(f"[{now()}] Trial {trial.number} finished with mean R2: {mean_score}")
        return mean_score

    def _run_optuna(self):
        """Runs Optuna hyperparameter optimization."""
        print(f"[{now()}] Starting Optuna hyperparameter optimization...")
        
        study_name = f"{self.trainer_model_type}_optimization"
        storage_name = f"sqlite:///{self.args.result_dir}/{study_name}.db"
        
        study = optuna.create_study(
            direction="maximize", # We want to maximize R2
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True
        )
        
        study.optimize(self._objective, n_trials=self.args.optuna_n_trials, timeout=self.args.optuna_timeout)
        
        print(f"[{now()}] Optuna optimization finished.")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial:")
        trial = study.best_trial
        
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        # Save best parameters
        best_params_path = os.path.join(self.args.result_dir, 'best_optuna_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(trial.params, f, indent=4)
        print(f"[{now()}] Best Optuna parameters saved to: {best_params_path}")

        # After Optuna, train the final model with the best parameters
        print(f"[{now()}] Training final model with best Optuna parameters...")
        self._train_final_model(model_params=trial.params)

    def _run_cross_validation(self, model_params: Dict[str, Any] = None, cv_results_filename: str = 'cross_validation_results.csv'):
        """Performs k-fold cross-validation."""
        print(f"[{now()}] Starting cross-validation...")
        kf = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=self.args.random_state)
        
        fold_metrics = []
        best_fold_score = -np.inf 
        
        # Determine model parameters
        if model_params is None:
            model_params = {}
            if self.args.model_type == 'mlp':
                model_params = {'max_iter': self.args.mlp_max_iter}


        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_val_data['X'])):
            print(f"[{now()}] --- Starting Fold {fold+1}/{self.args.num_folds} ---")
            
            fold_train_data = {'X': self.train_val_data['X'][train_idx], 'y': self.train_val_data['y'][train_idx]}
            fold_val_data = {'X': self.train_val_data['X'][val_idx], 'y': self.train_val_data['y'][val_idx]}
            
            # Handle IDs
            fold_train_ids = None
            fold_val_ids = None
            if 'ids' in self.train_val_data:
                # assuming ids is a numpy array or list
                fold_train_ids = np.array(self.train_val_data['ids'])[train_idx]
                fold_val_ids = np.array(self.train_val_data['ids'])[val_idx]
                fold_train_data['ids'] = fold_train_ids
                fold_val_data['ids'] = fold_val_ids

            trainer = MLTrainer(
                model_type=self.trainer_model_type,
                result_dir=self.args.result_dir,
                model_name=f"ml_model_fold_{fold+1}",
                model_params=model_params,
                is_cv=True  # Prevent saving config for each fold
            )
            
            model, history = trainer.train(
                train_data=fold_train_data,
                val_data=fold_val_data,
                target_names=self.args.target_columns,
                is_cv=True  # Inform the trainer it's a CV run
            )
            
            # Calculate metrics on ORIGINAL scale (inverse transform)
            # Predict validation set
            y_val_pred = model.predict(fold_val_data['X'])
            
            # Ensure 2D for inverse transform
            if y_val_pred.ndim == 1:
                y_val_pred = y_val_pred.reshape(-1, 1)
            
            y_val_true_scaled = fold_val_data['y']
            if y_val_true_scaled.ndim == 1:
                y_val_true_scaled = y_val_true_scaled.reshape(-1, 1)

            # Inverse transform
            y_val_pred_orig = self.scaler_y.inverse_transform(y_val_pred)
            y_val_true_orig = self.scaler_y.inverse_transform(y_val_true_scaled)
            
            metrics_for_df = {}
            
            # Iterate over targets to calculate metrics
            for i, target_name in enumerate(self.args.target_columns):
                y_true = y_val_true_orig[:, i]
                y_pred = y_val_pred_orig[:, i]
                
                metrics_for_df[f"{target_name}_r2"] = r2_score(y_true, y_pred)
                metrics_for_df[f"{target_name}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics_for_df[f"{target_name}_mae"] = mean_absolute_error(y_true, y_pred)

            fold_metrics.append(metrics_for_df)
            
            # --- Save Predictions for this Fold (Train/Val/Test) ---
            # Only save if NOT using Optuna (to avoid clutter, as Optuna trials already save this)
            if not self.args.use_optuna:
                # Create directory for this fold's predictions
                fold_pred_dir = os.path.join(self.args.result_dir, "predictions", f"fold_{fold+1}")
                os.makedirs(fold_pred_dir, exist_ok=True)
                
                # We need predictions for Train and Test as well to create "all_predictions.csv"
                # Train (fold)
                y_train_pred_fold = model.predict(fold_train_data['X'])
                if y_train_pred_fold.ndim == 1: y_train_pred_fold = y_train_pred_fold.reshape(-1, 1)
                y_train_pred_orig_fold = self.scaler_y.inverse_transform(y_train_pred_fold)
                y_train_true_orig_fold = self.scaler_y.inverse_transform(fold_train_data['y'].reshape(-1, 1) if fold_train_data['y'].ndim==1 else fold_train_data['y'])

                # Test (global)
                y_test_pred_fold = model.predict(self.test_data['X'])
                if y_test_pred_fold.ndim == 1: y_test_pred_fold = y_test_pred_fold.reshape(-1, 1)
                y_test_pred_orig_fold = self.scaler_y.inverse_transform(y_test_pred_fold)
                y_test_true_orig_fold = self.scaler_y.inverse_transform(self.test_data['y'].reshape(-1, 1) if self.test_data['y'].ndim==1 else self.test_data['y'])

                # Construct DataFrame
                # Train DF
                df_train_fold = pd.DataFrame()
                if fold_train_ids is not None:
                    df_train_fold["ID"] = fold_train_ids
                df_train_fold["Dataset"] = "Train"
                for idx, target in enumerate(self.args.target_columns):
                    df_train_fold[f"{target}_Actual"] = y_train_true_orig_fold[:, idx]
                    df_train_fold[f"{target}_Predicted"] = y_train_pred_orig_fold[:, idx]
                
                # Val DF
                df_val_fold = pd.DataFrame()
                if fold_val_ids is not None:
                    df_val_fold["ID"] = fold_val_ids
                df_val_fold["Dataset"] = "Validation"
                for idx, target in enumerate(self.args.target_columns):
                    df_val_fold[f"{target}_Actual"] = y_val_true_orig[:, idx]
                    df_val_fold[f"{target}_Predicted"] = y_val_pred_orig[:, idx]
                
                # Test DF
                df_test_fold = pd.DataFrame()
                if 'ids' in self.test_data:
                    df_test_fold["ID"] = self.test_data['ids']
                df_test_fold["Dataset"] = "Test"
                for idx, target in enumerate(self.args.target_columns):
                    df_test_fold[f"{target}_Actual"] = y_test_true_orig_fold[:, idx]
                    df_test_fold[f"{target}_Predicted"] = y_test_pred_orig_fold[:, idx]
                
                # Combine
                df_all_fold = pd.concat([df_train_fold, df_val_fold, df_test_fold], ignore_index=True)
                
                # Save
                all_pred_path = os.path.join(fold_pred_dir, "all_predictions.csv")
                df_all_fold.to_csv(all_pred_path, index=False)
                print(f"[{now()}] Saved all predictions for cv fold {fold+1} to {all_pred_path}")

            
            # Determine best fold score for logging (using average R2 across targets if multi-target)
            current_r2 = np.mean([metrics_for_df[f"{t}_r2"] for t in self.args.target_columns])
            if current_r2 > best_fold_score:
                best_fold_score = current_r2
                # No longer save the best model from a single fold.
                # The final model will be trained on all data after CV.

        # Calculate Mean and Std
        df_results = pd.DataFrame(fold_metrics)
        avg_metrics = df_results.mean().to_dict()
        std_metrics = df_results.std().to_dict()
        
        print(f"[{now()}] Cross-validation average metrics: {avg_metrics}")
        
        # Save flat average metrics for simple access
        with open(os.path.join(self.args.result_dir, 'cv_avg_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        # Save detailed CV results to CSV
        cv_csv_path = os.path.join(self.args.result_dir, cv_results_filename)
        df_results.to_csv(cv_csv_path, index=False)
        print(f"[{now()}] Saved detailed cross-validation results to: {cv_csv_path}")

        # Construct structured CV results with Mean and Std for ModelComparator
        # Expected format: { target: { metric_mean: val, metric_std: val } }
        structured_cv_results = {}
        
        # Helper to parse key "Target_metric" or "metric" (for overall)
        # However, MLTrainer compute_metrics returns keys like "val_UTS(MPa)_r2" 
        # which became "UTS(MPa)_r2" in fold_metrics
        
        for key in avg_metrics.keys():
            # Try to identify target and metric
            # We know keys are like "{target}_{metric}" or just "{metric}" for overall
            # But here we only really care about per-target metrics for the report
            
            # Check if key ends with known metric
            known_metrics = ['r2', 'rmse', 'mae', 'mape']
            matched_metric = None
            for m in known_metrics:
                if key.endswith(f"_{m}"):
                    matched_metric = m
                    break
            
            if matched_metric:
                # Extract target name
                # key is "{target}_{metric}"
                target = key[:-(len(matched_metric)+1)]
                
                if target not in structured_cv_results:
                    structured_cv_results[target] = {}
                
                structured_cv_results[target][f"{matched_metric}_mean"] = avg_metrics[key]
                structured_cv_results[target][f"{matched_metric}_std"] = std_metrics.get(key, 0.0)
        
        # Also add overall metrics if needed (optional)
        
        # Save structured results
        cv_results_path = os.path.join(self.args.result_dir, 'cross_validation_results.json')
        with open(cv_results_path, 'w', encoding='utf-8') as f:
            json.dump(structured_cv_results, f, indent=4)
        print(f"[{now()}] Saved detailed cross-validation results to: {cv_results_path}")

        # After CV, always train the final model on all data for test set evaluation
        print(f"[{now()}] Cross-validation finished. Training final model on all data...")
        self._train_final_model(model_params)

    def _train_final_model(self, model_params: Dict[str, Any] = None):
        """Trains a single model on the full train_val dataset."""
        print(f"[{now()}] Training final model on full data...")

        # Use a distinct model_name for the final model to avoid overwriting
        final_model_name = "final_best_model"

        # Prepare model parameters if not provided
        if model_params is None:
            model_params = {}
            if self.args.model_type == 'mlp':
                model_params = {'max_iter': self.args.mlp_max_iter}

        trainer = MLTrainer(
            model_type=self.trainer_model_type,
            result_dir=self.args.result_dir,
            model_name=final_model_name,
            model_params=model_params
        )
        
        self.best_model, history = trainer.train(
            train_data=self.train_val_data,
            val_data=self.test_data, # Using test set for final validation
            target_names=self.args.target_columns
        )
        
        self.best_model_path = os.path.join(self.args.result_dir, f'{final_model_name}.pkl')
        joblib.dump(self.best_model, self.best_model_path)
        print(f"[{now()}] Saved final best model to {self.best_model_path}")

        # Save training and validation history for all model types
        if history:
            loss_dir = os.path.join(self.args.result_dir, 'Loss')
            os.makedirs(loss_dir, exist_ok=True)
            
            # For MLP models specifically
            if self.trainer_model_type == 'ann' and 'loss_curve_' in history:
                mlp_history = {
                    'loss_curve': convert_numpy_types(history['loss_curve_']),
                    'best_loss': float(history['best_loss_']),
                    'n_iter': int(history['n_iter_'])
                }
                # Save history to JSON
                history_path = os.path.join(loss_dir, 'final_mlp_training_history.json')
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(mlp_history, f, indent=4)
                print(f"[{now()}] Saved final MLP training history to {history_path}")

                # Plot and save loss curve
                plt.figure(figsize=(10, 6))
                plt.plot(mlp_history['loss_curve'])
                plt.title('MLP Training Loss Curve')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.grid(True)
                plot_path = os.path.join(loss_dir, 'final_mlp_loss_curve.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"[{now()}] Saved final MLP loss curve plot to {plot_path}")
            
            # For all model types - save all metrics including train and val losses/scores
            all_metrics = {k: convert_numpy_types(v) for k, v in history.items() if not k.endswith('_')}
            metrics_path = os.path.join(loss_dir, 'training_validation_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=4)
            print(f"[{now()}] Saved training and validation metrics to {metrics_path}")
            
            # If there are both training and validation metrics, plot them together
            train_metrics = {k: v for k, v in all_metrics.items() if not k.startswith('val_')}
            val_metrics = {k.replace('val_', ''): v for k, v in all_metrics.items() if k.startswith('val_')}
            
            if train_metrics and val_metrics and 'loss' in train_metrics and 'loss' in val_metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(train_metrics['loss'], label='Training Loss')
                plt.plot(val_metrics['loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(loss_dir, 'train_val_loss.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"[{now()}] Saved training and validation loss plot to {plot_path}")


    def _evaluate_best_model(self):
        """Evaluates the best model on the test set and saves predictions."""
        print(f"[{now()}] Evaluating the best model on the test set...")
        
        # Get train-validation split for evaluation and plotting
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_val_data['X'], 
            self.train_val_data['y'],
            test_size=0.2, 
            random_state=self.args.random_state
        )

        train_data_split = {'X': X_train, 'y': y_train}
        val_data_split = {'X': X_val, 'y': y_val}

        evaluator = MLEvaluator(
            result_dir=self.args.result_dir,
            model_name="final_model_evaluation", # Use a specific name for final eval
            target_names=self.args.target_columns
        )
        
        metrics = evaluator.evaluate(
            model=self.best_model,
            train_data=train_data_split,
            val_data=val_data_split,
            test_data=self.test_data,
            scaler_y=self.scaler_y,
            save_predictions=False # Defer prediction saving to _save_predictions
        )
        print(f"[{now()}] Final test set evaluation metrics: {metrics}")
        
        # Save final metrics
        final_metrics_path = os.path.join(self.args.result_dir, "final_evaluation_metrics.json")
        # We need to clean the metrics dictionary from the dummy train data
        final_metrics = {k: v for k, v in metrics.items() if 'train' not in k}
        with open(final_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"[{now()}] Saved final evaluation metrics to: {final_metrics_path}")

        # Save predictions for training, validation and test sets in a single file
        self._save_predictions()

        # SHAP Analysis
        if self.args.run_shap_analysis:
            analyzer = MLShapAnalyzer(
                model=self.best_model,
                feature_names=self.feature_names,
                target_names=self.args.target_columns,
                result_dir=self.args.result_dir
            )
            analyzer.analyze(self.test_data['X'])

    def _save_predictions(self):
        """Save predictions for train, validation and test sets in a single file."""
        print(f"[{now()}] Saving predictions for train, validation, and test sets...")

        # Check if IDs are available
        has_ids = 'ids' in self.train_val_data and 'ids' in self.test_data

        # Get train-validation split for saving predictions separately
        if has_ids:
            X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                self.train_val_data['X'],
                self.train_val_data['y'],
                self.train_val_data['ids'],
                test_size=0.2,
                random_state=self.args.random_state
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                self.train_val_data['X'],
                self.train_val_data['y'],
                test_size=0.2,
                random_state=self.args.random_state
            )
        
        # Get predictions
        y_train_pred = self.best_model.predict(X_train)
        y_val_pred = self.best_model.predict(X_val)
        y_test_pred = self.best_model.predict(self.test_data['X'])
        
        # Ensure all y arrays are 2D before inverse_transform (保证所有y相关数组为2维)
        def ensure_2d(arr):
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr
        y_train_true = ensure_2d(y_train)
        y_val_true = ensure_2d(y_val)
        y_test_true = ensure_2d(self.test_data['y'])
        y_train_pred = ensure_2d(y_train_pred)
        y_val_pred = ensure_2d(y_val_pred)
        y_test_pred = ensure_2d(y_test_pred)
        # Inverse transform predictions and actual values back to original scale
        y_train_true = self.scaler_y.inverse_transform(y_train_true)
        y_val_true = self.scaler_y.inverse_transform(y_val_true)
        y_test_true = self.scaler_y.inverse_transform(y_test_true)
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred)
        y_val_pred = self.scaler_y.inverse_transform(y_val_pred)
        y_test_pred = self.scaler_y.inverse_transform(y_test_pred)
        # Create DataFrames for each set
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

        # Add ID columns if available
        if has_ids:
            train_df["ID"] = ids_train
            val_df["ID"] = ids_val
            test_df["ID"] = self.test_data['ids']

        # Add actual and predicted values for each target
        for i, target_name in enumerate(self.args.target_columns):
            train_df[f"{target_name}_Actual"] = y_train_true[:, i]
            train_df[f"{target_name}_Predicted"] = y_train_pred[:, i]
            train_df["Dataset"] = "Train"
            val_df[f"{target_name}_Actual"] = y_val_true[:, i]
            val_df[f"{target_name}_Predicted"] = y_val_pred[:, i]
            val_df["Dataset"] = "Validation"
            test_df[f"{target_name}_Actual"] = y_test_true[:, i]
            test_df[f"{target_name}_Predicted"] = y_test_pred[:, i]
            test_df["Dataset"] = "Test"
        # Combine all sets
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # Save to CSV
        predictions_dir = os.path.join(self.args.result_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        predictions_path = os.path.join(predictions_dir, "all_predictions.csv")
        combined_df.to_csv(predictions_path, index=False)
        print(f"[{now()}] Saved combined predictions to: {predictions_path}")
        # Save each set separately (分别保存训练集、验证集、测试集)
        train_path = os.path.join(predictions_dir, "train_predictions.csv")
        val_path = os.path.join(predictions_dir, "val_predictions.csv")
        test_path = os.path.join(predictions_dir, "test_predictions.csv")
        train_df.to_csv(train_path, index=False)  # Save train set predictions
        val_df.to_csv(val_path, index=False)      # Save validation set predictions
        test_df.to_csv(test_path, index=False)    # Save test set predictions
        print(f"[{now()}] Saved train/val/test predictions to: {train_path}, {val_path}, {test_path}")

    def _run_optuna(self):
        """
        Runs hyperparameter optimization using Optuna with cross-validation and pruning.
        """
        print(f"[{now()}] Starting Optuna hyperparameter optimization...")
        print(f"[{now()}] Each trial will be evaluated using 3-fold cross-validation for robustness.")
        print(f"[{now()}] Pruning is enabled to stop unpromising trials early.")

        # A pruner object to automatically stop unpromising trials
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
        
        study = optuna.create_study(
            direction='maximize', 
            study_name=self.args.study_name, 
            pruner=pruner,
            load_if_exists=True
        )
        study.optimize(self._objective, n_trials=self.args.n_trials, n_jobs=1) # n_jobs=1 for thread safety with file-based operations

        print(f"[{now()}] Optimization finished.")
        print(f"[{now()}] Best trial finished with avg R2 score: {study.best_value} and params: {study.best_params}")
        
        best_params = study.best_params.copy()
        
        # Save all best params to a JSON file for record-keeping
        best_params_path = os.path.join(self.args.result_dir, "optuna_best_params.json")
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4)
        print(f"[{now()}] Saved best Optuna parameters to: {best_params_path}")
        
        # Save Optuna Study History (Trials)
        try:
            trials_df = study.trials_dataframe()
            history_path = os.path.join(self.args.result_dir, "optuna_study_history.csv")
            trials_df.to_csv(history_path, index=False)
            print(f"[{now()}] Saved Optuna study history to: {history_path}")
        except Exception as e:
            print(f"[{now()}] Warning: Failed to save Optuna study history. Error: {e}")
            
        # --- Aggregate Cross-Validation Results from All Trials ---
        try:
            print(f"[{now()}] Aggregating cross-validation results from all Optuna trials...")
            trials_dir = os.path.join(self.args.result_dir, "predictions", "optuna_trials")
            if os.path.exists(trials_dir):
                all_cv_results = []
                # Traverse directories to find cross_validation_results.csv
                for trial_folder in os.listdir(trials_dir):
                    if trial_folder.startswith("trial_"):
                        trial_path = os.path.join(trials_dir, trial_folder)
                        cv_file = os.path.join(trial_path, "cross_validation_results.csv")
                        if os.path.exists(cv_file):
                            df_cv = pd.read_csv(cv_file)
                            df_cv["Trial"] = trial_folder
                            all_cv_results.append(df_cv)
                
                if all_cv_results:
                    combined_cv_results = pd.concat(all_cv_results, ignore_index=True)
                    # Sort nicely
                    combined_cv_results.sort_values(by=["Trial", "fold"], inplace=True)
                    
                    summary_path = os.path.join(self.args.result_dir, "predictions", "optuna_cv_results_summary.csv")
                    combined_cv_results.to_csv(summary_path, index=False)
                    print(f"[{now()}] Saved aggregated Optuna CV results to: {summary_path}")
                    
                    # Also save to root cross_validation_results.csv as per user expectation for "all trials results"
                    root_cv_path = os.path.join(self.args.result_dir, "cross_validation_results.csv")
                    combined_cv_results.to_csv(root_cv_path, index=False)
                    print(f"[{now()}] Saved all trials CV results to root: {root_cv_path}")
                else:
                    print(f"[{now()}] No cross_validation_results.csv found in trial folders.")
            else:
                print(f"[{now()}] No optuna_trials directory found at {trials_dir}.")
        except Exception as e:
            print(f"[{now()}] Warning: Failed to aggregate Optuna CV results. Error: {e}")
        
        # Reconstruct and clean parameters for MLP final model training
        if self.args.model_type == 'mlp':
            # Optuna's best_params dictionary contains the raw suggested values
            # (e.g., 'n_layers', 'n_units_l0'). We need to convert this into
            # the 'hidden_layer_sizes' tuple that MLPRegressor expects.
            if 'n_layers' in best_params:
                n_layers = best_params.pop('n_layers')
                layers = []
                for i in range(n_layers):
                    # Use .pop() to remove the key, cleaning the dict for the next step
                    layers.append(best_params.pop(f"n_units_l{i}"))
                best_params['hidden_layer_sizes'] = tuple(layers)

            # Ensure max_iter is preserved in the final parameters
            if 'max_iter' not in best_params:
                best_params['max_iter'] = self.args.mlp_max_iter

        if self.args.cross_validate:
            # If cross-validation is enabled, run CV with best params to get uncertainty stats
            print(f"[{now()}] Running final cross-validation with best Optuna params for statistical reporting...")
            # Use a distinctive name so we don't overwrite the all-trials summary
            self._run_cross_validation(model_params=best_params, cv_results_filename='best_model_cross_validation_results.csv')
            
        # ALWAYS Train final model with best Optuna params to produce the standard best model predictions (train/val/test)
        print(f"[{now()}] Training final model with best Optuna params...")
        self._train_final_model(model_params=best_params)
        
        # Note: _train_final_model handles saving history/plots/metrics for the final training run.
        # However, _run_cross_validation -> _train_final_model saves them too.
        
        if self.args.evaluate_after_train:
            self._evaluate_best_model()

    def _save_training_history(self, history: Dict[str, Any]):
        """Saves the training history (losses, metrics) and plots curves."""
        if not history:
            return

        loss_dir = os.path.join(self.args.result_dir, 'Loss')
        os.makedirs(loss_dir, exist_ok=True)
        
        # For MLP models specifically, save detailed history
        if self.trainer_model_type == 'ann' and 'loss_curve_' in history:
            mlp_history = {
                'loss_curve': convert_numpy_types(history.get('loss_curve_', [])),
                'best_loss': convert_numpy_types(history.get('best_loss_')),
                'n_iter': convert_numpy_types(history.get('n_iter_'))
            }
            history_path = os.path.join(loss_dir, 'final_mlp_training_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(mlp_history, f, indent=4)
            print(f"[{now()}] Saved MLP training history to {history_path}")

            # Plot and save loss curve for MLP
            if mlp_history['loss_curve']:
                plt.figure(figsize=(10, 6))
                plt.plot(mlp_history['loss_curve'])
                plt.title('MLP Training Loss Curve')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.grid(True)
                plot_path = os.path.join(loss_dir, 'final_mlp_training_loss_curve.png')
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"[{now()}] Saved MLP loss curve plot to {plot_path}")
        
        # For all model types - save all metrics including train and val losses/scores
        all_metrics = {k: convert_numpy_types(v) for k, v in history.items() if not k.endswith('_')}
        metrics_path = os.path.join(loss_dir, 'training_validation_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"[{now()}] Saved training and validation metrics to {metrics_path}")
        
        # If there are both training and validation metrics, plot them together
        train_metrics = {k.replace('train_', ''): v for k, v in all_metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_', ''): v for k, v in all_metrics.items() if k.startswith('val_')}
        
        # We only want to plot the loss curve
        if 'loss' in train_metrics and 'loss' in val_metrics and isinstance(train_metrics['loss'], list):
            plt.figure(figsize=(10, 6))
            plt.plot(train_metrics['loss'], label='Training Loss')
            plt.plot(val_metrics['loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(loss_dir, 'train_val_loss.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"[{now()}] Saved training and validation loss plot to {plot_path}")

