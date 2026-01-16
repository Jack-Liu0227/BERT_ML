"""
Main training and evaluation pipeline for CNN models.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import optuna
import joblib
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any, Union, cast
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from .utils import now
from .data_loader import load_data
from ..base.alloys_cnn_nn import AlloyCnnV2
from ..trainers.trainer_factory import TrainerFactory
from ..evaluators.evaluator_factory import EvaluatorFactory
from ..evaluators.base_evaluator import BaseEvaluator
from ..evaluators.alloys_evaluator import AlloysEvaluator


class CnnTrainingPipeline:
    """Pipeline for training and evaluating CNN models."""
    def __init__(self, args):
        """Initialize the CNN training pipeline."""
        self.args = args
        self._validate_args()
        self.train_val_data = None
        self.test_data = None
        self.feature_names = None
        self.feature_scaler = None
        self.target_scaler = None
        self.best_model_state_dict = None
        self.best_model_path = None
        
        # Create necessary directories
        os.makedirs(self.args.result_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.args.result_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def _validate_args(self):
        """Validate JSON arguments."""
        try:
            json.loads(self.args.cnn_1d_config)
            json.loads(self.args.cnn_2d_config)
            if self.args.other_feature_configs.lower() != 'none':
                json.loads(self.args.other_feature_configs)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model configuration arguments: {e}")

    def run(self):
        """Main entry point to run the pipeline."""
        self._prepare_data()
        
        if self.args.use_optuna:
            self._run_optuna()
        else:
            # This logic block is for standard training and CV without optuna
            if self.args.cross_validate:
                self._run_cross_validation()
            else:
                self._train_final_model()

            if self.args.evaluate_after_train:
                self._evaluate_best_model()

    def _prepare_data(self):
        """Prepare data for training."""
        print(f"[{now()}] Preparing data...")
        print(f"[{now()}] Loading data from: {self.args.data_file}")
        
        # Load data
        self.train_val_data, self.test_data, self.feature_names = load_data(
            self.args.data_file,
            self.args.target_columns,
            test_size=self.args.test_size,
            random_state=self.args.random_state
        )
        
        print(f"[{now()}] Found {len(self.feature_names)} feature columns and {len(self.args.target_columns)} target columns.")
        
        # Scale features
        feature_scaler = StandardScaler()
        self.train_val_data['X'] = feature_scaler.fit_transform(self.train_val_data['X'])
        self.test_data['X'] = feature_scaler.transform(self.test_data['X'])
        self.feature_scaler = feature_scaler
        
        # Scale targets
        target_scaler = StandardScaler()
        self.train_val_data['y'] = target_scaler.fit_transform(self.train_val_data['y'])
        self.target_scaler = target_scaler
        
        # Note: we don't scale test_data['y'] yet, as we want to keep it in original scale
        # It will be scaled when needed in the evaluation process
        
        # Save scalers
        scaler_dir = self.args.result_dir
        os.makedirs(scaler_dir, exist_ok=True)
        
        feature_scaler_path = os.path.join(scaler_dir, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(scaler_dir, 'target_scaler.pkl')
        
        with open(feature_scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)
        
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(target_scaler, f)
            
        print(f"[{now()}] Scalers saved to: {scaler_dir}")
    
    def _create_model(self, trial=None):
        """Creates and returns a CNN model instance based on the configuration."""
        
        # Initialize configurations
        cnn_1d_config = json.loads(self.args.cnn_1d_config)
        cnn_2d_config = json.loads(self.args.cnn_2d_config)
        prediction_hidden_dims = self.args.prediction_hidden_dims
        dropout_rate = self.args.dropout_rate

        # Hyperparameter search space for Optuna - deprecated, now handled in _objective
        if trial is not None:
            print(f"[{now()}] Warning: trial parameter in _create_model is deprecated, parameters should already be in args")

        # Load other feature configs
        other_feature_configs = None
        if self.args.other_feature_configs and self.args.other_feature_configs.lower() != 'none':
            other_feature_configs = json.loads(self.args.other_feature_configs)

        # Debug output for model configuration
        print(f"[{now()}] Creating model with configuration:")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Prediction hidden dims: {prediction_hidden_dims}")
        print(f"  - Features for 2D CNN: {self.args.features_for_2d_cnn}")
        
        # Create the model
        model = AlloyCnnV2(
            column_names=self.feature_names,
            output_dim=len(self.args.target_columns),
            cnn_1d_config=cnn_1d_config,
            cnn_2d_config=cnn_2d_config,
            features_for_2d_cnn=self.args.features_for_2d_cnn,
            other_feature_configs=other_feature_configs,
            prediction_hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )

        # Save model structure
        model_structure_path = os.path.join(self.args.result_dir, 'model_structure.txt')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.print_structure(file_path=model_structure_path)
        with open(model_structure_path, 'a') as f:
            f.write(f"\n\n--- Full Model ---\n{model}")
        print(f"[{now()}] Model structure saved to {model_structure_path}")

        # Use DataParallel if requested and available
        if self.args.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"[{now()}] Using {torch.cuda.device_count()} GPUs for training.")
            model = torch.nn.DataParallel(model)
            
        return model

    def _get_training_params(self):
        """Constructs training parameters dictionary."""
        return {
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
            'device': self.args.device,
            'early_stopping_patience': self.args.patience,
            'use_lr_scheduler': self.args.use_lr_scheduler,
            'lr_scheduler_patience': self.args.lr_scheduler_patience,
            'lr_scheduler_factor': self.args.lr_scheduler_factor,
        }

    def _run_cross_validation(self):
        """Performs k-fold cross-validation."""
        print(f"[{now()}] Starting {self.args.num_folds}-fold cross-validation...")
        kf = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=self.args.random_state)
        
        best_fold_val_loss = float('inf')
        best_fold_num = -1
        best_fold_history = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_val_data['X'])):
            print(f"[{now()}] --- Starting Fold {fold+1}/{self.args.num_folds} ---")
            
            fold_train_data = {'X': self.train_val_data['X'][train_idx], 'y': self.train_val_data['y'][train_idx]}
            fold_val_data = {'X': self.train_val_data['X'][val_idx], 'y': self.train_val_data['y'][val_idx]}
            
            model_fold = self._create_model()

            training_params = self._get_training_params()
            
            trainer = TrainerFactory.create_trainer(
                model_type='cnn', model=model_fold, result_dir=self.args.result_dir,
                model_name=f"alloy_cnn_fold_{fold+1}", target_names=self.args.target_columns,
                training_params=training_params
            )
            
            history, val_loss, best_state_dict = trainer.train(
                num_epochs=self.args.epochs,
                train_data=fold_train_data,
                val_data=fold_val_data,
                save_checkpoint=False  # Do not save checkpoints for each fold
            )

            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_num = fold + 1
                self.best_model_state_dict = best_state_dict
                best_fold_history = history
    
        print(f"[{now()}] Best fold: {best_fold_num} with validation loss: {best_fold_val_loss:.4f}")
        
        if best_fold_history:
            self._save_best_fold_history(best_fold_history, best_fold_num)
            
        self._save_best_model("cv_best_model", best_fold_num)

    def _save_best_fold_history(self, history: Dict[str, List[float]], fold_num: int):
        """Saves the training history of the best fold."""
        if not history:
            print(f"[{now()}] No history for the best fold was provided. Skipping history saving.")
            return

        loss_dir = os.path.join(self.args.result_dir, "loss")
        os.makedirs(loss_dir, exist_ok=True)

        # Save training and validation history data to a JSON file
        history_to_save = {
            "train_loss_curve": history.get('train_loss', []),
            "val_loss_curve": history.get('val_loss', []),
            "best_train_loss": min(history['train_loss']) if 'train_loss' in history and history['train_loss'] else None,
            "best_val_loss": min(history['val_loss']) if 'val_loss' in history and history['val_loss'] else None,
            "n_iter": len(history.get('train_loss', []))
        }
        json_path = os.path.join(loss_dir, "best_fold_training_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
        print(f"[{now()}] Saved best fold ({fold_num}) training loss data to: {json_path}")

        # Save the plot of training curves
        plot_path = os.path.join(loss_dir, f"best_fold_{fold_num}_training_curves.png")
        BaseEvaluator.plot_training_curves(
            history=history, 
            save_path=plot_path, 
            title=f"Best Fold ({fold_num}) Training Curves"
        )
        print(f"[{now()}] Saved best fold ({fold_num}) training plot to: {plot_path}")

    def _train_final_model(self):
        """Trains a single model on a train/validation split."""
        print(f"[{now()}] Starting training on a single data split...")
        train_X, val_X, train_y, val_y = train_test_split(
            self.train_val_data['X'], self.train_val_data['y'], test_size=0.25, random_state=self.args.random_state
        )
        final_train_data = {'X': train_X, 'y': train_y}
        final_val_data = {'X': val_X, 'y': val_y}

        model = self._create_model()

        final_trainer = TrainerFactory.create_trainer(
            model_type='cnn', model=model, result_dir=self.args.result_dir,
            model_name="final_model", target_names=self.args.target_columns,
            training_params=self._get_training_params()
        )
        history, _, best_state_dict = final_trainer.train(
            num_epochs=self.args.epochs,
            train_data=final_train_data,
            val_data=final_val_data,
            save_checkpoint=False  # Only save the best model
        )
        
        if best_state_dict:
            self.best_model_state_dict = best_state_dict
            self._save_best_model("best_model")
            
        if history:
            self._save_best_fold_history(history, 0)  # 0 indicates it's not from a fold

    def _save_best_model(self, name, fold_num=None):
        """Save the best model to a file."""
        if self.best_model_state_dict is None:
            print(f"[{now()}] No best model state dict found. Skipping saving.")
            return
            
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        if fold_num is not None:
            save_path = os.path.join(self.checkpoints_dir, f"{name}_{fold_num}.pt")
        else:
            save_path = os.path.join(self.checkpoints_dir, f"{name}.pt")
        
        # Create a complete save object that includes state_dict and model configuration
        save_object = {
            'model_state_dict': self.best_model_state_dict,
            'cnn_1d_config': self.args.cnn_1d_config,
            'cnn_2d_config': self.args.cnn_2d_config,
            'features_for_2d_cnn': self.args.features_for_2d_cnn,
            'other_feature_configs': self.args.other_feature_configs,
            'prediction_hidden_dims': self.args.prediction_hidden_dims,
            'dropout_rate': self.args.dropout_rate,
            'output_dim': len(self.args.target_columns)
        }
        
        torch.save(save_object, save_path)
        print(f"[{now()}] Saved best model to {save_path}")
        self.best_model_path = save_path

        config_data = { 'args': vars(self.args) }
        if fold_num is not None:
            config_data['best_fold'] = fold_num
        with open(os.path.join(self.args.result_dir, f"{name}_config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, default=str)

    def _evaluate_best_model(self):
        """Evaluates the best model on the test set."""
        print(f"[{now()}] Evaluating the best model...")
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            print(f"[{now()}] ERROR: Best model not found at {self.best_model_path}. Skipping evaluation.")
            return

        # Load the best model directly from disk
        loaded_object = torch.load(self.best_model_path, map_location=self.args.device or 'cpu')
        
        # Extract configuration from the saved model
        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            # Extract all configuration parameters from the saved model
            cnn_1d_config = loaded_object.get('cnn_1d_config')
            cnn_2d_config = loaded_object.get('cnn_2d_config')
            features_for_2d_cnn = loaded_object.get('features_for_2d_cnn')
            other_feature_configs = loaded_object.get('other_feature_configs')
            prediction_hidden_dims = loaded_object.get('prediction_hidden_dims')
            dropout_rate = loaded_object.get('dropout_rate')
            
            # Update args with the saved configuration
            if cnn_1d_config:
                self.args.cnn_1d_config = cnn_1d_config
            if cnn_2d_config:
                self.args.cnn_2d_config = cnn_2d_config
            if features_for_2d_cnn:
                self.args.features_for_2d_cnn = features_for_2d_cnn
            if other_feature_configs:
                self.args.other_feature_configs = other_feature_configs
            if prediction_hidden_dims:
                self.args.prediction_hidden_dims = prediction_hidden_dims
            if dropout_rate:
                self.args.dropout_rate = dropout_rate
                
            print(f"[{now()}] Updated args with configuration from saved model")
        
        # Try to load from best_config_path as well
        best_config_path = os.path.join(self.args.result_dir, "optuna_best_model_config.json")
        if os.path.exists(best_config_path):
            print(f"[{now()}] Loading best model configuration from: {best_config_path}")
            try:
                with open(best_config_path, 'r', encoding='utf-8') as f:
                    best_config = json.load(f)
                    
                # Update args with the best configuration
                for key, value in best_config.items():
                    if hasattr(self.args, key):
                        # Handle special case for prediction_hidden_dims which might be a list
                        if key == 'prediction_hidden_dims' and isinstance(value, list):
                            setattr(self.args, key, value)
                        else:
                            try:
                                # Try to convert string values to their original type
                                setattr(self.args, key, type(getattr(self.args, key))(value))
                            except (ValueError, TypeError):
                                # If conversion fails, use the value as is
                                setattr(self.args, key, value)
                print(f"[{now()}] Successfully loaded best model configuration.")
            except Exception as e:
                print(f"[{now()}] Warning: Failed to load best model configuration: {e}")

        # Create model with the exact same configuration as during training
        eval_model = self._create_model()
        
        # Set model to the appropriate device
        device = self.args.device or 'cpu'
        eval_model.to(device)
        
        # Load the best model weights
        if not self._load_best_model_weights(eval_model):
            print(f"[{now()}] Failed to load best model weights. Skipping evaluation.")
            return
            
        # Set model to evaluation mode
        eval_model.eval()

        # Create evaluator
        evaluator = EvaluatorFactory.create_evaluator(
            evaluator_type='alloys',
            result_dir=self.args.result_dir,
            model_name='best_model',
            target_names=self.args.target_columns,
            target_scaler=self.target_scaler
        )
        
        # Ensure the model is on the same device as the evaluator
        eval_model.to(evaluator.device)
        print(f"[{now()}] Model moved to {evaluator.device} for evaluation")
        
        # For evaluation after CV, the "train_data" for the evaluator should be the full train_val set
        if self.args.cross_validate:
            eval_train_data = {'X': self.train_val_data['X'], 'y': self.train_val_data['y']}
            eval_val_data = None # No separate validation set when using the full training data
        else:
            # For standard training, we split train_val into a smaller train and val set for evaluation
            train_X, val_X, train_y, val_y = train_test_split(
                self.train_val_data['X'], self.train_val_data['y'], 
                test_size=0.25, random_state=self.args.random_state
            )
            eval_train_data = {'X': train_X, 'y': train_y}
            eval_val_data = {'X': val_X, 'y': val_y}
        
        # Test set data - note that X is already scaled, and y is unscaled 
        # (since the pipeline preserves original targets)
        eval_test_data = self.test_data
        
        # Evaluate the model
        metrics = evaluator.evaluate_model(
            model=eval_model,
            train_data=eval_train_data,
            test_data=eval_test_data, 
            val_data=eval_val_data,
            save_prefix='final_', 
            feature_names=self.feature_names
        )
        
        print(f"[{now()}] Test set evaluation complete. Metrics: {metrics}")

    def _load_best_model_weights(self, model):
        """Load the weights of the best model into the given model."""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            print(f"[{now()}] ERROR: Best model not found at {self.best_model_path}")
            return False
            
        device = model.device if hasattr(model, 'device') else 'cpu'
        print(f"[{now()}] Loading best model weights from {self.best_model_path}")
        
        loaded_object = torch.load(self.best_model_path, map_location=device)
        
        # Check if we saved a complete configuration or just the state_dict
        if isinstance(loaded_object, dict) and 'model_state_dict' in loaded_object:
            model_state_dict = loaded_object['model_state_dict']
            
            # If we have a new model format with configuration
            if 'prediction_hidden_dims' in loaded_object:
                print(f"[{now()}] Found saved model configuration, recreating model with matching structure")
                
                # Create a new model with the saved configuration
                try:
                    new_model = AlloyCnnV2(
                        column_names=self.feature_names,
                        output_dim=loaded_object.get('output_dim', len(self.args.target_columns)),
                        cnn_1d_config=json.loads(loaded_object.get('cnn_1d_config', self.args.cnn_1d_config)),
                        cnn_2d_config=json.loads(loaded_object.get('cnn_2d_config', self.args.cnn_2d_config)),
                        features_for_2d_cnn=loaded_object.get('features_for_2d_cnn', self.args.features_for_2d_cnn),
                        other_feature_configs=self._parse_other_feature_configs(loaded_object.get('other_feature_configs')),
                        prediction_hidden_dims=loaded_object.get('prediction_hidden_dims', self.args.prediction_hidden_dims),
                        dropout_rate=loaded_object.get('dropout_rate', self.args.dropout_rate)
                    )
                    
                    # Transfer weights directly to the new model
                    new_model.load_state_dict(model_state_dict)
                    
                    # Now transfer the new model to the original model
                    # This should work since they should be the same structure
                    try:
                        model.load_state_dict(new_model.state_dict())
                        print(f"[{now()}] Successfully loaded best model weights into the model with the saved configuration")
                        return True
                    except Exception as e:
                        print(f"[{now()}] Error transferring model weights: {e}")
                        print(f"[{now()}] Will try to directly load weights into original model")
                except Exception as e:
                    print(f"[{now()}] Error recreating model with saved configuration: {e}")
                    print(f"[{now()}] Will try to directly load weights into original model")
        else:
            model_state_dict = loaded_object
        
        # Handle DataParallel state dict (has 'module.' prefix)
        if list(model_state_dict.keys())[0].startswith('module.') and not list(model.state_dict().keys())[0].startswith('module.'):
            # Remove 'module.' prefix from state_dict keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        elif not list(model_state_dict.keys())[0].startswith('module.') and list(model.state_dict().keys())[0].startswith('module.'):
            # Add 'module.' prefix to state_dict keys
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        
        # Try direct loading, which might fail if structures don't match
        try:
            model.load_state_dict(model_state_dict)
            print(f"[{now()}] Successfully loaded best model weights into the model.")
            return True
        except Exception as e:
            print(f"[{now()}] Error loading weights: {e}")
            print(f"[{now()}] Will use the original model without loading weights.")
            return False

    def _parse_other_feature_configs(self, config_str):
        """Safely parse other_feature_configs which could be None, 'None', or a JSON string."""
        if config_str is None or config_str == 'None':
            return None
        try:
            return json.loads(config_str)
        except (json.JSONDecodeError, TypeError):
            print(f"[{now()}] Warning: Failed to parse other_feature_configs: {config_str}")
            return None

    def _split_data_for_trial(self, trial: optuna.Trial):
        """Helper to split data for a single Optuna trial."""
        train_X, val_X, train_y, val_y = train_test_split(
            self.train_val_data['X'], self.train_val_data['y'],
            test_size=0.25, random_state=self.args.random_state
        )
        return {'X': train_X, 'y': train_y}, {'X': val_X, 'y': val_y}

    def _run_optuna(self):
        """Runs hyperparameter optimization using Optuna."""
        print(f"[{now()}] Starting Optuna hyperparameter optimization...")
        
        # Setup Optuna study
        load_if_exists = self.args.load_study_if_exists if hasattr(self.args, 'load_study_if_exists') else False
        
        if hasattr(self.args, 'study_storage') and self.args.study_storage:
            print(f"[{now()}] Using Optuna storage: {self.args.study_storage}")
            study = optuna.create_study(
                direction='minimize', 
                study_name=self.args.study_name, 
                storage=self.args.study_storage,
                load_if_exists=load_if_exists
            )
        else:
            print(f"[{now()}] Using in-memory Optuna storage")
            study = optuna.create_study(
                direction='minimize', 
                study_name=self.args.study_name
            )
        
        # Run optimization
        study.optimize(self._objective, n_trials=self.args.n_trials)

        print(f"[{now()}] Best trial finished with value: {study.best_value} and params: {study.best_params}")
        
        # Save best parameters to a JSON file
        best_params_path = os.path.join(self.args.result_dir, "optuna_best_params.json")
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(study.best_trial.params, f, indent=4)
        print(f"[{now()}] Saved best Optuna parameters to: {best_params_path}")

        # Update args with the best parameters from Optuna
        self._update_args_with_optuna_params(study.best_trial.params)

        # Persist the best model from the best trial
        best_trial = study.best_trial
        
        # Get the best model from the file saved during trial
        if best_trial.user_attrs.get("has_best_model", False):
            best_model_path = best_trial.user_attrs.get("best_model_path")
            if os.path.exists(best_model_path):
                print(f"[{now()}] Loading best model from: {best_model_path}")
                save_object = torch.load(best_model_path)
                self.best_model_state_dict = save_object['model_state_dict']
                
                # Save as the final best model
                self._save_best_model("optuna_best_model", best_trial.number)
                
                # Clean up trial model files, keeping only the best one
                self._cleanup_trial_models(best_trial.number)
                
                # Save the best trial's training history
                if "training_history" in best_trial.user_attrs:
                    history_data = best_trial.user_attrs.get("training_history")
                    
                    # Create loss directory
                    loss_dir = os.path.join(self.args.result_dir, "loss")
                    os.makedirs(loss_dir, exist_ok=True)
                    
                    # Save as the best model's training history
                    best_history_path = os.path.join(loss_dir, "best_model_training_history.json")
                    with open(best_history_path, 'w', encoding='utf-8') as f:
                        json.dump(history_data, f, indent=4)
                    
                    # Generate plot for the best model's training curves
                    history = {
                        'train_loss': history_data.get('train_loss_curve', []),
                        'val_loss': history_data.get('val_loss_curve', [])
                    }
                    plot_path = os.path.join(loss_dir, "best_model_training_curves.png")
                    BaseEvaluator.plot_training_curves(
                        history=history, 
                        save_path=plot_path, 
                        title=f"Best Model (Trial {best_trial.number}) Training Curves"
                    )
                    print(f"[{now()}] Saved best model training plot to: {plot_path}")
                
                if self.args.evaluate_after_train:
                    self._evaluate_best_model()
            else:
                print(f"[{now()}] Warning: Could not find best model at {best_model_path}")
        else:
            print(f"[{now()}] No best model found for the best trial. Skipping final evaluation.")
            
    def _update_args_with_optuna_params(self, best_params):
        """Update args with the best parameters found by Optuna."""
        print(f"[{now()}] Updating args with best Optuna parameters...")
        
        # Only continue if best_params is not empty
        if not best_params:
            print(f"[{now()}] No parameters to update.")
            return
            
        # Map Optuna parameter names to args attributes
        param_mapping = {
            'dropout_rate': 'dropout_rate',
            'learning_rate': 'learning_rate',
            'batch_size': 'batch_size',
            'weight_decay': 'weight_decay',
            'lr_scheduler_factor': 'lr_scheduler_factor',
        }
        
        # Update args with best parameters
        for optuna_param, args_attr in param_mapping.items():
            if optuna_param in best_params:
                setattr(self.args, args_attr, best_params[optuna_param])
                print(f"  - Set {args_attr} = {best_params[optuna_param]}")
        
        # Handle special cases for prediction_hidden_dims
        pred_hidden_dims = []
        pred_n_layers = best_params.get('pred_n_layers')
        if pred_n_layers:
            for i in range(pred_n_layers):
                dim_param = f'pred_hidden_dim_{i}'
                if dim_param in best_params:
                    pred_hidden_dims.append(best_params[dim_param])
            if pred_hidden_dims:
                self.args.prediction_hidden_dims = pred_hidden_dims
                print(f"  - Set prediction_hidden_dims = {pred_hidden_dims}")
        
        # Handle CNN configurations
        try:
            # Load current configs
            cnn_1d_config = json.loads(self.args.cnn_1d_config)
            cnn_2d_config = json.loads(self.args.cnn_2d_config)
            
            # Update 1D CNN configs
            for name, config in cnn_1d_config.items():
                n_cnn_layers_param = f'cnn_1d_{name}_layers'
                if n_cnn_layers_param in best_params:
                    n_cnn_layers = best_params[n_cnn_layers_param]
                    out_channels = []
                    for i in range(n_cnn_layers):
                        channel_param = f'cnn_1d_{name}_channels_{i}'
                        if channel_param in best_params:
                            out_channels.append(best_params[channel_param])
                    if out_channels:
                        config['out_channels'] = out_channels
                        print(f"  - Set {name} CNN out_channels = {out_channels}")
                
                # Update FC layers if present
                n_fc_layers_param = f'cnn_1d_{name}_fc_layers'
                if n_fc_layers_param in best_params and best_params[n_fc_layers_param] > 0:
                    n_fc_layers = best_params[n_fc_layers_param]
                    fc_layers = []
                    for i in range(n_fc_layers):
                        fc_dim_param = f'cnn_1d_{name}_fc_dim_{i}'
                        if fc_dim_param in best_params:
                            fc_layers.append(best_params[fc_dim_param])
                    if fc_layers:
                        config['fc_layers'] = fc_layers
                        print(f"  - Set {name} CNN fc_layers = {fc_layers}")
            
            # Update 2D CNN configs
            n_cnn_2d_layers_param = 'cnn_2d_layers'
            if n_cnn_2d_layers_param in best_params:
                n_cnn_2d_layers = best_params[n_cnn_2d_layers_param]
                out_channels = []
                for i in range(n_cnn_2d_layers):
                    channel_param = f'cnn_2d_channels_{i}'
                    if channel_param in best_params:
                        out_channels.append(best_params[channel_param])
                if out_channels:
                    cnn_2d_config['out_channels'] = out_channels
                    print(f"  - Set 2D CNN out_channels = {out_channels}")
            
            # Update 2D CNN FC layers if present
            n_fc_2d_layers_param = 'cnn_2d_fc_layers'
            if n_fc_2d_layers_param in best_params and best_params[n_fc_2d_layers_param] > 0:
                n_fc_2d_layers = best_params[n_fc_2d_layers_param]
                fc_layers = []
                for i in range(n_fc_2d_layers):
                    fc_dim_param = f'cnn_2d_fc_dim_{i}'
                    if fc_dim_param in best_params:
                        fc_layers.append(best_params[fc_dim_param])
                if fc_layers:
                    cnn_2d_config['fc_layers'] = fc_layers
                    print(f"  - Set 2D CNN fc_layers = {fc_layers}")
            
            # Save updated configs back to args
            self.args.cnn_1d_config = json.dumps(cnn_1d_config)
            self.args.cnn_2d_config = json.dumps(cnn_2d_config)
            print(f"  - Updated CNN configurations")
        except Exception as e:
            print(f"  - Warning: Failed to update CNN configurations: {e}")
        
        # If this is a trial's params (not best params), don't save to file
        if 'learning_rate' in best_params and 'batch_size' in best_params and 'weight_decay' in best_params:
            # Save the updated args to a config file
            updated_config_path = os.path.join(self.args.result_dir, "optuna_best_model_config.json")
            with open(updated_config_path, 'w', encoding='utf-8') as f:
                # Convert args to dict, handling non-serializable types
                args_dict = {k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)) 
                            for k, v in vars(self.args).items()}
                json.dump(args_dict, f, indent=4)
            print(f"[{now()}] Saved updated configuration to: {updated_config_path}")

    def _objective(self, trial: optuna.Trial):
        """Objective function for Optuna optimization."""
        print(f"[{now()}] Evaluating trial {trial.number}...")
        
        # Get hyperparameters for this trial
        params = {}
        
        # Add dropout rate
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)
        
        # Add prediction network parameters
        pred_n_layers = trial.suggest_int('pred_n_layers', 1, 3)
        params['pred_n_layers'] = pred_n_layers
        for i in range(pred_n_layers):
            params[f'pred_hidden_dim_{i}'] = trial.suggest_int(f'pred_hidden_dim_{i}', 32, 512, log=True)

        # Add 1D CNN parameters for each feature type
        cnn_1d_config = json.loads(self.args.cnn_1d_config)
        for name, config in cnn_1d_config.items():
            n_cnn_layers = trial.suggest_int(f'cnn_1d_{name}_layers', 1, 3)
            params[f'cnn_1d_{name}_layers'] = n_cnn_layers
            for i in range(n_cnn_layers):
                params[f'cnn_1d_{name}_channels_{i}'] = trial.suggest_int(f'cnn_1d_{name}_channels_{i}', 16, 64, log=True)
            
            n_fc_layers = trial.suggest_int(f'cnn_1d_{name}_fc_layers', 1, 2)
            params[f'cnn_1d_{name}_fc_layers'] = n_fc_layers
            for i in range(n_fc_layers):
                params[f'cnn_1d_{name}_fc_dim_{i}'] = trial.suggest_int(f'cnn_1d_{name}_fc_dim_{i}', 32, 256, log=True)

        # Add 2D CNN parameters
        n_cnn_2d_layers = trial.suggest_int('cnn_2d_layers', 1, 3)
        params['cnn_2d_layers'] = n_cnn_2d_layers
        for i in range(n_cnn_2d_layers):
            params[f'cnn_2d_channels_{i}'] = trial.suggest_int(f'cnn_2d_channels_{i}', 16, 64, log=True)
        
        n_fc_2d_layers = trial.suggest_int('cnn_2d_fc_layers', 1, 2)
        params['cnn_2d_fc_layers'] = n_fc_2d_layers
        for i in range(n_fc_2d_layers):
            params[f'cnn_2d_fc_dim_{i}'] = trial.suggest_int(f'cnn_2d_fc_dim_{i}', 32, 256, log=True)
            
        # Add training parameters
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        if self.args.use_lr_scheduler:
            params['lr_scheduler_factor'] = trial.suggest_float('lr_scheduler_factor', 0.1, 0.9)
        
        # Update args with the trial's parameters
        self._update_args_with_optuna_params(params)
        
        # Split data for this trial
        train_data, val_data = self._split_data_for_trial(trial)
        
        # Create model with trial-suggested hyperparameters - no need to pass trial here
        # since we've already updated args with the trial parameters
        model = self._create_model()
        
        # Get training parameters
        training_params = self._get_training_params()
        
        # Create trainer
        trainer = TrainerFactory.create_trainer(
            model_type='cnn', 
            model=model, 
            training_params=training_params,
            result_dir=self.args.result_dir,
            model_name=f"optuna_trial_{trial.number}",
            target_names=self.args.target_columns
        )
        
        # Train model
        num_epochs = self.args.epochs
        history, best_val_loss, best_model_state_dict = trainer.train(
            num_epochs=num_epochs,
            train_data=train_data,
            val_data=val_data,
            model_save_name=None,  # Don't save individual trial models
            save_checkpoint=False
        )
        
        # Store the training history in the trial's user attributes
        if history:
            history_to_save = {
                "train_loss_curve": history.get('train_loss', []),
                "val_loss_curve": history.get('val_loss', []),
                "best_train_loss": min(history['train_loss']) if 'train_loss' in history and history['train_loss'] else None,
                "best_val_loss": min(history['val_loss']) if 'val_loss' in history and history['val_loss'] else None,
                "n_iter": len(history.get('train_loss', []))
            }
            # Store the history in the trial's user attributes
            trial.set_user_attr("training_history", history_to_save)
        
        # Instead of storing the model state dict in the trial's user attributes,
        # save it to disk and store the path
        if best_model_state_dict is not None:
            # Create the checkpoints directory if it doesn't exist
            os.makedirs(os.path.join(self.args.result_dir, "checkpoints"), exist_ok=True)
            
            # Save the model to disk
            model_path = os.path.join(self.args.result_dir, "checkpoints", f"trial_{trial.number}_best_model.pt")
            
            # Create a complete save object that includes state_dict and model configuration
            save_object = {
                'model_state_dict': best_model_state_dict,
                'cnn_1d_config': self.args.cnn_1d_config,
                'cnn_2d_config': self.args.cnn_2d_config,
                'features_for_2d_cnn': self.args.features_for_2d_cnn,
                'other_feature_configs': self.args.other_feature_configs,
                'prediction_hidden_dims': self.args.prediction_hidden_dims,
                'dropout_rate': self.args.dropout_rate,
                'output_dim': len(self.args.target_columns)
            }
            
            # Save the model to disk
            torch.save(save_object, model_path)
            
            # Store only the path in the trial's user attributes
            trial.set_user_attr("best_model_path", model_path)
            trial.set_user_attr("has_best_model", True)
        
        return best_val_loss

    def _cleanup_trial_models(self, best_trial_number):
        """Clean up trial model files, keeping only the best one."""
        print(f"[{now()}] Cleaning up trial model files, keeping only the best model...")
        
        # Find all trial model files in the checkpoints directory
        checkpoints_dir = os.path.join(self.args.result_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return
            
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("trial_") and filename.endswith("_best_model.pt"):
                trial_number = int(filename.split("_")[1])
                if trial_number != best_trial_number:
                    file_path = os.path.join(checkpoints_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"[{now()}] Removed trial model file: {filename}")
                    except Exception as e:
                        print(f"[{now()}] Warning: Failed to remove trial model file {filename}: {e}") 