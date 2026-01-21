"""
Main training and evaluation pipeline.
"""
import os
import shutil
import json
import time
import torch
import joblib
import optuna
import numpy as np
import pandas as pd
import tempfile
from typing import Dict, List, Any, cast
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .utils import now
from .data_loader import load_data
from ..base.alloys_nn import AlloyNN
from ..trainers import TrainerFactory
from ..evaluators import EvaluatorFactory, AlloysEvaluator
from ..evaluators.base_evaluator import BaseEvaluator
from src.feature_engineering.utils import set_seed


class TrainingPipeline:
    def __init__(self, args):
        self.args = args
        self.train_val_data = None
        self.test_data = None
        self.feature_names = None
        self.scaler_X = None
        self.scaler_y = None
        self.best_model_path = None

    def _create_model(self, trial=None, config=None):
        """Creates and returns a model instance and its configuration."""
        if config:
            feature_type_config = config['feature_type_config']
            feature_hidden_dims = config['feature_hidden_dims']
            hidden_dims = config['hidden_dims']
            dropout_rate = config['dropout_rate']
        else:
            feature_type_config, feature_hidden_dims, hidden_dims, dropout_rate = self._get_model_config(trial)

        model_config = {
            'feature_type_config': feature_type_config,
            'feature_hidden_dims': feature_hidden_dims,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate
        }
        
        # This is where you can select the model based on self.args.model_type
        if self.args.model_type == 'nn':
            model = AlloyNN(
                column_names=self.feature_names,
                output_dim=len(self.args.target_columns),
                feature_type_config=feature_type_config,
                feature_hidden_dims=feature_hidden_dims,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
        # elif self.args.model_type == 'bert':
        #     # Placeholder for BERT model instantiation
        #     pass
        # elif self.args.model_type == 'cnn':
        #     # Placeholder for CNN model instantiation
        #     pass
        else:
            raise ValueError(f"Unknown model_type: {self.args.model_type}")

        # Centralize DataParallel wrapping
        if self.args.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"[{now()}] Using {torch.cuda.device_count()} GPUs for training.")
            model = torch.nn.DataParallel(model)
            
        return model, model_config

    def run(self):
        """Main entry point to run the pipeline."""
        # 设置随机种子以保证结果可复现
        set_seed(self.args.random_state)
        
        self._prepare_data()
        
        if self.args.use_optuna:
            self._run_optuna()
        else:
            self._run_standard_training()

        # Always evaluate after standard training if the flag is set
        if self.args.evaluate_after_train:
            self._evaluate_best_model()

    def _prepare_data(self):
        """Load, preprocess, and save scalers."""
        print(f"[{now()}] Preparing data...")
        self.train_val_data, self.test_data, self.feature_names = load_data(
            data_file=self.args.data_file,
            target_columns=self.args.target_columns,
            test_size=self.args.test_size,
            random_state=self.args.random_state,
            use_process_embedding=self.args.use_process_embedding,
            use_element_embedding=self.args.use_element_embedding,
            use_joint_composition_process_embedding=self.args.use_joint_composition_process_embedding,
            use_composition_feature=self.args.use_composition_feature,
            use_feature1=self.args.use_feature1,
            use_feature2=self.args.use_feature2,
            other_features_name=self.args.other_features_name,
            use_temperature=self.args.use_temperature
        )
        
        # 为了保证一致性，预先分割训练/验证数据
        # 这样在Optuna和最终评估时都使用相同的分割
        self.eval_train_X, self.eval_val_X, self.eval_train_y, self.eval_val_y = train_test_split(
            self.train_val_data['X'], self.train_val_data['y'], 
            test_size=0.25, random_state=self.args.random_state
        )
        
        # 如果有ID数据，也要分割
        if 'ids' in self.train_val_data:
            self.eval_train_ids, self.eval_val_ids = train_test_split(
                self.train_val_data['ids'], test_size=0.25, random_state=self.args.random_state
            )
        
        self.scaler_X = StandardScaler().fit(self.train_val_data['X'])
        self.train_val_data['X'] = self.scaler_X.transform(self.train_val_data['X']).astype(np.float32)
        self.test_data['X'] = self.scaler_X.transform(self.test_data['X']).astype(np.float32)
        
        # 对预分割的数据也进行标准化
        self.eval_train_X = self.scaler_X.transform(self.eval_train_X).astype(np.float32)
        self.eval_val_X = self.scaler_X.transform(self.eval_val_X).astype(np.float32)

        self.scaler_y = StandardScaler().fit(self.train_val_data['y'])
        self.train_val_data['y'] = self.scaler_y.transform(self.train_val_data['y']).astype(np.float32)
        self.test_data['y'] = self.scaler_y.transform(self.test_data['y']).astype(np.float32)
        
        # 对预分割的标签也进行标准化
        self.eval_train_y = self.scaler_y.transform(self.eval_train_y).astype(np.float32)
        self.eval_val_y = self.scaler_y.transform(self.eval_val_y).astype(np.float32)

        os.makedirs(self.args.result_dir, exist_ok=True)
        joblib.dump(self.scaler_X, os.path.join(self.args.result_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.args.result_dir, 'scaler_y.pkl'))
        print(f"[{now()}] Scalers saved to: {self.args.result_dir}")
        print(f"[{now()}] Data split: Train={len(self.eval_train_X)}, Val={len(self.eval_val_X)}, Test={len(self.test_data['X'])}")

    def _get_model_config(self, trial=None):
        """Constructs model configuration from args or Optuna trial."""
        # Feature configuration
        feature_type_config = {}
        if self.args.use_element_embedding or self.args.use_joint_composition_process_embedding:
            feature_type_config['emb'] = ['emb']
        if self.args.use_feature1:
            feature_type_config['feature1'] = ['feature1']
        if self.args.use_feature2:
            feature_type_config['feature2'] = ['feature2']
        if self.args.other_features_name and self.args.other_features_name != ['None']:
            feature_type_config['other_features'] = self.args.other_features_name
        
        # Hidden dimensions for feature processors
        feature_hidden_dims = {
            'emb': trial.suggest_categorical('emb_hidden_dim', [64, 128, 256, 512]) if trial and 'emb' in feature_type_config else self.args.emb_hidden_dim,
            'feature1': trial.suggest_categorical('feature1_hidden_dim', [64, 128, 256]) if trial and 'feature1' in feature_type_config else self.args.feature1_hidden_dim,
            'feature2': trial.suggest_categorical('feature2_hidden_dim', [64, 128, 256]) if trial and 'feature2' in feature_type_config else self.args.feature2_hidden_dim,
            'other_features': trial.suggest_categorical('other_features_hidden_dim', [64, 128, 256]) if trial and 'other_features' in feature_type_config else self.args.other_features_hidden_dim,
        }

        # Hidden dimensions for the main prediction network
        if trial:
            feature_raw_dims = {ftype: len([col for col in self.feature_names if any(k.lower() in col.lower() for k in keywords)]) for ftype, keywords in feature_type_config.items()}
            input_dim_for_pred = sum(feature_hidden_dims.get(ftype, 0) or raw_dim for ftype, raw_dim in feature_raw_dims.items())
            
            # 改进超参数搜索范围
            n_layers = trial.suggest_int('n_layers', 2, 6)  # 增加最小层数
            
            # 使用更保守的隐藏维度策略
            first_layer_factor = trial.suggest_float('first_layer_factor', 0.5, 2.0)
            reduction_factor = trial.suggest_float('reduction_factor', 0.4, 0.8)
            
            first_hidden_dim = max(64, int(input_dim_for_pred * first_layer_factor))
            hidden_dims = [first_hidden_dim]
            
            for i in range(1, n_layers):
                next_dim = max(32, int(hidden_dims[-1] * reduction_factor))
                if next_dim < 32:
                    break
                hidden_dims.append(next_dim)
        else:
            hidden_dims = self.args.hidden_dims

        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1) if trial else self.args.dropout_rate

        return feature_type_config, feature_hidden_dims, hidden_dims, dropout_rate

    def _run_standard_training(self):
        """Run training with or without cross-validation."""
        if self.args.cross_validate:
            self._run_cross_validation()
        else:
            self._train_final_model()

        # Always evaluate after standard training if the flag is set
        if self.args.evaluate_after_train:
            self._evaluate_best_model()

    def _run_cross_validation(self, model_config=None, trial_number=None, closest_dir_name="closest_to_mean_predictions"):
        """Performs k-fold cross-validation.
        
        Args:
            model_config: Optional dictionary containing model configuration (e.g., from Optuna).
                          If provided, this config is used for all folds.
            trial_number: Optional trial number if running as part of an Optuna study.
        """
        print(f"[{now()}] Starting cross-validation...")
        kf = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=self.args.random_state)
        
        best_fold_val_loss = float('inf')
        best_fold_num = -1
        best_fold_history = None
        best_model_state_dict = None
        best_model_config = None

        best_model_config = None
        
        all_fold_metrics = []
        all_fold_state_dicts = []

        # Determine output directories
        if trial_number is not None:
            # Structure: predictions/optuna_trials/trial_X/
            cv_output_dir = os.path.join(self.args.result_dir, "predictions", "optuna_trials", f"trial_{trial_number}")
            fold_pred_base_dir = cv_output_dir
        else:
            # Structure: result_dir/ (for csvs) and predictions/fold_X (for preds)
            cv_output_dir = self.args.result_dir
            fold_pred_base_dir = os.path.join(self.args.result_dir, "predictions")

        os.makedirs(cv_output_dir, exist_ok=True)


        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_val_data['X'])):
            print(f"[{now()}] --- Starting Fold {fold+1}/{self.args.num_folds} ---")
            
            fold_train_data = {'X': self.train_val_data['X'][train_idx], 'y': self.train_val_data['y'][train_idx]}
            fold_val_data = {'X': self.train_val_data['X'][val_idx], 'y': self.train_val_data['y'][val_idx]}
            
            # Handle IDs if present
            fold_train_ids, fold_val_ids = None, None
            if 'ids' in self.train_val_data:
                # Assuming ids is a numpy array or pandas Series/DataFrame that supports indexing
                ids_data = self.train_val_data['ids']
                if hasattr(ids_data, 'iloc'):
                    fold_train_ids = ids_data.iloc[train_idx]
                    fold_val_ids = ids_data.iloc[val_idx]
                else:
                    # Numpy array or list
                    fold_train_ids = np.array(ids_data)[train_idx]
                    fold_val_ids = np.array(ids_data)[val_idx]
                
                fold_train_data['ids'] = fold_train_ids
                fold_val_data['ids'] = fold_val_ids

            # Use provided config or create new one (which pulls from args)
            model_fold, model_config_fold = self._create_model(config=model_config)

            training_params = self._get_training_params()
            training_params['save_config_on_init'] = False
            training_params['save_checkpoints'] = False # Disable checkpoint saving for folds
            
            trainer = TrainerFactory.create_trainer(
                model_type=self.args.model_type, model=model_fold, result_dir=self.args.result_dir,
                model_name=f"alloy_nn_fold_{fold+1}", target_names=self.args.target_columns,
                train_data=fold_train_data, val_data=fold_val_data, training_params=training_params
            )
            
            history, val_loss, state_dict = trainer.train(num_epochs=self.args.epochs)
            
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_num = fold + 1
                best_model_state_dict = state_dict
                best_model_config = model_config_fold
                best_fold_history = history
            
            all_fold_state_dicts.append(state_dict)
            
            # Load best weights for this fold to perform detailed evaluation
            if best_model_state_dict:
                 # Helper to load state dict handling DataParallel
                 if isinstance(model_fold, torch.nn.DataParallel):
                      # If model is parallel but state_dict isn't (or vice versa), handle it. 
                      # But here we just trained it, so it should match. 
                      # Actually, standard practice: unwrap if needed or just load.
                      # Let's simple load.
                      model_fold.load_state_dict(best_model_state_dict)
                 else:
                      model_fold.load_state_dict(best_model_state_dict)
            
            model_fold.eval()
            
            # Predict on validation set
            X_val_tensor = torch.tensor(fold_val_data['X'], dtype=torch.float32).to(self.args.device)
            with torch.no_grad():
                y_val_pred_tensor = model_fold(X_val_tensor)
                y_val_pred = y_val_pred_tensor.cpu().numpy()
            
            # Inverse transform
            # Ensure 2D
            if y_val_pred.ndim == 1: y_val_pred = y_val_pred.reshape(-1, 1)
            y_val_true_scaled = fold_val_data['y']
            if y_val_true_scaled.ndim == 1: y_val_true_scaled = y_val_true_scaled.reshape(-1, 1)
            
            y_val_pred_orig = self.scaler_y.inverse_transform(y_val_pred)
            y_val_true_orig = self.scaler_y.inverse_transform(y_val_true_scaled)
            
            fold_res = {'fold': fold + 1}
            
            # Calculate metrics per target
            for i, target_name in enumerate(self.args.target_columns):
                y_true = y_val_true_orig[:, i]
                y_pred = y_val_pred_orig[:, i]
                
                fold_res[f"{target_name}_r2"] = r2_score(y_true, y_pred)
                fold_res[f"{target_name}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                fold_res[f"{target_name}_mae"] = mean_absolute_error(y_true, y_pred)
            
            all_fold_metrics.append(fold_res)

            # --- Save Predictions for this Fold (Train/Val/Test) ---
            # Create directory for this fold's predictions
            fold_pred_dir = os.path.join(fold_pred_base_dir, f"fold_{fold+1}")
            os.makedirs(fold_pred_dir, exist_ok=True)

            # 1. Train predictions (for this fold)
            model_fold.eval()
            with torch.no_grad():
                # Prepare train data tensor
                X_train_tensor = torch.tensor(fold_train_data['X'], dtype=torch.float32).to(self.args.device)
                y_train_pred_tensor = model_fold(X_train_tensor)
                y_train_pred = y_train_pred_tensor.cpu().numpy()
            
            if y_train_pred.ndim == 1: y_train_pred = y_train_pred.reshape(-1, 1)
            y_train_pred_orig = self.scaler_y.inverse_transform(y_train_pred)
            y_train_true_orig = self.scaler_y.inverse_transform(fold_train_data['y'].reshape(-1, 1) if fold_train_data['y'].ndim==1 else fold_train_data['y'])

            # 2. Test predictions (global)
            with torch.no_grad():
                X_test_tensor = torch.tensor(self.test_data['X'], dtype=torch.float32).to(self.args.device)
                y_test_pred_tensor = model_fold(X_test_tensor)
                y_test_pred = y_test_pred_tensor.cpu().numpy()
            
            if y_test_pred.ndim == 1: y_test_pred = y_test_pred.reshape(-1, 1)
            y_test_pred_orig = self.scaler_y.inverse_transform(y_test_pred)
            y_test_true_orig = self.scaler_y.inverse_transform(self.test_data['y'].reshape(-1, 1) if self.test_data['y'].ndim==1 else self.test_data['y'])

            # 3. Construct DataFrame
            df_train_fold = pd.DataFrame()
            if fold_train_ids is not None:
                df_train_fold['ID'] = fold_train_ids
            df_train_fold["Dataset"] = "Train"
            for idx, target in enumerate(self.args.target_columns):
                df_train_fold[f"{target}_Actual"] = y_train_true_orig[:, idx]
                df_train_fold[f"{target}_Predicted"] = y_train_pred_orig[:, idx]
            
            df_val_fold = pd.DataFrame()
            if fold_val_ids is not None:
                df_val_fold['ID'] = fold_val_ids
            df_val_fold["Dataset"] = "Validation"
            for idx, target in enumerate(self.args.target_columns):
                df_val_fold[f"{target}_Actual"] = y_val_true_orig[:, idx]
                df_val_fold[f"{target}_Predicted"] = y_val_pred_orig[:, idx]
            
            df_test_fold = pd.DataFrame()
            if 'ids' in self.test_data:
                df_test_fold['ID'] = self.test_data['ids']
            df_test_fold["Dataset"] = "Test"
            for idx, target in enumerate(self.args.target_columns):
                df_test_fold[f"{target}_Actual"] = y_test_true_orig[:, idx]
                df_test_fold[f"{target}_Predicted"] = y_test_pred_orig[:, idx]
            
            # Combine
            df_all_fold = pd.concat([df_train_fold, df_val_fold, df_test_fold], ignore_index=True)
            
            # Save
            all_pred_path = os.path.join(fold_pred_dir, "all_predictions.csv")
            df_all_fold.to_csv(all_pred_path, index=False)
            # print(f"[{now()}] Saved all predictions for fold {fold+1} to {all_pred_path}")

        avg_metrics = {}
        # Process and save aggregated CV metrics
        if all_fold_metrics:
            df_folds = pd.DataFrame(all_fold_metrics)
            
            # Save raw fold results to CSV
            # Save raw fold results to CSV
            cv_csv_path = os.path.join(cv_output_dir, 'cross_validation_results.csv')
            df_folds.to_csv(cv_csv_path, index=False)
            print(f"[{now()}] Saved detailed cross-validation results to: {cv_csv_path}")
            
            # Calculate and save summary stats
            summary = df_folds.describe().transpose()[['mean', 'std']]
            
            # Save summary stats to CSV
            # Save summary stats to CSV
            cv_stats_csv_path = os.path.join(cv_output_dir, 'cross_validation_stats.csv')
            summary.to_csv(cv_stats_csv_path)
            print(f"[{now()}] Saved cross-validation stats to: {cv_stats_csv_path}")

            avg_metrics = summary['mean'].to_dict()
            std_metrics = summary['std'].to_dict()
            
            # Construct a combined dict for output
            final_cv_summary = {}
            for k, v in avg_metrics.items():
                if k != 'fold':
                    final_cv_summary[f"{k}_mean"] = v
                    final_cv_summary[f"{k}_std"] = std_metrics.get(k, 0.0)
            
            # Save JSON
            # Save JSON
            with open(os.path.join(cv_output_dir, 'cv_avg_metrics.json'), 'w') as f:
                json.dump(final_cv_summary, f, indent=4)
            print(f"[{now()}] Saved cross-validation summary to: cv_avg_metrics.json")


        print(f"[{now()}] Best fold found: {best_fold_num} with validation loss: {best_fold_val_loss:.4f}")

        # --- Enhanced CV Results aggregation and saving ---
        # 1. Collect fold metrics
        # Note: 'history' contains the training history. We need the LAST (or best) metrics of each fold.
        # Since 'history' is overwritten in the loop, we should have probably collected it.
        # But wait, the loop doesn't save metrics per fold in a list currently. 
        # I need to modify the loop to collect `val_loss` and `val_r2` (and maybe others) for each fold.
        
        # This requires modifying the loop above.
        # Let's pivot to a slightly different strategy in this chunk - add the collection logic first, then saving.
        # But this replace block covers the end of the method. I should probably do a broader replace or two checks.
        
        # Let's save what we have first (best model), then I will assume I've modified the loop to collect 'all_fold_metrics'.
        
        if best_fold_history:
            loss_dir = os.path.join(self.args.result_dir, "loss")
            os.makedirs(loss_dir, exist_ok=True)
            
            history_plot_path = os.path.join(loss_dir, "best_fold_training_curves.png")
            BaseEvaluator.plot_training_curves(history=best_fold_history, save_path=history_plot_path, title=f"Best Fold ({best_fold_num}) Training Curves")
            print(f"[{now()}] Saved best fold training curves to: {history_plot_path}")

            history_df = pd.DataFrame(best_fold_history)
            history_csv_path = os.path.join(loss_dir, "best_fold_training_history.csv")
            history_df.to_csv(history_csv_path, index=False)
            print(f"[{now()}] Saved best fold training history to: {history_csv_path}")

        if best_model_state_dict:
            self._save_best_model(best_model_state_dict, best_model_config, best_fold_num)
        else:
            print(f"[{now()}] No best model found during cross-validation.")

        # Cache best fold info for Optuna CV runs
        self._last_cv_best_model_state_dict = best_model_state_dict
        self._last_cv_best_model_config = best_model_config
        self._last_cv_best_fold_num = best_fold_num

        # --- Closest to Mean Logic ---
        if all_fold_metrics and all_fold_state_dicts:
             # Calculate Mean Val R2 first (across all targets)
             fold_r2s = []
             for fm in all_fold_metrics:
                 # Filter keys ending with _r2
                 r2_vals = [v for k,v in fm.items() if k.endswith('_r2')]
                 if r2_vals:
                     fold_r2s.append(np.mean(r2_vals))
                 else:
                     fold_r2s.append(0.0) # Fallback
             
             mean_r2 = np.mean(fold_r2s)
             closest_idx = np.argmin(np.abs(np.array(fold_r2s) - mean_r2))
             closest_fold_num = all_fold_metrics[closest_idx]['fold']
             
             print(f"[{now()}] Mean Validation R2: {mean_r2:.4f}")
             print(f"[{now()}] Closest Fold to Mean: Fold {closest_fold_num} (R2: {fold_r2s[closest_idx]:.4f})")
             
             # Create output directory
             closest_result_dir = os.path.join(cv_output_dir, closest_dir_name)
             os.makedirs(closest_result_dir, exist_ok=True)
             
             # Save Fold Info
             info = {
                 "mean_val_r2": mean_r2,
                 "closest_fold": closest_fold_num,
                 "fold_val_r2": fold_r2s[closest_idx],
                 "all_fold_val_r2": fold_r2s
             }
             with open(os.path.join(closest_result_dir, "closest_to_mean_fold_info.json"), 'w') as f:
                 json.dump(info, f, indent=4)
                 
             # Save Closest Model
             closest_state_dict = all_fold_state_dicts[closest_idx]
             torch.save({
                 'model_state_dict': closest_state_dict,
                 'config': model_config if model_config else best_model_config # Use best_model_config if model_config is None
             }, os.path.join(closest_result_dir, "closest_to_mean_model.pt"))

             # Evaluate this model
             # Re-create model
             # If model_config was passed, use it. Otherwise use best_model_config (which was set during loop)
             # But best_model_config changes every fold? No, model_config_fold changes.
             # In loop: best_model_config = model_config_fold (if best).
             # If we use `model_config` arg properly, it is cleaner.
             # If model_config is None, we need to reconstruct config?
             # But in Optuna pipeline, model_config is passed.
             # In standard CV, model_config is None? 
             # Loop Line 266: `model_fold, model_config_fold = self._create_model(config=model_config)`
             # So `model_config_fold` is robust.
             # We didn't save `model_config_fold` for every trial in `all_fold_state_dicts`.
             # We only saved state_dict.
             # Assumption: Config is same for all folds in CV (usually true, except random seed?).
             # So we can use `best_model_config` as a proxy if `model_config` is None.
             eval_config = model_config if model_config else best_model_config
             
             if eval_config:
                 eval_model_closest, _ = self._create_model(config=eval_config)
                 eval_model_closest.load_state_dict(closest_state_dict)
                 eval_model_closest.to(self.args.device)
                 eval_model_closest.eval()
                 
                 # Create evaluators
                 evaluator_closest = cast(AlloysEvaluator, EvaluatorFactory.create_evaluator(
                    'alloys', result_dir=closest_result_dir, model_name=f"{closest_dir_name}_evaluation",
                    target_names=self.args.target_columns, target_scaler=self.scaler_y
                 ))
                 
                 # Evaluate
                 evaluator_closest.evaluate_model(
                    model=eval_model_closest, 
                    train_data=self.train_val_data, 
                    test_data=self.test_data, 
                    val_data=None, 
                    save_prefix=f"{closest_dir_name}_", 
                    feature_names=self.feature_names
                 )
             
             # Copy Predictions
             source_fold_dir = os.path.join(fold_pred_base_dir, f"fold_{closest_fold_num}")
             if os.path.exists(source_fold_dir):
                 for item in os.listdir(source_fold_dir):
                     s = os.path.join(source_fold_dir, item)
                     d = os.path.join(closest_result_dir, item)
                     if os.path.isfile(s) and not os.path.exists(d):
                         shutil.copy2(s, d)
                     elif os.path.isdir(s) and not os.path.exists(d):
                         shutil.copytree(s, d)
                 print(f"[{now()}] Copied predictions from fold {closest_fold_num} to {closest_result_dir}")

        return avg_metrics

    def _get_training_params(self, trial=None):
        """Constructs training parameters dictionary."""
        if trial:
            # 改进超参数搜索范围
            params = {
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'device': self.args.device,
                'patience': self.args.patience,
                'use_lr_scheduler': trial.suggest_categorical('use_lr_scheduler', [True, False]),
                'lr_scheduler_patience': trial.suggest_int('lr_scheduler_patience', 5, 20),
                'lr_scheduler_factor': trial.suggest_float('lr_scheduler_factor', 0.3, 0.8),
            }
        else:
            params = {
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'weight_decay': self.args.weight_decay,
                'device': self.args.device,
                'patience': self.args.patience,
                'use_lr_scheduler': self.args.use_lr_scheduler,
                'lr_scheduler_patience': self.args.lr_scheduler_patience,
                'lr_scheduler_factor': self.args.lr_scheduler_factor,
            }
        return params

    def _save_best_model(self, state_dict, config, fold_num=None):
        """Saves the best model's state dict and config."""
        checkpoints_dir = os.path.join(self.args.result_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
        
        if state_dict:
            save_obj = {
                'model_state_dict': state_dict,
                'config': config
            }
            torch.save(save_obj, self.best_model_path)
            fold_str = f" from fold {fold_num}" if fold_num is not None else ""
            print(f"[{now()}] Saved best model{fold_str} to {self.best_model_path}")

        # Save human-readable config
        full_config = {'args': vars(self.args), 'model_config': config}
        if fold_num is not None:
            key = 'best_fold' if self.args.cross_validate else 'best_trial'
            full_config[key] = fold_num
            
        with open(os.path.join(self.args.result_dir, "best_model_config.json"), 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=4, default=str)

    def _train_final_model(self):
        """Trains a single model on a single data split."""
        print(f"[{now()}] Training final model on a single data split...")
        # 使用预分割的数据保证一致性
        final_train_data = {'X': self.eval_train_X, 'y': self.eval_train_y}
        final_val_data = {'X': self.eval_val_X, 'y': self.eval_val_y}
        
        # 如果有ID数据，也要添加
        if hasattr(self, 'eval_train_ids'):
            final_train_data['ids'] = self.eval_train_ids
            final_val_data['ids'] = self.eval_val_ids

        model, model_config = self._create_model() # Create the final model

        final_trainer = TrainerFactory.create_trainer(
            model_type=self.args.model_type, model=model, result_dir=self.args.result_dir,
            model_name="best_model", target_names=self.args.target_columns,
            train_data=final_train_data, val_data=final_val_data,
            training_params=self._get_training_params()
        )
        history, _, best_state_dict = final_trainer.train(num_epochs=self.args.epochs)
        
        if history:
            loss_dir = os.path.join(self.args.result_dir, "loss")
            os.makedirs(loss_dir, exist_ok=True)
            
            history_plot_path = os.path.join(loss_dir, "training_curves.png")
            BaseEvaluator.plot_training_curves(history=history, save_path=history_plot_path, title="Training Curves")
            print(f"[{now()}] Saved training curves to: {history_plot_path}")

            history_df = pd.DataFrame(history)
            history_csv_path = os.path.join(loss_dir, "training_history.csv")
            history_df.to_csv(history_csv_path, index=False)
            print(f"[{now()}] Saved training history to: {history_csv_path}")

        # Save the best model from the single training run
        if best_state_dict:
            self._save_best_model(best_state_dict, model_config)
        else:
            print(f"[{now()}] No best model found during training.")

    def _evaluate_best_model(self):
        """Evaluates the best model on the test set."""
        print(f"[{now()}] Evaluating the best model...")
        if not self.best_model_path or not os.path.exists(self.best_model_path):
                 print(f"[{now()}] ERROR: Best model not found at {self.best_model_path}. Skipping evaluation.")
                 return

        device = self.args.device or 'cpu'
        loaded_object = torch.load(self.best_model_path, map_location=device)
        model_config = loaded_object['config']
        model_state_dict = loaded_object['model_state_dict']

        eval_model, _ = self._create_model(config=model_config) # Create model for evaluation
        self._load_best_model_weights(eval_model, model_state_dict)

        # 使用预分割的数据保证一致性
        eval_train_data = {'X': self.eval_train_X, 'y': self.eval_train_y}
        eval_val_data = {'X': self.eval_val_X, 'y': self.eval_val_y}
        
        # 如果有ID数据，也要添加
        if hasattr(self, 'eval_train_ids'):
            eval_train_data['ids'] = self.eval_train_ids
            eval_val_data['ids'] = self.eval_val_ids
        
        # The test data's 'y' is scaled (standardized) in _prepare_data method
        eval_test_data = self.test_data

        evaluator = cast(AlloysEvaluator, EvaluatorFactory.create_evaluator(
            'alloys', result_dir=self.args.result_dir, model_name='best_model_evaluation',
            target_names=self.args.target_columns, target_scaler=self.scaler_y
        ))
        
        evaluator.evaluate_model(
            model=eval_model, 
            train_data=eval_train_data,
            test_data=eval_test_data, 
            val_data=eval_val_data,
            save_prefix='best_model_', 
            feature_names=self.feature_names
        )

        # Save model structure
        model_structure_path = os.path.join(self.args.result_dir, "model_structure.txt")
        with open(model_structure_path, 'w', encoding='utf-8') as f:
            f.write(str(eval_model))
        print(f"[{now()}] Saved model structure to: {model_structure_path}")

    def _load_best_model_weights(self, model, model_state_dict=None):
        """Loads the weights from the best model checkpoint."""
        device = self.args.device or 'cpu'
        if model_state_dict is None:
            loaded_object = torch.load(self.best_model_path, map_location=device)
            model_state_dict = loaded_object.get("model_state_dict", loaded_object)
        
        is_parallel = isinstance(model, torch.nn.DataParallel)
        is_saved_parallel = any(k.startswith('module.') for k in model_state_dict.keys())

        if is_parallel and not is_saved_parallel:
            model.module.load_state_dict(model_state_dict)
        elif not is_parallel and is_saved_parallel:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        
        model.to(device)
        model.eval()
        print(f"[{now()}] Successfully loaded best model weights.")

    def _run_optuna(self):
        """Runs hyperparameter optimization using Optuna."""
        print(f"[{now()}] Starting Optuna hyperparameter optimization...")
        # 使用最小化负R²，等效于最大化R²
        study = optuna.create_study(
            direction='minimize', 
            study_name=self.args.study_name, 
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.args.random_state)
        )
        study.optimize(self._objective, n_trials=self.args.n_trials)

        print(f"[{now()}] Best trial finished with value: {study.best_value} and params: {study.best_params}")
        
        # Save best parameters and evaluate the best model
        best_params_path = os.path.join(self.args.result_dir, "optuna_best_params.json")
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(study.best_trial.params, f, indent=4)
        print(f"[{now()}] Saved best Optuna parameters to: {best_params_path}")

        # Save Optuna Study History (Trials)
        try:
            trials_df = study.trials_dataframe()
            history_path = os.path.join(self.args.result_dir, "optuna_study_history.csv")
            trials_df.to_csv(history_path, index=False)
            print(f"[{now()}] Saved Optuna study history to: {history_path}")
        except Exception as e:
            print(f"[{now()}] Warning: Failed to save Optuna study history. Error: {e}")

        # Persist the best model from the best trial
        best_trial = study.best_trial
        
        history = best_trial.user_attrs.get("training_history")
        if history:
            loss_dir = os.path.join(self.args.result_dir, "loss")
            os.makedirs(loss_dir, exist_ok=True)

            history_plot_path = os.path.join(loss_dir, "optuna_best_trial_training_curves.png")
            title = f"Best Optuna Trial ({best_trial.number}) Training Curves"
            BaseEvaluator.plot_training_curves(history=history, save_path=history_plot_path, title=title)
            print(f"[{now()}] Saved best trial training curves to: {history_plot_path}")

            history_df = pd.DataFrame(history)
            history_csv_path = os.path.join(loss_dir, "optuna_best_trial_training_history.csv")
            history_df.to_csv(history_csv_path, index=False)
            print(f"[{now()}] Saved best trial training history to: {history_csv_path}")

        best_model_state_dict = best_trial.user_attrs.get("best_model_state_dict")
        best_model_config = best_trial.user_attrs.get("model_config")

        if best_model_state_dict and best_model_config:
            self._save_best_model(best_model_state_dict, best_model_config, best_trial.number)
            
            # If cross-validation is requested, run it now with the best config to get uncertainty stats
            if self.args.cross_validate:
                print(f"[{now()}] Running final cross-validation with best Optuna config for statistical reporting...")
                
                # 1. Best Trial -> (Save to closest_to_mean_predictions in root)
                # Pass trial_number=None to save to root results (Standard)
                self._run_cross_validation(
                    model_config=best_model_config, 
                    trial_number=None,
                    closest_dir_name="closest_to_mean_predictions"
                )
                
                # 2. Global Mean Trial -> (Save to closest_to_global_mean_predictions in root)
                print(f"[{now()}] Finding and running Global Mean Trial...")
                valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                # Optuna minimizes negative R2, so t.value is -R2 (or Loss).
                all_values = [t.value for t in valid_trials if t.value is not None]
                
                if all_values:
                    global_mean_val = np.mean(all_values)
                    
                    # Find closest trial
                    closest_trial = min(valid_trials, key=lambda t: abs(t.value - global_mean_val))
                    print(f"  Global Mean Value: {global_mean_val:.4f}")
                    print(f"  Closest Trial: {closest_trial.number} (Value: {closest_trial.value:.4f})")
                    
                    global_config = closest_trial.user_attrs.get("model_config")
                    
                    if global_config:
                        print(f"[{now()}] Re-running cross-validation for Global Mean Trial...")
                        self._run_cross_validation(
                            model_config=global_config,
                            trial_number=None,
                            closest_dir_name="closest_to_global_mean_predictions"
                        )
                    else:
                        print(f"[{now()}] Warning: No model_config found for Global Mean Trial {closest_trial.number}")
            
            self._evaluate_best_model()
        else:
            print(f"[{now()}] No best model state dict or config found for best trial. Skipping evaluation.")

    def _objective(self, trial: optuna.Trial):
        """Objective function for Optuna optimization."""
        
        # 1. Generate model configuration for this trial
        # We call _create_model to sample hyperparameters and get the config
        # We don't need the model instance here if we are running CV, as CV will recreate it per fold
        _, model_config = self._create_model(trial)

        if self.args.cross_validate:
            # Case A: Run Cross-Validation for every trial
            print(f"[{now()}] [Trial {trial.number}] Running cross-validation...")
            
            # This will save results to predictions/optuna_trials/trial_{number}/
            avg_metrics = self._run_cross_validation(model_config=model_config, trial_number=trial.number)
            best_state = getattr(self, "_last_cv_best_model_state_dict", None)
            best_config = getattr(self, "_last_cv_best_model_config", None)
            best_fold = getattr(self, "_last_cv_best_fold_num", None)
            if best_state is not None and best_config is not None:
                trial.set_user_attr("best_model_state_dict", best_state)
                trial.set_user_attr("model_config", best_config)
                if best_fold is not None:
                    trial.set_user_attr("best_fold", best_fold)
            
            # Calculate objective value (Maximize R2 -> Minimize negative R2)
            # avg_metrics contains keys like "{target}_r2"
            r2_keys = [k for k in avg_metrics.keys() if k.endswith('_r2')]
            if r2_keys:
                avg_r2 = np.mean([avg_metrics[k] for k in r2_keys])
                return -avg_r2 # Optuna minimizes
            else:
                # Fallback if no R2 found (e.g. classification?), use loss if available or 0
                # Assuming regression tasks here
                print(f"[{now()}] Warning: No R2 metrics found in CV results. returning 0.")
                return 0.0

        else:
            # Case B: Standard Single Split (Original Logic)
            # 使用预分割的数据保证一致性
            train_data_split = {'X': self.eval_train_X, 'y': self.eval_train_y}
            val_data_split = {'X': self.eval_val_X, 'y': self.eval_val_y}
            
            # 如果有ID数据，也要添加
            if hasattr(self, 'eval_train_ids'):
                train_data_split['ids'] = self.eval_train_ids
                val_data_split['ids'] = self.eval_val_ids
    
            model, _ = self._create_model(config=model_config) # Re-create model with the config (or use trial)
    
            training_params = self._get_training_params(trial)
            training_params['save_config_on_init'] = False
            training_params['save_checkpoints'] = True # Enable for Optuna to save best trial model
            
            num_epochs = trial.suggest_int('num_epochs', 100, 300, step=10)
            
            # Use trial-specific directory for consistent saving even in single split
            trial_dir = os.path.join(self.args.result_dir, "predictions", "optuna_trials", f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            
            model_name = f"alloy_nn_optuna_trial_{trial.number}"
            trainer = TrainerFactory.create_trainer(
                model_type=self.args.model_type, model=model, result_dir=trial_dir, model_name=model_name,
                target_names=self.args.target_columns, train_data=train_data_split,
                val_data=val_data_split, training_params=training_params
            )
            history, best_val_loss, best_model_state_dict = trainer.train(num_epochs=num_epochs)
    
            trial.set_user_attr("training_history", history)
            if best_model_state_dict:
                trial.set_user_attr("best_model_state_dict", best_model_state_dict)
                trial.set_user_attr("model_config", model_config)
            
            # 计算平均验证R²作为优化目标（最大化）
            # 从历史记录中获取最佳验证R²
            if history and 'val_r2' in history:
                best_val_r2 = max(history['val_r2'])
                # Optuna进行最小化，所以返回负的R²
                return -best_val_r2
            else:
                # 如果没有R²数据，则使用验证损失
                return best_val_loss

