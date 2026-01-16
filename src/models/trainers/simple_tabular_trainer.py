"""
A simple trainer for a tabular model with two-part input.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import joblib
import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

from .base_trainer import BaseTrainer
from src.models.base.alloys_nn import PredictionNetwork
from src.models.visualization.plot_utils import plot_compare_scatter, plot_all_sets_compare_scatter
from src.models.evaluators.base_evaluator import BaseEvaluator

from src.models.evaluators.shap_analyzer import NeuralNetworkShapAnalyzer # New import

def now():
    """Return current time string for logging."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def load_data(data_file: str,
              target_columns: List[str], 
              test_size: float = 0.2,
              random_state: int = 42,
              processing_cols: List[str] = []) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    Load data from a CSV file, select features, and split into train/test sets.
    
    Args:
        data_file: Path to the CSV data file
        target_columns: List of target column names
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility 
        processing_cols: List of processing column names (optional)
    
    Returns:
        Tuple containing:
        - train_val_data: Dict with training/validation data
        - test_data: Dict with test data
        - selected_cols: List of selected feature column names
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    print(f"[{now()}] Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    
    # Validate target columns exist
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in data: {missing_targets}")

    # Get elemental composition columns (wt% and at%)
    elemental_cols = [col for col in data.columns if col.endswith('(wt%)') or col.endswith('(at%)')]
    
    # Select features based on whether processing_cols is provided
    if processing_cols:
        # Only use processing columns that exist in data
        valid_processing_cols = [col for col in processing_cols if col in data.columns]
        selected_cols = elemental_cols + valid_processing_cols
    else:
        # If no processing_cols, just use elemental columns
        selected_cols = elemental_cols
        
    print(f"[{now()}] Selected {len(selected_cols)} features for X: {selected_cols}")

    # Fill missing values with 0
    data[selected_cols] = data[selected_cols].fillna(0)
    data[target_columns] = data[target_columns].fillna(0)
    
    # Convert to numpy arrays
    X = data[selected_cols].values.astype('float32')
    y = data[target_columns].values.astype('float32')

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Package data into dictionaries
    train_val_data = {'X': X_train_val, 'y': y_train_val}
    test_data = {'X': X_test, 'y': y_test}
    
    return train_val_data, test_data, selected_cols


class SimpleTabularTrainer(BaseTrainer):
    """
    Trainer for the simple tabular model.
    """
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device, config, **kwargs):
        super().__init__(model=model, result_dir=config['result_dir'], model_name=config['model_name'], target_names=config['target_columns'])
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.early_stopping_patience = config.get('patience', 10)
        self.early_stopping_delta = config.get('early_stopping_delta', 1e-4)
        self.model.to(self.device)
        self.feature_names = config.get('feature_names', [])  # Store feature names

    def _reset_weights(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_targets, all_outputs = [], []
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_targets.append(batch_y.cpu().detach())
            all_outputs.append(outputs.cpu().detach())
        
        avg_loss = total_loss / len(self.train_loader)
        all_targets = torch.cat(all_targets).numpy()
        all_outputs = torch.cat(all_outputs).numpy()
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        mae = mean_absolute_error(all_targets, all_outputs)
        r2 = r2_score(all_targets, all_outputs)
        return {'loss': avg_loss, 'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': all_outputs, 'targets': all_targets}

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                all_targets.append(batch_y.cpu())
                all_outputs.append(outputs.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        all_targets = torch.cat(all_targets).numpy()
        all_outputs = torch.cat(all_outputs).numpy()

        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        mae = mean_absolute_error(all_targets, all_outputs)
        r2 = r2_score(all_targets, all_outputs)
        return {'loss': avg_loss, 'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': all_outputs, 'targets': all_targets}
    
    def train_cross_validation(self, num_epochs, kf, X_cv, y_cv, scaler_y, test_data):
        all_fold_val_metrics = []
        
        print(f"\n[{now()}] Starting {kf.get_n_splits()}-fold cross-validation...")
        plots_dir = os.path.join(self.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        for fold, (train_index, val_index) in enumerate(kf.split(X_cv)):
            print(f"\n[{now()}] --- Fold {fold+1}/{kf.get_n_splits()} ---")

            self.model.apply(self._reset_weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
            
            patience_counter = 0
            best_val_loss = float('inf')

            X_train, X_val = X_cv[train_index], X_cv[val_index]
            y_train, y_val = y_cv[train_index], y_cv[val_index]

            self.train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=self.config['batch_size'], shuffle=True)
            self.val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=self.config['batch_size'], shuffle=False)
            
            best_fold_val_metrics = {}
            history = {key: [] for key in ['train_loss', 'train_rmse', 'train_mae', 'train_r2', 'val_loss', 'val_rmse', 'val_mae', 'val_r2']}

            for epoch in range(num_epochs):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                
                # Store history
                for metric in ['loss', 'rmse', 'mae', 'r2']:
                    history[f'train_{metric}'].append(train_metrics[metric])
                    history[f'val_{metric}'].append(val_metrics[metric])

                print(f"[{now()}] Fold {fold+1} Epoch {epoch+1}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f} R2: {train_metrics['r2']:.4f} | Val Loss: {val_metrics['loss']:.4f} R2: {val_metrics['r2']:.4f}", flush=True)

                if val_metrics['loss'] < best_val_loss - self.early_stopping_delta:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    best_fold_val_metrics = val_metrics
                    # Save best model for this fold
                    best_path = os.path.join(self.checkpoints_dir, f'fold_{fold+1}_best.pth')
                    torch.save(self.model.state_dict(), best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"[{now()}] Early stopping triggered in fold {fold+1} after {epoch+1} epochs.")
                        break
            
            # Plot training curves for the fold
            save_path = os.path.join(plots_dir, f'training_curves_fold_{fold+1}.png')
            BaseEvaluator.plot_training_curves(None, history, save_path, title=f"Training Curves Fold {fold+1}")
            print(f"[{now()}] Training curves for fold {fold+1} saved to {save_path}")
            all_fold_val_metrics.append(best_fold_val_metrics)

        # Aggregate and print results
        mean_r2 = np.mean([m['r2'] for m in all_fold_val_metrics if m])
        std_r2 = np.std([m['r2'] for m in all_fold_val_metrics if m])
        mean_rmse = np.mean([m['rmse'] for m in all_fold_val_metrics if m])
        std_rmse = np.std([m['rmse'] for m in all_fold_val_metrics if m])
        
        print(f"\n[{now()}] Cross-validation finished.")
        print(f"Average Validation R2:   {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Average Validation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        # Identify best fold and load the best model
        best_fold_index = np.argmin([m['loss'] for m in all_fold_val_metrics if m])
        best_model_path = os.path.join(self.checkpoints_dir, f'fold_{best_fold_index+1}_best.pth')
        print(f"[{now()}] Loading best model from fold {best_fold_index+1} for final evaluation.")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        # Prepare data from the best fold and the hold-out test set
        best_fold_train_indices, best_fold_val_indices = list(kf.split(X_cv))[best_fold_index]
        
        X_train_best = X_cv[best_fold_train_indices]
        y_train_best = y_cv[best_fold_train_indices]
        X_val_best = X_cv[best_fold_val_indices]
        y_val_best = y_cv[best_fold_val_indices]

        # Get predictions for all sets
        with torch.no_grad():
            train_preds_scaled = self.model(torch.from_numpy(X_train_best).to(self.device)).cpu().numpy()
            val_preds_scaled = self.model(torch.from_numpy(X_val_best).to(self.device)).cpu().numpy()
            test_preds_scaled = self.model(torch.from_numpy(test_data['X']).to(self.device)).cpu().numpy()

        # Inverse transform
        train_true_unscaled = scaler_y.inverse_transform(y_train_best)
        train_pred_unscaled = scaler_y.inverse_transform(train_preds_scaled)
        val_true_unscaled = scaler_y.inverse_transform(y_val_best)
        val_pred_unscaled = scaler_y.inverse_transform(val_preds_scaled)
        test_true_unscaled = scaler_y.inverse_transform(test_data['y'])
        test_pred_unscaled = scaler_y.inverse_transform(test_preds_scaled)
        
        plot_save_all_path = os.path.join(plots_dir, "cross_validation_prediction_comparison_scatter.png")
        plot_all_sets_compare_scatter(
            train_data=(train_true_unscaled, train_pred_unscaled),
            val_data=(val_true_unscaled, val_pred_unscaled),
            test_data=(test_true_unscaled, test_pred_unscaled),
            target_names=self.config['target_columns'],
            save_path=plot_save_all_path
        )
        print(f"[{now()}] Aggregated cross-validation scatter plot saved to {plot_save_all_path}")

        plot_save_path = os.path.join(plots_dir, "prediction_comparison_scatter.png")
        plot_compare_scatter(
            train_true=train_true_unscaled,
            train_pred=train_pred_unscaled,
            test_true=test_true_unscaled,
            test_pred=test_pred_unscaled,
            target_names=self.config['target_columns'],
            save_path=plot_save_path
        )
        print(f"[{now()}] Prediction scatter plot saved to {plot_save_path}")
        
        # Perform SHAP analysis (if enabled)
        if self.config.get('run_shap_analysis', True) and self.feature_names:
            self._perform_shap_analysis(
                model=self.model,
                x_train_fold_tensor=torch.from_numpy(X_train_best),
                x_test_fold_tensor=torch.from_numpy(test_data['X']),
                feature_names=self.feature_names,
                target_names=self.target_names,
                fold_results_dir=plots_dir,
                device=self.device
            )

    def save_checkpoint(self, epoch, is_best=False):
        filename = f"checkpoint_{'best' if is_best else epoch}.pth"
        filepath = os.path.join(self.checkpoints_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.history['best_val_loss']
        }, filepath)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
        
    def _perform_shap_analysis(self, model, x_train_fold_tensor, x_test_fold_tensor, feature_names, target_names, fold_results_dir, device):
        """Perform SHAP analysis and save the results."""
        print(f"[{now()}] Starting SHAP analysis...")
        
        shap_analyzer = NeuralNetworkShapAnalyzer(
            model=model,
            feature_names=feature_names,
            target_names=target_names,
            result_dir=self.result_dir,
            device=device
        )
        
        train_data = x_train_fold_tensor.cpu().numpy()
        test_data = x_test_fold_tensor.cpu().numpy()

        # Use the entire training set as background data
        background_data = train_data

        # Use the entire test set for SHAP analysis
        analysis_data = test_data
        
        print(f"[{now()}] Using all {background_data.shape[0]} samples from the training set as background data.")
        print(f"[{now()}] Using all {analysis_data.shape[0]} samples from the test set for SHAP analysis.")
        
        # Analyze feature importance and generate plots
        feature_importance = shap_analyzer.analyze_feature_importance(
            X=analysis_data,
            background_data=background_data,
            max_display=min(30, len(feature_names))
        )
        
        # Print top 10 important features for each target
        print(f"\n[{now()}] SHAP Feature Importance Ranking:")
        for target, importance in feature_importance.items():
            print(f"\nTarget: {target}")
            
            try:
                # Ensure importance is a 1D array
                if isinstance(importance, np.ndarray) and importance.ndim > 1:
                    print(f"  Note: Feature importance is multi-dimensional, using mean for sorting.")
                    importance = np.mean(importance, axis=tuple(range(1, importance.ndim)))
                
                # Create feature importance ranking
                importance_pairs = list(zip(feature_names, importance))
                
                # Ensure importance value is a scalar; if not, take the mean of absolute values
                processed_pairs = []
                for feature, imp in importance_pairs:
                    if hasattr(imp, '__len__') and not isinstance(imp, (str, bytes)):
                        imp_value = float(np.abs(imp).mean())
                    else:
                        imp_value = float(imp)
                    processed_pairs.append((feature, imp_value))
                
                # Sort
                processed_pairs.sort(key=lambda x: x[1], reverse=True)
                top_features = processed_pairs[:10]
                
                # Print ranking
                for i, (feature, imp) in enumerate(top_features, 1):
                    # Convert any non-string feature names to string before formatting
                    feature_str = str(feature) if not isinstance(feature, str) else feature
                    print(f"  {i}. {feature_str}: {imp:.6f}")
                    
                # Prepare data for CSV
                feature_importance_df = pd.DataFrame(processed_pairs, columns=['Feature', 'Importance'])
                csv_save_path = os.path.join(fold_results_dir, f'feature_importance_{target}.csv')
                feature_importance_df.to_csv(csv_save_path, index=False)
                print(f"  Feature importance saved to: {csv_save_path}")

            except Exception as e:
                print(f"  Error processing feature importance: {str(e)}")
                print(f"  Importance data type: {type(importance)}, shape: {getattr(importance, 'shape', 'N/A')}")
                
        print(f"\n[{now()}] SHAP analysis complete. Detailed results and plots saved to: {fold_results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple Tabular Model Trainer')
    parser.add_argument('--data_file', type=str, default='datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv', help='Path to the dataset CSV file.')
    parser.add_argument('--result_dir', type=str, default='output/results/Ti_alloys/Xue/Mechanical/simple_tabular_trainer', help='Directory to save results.')
    parser.add_argument('--target_columns', type=str, nargs='+', default=['UTS(MPa)', 'El(%)'], help='List of target column names.')
    parser.add_argument('--processing_cols', type=str, nargs='+', default=[])
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset for the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--cross_validate', action='store_true', help='Enable K-fold cross-validation.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
    parser.add_argument('--run_shap_analysis', action='store_true', help='Run SHAP feature importance analysis after training.')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    # Load data
    train_val_data, test_data, feature_names = load_data(
        data_file=args.data_file,
        target_columns=args.target_columns,
        test_size=args.test_size,
        random_state=args.random_state,
        processing_cols=args.processing_cols
    )

    # Scale features and targets
    scaler_X = StandardScaler().fit(train_val_data['X'])
    train_val_data['X'] = scaler_X.transform(train_val_data['X'])
    test_data['X'] = scaler_X.transform(test_data['X'])
    joblib.dump(scaler_X, os.path.join(args.result_dir, 'scaler_X.pkl'))

    scaler_y = StandardScaler().fit(train_val_data['y'])
    train_val_data['y'] = scaler_y.transform(train_val_data['y'])
    test_data['y'] = scaler_y.transform(test_data['y'])
    joblib.dump(scaler_y, os.path.join(args.result_dir, 'scaler_y.pkl'))
    
    # Model parameters
    input_dim = train_val_data['X'].shape[1]
    output_dim = train_val_data['y'].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = vars(args)
    config['result_dir'] = args.result_dir
    config['model_name'] = 'SimpleTabularModel'
    config['feature_names'] = feature_names  # 添加特征名称到配置
        
    model = PredictionNetwork(input_dim, output_dim)
    with open(os.path.join(args.result_dir, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))
    if args.cross_validate:
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_state)

        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Optimizer will be re-initialized per fold
        criterion = nn.MSELoss()

        # Dummy loaders, will be replaced in CV loop
        dummy_loader = DataLoader(TensorDataset(torch.empty(0, input_dim), torch.empty(0, output_dim)))

        trainer_params = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'train_loader': dummy_loader,
            'val_loader': dummy_loader,
            'device': device,
            'config': config
        }
        trainer = SimpleTabularTrainer(**trainer_params)
        trainer.train_cross_validation(num_epochs=args.epochs, kf=kf, X_cv=train_val_data['X'], y_cv=train_val_data['y'], scaler_y=scaler_y, test_data=test_data)

    else:
        print(f"[{now()}] Starting single training run...")
        
        # Split train_val_data further into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            train_val_data['X'], train_val_data['y'], test_size=0.25, random_state=args.random_state # 0.25 * 0.8 = 0.2
        )

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = PredictionNetwork(input_dim, output_dim)
        with open(os.path.join(args.result_dir, 'model_architecture.txt'), 'w') as f:
            f.write(str(model))
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        trainer_params = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'device': device,
            'config': config
        }
        
        # We need to redefine BaseTrainer.__init__ to handle the config dict
        base_trainer_params = {
             'model': model,
             'result_dir': args.result_dir,
             'model_name': 'SimpleTabularModel',
             'target_names': args.target_columns,
             'early_stopping_patience': args.patience,
             'early_stopping_delta': 1e-4, # A sensible default
             'feature_names': feature_names
        }
        
        trainer = SimpleTabularTrainer(**base_trainer_params)
        
        # Manually loop epochs to collect full history for plotting
        history = {key: [] for key in ['train_loss', 'train_rmse', 'train_mae', 'train_r2', 'val_loss', 'val_rmse', 'val_mae', 'val_r2']}
        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = trainer.validate()

            # Store history
            for metric in ['loss', 'rmse', 'mae', 'r2']:
                history[f'train_{metric}'].append(train_metrics[metric])
                history[f'val_{metric}'].append(val_metrics[metric])
            
            print(f"[{now()}] Epoch {epoch+1}/{args.epochs} | Train Loss: {train_metrics['loss']:.4f} R2: {train_metrics['r2']:.4f} | Val Loss: {val_metrics['loss']:.4f} R2: {val_metrics['r2']:.4f}", flush=True)

            if val_metrics['loss'] < best_val_loss - trainer.early_stopping_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Use a consistent name for the best model
                torch.save(model.state_dict(), os.path.join(trainer.checkpoints_dir, 'best_model.pth'))
                print(f"[{now()}] Validation loss improved. Saved best model.")
            else:
                patience_counter += 1
                if patience_counter >= trainer.early_stopping_patience:
                    print(f"[{now()}] Early stopping triggered after {epoch+1} epochs.")
                    break

        # Plot training curves for the single run
        plots_dir = os.path.join(args.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, 'training_curves.png')
        BaseEvaluator.plot_training_curves(None, history, save_path, title="Training Curves")

        # --- Prediction and Plotting ---
        print(f"[{now()}] Training finished. Loading best model for evaluation.")
        best_model_path = os.path.join(trainer.checkpoints_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Get predictions for all sets
        with torch.no_grad():
            train_pred_scaled = model(torch.from_numpy(X_train).to(device)).cpu().numpy()
            val_pred_scaled = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
            test_pred_scaled = model(torch.from_numpy(test_data['X']).to(device)).cpu().numpy()

        # Inverse transform predictions and true values
        train_true_unscaled = scaler_y.inverse_transform(y_train)
        train_pred_unscaled = scaler_y.inverse_transform(train_pred_scaled)
        val_true_unscaled = scaler_y.inverse_transform(y_val)
        val_pred_unscaled = scaler_y.inverse_transform(val_pred_scaled)
        test_true_unscaled = scaler_y.inverse_transform(test_data['y'])
        test_pred_unscaled = scaler_y.inverse_transform(test_pred_scaled)

        plot_save_path = os.path.join(plots_dir, "prediction_comparison_scatter.png")
        plot_all_sets_compare_scatter(
            train_data=(train_true_unscaled, train_pred_unscaled),
            val_data=(val_true_unscaled, val_pred_unscaled),
            test_data=(test_true_unscaled, test_pred_unscaled),
            target_names=args.target_columns,
            save_path=plot_save_path
        )
        print(f"[{now()}] Comparison plot saved to {plot_save_path}")
        
        # If SHAP analysis is enabled, perform it
        if args.run_shap_analysis:
            trainer._perform_shap_analysis(
                model=model,
                x_train_fold_tensor=torch.from_numpy(X_train),
                x_test_fold_tensor=torch.from_numpy(test_data['X']),
                feature_names=feature_names,
                target_names=args.target_columns,
                fold_results_dir=plots_dir,
                device=device
            )

"""
python -m src.models.trainers.simple_tabular_trainer \
--cross_validate \
--num_folds 3 \
--epochs 200 \
--batch_size 32 \
--lr 0.001 \
--patience 200 \
--data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
--result_dir "output/results/Ti_alloys/Xue/Mechanical/simple_tabular_trainer_with_shap" \
--target_columns "UTS(MPa)" "El(%)" \
--processing_cols "Solution Temperature(℃)" "Solution Time(h)" "Aging Temperature(℃)" "Aging Time(h)" "Thermo-Mechanical Treatment Temperature(℃)" "Deformation(%)" \
--test_size 0.2 \
--random_state 42 \
--run_shap_analysis
"""