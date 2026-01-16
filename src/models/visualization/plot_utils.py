import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from typing import List, Dict, Optional, Tuple

def plot_training_curves(history, save_path, title="Training Curves"):
    """
    Plot training and validation curves
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot R² curve
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²', color='green')
    plt.title('Validation R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_curves(fold_histories, save_path, title="Cross-Validation Curves"):
    """
    Plot training curves for all folds
    
    Args:
        fold_histories (list): List of dictionaries containing fold histories
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Ensure all histories have the same length
    min_length = min(len(h['train_loss']) for h in fold_histories)
    fold_histories = [{k: v[:min_length] for k, v in h.items()} for h in fold_histories]
    
    # Plot loss curves for each fold
    plt.subplot(2, 2, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['train_loss'], label=f'Fold {i+1} Train', alpha=0.5)
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_loss'], label=f'Fold {i+1} Val', alpha=0.5)
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot R² curves for each fold
    plt.subplot(2, 2, 3)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_r2'], label=f'Fold {i+1}', alpha=0.5)
    plt.title('Validation R² Curves')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot mean and std of validation loss
    plt.subplot(2, 2, 4)
    val_losses = np.array([h['val_loss'] for h in fold_histories])
    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    epochs = range(1, len(mean_val_loss) + 1)
    
    plt.plot(epochs, mean_val_loss, label='Mean Val Loss', color='red')
    plt.fill_between(epochs, 
                     mean_val_loss - std_val_loss,
                     mean_val_loss + std_val_loss,
                     alpha=0.2, color='red', label='±1 Std')
    plt.title('Mean Validation Loss with Std')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(y_true, y_pred, target_names, save_dir):
    """
    Plot prediction scatter plots for each target
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        target_names (list): List of target names
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, target in enumerate(target_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, color='blue')
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        plt.plot(lims, lims, 'r--')
        plt.xlabel(f'True {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Prediction')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{target}_prediction.png'), dpi=300)
        plt.close()

def plot_compare_scatter(
    train_true: np.ndarray,
    train_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    target_names: List[str],
    save_path: str,
    metrics: Optional[Dict] = None
):
    """
    Generate and save a scatter plot comparing training and test set performance.
    
    Args:
        train_true (np.ndarray): True values for the training set.
        train_pred (np.ndarray): Predicted values for the training set.
        test_true (np.ndarray): True values for the test set.
        test_pred (np.ndarray): Predicted values for the test set.
        target_names (List[str]): Names of the target properties.
        save_path (str): Path to save the plot.
        metrics (Optional[Dict]): Dictionary with metrics for R2 and RMSE.
    """
    n_targets = len(target_names)
    
    # Define plot parameters
    plot_params = {
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3
    }
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update(plot_params)

    # Define colors
    colors = {
        'train': '#1f77b4',  # Blue
        'test': '#ff0000'    # red
    }
    
    # Ensure y_true and y_pred are 2D arrays if they exist
    if train_true is not None and train_true.ndim == 1:
        train_true = train_true.reshape(-1, 1)
        train_pred = train_pred.reshape(-1, 1)
    if test_true is not None and test_true.ndim == 1:
        test_true = test_true.reshape(-1, 1)
        test_pred = test_pred.reshape(-1, 1)

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 8), squeeze=False)
    
    for i in range(n_targets):
        ax = axes[0, i]
        
        # Determine plot limits based on available data
        all_vals = []
        if train_true is not None:
            all_vals.extend([train_true[:, i].min(), train_true[:, i].max(), train_pred[:, i].min(), train_pred[:, i].max()])
        if test_true is not None:
            all_vals.extend([test_true[:, i].min(), test_true[:, i].max(), test_pred[:, i].min(), test_pred[:, i].max()])
        
        if not all_vals: # Skip plot if no data is available at all
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            continue
            
        min_val, max_val = min(all_vals), max(all_vals)
        
        # Plot ideal line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
        
        # Plot scatter points for available data
        if train_true is not None:
            ax.scatter(train_true[:, i], train_pred[:, i], c=colors['train'], alpha=0.6, 
                      label='Train', s=50, edgecolors='white', linewidth=0.5)
        if test_true is not None:
            ax.scatter(test_true[:, i], test_pred[:, i], c=colors['test'], alpha=0.7, 
                      label='Test', s=50, marker='s', edgecolors='white', linewidth=0.5)

        ax.set_xlabel('True Values', fontsize=plot_params['axes.labelsize']+2, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=plot_params['axes.labelsize']+2, fontweight='bold')
        ax.set_title(target_names[i], fontsize=plot_params['axes.titlesize']+2, fontweight='bold', pad=15)
        ax.legend(frameon=True, fontsize=plot_params['legend.fontsize']+2)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate metrics for available data
        metric_text = ""
        if train_true is not None:
            train_r2 = r2_score(train_true[:, i], train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(train_true[:, i], train_pred[:, i]))
            metric_text += f"Train: R²={train_r2:.3f}, RMSE={train_rmse:.3f}\n"
        
        if test_true is not None:
            test_r2 = r2_score(test_true[:, i], test_pred[:, i])
            test_rmse = np.sqrt(mean_squared_error(test_true[:, i], test_pred[:, i]))
            metric_text += f"Test:  R²={test_r2:.3f}, RMSE={test_rmse:.3f}"
            
        # Position text at the bottom of the plot
        if metric_text:
            ax.text(0.5, -0.6, metric_text.strip(), transform=ax.transAxes, fontsize=plot_params['font.size'],
                     ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                     fc='white', ec='gray', alpha=0.9))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_sets_compare_scatter(
    train_data: Optional[Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray], 
    target_names: List[str],
    save_path: str,
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    metrics: Optional[Dict] = None
):
    """
    Generate and save a scatter plot comparing train, validation, and test set performance.
    
    Args:
        train_data (Optional[Tuple[np.ndarray, np.ndarray]]): Tuple of (true_values, predicted_values) for training set. Can be None.
        test_data (Tuple[np.ndarray, np.ndarray]): Tuple of (true_values, predicted_values) for test set.
        target_names (List[str]): Names of the target properties.
        save_path (str): Path to save the plot.
        val_data (Optional[Tuple[np.ndarray, np.ndarray]]): Tuple of (true_values, predicted_values) for validation set. Can be None.
        metrics (Optional[Dict]): Pre-calculated metrics for annotation. If None, they are calculated automatically.
    """
    n_targets = len(target_names)
    
    # Define plot parameters and style
    plot_params = {
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    }
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams.update(plot_params)

    # Unpack and ensure 2D arrays
    test_true, test_pred = np.atleast_2d(test_data[0]), np.atleast_2d(test_data[1])
    
    has_train_data = train_data is not None
    has_val_data = val_data is not None
    
    if has_train_data:
        train_true, train_pred = np.atleast_2d(train_data[0]), np.atleast_2d(train_data[1])
    else:
        train_true, train_pred = None, None
        
    if has_val_data:
        val_true, val_pred = np.atleast_2d(val_data[0]), np.atleast_2d(val_data[1])
    else:
        val_true, val_pred = None, None

    # Create figure with custom colors
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 8), squeeze=False)
    
    colors = {
        'train': '#1f77b4',  # Blue
        'val': '#ff7f0e',    # Orange  
        'test': '#ff0000'    # red
    }
    
    for i in range(n_targets):
        ax = axes[0, i]
        
        # Calculate plot limits
        all_values = []
        if has_train_data:
            all_values.extend([train_true[:, i], train_pred[:, i]])
        if has_val_data:
            all_values.extend([val_true[:, i], val_pred[:, i]])
        all_values.extend([test_true[:, i], test_pred[:, i]])
            
        all_values_flat = np.concatenate([arr.flatten() for arr in all_values])
        min_val, max_val = all_values_flat.min(), all_values_flat.max()
        plot_buffer = (max_val - min_val) * 0.05
        plot_min = min_val - plot_buffer
        plot_max = max_val + plot_buffer
        
        # Plot ideal line
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, label='Ideal')
        
        # Plot scatter points with enhanced visibility
        if has_train_data:
            ax.scatter(train_true[:, i], train_pred[:, i], c=colors['train'], alpha=0.6, 
                  label='Train', s=40, edgecolors='white', linewidth=0.5)
        if has_val_data:
            ax.scatter(val_true[:, i], val_pred[:, i], c=colors['val'], alpha=0.7,
                      label='Validation', marker='^', s=40, edgecolors='white', linewidth=0.5)
            
        ax.scatter(test_true[:, i], test_pred[:, i], c=colors['test'], alpha=0.7,
                  label='Test', marker='s', s=40, edgecolors='white', linewidth=0.5)

        # Set axis labels and title
        ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
        ax.set_title(target_names[i], fontsize=16, fontweight='bold', pad=10)
        
        # Customize grid and legend
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(frameon=True, fontsize=10, loc='upper left')
        
        # Set axis limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Calculate and format metrics
        metric_lines = []
        
        # Train metrics
        if has_train_data:
            train_r2 = r2_score(train_true[:, i], train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(train_true[:, i], train_pred[:, i]))
            metric_lines.append(f"Train: R²={train_r2:.3f}, RMSE={train_rmse:.3f}")
        
        # Validation metrics if available
        if has_val_data:
            val_r2 = r2_score(val_true[:, i], val_pred[:, i])
            val_rmse = np.sqrt(mean_squared_error(val_true[:, i], val_pred[:, i]))
            metric_lines.append(f"Val: R²={val_r2:.3f}, RMSE={val_rmse:.3f}")
        
        # Test metrics
        test_r2 = r2_score(test_true[:, i], test_pred[:, i])
        test_rmse = np.sqrt(mean_squared_error(test_true[:, i], test_pred[:, i]))
        metric_lines.append(f"Test: R²={test_r2:.3f}, RMSE={test_rmse:.3f}")
        
        metric_text = "\n".join(metric_lines)
        # Position text box below the plot area
        ax.text(0.5, -0.25, metric_text, transform=ax.transAxes, fontsize=10,
                ha='center', va='top', linespacing=1.2,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 0.92])  # Adjust layout to accommodate title and metrics
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, model_name, feature_names, save_dir):
    """
    Plot feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        model_name (str): Name of the model
        feature_names (list): List of feature names
        save_dir (str): Directory to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not support feature importance")
        return
        
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Save to CSV
    importance_csv_path = os.path.join(save_dir, f'{model_name}_feature_importance.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    
    # Plot top 20 features
    n_features = min(20, len(importances))
    plt.figure(figsize=(12, 8))
    plt.title(f'{model_name} - Feature Importance (Top {n_features})', fontsize=14)
    
    indices = np.argsort(importances)[::-1][:n_features]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_features))
    bars = plt.barh(range(n_features), importances[indices], color=colors)
    plt.yticks(range(n_features), [feature_names[i] for i in indices])
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', fontsize=9)
    
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                                target_names: list, save_path: str, 
                                title: str = "True vs. Predicted Values"):
    """
    Plot true vs. predicted values for each target and annotate with R2 and RMSE.

    Args:
        y_true (np.ndarray): True values (n_samples, n_targets)
        y_pred (np.ndarray): Predicted values (n_samples, n_targets)
        target_names (list): List of target names.
        save_path (str): Path to save the plot.
        title (str, optional): Overall title for the plot. Defaults to "True vs. Predicted Values".
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]
    if n_targets != len(target_names):
        raise ValueError(f"Mismatch between number of targets in y_true/y_pred ({n_targets}) and target_names ({len(target_names)})")

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 8), squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    metrics_summary = []

    for i, target_name in enumerate(target_names):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        metrics_summary.append(f"{target_name}: R2={r2:.3f}, RMSE={rmse:.3f}")

        ax.scatter(true_vals, pred_vals, alpha=0.7, label=f'Predictions')
        
        # Add 1:1 line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0], true_vals.min(), pred_vals.min()),
            max(ax.get_xlim()[1], ax.get_ylim()[1], true_vals.max(), pred_vals.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal y=x')
        
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"{target_name}\nR2={r2:.3f}, RMSE={rmse:.3f}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for metrics summary

    # Add a summary of metrics at the bottom if multiple targets
    if n_targets > 1:
        summary_text = "Overall Metrics: " + " | ".join(metrics_summary)
        fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"True vs. Predicted plot saved to {save_path}")

def plot_cv_error_boxplot(
    model_name: str,
    fold_predictions: List[Dict[str, np.ndarray]],
    target_names: List[str],
    save_dir: str
):
    """
    Generate and save a box plot of cross-validation errors and save the data.

    Args:
        model_name (str): Name of the model.
        fold_predictions (List[Dict[str, np.ndarray]]):
            A list of dictionaries, one for each fold. Each dictionary should
            contain 'true' and 'pred' numpy arrays for the validation set.
        target_names (List[str]): Names of the target properties.
        save_dir (str): Directory to save the plot and data CSV.
    """
    error_data = []
    for i, fold in enumerate(fold_predictions):
        y_true_fold = fold['true']
        y_pred_fold = fold['pred']

        if y_true_fold.ndim == 1:
            y_true_fold = y_true_fold.reshape(-1, 1)
        if y_pred_fold.ndim == 1:
            y_pred_fold = y_pred_fold.reshape(-1, 1)

        errors = y_true_fold - y_pred_fold

        for j, target in enumerate(target_names):
            for error_val in errors[:, j]:
                error_data.append({
                    'target': target,
                    'error': error_val,
                    'fold': i + 1
                })

    error_df = pd.DataFrame(error_data)

    # Save error data to CSV
    os.makedirs(save_dir, exist_ok=True)
    error_data_path = os.path.join(save_dir, f'{model_name}_cv_prediction_errors.csv')
    error_df.to_csv(error_data_path, index=False)
    print(f"CV prediction error data saved to: {error_data_path}")

    # Set up the plot
    plt.figure(figsize=(5 * len(target_names), 7))
    # Use hue to avoid FutureWarning, while achieving the same visual result
    sns.boxplot(x='target', y='error', data=error_df, hue='target', palette='viridis', legend=False)

    # Add a reference line at y=0
    plt.axhline(0, ls='--', color='red', zorder=1)

    plt.title(f'{model_name} Cross-Validation Prediction Error Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Target Property', fontsize=15, fontweight='bold')
    plt.ylabel('Prediction Error (True - Predicted)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, f'{model_name}_cv_error_boxplot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"CV error box plot saved to: {plot_path}")

def plot_loss_r2_curves(
    history: Dict[str, list],
    save_path: str,
    title: str = "Training and Validation Loss & R2 Curves"
):
    """
    在一个图中绘制训练损失、验证损失、训练R2和验证R2。
    Plot training/validation loss and R2 curves in a single figure.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_r2', 'val_r2'.
        save_path (str): Path to save the plot.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss curves (left y-axis)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red', linestyle='-')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Create a second y-axis for R2
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['train_r2'], label='Train R2', color='blue', linestyle='--')
    ax2.plot(epochs, history['val_r2'], label='Val R2', color='red', linestyle='--')
    ax2.set_ylabel('R2 Score')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage (can be removed or commented out):
if __name__ == '__main__':
    # Create dummy data for demonstration
    num_samples = 100
    targets = ['Yield Strength (MPa)', 'Elongation (%)']
    y_true_example = np.random.rand(num_samples, len(targets)) * 100
    y_pred_example = y_true_example * (1 + np.random.randn(num_samples, len(targets)) * 0.1) # Add some noise

    # Test case 1: Multiple targets
    plot_true_vs_pred_scatter(
        y_true=y_true_example, 
        y_pred=y_pred_example, 
        target_names=targets, 
        save_path='output/plots/example_true_vs_pred_multi.png',
        title='Example: True vs. Predicted (Multi-Target)'
    )
    print("Multi-target example plot generated.")

    # Test case 2: Single target
    y_true_single = np.random.rand(num_samples) * 50
    y_pred_single = y_true_single * (1 + np.random.randn(num_samples) * 0.15)
    plot_true_vs_pred_scatter(
        y_true=y_true_single,
        y_pred=y_pred_single,
        target_names=['Hardness (HV)'],
        save_path='output/plots/example_true_vs_pred_single.png',
        title='Example: True vs. Predicted (Single Target)'
    )
    print("Single-target example plot generated.")

    # Test case 3: Single target with 2D array input (should still work)
    y_true_single_2d = y_true_single.reshape(-1, 1)
    y_pred_single_2d = y_pred_single.reshape(-1, 1)
    plot_true_vs_pred_scatter(
        y_true=y_true_single_2d,
        y_pred=y_pred_single_2d,
        target_names=['Fracture Toughness (MPa m^0.5)'],
        save_path='output/plots/example_true_vs_pred_single_2d.png',
        title='Example: True vs. Predicted (Single Target, 2D input)'
    )
    print("Single-target 2D input example plot generated.") 