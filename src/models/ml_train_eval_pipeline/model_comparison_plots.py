"""
ML Model Comparison Plotting Module

Provides visualization functions for model comparison, including MAE, R², RMSE comparison charts.
Focus on test set results only with English-only labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Set font for better compatibility
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ModelComparisonPlotter:
    """
    Model Comparison Plotter - Test Set Results Only
    """

    def __init__(self, output_dir: str):
        """
        Initialize plotter

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
    def create_comprehensive_plots(self, df: pd.DataFrame, targets: List[str]):
        """
        Create comprehensive comparison plots for test set results only

        Args:
            df: DataFrame containing model comparison results
            targets: List of target variable names
        """
        # Create comprehensive comparison plot - all metrics in one figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ML Model Performance Comparison (Test Set Results)',
                    fontsize=16, fontweight='bold')

        # Prepare data
        models = df['Model'].tolist()

        # 1. R² Score comparison (test set only)
        ax1 = axes[0, 0]
        self._plot_metric(ax1, df, models, targets, 'R2', 'R² Score (Higher is Better)', higher_better=True)

        # 2. RMSE comparison (test set only)
        ax2 = axes[0, 1]
        self._plot_metric(ax2, df, models, targets, 'RMSE', 'RMSE (Lower is Better)', higher_better=False)

        # 3. MAE comparison (test set only)
        ax3 = axes[1, 0]
        self._plot_metric(ax3, df, models, targets, 'MAE', 'MAE (Lower is Better)', higher_better=False)

        # 4. Model ranking summary
        ax4 = axes[1, 1]
        self._create_ranking_plot(ax4, df, targets, models)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "comprehensive_model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Comprehensive comparison plot saved to: {plot_path}")

        return plot_path
    
    def _plot_metric(self, ax, df: pd.DataFrame, models: List[str], targets: List[str],
                    metric: str, title: str, higher_better: bool = True):
        """
        Plot single metric comparison (test set results only)
        """
        data = {}
        for target in targets:
            # Only use test set results (exclude CV results)
            col_name = f'{target}_{metric}'
            if col_name in df.columns:
                values = []
                for _, row in df.iterrows():
                    val = row[col_name]
                    if val != 'N/A' and pd.notna(val):
                        values.append(float(val))
                    else:
                        values.append(0 if higher_better else float('inf'))
                data[target] = values

        if data:
            x = np.arange(len(models))
            width = 0.35 if len(data) > 1 else 0.6

            for i, (target, values) in enumerate(data.items()):
                # Filter infinite values
                if not higher_better:
                    values = [v if v != float('inf') else 0 for v in values]

                offset = (i - len(data)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=target,
                             color=self.colors[i % len(self.colors)], alpha=0.8)

                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        format_str = '{:.3f}' if higher_better else '{:.2f}'
                        ax.text(bar.get_x() + bar.get_width()/2,
                               bar.get_height() + max(values)*0.01,
                               format_str.format(val),
                               ha='center', va='bottom', fontsize=9)

            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xlabel('Models')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if higher_better and metric == 'R2':
                ax.set_ylim(0, 1.1)
    
    def _create_ranking_plot(self, ax, df: pd.DataFrame, targets: List[str], models: List[str]):
        """
        Create model ranking plot based on test set results
        """
        # Calculate comprehensive score for each model
        model_scores = {}

        for model in models:
            model_row = df[df['Model'] == model].iloc[0]
            scores = []

            for target in targets:
                # R² score (higher is better, weight 0.5)
                r2_col = f'{target}_R2'
                if r2_col in df.columns and model_row[r2_col] != 'N/A':
                    r2_score = float(model_row[r2_col])
                    scores.append(r2_score * 0.5)

                # RMSE (lower is better, convert to 0-1 score, weight 0.25)
                rmse_col = f'{target}_RMSE'
                if rmse_col in df.columns and model_row[rmse_col] != 'N/A':
                    rmse_values = []
                    for _, row in df.iterrows():
                        if row[rmse_col] != 'N/A':
                            rmse_values.append(float(row[rmse_col]))
                    if rmse_values:
                        max_rmse = max(rmse_values)
                        min_rmse = min(rmse_values)
                        if max_rmse > min_rmse:
                            rmse_score = 1 - (float(model_row[rmse_col]) - min_rmse) / (max_rmse - min_rmse)
                            scores.append(rmse_score * 0.25)

                # MAE (lower is better, convert to 0-1 score, weight 0.25)
                mae_col = f'{target}_MAE'
                if mae_col in df.columns and model_row[mae_col] != 'N/A':
                    mae_values = []
                    for _, row in df.iterrows():
                        if row[mae_col] != 'N/A':
                            mae_values.append(float(row[mae_col]))
                    if mae_values:
                        max_mae = max(mae_values)
                        min_mae = min(mae_values)
                        if max_mae > min_mae:
                            mae_score = 1 - (float(model_row[mae_col]) - min_mae) / (max_mae - min_mae)
                            scores.append(mae_score * 0.25)

            model_scores[model] = np.mean(scores) if scores else 0

        # Sort and plot
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        models_sorted = [item[0] for item in sorted_models]
        scores_sorted = [item[1] for item in sorted_models]

        bars = ax.barh(models_sorted, scores_sorted, color=self.colors[:len(models_sorted)])

        # Add value labels
        for bar, score in zip(bars, scores_sorted):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontsize=10)

        ax.set_title('Overall Model Ranking (Test Set)', fontweight='bold')
        ax.set_xlabel('Overall Score')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Add ranking annotations
        for i, (model, score) in enumerate(sorted_models):
            rank_text = f"#{i+1}"
            ax.text(0.02, i, rank_text, ha='left', va='center',
                   fontweight='bold', fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    def create_individual_metric_plots(self, df: pd.DataFrame, targets: List[str]):
        """
        Create individual metric comparison plots (test set results only)

        Args:
            df: DataFrame containing model comparison results
            targets: List of target variable names

        Returns:
            List[str]: List of generated plot file paths
        """
        plot_paths = []

        # Create separate plots for each metric (test set only)
        for metric in ['R2', 'RMSE', 'MAE']:
            columns = [col for col in df.columns if f'_{metric}' in col and 'CV' not in col and 'std' not in col]
            if columns:
                ylabel = f'{metric} Score' if metric == 'R2' else metric
                filename = f'{metric.lower()}_comparison_test.png'
                higher_better = metric == 'R2'

                plot_path = self._create_metric_plot(df, columns, ylabel, filename, higher_better)
                if plot_path:
                    plot_paths.append(plot_path)

        return plot_paths
    
    def _create_metric_plot(self, df: pd.DataFrame, columns: List[str], ylabel: str,
                           filename: str, higher_better: bool = True):
        """
        Create comparison plot for a single metric (test set results only)
        """
        if not columns:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        models = df['Model'].tolist()
        x = np.arange(len(models))
        width = 0.8 / len(columns) if len(columns) > 1 else 0.6

        for i, col in enumerate(columns):
            values = []
            for _, row in df.iterrows():
                val = row[col]
                if val != 'N/A' and pd.notna(val):
                    values.append(float(val))
                else:
                    values.append(0 if higher_better else float('inf'))

            # Filter infinite values
            if not higher_better:
                values = [v if v != float('inf') else 0 for v in values]

            offset = (i - len(columns)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width,
                         label=col.replace('_R2', '').replace('_RMSE', '').replace('_MAE', ''),
                         color=self.colors[i % len(self.colors)], alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    format_str = '{:.3f}' if higher_better else '{:.2f}'
                    ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + max(values)*0.01,
                           format_str.format(val),
                           ha='center', va='bottom', fontsize=10)

        ax.set_title(f'{ylabel} Model Comparison (Test Set)',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Models')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if higher_better and 'R2' in ylabel:
            ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表] {ylabel} comparison plot saved to: {plot_path}")

        return plot_path


def create_model_comparison_plots(df: pd.DataFrame, output_dir: str, targets: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function: Create model comparison plots (test set results only)

    Args:
        df: DataFrame containing model comparison results
        output_dir: Output directory path
        targets: List of target variable names, auto-extracted from DataFrame if None

    Returns:
        Dict[str, Any]: Dictionary containing generated plot paths and statistics
    """
    # Auto-extract target variable names (test set results only)
    if targets is None:
        targets = []
        for col in df.columns:
            if '_R2' in col and 'CV' not in col and 'std' not in col:
                target = col.replace('_R2', '')
                if target not in targets:
                    targets.append(target)

    if not targets:
        print("⚠️ No target variables found, cannot generate comparison plots")
        return {}

    # Create plotter
    plotter = ModelComparisonPlotter(output_dir)

    # Generate plots
    results = {
        'targets': targets,
        'comprehensive_plot': None,
        'individual_plots': [],
        'models_count': len(df),
        'targets_count': len(targets)
    }

    try:
        # Create comprehensive comparison plot
        comprehensive_path = plotter.create_comprehensive_plots(df, targets)
        results['comprehensive_plot'] = comprehensive_path

        # Create individual metric plots
        individual_paths = plotter.create_individual_metric_plots(df, targets)
        results['individual_plots'] = individual_paths

        print(f"🎉 Successfully generated {len(individual_paths) + 1} comparison plots")

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        results['error'] = str(e)

    return results
