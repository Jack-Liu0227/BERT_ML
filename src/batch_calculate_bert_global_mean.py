"""
Batch script to process all BERT model (SciBERT, MatSciBERT, SteelBERT) Optuna trials across all alloys.
Calculates global mean R2 and identifies representative folds.
Results are organized by Alloy/BERT_Model/Results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
from typing import List, Dict, Tuple
import shutil
import json
import argparse

# Set plot style
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

def plot_bert_model_comparison(stats_df: pd.DataFrame, output_dir: str, alloy_name: str):
    """
    Plot BERT model comparison chart (similar to ML models comparison)
    """
    if stats_df.empty:
        return
    
    # Define model order and colors
    model_order = ['matscibert', 'scibert', 'steelbert']
    color_map = {
        'Ep(mV)': '#f4c542',  # Gold
        'UTS(MPa)': '#aacfef',  # Light Blue
        'El(%)': '#ffb3a7',  # Light Pink
        'YS(MPa)': '#c5e1a5'  # Light Green
    }
    
    properties = sorted(stats_df['Property'].unique())
    
    model_order_present = [m for m in model_order if m in stats_df['Model'].unique()]
    if not model_order_present or not properties:
        return
    
    x = np.arange(len(model_order_present))
    group_width = 0.8
    bar_width = group_width / max(len(properties), 1)
    
    plt.figure(figsize=(10, 8))
    
    for idx, prop in enumerate(properties):
        prop_data = stats_df[stats_df['Property'] == prop]
        if prop_data.empty:
            continue
        
        prop_data = prop_data.set_index('Model').reindex(model_order_present).reset_index()
        prop_data = prop_data.dropna(subset=['Global_Mean_R2'])
        if prop_data.empty:
            continue
        
        means = prop_data['Global_Mean_R2'].tolist()
        stds = prop_data['Global_Std_R2'].tolist()
        
        offset = (idx - (len(properties) - 1) / 2) * bar_width
        plt.bar(
            x + offset,
            means,
            yerr=stds,
            width=bar_width,
            capsize=4,
            color=color_map.get(prop, '#aacfef'),
            edgecolor='black',
            linewidth=1.2,
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
            label=prop
        )
    
    plt.xlabel('BERT Models', fontweight='bold', fontsize=18)
    plt.ylabel('R\u00b2', fontweight='bold', fontsize=18)
    plt.ylim(0, 1.0)
    
    # Capitalize model names for display
    display_names = [m.replace('matscibert', 'MatSciBERT').replace('scibert', 'SciBERT').replace('steelbert', 'SteelBERT') 
                    for m in model_order_present]
    plt.xticks(x, display_names, fontsize=16)
    plt.yticks(fontsize=16)
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='y', which='major', length=6, width=1.5)
    ax.tick_params(axis='y', which='minor', length=3, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend(frameon=True, fontsize=12, edgecolor='black')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"{alloy_name}_bert_models_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved BERT comparison plot: {output_path.name}")

def plot_diagonal_chart(file_path: Path, property_name: str, model_name: str, 
                       r2_value: float, output_dir: str,
                       output_prefix: str = "global_mean"):
    """
    Plot diagonal chart (Experimental vs Predicted) for train, validation, and test sets
    """
    try:
        df = pd.read_csv(file_path)
        
        actual_col = f"{property_name}_Actual"
        pred_col = f"{property_name}_Predicted"
        
        if actual_col not in df.columns or pred_col not in df.columns:
            return
        
        # Check if Dataset column exists
        has_dataset_col = 'Dataset' in df.columns
        
        plt.figure(figsize=(8, 8))
        
        # Define colors and markers for different datasets
        dataset_config = {
            'Train': {'color': '#4CAF50', 'marker': 'o', 'label': 'Training Set', 'alpha': 0.5, 's': 60},
            'Validation': {'color': '#2196F3', 'marker': 's', 'label': 'Validation Set', 'alpha': 0.6, 's': 70},
            'Test': {'color': '#FF5722', 'marker': '^', 'label': 'Test Set', 'alpha': 0.7, 's': 80}
        }
        
        # Alternative names for datasets
        dataset_aliases = {
            'Train': ['Train', 'train', 'Training', 'training'],
            'Validation': ['Validation', 'Valid', 'Val', 'validation', 'valid', 'val'],
            'Test': ['Test', 'test', 'Testing', 'testing']
        }
        
        min_val = float('inf')
        max_val = float('-inf')
        plotted_any = False
        metrics_text = []  # Store metrics for each dataset
        
        if has_dataset_col:
            # Plot each dataset separately
            for dataset_key, config in dataset_config.items():
                # Find matching rows
                matching_rows = pd.Series([False] * len(df))
                for alias in dataset_aliases[dataset_key]:
                    matching_rows |= (df['Dataset'] == alias)
                
                dataset_df = df[matching_rows].copy()
                valid_df = dataset_df[[actual_col, pred_col]].dropna()
                
                if len(valid_df) > 0:
                    plt.scatter(valid_df[actual_col], valid_df[pred_col], 
                               alpha=config['alpha'], s=config['s'], 
                               edgecolors='black', linewidth=0.8,
                               c=config['color'], marker=config['marker'],
                               label=config['label'])
                    
                    min_val = min(min_val, valid_df[actual_col].min(), valid_df[pred_col].min())
                    max_val = max(max_val, valid_df[actual_col].max(), valid_df[pred_col].max())
                    plotted_any = True
                    
                    # Calculate metrics
                    y_true = valid_df[actual_col].values
                    y_pred = valid_df[pred_col].values
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    
                    # Store metrics text
                    metrics_text.append(f"{config['label']}:\n  R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        else:
            # If no Dataset column, plot all data as one set
            valid_df = df[[actual_col, pred_col]].dropna()
            if len(valid_df) > 0:
                plt.scatter(valid_df[actual_col], valid_df[pred_col], 
                           alpha=0.6, s=80, edgecolors='black', linewidth=0.8,
                           c='#4CAF50', label='All Data')
                min_val = min(valid_df[actual_col].min(), valid_df[pred_col].min())
                max_val = max(valid_df[actual_col].max(), valid_df[pred_col].max())
                plotted_any = True
                
                # Calculate metrics
                y_true = valid_df[actual_col].values
                y_pred = valid_df[pred_col].values
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                metrics_text.append(f"All Data:\n  R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        
        if not plotted_any or min_val == float('inf'):
            plt.close()
            return
        margin = (max_val - min_val) * 0.05
        plt.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                'r--', linewidth=2.5, label='Perfect Prediction (y=x)')
        
        property_label = property_name.replace('_', ' ')
        plt.xlabel(f'Experimental {property_label}', fontsize=16, fontweight='bold')
        plt.ylabel(f'Predicted {property_label}', fontsize=16, fontweight='bold')
        plt.title(f"{model_name} - {property_label}", 
                  fontsize=18, fontweight='bold')
        
        plt.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black', fancybox=False)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axis('equal')
        plt.xlim(min_val - margin, max_val + margin)
        plt.ylim(min_val - margin, max_val + margin)
        
        # Add metrics text box in lower right corner
        if metrics_text:
            metrics_str = '\n\n'.join(metrics_text)
            plt.text(0.98, 0.02, metrics_str,
                    transform=plt.gca().transAxes,
                    fontsize=11,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
                    family='monospace')
        
        plt.tight_layout()
        
        safe_prop_name = property_name.replace('(', '').replace(')', '').replace('%', 'pct')
        output_path = Path(output_dir) / f"{output_prefix}_{safe_prop_name}_diagonal.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting {property_name}: {e}")

def process_bert_directory(model_dir: Path, alloy_name: str, alloy_subdir: str, summary_root: Path):
    """
    Process a single BERT model directory, calculate global mean, and save to summary folder.
    """
    optuna_trials_dir = model_dir / "predictions" / "optuna_trials"
    if not optuna_trials_dir.exists():
        return False

    model_type = model_dir.name # e.g., scibert, matscibert, steelbert
    
    # Create output directory: all_alloys_best_models_summary / Alloy / bert_models / ModelType
    output_dir = summary_root / alloy_name / "bert_models" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save internally for reference
    internal_output_dir = model_dir / "closest_to_global_mean_predictions"
    internal_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Processing {alloy_name}/{alloy_subdir}/{model_type}...")
    
    # 1. Collect all trial metrics from TEST SET only
    trial_data = []
    trial_folders = sorted([d for d in optuna_trials_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")], 
                          key=lambda x: int(x.name.split('_')[1]))
    
    for trial_dir in trial_folders:
        # Find all fold directories
        fold_dirs = sorted([d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
                          key=lambda x: int(x.name.split('_')[1]))
        
        for fold_dir in fold_dirs:
            try:
                fold_number = int(fold_dir.name.split('_')[1])
                prediction_file = fold_dir / "all_predictions.csv"
                
                if not prediction_file.exists():
                    continue
                
                # Read predictions and filter for TEST set only
                df_pred = pd.read_csv(prediction_file)
                if 'Dataset' not in df_pred.columns:
                    continue
                
                # Filter for test set
                df_test = df_pred[df_pred['Dataset'].isin(['Test', 'test', 'Testing', 'testing'])].copy()
                if len(df_test) == 0:
                    continue
                
                # Calculate R² for each property on TEST set
                data_point = {
                    'trial_id': trial_dir.name,
                    'fold': fold_number,
                    'file_path': str(prediction_file)
                }
                
                # Find all property columns (format: PropertyName_Actual and PropertyName_Predicted)
                actual_cols = [col for col in df_test.columns if col.endswith('_Actual')]
                
                for actual_col in actual_cols:
                    prop = actual_col.replace('_Actual', '')
                    pred_col = f"{prop}_Predicted"
                    
                    if pred_col not in df_test.columns:
                        continue
                    
                    # Calculate R² on test set
                    valid_rows = df_test[[actual_col, pred_col]].dropna()
                    if len(valid_rows) > 0:
                        test_r2 = r2_score(valid_rows[actual_col], valid_rows[pred_col])
                        data_point[prop] = test_r2
                
                if len(data_point) > 3:  # Has at least one property metric
                    trial_data.append(data_point)
                    
            except Exception as e:
                print(f"    Error processing {fold_dir}: {e}")

    if not trial_data:
        return False
        
    all_df = pd.DataFrame(trial_data)
    properties = [col for col in all_df.columns if col not in ['trial_id', 'fold', 'file_path']]
    
    print(f"    Found {len(trial_data)} total folds (Trial × Fold combinations) across {len(trial_folders)} trials")
    
    # 2. Calculate Global Mean Stats
    summary_results = []
    for prop in properties:
        target_mean = all_df[prop].mean()
        target_std = all_df[prop].std()
        
        # distance
        all_df[f'{prop}_dist'] = (all_df[prop] - target_mean).abs()
        closest_row = all_df.sort_values(by=f'{prop}_dist').iloc[0]
        
        closest_trial = closest_row['trial_id']
        closest_fold = closest_row['fold']
        closest_r2 = closest_row[prop]
        closest_file = Path(closest_row['file_path'])
        
        # Save results
        safe_prop = prop.replace('(', '').replace(')', '').replace('%', 'pct')
        
        # Destination names
        csv_name = f"global_mean_representative_{safe_prop}.csv"
        
        # Copy to internal
        shutil.copy2(closest_file, internal_output_dir / csv_name)
        # Copy to summary
        shutil.copy2(closest_file, output_dir / csv_name)
        
        # Plot to both
        plot_diagonal_chart(closest_file, prop, f"{model_type.capitalize()} (Global Mean)", closest_r2, str(internal_output_dir))
        plot_diagonal_chart(closest_file, prop, f"{model_type.capitalize()} (Global Mean)", closest_r2, str(output_dir))
        
        summary_results.append({
            'Alloy': alloy_name,
            'Model': model_type,
            'Property': prop,
            'Global_Mean_R2': target_mean,
            'Global_Std_R2': target_std,
            'Closest_Trial': closest_trial,
            'Closest_Fold': closest_fold,
            'Closest_R2': closest_r2
        })

    # Save local summary
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(internal_output_dir / "global_mean_summary.csv", index=False)
    summary_df.to_csv(output_dir / "global_mean_summary.csv", index=False)
    
    return summary_results

def main():
    parser = argparse.ArgumentParser(description="Batch process BERT model global means.")
    parser.add_argument("--base_dir", type=str, default="output/new_results_withuncertainty", help="Base directory of results")
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    summary_root = base_path / "all_alloys_best_models_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting batch BERT analysis in {base_path}...")
    
    all_summaries = []
    bert_models = ['scibert', 'matscibert', 'steelbert']
    
    # Dictionary to collect results by alloy for comparison plots
    alloy_results = {}
    
    # Traverse directory structure
    for alloy_dir in base_path.iterdir():
        if not alloy_dir.is_dir() or alloy_dir.name == "all_alloys_best_models_summary":
            continue
            
        for alloy_subdir in alloy_dir.iterdir():
            if not alloy_subdir.is_dir():
                continue
                
            for model_dir in alloy_subdir.iterdir():
                if model_dir.is_dir() and model_dir.name in bert_models:
                    results = process_bert_directory(model_dir, alloy_dir.name, alloy_subdir.name, summary_root)
                    if results:
                        all_summaries.extend(results)
                        
                        # Collect for comparison plot
                        if alloy_dir.name not in alloy_results:
                            alloy_results[alloy_dir.name] = []
                        alloy_results[alloy_dir.name].extend(results)
    
    # Generate comparison plots for each alloy
    print("\nGenerating BERT model comparison plots...")
    for alloy_name, results in alloy_results.items():
        if results:
            stats_df = pd.DataFrame(results)
            output_dir = summary_root / alloy_name / "bert_models"
            plot_bert_model_comparison(stats_df, output_dir, alloy_name)
                        
    if all_summaries:
        total_summary_df = pd.DataFrame(all_summaries)
        total_summary_path = summary_root / "ALL_BERT_MODELS_GLOBAL_MEAN_SUMMARY.csv"
        total_summary_df.to_csv(total_summary_path, index=False)
        print(f"\nCompleted! Total summary saved to {total_summary_path}")
    else:
        print("\nNo BERT trial results found.")

if __name__ == "__main__":
    main()
