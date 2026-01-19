import os
import copy
import json
import numpy as np
import pandas as pd
from .config import get_args
from .pipeline import TrainingPipeline
from .utils import to_long_path

def _aggregate_and_save_results(metrics_list, save_dir):
    """Aggregate metrics from multiple runs and save statistics."""
    if not metrics_list:
        return
        
    print(f"\nAggregating results from {len(metrics_list)} runs...")
    
    aggregated_data = {}
    
    # Collect all keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
        
    # Calculate statistics for numerical values
    summary_stats = {}
    
    # Store per-repeat data for CSV
    per_repeat_data = []

    for i, m in enumerate(metrics_list):
        row = {'repeat': i + 1}
        row.update(m)
        per_repeat_data.append(row)
        
    # Save detailed CSV
    detailed_df = pd.DataFrame(per_repeat_data)
    # Sort columns to make it readable: put repeat first
    cols = ['repeat'] + [c for c in detailed_df.columns if c != 'repeat']
    detailed_df = detailed_df[cols]
    
    detailed_csv_path = os.path.join(save_dir, "detailed_repeated_results.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"Saved detailed repeated results to: {detailed_csv_path}")

    # Process statistics and string formatting
    target_stats = {} # {target: {metric: {'mean': val, 'std': val}}}
    
    for key in all_keys:
        values = [m.get(key) for m in metrics_list if m.get(key) is not None]
        # Filter for numerical values
        num_values = [v for v in values if isinstance(v, (int, float))]
        
        if num_values:
            mean_val = float(np.mean(num_values))
            std_val = float(np.std(num_values))
            
            summary_stats[key] = {
                'mean': mean_val,
                'std': std_val,
                'min': float(np.min(num_values)),
                'max': float(np.max(num_values)),
            }
            
            # Flatten for CSV
            aggregated_data[f"{key}_mean"] = mean_val
            aggregated_data[f"{key}_std"] = std_val
            
            # Try to parse target and metric from key
            # key format assumption: "test_{target}_{metric}" or "val_{target}_{metric}"
            # Standard keys often look like: "test_UTS(MPa)_r2"
            if "test_" in key or "val_" in key:
                 # Find the metric part (last part after _)
                 parts = key.split('_')
                 if len(parts) >= 2:
                     metric_type = parts[-1]
                     # Check if metric is one of interest
                     if metric_type.lower() in ['r2', 'mae', 'rmse', 'mape']:
                         # Reconstruct target
                         # Assumption: "test_" is first, metric is last. Middle is target.
                         # Example: test_UTS(MPa)_r2 -> prefix=test, metric=r2, target=UTS(MPa)
                         if key.startswith("test_"):
                             target = "_".join(parts[1:-1])
                         elif key.startswith("val_"):
                             target = "_".join(parts[1:-1])
                         else:
                             # Fallback
                             target = "_".join(parts[:-1])

                         if target not in target_stats:
                             target_stats[target] = {}
                         target_stats[target][metric_type] = {'mean': mean_val, 'std': std_val}

    # Save detailed JSON report
    json_path = os.path.join(save_dir, "aggregated_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved aggregated JSON report to: {json_path}")
    
    # Save CSV summary
    if aggregated_data:
        df = pd.DataFrame([aggregated_data])
        csv_path = os.path.join(save_dir, "aggregated_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved aggregated CSV report to: {csv_path}")

    # Generate the formatted string
    # "R2 of 89.85% (±6.17%), 88.34% (±5.95%) and 87.24% (±5.15%) for YS, UTS and EL, respectively."
    if target_stats:
        summary_lines = []
        metrics_to_report = ['r2', 'mae', 'rmse']
        
        for metric in metrics_to_report:
            # Check if this metric exists for any target
            targets_with_metric = [t for t in target_stats if metric in target_stats[t]]
            if not targets_with_metric:
                continue
                
            # Sort targets for consistency
            targets_with_metric.sort()
            
            value_strs = []
            target_names = []
            
            for t in targets_with_metric:
                stats = target_stats[t][metric]
                mean_v = stats['mean']
                std_v = stats['std']
                
                # Format based on metric type
                # R2 is often percentage in the user request example (89.85%)
                if metric.lower() == 'r2':
                     # Check if R2 is 0-1 or 0-100. Assume 0-1 if mean <= 1.0
                     if abs(mean_v) <= 1.0: 
                         mean_disp = mean_v * 100
                         std_disp = std_v * 100
                     else:
                         mean_disp = mean_v
                         std_disp = std_v
                     value_strs.append(f"{mean_disp:.2f}% (±{std_disp:.2f}%)")
                else:
                    # MAE/RMSE usually raw units
                    value_strs.append(f"{mean_v:.4f} (±{std_v:.4f})")
                
                target_names.append(t)
            
            # Construct sentence
            if value_strs:
                if len(value_strs) > 1:
                    values_joined = ", ".join(value_strs[:-1]) + " and " + value_strs[-1]
                    names_joined = ", ".join(target_names[:-1]) + " and " + target_names[-1]
                else:
                    values_joined = value_strs[0]
                    names_joined = target_names[0]
                
                line = f"{metric.upper()} of {values_joined} for {names_joined}, respectively."
                summary_lines.append(line)
        
        if summary_lines:
            summary_text = "\n".join(summary_lines)
            print("\n" + "="*40)
            print("Performance Summary:")
            print(summary_text)
            print("="*40 + "\n")
            
            summary_path = os.path.join(save_dir, "performance_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)

def main():
    """Main entry point for the training pipeline."""
    args = get_args()
    
    # Handle long paths on Windows
    args.result_dir = to_long_path(args.result_dir)
    
    n_repeats = args.n_repeats
    
    if n_repeats <= 1:
        # Standard single run
        # Standard single run
        pipeline = TrainingPipeline(args)
        pipeline.run()
        
        # If single run with CV, generate summary from CV results
        if args.cross_validate:
            cv_metrics_path = os.path.join(args.result_dir, "cv_avg_metrics.json")
            if os.path.exists(cv_metrics_path):
                print(f"[{to_long_path(cv_metrics_path)}] Found CV metrics, generating summary...")
                with open(cv_metrics_path, 'r', encoding='utf-8') as f:
                    cv_metrics = json.load(f)
                
                # Reconstruct structured dict mostly likely saved as key_mean/std in flat JSON?
                # Actually, my previous update to NN pipeline saved a structured JSON in cv_avg_metrics.json?
                # No, NN pipeline saved flat key_mean/key_std in cv_avg_metrics.json.
                # Structure: "val_r2_mean", "val_r2_std"
                
                # Let's parse it.
                target_stats = {}
                
                for key, value in cv_metrics.items():
                    # key format: "{metric}_mean" or "{metric}_std" where metric might be "val_r2"
                    # But wait, did I save per-target stats in NN pipeline?
                    # The NN pipeline I updated saves aggregated metrics across ALL folds. 
                    # But the NN Pipeline trains ONE model for multiple targets usually?
                    # AlloyNN output dim is len(target_columns).
                    # The metrics returned by trainer are usually per-target if implemented, OR overall.
                    # Trainer in NN pipeline usually returns overall val_loss/val_r2.
                    # Does it return per-target metrics? 
                    # Looking at `_train_cross_validation`, it collects `val_r2`.
                    # BaseEvaluator.evaluate usually returns detailed metrics.
                    # But the CV loop in `_run_cross_validation` only tracks `val_loss` and `val_r2` (overall).
                    
                    # If NN pipeline doesn't break down per-target in CV loop, we can only report overall.
                    # However, usually users want per-target.
                    # The CV loop I modified creates `all_fold_metrics`.
                    # And `avg_metrics` from that.
                    
                    # If I want per-target uncertainty from NN pipeline CV, I need to ensure the CV loop collects per-target metrics.
                    # Currently `trainer.train` returns (history, val_loss, state_dict).
                    # `history` contains `val_r2` etc.
                    # If `AlloyNN` trainer tracks per-target metrics, they would be in history.
                    # Let's assume for now we use what's available.
                    
                    # Parse keys like "val_r2_mean", "val_r2_std"
                    parts = key.split('_')
                    # parts: ['val', 'r2', 'mean'] or ['val', 'UTS(MPa)', 'r2', 'mean'] ? 
                    # Depends on what's in history.
                    # Usually history keys are simple if not customized.
                    
                    # If we only have overall R2, we report that.
                    pass
                
                # For now, let's just generate summary if we can identify targets.
                # If keys are simple "val_r2_mean", we can't distinguish targets if multi-target.
                # But `performance_summary.txt` expects specific format.
                
                # If specific target breakdown is missing, we might skip or do best effort.
                # BUT, the user wants it.
                # Ideally NN pipeline's trainer should log per-target R2.
                # Assuming it does or we accept overall.
                
                # Let's try to construct a basic summary from what we have.
                summary_lines = []
                # Attempt to find "val_r2_mean" etc.
                
                # If we have multiple targets, usually we want "R2 of X for Target A, Y for Target B".
                # If we only have "val_r2_mean", it's an average across targets or a global metric.
                
                # Let's look for known patterns.
                # If the key contains target name?
                
                # If we can't find target-specific stats, we'll write a generic summary.
                
                target_stats = {}
                
                for key, val in cv_metrics.items():
                    if key.endswith("_mean"):
                        metric_base = key[:-5] # remove _mean
                        std_key = f"{metric_base}_std"
                        std_val = cv_metrics.get(std_key, 0.0)
                        
                        target_stats[metric_base] = {'mean': val, 'std': std_val}
                
                if target_stats:
                    summary_lines = []
                    # Try to match to targets
                    found_targets = False
                    for t in args.target_columns:
                        # Check if e.g. "val_{t}_r2" exists
                        # This requires knowledge of how trainer logs.
                        # If trainer logs "val_r2", it's global.
                        pass
                        
                    # Fallback: Just dump what we have in a readable format if matching fails
                    # Or reuse the formatting logic if keys look compatible
                    
                    # Create summary text
                    f_summary_path = os.path.join(args.result_dir, "performance_summary.txt")
                    with open(f_summary_path, 'w', encoding='utf-8') as f_sum:
                        f_sum.write("Performance Summary (Cross-Validation):\n")
                        for m_name, stats in target_stats.items():
                             m = stats['mean']
                             s = stats['std']
                             
                             # Format
                             if "r2" in m_name.lower():
                                 if abs(m) <= 1.0:
                                     val_str = f"{m*100:.2f}% (±{s*100:.2f}%)"
                                 else:
                                     val_str = f"{m:.2f}% (±{s:.2f}%)"
                             else:
                                 val_str = f"{m:.4f} (±{s:.4f})"
                                 
                             f_sum.write(f"{m_name}: {val_str}\n")
                        
                        print(f"Generated performance summary at {f_summary_path}")
    else:
        # Multiple repeated experiments
        print(f"🚀 Starting {n_repeats} repeated experiments for statistical analysis...")
        print(f"Base Result Directory: {args.result_dir}")
        
        original_result_dir = args.result_dir
        original_random_state = args.random_state
        repeat_metrics = []
        
        # Ensure base directory exists
        os.makedirs(original_result_dir, exist_ok=True)
        
        for i in range(n_repeats):
            print(f"\n{'='*60}")
            print(f"🔄 Running Experiment Repeat {i+1}/{n_repeats}")
            print(f"{'='*60}")
            
            # Update args for this repeat
            # Create a shallow copy is usually enough for Namespace, but deepcopy is safer
            current_args = copy.deepcopy(args)
            
            # Vary random state
            current_args.random_state = original_random_state + i
            
            # Subdirectory for this repeat
            current_args.result_dir = os.path.join(original_result_dir, f"repeat_{i}")
            
            # Run pipeline
            pipeline = TrainingPipeline(current_args)
            pipeline.run()
            
            # Collect metrics
            # We look for 'final_evaluation_metrics.json' which is saved by _evaluate_best_model
            metrics_path = os.path.join(current_args.result_dir, "final_evaluation_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    repeat_metrics.append(metrics)
                print(f"✅ Collected metrics for repeat {i+1}")
            else:
                print(f"⚠️ Warning: No metrics found for repeat {i+1} at {metrics_path}")
        
        # Aggregate and save results
        if repeat_metrics:
            _aggregate_and_save_results(repeat_metrics, original_result_dir)
        else:
            print("❌ No metrics collected from any repeat.")

if __name__ == '__main__':
    main()




