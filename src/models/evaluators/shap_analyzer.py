import os
import traceback
import torch
import numpy as np
try:
    import shap
except ImportError as e:
    print(f"[SHAP Analyzer] Warning: Could not import 'shap'. SHAP analysis will be unavailable. Error: {e}")
    shap = None
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Optional, Dict
from tqdm import tqdm
from .base_evaluator import BaseEvaluator

class NeuralNetworkShapAnalyzer(BaseEvaluator):
    """
    SHAP analyzer specifically designed for PyTorch neural network models.
    Uses DeepExplainer for efficient and accurate SHAP value computation.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 feature_names: List[str],
                 target_names: List[str],
                 result_dir: str,
                 device: torch.device):
        """
        Initializes the SHAP analyzer.

        Args:
            model (nn.Module): The trained PyTorch model to analyze.
            feature_names (List[str]): List of feature names.
            target_names (List[str]): List of target variable names.
            result_dir (str): Directory to save analysis results (plots and data).
            device (torch.device): The device the model is on (e.g., 'cuda' or 'cpu').
        """
        super().__init__(result_dir, "SHAP_Analysis", target_names)
        self.model = model.to(device)
        self.model.eval()
        self.feature_names = feature_names
        self.device = device
        
        print(f"[SHAP Analyzer] Initialized for PyTorch model analysis.")
        print(f"[SHAP Analyzer] Results will be saved in: {self.result_dir}")
        print(f"[SHAP Analyzer] Target names: {self.target_names}")
        print(f"[SHAP Analyzer] Using device: {self.device}")

    def _sanitize_filename(self, name: str) -> str:
        """Removes characters that are invalid for filenames."""
        return name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'percent')

    def analyze_feature_importance(self,
                                   X: np.ndarray,
                                   background_data: Optional[np.ndarray] = None,
                                   max_display: int = 20) -> Dict[str, np.ndarray]:
        """
        Performs SHAP analysis using KernelExplainer to determine feature importance.

        Args:
            X (np.ndarray): The data to explain (e.g., test set features).
            background_data (Optional[np.ndarray]): Background data for the explainer.
            max_display (int): Maximum number of features to display in summary plots.

        Returns:
            Dict[str, np.ndarray]: A dictionary of mean absolute SHAP values for each feature.
        """
        if shap is None:
            print("[SHAP Analyzer] SHAP support is unavailable (import failed). Skipping analysis.")
            return {}

        if background_data is None:
            background_data = X
            print(f"[SHAP Analyzer] No background data provided. Using input data X as the source.")

        # --- Data Validation ---
        print(f"[SHAP Analyzer] Analysis data shape: {X.shape}, type: {X.dtype}")
        print(f"[SHAP Analyzer] Background data shape: {background_data.shape}, type: {background_data.dtype}")
        if X.shape[1] != background_data.shape[1]:
            print(f"[SHAP Analyzer] FATAL: Mismatch in feature dimensions between analysis data ({X.shape[1]}) and background data ({background_data.shape[1]}).")
            return {}

        def model_wrapper(x_numpy: np.ndarray) -> np.ndarray:
            """Prediction function that handles numpy to tensor conversion for multi-output model."""
            x_tensor = torch.from_numpy(x_numpy).float().to(self.device)
            with torch.no_grad():
                return self.model(x_tensor).cpu().numpy()

        feature_importance_results = {}
            
        try:
            print(f"[SHAP Analyzer] Creating KernelExplainer for the multi-output model...")
            explainer = shap.KernelExplainer(model_wrapper, background_data)
            
            print(f"[SHAP Analyzer] Calculating SHAP values for all targets...")
            # For multi-output models, shap_values is a list of arrays, one for each output.
            # shap_values[i] corresponds to the SHAP values for the i-th output.
            shap_values = explainer.shap_values(X, l1_reg="auto") 
            print(f"[SHAP Analyzer] SHAP values calculated successfully.")

            # Iterate through each target to generate plots
            for i, target_name in enumerate(tqdm(self.target_names, desc="[SHAP] Plotting for Targets")):
                print(f"\n--- Processing Target: {target_name} ({i+1}/{len(self.target_names)}) ---")
                
                # Get SHAP values for the current target
                shap_values_target = shap_values[i] if isinstance(shap_values, list) else shap_values[:,:,i]
                sanitized_target_name = self._sanitize_filename(target_name)

                # Configure Matplotlib for CJK characters
                try:
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
                    plt.rcParams['axes.unicode_minus'] = False
                except Exception as e:
                    print(f"[SHAP Analyzer] Warning: Could not set preferred fonts for plotting. Error: {e}")

                # --- Generate and Save Beeswarm Plot (dot) ---
                plt.style.use('seaborn-v0_8-darkgrid')
                save_path = os.path.join(self.plots_dir, f'shap_beeswarm_{sanitized_target_name}.png')
                print(f"[SHAP Analyzer] Saving Beeswarm Plot for '{target_name}' to {save_path}")
                plt.figure()
                shap.summary_plot(shap_values_target, X, feature_names=self.feature_names, show=False, plot_type='dot', max_display=max_display)
                plt.title(f'SHAP Feature Importance for {target_name}')
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

                # --- Generate and Save Bar Plot ---
                plt.style.use('seaborn-v0_8-darkgrid')
                save_path = os.path.join(self.plots_dir, f'shap_bar_{sanitized_target_name}.png')
                print(f"[SHAP Analyzer] Saving Bar Plot for '{target_name}' to {save_path}")
                plt.figure()
                shap.summary_plot(shap_values_target, X, feature_names=self.feature_names, plot_type="bar", show=False, max_display=max_display)
                plt.title(f'Mean Absolute SHAP for {target_name}')
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                # --- Generate and Save Violin Plot ---
                plt.style.use('seaborn-v0_8-darkgrid')
                save_path = os.path.join(self.plots_dir, f'shap_violin_{sanitized_target_name}.png')
                print(f"[SHAP Analyzer] Saving Violin Plot for '{target_name}' to {save_path}")
                plt.figure()
                shap.summary_plot(shap_values_target, X, feature_names=self.feature_names, plot_type="violin", show=False, max_display=max_display)
                plt.title(f'SHAP Value Distribution for {target_name}')
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

                # Store mean absolute SHAP values for returning
                feature_importance_results[target_name] = np.abs(shap_values_target).mean(axis=0)

        except Exception as e:
            print(f"[SHAP Analyzer] FATAL: Failed to calculate or plot SHAP values. Error: {e}")
            print(traceback.format_exc())

        print(f"\n[SHAP Analyzer] Analysis complete. All plots saved in: {self.plots_dir}")
        
        # Save feature importance results to a JSON file
        feature_importance_path = os.path.join(self.result_dir, "feature_importance_results.json")
        self.save_metrics(feature_importance_results, feature_importance_path) # Using save_metrics from BaseEvaluator
        
        return feature_importance_results 


class MLShapAnalyzer(BaseEvaluator):
    """
    SHAP analyzer for traditional ML models (e.g., scikit-learn, XGBoost).
    Uses TreeExplainer for tree-based models and KernelExplainer for others.
    """
    def __init__(self,
                 model: any,
                 feature_names: List[str],
                 target_names: List[str],
                 result_dir: str):
        """
        Initializes the SHAP analyzer.

        Args:
            model (any): The trained model to analyze.
            feature_names (List[str]): List of feature names.
            target_names (List[str]): List of target variable names.
            result_dir (str): Directory to save analysis results.
        """
        super().__init__(result_dir, "SHAP_Analysis_ML", target_names)
        self.model = model
        self.feature_names = feature_names
        
        print(f"[SHAP Analyzer] Initialized for ML model analysis.")
        print(f"[SHAP Analyzer] Results will be saved in: {self.result_dir}")

    def _sanitize_filename(self, name: str) -> str:
        return name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'percent')

    def analyze(self,
                X: np.ndarray,
                max_display: int = 20):
        """
        Performs SHAP analysis to determine feature importance.

        Args:
            X (np.ndarray): The data to explain (e.g., test set features).
            max_display (int): Maximum number of features to display in plots.
        """
        print(f"[SHAP Analyzer] Analyzing feature importance for ML model...")
        
        if shap is None:
            print("[SHAP Analyzer] SHAP support is unavailable (import failed). Skipping analysis.")
            return
        
        try:
            # Check if the model is CatBoost and handle it specially to avoid crashes
            model_class_name = self.model.__class__.__name__
            is_catboost = 'CatBoost' in model_class_name or 'catboost' in model_class_name.lower()
            
            # For tree-based models like XGBoost and RandomForest, use the faster TreeExplainer.
            # For other models like SVR, KernelExplainer is used.
            if hasattr(self.model, 'feature_importances_'):
                print("[SHAP Analyzer] Tree-based model detected. Using TreeExplainer.")
                # Special handling for CatBoost to avoid memory crashes
                if is_catboost:
                    print("[SHAP Analyzer] CatBoost model detected. Using safer SHAP approach.")
                    # Use a smaller background dataset for CatBoost to avoid memory issues
                    n_background = min(50, X.shape[0] // 2)
                    background_indices = np.random.choice(X.shape[0], n_background, replace=False)
                    background_data = X[background_indices]
                    try:
                        explainer = shap.TreeExplainer(self.model, background_data)
                    except Exception as catboost_err:
                        print(f"[SHAP Analyzer] TreeExplainer failed for CatBoost: {catboost_err}")
                        print("[SHAP Analyzer] Skipping SHAP analysis for CatBoost due to compatibility issues.")
                        return
                else:
                    explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
            else:
                print("[SHAP Analyzer] Non-tree model detected. Using KernelExplainer.")
                # KernelExplainer requires a background dataset, so we summarize using k-means.
                # Ensure n_clusters is not greater than n_samples
                n_clusters = min(100, X.shape[0])
                try:
                    background_data = shap.kmeans(X, n_clusters)
                    print(f"[SHAP Analyzer] Summarized input data to {background_data.data.shape[0]} samples for KernelExplainer.")
                    explainer = shap.KernelExplainer(self.model.predict, background_data)
                    shap_values = explainer.shap_values(X)
                except Exception as kmeans_err:
                    print(f"[SHAP Analyzer] Error during KernelExplainer initialization: {kmeans_err}")
                    print("[SHAP Analyzer] Skipping SHAP analysis due to compatibility issues.")
                    return

            print("[SHAP Analyzer] SHAP values calculated successfully.")
            
            # SHAP values for multi-output models can be a list of arrays or a single 3D array.
            is_multi_output = isinstance(shap_values, list) or \
                            (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3)

            for i, target_name in enumerate(self.target_names):
                print(f"\n--- Processing Target: {target_name} ---")
                
                target_shap_values = shap_values
                if isinstance(shap_values, list):
                    # Handle case where shap_values is a list of arrays (one for each output)
                    target_shap_values = shap_values[i]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # Handle case where shap_values is a 3D array (n_samples, n_features, n_outputs)
                    target_shap_values = shap_values[:, :, i]
                elif len(self.target_names) > 1:
                    print(f"[SHAP Analyzer] WARNING: Model has multiple outputs but SHAP values are 2D. "
                          f"Plotting for the first target '{self.target_names[0]}' only.")
                    if i > 0:
                        break # Stop after plotting for the first target
                
                sanitized_target_name = self._sanitize_filename(target_name)
                
                # Configure Matplotlib for CJK characters
                try:
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
                    plt.rcParams['axes.unicode_minus'] = False
                except Exception as e:
                    print(f"[SHAP Analyzer] Warning: Could not set preferred fonts for plotting. Error: {e}")

                # Bar plot
                plt.figure()
                shap.summary_plot(target_shap_values, X, feature_names=self.feature_names, plot_type="bar", show=False, max_display=max_display)
                plt.title(f'Mean Absolute SHAP for {target_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'shap_bar_{sanitized_target_name}.png'), dpi=300)
                plt.close()

                # Beeswarm plot
                plt.figure()
                shap.summary_plot(target_shap_values, X, feature_names=self.feature_names, show=False, max_display=max_display)
                plt.title(f'SHAP Feature Importance for {target_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'shap_beeswarm_{sanitized_target_name}.png'), dpi=300)
                plt.close()

            print(f"\n[SHAP Analyzer] Analysis complete. Plots saved in: {self.plots_dir}")

        except Exception as e:
            print(f"[SHAP Analyzer] FATAL: Failed to calculate or plot SHAP values. Error: {e}")
            import traceback
            print(traceback.format_exc())
