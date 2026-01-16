"""
Neural Network Predictor for alloy property prediction
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from src.models.base.alloys_nn import AlloyNN
from src.models.visualization.plot_utils import plot_true_vs_pred_scatter

class NNPredictor:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize neural network predictor
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None  # Will be initialized in _load_model
        self.loaded_model_params = {} # To store params used for loading
        
    def _get_device(self, device: str) -> torch.device:
        """
        Get the appropriate device for model execution
        
        Args:
            device (str): Requested device ('cuda' or 'cpu')
            
        Returns:
            torch.device: The device to use
        """
        if device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("\nWarning: CUDA not available, using CPU instead")
            return torch.device('cpu')
    
    def _load_model(self, 
                    feature_names: List[str],
                    output_dim: int,
                    feature_type_config: Optional[Dict] = None,
                    feature_hidden_dims: Optional[Dict] = None,
                    prediction_hidden_dims: Optional[List[int]] = None,
                    dropout_rate: Optional[float] = None
                    ) -> AlloyNN:
        """
        Load the trained model.
        
        Args:
            feature_names (List[str]): List of feature names.
            output_dim (int): Output dimension of the model.
            feature_type_config (Optional[Dict]): Configuration for feature types. 
                                                  Defaults to AlloyNN's internal default if None.
            feature_hidden_dims (Optional[Dict]): Hidden dimensions for each feature type.
                                                 Defaults to AlloyNN's internal default if None.
            prediction_hidden_dims (Optional[List[int]]): Hidden dimensions for the prediction network.
                                                          Defaults to AlloyNN's internal default if None.
            dropout_rate (Optional[float]): Dropout rate. Defaults to AlloyNN's internal default if None.
            
        Returns:
            AlloyNN: The loaded model.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"\nLoading model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            model_state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # Prepare model arguments, using provided ones or letting AlloyNN use its defaults
            model_kwargs = {
                'column_names': feature_names,
                'output_dim': output_dim,
            }
            if feature_type_config is not None:
                model_kwargs['feature_type_config'] = feature_type_config
            if feature_hidden_dims is not None:
                model_kwargs['feature_hidden_dims'] = feature_hidden_dims
            if prediction_hidden_dims is not None:
                model_kwargs['hidden_dims'] = prediction_hidden_dims
            if dropout_rate is not None:
                model_kwargs['dropout_rate'] = dropout_rate

            model = AlloyNN(**model_kwargs)
            
            try:
                model.load_state_dict(model_state)
            except Exception as e:
                print(f"\nWarning: Failed to load model state dict directly: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(model_state, strict=False)
            
            model.to(self.device)
            model.eval()
            
            # Store parameters used to load this model instance
            self.loaded_model_params = {
                'feature_names': feature_names,
                'output_dim': output_dim,
                'feature_type_config': feature_type_config, # Store what was actually used (None or specific dict)
                'feature_hidden_dims': feature_hidden_dims, # Store what was actually used
                'prediction_hidden_dims': prediction_hidden_dims,
                'dropout_rate': dropout_rate
            }
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, 
                X: np.ndarray, 
                feature_names: List[str],
                output_dim: int, # Required: must match the trained model's output
                feature_type_config: Optional[Dict] = None,
                feature_hidden_dims: Optional[Dict] = None,
                prediction_hidden_dims: Optional[List[int]] = None,
                dropout_rate: Optional[float] = None
                ) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X (np.ndarray): Feature data for prediction.
            feature_names (List[str]): List of feature names.
            output_dim (int): Expected output dimension of the model.
            feature_type_config (Optional[Dict]): Model's feature type configuration.
            feature_hidden_dims (Optional[Dict]): Model's feature hidden dimensions.
            prediction_hidden_dims (Optional[List[int]]): Model's prediction network hidden dimensions.
            dropout_rate (Optional[float]): Model's dropout rate.
            
        Returns:
            np.ndarray: Model predictions.
        """
        # Check if model needs to be reloaded
        # This simple check reloads if feature names or key architectural params change.
        # A more robust system would compare all relevant stored params.
        reload_needed = False
        if self.model is None:
            reload_needed = True
        elif self.loaded_model_params.get('feature_names') != feature_names or \
             self.loaded_model_params.get('output_dim') != output_dim or \
             self.loaded_model_params.get('feature_type_config') != feature_type_config or \
             self.loaded_model_params.get('feature_hidden_dims') != feature_hidden_dims or \
             self.loaded_model_params.get('prediction_hidden_dims') != prediction_hidden_dims or \
             self.loaded_model_params.get('dropout_rate') != dropout_rate:
            reload_needed = True

        if reload_needed:
            print("Configuration changed or model not loaded. Reloading model.")
            self.model = self._load_model(
                feature_names=feature_names,
                output_dim=output_dim,
                feature_type_config=feature_type_config,
                feature_hidden_dims=feature_hidden_dims,
                prediction_hidden_dims=prediction_hidden_dims,
                dropout_rate=dropout_rate
            )
        
        # 转换为tensor并移动到正确的设备
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # 进行预测
        with torch.no_grad():
            predictions = self.model.predict(X_tensor)
        
        # 直接返回预测结果（已经是numpy数组）
        return predictions

    def predict_dataframe(self,
                          data_df: pd.DataFrame,
                          feature_names: List[str],
                          output_dim: int, # Required for model loading/prediction
                          target_names: Optional[List[str]] = None,
                          feature_type_config: Optional[Dict] = None,
                          feature_hidden_dims: Optional[Dict] = None,
                          prediction_hidden_dims: Optional[List[int]] = None,
                          dropout_rate: Optional[float] = None,
                          plot_save_path: Optional[str] = None,
                          plot_title: Optional[str] = "True vs. Predicted Values"
                          ) -> Dict[str, Any]:
        """
        Makes predictions on a DataFrame. If target columns are provided and found,
        it also extracts true values and generates a comparison plot.

        Args:
            data_df (pd.DataFrame): Input DataFrame with feature columns and optionally target columns.
            feature_names (List[str]): List of column names to use as features.
            output_dim (int): The output dimension of the neural network model.
            target_names (Optional[List[str]]): List of target column names.
            feature_type_config (Optional[Dict]): Model's feature type configuration.
            feature_hidden_dims (Optional[Dict]): Model's feature hidden dimensions.
            prediction_hidden_dims (Optional[List[int]]): Model's prediction network hidden dims.
            dropout_rate (Optional[float]): Model's dropout rate.
            plot_save_path (Optional[str]): Path to save the comparison plot if targets are available.
            plot_title (Optional[str]): Title for the comparison plot.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'predictions': pd.DataFrame of predicted values.
                - 'y_true': pd.DataFrame of true values (if targets available).
                - 'plot_path': Path to the saved plot (if targets available and plot_save_path provided).
        """
        # Validate feature_names
        missing_features = [name for name in feature_names if name not in data_df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_features}")

        X_predict_np = data_df[feature_names].values.astype(np.float32)

        # Get predictions using the core predict method
        predictions_np = self.predict(
            X=X_predict_np,
            feature_names=feature_names,
            output_dim=output_dim,
            feature_type_config=feature_type_config,
            feature_hidden_dims=feature_hidden_dims,
            prediction_hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # Determine column names for the prediction DataFrame
        # If target_names are provided and their count matches output_dim, use them.
        # Otherwise, use generic names like 'pred_0', 'pred_1', etc.
        if target_names and len(target_names) == output_dim:
            pred_column_names = target_names
        else:
            pred_column_names = [f"prediction_{i}" for i in range(output_dim)]
            if target_names: # Mismatch in count
                print(f"Warning: Number of target_names ({len(target_names)}) does not match model output_dim ({output_dim}). Using generic prediction column names.")


        predictions_df = pd.DataFrame(predictions_np, columns=pred_column_names, index=data_df.index)
        result = {'predictions': predictions_df}

        # Handle targets if provided
        y_true_df = None
        actual_target_names_found = []

        if target_names:
            valid_targets_in_df = [name for name in target_names if name in data_df.columns]
            if len(valid_targets_in_df) == len(target_names): # All specified targets found
                actual_target_names_found = valid_targets_in_df
                y_true_np = data_df[actual_target_names_found].values.astype(np.float32)
                y_true_df = pd.DataFrame(y_true_np, columns=actual_target_names_found, index=data_df.index)
                result['y_true'] = y_true_df
                
                if y_true_np.shape[1] != output_dim:
                    print(f"Warning: The number of found target columns ({y_true_np.shape[1]}) in data_df "
                          f"does not match the model's output_dim ({output_dim}). "
                          "Plotting might be misleading if target_names for predictions are not aligned.")

                if plot_save_path:
                    print(f"\nPlotting true vs. predicted values... saving to {plot_save_path}")
                    # Ensure y_true_np and predictions_np have same number of columns for plotting
                    # This typically means target_names used for pred_column_names should align with actual_target_names_found
                    if predictions_np.shape[1] == y_true_np.shape[1]:
                         plot_true_vs_pred_scatter(
                            y_true=y_true_np,
                            y_pred=predictions_np,
                            target_names=actual_target_names_found, # Use names of actual found targets for plotting
                            save_path=plot_save_path,
                            title=f"{plot_title}\nModel: {os.path.basename(self.model_path)}"
                        )
                         result['plot_path'] = plot_save_path
                    else:
                        print(f"Warning: Cannot plot. Number of predicted columns ({predictions_np.shape[1]}) "
                              f"does not match number of true target columns ({y_true_np.shape[1]}).")
            elif valid_targets_in_df:
                print(f"Warning: Some target columns specified in target_names were found ({valid_targets_in_df}), "
                      f"but not all ({target_names}). Predictions will be made, but y_true and plotting might be affected.")
            else: # No specified targets found
                print("Warning: None of the specified target_names found in DataFrame. Only making predictions.")
        
        return result

if __name__ == '__main__':
    """
    Example usage of the NNPredictor class.
    
    Instructions:
    1. Ensure the model_path and data_path variables below point to your actual files.
    2. !! IMPORTANT !! Verify that the hyperparameters (output_dim, hidden_dims, etc.)
       hardcoded in the NNPredictor._load_model() method match the architecture
       of the model specified by model_path. If they don't match, loading the model
       will likely fail or produce incorrect results. You may need to modify
       NNPredictor to accept these hyperparameters or load them from the checkpoint.
    3. Identify your target column(s) in the data_path CSV file.
    """
    import pandas as pd

    # --- Configuration ---
    model_path = "Features/Ni_alloys/4090/composition_texts/best_alloy_nn_model_state_dict.pt"
    data_path = "Features/Ni_alloys/4090/composition_texts/features.csv"

    target_columns = ["Solidus Temperature (℃)", "Liquidus Temperature (℃)"] 
    
    actual_output_dim = len(target_columns)
    
    model_feature_type_config = { # As per AlloyNN defaults
        'emb': ['emb'], # Assuming your embedding columns contain 'emb'
        'feature1': ['feature1'], # If you have 'feature1' type features
        'feature2': ['feature2']  # If you have 'feature2' type features
        # Add other feature types if your model was trained with them
    }

    
    model_feature_hidden_dims = { # As per AlloyNN defaults
        'emb': 256,
        'feature1': 0,
        'feature2': 0
        # Match these to your trained model's FeatureProcessor settings
    }
    
    model_prediction_hidden_dims = [2048, 1024,512,256,128] # As per AlloyNN default for PredictionNetwork
    model_dropout_rate = 0.0 # As per AlloyNN default
    
    print(f"--- Running NNPredictor Example ---")
    print(f"Model path: {model_path}")
    print(f"Data path: {data_path}")

    # --- Load Data ---
    try:
        data_df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}. Shape: {data_df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please check the path and try again.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- Prepare Features ---
    # Assumes all columns not in target_columns are features
    feature_names = [col for col in data_df.columns if col not in target_columns]
    
    if not feature_names:
        print("Error: No feature columns found. Please check target_columns and data file.")
        exit()
    
    if any(tc not in data_df.columns for tc in target_columns):
        print(f"Error: One or more target columns {target_columns} not found in the data file.")
        print(f"Available columns: {data_df.columns.tolist()}")
        exit()

    X_predict = data_df[feature_names].values.astype(np.float32)
    y_true = data_df[target_columns].values.astype(np.float32) # For reference, not used by predictor directly
    
    print(f"Features shape (X_predict): {X_predict.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"First 5 feature names: {feature_names[:5]}")
    print(f"Targets shape (y_true): {y_true.shape}")

    
    # For this example, let's check the expected output_dim
    expected_output_dim = actual_output_dim
    print(f"Expected model output_dim based on target_columns: {actual_output_dim}")
    print("Using the following architecture parameters for loading:")
    print(f"  feature_type_config: {model_feature_type_config}")
    print(f"  feature_hidden_dims: {model_feature_hidden_dims}")
    print(f"  prediction_hidden_dims: {model_prediction_hidden_dims}")
    print(f"  dropout_rate: {model_dropout_rate}")
    print("Please ensure these match your saved model's architecture if they are not None.")

    predictor = NNPredictor(model_path=model_path, device='cuda') # or 'cpu'

    # --- Make Predictions using predict_dataframe ---
    try:
        print(f"\nCalling predict_dataframe...")
        
        # Define where to save the plot
        plot_output_dir = os.path.join(os.path.dirname(model_path), "../plots_from_predictor") # Example: save in parent_dir_of_model/plots_from_predictor/
        os.makedirs(plot_output_dir, exist_ok=True)
        prediction_plot_path = os.path.join(plot_output_dir, f"{os.path.splitext(os.path.basename(data_path))[0]}_prediction_plot.png")

        results = predictor.predict_dataframe(
            data_df=data_df,
            feature_names=feature_names,
            output_dim=actual_output_dim,
            target_names=target_columns, # Pass target_columns to enable y_true extraction and plotting
            feature_type_config=model_feature_type_config,
            feature_hidden_dims=model_feature_hidden_dims,
            prediction_hidden_dims=model_prediction_hidden_dims,
            dropout_rate=model_dropout_rate,
            plot_save_path=prediction_plot_path,
            plot_title=f"Predictions for {os.path.basename(data_path)}"
        )

        predictions_df = results.get('predictions')
        print(f"\nPredictions DataFrame shape: {predictions_df.shape}")
        if not predictions_df.empty:
            print("First 5 predictions:")
            print(predictions_df.head())

        if 'y_true' in results:
            y_true_df = results.get('y_true')
            print(f"\nTrue Values DataFrame shape: {y_true_df.shape}")
            if not y_true_df.empty:
                print("First 5 true values:")
                print(y_true_df.head())
        
        if 'plot_path' in results and results['plot_path']:
            print(f"Comparison plot saved to: {results['plot_path']}")
        
    except FileNotFoundError as e:
        print(f"Error: A file was not found. {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except RuntimeError as e:
        print(f"RuntimeError during prediction: {e}")
        print("This might be due to a mismatch between the saved model's architecture and the one defined in NNPredictor._load_model() or parameters passed.")
        print("Please check the output_dim and other hyperparameters.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\n--- NNPredictor Example Finished ---")
# python -m src.models.predictors.nn_predictor