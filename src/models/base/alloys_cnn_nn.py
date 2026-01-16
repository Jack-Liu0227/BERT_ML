"""
Alloy property prediction neural network with combined 1D and 2D CNN feature extractors.
"""

import torch
from torch import nn
from typing import List, Dict, Optional, Any
import numpy as np
import json

class FcUnit(nn.Module):
    """
    A single MLP layer with optional normalization and activation.
    It orders the operations as: fc -> norm -> act_fn -> dropout
    """
    def __init__(self, in_features: int, out_features: int, normalization: str = 'none', activation: str = 'relu', dropout_prob: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(out_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(out_features)
        elif normalization == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        self.act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class Cnn1dUnit(nn.Module):
    """
    A single 1D CNN layer, followed by an activation and pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 1, padding: int = 1, pooling: str = 'max', activation: str = 'relu'):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size) if pooling == 'max' else nn.AvgPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act_fn(self.conv1d(x)))

class Cnn2dUnit(nn.Module):
    """
    A single 2D CNN layer, followed by an activation and pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 1, padding: int = 1, pooling: str = 'max', activation: str = 'relu'):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size) if pooling == 'max' else nn.AvgPool2d(kernel_size=kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act_fn(self.conv2d(x)))

class PredictionNetwork(nn.Module):
    """
    Prediction network module with customizable fully connected layers.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 128], dropout_rate: float = 0.0, activation: str = 'relu'):
        super(PredictionNetwork, self).__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        
        act_fn = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(negative_slope=0.01)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AlloyCnnV2(nn.Module):
    """
    An alloy property prediction model that combines 1D and 2D CNN-based feature extractors,
    inspired by architectures that process multiple embedding types with dedicated processing paths.
    This version aligns with the structure of CustomSimpleModel from reg_v1.py.
    """
    def __init__(self, 
                 column_names: List[str], 
                 output_dim: int,
                 cnn_1d_config: Dict[str, Dict[str, Any]],
                 cnn_2d_config: Dict[str, Any],
                 features_for_2d_cnn: List[str],
                 other_feature_configs: Optional[Dict[str, Any]],
                 prediction_hidden_dims: List[int],
                 dropout_rate: float = 0.0,
                 prediction_activation: str = 'relu'):
        super().__init__()
        self.column_names = column_names
        self.type2indices: Dict[str, List[int]] = {}

        # --- 1. 1D CNN Processors ---
        self.cnn_1d_networks = nn.ModuleDict()
        self.cnn_1d_output_dims: Dict[str, int] = {}

        for ftype, config in cnn_1d_config.items():
            keyword = config.get('keyword')
            if not keyword:
                raise ValueError(f"Each item in `cnn_1d_config` must have a 'keyword'. Missing in '{ftype}'.")
            
            indices = [i for i, col in enumerate(column_names) if keyword.lower() in col.lower()]
            if not indices:
                print(f"Warning: No columns found for 1D CNN feature type '{ftype}' with keyword '{keyword}'. Skipping.")
                continue
            
            self.type2indices[ftype] = indices
            dim = len(indices)

            # --- New: Pre-CNN FC layers (equivalent to simple_layer in CustomSimpleModel) ---
            pre_cnn_fc_layers: List[nn.Module] = []
            in_dim_pre = dim
            for out_dim_pre in config.get('pre_cnn_fc_layers', []):
                pre_cnn_fc_layers.append(FcUnit(in_dim_pre, out_dim_pre, dropout_prob=dropout_rate))
                in_dim_pre = out_dim_pre
            
            # Use a dummy tensor to find the output dimension of pre_cnn_fc_layers
            with torch.no_grad():
                temp_pre_cnn_seq = nn.Sequential(*pre_cnn_fc_layers)
                dummy_out_pre = temp_pre_cnn_seq(torch.randn(1, dim))
                pre_cnn_out_dim = dummy_out_pre.shape[1]

            # Build CNN layers for this path
            cnn_layers: List[nn.Module] = []
            in_channels = 1
            # Input to CNN is the output of pre_cnn_fc_layers, so its length is pre_cnn_out_dim
            current_dim = pre_cnn_out_dim 
            for out_c in config.get('out_channels', []):
                # Calculate output dimension after conv and pool
                k = config.get('kernel_size', 3)
                p = config.get('padding', 1)
                s = 1
                
                # Conv1d output size: floor((L_in + 2*padding - kernel_size)/stride) + 1
                conv_out_dim = np.floor((current_dim + 2 * p - k) / s) + 1
                # Pool1d output size: floor((L_in - kernel_size)/kernel_size) + 1
                pool_out_dim = np.floor((conv_out_dim - k) / k) + 1
                
                cnn_layers.append(Cnn1dUnit(in_channels, out_c, kernel_size=k, padding=p, activation=config.get('activation', 'relu'), pooling=config.get('pooling', 'max')))
                in_channels = out_c
                current_dim = int(pool_out_dim)

            cnn_layers.append(nn.Flatten())
            
            # Use a dummy tensor to find the flattened output dimension of CNN part
            with torch.no_grad():
                # We need to reshape the output of pre-cnn fc to be (batch, 1, dim) for CNN
                dummy_input_cnn = torch.randn(1, 1, pre_cnn_out_dim)
                temp_cnn_seq = nn.Sequential(*cnn_layers)
                dummy_out_cnn = temp_cnn_seq(dummy_input_cnn)
                cnn_out_dim = dummy_out_cnn.shape[1]

            # Build FC layers for this path (post-CNN)
            fc_layers_1d: List[nn.Module] = []
            in_dim = cnn_out_dim
            for out_dim_fc in config.get('fc_layers', []):
                fc_layers_1d.append(FcUnit(in_dim, out_dim_fc, dropout_prob=dropout_rate))
                in_dim = out_dim_fc

            self.cnn_1d_networks[ftype] = nn.Sequential(*(pre_cnn_fc_layers + cnn_layers + fc_layers_1d))
            self.cnn_1d_output_dims[ftype] = in_dim
        
        # --- 2. 2D CNN Processor ---
        self.features_for_2d_cnn = features_for_2d_cnn
        self.cnn_2d_network: Optional[nn.Module] = None
        self.cnn_2d_output_dim = 0
        
        dims_2d = [len(self.type2indices[ftype]) for ftype in self.features_for_2d_cnn if ftype in self.type2indices]
        
        if self.features_for_2d_cnn and not dims_2d:
             raise ValueError("None of the features specified in `features_for_2d_cnn` were found in the data based on `cnn_1d_config` keywords.")
        if dims_2d and len(set(dims_2d)) != 1:
            raise ValueError(f"All features for 2D CNN must have the same dimension, but found dimensions: {dims_2d}")
        
        if dims_2d:
            num_feature_types_for_2d = len(self.features_for_2d_cnn)
            embedding_dim_for_2d = dims_2d[0]
            
            cnn_layers_2d: List[nn.Module] = []
            in_channels_2d = 1
            for out_c in cnn_2d_config.get('out_channels', []):
                cnn_layers_2d.append(Cnn2dUnit(in_channels_2d, out_c, kernel_size=cnn_2d_config.get('kernel_size', 2), padding=cnn_2d_config.get('padding', 1), activation=cnn_2d_config.get('activation', 'relu'), pooling=cnn_2d_config.get('pooling', 'max')))
                in_channels_2d = out_c
            cnn_layers_2d.append(nn.Flatten())
            
            temp_cnn_seq_2d = nn.Sequential(*cnn_layers_2d)
            with torch.no_grad():
                dummy_input_2d = torch.randn(1, 1, num_feature_types_for_2d, embedding_dim_for_2d)
                dummy_out_2d = temp_cnn_seq_2d(dummy_input_2d)
                cnn_out_dim_2d = dummy_out_2d.shape[1]
            
            fc_layers_2d: List[nn.Module] = []
            in_dim_2d = cnn_out_dim_2d
            for out_dim_fc_2d in cnn_2d_config.get('fc_layers', []):
                fc_layers_2d.append(FcUnit(in_dim_2d, out_dim_fc_2d, dropout_prob=dropout_rate))
                in_dim_2d = out_dim_fc_2d

            self.cnn_2d_network = nn.Sequential(*(cnn_layers_2d + fc_layers_2d))
            self.cnn_2d_output_dim = in_dim_2d

        # --- 3. Other Features (Bypassing CNN) ---
        self.other_feature_indices: List[int] = []
        if other_feature_configs:
            for ftype, config in other_feature_configs.items():
                keyword = config.get('keyword')
                if not keyword:
                    raise ValueError(f"Each item in `other_feature_configs` must have a 'keyword'. Missing in '{ftype}'.")
                
                indices = [i for i, col in enumerate(column_names) if keyword.lower() in col.lower()]
                if not indices:
                    print(f"Warning: No columns found for other feature '{ftype}' with keyword '{keyword}'. Skipping.")
                    continue
                self.other_feature_indices.extend(indices)
        self.other_feature_indices = sorted(list(set(self.other_feature_indices)))

        # --- 4. Final Prediction Network ---
        total_input_dim = sum(self.cnn_1d_output_dims.values()) + self.cnn_2d_output_dim + len(self.other_feature_indices)
        if total_input_dim == 0:
            raise ValueError("No features were processed. Check configurations and data.")
            
        self.prediction_network = PredictionNetwork(total_input_dim, output_dim, prediction_hidden_dims, dropout_rate, activation=prediction_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed_features: List[torch.Tensor] = []

        # 1. Deconstruct the sequential model for 1D CNN paths to handle intermediate reshaping
        for ftype, network in self.cnn_1d_networks.items():
            # Find where CNN layers start
            split_point = 0
            for i, layer in enumerate(network):
                if isinstance(layer, Cnn1dUnit):
                    split_point = i
                    break
            
            pre_cnn_layers = network[:split_point]
            cnn_and_post_layers = network[split_point:]

            # Apply pre-cnn fc layers
            indices = self.type2indices[ftype]
            feature_input = x[:, indices]
            pre_cnn_output = pre_cnn_layers(feature_input)

            # Reshape for CNN and apply rest of the network
            cnn_input = pre_cnn_output.unsqueeze(1)
            final_output = cnn_and_post_layers(cnn_input)
            processed_features.append(final_output)

        # 2D CNN path
        if self.cnn_2d_network and self.features_for_2d_cnn:
            features_to_stack = [x[:, self.type2indices[ftype]] for ftype in self.features_for_2d_cnn if ftype in self.type2indices]
            if features_to_stack:
                stacked_features = torch.stack(features_to_stack, dim=1)
                input_2d = stacked_features.unsqueeze(1)
                processed_features.append(self.cnn_2d_network(input_2d))
        
        # Other features path
        if self.other_feature_indices:
            processed_features.append(x[:, self.other_feature_indices])
        
        if not processed_features:
            raise RuntimeError("No features were processed during forward pass. Check model configuration.")
            
        final_concat = torch.cat(processed_features, dim=1)
        return self.prediction_network(final_concat)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return self(X_tensor).cpu().numpy()
    
    def print_structure(self, file_path: Optional[str] = None):
        """
        Prints the model structure as a JSON string and optionally saves it to a file.
        
        Args:
            file_path (Optional[str]): If provided, the structure is written to this file.
        """
        structure_info = {
            "model_class": self.__class__.__name__,
            "output_dim": self.prediction_network.network[-1].out_features,
            "1d_cnn_paths": {ftype: str(net) for ftype, net in self.cnn_1d_networks.items()},
            "2d_cnn_path": str(self.cnn_2d_network) if self.cnn_2d_network else "Not used",
            "other_features_count": len(self.other_feature_indices),
            "prediction_network": str(self.prediction_network)
        }
        structure_str = json.dumps(structure_info, indent=2)
        print(structure_str)
        if file_path:
            try:
                with open(file_path, 'w') as f: # Use 'w' to create/overwrite
                    f.write(structure_str)
            except IOError as e:
                print(f"Warning: Could not write model structure to {file_path}. Error: {e}")
