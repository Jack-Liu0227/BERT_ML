"""
Neural network model for alloy property prediction
"""

import torch
from torch import nn

class FeatureProcessor(nn.Module):
    """
    General feature processing module supporting multiple feature types, with emb features processed separately
    Args:
        column_names (list): List of feature column names
        feature_type_config (dict): {type: [keywords]} Feature type and their keywords
        feature_hidden_dims (dict): {type: hidden_dim} Hidden dim for each feature type. If hidden_dim=0, the raw features will be concatenated directly; if hidden_dim>0, features will be processed by a fully connected layer before concatenation.
        exclude_emb (bool): Whether to exclude 'emb' features from processing. Defaults to False, as 'emb' is expected to be a standard input feature for AlloyNN.
    """
    def __init__(self, column_names, feature_type_config, feature_hidden_dims, exclude_emb=False):
        """
        Initialize feature processing module
        
        Args:
            column_names (list): List of feature column names, e.g. ['Fe', 'Cu', 'Ni', 'emb_1', 'emb_2']
            feature_type_config (dict): Feature type config dict defining keywords for each type, e.g. {'comp': ['Fe', 'Cu'], 'emb': ['emb']}
            feature_hidden_dims (dict): Hidden dimensions for each feature type, e.g. {'comp': 64, 'emb': 32}
            exclude_emb (bool): Whether to exclude 'emb' features from processing.
        """
        super().__init__()
        
        # 保存输入参数 / Save input parameters               
        self.column_names = column_names
        self.feature_type_config = feature_type_config  
        self.feature_hidden_dims = feature_hidden_dims
        self.exclude_emb = exclude_emb
        
        # 初始化映射字典 / Initialize mapping dictionaries
        self.type2cols = {}      # 特征类型到列名的映射 / Mapping from type to column names
        self.type2indices = {}   # 特征类型到列索引的映射 / Mapping from type to column indices  
        self.type2dim = {}       # 特征类型到输入维度的映射 / Mapping from type to input dimensions
        self.type2fc = nn.ModuleDict()  # 特征类型到全连接层的映射 / Mapping from type to FC layers
        
        # 记录处理后的总维度 / Track total processed dimension
        processed_dim = 0
        
        # 处理每种特征类型 / Process each feature type
        for ftype, keywords in feature_type_config.items():
            # If exclude_emb is True and ftype is 'emb', skip
            if exclude_emb and ftype == 'emb':
                continue
            
            # Select columns by keywords, case-insensitive
            cols = [col for col in column_names if any(k.lower() in col.lower() for k in keywords)]
            self.type2cols[ftype] = cols
            
            # Select indices by keywords, case-insensitive
            indices = [i for i, col in enumerate(column_names) if any(k.lower() in col.lower() for k in keywords)]
            self.type2indices[ftype] = indices
            
            dim = len(cols)
            self.type2dim[ftype] = dim
            hidden_dim = feature_hidden_dims.get(ftype, 0)
            # If hidden_dim > 0, use FC layer; if 0, use raw features
            if dim > 0 and hidden_dim > 0:
                self.type2fc[ftype] = nn.Sequential(
                    nn.Linear(dim, hidden_dim)
                )
                processed_dim += hidden_dim
            elif dim > 0 and hidden_dim == 0:
                processed_dim += dim
        self.processed_dim = processed_dim

    def forward(self, x):
        """
        前向传播函数，处理不同类型的特征并拼接
        Forward pass function that processes different types of features and concatenates them

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, total_features)
                            Input tensor with shape (batch_size, total_features)

        Returns:
            torch.Tensor: 拼接后的特征张量，形状为 (batch_size, processed_dim)
                         Concatenated feature tensor with shape (batch_size, processed_dim)
        """
        result_tensors = []
        
        for ftype in self.type2cols:
            # 如果是emb特征且exclude_emb为True，则跳过
            if self.exclude_emb and ftype == 'emb':
                continue
                
            indices = self.type2indices[ftype]
            hidden_dim = self.feature_hidden_dims.get(ftype, 0)
            
            if indices:
                # Ensure features are on the same device as the model
                features = x[:, indices].to(x.device)
                if hidden_dim > 0 and ftype in self.type2fc:
                    out = self.type2fc[ftype](features)
                    result_tensors.append(out)
                elif hidden_dim == 0:
                    result_tensors.append(features)
                    
        if not result_tensors:
            # Return a zero tensor with the correct batch size and device if no features are processed
            return torch.zeros(x.size(0), 0, device=x.device)
            
        return torch.cat(result_tensors, dim=1)

class PredictionNetwork(nn.Module):
    """
    Prediction network module with customizable fully connected layers.
    预测网络模块，可自定义全连接层结构
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output dimension
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout rate
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout_rate=0.0):
        super(PredictionNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class AlloyNN(nn.Module):
    """
    通用合金性能预测神经网络，支持多种特征类型
    General alloy property prediction neural network supporting multiple feature types
    Args:
        column_names (list): List of feature column names
        output_dim (int): Output dimension
        feature_type_config (dict): {type: [keywords]} 特征类型及其关键词
        feature_hidden_dims (dict): {type: hidden_dim} 每种特征类型的隐藏层维度
        hidden_dims (list): List of hidden layer dimensions for prediction network
        dropout_rate (float): Dropout rate
    """
    def __init__(self, column_names, output_dim=3,
                 feature_type_config=None,
                 feature_hidden_dims=None,
                 hidden_dims=[256, 128], dropout_rate=0.0):
        super(AlloyNN, self).__init__()
        # 默认支持的特征类型及关键词
        if feature_type_config is None:
            feature_type_config = {
                'raw': column_names  # 所有特征都归为 raw
            }
        # 默认隐藏层维度
        if feature_hidden_dims is None:
            feature_hidden_dims = {
                'raw': 0
            }
        
        self.use_feature_processor = any(v > 0 for v in feature_hidden_dims.values())

        if self.use_feature_processor:
            self.feature_processor = FeatureProcessor(
                column_names=column_names,
                feature_type_config=feature_type_config,
                feature_hidden_dims=feature_hidden_dims,
                exclude_emb=False
            )
            # The input dimension for the prediction network is the output of the feature processor
            input_dim_for_pred = self.feature_processor.processed_dim
        else:
            self.feature_processor = None
            # When bypassing, the input to prediction network is the raw feature vector
            input_dim_for_pred = len(column_names)

        self.prediction_network = PredictionNetwork(
            input_dim=input_dim_for_pred,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        if self.use_feature_processor and self.feature_processor is not None:
            processed_features = self.feature_processor(x)
        else:
            processed_features = x
        output = self.prediction_network(processed_features)
        return output

    def predict(self, X):
        """
        Make predictions for input features
        Args:
            X: Input features (numpy array or torch tensor)
        Returns:
            numpy array: Model predictions
        """
        self.eval()  # Set model to evaluation mode
        # Convert input to tensor if it's not already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        # Move input to the same device as the model
        device = next(self.parameters()).device
        X = X.to(device)
        # Make predictions
        with torch.no_grad():
            predictions = self(X)
        # Convert predictions to numpy array
        return predictions.cpu().numpy()
    def print_structure(self, file_path=None):
        """
        打印模型结构，包括特征处理和预测网络层级结构
        Print the model structure, including feature processor and prediction network
        Args:
            file_path (str, optional): If provided, save the structure to this file
        """
        print("[AlloyNN] 正在输出模型结构... (Printing model structure...)")
        lines = []
        lines.append("\n========== AlloyNN Model Structure ==========")
        
        if self.use_feature_processor and self.feature_processor is not None:
            lines.append("FeatureProcessor:")
            for ftype, cols in self.feature_processor.type2cols.items():
                msg = f"  - {ftype}: {len(cols)} features -> hidden_dim {self.feature_processor.feature_hidden_dims.get(ftype, 0)}"
                lines.append(msg)
                lines.append(f"    columns: {cols}")
            lines.append(f"  Total processed dim: {self.feature_processor.processed_dim}")
            prev_dim = self.feature_processor.processed_dim
        else:
            lines.append("FeatureProcessor: Bypassed (all hidden_dims are 0)")
            # Assuming the number of columns in the input tensor `x` is the input to the prediction network
            prev_dim = self.prediction_network.input_dim

        lines.append("\nPredictionNetwork:")
        for i, hidden_dim in enumerate(self.prediction_network.hidden_dims):
            msg = f"  Layer {i}: Linear({prev_dim}, {hidden_dim}) + ReLU" + (f" + Dropout({self.prediction_network.dropout_rate})" if self.prediction_network.dropout_rate > 0 else "")
            lines.append(msg)
            prev_dim = hidden_dim
        
        out_msg = f"  Output: Linear({prev_dim}, {self.prediction_network.output_dim})"
        lines.append(out_msg)
        lines.append("============================================\n")
        
        structure_str = '\n'.join(lines)
        print(structure_str)
        
        if file_path is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(structure_str)
            print(f"[AlloyNN] 模型结构已保存到: {file_path} (Model structure saved to: {file_path})")
