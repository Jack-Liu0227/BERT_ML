"""
BERT-based model for alloy property prediction
"""

import os
import json
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from datasets import Dataset
from typing import Optional, Dict, List, Union, Any
from peft import get_peft_model, LoraConfig, TaskType
from src.models.base.alloys_nn import FeatureProcessor

class AlloyDataset(Dataset):
    """
    Dataset class for alloy data with text and numerical features
    
    Args:
        texts (List[str]): List of text descriptions
        features (Optional[np.ndarray]): Numerical features
        targets (Optional[np.ndarray]): Target values
        tokenizer (Optional[AutoTokenizer]): BERT tokenizer
        max_length (int): Maximum sequence length
        use_features (bool): Whether to use numerical features
    """
    def __init__(self, texts: List[str], 
                 features: Optional[torch.Tensor] = None,
                 targets: Optional[torch.Tensor] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 max_length: int = 128,
                 use_features: bool = True):
        self.texts = texts
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_feature1 = use_feature1
        self.use_feature2 = use_feature2
        self.use_other_features = use_other_features
        
        if tokenizer is not None:
            self.encodings = tokenizer(texts, 
                                     truncation=True,
                                     padding=True,
                                     max_length=max_length,
                                     return_tensors='pt')

    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset"""
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        
        if self.features is not None and self.use_features:
            item['features'] = self.features[idx]
            
        if self.targets is not None:
            item['labels'] = self.targets[idx]
            
        return item

class PredictionNetwork(nn.Module):
    """
    Flexible MLP classifier for alloy property prediction.
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output dimension
        hidden_layers (Union[int, List[int]]): Number or list of hidden units for classifier MLP
        dropout_rate (float): Dropout rate
    """
    def __init__(self, input_dim, output_dim, hidden_layers=1, dropout_rate=0.1):
        super(PredictionNetwork, self).__init__()
        if isinstance(hidden_layers, int):
            hidden_sizes = [256] * hidden_layers
        else:
            hidden_sizes = list(hidden_layers)
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AlloyModel(nn.Module):
    """
    BERT-based model for alloy property prediction with fine-tuning support
    
    Args:
        pretrained_model_name (str): Name of pretrained BERT model
        output_dim (int): Output dimension
        dropout_rate (float): Dropout rate
        feature_config (Optional[Dict]): Configuration for the FeatureProcessor. 
            - In feature_config['feature_hidden_dims'], if a feature type's hidden_dim is 0, the raw features will be used.
            - If hidden_dim > 0, features will be processed by a fully connected layer.
            - This applies to the 'emb' type from BERT as well. If the 'emb' hidden_dim is 0, the raw BERT output is used.
        use_lora (bool): Whether to use LoRA for fine-tuning
        lora_config (Optional[Dict]): LoRA configuration
        classifier_hidden_layers (Union[int, List[int]]): Number or list of hidden units for classifier MLP
        use_batch_norm (bool): Whether to use BatchNorm after feature concatenation.
        batch_norm_momentum (float): Momentum for the BatchNorm layer.
    """

    def __init__(self, 
                 pretrained_model_name: str,
                 output_dim: int = 3,
                 dropout_rate: float = 0.1,
                 feature_config: Optional[Dict] = None,
                 use_lora: bool = False,
                 lora_config: Optional[Dict] = None,
                 classifier_hidden_layers: Union[int, list] = 1,
                 use_batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1):
        super(AlloyModel, self).__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.use_features = feature_config is not None
        self.use_batch_norm = use_batch_norm
        
        # Apply LoRA if specified
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": TaskType.FEATURE_EXTRACTION
                }
            self.bert = get_peft_model(self.bert, LoraConfig(**lora_config))
        
        # --- Feature processing ---
        bert_dim = self.bert.config.hidden_size
        self.feature_processor = None
        self.bert_fcl = None

        if self.use_features:
            # Process BERT embedding based on 'emb' hidden_dim
            emb_hidden_dim = feature_config.get('feature_hidden_dims', {}).get('emb', 0)
            if emb_hidden_dim > 0:
                self.bert_fcl = nn.Linear(bert_dim, emb_hidden_dim)
                final_bert_dim = emb_hidden_dim
            else:
                final_bert_dim = bert_dim  # Use raw BERT output dim

            # Process other numerical features from the input data
            feature_config['exclude_emb'] = True
            self.feature_processor = FeatureProcessor(**feature_config)
            
            combined_dim = final_bert_dim + self.feature_processor.processed_dim
        else:
            # No external features are being used, so the combined dimension is just the BERT dimension
            combined_dim = bert_dim
        
        # Optional BatchNorm layer
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(combined_dim, momentum=batch_norm_momentum)
        else:
            self.batch_norm = None

        # Use the new PredictionNetwork class for the classifier
        self.classifier = PredictionNetwork(
            input_dim=combined_dim,
            output_dim=output_dim,
            hidden_layers=classifier_hidden_layers,
            dropout_rate=dropout_rate
        )
    

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (Optional[torch.Tensor]): Token type IDs for BERT
            features (Optional[torch.Tensor]): Additional numerical features
            labels (Optional[torch.Tensor]): Target labels
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs including logits and loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use pooler_output if available, otherwise use [CLS] token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]

        # Pass BERT output through bert_fcl if it exists
        if hasattr(self, 'bert_fcl') and self.bert_fcl is not None:
            pooled_output = self.bert_fcl(pooled_output)

        # Process and combine with numerical features if available
        if self.feature_processor is not None and features is not None:
            processed_features = self.feature_processor(features)
            if processed_features.numel() > 0:  # If there are processed features
                combined_features = torch.cat([pooled_output, processed_features], dim=1)
            else:
                combined_features = pooled_output
        else:
            combined_features = pooled_output
        

        # Apply optional BatchNorm
        if self.batch_norm is not None:
            combined_features = self.batch_norm(combined_features)

        # Get predictions
        logits = self.classifier(combined_features)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def save_pretrained(self, save_directory: str):
        """
        Save model and tokenizer
        
        Args:
            save_directory (str): Directory to save the model
        """
        # 创建保存目录
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存 BERT 模型
        self.bert.save_pretrained(save_directory)
        
        # 保存分类器
        classifier_path = os.path.join(save_directory, 'classifier.pt')
        torch.save(self.classifier.state_dict(), classifier_path)
        
        # 保存特征处理器
        if self.use_features:
            feature_processor_path = os.path.join(save_directory, 'feature_processor.pt')
            torch.save(self.feature_processor.state_dict(), feature_processor_path)
        
        # 保存配置
        config = {
            'pretrained_model_name': self.bert.config._name_or_path,
            'use_features': self.use_features,
            'use_lora': hasattr(self.bert, 'peft_config'),
            'use_batch_norm': self.use_batch_norm,
            'batch_norm_momentum': self.batch_norm.momentum if self.use_batch_norm else None
        }
        
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_pretrained(cls, 
                       pretrained_model_name_or_path: str,
                       **kwargs) -> 'AlloyModel':
        """
        Load model from pretrained
        
        Args:
            pretrained_model_name_or_path (str): Path to pretrained model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            AlloyModel: Loaded model
        """
        # 加载配置
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = kwargs
        
        # 创建模型
        model = cls(
            pretrained_model_name=config.get('pretrained_model_name', pretrained_model_name_or_path),
            output_dim=config.get('output_dim', 3),
            dropout_rate=config.get('dropout_rate', 0.1),
            feature_config=config.get('feature_config', None),
            use_lora=config.get('use_lora', False),
            classifier_hidden_layers=config.get('classifier_hidden_layers', 1),
            use_batch_norm=config.get('use_batch_norm', False),
            batch_norm_momentum=config.get('batch_norm_momentum', 0.1)
        )
        
        # 加载 BERT 模型
        model.bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
        # 加载分类器
        classifier_path = os.path.join(pretrained_model_name_or_path, 'classifier.pt')
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path))
        
        # 加载特征处理器
        if model.use_features:
            feature_processor_path = os.path.join(pretrained_model_name_or_path, 'feature_processor.pt')
            if os.path.exists(feature_processor_path):
                model.feature_processor.load_state_dict(torch.load(feature_processor_path))
        
        return model 

