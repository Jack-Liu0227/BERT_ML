"""
BERT-based model for alloy property prediction using the 'adapters' library.
"""

import os
import json
import torch
from torch import nn
from transformers import AutoModel
from peft import get_peft_model, TaskType, IA3Config
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Optional, Dict, List, Union, Any
from src.models.base.alloys_nn import FeatureProcessor
from adapters import AutoAdapterModel


class AlloyDataset(Dataset):
    """
    Dataset class for alloy data with text and numerical features.
    (This is a copy from alloys_bert.py and can be refactored into a common module)
    
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
        self.use_features = use_features
        
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
        
class AlloyAdapterModel(nn.Module):
    """
    BERT-based model for alloy property prediction with PEFT's (IA)^3 adapter support.
    
    Args:
        pretrained_model_name (str): Name of pretrained BERT model
        output_dim (int): Output dimension
        dropout_rate (float): Dropout rate
        feature_config (Optional[Dict]): Configuration for the FeatureProcessor.
        use_adapter (bool): Whether to use an (IA)^3 adapter for fine-tuning.
        classifier_hidden_layers (Union[int, List[int]]): Classifier MLP hidden layers.
        use_batch_norm (bool): Whether to use BatchNorm after feature concatenation.
        batch_norm_momentum (float): Momentum for the BatchNorm layer.
        target_modules (Optional[List[str]]): List of modules to apply (IA)^3 to.
    """

    def __init__(self, 
                 pretrained_model_name: str,
                 output_dim: int = 3,
                 dropout_rate: float = 0.1,
                 feature_config: Optional[Dict] = None,
                 use_adapter: bool = False,
                 classifier_hidden_layers: Union[int, list] = 1,
                 use_batch_norm: bool = False,
                 batch_norm_momentum: float = 0.1,
                 target_modules: Optional[List[str]] = None):
        super(AlloyAdapterModel, self).__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.use_features = feature_config is not None
        self.use_batch_norm = use_batch_norm
        self.use_adapter = use_adapter
        
        # Load pretrained BERT model
        bert_model = AutoModel.from_pretrained(pretrained_model_name)
        
        # Add (IA)^3 adapter using PEFT if specified
        if self.use_adapter:
            if target_modules is None:
                # Default target modules for BERT-like models
                target_modules = ["query", "key", "value", "intermediate.dense", "output.dense"]
            
            # Ensure feedforward_modules is a subset of target_modules
            defined_feedforward_modules = ["intermediate.dense"]
            actual_feedforward_modules = [m for m in defined_feedforward_modules if m in target_modules]

            peft_config = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
                feedforward_modules=actual_feedforward_modules,
                inference_mode=False,
            )
            self.bert = get_peft_model(bert_model, peft_config)
            self.bert.print_trainable_parameters()
        else:
            self.bert = bert_model
        
        # --- Feature processing ---
        bert_dim = self.bert.config.hidden_size
        self.feature_processor = None
        self.bert_fcl = None

        if self.use_features:
            emb_hidden_dim = feature_config.get('feature_hidden_dims', {}).get('emb', 0)
            if emb_hidden_dim > 0:
                self.bert_fcl = nn.Linear(bert_dim, emb_hidden_dim)
                final_bert_dim = emb_hidden_dim
            else:
                final_bert_dim = bert_dim

            feature_config['exclude_emb'] = True
            self.feature_processor = FeatureProcessor(**feature_config)
            combined_dim = final_bert_dim + self.feature_processor.processed_dim
        else:
            combined_dim = bert_dim
        
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(combined_dim, momentum=batch_norm_momentum)
        else:
            self.batch_norm = None

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
        """Forward pass of the model."""
        # PEFT model forward pass for feature extraction
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # The output of a PEFT model in feature extraction mode is the last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        # Use the hidden state of the [CLS] token
        pooled_output = last_hidden_state[:, 0]
        
        if hasattr(self, 'bert_fcl') and self.bert_fcl is not None:
            pooled_output = self.bert_fcl(pooled_output)

        if self.feature_processor is not None and features is not None:
            processed_features = self.feature_processor(features)
            combined_features = torch.cat([pooled_output, processed_features], dim=1) if processed_features.numel() > 0 else pooled_output
        else:
            combined_features = pooled_output

        if self.batch_norm is not None:
            combined_features = self.batch_norm(combined_features)

        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)
        
        return {'logits': logits, 'loss': loss}
        
    def save_model(self, save_directory):
        """Saves the model components."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the PEFT adapter or the full model
        self.bert.save_pretrained(save_directory)

        # Save custom components
        custom_components = {
            'classifier': self.classifier.state_dict(),
            'bert_fcl': self.bert_fcl.state_dict() if self.bert_fcl else None,
            'batch_norm': self.batch_norm.state_dict() if self.batch_norm else None
        }
        torch.save(custom_components, os.path.join(save_directory, 'custom_components.pth'))

        # Save config
        config = {
            'pretrained_model_name': self.pretrained_model_name,
            'output_dim': self.classifier.network[-1].out_features,
            'dropout_rate': self.classifier.network[2].p if isinstance(self.classifier.network[2], nn.Dropout) else 0.1,
            'feature_config': self.feature_processor.config if self.use_features else None,
            'use_adapter': self.use_adapter,
            'classifier_hidden_layers': [l.out_features for l in self.classifier.network if isinstance(l, nn.Linear)][:-1],
            'use_batch_norm': self.use_batch_norm,
            'batch_norm_momentum': self.batch_norm.momentum if self.batch_norm else 0.1,
            'target_modules': self.bert.peft_config.get('default', {}).target_modules if self.use_adapter else None
        }
        with open(os.path.join(save_directory, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_pretrained(cls, load_directory):
        """Loads a model from a directory."""
        # Load config
        with open(os.path.join(load_directory, 'model_config.json'), 'r') as f:
            config = json.load(f)

        model = cls(
            pretrained_model_name=config['pretrained_model_name'],
            output_dim=config['output_dim'],
            dropout_rate=config['dropout_rate'],
            feature_config=config.get('feature_config'),
            use_adapter=config.get('use_adapter', False),
            classifier_hidden_layers=config.get('classifier_hidden_layers', 1),
            use_batch_norm=config.get('use_batch_norm', False),
            batch_norm_momentum=config.get('batch_norm_momentum', 0.1),
            target_modules=config.get('target_modules')
        )

        # Load PEFT model
        # The from_pretrained method of a PeftModel will load the base model and the adapter
        # config automatically. We pass the base model instance to our class constructor
        # and it gets wrapped. Here we just need to ensure the adapter weights are loaded.
        # This seems redundant if the PeftModel is created fresh, so we rely on the init.
        # Instead, we just need to load our custom parts.

        # Load custom components
        custom_components_path = os.path.join(load_directory, 'custom_components.pth')
        if os.path.exists(custom_components_path):
            custom_states = torch.load(custom_components_path)
            model.classifier.load_state_dict(custom_states['classifier'])
            if model.bert_fcl and 'bert_fcl' in custom_states and custom_states['bert_fcl']:
                model.bert_fcl.load_state_dict(custom_states['bert_fcl'])
            if model.batch_norm and 'batch_norm' in custom_states and custom_states['batch_norm']:
                model.batch_norm.load_state_dict(custom_states['batch_norm'])
        
        return model 