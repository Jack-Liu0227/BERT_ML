"""
BERT model trainer implementation
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from .base_trainer import BaseTrainer
from ..base.alloys_bert import AlloyModel, AlloyDataset

class BERTTrainer(BaseTrainer):
    """
    Trainer for BERT-based alloy property prediction models
    
    Args:
        result_dir (str): Directory to save training results
        model_name (str): Name of the model
        target_names (List[str]): List of target names
        train_data (Dict): Training data
        val_data (Dict): Validation data
        model_params (Dict): Model parameters
        training_params (Dict): Training parameters
    """
    def __init__(self,
                 result_dir: str,
                 model_name: str,
                 target_names: List[str],
                 train_data: Dict[str, Any],
                 val_data: Dict[str, Any],
                 model_params: Dict[str, Any] = None,
                 training_params: Dict[str, Any] = None):
        """
        Initialize BERT trainer
        
        Args:
            result_dir (str): Directory to save training results
            model_name (str): Name of the model
            target_names (List[str]): List of target names
            train_data (Dict): Training data
            val_data (Dict): Validation data
            model_params (Dict): Model parameters
            training_params (Dict): Training parameters
        """
        self.model_params = model_params or {}
        self.training_params = training_params or {}
        # 先用参数 target_names 创建模型
        model = self._create_model(target_names)
        # 初始化父类，父类会设置 self.target_names
        super().__init__(
            model=model,
            result_dir=result_dir,
            model_name=model_name,
            target_names=target_names,
            early_stopping_patience=self.training_params.get('early_stopping_patience', 10),
            early_stopping_delta=self.training_params.get('early_stopping_delta', 1e-4)
        )
        # 创建数据集
        self.train_dataset = self._create_dataset(train_data)
        self.val_dataset = self._create_dataset(val_data)
        # 创建训练器
        self.trainer = self._create_trainer()
        # 保存配置
        self._save_config()
        
    def _create_dataset(self, data: Dict) -> Dataset:
        """
        Create dataset for training/validation
        
        Args:
            data (Dict): Data dictionary containing texts, targets, and features
            
        Returns:
            Dataset: HuggingFace dataset
        """
        # 准备数据
        dataset_dict = {
            'text': data['texts'],
            'labels': data['targets']
        }
        
        # 如果有特征，添加到数据集
        if data.get('features') is not None:
            dataset_dict['features'] = data['features']
        
        # 创建数据集
        dataset = Dataset.from_dict(dataset_dict)
        
        # 对文本进行编码
        def encode_text(example):
            # 确保文本是字符串类型
            text = str(example['text'])
            # 使用 tokenizer 进行编码，确保启用 padding 和 truncation
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.model_params.get('max_length', 128),
                return_tensors=None  # 让 tokenizer 返回列表而不是张量
            )
            return encoded
        
        # 应用编码
        dataset = dataset.map(
            encode_text,
            batched=False,  # 改为 False，一次处理一个样本
            remove_columns=['text']  # 移除原始文本列
        )
        
        return dataset
        
    def _create_model(self, target_names: List[str]) -> AlloyModel:
        """
        Create BERT model
        
        Returns:
            AlloyModel: Initialized model
        """
        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_params.get('pretrained_model_name', 'bert-base-uncased')
        )
        
        # 创建模型
        model = AlloyModel(
            pretrained_model_name=self.model_params.get('pretrained_model_name', 'bert-base-uncased'),
            feature_dim=self.model_params.get('feature_dim', 0),
            output_dim=len(target_names),
            dropout_rate=self.model_params.get('dropout_rate', 0.1),
            use_features=self.model_params.get('use_features', True),
            use_lora=self.model_params.get('use_lora', False)
        )
        
        return model
        
    def _create_trainer(self) -> Trainer:
        """
        Create HuggingFace trainer
        
        Returns:
            Trainer: Initialized trainer
        """
        # 检测分布式环境
        is_distributed = int(os.environ.get('LOCAL_RANK', -1)) != -1
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 计算每个设备的batch size
        per_device_batch_size = self.training_params.get('batch_size', 32)
        if is_distributed:
            # 在分布式环境中，总batch size会乘以GPU数量
            total_batch_size = per_device_batch_size * world_size
            print(f"\nDistributed training detected:")
            print(f"World size: {world_size}")
            print(f"Local rank: {local_rank}")
            print(f"Per device batch size: {per_device_batch_size}")
            print(f"Total batch size: {total_batch_size}")
        else:
            print(f"\nSingle GPU training:")
            print(f"Batch size: {per_device_batch_size}")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=os.path.join(self.result_dir, self.model_name),
            num_train_epochs=self.training_params.get('num_epochs', 3),
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            warmup_steps=self.training_params.get('warmup_steps', 500),
            weight_decay=self.training_params.get('weight_decay', 0.01),
            logging_dir=os.path.join(self.result_dir, 'logs'),
            logging_steps=self.training_params.get('logging_steps', 100),
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_total_limit=1,  # 只保留最好的模型
            remove_unused_columns=False,
            report_to=[],  # 禁用所有报告工具，包括 wandb
            # 分布式训练参数
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
            ddp_backend='nccl' if torch.cuda.is_available() else 'gloo',
            # 根据是否分布式设置不同的保存策略
            save_on_each_node=is_distributed,  # 在分布式训练时在每个节点保存
            # 梯度累积步数，用于控制总batch size
            gradient_accumulation_steps=1,
            # 是否使用混合精度训练
            fp16=torch.cuda.is_available(),  # 如果有GPU就使用混合精度
            # 是否使用梯度检查点以节省显存
            gradient_checkpointing=self.training_params.get('use_lora', False),  # 如果使用LoRA就启用梯度检查点
            # 禁用分布式训练初始化（由 torchrun 处理）
            ddp_timeout=1800,  # 增加超时时间
            # 使用新的设备选择参数
            use_cpu=not torch.cuda.is_available(),
            # 使用新的评估策略参数
            eval_strategy='epoch'
        )
        
        # 创建训练器
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self._compute_metrics
        )
        
    def _compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics
        
        Args:
            eval_pred: Tuple of predictions and labels
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        predictions, labels = eval_pred
        metrics = {}
        
        # 计算每个目标的指标
        for i, target in enumerate(self.target_names):
            target_pred = predictions[:, i]
            target_true = labels[:, i]
            
            # MSE
            metrics[f"{target}_mse"] = np.mean((target_pred - target_true) ** 2)
            
            # MAE
            metrics[f"{target}_mae"] = np.mean(np.abs(target_pred - target_true))
            
            # R²
            ss_res = np.sum((target_true - target_pred) ** 2)
            ss_tot = np.sum((target_true - np.mean(target_true)) ** 2)
            metrics[f"{target}_r2"] = 1 - (ss_res / (ss_tot + 1e-8))
        
        # 计算平均指标
        metrics['mean_mse'] = np.mean([metrics[f"{target}_mse"] for target in self.target_names])
        metrics['mean_mae'] = np.mean([metrics[f"{target}_mae"] for target in self.target_names])
        metrics['mean_r2'] = np.mean([metrics[f"{target}_r2"] for target in self.target_names])
        
        return metrics
        
    def train(self, train_data: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model"""
        # 创建数据集
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)
        
        # 训练模型
        print("\nStarting training...")
        print(f"Model: {self.model_name}")
        print(f"Targets: {', '.join(self.target_names)}")
        
        # 设置训练参数，确保保存最佳模型
        self.trainer.args.save_strategy = "epoch"
        self.trainer.args.evaluation_strategy = "epoch"
        self.trainer.args.load_best_model_at_end = True
        self.trainer.args.metric_for_best_model = "eval_loss"
        self.trainer.args.greater_is_better = False
        self.trainer.args.save_total_limit = 1  # 只保留最好的模型
        
        train_result = self.trainer.train()
        
        # 保存训练历史
        history = {
            'train_loss': train_result.training_loss,
            'train_metrics': train_result.metrics,
            'train_runtime': train_result.metrics.get('train_runtime', 0),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
            'train_steps_per_second': train_result.metrics.get('train_steps_per_second', 0),
            'epoch': train_result.metrics.get('epoch', 0)
        }
        
        # 保存训练历史
        history_file = os.path.join(self.result_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        # 保存最佳模型信息
        best_model_info = {
            'best_model_path': self.trainer.state.best_model_checkpoint,
            'best_metric': self.trainer.state.best_metric,
            'epoch': train_result.metrics.get('epoch', 0)
        }
        best_model_info_file = os.path.join(self.result_dir, 'best_model_info.json')
        with open(best_model_info_file, 'w') as f:
            json.dump(best_model_info, f, indent=4)
        
        print(f"\nTraining completed in {history['train_runtime']:.2f} seconds")
        print(f"Training loss: {history['train_loss']:.4f}")
        print(f"Best model saved at: {best_model_info['best_model_path']}")
        print(f"Best metric value: {best_model_info['best_metric']:.4f}")
        
        return history
        
    def predict(self, data: Dict) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data (Dict): Data dictionary containing texts and features
            
        Returns:
            np.ndarray: Model predictions
        """
        # 创建数据集
        dataset = self._create_dataset(data)
        
        # 获取预测结果
        predictions = self.trainer.predict(dataset)
        
        return predictions.predictions
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Dictionary of metrics for this epoch
        """
        # 训练一个 epoch
        train_result = self.trainer.train()
        
        # 提取指标
        metrics = {
            'loss': train_result.training_loss
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dict[str, float]: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if 'numerical_features' in batch:
                    numerical_features = batch['numerical_features'].to(self.device)
                else:
                    numerical_features = None
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features
                )
                loss = self.criterion(outputs, labels)
                
                # Store predictions and targets
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Compute metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute R² score for each target
        r2_scores = []
        for i in range(all_targets.shape[1]):
            target_preds = all_preds[:, i]
            target_true = all_targets[:, i]
            r2 = 1 - np.sum((target_true - target_preds) ** 2) / np.sum((target_true - np.mean(target_true)) ** 2)
            r2_scores.append(r2)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'r2': np.mean(r2_scores)
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        # 保存检查点
        checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f'{self.model_name}_epoch_{epoch+1}.pt'
        )
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        
        # 如果是最佳模型，保存到最佳模型目录
        if is_best:
            best_model_path = os.path.join(
                self.result_dir,
                f'{self.model_name}_best'
            )
            self.model.save_pretrained(best_model_path)
            
            # 保存最佳模型信息
            best_model_info = {
                'epoch': epoch,
                'checkpoint_path': checkpoint_path
            }
            info_path = os.path.join(self.result_dir, 'best_model_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(best_model_info, f, indent=4)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        # 加载模型
        self.model = self._create_model(self.target_names).from_pretrained(checkpoint_path)
        # 更新训练器
        self.trainer = self._create_trainer()
        print("Checkpoint loaded successfully")
    
    def train_cross_validation(self,
                             num_epochs: int,
                             num_folds: int,
                             fold_data: List[Dict[str, Any]],
                             callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Train model using cross-validation
        
        Args:
            num_epochs (int): Number of epochs per fold
            num_folds (int): Number of cross-validation folds
            fold_data (List[Dict[str, Any]]): List of fold data dictionaries
            callbacks (Optional[List[Callable]]): List of callback functions
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        print(f"\nTraining {self.model_name} with {num_folds}-fold cross-validation...")
        start_time = time.time()
        
        # 初始化结果字典
        cv_results = {
            'fold_metrics': [],
            'best_fold': 0,
            'best_val_loss': float('inf'),
            'mean_val_r2': 0.0,
            'std_val_r2': 0.0,
            'mean_val_loss': 0.0,
            'std_val_loss': 0.0,
            'training_time': 0.0
        }
        
        # 训练每个折
        for fold in range(num_folds):
            print(f"\nTraining fold {fold+1}/{num_folds}")
            
            # 更新数据集
            self.train_dataset = self._create_dataset(fold_data[fold])
            self.val_dataset = self._create_dataset(fold_data[fold])
            
            # 更新训练器
            self.trainer = self._create_trainer()
            
            # 训练模型
            train_result = self.trainer.train()
            
            # 保存模型
            fold_model_path = os.path.join(
                self.result_dir,
                f'{self.model_name}_fold_{fold+1}'
            )
            self.model.save_pretrained(fold_model_path)
            
            # 获取验证指标
            eval_results = self.trainer.evaluate()
            
            # 存储折的结果
            fold_metrics = {
                'fold': fold,
                'val_loss': eval_results['eval_loss'],
                'val_r2': eval_results['mean_r2'],
                'metrics': eval_results
            }
            cv_results['fold_metrics'].append(fold_metrics)
            
            # 更新最佳模型
            if eval_results['eval_loss'] < cv_results['best_val_loss']:
                cv_results['best_val_loss'] = eval_results['eval_loss']
                cv_results['best_fold'] = fold
                
                # 保存最佳模型
                best_model_path = os.path.join(
                    self.result_dir,
                    f'{self.model_name}_best'
                )
                self.model.save_pretrained(best_model_path)
        
        # 计算平均指标
        val_losses = [metrics['val_loss'] for metrics in cv_results['fold_metrics']]
        val_r2s = [metrics['val_r2'] for metrics in cv_results['fold_metrics']]
        
        cv_results['mean_val_loss'] = np.mean(val_losses)
        cv_results['std_val_loss'] = np.std(val_losses)
        cv_results['mean_val_r2'] = np.mean(val_r2s)
        cv_results['std_val_r2'] = np.std(val_r2s)
        cv_results['training_time'] = time.time() - start_time
        
        # 保存交叉验证结果
        cv_results_path = os.path.join(self.logs_dir, 'cv_results.json')
        with open(cv_results_path, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, indent=4)
        
        print(f"\nCross-validation completed in {cv_results['training_time']:.2f} seconds")
        print(f"Best model from fold {cv_results['best_fold']+1}")
        print(f"Mean validation R²: {cv_results['mean_val_r2']:.4f} ± {cv_results['std_val_r2']:.4f}")
        print(f"Mean validation loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
        
        return cv_results
    
    def _save_config(self):
        """Save training configuration"""
        config = {
            'model_params': self.model_params,
            'training_params': self.training_params,
            'target_names': self.target_names,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_delta': self.early_stopping_delta
        }
        
        config_path = os.path.join(self.result_dir, f'{self.model_name}_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4) 