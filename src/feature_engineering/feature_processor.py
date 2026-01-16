import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
from typing import Optional, Tuple, Union

from src.feature_engineering.data_processor import DataProcessor
from src.feature_engineering.feature_extractor import FeatureExtractor
from src.feature_engineering.embedding_manager import EmbeddingManager
from src.feature_engineering.utils import setup_logging, ensure_dir, save_features, standardize_features, save_feature_names

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特征处理器主类"""
    
    def __init__(self,
                 data_path=None,
                 random_state=110,
                 use_process_embedding=True,
                 use_element_embedding=True,
                 # Removed use_matminer as FeatureExtractor does not support it directly
                 use_feature1=False,
                 use_feature2=False,
                 use_composition_feature=False,
                 standardize_features=False,
                 use_temperature=False,
                 feature_dir=None,
                 log_dir=None,
                 target_columns=None,
                 model_name="steelbert",
                 other_features_name=None,
                 use_joint_composition_process_embedding=False,
                 processing_text_column="Processing_Description"):
        """
        初始化特征处理器

        Args:
            data_path: 数据文件路径
            random_state: 随机种子
            use_process_embedding: 是否使用工艺描述嵌入向量
            use_composition_embedding: 是否使用元素组分嵌入向量
            # use_matminer: 是否使用Matminer特征 (Removed from args, handled as warning)
            use_feature1: 是否使用Feature1特征
            use_feature2: 是否使用Feature2特征
            use_composition_feature: 是否使用元素组分特征
            standardize_features: 是否标准化特征
            use_temperature: 是否使用温度特征
            feature_dir: 特征存储目录
            log_dir: 日志存储目录
            target_columns: 目标变量列名列表
            model_name: 模型名称（如 "steelbert", "matscibert", "scibert"），默认为 "steelbert"
            other_features_name: list, user-specified feature column names
            use_joint_composition_process_embedding: 是否使用联合组分+工艺BERT嵌入
            processing_text_column: str, 工艺描述文本列名，用于生成嵌入向量，默认为 "Processing_Description"
        """
        self.data_path = data_path
        self.random_state = random_state
        self.use_process_embedding = use_process_embedding
        self.use_element_embedding = use_element_embedding
        self.use_matminer = False # Always set to False as FeatureExtractor does not have it
        self.use_feature1 = use_feature1
        self.use_feature2 = use_feature2
        self.use_composition_feature = use_composition_feature
        self.other_features_name = other_features_name
        self.standardize_features = standardize_features
        self.use_temperature = use_temperature
        self.target_columns = target_columns if target_columns is not None else ['YS(MPa)', 'UTS(MPa)', 'El(%)']
        self.use_joint_composition_process_embedding = use_joint_composition_process_embedding
        self.model_name = model_name  # 保存模型名称供后续使用
        self.processing_text_column = processing_text_column  # 工艺描述文本列名

        # Debug prints for received parameters
        print(f"[DEBUG FeatureProcessor.__init__] Received use_process_embedding: {use_process_embedding}")
        print(f"[DEBUG FeatureProcessor.__init__] Received use_element_embedding: {use_element_embedding}")
        print(f"[DEBUG FeatureProcessor.__init__] Received use_composition_feature: {use_composition_feature}")
        print(f"[DEBUG FeatureProcessor.__init__] Received use_joint_composition_process_embedding: {use_joint_composition_process_embedding}")
        print(f"[DEBUG FeatureProcessor.__init__] Received standardize_features: {standardize_features}")
        print(f"[DEBUG FeatureProcessor.__init__] Using model: {model_name}")

        # 设置特征目录
        if feature_dir is not None:
            self.feature_dir = feature_dir

        # 设置日志目录
        if log_dir is not None:
            self.log_dir = log_dir

        
        # 创建目录
        ensure_dir(self.feature_dir)
        ensure_dir(self.log_dir)
        
        # 设置日志
        setup_logging(log_name=f"{Path(self.feature_dir).name}.log", log_dir=self.log_dir)
        
        # 初始化组件
        self.data_processor = DataProcessor(data_path)
        self.feature_extractor = FeatureExtractor()

        # 初始化 EmbeddingManager
        self.embedding_manager = EmbeddingManager(model_name=model_name)
        
        # 初始化数据和特征变量
        self.data = None
        self.X: pd.DataFrame = pd.DataFrame() # Initialize as empty DataFrame
        self.y: Optional[Union[pd.DataFrame, pd.Series]] = pd.DataFrame() # Initialize as empty DataFrame
        self.scaler = None
        
        logger.info(f"FeatureProcessor initialized")
        logger.info(f"Feature configuration: process_embedding={self.use_process_embedding}, "
                   f"element_embedding={self.use_element_embedding}, matminer={self.use_matminer}, "
                   f"feature1={self.use_feature1}, feature2={self.use_feature2}, composition_feature={self.use_composition_feature}, "
                   f"temperature={self.use_temperature}")
        logger.info(f"Feature will be stored in: {self.feature_dir}")

    @staticmethod
    def build_feature_path(base_dir: str, model_name: str, dataset_path: str) -> str:
        """
        构建符合新目录结构的特征路径

        Args:
            base_dir: 基础目录，通常是 "Features"
            model_name: 模型名称（steelbert, matscibert, scibert, traditionalModel）
            dataset_path: 数据集路径，例如 "Al_Alloys/USTB_new/all_features"

        Returns:
            完整的特征目录路径，格式为 "Features/{model_name}/{dataset_path}"

        Example:
            >>> FeatureProcessor.build_feature_path("Features", "steelbert", "Steel/USTB_steel/all_features")
            'Features/steelbert/Steel/USTB_steel/all_features'
        """
        return os.path.join(base_dir, model_name, dataset_path)

    @staticmethod
    def build_output_path(base_dir: str, model_name: str, experiment_path: str) -> str:
        """
        构建符合新目录结构的输出路径

        Args:
            base_dir: 基础目录，通常是 "output"
            model_name: 模型名称（steelbert, matscibert, scibert, traditionalModel）
            experiment_path: 实验路径，例如 "Steel/USTB_steel/NN_experiment"

        Returns:
            完整的输出目录路径，格式为 "output/{model_name}/{experiment_path}"

        Example:
            >>> FeatureProcessor.build_output_path("output", "steelbert", "Steel/USTB_steel/NN_cv5")
            'output/steelbert/Steel/USTB_steel/NN_cv5'
        """
        return os.path.join(base_dir, model_name, experiment_path)

    @staticmethod
    def make_composition_string_from_columns(df: pd.DataFrame, mode: str = 'wt') -> list:
        """
        将元素含量列拼接为组分字符串，如Al0.85Cu0.01...，并标注at%或wt%。
        Args:
            df: 包含元素含量的DataFrame
            mode: 'wt'或'at'，决定使用(wt%)还是(at%)列
        Returns:
            list: 每一行的组分字符串
        """
        if mode == 'wt':
            suffix = '(wt%)'
        else:
            suffix = '(at%)'
        element_cols = [col for col in df.columns if col.endswith(suffix)]
        comp_strs = []
        for _, row in df.iterrows():
            parts = []
            for col in element_cols:
                val = row[col]
                ele = col.replace(suffix, '').strip()
                if val > 0:
                    parts.append(f"{ele}{val:.4f}")
            comp_str = ''.join(parts) + (f' {mode}%')
            comp_strs.append(comp_str)
        return comp_strs

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行特征处理流程 (Feature processing pipeline)"""
        # 加载和预处理数据
        required_cols = []
        if self.use_process_embedding:
            required_cols.append(self.processing_text_column)
        required_cols.extend(self.target_columns)
        
        self.data = self.data_processor.process(required_cols)
        
        # Initialize self.X and self.y with correct indices based on self.data, or as empty DataFrames
        if self.data is None or self.data.empty:
            logger.warning("Input data is empty or None after DataProcessor.process. Cannot generate features.")
            # Return empty DataFrames with a default RangeIndex if self.data is None, or with its index if empty
            empty_index = pd.RangeIndex(start=0, stop=0) 
            return pd.DataFrame(index=empty_index, columns=pd.Index([])), pd.DataFrame(index=empty_index, columns=pd.Index([]))

        self.X = pd.DataFrame(index=self.data.index, columns=pd.Index([])) # Initialize with correct index and empty columns
        self.y = pd.DataFrame(index=self.data.index, columns=pd.Index([])) # Initialize with correct index and empty columns
        
        print(f"[DEBUG FeatureProcessor.process()] Current self.data columns: {self.data.columns.tolist()}")
        print(f"[DEBUG FeatureProcessor.process()] Current self.data is empty: {self.data.empty}")

        # 提取工艺描述嵌入向量
        if self.use_process_embedding:
            if self.processing_text_column not in self.data.columns:
                logger.warning(f"'{self.processing_text_column}' column not found in data. Skipping process embedding.")
            else:
                process_embeddings = self.embedding_manager.get_process_embeddings(
                    self.data[self.processing_text_column].tolist()
                )
                proc_emb_features = pd.DataFrame(
                    process_embeddings,
                    index=self.data.index,
                    columns=pd.Index([f"proc_emb_{i+1}" for i in range(process_embeddings.shape[1])]) # Use pd.Index
                )
                self.X = pd.concat([self.X, proc_emb_features], axis=1)
        
        # 提取元素组分嵌入向量
        if self.use_element_embedding:
            # 创建元素成分字典列表
            compositions = []
            element_cols = [col for col in self.data.columns if col.endswith(('(at%)', '(wt%)'))]
            if not element_cols:
                logger.warning("No element composition columns found (e.g., Al(wt%), Ti(at%)). Skipping element embedding.")
            else:
                for _, row in self.data.iterrows():
                    comp = {col.split('(')[0]: row[col] for col in element_cols if row[col] > 0}
                    compositions.append(comp)
                
                comp_embeddings = self.embedding_manager.get_composition_embeddings(compositions)
                comp_emb_features = pd.DataFrame(
                    comp_embeddings,
                    index=self.data.index,
                    columns=pd.Index([f"ele_emb_{i+1}" for i in range(comp_embeddings.shape[1])]) # Use pd.Index
                )
                self.X = pd.concat([self.X, comp_emb_features], axis=1)
        
        # 提取Matminer特征
        if self.use_matminer:
            logger.warning("Matminer features requested but FeatureExtractor does not support 'extract_matminer_features'. Skipping.")
            # formula = self.feature_extractor.create_formula(self.data)
            # matminer_features = self.feature_extractor.extract_matminer_features(
            #     pd.concat([self.data, formula], axis=1)
            # )
            # if not matminer_features.empty:
            #     matminer_features.columns = [f"mat_{col}" for col in matminer_features.columns]
            #     self.X = pd.concat([self.X, matminer_features], axis=1)
        
        # 提取Feature1特征
        if self.use_feature1:
            features1 = self.feature_extractor.extract_feature1_features(self.data)
            if not features1.empty:
                features1.columns = pd.Index([col for col in features1.columns]) # Use pd.Index
                self.X = pd.concat([self.X, features1], axis=1)
                
        # 提取Feature2特征
        if self.use_feature2:
            features2 = self.feature_extractor.extract_feature2_features(self.data)
            if not features2.empty:
                features2.columns = pd.Index([col for col in features2.columns]) # Use pd.Index
                self.X = pd.concat([self.X, features2], axis=1)
        
        # 提取元素比例信息
        if self.use_composition_feature:
            composition_features = self.feature_extractor.extract_composition_features(self.data)
            if not composition_features.empty:
                composition_features.columns = pd.Index([col for col in composition_features.columns]) # Use pd.Index
                self.X = pd.concat([self.X, composition_features], axis=1)
                
        # 提取其他指定特征 (Extract user-specified features)
        if self.other_features_name is not None:
            other_features = self.feature_extractor.extract_other_features(self.data, columns=self.other_features_name)
            if not other_features.empty:
                other_features.columns = pd.Index([col for col in other_features.columns]) # Use pd.Index
                self.X = pd.concat([self.X, other_features], axis=1)

        # 提取温度特征
        if self.use_temperature:
            temp_features = self.feature_extractor.extract_temperature_features(self.data)
            if not temp_features.empty:
                self.X = pd.concat([self.X, temp_features], axis=1)

        # 联合组分+工艺BERT嵌入
        if self.use_joint_composition_process_embedding:
            # 自动识别at%或wt%
            at_cols = [col for col in self.data.columns if col.endswith('(at%)')]
            wt_cols = [col for col in self.data.columns if col.endswith('(wt%)')]
            if at_cols:
                mode = 'at'
            elif wt_cols:
                mode = 'wt'
            else:
                logger.warning('No element composition columns with (at%) or (wt%) found for joint embedding. Skipping.')
                # Return empty DataFrames as no features can be generated for this path
                empty_index = self.data.index if (self.data is not None and not self.data.empty) else pd.RangeIndex(start=0, stop=0)
                return pd.DataFrame(index=empty_index, columns=pd.Index([])), pd.DataFrame(index=empty_index, columns=pd.Index([]))

            comp_strs = self.make_composition_string_from_columns(self.data, mode=mode)
            proc_strs = self.data[self.processing_text_column].astype(str).tolist()
            compositions = []
            for comp_str in comp_strs:
                comp_dict = {}
                for m in re.finditer(r'([A-Z][a-z]*)([0-9.]+)', comp_str):
                    ele, val = m.group(1), float(m.group(2))
                    comp_dict[ele] = val
                compositions.append(comp_dict)
            joint_embs = self.embedding_manager.get_batch_joint_composition_process_embeddings(
                compositions, proc_strs, use_atom_percent=(mode=='at'))
            joint_emb_features = pd.DataFrame(
                joint_embs,
                index=self.data.index,
                columns=pd.Index([f"joint_emb_{i+1}" for i in range(joint_embs.shape[1])]) # Use pd.Index
            )
            self.X = pd.concat([self.X, joint_emb_features], axis=1)

        # Initialize self.y to an empty DataFrame if no targets are found
        available_targets = [col for col in self.target_columns if col in self.data.columns]
        if available_targets:
            # Explicitly create DataFrame to satisfy linter
            self.y = pd.DataFrame(self.data[available_targets].values,
                                  index=self.data.index,
                                  columns=pd.Index(available_targets))
        else:
            # Ensure self.y is always a DataFrame, even if no target columns are specified or found
            self.y = pd.DataFrame(index=self.data.index, columns=pd.Index([])) # Use pd.Index

        # Final check to ensure self.X is a DataFrame and has the correct index
        if self.X.empty and not self.data.empty:
            self.X = pd.DataFrame(index=self.data.index, columns=pd.Index([])) # Use pd.Index
        elif self.X is None: # Should not happen with X initialized as empty DF
            self.X = pd.DataFrame(index=self.data.index, columns=pd.Index([])) # Use pd.Index
        
        # 删除包含缺失值的行
        rows_before = len(self.X)
        self.X = self.X.dropna()
        rows_after = len(self.X)
        if rows_before > rows_after:
            logger.info(f"已删除{rows_before - rows_after}行包含缺失值的数据，剩余{rows_after}行")
        # 与X对齐y，便于保存与追溯
        if not self.y.empty:
            self.y = self.y.loc[self.X.index]
        
        # 标准化特征
        if self.standardize_features:
            self.X, self.scaler = standardize_features(self.X, available_targets)
        
        # 保存特征，并额外保存与特征对齐的ID，便于后续溯源
        id_col = None
        for c in ["ID", "Id", "id"]:
            if c in self.data.columns:
                id_col = c
                break
        ids_to_save = None
        if id_col is not None:
            try:
                ids_to_save = self.data.loc[self.X.index, id_col]
            except Exception:
                ids_to_save = self.data[id_col]

        save_features(self.X, self.y, self.feature_dir, ids=ids_to_save, id_column_name=(id_col or "ID"))
        
        # 保存特征名称
        feature_info = {
            "工艺描述嵌入向量": self.use_process_embedding,
            "元素组分嵌入向量": self.use_element_embedding,
            "Matminer特征": self.use_matminer,
            "Feature1特征": self.use_feature1,
            "Feature2特征": self.use_feature2,
            "元素组分特征": self.use_composition_feature,
            "温度特征": self.use_temperature
        }
        save_feature_names(self.X, self.feature_dir, feature_info)
        
        logger.info("特征处理完成")
        return self.X, self.y 