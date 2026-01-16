import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DebertaV2Model
import logging
from pathlib import Path
from .bert_model_loader import BERTModelLoader

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """管理元素和工艺描述的embedding"""

    def __init__(self, model_name: str = "steelbert", embeddings_path=None):
        """
        初始化embedding管理器

        Args:
            model_name: 模型名称（如 "steelbert", "matscibert", "scibert"），默认为 "steelbert"
            embeddings_path: 预生成的元素嵌入向量文件路径（可选，默认自动根据模型名称确定）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # 自动确定嵌入向量文件路径
        if embeddings_path is None:
            self.embeddings_path = self._get_default_embeddings_path(model_name)
        else:
            self.embeddings_path = embeddings_path

        logger.info(f"使用模型名称: {model_name}")
        logger.info(f"元素嵌入向量路径: {self.embeddings_path}")

        # 初始化模型加载器
        self.model_loader = BERTModelLoader()

        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self.hidden_size = 768  # 默认BERT模型隐藏层大小

        # 加载模型
        self._load_model()

        # 加载预生成的元素嵌入向量
        self.element_embeddings = self._load_element_embeddings()

    def _get_default_embeddings_path(self, model_name: str) -> str:
        """
        根据模型名称获取默认的嵌入向量文件路径

        Args:
            model_name: 模型名称

        Returns:
            str: 嵌入向量文件路径
        """
        # 新的路径结构：element_embeddings/{model_name}/element_embeddings.npy
        new_path = f"element_embeddings/{model_name}/element_embeddings.npy"

        # 向后兼容：检查旧路径
        legacy_path = "element_embeddings/element_embeddings.npy"

        # 如果是 steelbert 且旧路径存在，使用旧路径（向后兼容）
        if model_name == "steelbert" and os.path.exists(legacy_path) and not os.path.exists(new_path):
            logger.info(f"检测到旧版本的元素嵌入文件: {legacy_path}")
            logger.info(f"建议迁移到新路径: {new_path}")
            return legacy_path

        return new_path

    def _migrate_legacy_embeddings(self):
        """
        迁移旧版本的元素嵌入文件到新的目录结构
        """
        legacy_path = "element_embeddings/element_embeddings.npy"
        new_path = "element_embeddings/steelbert/element_embeddings.npy"

        if os.path.exists(legacy_path) and not os.path.exists(new_path):
            try:
                logger.info("正在迁移旧版本的元素嵌入文件...")
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                import shutil
                shutil.copy2(legacy_path, new_path)
                logger.info(f"✓ 迁移完成: {legacy_path} -> {new_path}")
                logger.info(f"原文件保留在: {legacy_path}")
            except Exception as e:
                logger.warning(f"迁移失败: {str(e)}")

    def _load_model(self):
        """加载NLP模型，支持多卡"""
        logger.info("正在加载预训练语言模型...")
        try:
            # 使用模型加载器加载模型
            logger.info(f"使用 BERTModelLoader 加载模型: {self.model_name}")
            self.tokenizer, self.model = self.model_loader.load_model(self.model_name)

            # 将模型移到设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)

            # 多卡支持
            if torch.cuda.device_count() > 1:
                logger.info(f"检测到 {torch.cuda.device_count()} 张GPU，启用DataParallel")
                self.model = torch.nn.DataParallel(self.model)

            self.hidden_size = self.model.module.config.hidden_size if hasattr(self.model, "module") else self.model.config.hidden_size
            logger.info(f"语言模型加载完成，使用设备: {self.device}, 隐藏层大小: {self.hidden_size}")

        except Exception as e:
            logger.error(f"加载语言模型时出错: {str(e)}")
            raise RuntimeError(f"无法加载模型: {str(e)}")
    
    def _load_element_embeddings(self):
        """加载预生成的元素embedding，如果不存在则自动生成"""
        try:
            # 检查文件是否存在
            if os.path.exists(self.embeddings_path):
                logger.info(f"从缓存加载元素嵌入向量: {self.embeddings_path}")
                embeddings = np.load(self.embeddings_path, allow_pickle=True).item()
                logger.info(f"✓ 成功加载 {len(embeddings)} 个元素的嵌入向量（来自缓存）")
                return embeddings

            # 文件不存在，自动生成
            logger.warning(f"元素嵌入向量缓存不存在: {self.embeddings_path}")
            logger.info(f"正在为模型 '{self.model_name}' 自动生成元素嵌入向量...")

            embeddings = self._generate_element_embeddings()

            # 保存到缓存
            self._save_element_embeddings(embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"加载/生成元素嵌入向量时出错: {str(e)}")
            return {}

    def _generate_element_embeddings(self):
        """
        自动生成所有元素的嵌入向量

        Returns:
            dict: 元素名称到嵌入向量的映射
        """
        logger.info("=" * 60)
        logger.info("开始生成元素嵌入向量")
        logger.info("=" * 60)

        # 周期表所有118种元素
        elements = [
            'H',  'He',
            'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
            'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I',  'Xe',
            'Cs', 'Ba',
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra',
            'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'
        ]

        embeddings_dict = {}
        total = len(elements)

        for idx, element in enumerate(elements, 1):
            try:
                embedding = self._get_element_embedding(element)
                embeddings_dict[element] = embedding

                # 每10个元素打印一次进度
                if idx % 10 == 0 or idx == total:
                    logger.info(f"进度: {idx}/{total} ({idx*100//total}%) - 最新: {element}")

            except Exception as e:
                logger.error(f"生成元素 {element} 的嵌入向量时出错: {str(e)}")

        logger.info("=" * 60)
        logger.info(f"✓ 元素嵌入向量生成完成，共 {len(embeddings_dict)} 个元素")
        logger.info("=" * 60)

        return embeddings_dict

    def _save_element_embeddings(self, embeddings_dict):
        """
        保存元素嵌入向量到文件

        Args:
            embeddings_dict: 元素名称到嵌入向量的映射
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

            # 保存
            np.save(self.embeddings_path, embeddings_dict)
            logger.info(f"✓ 元素嵌入向量已保存到: {self.embeddings_path}")

            # 保存信息文件
            info_path = self.embeddings_path.replace('.npy', '_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"模型名称: {self.model_name}\n")
                f.write(f"元素数量: {len(embeddings_dict)}\n")
                f.write(f"嵌入维度: {next(iter(embeddings_dict.values())).shape[0]}\n")
                f.write(f"\n包含的元素:\n")
                for element in sorted(embeddings_dict.keys()):
                    f.write(f"  - {element}\n")

            logger.info(f"✓ 信息文件已保存到: {info_path}")

        except Exception as e:
            logger.error(f"保存元素嵌入向量时出错: {str(e)}")
    
    def get_element_embedding(self, element):
        """获取元素的embedding"""
        if element in self.element_embeddings:
            return self.element_embeddings[element]
        else:
            logger.warning(f"元素 {element} 的嵌入向量不存在，尝试生成...")
            try:
                embedding = self._get_element_embedding(element)
                self.element_embeddings[element] = embedding
                return embedding
            except Exception as e:
                logger.error(f"生成元素 {element} 的嵌入向量时出错: {str(e)}")
                raise ValueError(f"无法获取元素 {element} 的嵌入向量")
    
    def _get_element_embedding(self, element):
        """获取单个元素的embedding"""
        inputs = self.tokenizer(element, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            element_embedding = last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return element_embedding
    
    def get_process_embeddings(self, texts, batch_size=500):
        """
        获取工艺描述的embedding，分批处理，支持多卡
        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数
        Returns:
            numpy.ndarray: shape=(len(texts), hidden_size)
        """
        if not isinstance(texts, list):
            texts = [texts]
        texts = [str(text) for text in texts]
        all_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                embeddings = last_hidden_state[:, 0, :]  # shape: (batch, hidden)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)
    
    def get_composition_embeddings(self, compositions):
        """
        计算合金成分的embedding
        
        Args:
            compositions: 包含元素及其含量的字典列表
                        例如: [{'Al': 90, 'Cu': 10}, {'Fe': 80, 'Ni': 20}]
        
        Returns:
            numpy.ndarray: 成分embedding数组
        """
        embeddings = []
        
        for comp in compositions:
            element_weighted_embeddings = []
            
            for element, percentage in comp.items():
                if percentage > 0:
                    element_embedding = self.get_element_embedding(element)
                    weighted_embedding = element_embedding * (percentage / 100)
                    element_weighted_embeddings.append(weighted_embedding)
            
            if element_weighted_embeddings:
                comp_embedding = np.sum(element_weighted_embeddings, axis=0)
            else:
                comp_embedding = np.zeros(self.hidden_size)
            
            embeddings.append(comp_embedding)
        
        return np.array(embeddings)
    
    # def get_joint_composition_process_embedding(self, composition: dict, process: str, use_atom_percent: bool = True) -> np.ndarray:
    #     """
    #     Get a joint embedding by concatenating the composition and process description as a single sentence.
    #     Args:
    #         composition (dict): e.g. {'Al': 90, 'Cu': 10}
    #         process (str): process description
    #         use_atom_percent (bool): If True, use atom percent; else use weight percent
    #     Returns:
    #         np.ndarray: embedding vector for the joint description
    #     """
    #     # Build composition description
    #     comp_items = []
    #     for element, percent in composition.items():
    #         if percent > 0:
    #             comp_items.append(f"{element} {percent:.2f}")
    #     comp_str = ''.join(comp_items)
    #     if use_atom_percent:
    #         comp_desc = f"The atom percent composition is {comp_str}."
    #     else:
    #         comp_desc = f"The weight percent composition is {comp_str}."
    #     # Build process description
    #     process_desc = f"The processing is {process}."
    #     # Combine
    #     joint_text = comp_desc + ' ' + process_desc
    #     # Tokenize and get embedding
    #     self.model.eval()
    #     inputs = self.tokenizer(joint_text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     with torch.no_grad():
    #         outputs = self.model(**inputs, output_hidden_states=True)
    #         hidden_states = outputs.hidden_states
    #         last_hidden_state = hidden_states[-1]
    #         embedding = last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    #     return embedding
    
    def get_batch_joint_composition_process_embeddings(self, compositions: list, processes: list, use_atom_percent: bool = True) -> np.ndarray:
        """
        Get joint embeddings for a batch of (composition, process) pairs, with scientific-paper style English description.
        Args:
            compositions (list of dict): e.g. [{'Al': 90, 'Cu': 10}, ...]
            processes (list of str): process description for each sample
            use_atom_percent (bool): If True, use at.%; else use wt.%
        Returns:
            np.ndarray: shape=(len(compositions), hidden_size)
        """
        assert len(compositions) == len(processes), "compositions and processes must have the same length"
        unit = "at.%" if use_atom_percent else "wt.%"
        all_texts = []
        for comp, proc in zip(compositions, processes):
            comp_items = []
            for element, percent in comp.items():
                if percent > 0:
                    comp_items.append(f"{element}{percent}")
            comp_str = '-'.join(comp_items)+f" ({unit})"
            comp_desc = f"The nominal composition of the alloy is: {comp_str}."
            process_desc = f"The processing route is: {proc}."
            joint_text = comp_desc + ' ' + process_desc
            print(joint_text)
            all_texts.append(joint_text)
        # Tokenize and get embeddings in batch
        self.model.eval()
        inputs = self.tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            embeddings = last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings 