"""
BERT 模型加载器
支持从本地路径加载不同的 BERT 模型（SteelBERT、MatSciBERT、SciBERT 等）
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


class BERTModelLoader:
    """
    BERT 模型加载器类
    支持通过配置切换不同的本地 BERT 模型
    """
    
    # 模型配置字典：映射模型名称到本地路径
    MODEL_CONFIGS = {
        "steelbert": {
            "path": "./models/steelbert",
            "description": "SteelBERT - 专门针对钢铁材料训练的 BERT 模型"
        },
        "matscibert": {
            "path": "./models/matscibert",
            "description": "MatSciBERT - 材料科学领域的 BERT 模型"
        },
        "scibert": {
            "path": "./models/scibert",
            "description": "SciBERT - 科学文献领域的 BERT 模型"
        }
    }
    
    # 兼容旧路径的映射
    LEGACY_PATHS = {
        "./SteelBERTmodel": "steelbert"
    }
    
    def __init__(self, project_root: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            project_root: 项目根目录路径，用于解析相对路径。如果为 None，使用当前工作目录
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        logger.info(f"BERTModelLoader 初始化，项目根目录: {self.project_root}")
    
    def _resolve_path(self, path: str) -> Path:
        """
        解析路径为绝对路径
        
        Args:
            path: 相对或绝对路径
            
        Returns:
            Path: 绝对路径对象
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.project_root / path_obj).resolve()
    
    def _validate_model_path(self, model_path: Path) -> bool:
        """
        验证模型路径是否存在且包含必要文件
        
        Args:
            model_path: 模型目录路径
            
        Returns:
            bool: 路径是否有效
        """
        if not model_path.exists():
            logger.error(f"模型路径不存在: {model_path}")
            return False
        
        if not model_path.is_dir():
            logger.error(f"模型路径不是目录: {model_path}")
            return False
        
        # 检查必要的模型文件
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        # 检查 config.json
        if not (model_path / "config.json").exists():
            logger.error(f"缺少 config.json 文件: {model_path}")
            return False
        
        # 检查至少有一个模型权重文件
        has_model_file = any((model_path / f).exists() for f in model_files)
        if not has_model_file:
            logger.error(f"缺少模型权重文件 (pytorch_model.bin 或 model.safetensors): {model_path}")
            return False
        
        logger.info(f"模型路径验证通过: {model_path}")
        return True
    
    def _auto_migrate_legacy_model(self, legacy_path: Path, target_path: Path) -> bool:
        """
        自动迁移旧模型到新位置

        Args:
            legacy_path: 旧模型路径
            target_path: 新模型路径

        Returns:
            bool: 是否成功迁移
        """
        try:
            import shutil

            logger.info(f"检测到旧模型路径: {legacy_path}")
            logger.info(f"准备自动迁移到: {target_path}")

            # 验证旧路径
            if not self._validate_model_path(legacy_path):
                logger.warning(f"旧路径验证失败，跳过迁移: {legacy_path}")
                return False

            # 检查目标路径是否已存在
            if target_path.exists():
                logger.info(f"目标路径已存在，跳过迁移: {target_path}")
                return True

            # 创建目标目录的父目录
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 移动模型
            logger.info("正在移动模型文件...")
            shutil.move(str(legacy_path), str(target_path))
            logger.info(f"✓ 模型已成功迁移到: {target_path}")

            return True

        except Exception as e:
            logger.error(f"自动迁移失败: {str(e)}")
            logger.info("请手动运行迁移脚本: python scripts/migrate_steelbert.py")
            return False

    def get_model_path(self, model_name: str) -> str:
        """
        根据模型名称获取本地路径

        Args:
            model_name: 模型名称（如 "steelbert", "matscibert"）或直接的路径

        Returns:
            str: 模型的本地路径

        Raises:
            ValueError: 如果模型名称未配置或路径无效
        """
        # 检查是否是旧路径格式
        if model_name in self.LEGACY_PATHS:
            logger.warning(f"检测到旧路径格式 '{model_name}'，自动映射到 '{self.LEGACY_PATHS[model_name]}'")

            # 尝试自动迁移
            legacy_path = self._resolve_path(model_name)
            target_model_name = self.LEGACY_PATHS[model_name]
            target_config = self.MODEL_CONFIGS[target_model_name]
            target_path = self._resolve_path(target_config["path"])

            # 如果旧路径存在且新路径不存在，执行自动迁移
            if legacy_path.exists() and not target_path.exists():
                logger.info("=" * 60)
                logger.info("检测到需要迁移的旧模型")
                logger.info("=" * 60)
                if self._auto_migrate_legacy_model(legacy_path, target_path):
                    logger.info("=" * 60)
                    logger.info("✓ 自动迁移完成")
                    logger.info("=" * 60)
                else:
                    logger.warning("自动迁移失败，将尝试使用旧路径")

            model_name = target_model_name

        # 如果是已配置的模型名称
        if model_name.lower() in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model_name.lower()]
            model_path = self._resolve_path(config["path"])

            # 如果配置的路径不存在，检查是否有旧路径可用
            if not model_path.exists():
                # 检查旧路径
                for legacy_path_str, legacy_model_name in self.LEGACY_PATHS.items():
                    if legacy_model_name == model_name.lower():
                        legacy_path = self._resolve_path(legacy_path_str)
                        if legacy_path.exists() and self._validate_model_path(legacy_path):
                            logger.warning(f"配置路径不存在，使用旧路径: {legacy_path}")
                            return str(legacy_path)

            if not self._validate_model_path(model_path):
                raise ValueError(
                    f"模型路径无效: {model_path}\n"
                    f"请运行以下命令之一:\n"
                    f"  - 下载模型: python scripts/download_matscibert.py\n"
                    f"  - 迁移模型: python scripts/migrate_steelbert.py"
                )

            logger.info(f"使用配置的模型: {config['description']}")
            logger.info(f"模型路径: {model_path}")
            return str(model_path)

        # 如果是直接路径
        model_path = self._resolve_path(model_name)
        if self._validate_model_path(model_path):
            logger.info(f"使用自定义路径: {model_path}")
            return str(model_path)

        raise ValueError(
            f"无效的模型名称或路径: '{model_name}'\n"
            f"支持的模型名称: {list(self.MODEL_CONFIGS.keys())}\n"
            f"或提供有效的本地模型路径"
        )
    
    def load_model(self, model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """
        加载指定的 BERT 模型和 tokenizer
        
        Args:
            model_name: 模型名称（如 "steelbert", "matscibert"）或本地路径
            
        Returns:
            Tuple[PreTrainedTokenizer, PreTrainedModel]: (tokenizer, model) 元组
            
        Raises:
            RuntimeError: 如果模型加载失败
        """
        try:
            model_path = self.get_model_path(model_name)
            
            logger.info(f"正在从本地加载模型: {model_path}")
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            logger.info("Tokenizer 加载成功")
            
            # 加载模型
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True
            )
            logger.info("模型加载成功")
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise RuntimeError(f"无法加载模型 '{model_name}': {str(e)}")
    
    @classmethod
    def list_available_models(cls) -> dict:
        """
        列出所有可用的预配置模型
        
        Returns:
            dict: 模型配置字典
        """
        return cls.MODEL_CONFIGS.copy()
    
    @staticmethod
    def download_model_from_huggingface(
        model_id: str,
        save_path: str,
        force_download: bool = False
    ) -> None:
        """
        从 Hugging Face 下载模型到本地
        
        Args:
            model_id: Hugging Face 模型 ID（如 "m3rg-iitd/matscibert"）
            save_path: 本地保存路径
            force_download: 是否强制重新下载
            
        Raises:
            RuntimeError: 如果下载失败
        """
        try:
            save_path_obj = Path(save_path)
            
            # 检查路径是否已存在
            if save_path_obj.exists() and not force_download:
                logger.warning(f"目标路径已存在: {save_path}，跳过下载")
                logger.info("如需重新下载，请设置 force_download=True")
                return
            
            # 创建目录
            save_path_obj.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"正在从 Hugging Face 下载模型: {model_id}")
            logger.info(f"保存路径: {save_path}")
            
            # 下载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(save_path)
            logger.info("Tokenizer 下载完成")
            
            # 下载模型
            model = AutoModel.from_pretrained(model_id)
            model.save_pretrained(save_path)
            logger.info("模型下载完成")
            
            logger.info(f"模型已成功下载到: {save_path}")
            
        except Exception as e:
            logger.error(f"下载模型时出错: {str(e)}")
            raise RuntimeError(f"无法下载模型 '{model_id}': {str(e)}")

