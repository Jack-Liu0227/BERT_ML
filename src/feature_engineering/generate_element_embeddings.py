import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_engineering.embedding_manager import EmbeddingManager
from src.feature_engineering.utils import setup_logging, ensure_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成元素嵌入向量")
    parser.add_argument(
        '--model_name',
        type=str,
        default='steelbert',
        choices=['steelbert', 'matscibert', 'scibert'],
        help='BERT模型名称（如 steelbert, matscibert, scibert），默认为 steelbert'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成（即使缓存已存在）'
    )
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()

    # 设置日志
    log_dir = "logs"
    ensure_dir(log_dir)
    setup_logging(log_name="element_embeddings.log", log_dir=log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("元素嵌入向量生成工具")
    logger.info("=" * 60)
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"强制重新生成: {args.force}")
    logger.info("")

    # 确定保存路径
    save_dir = f"element_embeddings/{args.model_name}"
    embeddings_file = os.path.join(save_dir, "element_embeddings.npy")

    # 检查是否已存在
    if os.path.exists(embeddings_file) and not args.force:
        logger.info(f"元素嵌入向量缓存已存在: {embeddings_file}")
        logger.info("如需重新生成，请使用 --force 参数")
        logger.info("")

        # 显示现有缓存信息
        try:
            embeddings = np.load(embeddings_file, allow_pickle=True).item()
            logger.info(f"缓存信息:")
            logger.info(f"  - 元素数量: {len(embeddings)}")
            logger.info(f"  - 嵌入维度: {next(iter(embeddings.values())).shape[0]}")
            logger.info("")
        except Exception as e:
            logger.warning(f"读取缓存信息失败: {str(e)}")

        return

    # 初始化嵌入向量管理器
    logger.info("初始化嵌入向量管理器...")
    logger.info(f"使用模型: {args.model_name}")
    manager = EmbeddingManager(model_name=args.model_name)
    
    # 注意：EmbeddingManager 初始化时会自动生成元素嵌入向量
    # 这里只需要确认生成完成
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 元素嵌入向量生成完成")
    logger.info("=" * 60)
    logger.info(f"保存位置: {embeddings_file}")
    logger.info(f"元素数量: {len(manager.element_embeddings)}")
    logger.info("")

    # 创建可视化数据（可选）
    try:
        if len(manager.element_embeddings) > 0:
            # 使用PCA降维到2维进行可视化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)

            # 准备数据
            elements_list = []
            embeddings_list = []
            for element, embedding in manager.element_embeddings.items():
                elements_list.append(element)
                embeddings_list.append(embedding)

            # 降维
            embeddings_2d = pca.fit_transform(embeddings_list)

            # 创建可视化数据文件
            viz_file = os.path.join(save_dir, "element_embeddings_2d.csv")
            viz_data = pd.DataFrame({
                'Element': elements_list,
                'PC1': embeddings_2d[:, 0],
                'PC2': embeddings_2d[:, 1]
            })
            viz_data.to_csv(viz_file, index=False, encoding='utf-8')
            logger.info(f"✓ 2D可视化数据已保存到: {viz_file}")
    except Exception as e:
        logger.warning(f"创建可视化数据时出错: {str(e)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("所有操作完成！")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 