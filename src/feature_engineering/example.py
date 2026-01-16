import os
import pandas as pd
from pathlib import Path
import sys
import argparse
import torch
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.feature_engineering.feature_processor import FeatureProcessor

def str2bool(v):
    """
    将字符串转换为布尔值，支持多种写法
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Feature processing pipeline")
    parser.add_argument('--data_path', type=str,
                        default="datasets/Ni_alloys/NiAlloy-Tm-CALPHAD-693K-descriptor_withProcessing.csv",
                        help="数据文件路径")
    parser.add_argument('--feature_dir', type=str,
                        default="Features/Ni_alloys/NiAlloy-Tm-CALPHAD-693K-descriptor_withProcessing",
                        help="特征保存目录")
    parser.add_argument('--log_dir', type=str, default=None, help="日志保存目录")
    parser.add_argument('--use_process_embedding', type=str2bool, default=True, help="是否使用工艺描述嵌入向量")
    parser.add_argument('--use_element_embedding', type=str2bool, default=True, help="是否使用元素组分嵌入向量")
    parser.add_argument('--use_feature1', type=str2bool, default=False, help="是否使用Feature1特征")
    parser.add_argument('--use_feature2', type=str2bool, default=False, help="是否使用Feature2特征")
    parser.add_argument('--use_composition_feature', type=str2bool, default=False, help="是否使用元素组分特征")
    parser.add_argument('--standardize_features', type=str2bool, default=False, help="是否标准化特征")
    parser.add_argument('--use_temperature', type=str2bool, default=False, help="是否使用温度特征")
    parser.add_argument('--batch_size', type=int, default=500, help="嵌入计算的batch size")
    parser.add_argument('--target_columns', type=str, nargs='+',
                        default=["YS(MPa)","UTS(MPa)" "El(%)"],
                        help="目标变量列名")
    parser.add_argument('--other_features_name', type=str, nargs='+', default=None, help="自定义特征列名 (可多个)")
    parser.add_argument('--use_joint_composition_process_embedding', type=str2bool, default=False, help="是否使用组分+工艺联合BERT嵌入")
    parser.add_argument('--model_name', type=str, default='steelbert',
                        choices=['steelbert', 'matscibert', 'scibert'],
                        help="BERT模型名称 (如 'steelbert', 'matscibert', 'scibert')，默认为 'steelbert'")
    return parser.parse_args()

def main():
    args = parse_args()
    # 这里可以直接用 args.xxx 访问参数
    print("数据路径:", args.data_path)
    print("特征保存目录:", args.feature_dir)
    print("日志目录:", args.log_dir)
    print("是否用工艺描述嵌入:", args.use_process_embedding)
    print("是否用组分嵌入:", args.use_element_embedding)
    print("是否用Feature1特征:", args.use_feature1)
    print("是否用Feature2特征:", args.use_feature2)
    print("是否用元素组分特征:", args.use_composition_feature)
    print("是否标准化特征:", args.standardize_features)
    print("是否用温度特征:", args.use_temperature)
    print("是否用其他特征:", args.other_features_name)
    print("batch size:", args.batch_size)
    print("目标列:", args.target_columns)

    # 设置日志目录
    log_dir = args.log_dir if args.log_dir else os.path.join(args.feature_dir, "logs")

    # 初始化特征处理器
    print(f"使用 BERT 模型: {args.model_name}")
    processor = FeatureProcessor(
        data_path=args.data_path,
        use_process_embedding=args.use_process_embedding,
        use_element_embedding=args.use_element_embedding,
        use_feature1=args.use_feature1,
        use_feature2=args.use_feature2,
        use_composition_feature=args.use_composition_feature,
        standardize_features=args.standardize_features,
        use_temperature=args.use_temperature,
        feature_dir=args.feature_dir,
        log_dir=log_dir,
        target_columns=args.target_columns,
        model_name=args.model_name,
        other_features_name=args.other_features_name,
        use_joint_composition_process_embedding=args.use_joint_composition_process_embedding
    )

    # 执行特征处理，增加处理进度打印
    # 如果用工艺描述嵌入，重载get_process_embeddings方法以打印进度
    if args.use_process_embedding:
        orig_get_process_embeddings = processor.embedding_manager.get_process_embeddings
        def get_process_embeddings_with_progress(texts, batch_size=args.batch_size):
            if not isinstance(texts, list):
                texts = [str(texts)]
            texts = [str(t) for t in texts]
            all_embeddings = []
            total = len(texts)
            processor.embedding_manager.model.eval()
            with torch.no_grad():
                for i in range(0, total, batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = processor.embedding_manager.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(processor.embedding_manager.device) for k, v in inputs.items()}
                    outputs = processor.embedding_manager.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    last_hidden_state = hidden_states[-1]
                    embeddings = last_hidden_state[:, 0, :]
                    all_embeddings.append(embeddings.cpu().numpy())
                    # 每处理100行打印一次进度
                    if (i // batch_size * batch_size) % 100 == 0:
                        print(f"Processed {min(i+batch_size, total)}/{total} process descriptions...")
            return np.concatenate(all_embeddings, axis=0)
        processor.embedding_manager.get_process_embeddings = get_process_embeddings_with_progress

    X, y = processor.process()

    # 打印特征信息
    print("\n特征处理完成！")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print("\n特征目录:", processor.feature_dir)
    print("特征已保存到:", os.path.join(processor.feature_dir, "features_with_id.csv"))
    print("目标变量已保存到:", os.path.join(processor.feature_dir, "target_with_id.csv"))
    print("特征名称已保存到:", os.path.join(processor.feature_dir, "feature_names.txt"))

if __name__ == "__main__":
    main()

# """
# ============================================================================
# 使用示例 - 支持多模型切换
# ============================================================================
#
# 可用的模型：
#   - steelbert   : 钢铁材料专用BERT模型（默认）
#   - matscibert  : 材料科学通用BERT模型
#   - scibert     : 科学文献BERT模型
#
# 注意：
#   1. 首次使用某个模型时，会自动生成并缓存元素嵌入向量
#   2. 后续使用会直接从缓存加载，显著提升性能
#   3. 不同模型的缓存独立存储在 element_embeddings/{model_name}/ 目录下
# ============================================================================

# ============================================================================
# 推荐：使用新的多模型目录结构
# 格式：Features/{model_name}/{dataset_path}/
# 便于管理和比较不同模型的特征
# ============================================================================

# 示例 1: 使用 SteelBERT 模型（推荐新路径结构）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Al_Alloys\USTB\USTB_Al_alloys_processed_withID.csv" `
#     --feature_dir Features\steelbert\Al_Alloys\USTB_new\all_features `
#     --log_dir Features\steelbert\Al_Alloys\USTB_new\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --use_joint_composition_process_embedding False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)"

# 示例 2: 使用 MatSciBERT 模型（推荐新路径结构）
# python src/feature_engineering/example.py `
#     --model_name matscibert `
#     --data_path "datasets\Al_Alloys\USTB\USTB_Al_alloys_processed_withID.csv" `
#     --feature_dir Features\matscibert\Al_Alloys\USTB_new\all_features `
#     --log_dir Features\matscibert\Al_Alloys\USTB_new\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)"

# 示例 3: 使用 SciBERT 模型（推荐新路径结构）
# python src/feature_engineering/example.py `
#     --model_name scibert `
#     --data_path "datasets\Al_Alloys\USTB\USTB_Al_alloys_processed_withID.csv" `
#     --feature_dir Features\scibert\Al_Alloys\USTB_new\all_features `
#     --log_dir Features\scibert\Al_Alloys\USTB_new\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)"

# 示例 4: 传统特征（不使用 BERT 嵌入）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Al_Alloys\USTB\USTB_Al_alloys_processed_withID.csv" `
#     --feature_dir Features\traditionalModel\Al_Alloys\USTB_new\composition_features `
#     --log_dir Features\traditionalModel\Al_Alloys\USTB_new\composition_features\logs `
#     --use_process_embedding False `
#     --use_element_embedding False `
#     --use_composition_feature True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)"

# ============================================================================
# 更多数据集示例
# ============================================================================

# 高熵合金数据集（训练集）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\HEA_data\RoomTemperature_HEA_train_with_ID.csv" `
#     --feature_dir Features/HEA_data/yasir_data_half/ID/all_features `
#     --log_dir Features/HEA_data/yasir_data_half/ID/all_features/logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature True `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 高熵合金数据集（完整）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path   "datasets\HEA_data\RoomTemperature_HEA_withID.csv" `
#     --feature_dir Features/HEA_data/yasir_data/ID/all_features `
#     --log_dir Features/HEA_data/yasir_data/ID/all_features/logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 钛合金数据集
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" `
#     --feature_dir Features/Ti_alloys/Xue/ID/all_features `
#     --log_dir Features/Ti_alloys/Xue/ID/all_features/logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "El(%)"

# 铌合金数据集（包含温度特征和联合嵌入）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Nb_Alloys\Nb_cleandata\Nb_clean_with_processing_sequence_withID.csv" `
#     --feature_dir Features/Nb_Alloys/Nb_cleandata/ID/all_features `
#     --log_dir Features/Nb_Alloys/Nb_cleandata/ID/all_features/logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature True `
#     --use_joint_composition_process_embedding True `
#     --standardize_features False `
#     --use_temperature True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 钢铁数据集
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Steel\USTB_steel_processed_withID.csv" `
#     --feature_dir Features\Steel\USTB_steel\all_features_withID `
#     --log_dir Features\Steel\USTB_steel\all_features_withID\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --use_joint_composition_process_embedding False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 铌合金数据集 - 高温压缩强度（仅元素嵌入）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets/Nb_Alloys/Harbin/HTC_processed_withID.csv" `
#     --feature_dir Features/Nb_Alloys/Harbin/HTC/all_features_withID `
#     --log_dir Features/Nb_Alloys/Harbin/HTC/all_features_withID/logs `
#     --use_process_embedding False `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --use_joint_composition_process_embedding False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "high-temperature compressive strength(MPa)"

# 铌合金数据集 - 断裂韧性（仅元素嵌入）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets/Nb_Alloys/Harbin/KQ_processed_withID.csv" `
#     --feature_dir Features/Nb_Alloys/Harbin/KQ/all_features_withID `
#     --log_dir Features/Nb_Alloys/Harbin/KQ/all_features_withID/logs `
#     --use_process_embedding False `
#     --use_element_embedding True `
#     --use_feature1 False `
#     --use_feature2 False `
#     --use_composition_feature False `
#     --use_joint_composition_process_embedding False `
#     --standardize_features False `
#     --use_temperature False `
#     --batch_size 2048 `
#     --target_columns "KQ(MPa·m^(1/2))"

# ============================================================================
# 模型比较示例 - 使用不同模型处理相同数据集（推荐新路径结构）
# 注意：路径中包含模型名称，便于后续比较不同模型的效果
# ============================================================================

# 使用 SteelBERT 处理钢铁数据
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Steel\USTB_steel_processed_withID.csv" `
#     --feature_dir Features\steelbert\Steel\USTB_steel\all_features `
#     --log_dir Features\steelbert\Steel\USTB_steel\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 使用 MatSciBERT 处理钢铁数据（对比）
# python src/feature_engineering/example.py `
#     --model_name matscibert `
#     --data_path "datasets\Steel\USTB_steel_processed_withID.csv" `
#     --feature_dir Features\matscibert\Steel\USTB_steel\all_features `
#     --log_dir Features\matscibert\Steel\USTB_steel\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 使用 SciBERT 处理钢铁数据（对比）
# python src/feature_engineering/example.py `
#     --model_name scibert `
#     --data_path "datasets\Steel\USTB_steel_processed_withID.csv" `
#     --feature_dir Features\scibert\Steel\USTB_steel\all_features `
#     --log_dir Features\scibert\Steel\USTB_steel\all_features\logs `
#     --use_process_embedding True `
#     --use_element_embedding True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# 使用传统特征作为基线（对比）
# python src/feature_engineering/example.py `
#     --model_name steelbert `
#     --data_path "datasets\Steel\USTB_steel_processed_withID.csv" `
#     --feature_dir Features\traditionalModel\Steel\USTB_steel\composition_features `
#     --log_dir Features\traditionalModel\Steel\USTB_steel\composition_features\logs `
#     --use_process_embedding False `
#     --use_element_embedding False `
#     --use_composition_feature True `
#     --batch_size 2048 `
#     --target_columns "UTS(MPa)" "YS(MPa)" "El(%)"

# ============================================================================
# 缓存管理命令
# ============================================================================
#
# 查看所有模型的缓存状态：
#   python scripts/test_multi_model_cache.py
#
# 为特定模型生成元素嵌入向量缓存：
#   python src/feature_engineering/generate_element_embeddings.py --model_name steelbert
#   python src/feature_engineering/generate_element_embeddings.py --model_name matscibert
#   python src/feature_engineering/generate_element_embeddings.py --model_name scibert
#
# 强制重新生成缓存：
#   python src/feature_engineering/generate_element_embeddings.py --model_name steelbert --force
#
# 迁移旧版本的元素嵌入文件：
#   python scripts/migrate_element_embeddings.py
#
# 自动设置所有模型（包括迁移）：
#   python scripts/auto_setup_models.py
#
# ============================================================================
# """