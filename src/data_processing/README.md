# Alloy Data Processing

This module provides the core data splitting and OOD (Out-of-Distribution) partitioning architecture for alloy property prediction.

## 模块结构

```
data_processing/
├── canonical_split_registry.py                     # 标准划分定义与注册表
├── ood_split_cache.py                              # OOD 划分落盘缓存管理
├── strength_ood_registry.py                        # OOD 处理器工厂注册表
├── strength_ood_common.py                          # 划分基础数据结构 (PreparedFold, PreparedSplit)
├── strength_extrapolation_data_processor.py        # 极值目标外推划分
├── strength_sparse_x_single_data_processor.py      # X 空间单样本稀疏划分
├── strength_sparse_x_cluster_data_processor.py     # X 空间聚类稀疏划分
├── strength_sparse_y_single_data_processor.py      # Y 空间单样本稀疏划分
├── strength_sparse_y_cluster_data_processor.py     # Y 空间聚类稀疏划分
├── strength_loco_data_processor.py                 # 留一聚类划分 (Leave-One-Cluster-Out)
├── strength_random_cv_baseline_data_processor.py   # 同分布随机对照交叉验证
├── strength_hybrid_extrapolation_data_processor.py # 混合外推划分
├── utils.py                                        # 材料物理与成分工具函数
└── constants.py                                    # 物理常量与单位范围
```

## 支持的划分策略 (Split Strategies)

1. **`target_extrapolation`**：目标极值（如按高温或高强度上下分位数）划分。
2. **`sparse_x_single` / `sparse_x_cluster`**：特征空间（X-space）单点或聚类稀疏采样。
3. **`sparse_y_single` / `sparse_y_cluster`**：目标空间（Y-space）单点或聚类稀疏采样。
4. **`loco`**：留一聚类划分（Leave-One-Cluster-Out）。
5. **`random_cv_baseline`**：随机多折交叉验证基线对照。
6. **`hybrid_extrapolation_*`**：混合极值与多重 OOD 划分。
