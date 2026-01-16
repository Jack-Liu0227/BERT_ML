# 端到端机器学习流水线 / End-to-End ML Pipeline

## 概述 / Overview

这是一个综合性的端到端机器学习流水线，整合了特征工程、模型训练（传统ML和神经网络）、评估为统一工作流。

This is a comprehensive end-to-end machine learning pipeline that integrates feature engineering, model training (traditional ML and neural networks), and evaluation into a unified workflow.

## 主要特性 / Key Features

- ✅ **特征生成** / Feature Generation
  - 组分特征 / Composition features
  - 元素嵌入（SciBERT, SteelBERT, MatSciBERT）/ Element embeddings
  - 工艺嵌入 / Process embeddings
  - 温度特征 / Temperature features

- ✅ **模型训练** / Model Training
  - 传统ML模型：XGBoost, Random Forest, MLP, LightGBM, CatBoost
  - 神经网络模型：AlloyNN
  - 交叉验证 / Cross-validation
  - Optuna超参数优化 / Hyperparameter optimization

- ✅ **模型评估** / Model Evaluation
  - 标准评估指标（R², RMSE, MAE）/ Standard metrics
  - SHAP分析（传统ML）/ SHAP analysis
  - 预测图表 / Prediction plots

- ✅ **智能特性** / Smart Features
  - 自动检测特征文件，跳过重复生成 / Auto-detect feature files
  - 自动推断合金类型和数据集名称 / Auto-infer alloy type and dataset name
  - 标准化目录结构 / Standardized directory structure
  - 向后兼容现有工作流 / Backward compatible with existing workflows

## 快速开始 / Quick Start

### 示例 1: 传统ML模型 + 组分特征

```bash
python -m src.pipelines.end_to_end_pipeline \
    --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
    --result_dir "output/results/Ti_alloys/Xue/tradition/" \
    --target_columns "UTS(MPa)" "El(%)" \
    --processing_cols "Solution Temperature(℃)" "Solution Time(h)" "Aging Temperature(℃)" "Aging Time(h)" \
    --models xgboost sklearn_rf lightgbm \
    --use_composition_feature True \
    --embedding_type tradition \
    --cross_validate --num_folds 9 \
    --evaluate_after_train \
    --use_optuna --n_trials 50
```

### 示例 2: 神经网络 + 嵌入特征

```bash
python -m src.pipelines.end_to_end_pipeline \
    --data_file "Features/Steel/USTB_steel/matscibert/features_with_id.csv" \
    --result_dir "output/results/Steel/USTB_steel/matscibert/NN_opt" \
    --target_columns "UTS(MPa)" "YS(MPa)" "El(%)" \
    --use_nn \
    --use_element_embedding True \
    --use_process_embedding True \
    --embedding_type matscibert \
    --cross_validate --num_folds 9 \
    --epochs 200 --batch_size 256 \
    --evaluate_after_train \
    --use_optuna --n_trials 50
```

## 参数说明 / Parameters

### 必需参数 / Required

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_file` | 输入数据文件路径 | `datasets/Ti_alloys/data.csv` |
| `--result_dir` | 结果输出目录 | `output/results/Ti_alloys/` |
| `--target_columns` | 目标预测列名 | `"UTS(MPa)" "El(%)"` |
| `--embedding_type` | 嵌入类型 | `tradition/scibert/steelbert/matscibert` |

### 模型选择 / Model Selection (二选一)

| 参数 | 说明 | 示例 |
|------|------|------|
| `--models` | 传统ML模型列表 | `xgboost sklearn_rf mlp lightgbm catboost` |
| `--use_nn` | 使用神经网络模型 | (flag) |

### 特征配置 / Feature Configuration

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_composition_feature` | 使用组分特征 | `False` |
| `--use_element_embedding` | 使用元素嵌入 | `False` |
| `--use_process_embedding` | 使用工艺嵌入 | `False` |
| `--use_temperature` | 使用温度特征 | `False` |
| `--processing_cols` | 处理参数列名 | `[]` |

### 训练配置 / Training Configuration

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cross_validate` | 启用交叉验证 | `False` |
| `--num_folds` | 交叉验证折数 | `9` |
| `--test_size` | 测试集比例 | `0.2` |
| `--random_state` | 随机种子 | `42` |

### 神经网络参数 / Neural Network

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 最大训练轮数 | `200` |
| `--patience` | 早停耐心值 | `200` |
| `--batch_size` | 训练批次大小 | `256` |

### 优化配置 / Optimization

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_optuna` | 启用Optuna优化 | `False` |
| `--n_trials` | Optuna试验次数 | `50` |

### 评估配置 / Evaluation

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--evaluate_after_train` | 训练后评估 | `False` |
| `--run_shap_analysis` | 运行SHAP分析 | `False` |

## 目录结构 / Directory Structure

### 输入特征目录 / Input Features
```
Features/
└── {alloy_type}/           # 合金类型 (Ti, Al, HEA, Nb, Steel)
    └── {dataset_name}/     # 数据集名称
        ├── tradition/      # Traditional composition features only
        ├── scibert/        # Features with SciBERT embeddings
        ├── steelbert/      # Features with SteelBERT embeddings
        └── matscibert/     # Features with MatSciBERT embeddings
            ├── features_with_id.csv
            ├── target_with_id.csv
            └── feature_names.txt
```

### 输出结果目录 / Output Results
```
output/results/
└── {alloy_type}/
    └── {dataset_name}/
        └── {embedding_type}/
            ├── models/
            ├── predictions/
            ├── evaluations/
            └── logs/
```

## 工作流程 / Workflow

1. **参数验证** / Parameter Validation
   - 检查参数有效性和文件存在性

2. **特征生成** / Feature Generation
   - 自动检测输入是否为特征文件
   - 如果是原始数据，生成特征并保存

3. **模型训练** / Model Training
   - 传统ML：多模型对比、交叉验证、Optuna优化
   - 神经网络：交叉验证、Optuna优化、早停

4. **模型评估** / Model Evaluation
   - 计算评估指标（R², RMSE, MAE）
   - 生成预测图表
   - SHAP分析（传统ML）

## 注意事项 / Notes

1. ⚠️ 模型选择互斥：不能同时指定 `--use_nn` 和 `--models`
2. ⚠️ 嵌入类型非tradition时，必须指定至少一种嵌入特征
3. ✅ 特征文件自动检测：位于Features/目录或文件名包含'feature'
4. ✅ 结果目录自动创建：遵循标准化目录结构
5. 💡 交叉验证推荐折数：9折（可根据数据量调整）
6. 💡 Optuna优化推荐试验次数：30-50次

## 支持的合金类型 / Supported Alloy Types

- Ti_alloys (钛合金)
- Al_alloys (铝合金)
- Nb_alloys (铌合金)
- HEA (高熵合金)
- Steel (钢铁)

