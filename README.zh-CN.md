# 基于 BERT 的材料性能预测

[English](./README.md)

本仓库包含基于材料领域预训练语言模型的材料性能预测代码与资源，覆盖钛合金、铝合金、钢铁材料和高熵合金等多类数据集。

## 项目简介

项目使用面向材料科学的预训练 BERT 模型，例如 MatSciBERT、SciBERT 和 SteelBERT，预测关键力学性能，包括屈服强度 `YS`、抗拉强度 `UTS` 和延伸率 `EL`。仓库提供从数据预处理、特征工程到模型训练、评估和可视化的完整流程。

## 环境安装

建议使用 Conda 或 pip 进行环境配置。

### 前置要求

- 推荐安装 Anaconda 或 Miniconda
- Python 版本以 [`environment.yml`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/environment.yml) 为准

### 方式一：Conda

```bash
conda env create -f environment.yml
conda activate llm
```

### 方式二：pip

```bash
pip install -r requirements.txt
```

## 模型准备

运行训练或预测脚本前，需要先下载所需的预训练 BERT 模型：

```bash
python models/download_bert_models.py
```

其中 `SteelBERT` 是 Hugging Face 上的受限仓库，需要先完成：

1. 执行 `huggingface-cli login`
2. 在 [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT) 申请访问权限

## TabPFN

仓库中还包含面向合金性能预测的 TabPFN 工作流。

- 总览与使用说明：[`src/TabPFN/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/README.md)
- 基础回归脚本：`python src/TabPFN/train_tabpfn.py`
- 微调脚本：`python src/TabPFN/finetune_tabpfn.py`
- 输出目录：
  - `output/TabPFN_results/`
  - `output/TabPFN_finetune_results/`

## 项目结构

- `models/`：预训练模型目录，例如 MatSciBERT、SciBERT、SteelBERT
- `datasets/`：各类合金体系的原始与处理后数据
- `src/`：训练与预测核心代码
- `src/pipelines/`：批量实验配置与执行脚本
- `src/TabPFN/`：TabPFN 回归与微调流程
- `Scripts/`：绘图和批处理工具脚本
- `output/`：结果与日志输出目录

## 使用说明

主实验入口为 [`src.pipelines.run_batch`](./src/pipelines/README.md)，支持预定义配置、自定义运行、断点续跑和进度管理。

### 基础命令

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

### 完整实验

```bash
python -m src.pipelines.run_batch --all
python -m src.pipelines.run_batch --experiment1
python -m src.pipelines.run_batch --experiment2
```

### 自定义运行

```bash
python -m src.pipelines.run_batch --custom \
    --embedding_type steelbert \
    --use_nn \
    --epochs 100

python -m src.pipelines.run_batch --custom \
    --embedding_type tradition \
    --models xgboost lightgbm \
    --use_composition_feature
```

### 高级功能

```bash
python -m src.pipelines.run_batch --all --dry_run
python -m src.pipelines.run_batch --all --resume
python -m src.pipelines.run_batch --show_progress
python -m src.pipelines.run_batch --clear_progress
```

## 配置文件

- [`src/pipelines/batch_configs.py`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/pipelines/batch_configs.py)
  - `ALLOY_CONFIGS`：定义各合金的数据路径、目标列和处理参数
  - `BATCH_CONFIGS`：定义实验级别的模型和训练参数

## 相关文档

- 英文首页： [README.md](./README.md)
- 模型说明：[`src/models/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/models/README.md)
- 流水线说明：[`src/pipelines/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/pipelines/README.md)
- TabPFN 说明：[`src/TabPFN/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/README.md)
