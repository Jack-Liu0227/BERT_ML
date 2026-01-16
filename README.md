# BERT-Based Material Property Prediction / 基于 BERT 的材料性能预测

This repository contains code and resources for predicting material properties using advanced language models like MatSciBERT, SciBERT, and SteelBERT. The project focuses on various alloys including Titanium, Aluminum, Steel, and High-Entropy Alloys (HEAs).

> **Navigation**:
>
> - [🇬🇧 English Documentation](#english-documentation)
> - [🇨🇳 中文说明 (Chinese Documentation)](#chinese-documentation)

---

<div id="english-documentation"></div>

## 🇬🇧 English Documentation

### 📖 Introduction

This project leverages pre-trained BERT models adapted for material science to predict key mechanical properties such as Yield Strength (YS), Ultimate Tensile Strength (UTS), and Elongation (EL). It provides a complete pipeline from data preprocessing and feature engineering to model training and visualization.

### 🛠️ Environment Setup & Reproduction

To ensure that you can fully reproduce our results, we have exported the exact environment configurations. You can set up the environment using either Conda or Pip.

#### Prerequisites
- Anaconda or Miniconda (recommended)
- Python (as specified in `environment.yml`)

#### Method 1: Using Conda (Recommended)
This method ensures that all binary dependencies and specific versions are installed correctly.

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate llm
```

#### Method 2: Using Pip
If you prefer a standard pip workflow:

```bash
pip install -r requirements.txt
```

### 📥 Model Preparation

Before running the training or prediction scripts, you need to download the pre-trained BERT models (MatSciBERT, SciBERT, SteelBERT). We provided a script to automate this process.

```bash
# Download all required models
python models/download_bert_models.py
```

> **Note**: **SteelBERT** is a gated repository on Hugging Face. You must:
> 1. Run `huggingface-cli login` to authenticate.
> 2. Request access at [https://huggingface.co/MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT).

### 📂 Project Structure

- **`models/`**: Stores pre-trained models (MatSciBERT, SciBERT, etc.).
- **`datasets/`**: Raw and processed datasets for different alloy systems.
- **`src/`**: Source code for core logic (training, prediction).
    - **`src/pipelines/`**: Contains batch processing configs and scripts.
- **`Scripts/`**: Utility scripts for plotting and batch processing.
- **`output/`**: Directory where results and logs are saved.

### 🚀 Usage

The project provides a powerful batch processing system to run experiments across multiple alloy types and models.

#### Batch Execution (`src.pipelines.run_batch`)

The main entry point for running experiments is `src.pipelines.run_batch`. It supports:
- Running predefined experimental configurations.
- Running custom combinations of models and alloys.
- Resuming interrupted runs.
- Parallel execution.

**1. Basic Commands**

```bash
# List all available configurations and alloy types
python -m src.pipelines.run_batch --list

# Run a specific configuration (e.g., matching traditional ML models)
python -m src.pipelines.run_batch --config experiment1_all_ml_models

# Run multiple configurations
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

**2. Running Complete Experiments**

We have defined comprehensive experiment sets (Experiment 1 & 2):

```bash
# Run ALL experiments (Experiment 1 + Experiment 2a/2b/2c)
python -m src.pipelines.run_batch --all

# Run only Experiment 1 (Traditional ML Models: XGBoost, RF, MLP, etc.)
python -m src.pipelines.run_batch --experiment1

# Run only Experiment 2 (Neural Networks with BERT Embeddings)
python -m src.pipelines.run_batch --experiment2
```

**3. Custom Execution**

You can also run custom setups without using predefined configs:

```bash
# Run SteelBERT encoding for all alloys using Neural Networks
python -m src.pipelines.run_batch --custom \
    --embedding_type steelbert \
    --use_nn \
    --epochs 100

# Run specific traditional models with custom settings
python -m src.pipelines.run_batch --custom \
    --embedding_type tradition \
    --models xgboost lightgbm \
    --use_composition_feature
```

**4. Advanced Features**

- **Preview (Dry Run)**: See what commands will be executed without running them.
  ```bash
  python -m src.pipelines.run_batch --all --dry_run
  ```

- **Resume Support**: If a task is interrupted, use `--resume` to continue from where it left off.
  ```bash
  python -m src.pipelines.run_batch --all --resume
  ```

- **Progress Management**:
  ```bash
  # View current progress
  python -m src.pipelines.run_batch --show_progress
  
  # Clear progress history
  python -m src.pipelines.run_batch --clear_progress
  ```

#### Configuration Files

- **`src/pipelines/batch_configs.py`**:
  - `ALLOY_CONFIGS`: Defines dataset paths, target columns, and processing parameters for each alloy (Ti, Al, HEA, Steel, etc.).
  - `BATCH_CONFIGS`: Defines experiment parameters (models to use, validation settings, hyperparameters).

---

<div id="chinese-documentation"></div>

## 🇨🇳 中文说明 (Chinese Documentation)

[⬆️ Back to Top / 回到顶部](#bert-based-material-property-prediction--基于-bert-的材料性能预测)

### 📖 项目简介

本项目利用针对材料科学预训练的 BERT 模型（如 MatSciBERT, SciBERT, SteelBERT）来预测关键的机械性能，例如屈服强度 (YS)、由于抗拉强度 (UTS) 和延伸率 (EL)。主要研究对象涵盖钛合金、铝合金、钢材以及高熵合金 (HEAs)。

### 🛠️ 环境安装与复现

为了确保您可以完全复现我们的项目结果，我们导出了详细的环境配置文件。您可以选择使用 Conda 或 Pip 进行安装。

#### 前置要求
- Anaconda 或 Miniconda (推荐)
- Python (版本详见 `environment.yml`)

#### 方法 1: 使用 Conda (推荐)
此方法能最准确地还原运行环境，包含所有依赖库。

```bash
# 从 yaml 文件创建环境
conda env create -f environment.yml

# 激活环境
conda activate llm
```

#### 方法 2: 使用 Pip
如果您更习惯使用 pip：

```bash
pip install -r requirements.txt
```

### 📥 模型准备

在开始训练或预测之前，您需要下载预训练的 BERT 模型 (MatSciBERT, SciBERT, SteelBERT)。我们提供了一个自动化脚本来完成此步骤。

```bash
# 下载所有需要的模型
python models/download_bert_models.py
```

> **注意**: **SteelBERT** 是 Hugging Face 上的受控仓库。您必须：
> 1. 运行 `huggingface-cli login` 进行登录。
> 2. 在 [https://huggingface.co/MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT) 申请访问权限。

### 📂 项目结构

- **`models/`**: 存放预训练模型 (MatSciBERT, SciBERT 等)。
- **`datasets/`**: 不同合金体系的原始及处理后的数据集。
- **`src/`**: 核心源代码 (训练、预测逻辑)。
    - **`src/pipelines/`**: 包含批量处理的配置和脚本。
- **`Scripts/`**: 用于绘图和批处理的工具脚本。
- **`output/`**: 结果和日志输出目录。

### 🚀 使用说明

本项目提供了一个强大的批量处理系统，可以针对多种合金类型和模型运行实验。

#### 批量执行 (`src.pipelines.run_batch`)

运行实验的主要入口是 `src.pipelines.run_batch`。它支持：
- 运行预定义的实验配置。
- 运行自定义的模型和合金组合。
- 断点续传（从中断处继续运行）。
- 并行执行。

**1. 基础命令**

```bash
# 列出所有可用配置和合金类型
python -m src.pipelines.run_batch --list

# 运行特定配置（例如：传统机器学习模型对比）
python -m src.pipelines.run_batch --config experiment1_all_ml_models

# 同时运行多个配置
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

**2. 运行完整实验**

我们定义了完整的实验集（实验 1 和 实验 2）：

```bash
# 运行所有实验（实验 1 + 实验 2a/2b/2c）
python -m src.pipelines.run_batch --all

# 仅运行实验 1（传统 ML 模型：XGBoost, RF, MLP 等）
python -m src.pipelines.run_batch --experiment1

# 仅运行实验 2（神经网络 + BERT 嵌入）
python -m src.pipelines.run_batch --experiment2
```

**3. 自定义运行**

您也可以在不使用预定义配置的情况下进行自定义设置：

```bash
# 使用神经网络运行 SteelBERT 编码（针对所有合金）
python -m src.pipelines.run_batch --custom \
    --embedding_type steelbert \
    --use_nn \
    --epochs 100

# 运行特定的传统模型并使用自定义设置
python -m src.pipelines.run_batch --custom \
    --embedding_type tradition \
    --models xgboost lightgbm \
    --use_composition_feature
```

**4. 高级功能**

- **预览模式 (Dry Run)**: 查看将要执行的命令而不实际运行。
  ```bash
  python -m src.pipelines.run_batch --all --dry_run
  ```

- **断点续传**: 如果任务中断，使用 `--resume` 从中断处继续执行。
  ```bash
  python -m src.pipelines.run_batch --all --resume
  ```

- **进度管理**:
  ```bash
  # 查看当前进度
  python -m src.pipelines.run_batch --show_progress
  
  # 清除进度记录
  python -m src.pipelines.run_batch --clear_progress
  ```

#### 配置文件

- **`src/pipelines/batch_configs.py`**:
  - `ALLOY_CONFIGS`: 定义每种合金（Ti, Al, HEA, Steel 等）的数据集路径、目标列和处理参数。
  - `BATCH_CONFIGS`: 定义实验参数（使用的模型、验证设置、超参数）。
