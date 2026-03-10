# BERT-Based Material Property Prediction

[中文说明](./README.zh-CN.md)

This repository contains code and resources for predicting material properties using advanced language models such as MatSciBERT, SciBERT, and SteelBERT. The project focuses on multiple alloy systems, including Titanium, Aluminum, Steel, and High-Entropy Alloys (HEAs).

## Introduction

This project uses pretrained BERT models adapted for materials science to predict key mechanical properties such as Yield Strength (`YS`), Ultimate Tensile Strength (`UTS`), and Elongation (`EL`). It provides a full workflow from data preprocessing and feature engineering to model training, evaluation, and visualization.

## Environment Setup

To reproduce the project environment, use either Conda or pip.

### Prerequisites

- Anaconda or Miniconda is recommended
- Python version should match [`environment.yml`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/environment.yml)

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate llm
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## Model Preparation

Before running training or prediction scripts, download the required pretrained BERT models:

```bash
python models/download_bert_models.py
```

`SteelBERT` is a gated Hugging Face repository. You need to:

1. Run `huggingface-cli login`
2. Request access at [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT)

## TabPFN

This repository also includes a dedicated TabPFN workflow for alloy property prediction.

- Overview and usage: [`src/TabPFN/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/README.md)
- Direct regression script: `python src/TabPFN/train_tabpfn.py`
- Fine-tuning script: `python src/TabPFN/finetune_tabpfn.py`
- Result directories:
  - `output/TabPFN_results/`
  - `output/TabPFN_finetune_results/`

## Project Structure

- `models/`: pretrained models such as MatSciBERT, SciBERT, and SteelBERT
- `datasets/`: raw and processed datasets for alloy systems
- `src/`: source code for training and prediction logic
- `src/pipelines/`: batch experiment configuration and execution scripts
- `src/TabPFN/`: TabPFN regression and fine-tuning workflow
- `Scripts/`: utility scripts for plotting and batch processing
- `output/`: generated results and logs

## Usage

The main experiment entry point is [`src.pipelines.run_batch`](./src/pipelines/README.md). It supports predefined configurations, custom runs, resume mode, and progress tracking.

### Basic Commands

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

### Complete Experiment Sets

```bash
python -m src.pipelines.run_batch --all
python -m src.pipelines.run_batch --experiment1
python -m src.pipelines.run_batch --experiment2
```

### Custom Runs

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

### Advanced Features

```bash
python -m src.pipelines.run_batch --all --dry_run
python -m src.pipelines.run_batch --all --resume
python -m src.pipelines.run_batch --show_progress
python -m src.pipelines.run_batch --clear_progress
```

## Configuration Files

- [`src/pipelines/batch_configs.py`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/pipelines/batch_configs.py)
  - `ALLOY_CONFIGS`: dataset paths, targets, and processing parameters for each alloy
  - `BATCH_CONFIGS`: experiment-level model and training settings

## Documentation

- Chinese project overview: [README.zh-CN.md](./README.zh-CN.md)
- Model suite details: [`src/models/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/models/README.md)
- Pipeline details: [`src/pipelines/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/pipelines/README.md)
- TabPFN details: [`src/TabPFN/README.md`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/README.md)
## Contact

- Email: `liu_yujie@stu.xjtu.edu.cn`
