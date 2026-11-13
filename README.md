# BERT_ML: Alloy Property Prediction Benchmark & Experiment Suite

[中文说明](./README.zh-CN.md)

BERT_ML is a modular, reproducible research benchmark for predicting mechanical properties of multi-component alloys across both In-Distribution (I.I.D.) and Out-of-Distribution (OOD) regimes. The repository integrates traditional tabular machine learning, deep neural networks, multimodal materials language model embeddings (SciBERT, SteelBERT, MatSciBERT), Tabular Prior-data Fitted Networks (TabPFN v2), and LLM-based property predictors (LLMProp).

---

## Key Features

- **4 Active Model Families**:
  - **Traditional ML**: XGBoost, LightGBM, CatBoost, Random Forest, ExtraTrees, SVR, Ridge, Lasso, ElasticNet, and MLP with Optuna hyperparameter optimization, multi-output regression, and SHAP interpretability.
  - **Deep Neural Networks (AlloyNN)**: Feature-fusion networks combining compositional fractions, process parameters, and text embeddings.
  - **Domain BERT + NN Fusion**: Joint architectures integrating pre-trained domain language models (SciBERT, SteelBERT, MatSciBERT) with process/composition features.
  - **Tabular Foundation Models (TabPFN v2)**: In-context tabular inference with local checkpoints and API integration.
  - **LLM-Based Regression (LLMProp)**: Textual prompt-driven property prediction.
- **5 Curated Alloy Systems**:
  - Titanium alloys (Ti), Aluminum alloys (Al), High-Entropy Alloys (HEA), Carbon/Structural Steels (Steel), and Matbench Steels.
  - Targets: Yield Strength (YS(MPa)), Ultimate Tensile Strength (UTS(MPa)), and Elongation (El(%)).
- **8 Comprehensive OOD Data Splitting Protocols**:
  - Extreme Target Extrapolation (	arget_extrapolation: Top 20% / Bottom 20%).
  - Composition / Feature Space Sparsity (sparse_x_single, sparse_x_cluster).
  - Target / Property Space Sparsity (sparse_y_single, sparse_y_cluster).
  - Unseen Alloy Cluster Generalization (loco: Leave-One-Cluster-Out).
  - Standard In-Distribution Validation (
andom_cv_baseline: Multi-fold CV).
  - Hybrid Extrapolation Combinations (hybrid_extrapolation_*).
- **Modular Architecture**:
  - src/: Core dataset splitting, feature engineering, and model training pipelines.
  - Scripts/: Offline post-hoc uncertainty calibration, distribution shift analysis (Wasserstein distance / W-jump), reporting, and publication-ready figure rendering.

---

## Directory Structure

\BERT_ML/
├── datasets/                  # Curated benchmark datasets
│   ├── Ti_alloys/             # Titanium alloy datasets
│   ├── Al_Alloys/             # Aluminum alloy datasets
│   ├── HEA_data/              # High-entropy alloy datasets
│   ├── Steel/                 # Steel mechanical property datasets
│   └── matbench_convert/      # Matbench benchmark datasets
├── models/                    # Model weights & download utilities
│   ├── download_bert_models.py
│   ├── download_llmprop_models.py
│   └── tabpfn/                # TabPFN v2 local model weights
├── src/                       # Core training and evaluation framework
│   ├── data_processing/       # OOD splitting processors, registries & split cache
│   ├── feature_engineering/   # Composition features, BERT loader & embeddings
│   ├── models/                # ML/NN/BERT architectures, trainers & evaluators
│   ├── pipelines/             # Batch experiment runners & multi-fold CV sweeps
│   ├── TabPFN/                # TabPFN training, OOD pipelines & prediction alignment
│   ├── LLMProp/               # LLMProp fine-tuning & OOD evaluation adapter
│   └── training_size_ood/     # Sample-size scaling experiment runner
├── Scripts/                   # Analysis, calibration, and visualization suite
│   ├── analysis/              # Space shifts, W-jump analysis & data audits
│   ├── calibration/           # Post-hoc confidence & uncertainty calibration
│   ├── figures/               # Paper figures & interactive explainers
│   ├── maintenance/           # Result repair, audit & reset utilities
│   └── reporting/             # Batch summarization & benchmark tables
└── tests/                     # Automated pytest verification test suite
\
---

## Quick Start

### 1. Environment Setup

Create the Conda environment from environment.yml (Python 3.10 recommended):

\\ash
conda env create -f environment.yml
conda activate llm
\
Alternatively, install dependencies via pip:

\\ash
pip install -r requirements.txt
\
### 2. Download Base Model Weights

Download domain BERT and LLMProp checkpoints into models/:

\\ash
python models/download_bert_models.py
python models/download_llmprop_models.py
\
> **Note**: SteelBERT is hosted as a gated repository on Hugging Face. Run \huggingface-cli login\ and request access at [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT).

---

## Running Experiments

All pipelines are designed as pure Python module entrypoints (python -m ...). No shell or batch script wrappers are required.

### 1. In-Distribution (I.I.D.) Batch Experiments

List available batch configurations:
\\ash
python -m src.pipelines.run_batch --list
\
Run a specific experiment:
\\ash
python -m src.pipelines.run_batch --config experiment1_all_ml_models
\
Dry-run all configured standard batches:
\\ash
python -m src.pipelines.run_batch --all --dry_run
\
### 2. Out-of-Distribution (OOD) Benchmark Matrix

List available OOD configurations (64 batch setups across 5 alloy systems and 8 splitting strategies):
\\ash
python -m src.pipelines.run_batch_ood --list
\
Run target extrapolation with traditional ML:
\\ash
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation
\
Run LOCO (Leave-One-Cluster-Out) with SciBERT + NN:
\\ash
python -m src.pipelines.run_batch_ood --config experiment2a_all_nn_scibert_loco
\
Preview task execution without running:
\\ash
python -m src.pipelines.run_batch_ood --all --dry_run
\
### 3. Multi-Seed Cross-Validation Sweeps

Run systematic k-fold sweeps across seeds and models:
\\ash
# Dry-run validation
python -m src.pipelines.run_cv_k_sweep --dry_run

# Execute 5-fold CV across seeds 0 to 4 for Random CV baseline
python -m src.pipelines.run_cv_k_sweep --method random_cv_baseline --k 5 --seeds 0 1 2 3 4 --workers 2
\
### 4. TabPFN Foundation Model Experiments

Run TabPFN on in-distribution or OOD splits:
\\ash
# List available TabPFN OOD configs
python -m src.TabPFN.run_tabpfn_ood_batch --list

# Run TabPFN local regression on all standard tasks
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric

# Run TabPFN on OOD splits
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
\
### 5. LLMProp Experiments

Run LLMProp property prediction under OOD protocols:
\\ash
python -m src.LLMProp.run_llmprop_ood --help
\
---

## Offline Analysis, Calibration & Figures

Modules under Scripts/ provide post-processing, verification, and visualization tools:

- **Uncertainty Calibration**:
  \\ash
  python -m Scripts.calibration.cli --help
  \- **Publication Figures (Figures 3 & 4)**:
  \\ash
  python -m Scripts.figures.build_ood_figures_3_4 --help
  \- **TabPFN Triptych Comparison**:
  \\ash
  python -m Scripts.figures.build_bestplus_tabpfn_triptych --help
  \
---

## Verification & Testing

The repository includes a comprehensive pytest suite ensuring full coverage of split registries, model adapters, calibration routines, and CLI entrypoints:

\\ash
pytest tests/
\
All 267 test cases pass with 0 errors.

---

## License & Citation

This project is licensed under the MIT License. If you use this codebase or benchmark data in your research, please cite our corresponding publications.
