# BERT_ML: Alloy Property Prediction Experiments

[中文说明](./README.zh-CN.md)

This repository contains Python workflows for alloy property prediction with traditional ML, BERT-based embeddings, TabPFN, and LLMProp. The project is organized for full experiment reproducibility: all runnable experiment entrypoints are Python modules or Python scripts, model assets live under `models/`, and historical repair utilities are archived separately.

## Quick Start

Create the environment from the project root:

```bash
conda env create -f environment.yml
conda activate llm
```

or install with pip:

```bash
pip install -r requirements.txt
```

Download model assets when they are not already present:

```bash
python models/download_bert_models.py
python models/download_llmprop_models.py
```

`SteelBERT` is a gated Hugging Face model. Log in with `huggingface-cli login` and request access at [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT).

## Python-Only Entrypoints

Shell wrappers are intentionally not used. Run experiments with Python only:

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --all --dry_run
```

OOD experiments:

```bash
python -m src.pipelines.run_batch_ood --list
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation --dry_run
python -m src.pipelines.run_cv_k_sweep --dry_run
```

TabPFN experiments:

```bash
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
python -m src.TabPFN.run_tabpfn_ood_batch --list
```

LLMProp OOD single-run entrypoint:

```bash
python -m src.LLMProp.run_llmprop_ood --help
```

## Model Assets

Model assets are centralized under `models/`:

- `models/matscibert/`
- `models/scibert/`
- `models/steelbert/`
- `models/llmprop/`
- `models/tabpfn/tabpfn-v2-regressor.ckpt`

Generated training checkpoints stay under `output/**/checkpoints/` because they are experiment outputs, not reusable base assets.

## Data And Outputs

Tracked input datasets are under `datasets/`. The active configuration files reference these normalized CSV paths:

- `datasets/Ti_alloys/titanium.csv`
- `datasets/Al_Alloys/aluminum.csv`
- `datasets/HEA_data/hea.csv`
- `datasets/Steel/steel.csv`
- `datasets/HEA_data/Pitting_potential_data_xiongjie.csv`
- `datasets/matbench_convert/matbench_steels.csv`
- `datasets/matbench_convert/matbench_steels_ood.csv`

Generated features, logs, figures, and model outputs are written to ignored directories such as `Features/`, `Features_extrapolation/`, `output/`, and `logs/`.

## Reporting Scripts

Active analysis and figure scripts remain in `Scripts/`. Examples:

```bash
python Scripts/batch_summarize_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_bert_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results
python Scripts/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
```

Optional external OOD sources are configured in `Scripts/external_ood_model_sources.yaml`. Set `EXTERNAL_OOD_ROOT` before running combined summaries if external LLM results should be merged.

## Documentation

- [English reproducibility guide](./docs/reproducibility.md)
- [中文复现指南](./docs/reproducibility.zh-CN.md)
- [Script inventory](./docs/script_inventory.md)
- [脚本清单](./docs/script_inventory.zh-CN.md)
- [Batch run guide](./src/pipelines/BATCH_RUN_GUIDE.md)
- [TabPFN guide](./src/TabPFN/README.md)

Historical local repair and migration utilities are preserved in `archive/legacy_scripts/`. They are not recommended reproduction entrypoints.
