# Reproducibility Guide

This guide records the Python-only workflow for reproducing the project experiments.

## 1. Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate llm
```

or install dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

The `environment.yml` file is portable and does not include a local `prefix`.

## 2. Model Assets

All reusable model assets belong under `models/`.

Required or supported locations:

- `models/matscibert/`
- `models/scibert/`
- `models/steelbert/`
- `models/llmprop/`
- `models/tabpfn/tabpfn-v2-regressor.ckpt`

Download missing BERT and LLMProp assets:

```bash
python models/download_bert_models.py
python models/download_llmprop_models.py
```

`SteelBERT` requires Hugging Face access approval. TabPFN fine-tuning resolves its checkpoint in this order:

1. `TABPFN_REGRESSOR_MODEL_PATH`
2. `models/tabpfn/tabpfn-v2-regressor.ckpt`
3. user-level TabPFN cache locations
4. legacy project-root checkpoint path

## 3. Standard Experiments

Inspect and run standard experiments:

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --all --dry_run
python -m src.pipelines.run_batch --all
```

Run specific configs:

```bash
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --config experiment2a_all_nn_scibert experiment2b_all_nn_steelbert
```

## 4. OOD Experiments

Inspect and run OOD experiments:

```bash
python -m src.pipelines.run_batch_ood --list
python -m src.pipelines.run_batch_ood --all --dry_run
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation
```

Run k-sweep experiments:

```bash
python -m src.pipelines.run_cv_k_sweep --dry_run
```

## 5. TabPFN And LLMProp

TabPFN direct and OOD runs:

```bash
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
python -m src.TabPFN.run_tabpfn_ood_batch --list
```

LLMProp single OOD run:

```bash
python -m src.LLMProp.run_llmprop_ood --help
```

## 6. Summaries And Figures

Use active Python scripts in `Scripts/`:

```bash
python Scripts/batch_summarize_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_bert_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results
python Scripts/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
```

Set `EXTERNAL_OOD_ROOT` only when external LLM OOD results should be merged.

## 7. Outputs

Generated artifacts are intentionally ignored:

- `Features/`
- `Features_extrapolation/`
- `output/`
- `logs/`
- `.batch_progress*.json`

Training checkpoints remain under each run's output directory. They are not moved into `models/` unless they become reusable base assets.
