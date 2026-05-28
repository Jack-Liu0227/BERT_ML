# Batch Run Guide / 批量运行指南

This project uses Python-only batch entrypoints. Do not use PowerShell, bat, sh, or cmd wrappers for reproduction.

本项目的批量实验统一使用 Python 入口；复现时不要使用 PowerShell、bat、sh 或 cmd 包装脚本。

## Standard Experiments

List available standard batch configs:

```bash
python -m src.pipelines.run_batch --list
```

Run one or more configs:

```bash
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

Run all standard configs or preview commands:

```bash
python -m src.pipelines.run_batch --all
python -m src.pipelines.run_batch --all --dry_run
python -m src.pipelines.run_batch --all --resume
```

Current standard configs are defined in `src/pipelines/batch_configs.py`:

- `experiment1_all_ml_models`
- `experiment2a_all_nn_scibert`
- `experiment2b_all_nn_steelbert`
- `experiment2c_all_nn_matscibert`

## OOD Experiments

List OOD data configs, methods, and batch configs:

```bash
python -m src.pipelines.run_batch_ood --list
```

Run or preview OOD configs:

```bash
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation --dry_run
python -m src.pipelines.run_batch_ood --all --dry_run
```

Run k-sweep experiments:

```bash
python -m src.pipelines.run_cv_k_sweep --dry_run
python -m src.pipelines.run_cv_k_sweep --methods random_cv_baseline loco --k_values 3 5
```

OOD data configs are defined in `src/pipelines/batch_configs_ood.py`.

## Custom Standard Runs

Run selected alloys and models:

```bash
python -m src.pipelines.run_batch --custom \
    --alloy_types Ti Al \
    --embedding_type tradition \
    --use_composition_feature \
    --models xgboost lightgbm \
    --cross_validate \
    --num_folds 9
```

Run a BERT/NN custom config:

```bash
python -m src.pipelines.run_batch --custom \
    --embedding_type steelbert \
    --use_element_embedding \
    --use_process_embedding \
    --use_nn \
    --cross_validate \
    --num_folds 9 \
    --use_optuna \
    --n_trials 30 \
    --evaluate_after_train
```

## Active Dataset Keys

Standard configs currently use:

| Key | Data file | Targets |
| --- | --- | --- |
| `Ti` | `datasets/Ti_alloys/titanium.csv` | `UTS(MPa)`, `El(%)` |
| `Al` | `datasets/Al_Alloys/aluminum.csv` | `UTS(MPa)` |
| `HEA` | `datasets/HEA_data/hea.csv` | `YS(MPa)`, `UTS(MPa)`, `El(%)` |
| `Steel` | `datasets/Steel/steel.csv` | `YS(MPa)`, `UTS(MPa)`, `El(%)` |
| `HEA_corrosion` | `datasets/HEA_data/Pitting_potential_data_xiongjie.csv` | `Ep(mV)` |

OOD configs additionally include:

| Key | Data file | Targets |
| --- | --- | --- |
| `MatbenchSteels` | `datasets/matbench_convert/matbench_steels_ood.csv` | `yield strength` |

## Outputs

Generated files are written under ignored output directories such as:

- `Features/`
- `Features_extrapolation/`
- `output/`
- `logs/`

Progress files such as `.batch_progress_*.json` are temporary and ignored.

## Notes

- Use `--dry_run` before long experiments to inspect commands.
- Use `--resume` to continue from existing progress files.
- Generated model checkpoints remain under each run's `output/**/checkpoints/` directory.
