# Scripts

`Scripts/` only keeps the core OOD reporting and final-figure pipeline. Historical repair, export, Optuna, 2D-map, and one-off utilities were removed from this active directory; archived utilities live under `archive/legacy_scripts/`.

Run every command from the repository root.

## Core Workflow

1. Summarize each model family into the canonical OOD schema.
2. Merge family summaries into one combined report.
3. Build final per-task triptych figures from the combined report.

```bash
python Scripts/batch_summarize_extrapolation_results.py
python Scripts/batch_summarize_bert_extrapolation_results.py
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_llmprop_ood_results.py
python Scripts/batch_summarize_combined_ood_reports.py
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
```

## Retained CLI Scripts

### `batch_summarize_extrapolation_results.py`

Summarizes Traditional-ML OOD results into the canonical schema.

Default inputs and outputs:

- input: `output/ood_results`
- output: `output/ood_summary_reports/Traditional`

Common usage:

```bash
python Scripts/batch_summarize_extrapolation_results.py
python Scripts/batch_summarize_extrapolation_results.py --base-dir output/ood_results --output-dir output/ood_summary_reports/Traditional
```

### `batch_summarize_bert_extrapolation_results.py`

Summarizes BERT/NN OOD results into the canonical schema.

Default inputs and outputs:

- input: `output/ood_results`
- output: `output/ood_summary_reports/BERT`
- optional progress filter: `.batch_progress_ood.json`

Common usage:

```bash
python Scripts/batch_summarize_bert_extrapolation_results.py
python Scripts/batch_summarize_bert_extrapolation_results.py --success-only
python Scripts/batch_summarize_bert_extrapolation_results.py --tracked-config experiment2a_all_nn_scibert_extrapolation
```

### `batch_summarize_tabpfn_extrapolation_results.py`

Summarizes TabPFN OOD results into the canonical schema.

Default inputs and outputs:

- input: `output/ood_results_TabPFN-2.5-Plus-Numeric` and `output/ood_results_TabPFN-2.5-Plus-Text`
- output: `output/ood_summary_reports/TabPFN`

Common usage:

```bash
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_tabpfn_extrapolation_results.py --base-dir output/ood_results_TabPFN-2.5-Plus-Numeric
python Scripts/batch_summarize_tabpfn_extrapolation_results.py --base-dirs output/ood_results_TabPFN-2.5-Plus-Numeric output/ood_results_TabPFN-2.5-Plus-Text
```

### `batch_summarize_llmprop_ood_results.py`

Summarizes LLMProp OOD results into the canonical schema.

Default inputs and outputs:

- input: `output/ood_results`
- output: `output/ood_summary_reports/LLMProp`

Common usage:

```bash
python Scripts/batch_summarize_llmprop_ood_results.py
python Scripts/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results --output-dir output/ood_summary_reports/LLMProp
```

### `batch_summarize_combined_ood_reports.py`

Merges Traditional, BERT, TabPFN, LLMProp, and optional external OOD summaries into one report.

Default inputs and outputs:

- input root: `output/ood_summary_reports`
- output: `output/ood_summary_reports/Combined`
- optional external config: `Scripts/external_ood_model_sources.yaml`

Common usage:

```bash
python Scripts/batch_summarize_combined_ood_reports.py
python Scripts/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports --output-dir output/ood_summary_reports/Combined
```

Optional external LLM results are loaded only when their configured root exists. To use the default external config:

```bash
EXTERNAL_OOD_ROOT=/path/to/external/ood/results python Scripts/batch_summarize_combined_ood_reports.py
```

On Windows PowerShell, set the environment variable before running Python:

```text
$env:EXTERNAL_OOD_ROOT="D:\path\to\external\ood\results"
python Scripts/batch_summarize_combined_ood_reports.py
```

### `build_bestplus_tabpfn_triptych.py`

Builds final per-task triptych figures from the combined summary. It does not read raw model outputs; it reads the combined canonical summary.

Default inputs and outputs:

- input: `output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv`
- default config: `Scripts/build_bestplus_tabpfn_triptych.config.yaml`
- paper config: `Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml`
- output: `output/ood_summary_reports/Combined/figure/per_task_bestplus_tabpfn`

Common usage:

```bash
python Scripts/build_bestplus_tabpfn_triptych.py
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
python Scripts/build_bestplus_tabpfn_triptych.py --summary-csv output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv --output-dir output/ood_summary_reports/Combined/figure/per_task_bestplus_tabpfn_paper
```

## Helper And Configuration Files

The following files are required by the retained CLI scripts:

- `_raw_prediction_stats.py`: prediction and Optuna trial metric readers.
- `_ood_summary_common.py`: canonical OOD schema, exports, ranking, and plotting helpers.
- `_external_ood_sources.py`: optional external result loader for combined reports.
- `_bestplus_tabpfn_triptych_config.py`: triptych config loader and defaults.
- `_bestplus_tabpfn_triptych_plotting.py`: triptych plotting implementation.
- `OOD_SUMMARY_SCHEMA.md`: canonical schema description.
- `external_ood_model_sources.yaml`: optional external LLM source mapping.
- `build_bestplus_tabpfn_triptych.config.yaml`: default figure config.
- `build_bestplus_tabpfn_triptych.paper.config.yaml`: paper-oriented figure config.

Do not run helper modules directly.

## Removed From Active Scripts

The active directory no longer contains:

- shell wrappers: `.ps1`, `.bat`, `.sh`, `.cmd`
- historical Optuna-only scripts
- one-off export scripts
- 2D-map rendering and split-trace refresh scripts
- local repair or rerun utilities

Use the Python module entrypoints in `src/` to generate experiments, then use the retained scripts here to summarize and plot results.
