# OOD summary schema

This repo now uses a canonical OOD summary schema across `BERT`, `Traditional`, `TabPFN`, and `Combined`.

## One row means

Each family summary row corresponds to:

- `alloy_family`
- `dataset_name`
- `property`
- `ood_method`
- `model`

For `BERT` and `Traditional`, the row is built from all available trial/fold results of that model under the fixed case above.

For `TabPFN`, the row is built from the available run units for that case:

- single-run methods: one run
- `LOCO`: all fold metrics under that case

## Canonical metric fields

- `trial_count`: number of trial groups contributing to the row
- `fold_count`: number of fold/run records contributing to the row
- `summary_test_r2`
- `summary_test_r2_std`
- `summary_test_mae`
- `summary_test_mae_std`
- `summary_test_rmse`
- `summary_test_rmse_std`

For `BERT` and `Traditional`:

- first compute trial-level means from fold-level metrics
- then compute the model-level mean/std across trial means

For `TabPFN`:

- non-LOCO methods use the single available run
- `LOCO` uses the mean/std across fold metrics

## Representative result fields

- `representative_selection_mode`
- `representative_trial_id`
- `representative_fold`
- `representative_test_r2`
- `representative_test_mae`
- `representative_test_rmse`
- `representative_predictions_file`
- `representative_plot_file`

Representative selection is for artifact tracing only. It does not decide the family-best model.

### Representative selection rule

For multi-fold/multi-trial families, choose the fold/run with:

1. smallest `abs(representative_test_r2 - summary_test_r2)`
2. higher `representative_test_r2`
3. smaller trial id / fold index / stable file order

## Family-best fields

- `family_best_metric`
- `family_rank_score`
- `rank_within_family`
- `is_family_best`

Family-best is ranked within each:

- `alloy_family`
- `dataset_name`
- `property`
- `ood_method`
- `model_family`

using:

1. `summary_test_r2` descending
2. `summary_test_r2_std` ascending
3. `summary_test_mae` ascending
4. `model` ascending

## Downstream consumers

The following scripts now read the canonical schema directly:

- `batch_summarize_bert_extrapolation_results.py`
- `batch_summarize_extrapolation_results.py`
- `batch_summarize_tabpfn_extrapolation_results.py`
- `batch_summarize_combined_ood_reports.py`
- `build_bestplus_tabpfn_triptych.py`

`Combined` and triptych outputs should always use `summary_test_*` as the plotting and comparison metrics.
