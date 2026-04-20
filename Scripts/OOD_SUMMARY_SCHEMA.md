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

Special case for `LOCO`:

- `BERT` and `Traditional` first align on the outer `fold_*` split
- within each outer fold, re-scan `optuna_trials` and choose the best trial by:
  1. smallest inner-fold mean test MAE
  2. smallest inner-fold std test MAE
  3. higher inner-fold mean test R2
  4. smaller trial number
- then read the selected outer fold's final `OODTest` / test predictions from the fold root
- `summary_test_*` is the mean/std across those outer-fold final-test metrics, not the inner-fold CV means

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

- non-LOCO methods:
  - first compute trial-level means from fold-level metrics
  - then compute the model-level mean/std across trial means
- `LOCO`:
  - first choose one best trial inside each outer `fold_*`
  - then compute the mean/std across the corresponding outer-fold final `OODTest` metrics
  - `trial_count` / `fold_count` both mean the number of outer folds that actually contributed valid final-test metrics

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

For multi-fold/multi-trial families, representative selection is for tracing only.

- default rule:
  1. smallest `abs(representative_test_r2 - summary_test_r2)`
  2. higher `representative_test_r2`
  3. smaller trial id / fold index / stable file order
- `LOCO` for `BERT` and `Traditional`:
  1. smallest `abs(selected_outer_fold_oodtest_mae - summary_test_mae)`
  2. smaller selected outer-fold `OODTest` MAE
  3. higher selected outer-fold `OODTest` R2
  4. smaller outer fold index

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

`Combined` and triptych outputs should use the canonical LOCO-aligned final-test values for plotting and comparison.

In practice, `per_task_bestplus_tabpfn` applies a LOCO-only override for `BERT` and `Traditional` so their `plot_test_*` values are aligned to these canonical LOCO final-test `summary_test_*` values before family-best selection, plotting, and ranking.
