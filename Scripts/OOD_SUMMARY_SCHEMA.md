# OOD summary schema

This repo now uses a canonical OOD summary schema across `BERT`, `Traditional`, `TabPFN`, and `Combined`.

Supported OOD methods in the reporting layer are:

- `RandomCV`
- `Extrapolation`
- `LOCO`
- `SparseXcluster`
- `SparseXsingle`
- `SparseYcluster`
- `SparseYsingle`

## One row means

Each family summary row corresponds to:

- `alloy_family`
- `dataset_name`
- `property`
- `ood_method`
- `model`

For every family, the canonical `summary_test_*` fields now align to the final exported OOD-test semantics used by `artifact_test_*` / `plot_test_*`.

Special case for outer-fold methods such as `LOCO` and `RandomCV`:

- `BERT` and `Traditional` first align on the outer `fold_*` split
- within each outer fold, re-scan `optuna_trials` and choose the best trial by:
  1. smallest inner-fold mean test MAE
  2. smallest inner-fold std test MAE
  3. higher inner-fold mean test R2
  4. smaller trial number
- then read the selected outer fold's final `OODTest` / test predictions from the fold root
- `summary_test_*` is the mean/std across those outer-fold final-test metrics, not the inner-fold CV means

For `RandomCV`, the outer `fold_*` directories come from a standard shuffled 5-fold random split of the full dataset.
- `TabPFN` uses the mean/std across fold-level final `ood_test` metrics and the canonical artifact value is the same fold aggregate

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
  - optuna / CV artifacts are still scanned for provenance and representative-trial tracing
  - but canonical `summary_test_*` is overwritten to the case-level final OOD-test result from the selected exported model
  - therefore the canonical non-LOCO `summary_test_*_std` is `0.0`
- `LOCO`:
  - first choose one best trial inside each outer `fold_*`
  - then compute the mean/std across the corresponding outer-fold final `OODTest` metrics
  - `trial_count` / `fold_count` stay as the already-exported aggregation metadata for that row

For `TabPFN`:

- non-LOCO methods use the single available final `ood_test` run
- `LOCO` uses the mean/std across fold-level final `ood_test` metrics, and canonical artifact/plot metrics use the same fold aggregate

For external `LLM` rows such as `gpt-5.4`:

- `summary_test_*`, `artifact_test_*`, and `plot_test_*` already refer to the same final OOD-test semantics

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
- non-LOCO single-run canonical rows:
  - the representative result is rewritten to the same case-level final OOD-test artifact used by `summary_test_*`
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

`Combined` and triptych outputs should use the canonical final OOD-test-aligned values for plotting and comparison.

In practice, `summary_test_*`, `artifact_test_*`, and `plot_test_*` are intended to be numerically aligned for every exported row. Any remaining family-specific extra metadata is for provenance only.
