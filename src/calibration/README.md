# Calibration Module

This module owns non-GPT regression calibration. It discovers existing tuned
models, audits parameter provenance, freezes effective model configurations,
normalizes real uncertainty estimates, assigns confidence tiers, and computes
calibration summaries.

## Commands

```powershell
python -m src.calibration.cli --stage audit-parameters
python -m src.calibration.cli --stage select-best-parameters
python -m src.calibration.cli --stage uncertainty
python -m src.calibration.cli --stage calibration
python -m src.calibration.cli --stage all
```

The rebuilt full workflow uses only the frozen v1 task/model registry and the
canonical files below `output/ood_splits`:

```powershell
python -m src.calibration.cli --stage reset-output
python -m src.calibration.cli --stage reset-output --execute
python -m src.calibration.cli --stage full-index --scope ti
python -m src.calibration.cli --stage full-run --scope ti --resume --jobs 1
python -m src.calibration.cli --stage full-aggregate --scope ti
python -m src.calibration.cli --stage full-verify --scope ti
```

Run the same four `full-*` stages with `--scope all` after the Ti gate passes.
`full-index`, `full-run`, and `full-verify` reject snapshots other than v1.

## Output layout

`output/calibration/parameter_freeze/` is protected parameter evidence and is
never a model-result directory. New predictions are stored by task and
protocol:

```text
output/calibration/
  manifest.json                         # Locked snapshot and split-root identity.
  audit/                                # Inventories, protected hashes, verification reports.
  status/run_status.csv                 # Unit status, attempt count, and latest error.
  cache/features/...                    # Reusable features for the same canonical split.
  results/<task>/<protocol>/<split>/<model>/
    member_predictions.csv              # Seed ensemble or MC-dropout member predictions.
    predictive_quantiles.csv            # TabPFN q10/q50/q90 only; copied from verified OOD output.
    sample_uncertainty.csv               # Predictive mean, absolute error, and uncertainty per sample.
    metrics.json                         # Metrics recomputed from sample_uncertainty.csv.
    manifest.json                        # Snapshot, parameter, canonical split, and file hashes.
    logs/                                # Unit logs retained after successful promotion.
  results/<task>/<protocol>/aggregate/  # Protocol-pooled confidence tiers and error summaries.
  aggregate/                            # Cross-task concatenated calibration tables.
  failed_attempts/                      # Failed temporary runs; excluded from final aggregation.
```

Snapshot IDs are metadata, not path components. A changed frozen v1 file makes
resume fail and requires another protected reset/rebuild.

Ti-only frozen-snapshot pilot:

```powershell
python -m src.calibration.cli --stage ti-index --snapshot-version v1
python -m src.calibration.cli --stage ti-uncertainty --snapshot-version v1 --resume --ti-adapters reuse
python -m src.calibration.cli --stage ti-uncertainty --snapshot-version v1 --resume --ti-adapters llmprop
python -m src.calibration.cli --stage ti-uncertainty --snapshot-version v1 --resume --ti-adapters tabpfn
python -m src.calibration.cli --stage ti-calibration --snapshot-version v1
```

The combined command is:

```powershell
python -m src.calibration.cli --stage ti-pilot --snapshot-version v1 --resume
```

The pilot reads model parameters only from `frozen_model_params_v1.yaml`. The
snapshot contains one entry per task/model; protocol and split directories
provide data and prediction provenance only. Uncertainty uses repeated model
predictions, MC-dropout for LLMProp, and API predictive quantiles for TabPFN.
It never runs local TabPFN V2 and writes only to `output/calibration/pilots/ti/`.

Generated artifacts are written under `output/calibration/`. Historical
split-level snapshots remain evidence only and cannot override the active
task/model registry.

## Parameter selection

The default `select-best-parameters` mode is `existing_results_only`. It scans
all existing RandomCV trials in all five outer folds and aligns identical
parameters by canonical parameter hash, never by trial number. Validation
predictions provide the selection metric:

```text
MAE_fold = mean(abs(y_true - y_pred))
mean_available_MAE = mean(MAE_fold for folds where that exact parameter exists)
```

The lowest `mean_available_MAE` is frozen for each task/model. The snapshot
records `observed_fold_count` and `observed_fold_ids`; absent folds are not
imputed. R2 and RMSE are diagnostics. This mode performs no model retraining and
does not use Extrapolation, LOCO, Sparse, or other OOD metrics for selection.

Strict five-fold reevaluation remains available by setting:

```yaml
parameter_selection:
  mode: strict_five_fold
```

The missing-evaluation queue is resumable. Each completed candidate/fold writes
`output/calibration/parameter_evaluations/<task>/<model>/<parameter_hash>/fold_N/`
with `predictions/validation_predictions.csv` and `manifest.json`. The evaluator
uses only the RandomCV outer-train data and its inner validation folds; the OOD
test file is never used for parameter selection. After any batch, rerun
`select-best-parameters` to rebuild the score matrix. The v1 snapshot is written
only after all required candidate/fold scores are complete.

LLMProp fold checkpoints are temporary: the evaluator extracts validation
predictions and removes the checkpoints it created for that fold. This keeps
the resumable evidence small while retaining the CSV and manifest needed for
selection.

The optional `evaluate-missing-parameters` stage belongs only to strict mode.
It is not required for the default existing-results-only workflow.

## Data rule

Uncertainty must come from a model-provided uncertainty column or repeated
predictions for the same sample. The module never derives uncertainty from
absolute prediction error.

Add prediction files to `uncertainty.prediction_sources` in
`calibration.config.yaml`. Each source can define `metadata` and `column_map`
for task, protocol, split, model, sample, repeat, target, prediction, and
uncertainty columns.
