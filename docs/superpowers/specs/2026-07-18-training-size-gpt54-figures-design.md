# Training-Size GPT-5.4 Figure Integration Design

## Scope

Extend the existing Figure 4 task-figure pipeline without changing trained-model outputs.
The existing `figures/` output remains the default BERT-inclusive view. A second
`figures_no_bert/` output presents the same eight tasks without `BERT-best`.

## GPT-5.4 adapter

The adapter scans a few-shot training-size root and converts complete result triples
into the current `degradation_ratio_by_repeat.csv` row contract:

- `inner_random` supplies `random_cv_mae`.
- `test_extrapolation_high20` supplies `top20_property_extreme` OOD MAE.
- `loco_farthest_cluster` supplies `top_tail_distant_loco` OOD MAE.
- `training_size_ood_split_summary.json` supplies task, fraction, repeat, snapshot,
  subset hash, provider, and model metadata.
- `completion_marker.json` and `metrics.json` provide completion and metric evidence.

The adapter writes a coverage audit and normalized GPT CSV. Strict mode includes only
full-test runs whose prediction counts match `csv/split_audit.csv`. Preview mode may
include completed sampled runs and labels the audit as partial.

## Figure behavior

The plotting layer accepts a dynamic ordered method list and method styles. The normal
entry point renders the five existing methods and appends `GPT-5.4` when eligible rows
exist. The no-BERT entry point removes `BERT-best` and appends GPT in the same way.
Both outputs use identical representative Traditional selection, bootstrap settings,
task filtering, Panel A source, and output layout.

## Compatibility

Without `--gpt-root`, existing commands and output content remain unchanged. An
incomplete GPT root does not contaminate formal curves: the figures render available
established methods and the audit explains why GPT rows were excluded. No model
training, prediction calls, or checkpoint operations are performed.
