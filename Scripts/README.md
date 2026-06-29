# Scripts

Run commands from the repository root. `Scripts/` now keeps executable Python
entrypoints grouped by workflow, while reusable helpers live in `Scripts/lib/`
and YAML configuration lives under `configs/`.

## Directory Layout

- `reporting/`: OOD summary and combined-report entrypoints.
- `figures/`: final figure and W-error visualization entrypoints.
- `analysis/`: X-space, sample-W, top-tail, and diagnostic analysis entrypoints.
- `lib/`: private helper modules imported by scripts; do not run these directly.

## Core OOD Reporting Workflow

```bash
python Scripts/reporting/batch_summarize_extrapolation_results.py
python Scripts/reporting/batch_summarize_bert_extrapolation_results.py
python Scripts/reporting/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/reporting/batch_summarize_llmprop_ood_results.py
python Scripts/reporting/batch_summarize_combined_ood_reports.py
python Scripts/figures/build_bestplus_tabpfn_triptych.py --config configs/figures/build_bestplus_tabpfn_triptych.paper.config.yaml
```

## Reporting Scripts

### `reporting/batch_summarize_extrapolation_results.py`

Summarizes Traditional-ML OOD results into the canonical schema.

```bash
python Scripts/reporting/batch_summarize_extrapolation_results.py
python Scripts/reporting/batch_summarize_extrapolation_results.py --base-dir output/ood_results --output-dir output/ood_summary_reports/Traditional
```

### `reporting/batch_summarize_bert_extrapolation_results.py`

Summarizes BERT/NN OOD results into the canonical schema.

```bash
python Scripts/reporting/batch_summarize_bert_extrapolation_results.py
python Scripts/reporting/batch_summarize_bert_extrapolation_results.py --success-only
python Scripts/reporting/batch_summarize_bert_extrapolation_results.py --tracked-config experiment2a_all_nn_scibert_extrapolation
```

### `reporting/batch_summarize_tabpfn_extrapolation_results.py`

Summarizes TabPFN OOD results into the canonical schema.

```bash
python Scripts/reporting/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/reporting/batch_summarize_tabpfn_extrapolation_results.py --base-dir output/ood_results_TabPFN-2.5-Plus-Numeric
python Scripts/reporting/batch_summarize_tabpfn_extrapolation_results.py --base-dirs output/ood_results_TabPFN-2.5-Plus-Numeric output/ood_results_TabPFN-2.5-Plus-Text
```

### `reporting/batch_summarize_llmprop_ood_results.py`

Summarizes LLMProp OOD results into the canonical schema.

```bash
python Scripts/reporting/batch_summarize_llmprop_ood_results.py
python Scripts/reporting/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results --output-dir output/ood_summary_reports/LLMProp
```

### `reporting/batch_summarize_combined_ood_reports.py`

Merges Traditional, BERT, TabPFN, LLMProp, and optional external OOD summaries.

Default external config:
`configs/ood_reporting/external_ood_model_sources.yaml`

```bash
python Scripts/reporting/batch_summarize_combined_ood_reports.py
python Scripts/reporting/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports --output-dir output/ood_summary_reports/Combined
```

On Windows PowerShell, enable optional external results before running:

```text
$env:EXTERNAL_OOD_ROOT="D:\path\to\external\ood\results"
python Scripts/reporting/batch_summarize_combined_ood_reports.py
```

## Figure Scripts

### `figures/build_bestplus_tabpfn_triptych.py`

Builds final per-task triptych figures from the combined canonical summary.

Default configs:

- `configs/figures/build_bestplus_tabpfn_triptych.config.yaml`
- `configs/figures/build_bestplus_tabpfn_triptych.paper.config.yaml`

```bash
python Scripts/figures/build_bestplus_tabpfn_triptych.py
python Scripts/figures/build_bestplus_tabpfn_triptych.py --config configs/figures/build_bestplus_tabpfn_triptych.paper.config.yaml
python Scripts/figures/build_bestplus_tabpfn_triptych.py --summary-csv output/ood_summary_reports/Combined/data/all_model_families_ood_summary.csv --output-dir output/ood_summary_reports/Combined/figure/per_task_bestplus_tabpfn_paper
```

### `figures/plot_w_error_relationships.py`

Builds per-sample W/error relationship figures and long-form tables.

```bash
python Scripts/figures/plot_w_error_relationships.py --scope both
python Scripts/figures/plot_w_error_relationships.py --scope hybrid --case-contains Al__aluminum
```

### `figures/plot_hybrid_method_visualizations.py`

Builds method-aware Hybrid Set A/B W-severity visualizations.

```bash
python Scripts/figures/plot_hybrid_method_visualizations.py
python Scripts/figures/plot_hybrid_method_visualizations.py --overview-only
```

### `figures/plot_ood_xspace_figures.py`

Builds figures from X-space OOD score outputs.

```bash
python Scripts/figures/plot_ood_xspace_figures.py
```

## Analysis Scripts

### `analysis/analyze_top_tail_loco_proximity.py`

Analyzes top-tail proximity to LOCO clusters.

Default config:
`configs/analysis/top_tail_loco_proximity_outliers.yaml`

```bash
python Scripts/analysis/analyze_top_tail_loco_proximity.py
python Scripts/analysis/analyze_top_tail_loco_proximity.py --outlier-policy steel_core
```

### `analysis/score_ood_splits.py`

Scores canonical OOD split geometry.

```bash
python Scripts/analysis/score_ood_splits.py
```

### `analysis/export_ood_sample_w_reports.py`

Exports sample-level W reports from OOD score outputs.

```bash
python Scripts/analysis/export_ood_sample_w_reports.py
```

### `analysis/export_three_space_w_ood_report.py`

Combines X/Y/Z-space sample-W reports.

```bash
python Scripts/analysis/export_three_space_w_ood_report.py
```

### `analysis/export_wx_v2_sample_w_reports.py`

Builds sample-W reports using Wx v2 mixed-OT scores.

```bash
python Scripts/analysis/export_wx_v2_sample_w_reports.py
```

### `analysis/generate_matbench_embedding_background.py`

Generates Matbench Steel embedding background data.

```bash
python Scripts/analysis/generate_matbench_embedding_background.py
```

### `analysis/summarize_w_error_trend_slopes.py`

Summarizes W/error trend slopes from generated long tables.

```bash
python Scripts/analysis/summarize_w_error_trend_slopes.py
```

## Notes

- Helper modules in `Scripts/lib/` are implementation details.
- YAML configuration belongs under `configs/`, not `Scripts/`.
- Historical one-off utilities should be archived in `archive/legacy_scripts/`.
- Do not add `.ps1`, `.bat`, `.sh`, or `.cmd` wrappers; use Python commands directly.
