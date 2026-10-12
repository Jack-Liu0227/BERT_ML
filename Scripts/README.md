# Scripts

Run commands from the repository root. `Scripts/` keeps the data-processing
entrypoints: result summarization, OOD/space analysis, result-table rebuilds,
and the figure pipelines that are wired into the training code. Reusable helpers
live in `Scripts/lib/` and YAML configuration lives under `configs/`.

## Scope

Scripts in this folder are expected to read experiment outputs and write
machine-readable results (`.csv`, `.json`), plus the figure PNG/PDF files that
the reporting pipelines already produce.

Deliverable-only generators were removed on purpose. There is no longer any
script here that builds a Word document, a presentation, an Excel workbook, or a
standalone HTML viewer/gallery. Layout and slide decks are produced outside this
repository from the CSV/PNG outputs. Removed entrypoints (still reachable via
git history):

| Removed script | Why |
| --- | --- |
| `analysis/build_top_tail_loco_sensitivity_docx.py` | generated a `.docx` method note |
| `figures/export_calibration_figure3_data.py` | built a standalone HTML delivery viewer |
| `figures/ood_figures_3_4/build_figure4_workbooks.mjs` | generated Excel workbooks |
| `figures/ood_figures_3_4/build_training_size_gallery.py` | generated an HTML gallery |
| `figures/ood_figures_3_4/export_figure4_data_viewer.py` | generated an HTML/Excel viewer |
| `maintenance/build_hybrid_data_viewer.py` | generated an HTML viewer |
| `maintenance/build_triptych_viewer.py` | generated an HTML viewer |
| `maintenance/sync_hybrid_confidence_results.py` | figure sync that also rebuilt the HTML viewer |
| `ood_w_error_rebuild/export_pure_ood_yz_table.py` | exported a Word-ready HTML table preview |

## Directory Layout

- `reporting/`: OOD summary and combined-report entrypoints.
- `analysis/`: split scoring, X-space/sample-W export, top-tail, and diagnostic analysis.
- `figures/`: figure builds that remain part of the reproducible pipeline (including the Figure 3/4 modules used by `src/training_size_ood`).
- `maintenance/`: catalog and compact-hierarchy rebuild utilities for published results.
- `ood_w_error_rebuild/`: W/error result rebuilding, join audits, and W-severity publication tables.
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

### `figures/build_ood_figures_3_4.py`

Audits, tabulates, and renders the OOD calibration/training-size figures.

Default config:
`configs/figures/ood_figures_3_4.config.yaml`

```bash
python Scripts/figures/build_ood_figures_3_4.py
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

### `analysis/generate_ood_splits.py`

Pre-generates the canonical OOD split caches shared by every model family, so
parallel training workers never race on the same split-cache directory.

```bash
python -m Scripts.analysis.generate_ood_splits --alloys Aluminum_2
python -m Scripts.analysis.generate_ood_splits --alloys Aluminum_2 --execute
```

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

- Helper modules in `Scripts/lib/` are implementation details. Note that
  `.gitignore` currently excludes `lib/`, so `Scripts/lib/` must be force-added
  (`git add -f Scripts/lib`) to keep a fresh clone runnable.
- YAML configuration belongs under `configs/`, not `Scripts/`.
- Historical one-off utilities should be archived in `archive/legacy_scripts/`.
- Do not add `.ps1`, `.bat`, `.sh`, or `.cmd` wrappers; use Python commands directly.
