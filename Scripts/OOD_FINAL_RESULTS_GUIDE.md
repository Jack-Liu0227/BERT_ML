# OOD Final Results Guide

This guide explains how to refresh the final OOD result packages after GPT
prediction/confidence data changes.

## Final Result Folders

- Standard OOD:
  `D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD`
- Hybrid OOD:
  `D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD`

The folder name `HYBIRD` is kept because the existing final-paper artifact
directory uses this spelling.

## Set A / Set B Definition

Hybrid Set A and Set B are fixed as:

- Set A = `test_extrapolation_high20` = top20% extrapolation samples.
- Set B = `test_inner_ood` = inner OOD samples.

Do not infer Set A/B only from a folder name. Hybrid visualizations should use
`source_split_dir\test_sets\test_extrapolation_high20.csv` and
`source_split_dir\test_sets\test_inner_ood.csv` to relabel sample IDs when
those files are available.

## Dependency Map

### GPT-dependent outputs

Refresh these when GPT prediction or confidence files change:

- `confidence_ood_relationship`
- `triptych`
- `triptych_randomcv` in standard OOD when GPT participates in the comparison
- `per_task_extreme_ood_model_analysis` when GPT/LLM performance rows change
- `w_error_relationship`
- `hybrid_method_visualizations`

### Mostly split/W-distance-dependent outputs

These usually do not change when only GPT predictions change, but should be
validated or regenerated for a consistent final package:

- `panel_distribution`
- `wasserstein_distance_standardized`

## Important Figure Semantics

- Hybrid `figure_1_severity_decomposition` uses `W_p90`.
- Hybrid `figure_8_yz_ood_map` uses `W_mean`.
- In `figure_8_yz_ood_map`, x = `Wy_mean` from `Y-space` `W_mean`, and y =
  `Wz_mean` from `Z-space` `W_mean`.
- In `figure_8_yz_ood_map`, Set A/Set B are overlaid in one shared coordinate
  system. Marker shape encodes the set, and color encodes the method.

## Recommended Run Order

Run commands from `D:\XJTU\ImportantFile\auto-design-alloy\BERT_ML` unless
noted otherwise.

1. Confirm GPT source files are present.

   Standard confidence analysis expects:

   ```powershell
   D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\output\ood\k5\<strength_ood_*>\<task>\[fold_*]\openai\gpt-5.4\predictions.csv
   ```

2. Refresh standard OOD confidence tables if the source structure above is
   populated.

   ```powershell
   python D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\scripts\analyze_confidence_ood_relationship.py `
     --input-root D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\output\ood\k5 `
     --ood-analysis-dir "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\per_task_extreme_ood_model_analysis" `
     --output-dir "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\confidence_ood_relationship"
   ```

3. Redraw standard OOD confidence figures from the current CSV tables.

   ```powershell
   python D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided\scripts\build_npj_confidence_ood_figures.py `
     --input-dir "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\confidence_ood_relationship\csv" `
     --output-dir "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\confidence_ood_relationship\figures\npj" `
     --final-summary-layout `
     --clean-output-dir
   ```

4. Refresh standard OOD triptych artifacts.

   ```powershell
   python "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\triptych\Plot.py"
   python "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\triptych_randomcv\Plot.py"
   ```

5. Validate or refresh standard OOD split/W-only artifacts.

   ```powershell
   python "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\panel_distribution\Plot.py"
   python "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\wasserstein_distance_standardized\Plot.py"
   ```

6. Refresh the Hybrid OOD final package.

   Run from `D:\XJTU\ImportantFile\auto-design-alloy\fewshot-guided`:

   ```powershell
   python scripts\build_ood_hybird_results.py `
     --clean `
     --bert-combined-summary-root D:\XJTU\ImportantFile\auto-design-alloy\BERT_ML\output\ood_summary_reports_hybrid\Combined `
     --output-root "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD"
   ```

7. Refresh W-error relationship artifacts.

   Full outputs can be slow because they join per-sample W values to many model
   prediction files and render many figures. For figure-8-only work, the hybrid
   method visualizer only needs `test_extrapolation_high20\csv\w_error_samples_long.csv`
   and `test_inner_ood\csv\w_error_samples_long.csv`.

   ```powershell
   python Scripts\plot_w_error_relationships.py `
     --scope both `
     --ood-output "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\w_error_relationship" `
     --hybrid-output "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD\w_error_relationship" `
     --formats png pdf svg
   ```

   Faster Hybrid CSV-focused run:

   ```powershell
   python Scripts\plot_w_error_relationships.py `
     --scope hybrid `
     --hybrid-output "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD\w_error_relationship" `
     --formats png `
     --phase-diagram-spaces none
   ```

8. Refresh Hybrid method visualizations.

   ```powershell
   python Scripts\plot_hybrid_method_visualizations.py `
     --hybrid-root "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD\w_error_relationship" `
     --pure-root "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD\w_error_relationship" `
     --triptych-root "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD\triptych" `
     --output-dir "D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\最终结果图\OOD HYBIRD\hybrid_method_visualizations" `
     --formats png pdf svg
   ```

## Validation Checklist

Run:

```powershell
pytest tests\test_plot_w_error_relationships.py tests\test_plot_hybrid_method_visualizations.py -q
```

Check:

- `OOD HYBIRD\manifest.json`
- `OOD HYBIRD\validation_summary.json`
- `OOD HYBIRD\hybrid_method_visualizations\csv\figure_manifest.csv`
- `OOD HYBIRD\hybrid_method_visualizations\csv\hybrid_set_ab_yz_ood_map_summary.csv`
- `OOD HYBIRD\hybrid_method_visualizations\data\Set_A\hybrid_yz_ood_map_summary.csv`
- `OOD HYBIRD\hybrid_method_visualizations\data\Set_B\hybrid_yz_ood_map_summary.csv`
- `OOD\confidence_ood_relationship\csv\confidence_by_prediction_file.csv`
- `OOD HYBIRD\confidence_ood_relationship\csv\confidence_by_prediction_file.csv`

For `Ti-UTS` hybrid figure 8:

- Set A must correspond to `test_extrapolation_high20`.
- Set B must correspond to `test_inner_ood`.
- The figure should have one data axes.
- Set A should use circle markers.
- Set B should use square markers.
- There should be 6 method-level Set A points and 6 method-level Set B points
  after method aggregation.

## Common Problems

- Old caches: `output\ood_xspace_scores\hybrid_w_error_cache` can preserve old W
  tables. Use `--force-recompute-hybrid-w` only when split files or W feature
  generation changed; it can be slow.
- AB reversed: always inspect `source_split_dir\test_sets\*.csv` when a plot
  looks inverted.
- Mixed output roots: avoid mixing project-local `output\...` copies with final
  manuscript folders unless you deliberately use a known complete cache as an
  input.
- Standard confidence source missing: if `analyze_confidence_ood_relationship.py`
  reports no `openai/gpt-5.4` predictions, the expected `strength_ood_*` input
  tree has not been populated.
- `HYBIRD` spelling: final paths intentionally use `OOD HYBIRD`.

## Latest Refresh Notes

The 2026-06-24 refresh rebuilt the Hybrid OOD final package with
`build_ood_hybird_results.py --clean`. Its `validation_summary.json` passed all
checks, including required main-source completeness and Matbench Steel-YS
GPT-5.4 rows.

The standard OOD triptych and triptych_randomcv figures were regenerated.
Standard OOD confidence figures were redrawn from the existing CSV tables.
However, standard OOD confidence bottom tables were not recomputed because the
expected `fewshot-guided\output\ood\k5\...\openai\gpt-5.4\predictions.csv`
source tree was not present.

Standard OOD `panel_distribution\Plot.py` exposed a source schema issue in the
Matbench panel CSV: missing `display_name`, `group_label`, and `method`.
Standard OOD `wasserstein_distance_standardized\Plot.py` exposed a missing
source summary file:
`D:\XJTU\已完成论文数据汇总\Fewshot\预处理汇总数据\OOD\comparison_outputs\wasserstein_distance_standardized\ood_wasserstein_summary.csv`.

Full Hybrid figure rendering in `plot_w_error_relationships.py` was too slow
for an interactive refresh, so the script now supports `--csv-only`. The final
Hybrid `w_error_relationship` folder was refreshed with `--csv-only`, producing
complete `combined`, `test_extrapolation_high20`, and `test_inner_ood` CSV
tables in the final `OOD HYBIRD` directory. The final
`hybrid_method_visualizations` folder was then regenerated from those final
W-error CSV tables plus the freshly rebuilt final Hybrid triptych folder.
