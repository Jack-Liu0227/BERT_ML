# Script Inventory

All active reproduction entrypoints are Python-only.

## Core Python Modules

| Entrypoint | Purpose |
| --- | --- |
| `python -m src.pipelines.run_batch` | Standard traditional ML and BERT/NN batch experiments. |
| `python -m src.pipelines.run_batch_ood` | OOD batch experiments. |
| `python -m src.pipelines.run_cv_k_sweep` | RandomCV and LOCO k-sweep experiments. |
| `python -m src.TabPFN.train_tabpfn` | TabPFN direct regression. |
| `python -m src.TabPFN.train_tabpfn_ood` | TabPFN single/all OOD runs. |
| `python -m src.TabPFN.run_tabpfn_ood_batch` | TabPFN OOD batch configs. |
| `python -m src.LLMProp.run_llmprop_ood` | LLMProp single OOD run. |

## Active `Scripts/`

| Script | Status |
| --- | --- |
| `batch_summarize_extrapolation_results.py` | Active OOD summary utility. |
| `batch_summarize_bert_extrapolation_results.py` | Active OOD summary utility. |
| `batch_summarize_tabpfn_extrapolation_results.py` | Active OOD summary utility. |
| `batch_summarize_llmprop_ood_results.py` | Active OOD summary utility. |
| `batch_summarize_combined_ood_reports.py` | Active combined reporting utility. |
| `build_bestplus_tabpfn_triptych.py` | Active figure builder. |

Helper modules beginning with `_` are implementation details used by active scripts.

## Archived Legacy Scripts

The following scripts were moved to `archive/legacy_scripts/`:

- `copy_prior_models.py`
- `merge_references.py`
- `monitor_ood_repair_progress.py`
- `repair_ood_sparse_loco.py`
- `run_ood_missing_with_llm_env.py`
- `run_tabpfn25_ood_inconsistent_with_llm_env.py`
- `run_traditional_ood_final_missing_with_llm_env.py`
- `upsert_matbench_steel_summary.py`

They are retained for traceability. Several depend on historical local paths, external result folders, or repair manifests, so they are not recommended reproduction entrypoints.

Other historical Scripts utilities were removed from the active tree because they are not part of the core OOD summary and final-figure pipeline.

## Disallowed Script Wrappers

The project should not add or use `.ps1`, `.bat`, `.sh`, or `.cmd` wrappers. Use Python commands directly.
