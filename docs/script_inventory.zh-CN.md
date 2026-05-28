# 脚本清单

所有 active 复现入口都只使用 Python。

## 核心 Python 模块

| 入口 | 用途 |
| --- | --- |
| `python -m src.pipelines.run_batch` | 标准传统 ML 与 BERT/NN 批量实验。 |
| `python -m src.pipelines.run_batch_ood` | OOD 批量实验。 |
| `python -m src.pipelines.run_cv_k_sweep` | RandomCV 与 LOCO k-sweep 实验。 |
| `python -m src.TabPFN.train_tabpfn` | TabPFN 基础回归。 |
| `python -m src.TabPFN.train_tabpfn_ood` | TabPFN 单任务或全量 OOD 实验。 |
| `python -m src.TabPFN.run_tabpfn_ood_batch` | TabPFN OOD 批量配置入口。 |
| `python -m src.LLMProp.run_llmprop_ood` | LLMProp 单个 OOD 任务入口。 |

## Active `Scripts/`

| 脚本 | 状态 |
| --- | --- |
| `batch_summarize_extrapolation_results.py` | active OOD 汇总工具。 |
| `batch_summarize_bert_extrapolation_results.py` | active OOD 汇总工具。 |
| `batch_summarize_tabpfn_extrapolation_results.py` | active OOD 汇总工具。 |
| `batch_summarize_llmprop_ood_results.py` | active OOD 汇总工具。 |
| `batch_summarize_combined_ood_reports.py` | active 综合报告工具。 |
| `build_bestplus_tabpfn_triptych.py` | active 绘图工具。 |

以下划线开头的 helper 模块仅供 active 脚本导入，不作为直接 CLI 入口。

## 已归档 Legacy 脚本

以下脚本已移动到 `archive/legacy_scripts/`：

- `copy_prior_models.py`
- `merge_references.py`
- `monitor_ood_repair_progress.py`
- `repair_ood_sparse_loco.py`
- `run_ood_missing_with_llm_env.py`
- `run_tabpfn25_ood_inconsistent_with_llm_env.py`
- `run_traditional_ood_final_missing_with_llm_env.py`
- `upsert_matbench_steel_summary.py`

它们用于追溯历史修复和迁移流程。部分脚本依赖旧本机路径、外部结果目录或历史 repair manifest，因此不是推荐复现入口。

其他历史 Scripts 工具已从 active 目录删除，因为它们不属于核心 OOD 汇总和最终图表链路。

## 禁止的包装脚本

项目不应新增或使用 `.ps1`、`.bat`、`.sh`、`.cmd` 包装脚本。请直接使用 Python 命令。
