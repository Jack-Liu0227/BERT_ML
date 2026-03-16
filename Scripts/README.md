# Scripts

这个目录放项目里的批处理脚本和使用说明，便于统一管理和直接调用。

## 当前脚本

| 文件 | 用途 |
| --- | --- |
| `merge_references.py` | 按 `ID` 将原始数据集里的参考文献信息补回到目标 CSV。 |
| `select_best_model_and_plot.py` | 从常规模型结果中筛选最佳模型，生成代表性结果和汇总图。 |
| `batch_calculate_bert_global_mean.py` | 批量处理 BERT 模型 Optuna 结果，选择最接近全局均值的代表 fold。 |
| `batch_summarize_extrapolation_results.py` | 批量汇总 `output\extrapolation_results`，比较各模型的外推测试表现并导出最佳模型结果。 |
| `optuna_plotting_scripts_usage.md` | 旧有 Optuna 绘图与分析脚本的详细说明。 |

## 使用方式

建议在项目根目录 `D:\XJTU\ImportantFile\auto-design-alloy\BERT_ML` 下执行。

```bash
python Scripts/select_best_model_and_plot.py
python Scripts/batch_calculate_bert_global_mean.py --base_dir "output\new_results_withuncertainty"
python Scripts/batch_summarize_extrapolation_results.py --base-dir "output\extrapolation_results"
```

## 外推结果汇总脚本

`batch_summarize_extrapolation_results.py` 会自动递归查找：

```text
output\extrapolation_results\<alloy_family>\<dataset>\<target>\<feature_mode>\model_comparison
```

对每个 case，它会：

- 读取每个模型的 `final_model_evaluation_metrics.json`
- 读取 `closest_to_global_mean_trial_fold\closest_to_global_mean_trial_fold_metrics.json`
- 生成单个 case 的模型对比图
- 选择 `final_test_r2` 最好的模型
- 复制最佳模型的最终预测、代表性外推测试结果和对应图片
- 在 `output\extrapolation_results\all_extrapolation_summary` 下输出总汇总 CSV

主要输出包括：

- `ALL_EXTRAPOLATION_MODEL_SUMMARY.csv`
- `ALL_BEST_EXTRAPOLATION_MODELS.csv`
- `BEST_MODEL_FINAL_TEST_R2_PIVOT.csv`
- 每个 case 下的 `extrapolation_model_summary.csv`
- 每个 case 下的 `best_extrapolation_model_<target>.csv`

## 说明

- 这些脚本原先在 `src` 下，现已迁移到 `Scripts`。
- `optuna_plotting_scripts_usage.md` 仍然可作为历史脚本说明参考，但命令路径请改为 `python Scripts/...`。
## Additional Script

- `batch_summarize_tabpfn_extrapolation_results.py`
  Summarize `output\extrapolation_results_tabpfn`, export train/test CSV summaries,
  group results by target, and generate interactive overview HTML plots.

Example:
```bash
python Scripts/batch_summarize_tabpfn_extrapolation_results.py --base-dir "output\extrapolation_results_tabpfn"
```

Main outputs:
- `output\extrapolation_results_tabpfn\all_tabpfn_extrapolation_summary\TABPFN_EXTRAPOLATION_TRAIN_SUMMARY.csv`
- `output\extrapolation_results_tabpfn\all_tabpfn_extrapolation_summary\TABPFN_EXTRAPOLATION_TEST_SUMMARY.csv`
- `output\extrapolation_results_tabpfn\all_tabpfn_extrapolation_summary\by_target\*.csv`
- `output\extrapolation_results_tabpfn\all_tabpfn_extrapolation_summary\plots\overview\*.html`
- `output\extrapolation_results_tabpfn\all_tabpfn_extrapolation_summary\plots\overview\overview_dashboard.html`
