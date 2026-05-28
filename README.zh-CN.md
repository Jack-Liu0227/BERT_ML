# BERT_ML：合金性能预测实验

[English](./README.md)

本仓库包含面向合金性能预测的 Python 工作流，覆盖传统机器学习、BERT 嵌入、TabPFN 和 LLMProp。项目按“全部实验可复现”整理：所有可运行入口均为 Python 模块或 Python 脚本，基础模型资产统一放在 `models/`，历史修复脚本单独归档。

## 快速开始

在项目根目录创建环境：

```bash
conda env create -f environment.yml
conda activate llm
```

或使用 pip：

```bash
pip install -r requirements.txt
```

如果本地没有模型资产，先下载：

```bash
python models/download_bert_models.py
python models/download_llmprop_models.py
```

`SteelBERT` 是 Hugging Face 受限模型，需要先运行 `huggingface-cli login`，并在 [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT) 申请访问权限。

## 仅使用 Python 入口

项目不使用 PowerShell、bat、sh 或 cmd 包装脚本。批量实验统一用 Python 命令：

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --all --dry_run
```

OOD 实验：

```bash
python -m src.pipelines.run_batch_ood --list
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation --dry_run
python -m src.pipelines.run_cv_k_sweep --dry_run
```

TabPFN 实验：

```bash
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
python -m src.TabPFN.run_tabpfn_ood_batch --list
```

LLMProp OOD 单任务入口：

```bash
python -m src.LLMProp.run_llmprop_ood --help
```

## 模型资产

模型资产统一放在 `models/`：

- `models/matscibert/`
- `models/scibert/`
- `models/steelbert/`
- `models/llmprop/`
- `models/tabpfn/tabpfn-v2-regressor.ckpt`

训练过程中生成的 checkpoint 仍保存在 `output/**/checkpoints/`，因为它们是实验输出，不是基础模型资产。

## 数据与输出

已跟踪输入数据位于 `datasets/`。当前配置实际引用的标准化 CSV 包括：

- `datasets/Ti_alloys/titanium.csv`
- `datasets/Al_Alloys/aluminum.csv`
- `datasets/HEA_data/hea.csv`
- `datasets/Steel/steel.csv`
- `datasets/HEA_data/Pitting_potential_data_xiongjie.csv`
- `datasets/matbench_convert/matbench_steels.csv`
- `datasets/matbench_convert/matbench_steels_ood.csv`

生成特征、日志、图和模型输出写入被忽略的目录，例如 `Features/`、`Features_extrapolation/`、`output/` 和 `logs/`。

## 汇总与绘图脚本

当前仍在使用的分析和绘图脚本保留在 `Scripts/`。示例：

```bash
python Scripts/batch_summarize_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_bert_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results
python Scripts/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
```

外部 OOD 结果源由 `Scripts/external_ood_model_sources.yaml` 配置。如需合并外部 LLM 结果，运行 combined summary 前设置 `EXTERNAL_OOD_ROOT`。

## 文档

- [英文复现指南](./docs/reproducibility.md)
- [中文复现指南](./docs/reproducibility.zh-CN.md)
- [英文脚本清单](./docs/script_inventory.md)
- [中文脚本清单](./docs/script_inventory.zh-CN.md)
- [批量运行指南](./src/pipelines/BATCH_RUN_GUIDE.md)
- [TabPFN 指南](./src/TabPFN/README.md)

历史本机修复和迁移工具保存在 `archive/legacy_scripts/`。它们不是推荐复现入口。
