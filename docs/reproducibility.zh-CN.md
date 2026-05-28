# 复现指南

本文档记录项目的 Python-only 复现实验流程。

## 1. 环境

创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate llm
```

或在已有环境中安装依赖：

```bash
pip install -r requirements.txt
```

`environment.yml` 已移除本机 `prefix`，可在其他机器复用。

## 2. 模型资产

可复用模型资产统一放在 `models/`。

当前支持的位置：

- `models/matscibert/`
- `models/scibert/`
- `models/steelbert/`
- `models/llmprop/`
- `models/tabpfn/tabpfn-v2-regressor.ckpt`

缺少 BERT 或 LLMProp 资产时运行：

```bash
python models/download_bert_models.py
python models/download_llmprop_models.py
```

`SteelBERT` 需要 Hugging Face 访问权限。TabPFN 微调 checkpoint 的搜索顺序是：

1. `TABPFN_REGRESSOR_MODEL_PATH`
2. `models/tabpfn/tabpfn-v2-regressor.ckpt`
3. 用户级 TabPFN cache 路径
4. legacy 根目录 checkpoint 路径

## 3. 标准实验

查看和运行标准实验：

```bash
python -m src.pipelines.run_batch --list
python -m src.pipelines.run_batch --all --dry_run
python -m src.pipelines.run_batch --all
```

运行指定配置：

```bash
python -m src.pipelines.run_batch --config experiment1_all_ml_models
python -m src.pipelines.run_batch --config experiment2a_all_nn_scibert experiment2b_all_nn_steelbert
```

## 4. OOD 实验

查看和运行 OOD 实验：

```bash
python -m src.pipelines.run_batch_ood --list
python -m src.pipelines.run_batch_ood --all --dry_run
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation
```

运行 k-sweep：

```bash
python -m src.pipelines.run_cv_k_sweep --dry_run
```

## 5. TabPFN 和 LLMProp

TabPFN 基础和 OOD 实验：

```bash
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
python -m src.TabPFN.run_tabpfn_ood_batch --list
```

LLMProp 单个 OOD 入口：

```bash
python -m src.LLMProp.run_llmprop_ood --help
```

## 6. 汇总与绘图

使用 `Scripts/` 下的 active Python 脚本：

```bash
python Scripts/batch_summarize_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_bert_extrapolation_results.py --base-dir output/ood_results
python Scripts/batch_summarize_tabpfn_extrapolation_results.py
python Scripts/batch_summarize_llmprop_ood_results.py --base-dir output/ood_results
python Scripts/batch_summarize_combined_ood_reports.py --reports-root output/ood_summary_reports
python Scripts/build_bestplus_tabpfn_triptych.py --config Scripts/build_bestplus_tabpfn_triptych.paper.config.yaml
```

只有需要合并外部 LLM OOD 结果时，才设置 `EXTERNAL_OOD_ROOT`。

## 7. 输出

以下生成物默认被忽略：

- `Features/`
- `Features_extrapolation/`
- `output/`
- `logs/`
- `.batch_progress*.json`

训练 checkpoint 保留在各自 run 的 output 目录下。除非它们成为可复用基础模型资产，否则不移动到 `models/`。
