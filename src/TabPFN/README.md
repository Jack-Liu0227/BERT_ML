# TabPFN for Alloy Property Prediction

This directory contains the TabPFN-based workflow used in this project for alloy mechanical property prediction.

## 中文速览

`src/TabPFN` 提供了面向合金性能预测的 TabPFN 工作流，覆盖基础回归与微调两类用法。

- 基础回归脚本：`python src/TabPFN/train_tabpfn.py`
- 微调脚本：`python src/TabPFN/finetune_tabpfn.py`
- 支持数据集：`Ti`、`Al`、`HEA`、`Steel`
- 常见预测目标：`YS(MPa)`、`UTS(MPa)`、`El(%)`
- 基础结果目录：`output/TabPFN_results/`
- 微调结果目录：`output/TabPFN_finetune_results/`

脚本当前采用交互式运行方式：

- 输入 `1` 运行全部实验
- 输入 `2` 选择单个合金和目标列

如果需要本地 checkpoint 微调，仓库根目录已包含 `tabpfn-v2-regressor.ckpt`，也可以通过环境变量 `TABPFN_REGRESSOR_MODEL_PATH` 指定自定义路径。

## 1. What is TabPFN

TabPFN (Tabular Prior-data Fitted Network) is a pretrained transformer for tabular learning. In this project it is used for small and medium-sized alloy datasets to predict targets such as `YS(MPa)`, `UTS(MPa)`, and `El(%)`.

Compared with conventional machine learning pipelines, TabPFN is suitable for:

- fast baseline experiments
- smaller tabular datasets
- quick comparison against XGBoost, CatBoost, random forest, and neural networks

This implementation uses `TabPFNRegressor` for direct regression. The default open model version is `v2`.

## 2. Supported datasets

- `Ti`: `UTS(MPa)`, `El(%)`
- `Al`: `UTS(MPa)`
- `HEA`: `YS(MPa)`, `UTS(MPa)`, `El(%)`
- `Steel`: `YS(MPa)`, `UTS(MPa)`, `El(%)`

Dataset paths, feature columns, target columns, and reference prediction CSV paths are defined in [`tabpfn_configs.py`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/tabpfn_configs.py).

## 3. File layout

```text
src/TabPFN/
|- __init__.py
|- data_processor.py                # Data loading, cleaning, split, scaling
|- tabpfn_configs.py               # Dataset and model configuration
|- train_tabpfn.py                 # Direct TabPFN regression experiments
|- finetune_tabpfn.py              # TabPFN fine-tuning experiments
|- prediction_alignment.py         # Align exported predictions by ID
|- align_predictions.py            # Alignment helpers
|- compare_finetune_results.py     # Fine-tune result comparison
`- README.md
```

## 4. Installation

Install project dependencies first:

```bash
pip install -r requirements.txt
```

If `tabpfn` is not available in your environment, install or upgrade it explicitly:

```bash
pip install -U tabpfn
```

Fine-tuning additionally requires PyTorch with a working CPU or CUDA runtime.

## 5. Quick start

Run from the project root:

```bash
cd src/TabPFN
python train_tabpfn.py
```

The script is interactive. It will ask you to choose:

- `1`: run all TabPFN experiments for all supported alloys and targets
- `2`: run one specific alloy-target experiment

Example interactive flow:

```text
Available options:
1. Run all experiments
2. Run single experiment

Enter your choice (1 or 2, default=1): 2
Enter alloy type (e.g., Ti): Ti
Enter target column: UTS(MPa)
```

## 6. Fine-tuning workflow

For fine-tuning, use:

```bash
cd src/TabPFN
python finetune_tabpfn.py
```

This script is also interactive:

- `1`: run all fine-tune experiments
- `2`: run one alloy-target fine-tune experiment

Default fine-tune parameters are defined in [`finetune_tabpfn.py`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/src/TabPFN/finetune_tabpfn.py):

- `device="cuda"`
- `epochs=300`
- `learning_rate=1e-5`
- `validation_split_ratio=0.1`
- `early_stopping=True`
- `early_stopping_patience=5`

If CUDA is unavailable, the script automatically falls back to CPU.

## 7. Local checkpoint usage

This repository already includes a local checkpoint file at:

- [`tabpfn-v2-regressor.ckpt`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/tabpfn-v2-regressor.ckpt)

`finetune_tabpfn.py` prefers a local open `v2` regressor checkpoint to avoid gated model downloads. It resolves the checkpoint in this order:

1. `TABPFN_REGRESSOR_MODEL_PATH` environment variable
2. `%APPDATA%\\tabpfn\\tabpfn-v2-regressor.ckpt`
3. `~/.cache/tabpfn/tabpfn-v2-regressor.ckpt`
4. `tabpfn-v2-regressor.ckpt` in the current project

If you want to force a custom checkpoint path:

```powershell
$env:TABPFN_REGRESSOR_MODEL_PATH="D:\path\to\tabpfn-v2-regressor.ckpt"
python src\TabPFN\finetune_tabpfn.py
```

## 8. Output files

Direct regression results are saved to:

- `output/TabPFN_results/<Alloy>/`
- `output/TabPFN_results/summary_results.csv`

Fine-tune results are saved to:

- `output/TabPFN_finetune_results/<Alloy>/`
- `output/TabPFN_finetune_results/summary_results.csv`
- `output/TabPFN_finetune_results/<Alloy>/checkpoints/`

Typical generated files include:

- `*_predictions.png`
- `*_all_predictions.csv`
- `*_epoch_metrics.csv` for fine-tuning
- `*_loss_curve.png` and `*_r2_curve.png` for fine-tuning

Prediction CSV files are aligned by `ID` when a reference CSV is configured, which makes cross-model comparison easier.

## 9. Programmatic usage

### Direct regression

```python
from pathlib import Path
from src.TabPFN.train_tabpfn import run_single_experiment

base_path = Path(".").resolve()

results = run_single_experiment(
    alloy_type="Ti",
    target_col="UTS(MPa)",
    base_path=str(base_path),
)
```

### Fine-tuning

```python
from pathlib import Path
from src.TabPFN.finetune_tabpfn import run_single_finetune_experiment

base_path = Path(".").resolve()

results = run_single_finetune_experiment(
    alloy_type="Steel",
    target_col="YS(MPa)",
    base_path=str(base_path),
    finetune_params={
        "device": "cuda",
        "epochs": 100,
        "learning_rate": 1e-5,
    },
)
```

## 10. Notes

- TabPFN is most suitable for relatively small tabular datasets.
- This module currently uses fixed dataset definitions from `tabpfn_configs.py`.
- The training scripts apply `scale=True` and `drop_na=True` by default.
- Exported predictions keep both train and test rows and include the `Dataset` column for separation.

## 11. Troubleshooting

### `ImportError: No module named 'tabpfn'`

```bash
pip install -U tabpfn
```

### Fine-tuning is unavailable

The installed `tabpfn` version may not include `tabpfn.finetuning`. Upgrade the package and confirm that PyTorch is installed correctly.

### CUDA errors

Use CPU instead:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
python src\TabPFN\finetune_tabpfn.py
```

### Checkpoint not found

Make sure [`tabpfn-v2-regressor.ckpt`](/D:/XJTU/ImportantFile/auto-design-alloy/BERT_ML/tabpfn-v2-regressor.ckpt) exists in the project root, or set `TABPFN_REGRESSOR_MODEL_PATH` to the correct file.
