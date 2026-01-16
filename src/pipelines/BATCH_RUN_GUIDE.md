# 批量运行指南 / Batch Run Guide

本指南介绍如何使用批量运行脚本来运行多个合金类型的训练任务。

## 重构说明

**重要变更**：批量运行脚本已重构，现在使用统一的 `run_batch.py` 脚本。

### 文件结构

- `src/pipelines/batch_configs.py`: 配置文件（包含 `ALLOY_CONFIGS` 和 `BATCH_CONFIGS`）
- `src/pipelines/run_batch.py`: 统一的批量运行脚本
- `src/pipelines/end_to_end_pipeline.py`: 单个流水线脚本
- `run_all_experiments.bat`: Windows批处理脚本
- `run_all_experiments.sh`: Linux/Mac shell脚本

### 配置分离

- **ALLOY_CONFIGS**: 合金数据配置（数据文件路径、目标列、工艺参数等）
- **BATCH_CONFIGS**: 批量运行任务配置（实验参数、模型选择等）

## 使用方法

### 方法1: 列出所有可用配置

```bash
python -m src.pipelines.run_batch --list
```

输出示例：
```
可用的批量运行配置 / Available Batch Configurations:

  * experiment1_all_ml_models
    【实验1】所有合金 × 5个传统ML模型对比（XGBoost, RF, MLP, LightGBM, CatBoost）

  * experiment2a_all_nn_scibert
    【实验2a】所有合金 × 神经网络 + SciBERT嵌入

  * experiment2b_all_nn_steelbert
    【实验2b】所有合金 × 神经网络 + SteelBERT嵌入

  * experiment2c_all_nn_matscibert
    【实验2c】所有合金 × 神经网络 + MatSciBERT嵌入
```

### 方法2: 运行预定义配置

```bash
# 运行单个配置
python -m src.pipelines.run_batch --config experiment1_all_ml_models

# 运行多个配置
python -m src.pipelines.run_batch --config experiment1_all_ml_models experiment2a_all_nn_scibert
```

### 方法3: 运行完整实验

```bash
# 运行所有实验（实验1 + 实验2）= 48个任务
python -m src.pipelines.run_batch --all

# 仅运行实验1（传统ML模型）= 30个任务
python -m src.pipelines.run_batch --experiment1

# 仅运行实验2（神经网络 + BERT嵌入）= 18个任务
python -m src.pipelines.run_batch --experiment2
```

### 方法4: 预览命令（不实际执行）

```bash
python -m src.pipelines.run_batch --all --dry_run
```

### 方法5: 自定义参数运行

```bash
python -m src.pipelines.run_batch --custom \
    --embedding_type steelbert \
    --use_composition_feature \
    --use_element_embedding \
    --use_process_embedding \
    --use_nn \
    --cross_validate \
    --num_folds 9 \
    --use_optuna \
    --n_trials 30 \
    --evaluate_after_train
```

#### 自定义参数示例

**运行特定合金类型**：
```bash
python -m src.pipelines.run_batch --custom \
    --alloy_types Ti Al \
    --embedding_type tradition \
    --use_composition_feature \
    --models xgboost lightgbm \
    --cross_validate \
    --num_folds 9
```

**排除特定合金类型**：
```bash
python -m src.pipelines.run_batch --custom \
    --exclude_alloys HEA_full HEA_half \
    --embedding_type matscibert \
    --use_composition_feature \
    --use_element_embedding \
    --use_process_embedding \
    --use_nn
```

## 实验说明

### 实验1: 传统ML模型对比
- **合金类型**: Ti, Al, HEA_full, HEA_half, Nb, Steel（6个）
- **模型**: xgboost, sklearn_rf, mlp, lightgbm, catboost（5个）
- **嵌入类型**: tradition（仅组分特征）
- **特征配置**: use_composition_feature=True, use_element_embedding=False, use_process_embedding=False
- **训练方式**: cross_validate=True, num_folds=9
- **总任务数**: 6 × 5 = 30个任务

### 实验2: 神经网络 + BERT嵌入对比
- **合金类型**: Ti, Al, HEA_full, HEA_half, Nb, Steel（6个）
- **模型**: 神经网络（AlloyNN）
- **嵌入类型**: scibert, steelbert, matscibert（3种）
- **特征配置**: use_composition_feature=True, use_element_embedding=True, use_process_embedding=True
- **训练方式**: cross_validate=True, num_folds=9, use_optuna=True, n_trials=30
- **总任务数**: 6 × 3 = 18个任务

### 总计
- **总任务数**: 30 + 18 = 48个训练任务

## 支持的合金类型

| 合金类型 | 描述 | 数据文件 |
|---------|------|---------|
| `Ti` | 钛合金 | `datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv` |
| `Al` | 铝合金 | `datasets/Al_Alloys/USTB/USTB_Al_alloys_processed_withID.csv` |
| `HEA_full` | 高熵合金（完整数据集） | `datasets/HEA_data/RoomTemperature_HEA_with_ID.csv` |
| `HEA_half` | 高熵合金（一半数据集） | `datasets/HEA_data/RoomTemperature_HEA_train_with_ID.csv` |
| `Nb` | 铌合金 | `datasets/Nb_Alloys/Nb_cleandata/Nb_clean_with_processing_sequence_withID.csv` |
| `Steel` | 钢铁 | `datasets/Steel/USTB_steel_processed_withID.csv` |

## 自定义配置

你可以在 `src/pipelines/batch_configs.py` 中添加自己的配置：

```python
BATCH_CONFIGS = {
    "my_custom_config": {
        "description": "我的自定义配置",
        "alloy_types": ["Ti", "Al"],  # None表示所有合金
        "exclude_alloys": [],
        "embedding_type": "steelbert",
        "use_composition_feature": True,
        "use_element_embedding": True,
        "use_process_embedding": True,
        "use_temperature": False,
        "models": None,  # None表示不使用传统ML模型
        "use_nn": True,
        "cross_validate": True,
        "num_folds": 9,
        "test_size": 0.2,
        "random_state": 42,
        "epochs": 200,
        "patience": 30,
        "batch_size": 256,
        "use_optuna": True,
        "n_trials": 30,
        "evaluate_after_train": True,
        "run_shap_analysis": False,
    }
}
```

然后运行：

```bash
python -m src.pipelines.run_batch --config my_custom_config
```

## 使用Shell脚本

### Windows

```bash
# 激活环境
conda activate llm

# 运行批处理脚本（交互式菜单）
run_all_experiments.bat
```

菜单选项：
1. 仅运行实验1（传统ML模型）
2. 仅运行实验2（神经网络 + BERT嵌入）
3. 运行所有实验（实验1 + 实验2）
4. 预览命令（不实际执行）

### Linux/Mac

```bash
# 激活环境
source activate llm

# 运行shell脚本（交互式菜单）
bash run_all_experiments.sh
```

菜单选项同上。

## 注意事项

1. **资源消耗**: 批量运行会消耗大量计算资源，建议在服务器上运行
2. **时间估算**: 每个合金类型的训练时间取决于数据集大小和模型复杂度
   - 传统ML模型：约10-30分钟/合金
   - 神经网络（无Optuna）：约30-60分钟/合金
   - 神经网络（有Optuna）：约2-4小时/合金
3. **错误处理**: 如果某个合金类型失败，脚本会继续运行其他合金类型
4. **日志文件**: 运行日志会保存在 `batch_run_*.log` 文件中
5. **结果目录**: 结果保存在 `output/results/{alloy_type}/{dataset_name}/{embedding_type}/` 目录下

## 快速开始

### 最简单的方式

```bash
# Windows
run_all_experiments.bat

# Linux/Mac
bash run_all_experiments.sh
```

### 命令行方式

```bash
# 激活环境
conda activate llm  # Windows
source activate llm  # Linux/Mac

# 列出所有配置
python -m src.pipelines.run_batch --list

# 运行所有实验
python -m src.pipelines.run_batch --all

# 预览命令
python -m src.pipelines.run_batch --all --dry_run
```

## 常见问题

### Q: 如何只运行某几个合金类型？

A: 使用自定义模式：
```bash
python -m src.pipelines.run_batch --custom \
    --alloy_types Ti Al \
    --embedding_type tradition \
    --use_composition_feature \
    --models xgboost lightgbm \
    --cross_validate
```

### Q: 如何修改交叉验证的折数？

A: 在自定义模式中指定 `--num_folds`：
```bash
python -m src.pipelines.run_batch --custom \
    --embedding_type tradition \
    --use_composition_feature \
    --models xgboost \
    --cross_validate \
    --num_folds 5
```

### Q: 如何查看将要执行的命令？

A: 使用 `--dry_run` 参数：
```bash
python -m src.pipelines.run_batch --all --dry_run
```

### Q: 如何添加新的合金类型？

A: 在 `src/pipelines/batch_configs.py` 的 `ALLOY_CONFIGS` 中添加：
```python
ALLOY_CONFIGS = {
    # ... 现有配置 ...
    "NewAlloy": {
        "raw_data": "datasets/NewAlloy/data.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        "processing_cols": ["Temperature", "Time"],
        "processing_text_column": "Processing_Description",
        "description": "新合金数据集"
    }
}
```

