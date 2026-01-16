# Alloy Property Prediction Model Suite / 合金性能预测模型套件

本目录包含多种合金性能预测模型的定义、训练、评估与测试脚本，支持神经网络、BERT、XGBoost、随机森林等多种方法。

This directory contains definitions, training, evaluation, and test scripts for various alloy property prediction models, supporting neural networks, BERT, XGBoost, Random Forest, and more.

## 目录结构 / Directory Structure

```
models/
├── base/           # 基础模型定义 / Base model definitions
│   ├── alloys_nn.py       # 神经网络模型定义 / Neural network model definition
│   └── alloys_bert.py     # BERT模型定义 / BERT model definition
├── trainers/       # 各类模型训练器 / Model trainers
│   ├── base_trainer.py    # 基础训练器 / Base trainer
│   ├── nn_trainer.py      # 神经网络训练器 / Neural network trainer
│   ├── bert_trainer.py    # BERT训练器 / BERT trainer
│   ├── ml_trainer.py      # 机器学习模型训练器 / Machine learning trainer
│   └── trainer_factory.py # 训练器工厂 / Trainer factory
├── evaluators/     # 评估器 / Evaluators
│   ├── base_evaluator.py  # 基础评估器 / Base evaluator
│   ├── alloys_evaluator.py # 合金模型评估器 / Alloy model evaluator
│   ├── bert_evaluator.py  # BERT模型评估器 / BERT model evaluator
│   ├── ml_evaluator.py    # 机器学习模型评估器 / Machine learning model evaluator
│   └── evaluator_factory.py # 评估器工厂 / Evaluator factory
├── utils/          # 工具函数 / Utility functions
│   └── json_utils.py      # JSON工具 / JSON utilities
├── visualization/  # 可视化工具 / Visualization tools
│   └── plot_utils.py      # 绘图工具 / Plotting utilities
├── test_trainer.py        # 神经网络/BERT训练测试脚本 / NN/BERT test script
└── test_ml_trainer.py     # 机器学习模型训练测试脚本 / ML model test script
```

## 主要功能 / Main Features

- **模型定义 / Model Definition**：支持神经网络、BERT、XGBoost、随机森林等 / Supports NN, BERT, XGBoost, RF, etc.
- **训练器 / Trainer**：统一接口，支持早停、交叉验证、模型保存 / Unified interface, early stopping, cross-validation, model saving
- **评估器 / Evaluator**：自动计算RMSE、MAE、R²等指标，支持可视化 / Auto metrics (RMSE, MAE, R²), visualization
- **测试脚本 / Test Script**：一键训练与评估 / One-click train & eval

## 快速开始 / Quick Start

### 1. 环境准备 / Environment Setup

建议使用conda环境，安装依赖：/ Use conda environment and install dependencies:

```bash
conda activate llm
pip install -r requirements.txt
```

### 2. 神经网络/深度学习模型训练 / NN/Deep Learning Training

```bash
python src/models/test_trainer.py \
  --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
  --models_dir "model_checkpoints/nn" \
  --result_dir "output/results/nn" \
  --epochs 200 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --weight_decay 1e-5 \
  --emb_hidden_dim 256 \
  --acta_hidden_dim 0 \
  --hidden_dims 256 128 \
  --dropout_rate 0.0 \
  --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
  --patience 20 \
  --evaluate_after_train \
  --plot_compare
```

### 3. 机器学习模型训练（如XGBoost）/ ML Model Training (e.g. XGBoost)

```bash
python src/models/test_ml_trainer.py \
  --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
  --models_dir "model_checkpoints/xgb" \
  --result_dir "output/results/xgb" \
  --model_type "xgb" \
  --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
  --evaluate_after_train \
  --plot_compare
```

### 4. 使用其他机器学习模型 / Using Other ML Models

本套件支持多种机器学习模型，包括XGBoost、随机森林、SVR、Ridge、Lasso、ElasticNet等。只需修改 `--model_type` 参数即可切换模型。所有单目标模型（如SVR、Ridge、Lasso、ElasticNet）已通过 `MultiOutputRegressor` 自动支持多目标回归，无需额外修改。XGBoost 和随机森林原生支持多目标回归，无需包装。例如：

- **随机森林 / Random Forest**:
  ```bash
  python src/models/test_ml_trainer.py \
    --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
    --models_dir "model_checkpoints/rf" \
    --result_dir "output/results/rf" \
    --model_type "rf" \
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
    --evaluate_after_train \
    --plot_compare
  ```

- **支持向量回归 / Support Vector Regression**:
  ```bash
  python src/models/test_ml_trainer.py \
    --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
    --models_dir "model_checkpoints/svr" \
    --result_dir "output/results/svr" \
    --model_type "svr" \
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
    --evaluate_after_train \
    --plot_compare
  ```

- **Ridge回归 / Ridge Regression**:
  ```bash
  python src/models/test_ml_trainer.py \
    --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
    --models_dir "model_checkpoints/ridge" \
    --result_dir "output/results/ridge" \
    --model_type "ridge" \
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
    --evaluate_after_train \
    --plot_compare
  ```

- **Lasso回归 / Lasso Regression**:
  ```bash
  python src/models/test_ml_trainer.py \
    --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
    --models_dir "model_checkpoints/lasso" \
    --result_dir "output/results/lasso" \
    --model_type "lasso" \
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
    --evaluate_after_train \
    --plot_compare
  ```

- **ElasticNet回归 / ElasticNet Regression**:
  ```bash
  python src/models/test_ml_trainer.py \
    --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
    --models_dir "model_checkpoints/elastic" \
    --result_dir "output/results/elastic" \
    --model_type "elastic" \
    --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
    --evaluate_after_train \
    --plot_compare
  ```

### 5. 交叉验证 / Cross-validation

```bash
python src/models/test_ml_trainer.py \
  --data_file "Features/Al_alloys_Features/Al_mechanical_dataset_clean_Proc_Comp_Embeddings/containAll/NN/features.csv" \
  --models_dir "model_checkpoints/xgb_cv" \
  --result_dir "output/results/xgb_cv" \
  --model_type "xgb" \
  --target_columns "YS(MPa)" "UTS(MPa)" "El(%)" \
  --cross_validate \
  --num_folds 5 \
  --evaluate_after_train \
  --plot_compare
```

## 结果输出 / Output

- 训练日志、模型权重、评估指标、可视化图片均保存在 `result_dir` 和 `models_dir` 下。
- 评估指标包括RMSE、MAE、R²等，支持多目标回归。

All logs, model weights, metrics, and plots are saved in `result_dir` and `models_dir`. Metrics include RMSE, MAE, R², and support multi-target regression.

## 常见问题 / FAQ

- **XGBoost预测报错 / XGBoost prediction error**：请确保输入为numpy数组，且模型为MLTrainer包装对象。/ Make sure input is numpy array and model is MLTrainer wrapper.
- **环境问题 / Environment issues**：请先激活llm环境，确保依赖齐全。/ Activate llm env and check dependencies.
- **自定义模型/特征 / Custom model/features**：可在`base/`和`trainers/`目录下扩展。/ Extend in `base/` and `trainers/`.

## 联系方式 / Contact

如有问题请联系开发者或提交issue。/ Contact developer or submit issue if any problem.

---

**English version available upon request.**