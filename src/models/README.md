# Alloy Property Prediction Model Suite / 合金性能预测模型套件

本目录包含合金性能预测模型的定义、训练与评估管线，支持传统机器学习 (ML)、深度神经网络 (NN)、BERT+NN、TabPFN 与 LLMProp。

This directory contains definitions, training, and evaluation pipelines for alloy property prediction models, supporting traditional ML, NN, BERT+NN, TabPFN, and LLMProp.

## 目录结构 / Directory Structure

\models/
├── base/           # 基础模型定义 / Base model definitions
│   ├── alloys_nn.py           # 神经网络结构 / Neural network architectures
│   ├── alloys_bert.py         # BERT+NN 联合模型 / BERT+NN joint model
│   └── alloys_bert_adapter.py # BERT 适配层 / BERT adapter layer
├── trainers/       # 各类模型训练器 / Model trainers
│   ├── base_trainer.py        # 训练器基类 / Base trainer
│   ├── nn_trainer.py          # 神经网络训练器 / Neural network trainer
│   ├── bert_trainer.py        # BERT 训练器 / BERT trainer
│   ├── ml_trainer.py          # 机器学习训练器 (XGB, LGBM, RF, CatBoost, etc.)
│   ├── simple_tabular_trainer.py # 简单表格训练器
│   └── trainer_factory.py     # 训练器工厂 / Trainer factory
├── evaluators/     # 评估器与解释器 / Evaluators & Explainers
│   ├── base_evaluator.py      # 评估器基类 / Base evaluator
│   ├── alloys_evaluator.py    # 合金通用评估器 / Generic alloy evaluator
│   ├── bert_evaluator.py      # BERT 评估器 / BERT evaluator
│   ├── ml_evaluator.py        # 机器学习评估器 / ML evaluator
│   ├── shap_analyzer.py       # SHAP 归因分析器 / SHAP attribution
│   └── evaluator_factory.py   # 评估器工厂 / Evaluator factory
├── ml_train_eval_pipeline/    # 传统机器学习训练与比较管线
├── nn_train_eval_pipeline/    # 神经网络训练与比较管线
├── predictors/     # 推理预测器 / Inference predictors
│   ├── ml_predictor.py
│   └── nn_predictor.py
├── utils/          # 工具函数 / Utilities
│   ├── frozen_params.py       # 参数冻结与复现工具
│   └── json_utils.py          # JSON 序列化工具
└── visualization/  # 可视化转发层 / Visualization shim
    └── plot_utils.py          # 转发至 Scripts.figures.plot_utils
\
## 支持模型架构 / Supported Architectures

1. **传统机器学习 (Traditional ML)**: XGBoost, LightGBM, CatBoost, Random Forest, ExtraTrees, SVR, Ridge, Lasso, ElasticNet, MLP.
2. **深度神经网络 (Deep NN)**: 多层感知机与工艺/成分特征融合网络.
3. **BERT+NN (Multimodal)**: 材料专属语言模型 (SciBERT, SteelBERT, MatSciBERT) + 工艺/成分特征融合.
4. **表格基础模型 (TabPFN)**: 位于 \src/TabPFN/\.
5. **大语言模型属性预测 (LLMProp)**: 位于 \src/LLMProp/\.
