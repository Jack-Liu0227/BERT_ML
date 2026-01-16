# Summary of Model Training Methods

This document provides a formal overview of the methodologies employed for model training and evaluation.

---

## Classical Machine Learning Models

### English

To establish a predictive baseline, a systematic evaluation of several machine learning models was performed. The candidate algorithms, as implemented in the codebase, include XGBoost, LightGBM, CatBoost, Random Forest (`sklearn_rf`), and a Multi-layer Perceptron (MLP).

The model training and evaluation protocol is grounded in the framework's specific configuration. To ensure statistical robustness, a 9-fold cross-validation (`num_folds: 9`) methodology is implemented. Hyperparameter optimization is conducted using the Optuna framework (`use_optuna: True`), where a search is performed over 30 trials (`n_trials: 30`) to identify the optimal hyperparameter configuration for each model. For the tree-based models (XGBoost, LightGBM, CatBoost, Random Forest), key parameters such as `n_estimators`, `max_depth`, and `learning_rate` are tuned. For the MLP model, the framework specifically utilizes the `Adam` optimization algorithm (`solver: 'adam'`). Its architecture (e.g., number and size of hidden layers) and regularization parameter (`alpha`) are optimized, with training conducted for a maximum of 300 iterations (`mlp_max_iter: 300`).

Model performance was quantified using two primary metrics: the Root Mean Squared Error (RMSE) and the coefficient of determination (R²). The R² score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables, is calculated as follows:
$R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$
where $y_i$ is the observed value, $\hat{y}_i$ is the value predicted by the model, and $\bar{y}$ is the mean of the observed values. An R² score of 1.0 indicates a perfect prediction of the observed variance.

### 中文 (Chinese)

为建立一个预测基准，我们对多种机器学习模型进行了系统性评估。根据代码库的实现，候选算法包括 XGBoost、LightGBM、CatBoost、随机森林（`sklearn_rf`）以及一个多层感知机（MLP）。

模型训练与评估协议基于框架中的特定配置。为确保统计鲁棒性，我们实施了9折交叉验证（`num_folds: 9`）方法。超参数优化通过Optuna框架进行（`use_optuna: True`），在30次试验（`n_trials: 30`）中为每个模型识别最佳超参数配置。对于基于树的模型（XGBoost, LightGBM, CatBoost, 随机森林），关键参数如 `n_estimators`、`max_depth` 和 `learning_rate` 会被调优。对于MLP模型，该框架明确使用了 `Adam` 优化算法（`solver: 'adam'`）。其架构（例如，隐藏层的数量和大小）和正则化参数（`alpha`）都会被优化，并且模型最多训练300次迭代（`mlp_max_iter: 300`）。

模型性能通过两个主要指标进行量化：均方根误差（RMSE）和决定系数（R²）。R²分数衡量了因变量方差中可由自变量预测的比例，其计算公式如下：
$R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$
其中，$y_i$是观测值，$\hat{y}_i$是模型预测值，$\bar{y}$是观测值的平均值。R²分数为1.0表示模型能完美预测观测值的方差。

---

## Custom Multi-Branch Neural Network

### English

For advanced feature representation, a custom multi-branch neural network, `AlloyNN`, was developed. This architecture is designed to process diverse feature sets, including pre-computed embeddings from models like SciBERT or MatSciBERT, alongside other numerical features.

The `AlloyNN` architecture consists of two main stages. The first is a `FeatureProcessor` module, which implements a parallel-branch design. Distinct feature groups (e.g., compositional embeddings, processing parameters) are independently transformed by dedicated linear layers. The outputs of these branches are then concatenated into a single, unified feature vector. This vector is subsequently passed to the second stage, a `PredictionNetwork`. This final module is a standard Multi-Layer Perceptron (MLP) that acts as the regressor, mapping the high-dimensional feature representation to the target material properties.

The training protocol is configured to minimize the Mean Squared Error (MSE) loss function using the `Adam` optimizer. Models are trained for a maximum of 200 epochs with a batch size of 256, incorporating an early stopping criterion with a patience of 30 epochs to prevent overfitting. A systematic hyperparameter and architecture optimization is performed using the Optuna framework over 30 trials. The search space includes:
- **Training Parameters**: `learning_rate` (1e-5 to 1e-2), `weight_decay` (1e-6 to 1e-3), and `batch_size` (16 to 256).
- **LR Scheduler**: The use of a learning rate scheduler, along with its `patience` and `factor`.
- **Architecture**: The `dropout_rate`, the hidden dimensions of the `FeatureProcessor` branches, and the structure of the final `PredictionNetwork` (number of layers and their sizes).

### 中文 (Chinese)

为了实现高级特征表示，我们开发了一个名为 `AlloyNN` 的自定义多分支神经网络。该架构旨在处理多样化的特征集，包括来自SciBERT或MatSciBERT等模型的预计算嵌入以及其他数值特征。

`AlloyNN` 的架构包含两个主要阶段。第一阶段是 `FeatureProcessor` 模块，它实现了一种并行分支设计。不同的特征组（例如，成分嵌入、工艺参数）由专用的线性层独立进行变换。这些分支的输出随后被拼接成一个统一的特征向量。该向量接着被传递到第二阶段——`PredictionNetwork`。这个最终模块是一个标准的多层感知机（MLP），作为回归器，将高维特征表示映射到目标材料属性。

训练协议配置为使用 `Adam` 优化器最小化均方误差（MSE）损失函数。模型最多训练200个周期，批量大小为256，并结合了耐心值为30个周期的早停标准以防止过拟合。我们使用Optuna框架在30次试验中进行了系统的超参数和架构优化。其搜索空间包括：
- **训练参数**: `learning_rate` (1e-5 至 1e-2), `weight_decay` (1e-6 至 1e-3), 和 `batch_size` (16 至 256)。
- **学习率调度器**: 是否使用学习率调度器，及其 `patience` 和 `factor` 参数。
- **架构**: `dropout_rate`、`FeatureProcessor` 分支的隐藏维度，以及最终 `PredictionNetwork` 的结构（层数和大小）。

