# 机器学习模型对比工具 / ML Model Comparison Tool

## 🎯 项目重构总结 / Project Refactoring Summary

本项目已完成重构，将复杂的单文件脚本拆分为模块化的结构，提供更清晰的代码组织和更好的用户体验。

This project has been refactored, splitting complex single-file scripts into modular structure for clearer code organization and better user experience.

## 📁 文件结构 / File Structure

### 核心模块 / Core Modules
- `model_comparator.py` - 模型对比器核心逻辑
- `config_manager.py` - 配置管理和参数解析
- `model_comparison_example.py` - 简化的主接口
- `model_comparison_cli.py` - 命令行工具

### 支持文件 / Supporting Files
- `pipeline.py` - ML训练管道
- `data_loader.py` - 数据加载器
- `utils.py` - 工具函数
- `config.py` - 基础配置
- `example_config.json` - 示例配置文件

## 🚀 使用方法 / Usage

### 1. 命令行工具（推荐）/ CLI Tool (Recommended)

**重要**: 请在项目根目录下使用模块方式运行，以避免导入错误。

#### 查看示例命令 / View Example Commands
```bash
# 在项目根目录下运行
python -m src.models.ml_train_eval_pipeline.model_comparison_cli
```

#### 基础对比 / Basic Comparison
```bash
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
    --result_dir "output/results/Ti_alloys/basic_comparison" \
    --target_columns "UTS(MPa)" "El(%)" \
    --models xgboost sklearn_rf mlp \
    --use_composition_feature
```

#### 完整对比 / Full Comparison
```bash
python -m src.models.ml_train_eval_pipeline.model_comparison_cli \
    --data_file "datasets/Ti_alloys/Titanium_Alloy_Dataset_Processed_cleaned.csv" \
    --result_dir "output/results/Ti_alloys/full_comparison" \
    --target_columns "UTS(MPa)" "El(%)" \
    --models xgboost sklearn_rf mlp sklearn_svr lightgbm catboost \
    --processing_cols "Solution Temperature()" "Solution Time(h)" "Aging Temperature()" "Aging Time(h)" "Thermo-Mechanical Treatment Temperature()" "Deformation(%)" \
    --use_composition_feature \
    --use_optuna \
    --n_trials 20 \
    --num_folds 5
```

### 2. 配置文件方式 / Configuration File Method
```bash
python -m src.models.ml_train_eval_pipeline.model_comparison_example --config example_config.json
```

### 3. 代码调用 / Code Integration
```python
from model_comparison_example import run_model_comparison

config = {
    'data_file': 'path/to/data.csv',
    'result_dir': 'output/results',
    'target_columns': ['UTS(MPa)', 'El(%)'],
    'models': ['xgboost', 'sklearn_rf', 'mlp'],
    'use_composition_feature': True,
    'use_optuna': True,
    'n_trials': 20
}

results = run_model_comparison(config)
```

## 📊 支持的模型 / Supported Models

- `xgboost` - XGBoost
- `lightgbm` - LightGBM
- `catboost` - CatBoost
- `sklearn_rf` - Random Forest
- `sklearn_svr` - Support Vector Regression
- `sklearn_gpr` - Gaussian Process Regression
- `mlp` - Multi-Layer Perceptron

## 🔧 主要参数 / Key Parameters

### 必需参数 / Required Parameters
- `--data_file` - 数据文件路径
- `--result_dir` - 结果保存目录
- `--target_columns` - 目标列名
- `--models` - 要对比的模型

### 常用可选参数 / Common Optional Parameters
- `--use_composition_feature` - 使用成分特征
- `--use_optuna` - 启用Optuna超参数优化
- `--n_trials` - Optuna优化试验次数
- `--num_folds` - 交叉验证折数
- `--processing_cols` - 处理工艺列名

## 📈 输出结果 / Output Results

运行完成后，在指定的结果目录下会生成：

```
result_dir/
└── model_comparison/
    ├── model_comparison_results.csv      # 对比结果表格
    ├── model_comparison_plots.png        # 可视化图表
    ├── comparison_report.txt             # 详细文本报告
    ├── xgboost_results/                  # XGBoost详细结果
    ├── sklearn_rf_results/               # Random Forest详细结果
    └── ...                               # 其他模型结果
```

## ✨ 重构改进 / Refactoring Improvements

### 之前 / Before
- ❌ 单个复杂文件（800+ 行）
- ❌ 硬编码配置
- ❌ 难以维护和扩展
- ❌ 重复代码

### 现在 / Now
- ✅ 模块化设计（4个核心模块）
- ✅ 灵活的配置系统
- ✅ 清晰的代码结构
- ✅ 易于维护和扩展
- ✅ 多种使用方式
- ✅ 详细的示例和文档

## 💡 使用建议 / Usage Tips

1. **新手用户**: 在项目根目录运行 `python -m src.models.ml_train_eval_pipeline.model_comparison_cli` 查看示例
2. **快速测试**: 使用少量模型和试验次数
3. **生产环境**: 使用完整配置获得最佳结果
4. **批量运行**: 创建多个配置文件
5. **自动化**: 在代码中调用 `run_model_comparison()` 函数
6. **重要**: 必须添加 `--use_composition_feature` 参数来启用成分特征，否则会出现"No features selected"错误

## 🔍 获取帮助 / Get Help

```bash
python -m src.models.ml_train_eval_pipeline.model_comparison_cli --help
```

## ⚠️ 常见问题 / Common Issues

1. **ModuleNotFoundError**: 请确保在项目根目录下使用 `python -m` 方式运行
2. **No features selected**: 请添加 `--use_composition_feature` 参数
3. **路径错误**: 数据文件路径应相对于项目根目录
