# Optuna 绘图脚本使用说明

## 概述

本文档提供五个用于分析和可视化 Optuna 优化结果的 Python 脚本的详细使用说明。这些脚本可以处理不同合金数据集和机器学习模型的交叉验证结果。

---

## 脚本概览

| 脚本名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `select_best_model_and_plot.py` | **自动选择最佳模型并绘图** | 基础目录 | 模型对比图、最佳模型结果、跨合金汇总 |

**主要功能**：
- 自动递归搜索所有 `model_comparison` 目录
- 比较所有模型性能（CatBoost, LightGBM, MLP, RF, XGB）
- 选择R²最高的模型
- 从最佳模型中选择最接近平均值的代表性结果
- **跨合金汇总**：将所有结果集中到统一文件夹

---

## 1. plot_optuna_trials_comparison.py

### 功能描述
绘制不同模型的性能对比图，读取每个 trial 目录中的 `all_predictions.csv` 文件，计算所有 trials 的均值和标准差。

### 使用方法

#### **示例：处理 HEA 腐蚀数据集**

```bash
# 处理 HEA 腐蚀点蚀电位数据
python src/plot_optuna_trials_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"

# 或使用当前目录
cd "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
python src/plot_optuna_trials_comparison.py
```

#### **运行输出示例**
```
Processing directory: D:\...\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison
Processing catboost...
  Ep(mV): 225 test sets, R2=0.7182+/-0.1253
Processing lightgbm...
  Ep(mV): 230 test sets, R2=0.6161+/-0.1023
Processing mlp...
  Ep(mV): 235 test sets, R2=0.8142+/-0.1723
Processing sklearn_rf...
  Ep(mV): 230 test sets, R2=0.6263+/-0.1559
Processing xgboost...
  Ep(mV): 225 test sets, R2=0.7413+/-0.1639

Successfully loaded data for models: ['CatBoost', 'LightGBM', 'MLP', 'RF', 'XGB']
Plotting targets: ['EP']

Plot saved to: ...\model_comparison\optuna_trials_model_comparison.png
Individual plot saved to: ...\model_comparison\optuna_trials_model_comparison_EP.png
```

### 输入要求
- 目录结构：
  ```
  model_comparison/
    ├── catboost_results/
    │   └── predictions/
    │       └── optuna_trials/
    │           ├── trial_0/
    │           │   ├── fold_0/
    │           │   │   └── all_predictions.csv
    │           │   ├── fold_1/
    │           │   └── ...
    │           └── trial_1/
    ├── lightgbm_results/
    ├── mlp_results/
    ├── sklearn_rf_results/
    └── xgboost_results/
  ```

### 输出文件
- `optuna_trials_model_comparison.png` - 所有目标属性的综合对比图
- `optuna_trials_model_comparison_EP.png` - 点蚀电位（EP）单独对比图
- `optuna_trials_model_comparison_UTS.png` - 抗拉强度（UTS）单独对比图
- `optuna_trials_model_comparison_EL.png` - 延伸率（EL）单独对比图
- `optuna_trials_model_comparison_YS.png` - 屈服强度（YS）单独对比图

### 特性说明
- 从所有 Optuna trials 的测试集中计算 R² 均值和标准差
- 可通过 `PLOT_CONFIG` 字典自定义绘图参数
- 支持多个目标属性：UTS(MPa), El(%), YS(MPa), Ep(mV)
- 使用 Times New Roman 字体，DPI=300 的出版质量图表

---

## 2. batch_plot_optuna_trials.py

### 功能描述
批量脚本，自动查找所有 `model_comparison` 目录并逐个处理，生成对比图。

### 使用方法

#### **示例：批量处理所有合金数据**

```bash
# 处理 new_results_withuncertainty 下的所有数据
python src/batch_plot_optuna_trials.py "output\new_results_withuncertainty"

# 默认处理 output\new_results_withuncertainty
python src/batch_plot_optuna_trials.py
```

#### **运行输出示例**
```
Searching for model_comparison directories in: D:\...\output\new_results_withuncertainty

Found 12 model_comparison directories:
  1. D:\...\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison
  2. D:\...\Steel\USTB_steel\tradition\model_comparison
  3. D:\...\Al\USTB_Al_alloys\tradition\model_comparison
  ...

================================================================================
Starting batch processing...
================================================================================

================================================================================
Processing: D:\...\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison
================================================================================

[OK] Successfully processed: ...
...

================================================================================
BATCH PROCESSING SUMMARY
================================================================================

Total directories processed: 12
Successful: 12
Failed: 0

================================================================================
Batch processing complete!
================================================================================

================================================================================
Collecting specific plots to: D:\...\new_results_withuncertainty\optuna_comparison_summary
================================================================================

[COLLECT] optuna_trials_model_comparison.png -> optuna_comparison_summary/HEA_corrosion_Pitting potential data_xiongjie_tradition_optuna_trials_model_comparison.png
...

Successfully collected 12 plots into optuna_comparison_summary/
```

### 输入要求
- 基础目录包含多个合金子目录
- 每个子目录中有 `model_comparison` 文件夹

### 输出
- 每个 `model_comparison` 目录中生成对比图
- 汇总目录：`optuna_comparison_summary/` 包含所有图表（带描述性名称）

### 特性说明
- 递归搜索 `model_comparison` 目录
- 批量处理并显示进度
- 错误处理和超时保护（每个目录 60 秒）
- 自动收集所有图表到汇总文件夹

---

## 3. collect_all_optuna_results.py

### 功能描述
收集并汇总所有合金数据集的 Optuna trials 结果，创建统一的 CSV 文件用于后续绘图和分析。

### 使用方法

#### **示例：收集所有结果**

```bash
# 收集 new_results_withuncertainty 下的所有结果
python src/collect_all_optuna_results.py "output\new_results_withuncertainty"

# 默认处理 output\new_results_withuncertainty
python src/collect_all_optuna_results.py
```

#### **运行输出示例**
```
Collecting results from: D:\...\output\new_results_withuncertainty

Found 12 model_comparison directories

Processing: HEA_corrosion / Pitting potential data_xiongjie
  Directory: ...\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison
  Model: CatBoost
    Ep(mV): 225 test sets, R2=0.7182+/-0.1253
  Model: LightGBM
    Ep(mV): 230 test sets, R2=0.6161+/-0.1023
  Model: MLP
    Ep(mV): 235 test sets, R2=0.8142+/-0.1723
  Model: RF
    Ep(mV): 230 test sets, R2=0.6263+/-0.1559
  Model: XGB
    Ep(mV): 225 test sets, R2=0.7413+/-0.0639

Processing: Steel / USTB_steel
...

================================================================================
Results saved to: D:\...\new_results_withuncertainty\all_optuna_trials_results_summary.csv
================================================================================

Summary:
  Total records: 60
  Alloys: 5 (HEA_corrosion, Steel, Al, Ti, Nb)
  Models: 5 (CatBoost, LightGBM, MLP, RF, XGB)
  Targets: 3 (EP, UTS, EL)

R² Statistics by Alloy:
Alloy           Mean R²    Std R²     Min R²     Max R²    
---------------------------------------------------------------
HEA_corrosion   0.6832     0.0753     0.6161     0.8142    
Steel           0.8234     0.0421     0.7456     0.8923    
Al              0.7654     0.0612     0.6543     0.8765    
...

R² Statistics by Model:
Model           Mean R²    Std R²     Min R²     Max R²    
---------------------------------------------------------------
CatBoost        0.7234     0.0543     0.6234     0.8234    
LightGBM        0.6543     0.0612     0.5432     0.7654    
MLP             0.7876     0.0987     0.6543     0.9123    
RF              0.6234     0.0876     0.5123     0.7456    
XGB             0.7456     0.0432     0.6789     0.8345    

R² Statistics by Target:
Target          Mean R²    Std R²     Min R²     Max R²    
---------------------------------------------------------------
EP              0.6832     0.0753     0.6161     0.8142    
UTS             0.8234     0.0421     0.7456     0.8923    
EL              0.7123     0.0612     0.6234     0.8234    

================================================================================
Creating pivot tables (compact format with mean and std)...
================================================================================

EP:
  File: D:\...\new_results_withuncertainty\EP_alloy_vs_model.csv
  Shape: (5, 10) (Alloys x Metrics)
  Columns: ['CatBoost_Mean', 'CatBoost_Std', 'LightGBM_Mean', 'LightGBM_Std', 'MLP_Mean', 'MLP_Std', 'RF_Mean', 'RF_Std', 'XGB_Mean', 'XGB_Std']

UTS:
  File: D:\...\new_results_withuncertainty\UTS_alloy_vs_model.csv
  ...

================================================================================
Total files created: 3 CSV files
Format: Each file contains Mean and Std for all models
================================================================================
```

### 输入要求
- 基础目录结构：
  ```
  new_results_withuncertainty/
    ├── HEA_corrosion/
    │   └── Pitting potential data_xiongjie/
    │       └── tradition/
    │           └── model_comparison/
    │               ├── catboost_results/
    │               ├── xgboost_results/
    │               └── ...
    ├── Steel/
    ├── Al/
    └── ...
  ```

### 输出文件

1. **主要 CSV 文件**：`all_optuna_trials_results_summary.csv`
   - 列：Alloy, Dataset, Model, Target, Target_Full, R2_Mean, R2_Std, N_Test_Sets, Directory

2. **透视表**（紧凑格式）：
   - `EP_alloy_vs_model.csv` - 每个模型的均值和标准差
   - `UTS_alloy_vs_model.csv`
   - `EL_alloy_vs_model.csv`
   - `YS_alloy_vs_model.csv`
   - 格式：`{Model}_Mean`, `{Model}_Std` 列

3. **控制台输出**：
   - 按合金、模型和目标属性的汇总统计
   - R² 统计信息（均值、标准差、最小值、最大值）

### 特性说明
- 从路径自动提取合金和数据集名称
- 全面的统计报告
- 生成透视表便于比较

---

## 4. plot_custom_comparison.py

### 功能描述
处理目录以创建带有 bootstrap 置信区间的自定义对比图。自动搜索模型结果目录。

### 使用方法

#### **示例：自定义对比分析**

```bash
# 处理特定目录
python src/plot_custom_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"

# 处理当前目录（无参数）
cd "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
python src/plot_custom_comparison.py

# 处理多个目录
python src/plot_custom_comparison.py "output\new_results_withuncertainty\HEA_corrosion" "output\new_results_withuncertainty\Steel"
```

### 输入要求
- 包含模型结果子目录的目录
- 每个模型文件夹应包含 `predictions/optuna_trials/*/all_predictions.csv`

### 输出
- 带有 bootstrap 置信区间的自定义对比图
- 带不确定性估计的 R² 分数

### 特性说明
- Bootstrap 重采样（n=50）用于置信区间
- 递归目录搜索
- 灵活输入：单个或多个目录

---

## 5. calculate_r2_average_and_plot.py

### 功能描述
计算所有合金 `all_predictions.csv` 文件的 R² 中位数并生成对角线图（预测值 vs 实际值）。优先使用 CatBoost 模型结果。

### 使用方法

#### **示例：R² 中位数分析**

```bash
python src/calculate_r2_average_and_plot.py
```

**默认路径**：脚本使用以下默认路径（也可以通过命令行参数指定）：

```python
# 默认路径
base_dir = r"output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
```

#### **使用其他路径**

可以通过命令行参数指定其他路径：
```bash
python src/calculate_r2_average_and_plot.py "output\new_results_withuncertainty"
```

### 运行输出示例
```
================================================================================
开始计算所有合金的 R2 中位数
================================================================================

找到 1145 个 all_predictions.csv 文件
catboost 文件数: 187
使用文件数: 187

保存所有 R2 结果: D:\...\r2_median_analysis\all_r2_results.csv

================================================================================
R2 中位数统计
================================================================================

                           count   min  median   mean    max    std
alloy  property                                                    
HEA    Ep(mV)               225  0.45   0.7182  0.7182  0.92  0.1253

保存中位数统计: D:\...\r2_median_analysis\r2_median_statistics.csv

================================================================================
绘制图表
================================================================================

保存图表: D:\...\r2_median_analysis\r2_distribution_by_alloy_Ep(mV).png
保存图表: D:\...\r2_median_analysis\diagonal_plot_HEA_Ep(mV).png
保存最佳文件: D:\...\r2_median_analysis\selected_predictions\HEA\all_predictions_EpmV.csv
保存图表: D:\...\r2_median_analysis\all_alloys_diagonal_plot_Ep(mV).png

================================================================================
完成!
================================================================================
```

### 输入要求
- 基础目录包含 `all_predictions.csv` 文件
- 文件必须包含：
  - `Dataset` 列（用于过滤测试集）
  - `{Property}_Actual` 和 `{Property}_Predicted` 列

### 输出文件

1. **CSV 汇总**：
   - `all_r2_results.csv` - 所有 R² 分数及元数据
   - `r2_median_statistics.csv` - 按合金和属性的统计信息

2. **图表**：
   - `r2_distribution_by_alloy_{property}.png` - 每个属性的箱线图
   - `diagonal_plot_{alloy}_{property}.png` - 单个对角线图
   - `all_alloys_diagonal_plot_{property}.png` - 所有合金的网格图

3. **筛选的预测结果**：
   - `selected_predictions/{alloy}/all_predictions_{property}.csv` - 最佳性能的预测结果

### 特性说明
- 自动优先选择 CatBoost 模型
- 测试集过滤
- 均值 R² 文件选择（最接近平均值）
- 多属性支持
- 图表支持中文字体

---

## 工作流程推荐

### 标准分析流程

#### **以 HEA 腐蚀数据为例**

**第 1 步：收集所有结果**
```bash
python src/collect_all_optuna_results.py "output\new_results_withuncertainty"
```
→ 生成汇总 CSV 和透视表

**第 2 步：批量绘图**
```bash
python src/batch_plot_optuna_trials.py "output\new_results_withuncertainty"
```
→ 为所有数据集创建对比图

**第 3 步：详细分析（针对特定数据集）**
```bash
# 使用默认路径或指定路径
python src/calculate_r2_average_and_plot.py
```
→ 生成对角线图和 R² 统计信息

**第 4 步：自定义对比（可选）**
```bash
python src/plot_custom_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
```
→ 针对特定目录的自定义分析

---

## 完整使用示例：HEA 腐蚀数据分析

### 场景：分析高熵合金腐蚀性能预测

```bash
# 1. 单个数据集对比
python src/plot_optuna_trials_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"

# 输出：
# - optuna_trials_model_comparison.png (综合对比)
# - optuna_trials_model_comparison_EP.png (Ep 单独对比)

# 2. 收集所有合金的结果
python src/collect_all_optuna_results.py "output\new_results_withuncertainty"

# 输出：
# - all_optuna_trials_results_summary.csv
# - EP_alloy_vs_model.csv
# - UTS_alloy_vs_model.csv
# - EL_alloy_vs_model.csv

# 3. 批量处理所有数据集
python src/batch_plot_optuna_trials.py "output\new_results_withuncertainty"

# 输出：
# - 每个 model_comparison 目录中的对比图
# - optuna_comparison_summary/ 文件夹包含所有图表

# 4. R² 中位数和对角线图分析
# 使用默认路径 (HEA腐蚀数据)
python src/calculate_r2_average_and_plot.py

# 输出：
# - all_r2_results.csv
# - r2_median_statistics.csv
# - diagonal_plot_HEA_Ep(mV).png
# - r2_distribution_by_alloy_Ep(mV).png
```

---

## 常用参数

### 支持的模型
- CatBoost (`catboost`)
- LightGBM (`lightgbm`)
- XGBoost (`xgboost`)
- Random Forest (`sklearn_rf`)
- MLP (`mlp`)

### 支持的目标属性
- **UTS(MPa)** → 显示为 "UTS" (抗拉强度)
- **El(%)** → 显示为 "EL" (延伸率)
- **YS(MPa)** → 显示为 "YS" (屈服强度)
- **Ep(mV)** → 显示为 "EP" (点蚀电位)

### 绘图配置
大多数脚本使用出版质量设置：
- **字体**：Times New Roman，大小 22
- **DPI**：300
- **Y 轴**：R² 范围 [0, 1]
- **颜色方案**：
  - UTS（蓝色 #aacfef）
  - EL（粉色 #ffb3a7）
  - YS（绿色 #c5e1a5）
  - EP（金色 #f4c542）

---

## 故障排除

### 问题：找不到模型目录
**解决方案**：
- 确保目录结构符合预期格式，包含 `*_results` 子目录
- 检查路径是否指向 `model_comparison` 目录

**示例**：
```bash
# ❌ 错误
python src/plot_optuna_trials_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie"

# ✅ 正确
python src/plot_optuna_trials_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
```

### 问题：找不到测试集数据
**解决方案**：
- 验证 `all_predictions.csv` 包含 `Dataset` 列且有 "Test" 值

### 问题：找不到 optuna_trials 目录
**解决方案**：
- 检查是否运行了 Optuna 优化
- 确认结果在 `predictions/optuna_trials/` 中

### 问题：中文字符显示为方框
**解决方案**：
- 安装思源黑体（SimHei）字体
- 或修改 `calculate_r2_average_and_plot.py` 中的字体设置：
```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
```

### 问题：No valid targets found
**解决方案**：
- 检查 `PLOT_CONFIG` 中的 `target_names` 映射
- 确保数据文件中的目标属性名称与配置匹配
- 例如：如果数据中有 `Ep(mV)` 但配置中没有，需要添加：
```python
'target_names': {
    'UTS(MPa)': 'UTS',
    'El(%)': 'EL',
    'YS(MPa)': 'YS',
    'Ep(mV)': 'EP'  # 添加新的目标属性
},
```

---

## 快速参考命令

### HEA 腐蚀数据分析快速命令

```bash
# 单个数据集分析
python src/plot_optuna_trials_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"

# 收集所有结果
python src/collect_all_optuna_results.py "output\new_results_withuncertainty"

# 批量处理
python src/batch_plot_optuna_trials.py "output\new_results_withuncertainty"

# R² 分析（默认HEA腐蚀数据路径）
python src/calculate_r2_average_and_plot.py

# 自定义对比
python src/plot_custom_comparison.py "output\new_results_withuncertainty\HEA_corrosion\Pitting potential data_xiongjie\tradition\model_comparison"
```

---

## 版本信息
- **创建日期**：2026-01-15
- **兼容 Python 版本**：3.7+
- **必需依赖**：pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 注意事项

- 脚本设计用于批量处理 Optuna 交叉验证结果
- 所有脚本都能优雅地处理缺失数据
- 保存中间结果以便恢复中断的分析
- 可通过 `PLOT_CONFIG` 字典自定义绘图配置
- 确保路径使用正确的斜杠（Windows 使用反斜杠 `\` 或双斜杠 `\\`）
