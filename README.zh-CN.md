# BERT_ML：合金性能预测基准与实验套件

[English](./README.md)

BERT_ML 是一个模块化、严格可复现的合金力学性能预测科研基准平台，系统覆盖独立同分布（I.I.D.）与分布外泛化（OOD, Out-of-Distribution）场景。项目集成了传统表格机器学习、深度神经网络（AlloyNN）、材料领域预训练语言模型嵌入融合（SciBERT、SteelBERT、MatSciBERT）、表格先验大模型（TabPFN v2）以及基于大语言模型的属性预测器（LLMProp）。

---

## 核心特性

- **4 大核心模型家族**：
  - **传统机器学习 (Traditional ML)**：XGBoost、LightGBM、CatBoost、Random Forest、ExtraTrees、SVR、Ridge、Lasso、ElasticNet、MLP，内置 Optuna 超参数优化、多目标回归支持与 SHAP 可解释性分析。
  - **深度神经网络 (AlloyNN)**：工艺参数、组分比例与嵌入向量的多路特征融合网络。
  - **领域 BERT + NN 联合架构**：集成 SciBERT、SteelBERT、MatSciBERT 领域专属语言模型表征与物理组分/工艺特征的端到端融合模型。
  - **表格基础模型 (TabPFN v2)**：支持本地权重与 API 调用的上下文学习表格模型。
  - **大语言模型回归 (LLMProp)**：基于提示工程与文本微调的合金属性回归模型。
- **5 大合金体系与多属性预测**：
  - 钛合金 (Ti)、铝合金 (Al)、高熵合金 (HEA)、结构/碳素钢 (Steel)、Matbench 钢基准数据集。
  - 预测目标涵盖：屈服强度 (YS(MPa))、抗拉强度 (UTS(MPa))、延伸率 (El(%))。
- **8 种系统性 OOD 数据划分机制**：
  - 极值目标外推 (	arget_extrapolation：前 20% / 后 20% 性能区间外推)。
  - 成分/特征空间稀疏外推 (sparse_x_single, sparse_x_cluster)。
  - 属性/目标空间稀疏外推 (sparse_y_single, sparse_y_cluster)。
  - 未见合金聚类留出验证 (loco: Leave-One-Cluster-Out)。
  - 经典分布内基线验证 (
andom_cv_baseline：标准多折交叉验证)。
  - 复合混合外推方案 (hybrid_extrapolation_*)。
- **职责清晰的工程架构**：
  - src/：严格聚焦于核心数据划分、特征生成与模型训练/评估管线。
  - Scripts/：承载后验不确定性标定、分布偏移度量（Wasserstein 距离 / W-jump 审计）、批量总结报告与论文图表生成。

---

## 项目目录结构

\BERT_ML/
├── datasets/                  # 标准化基准数据集
│   ├── Ti_alloys/             # 钛合金数据集
│   ├── Al_Alloys/             # 铝合金数据集
│   ├── HEA_data/              # 高熵合金数据集
│   ├── Steel/                 # 钢铁力学性能数据集
│   └── matbench_convert/      # Matbench 转换数据集
├── models/                    # 模型权重与下载工具
│   ├── download_bert_models.py
│   ├── download_llmprop_models.py
│   └── tabpfn/                # TabPFN v2 本地模型权重
├── src/                       # 核心训练与评估管线
│   ├── data_processing/       # OOD 划分处理器、注册表与划分缓存
│   ├── feature_engineering/   # 组分特征生成、BERT 模型加载与嵌入
│   ├── models/                # 模型架构定义、训练器 (Trainers) 与评估器 (Evaluators)
│   ├── pipelines/             # 批处理实验运行器与多折 CV 扫描
│   ├── TabPFN/                # TabPFN 训练、OOD 管线与预测对齐
│   ├── LLMProp/               # LLMProp 微调与 OOD 评估适配器
│   └── training_size_ood/     # 训练样本量扩展实验运行器
├── Scripts/                   # 离线分析、标定与可视化工具集
│   ├── analysis/              # 空间分布偏移、W-jump 审计与数据分析
│   ├── calibration/           # 后验置信度与不确定性标定
│   ├── figures/               # 论文图表与交互式可视化解释器
│   ├── maintenance/           # 结果修复、数据审计与输出重置
│   └── reporting/             # 批量统计汇总与基准报告
└── tests/                     # 自动化 pytest 验证测试集
\
---

## 快速开始

### 1. 环境配置

推荐使用 Conda 创建独立虚拟环境（Python 3.10）：

\\ash
conda env create -f environment.yml
conda activate llm
\
或使用 pip 安装依赖：

\\ash
pip install -r requirements.txt
\
### 2. 下载基础模型资产

运行下载脚本获取领域 BERT 与 LLMProp 模型权重：

\\ash
python models/download_bert_models.py
python models/download_llmprop_models.py
\
> **说明**：SteelBERT 属于 Hugging Face 权限受限模型，请先执行 \huggingface-cli login\，并在 [MGE-LLMs/SteelBERT](https://huggingface.co/MGE-LLMs/SteelBERT) 页面申请访问权限。

---

## 实验运行指南

所有实验管线均采用原生 Python 模块入口（python -m ...），无需使用任何 shell 或 batch 包装脚本。

### 1. 分布内 (I.I.D.) 批处理实验

查看所有支持的分布内批处理配置：
\\ash
python -m src.pipelines.run_batch --list
\
运行指定实验（如传统机器学习模型）：
\\ash
python -m src.pipelines.run_batch --config experiment1_all_ml_models
\
空跑预览全部标准批处理实验：
\\ash
python -m src.pipelines.run_batch --all --dry_run
\
### 2. 分布外 (OOD) 泛化基准矩阵

查看所有支持的 64 组 OOD 实验配置（覆盖 5 类合金体系 × 8 种划分策略）：
\\ash
python -m src.pipelines.run_batch_ood --list
\
运行传统 ML 的极值目标外推实验：
\\ash
python -m src.pipelines.run_batch_ood --config experiment1_all_ml_models_extrapolation
\
运行 SciBERT + NN 的 LOCO（留一合金聚类）实验：
\\ash
python -m src.pipelines.run_batch_ood --config experiment2a_all_nn_scibert_loco
\
空跑校验全部 OOD 实验矩阵：
\\ash
python -m src.pipelines.run_batch_ood --all --dry_run
\
### 3. 多随机种子交叉验证扫描 (CV Sweep)

执行严格的多种子多折交叉验证参数扫描：
\\ash
# 空跑校验扫描任务计划
python -m src.pipelines.run_cv_k_sweep --dry_run

# 运行 Random CV 基准（5折，种子 0~4，2个并行工作进程）
python -m src.pipelines.run_cv_k_sweep --method random_cv_baseline --k 5 --seeds 0 1 2 3 4 --workers 2
\
### 4. TabPFN 表格基础模型实验

执行 TabPFN 分布内与分布外实验：
\\ash
# 查看 TabPFN OOD 配置列表
python -m src.TabPFN.run_tabpfn_ood_batch --list

# 运行 TabPFN 本地回归基线
python -m src.TabPFN.train_tabpfn --all --backend local --feature_mode numeric

# 运行 TabPFN OOD 评估
python -m src.TabPFN.train_tabpfn_ood --all --backend local --feature_mode numeric
\
### 5. LLMProp 实验

运行大语言模型属性回归实验：
\\ash
python -m src.LLMProp.run_llmprop_ood --help
\
---

## 离线分析、标定与论文成图

Scripts/ 目录提供了完整的实验后处理与可视化管线：

- **后验置信度标定**：
  \\ash
  python -m Scripts.calibration.cli --help
  \- **论文核心图表生成（Figure 3 与 Figure 4）**：
  \\ash
  python -m Scripts.figures.build_ood_figures_3_4 --help
  \- **TabPFN 三联对比图生成**：
  \\ash
  python -m Scripts.figures.build_bestplus_tabpfn_triptych --help
  \
---

## 测试与质量验证

项目包含完备的自动化 pytest 测试套件，覆盖划分注册表、模型适配器、标定算法与命令行接口：

\\ash
pytest tests/
\
全量 267 个测试用例全部通过（0 failures, 0 errors）。

---

## 许可证与引用

本项目采用 MIT 许可证。如果您在学术研究中使用了本代码库或实验基准，请引用相关论文成果。
