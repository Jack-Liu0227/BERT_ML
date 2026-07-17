(() => {
  if (!document.documentElement.lang.toLowerCase().startsWith("en")) return;

  const translations = new Map([
    ["Calibration 结果是怎样得到的", "How the Calibration Results Were Produced"],
    ["Calibration 结果获取全过程", "Calibration Results: End-to-End Workflow"],
    ["中文", "ZH"],
    ["通俗解释", "Plain explanation"],
    ["技术细节", "Technical details"],
    ["文件证据", "File evidence"],
    ["阅读路径", "Reading path"],
    ["全过程总览", "Workflow overview"],
    ["Calibration 是什么", "What calibration means"],
    ["数据与任务范围", "Data and task scope"],
    ["参数如何确定", "How parameters were fixed"],
    ["七种 OOD 测试", "Seven OOD protocols"],
    ["置信度如何产生", "How confidence was produced"],
    ["GPT 如何并入", "How GPT was integrated"],
    ["指标如何计算", "How metrics were calculated"],
    ["结果如何解读", "How to read the results"],
    ["如何保证可信", "How reliability was verified"],
    ["页面中的数字由 verified calibration 文件自动生成，不是手工填写。", "Every number on this page is generated from verified calibration artifacts, not entered by hand."],
    ["从一条材料性能预测，到最终 Calibration 图", "From one material-property prediction to the final calibration figures"],
    ["这不是直接把几个模型的误差画在一起。完整过程包含数据划分、参数冻结、重复预测、不确定性计算、置信层分组、统一指标和独立验证。", "This is more than plotting several models' errors together. The full workflow covers data splitting, parameter freezing, repeated prediction, uncertainty estimation, confidence-tier assignment, unified metrics, and independent verification."],
    ["冻结的 task-model 参数配置", "Frozen task-model parameter configurations"],
    ["统一 OOD 测试协议", "Unified OOD test protocols"],
    ["共同 task-protocol 案例", "Common task-protocol cases"],
    ["最终比较的方法序列", "Methods in the final comparison"],
    ["统一后的样本级记录", "Integrated sample-level records"],
    ["通过验证的完整运行单元", "Fully verified run units"],
    ["整理材料数据", "Prepare material data"],
    ["统一任务、特征、目标和样本 ID。", "Standardize tasks, features, targets, and sample IDs."],
    ["选择并冻结参数", "Select and freeze parameters"],
    ["只使用 RandomCV 验证结果，不看 OOD 测试答案。", "Use RandomCV validation results only, without consulting OOD test outcomes."],
    ["建立 OOD 划分", "Create OOD splits"],
    ["模拟随机、跨簇、外推和稀疏区域。", "Simulate random, cross-cluster, extrapolation, and sparse-region settings."],
    ["生成重复预测", "Generate repeated predictions"],
    ["ensemble、MC-dropout、quantile 或 GPT 语义置信度。", "Use ensembles, MC dropout, predictive quantiles, or GPT semantic confidence."],
    ["统一样本表", "Build one sample table"],
    ["所有方法对齐为 y_true、y_pred、error 和 confidence。", "Align every method to y_true, y_pred, error, and confidence."],
    ["计算 Calibration", "Calculate calibration"],
    ["NMAE、判别分数、Spearman 和 bootstrap CI。", "Compute NMAE, discrimination, Spearman correlation, and bootstrap confidence intervals."],
    ["验证并发布", "Verify and publish"],
    ["检查数量、快照、哈希、图片和完整覆盖。", "Check counts, snapshots, hashes, figures, and complete coverage."],
    ["通俗理解", "Plain-language view"],
    ["可以把这条流程看成一次封闭考试：先在练习题上确定模型设置，再把设置锁住，最后让所有模型参加相同的七类陌生题。Calibration 关注的不是谁答对最多，而是谁能够诚实地知道自己什么时候可能答错。", "Think of the workflow as a closed-book examination. Model settings are chosen on practice questions, locked, and then every model takes the same seven kinds of unfamiliar test. Calibration does not ask who answers the most questions correctly; it asks who can honestly recognize when it may be wrong."],
    ["技术定义", "Technical definition"],
    ["超参数冻结与 OOD 评估严格分离。最终比较使用 8 个共同任务、7 个协议和 6 个方法，共 336 个 task-protocol-method 组合；所有方法各贡献 9,231 条样本记录。", "Hyperparameter freezing is strictly separated from OOD evaluation. The final comparison uses 8 common tasks, 7 protocols, and 6 methods, giving 336 task-protocol-method combinations. Each method contributes 9,231 sample records."],
    ["证据入口", "Evidence entry points"],
    ["总范围、输入哈希、输出哈希和验证状态来自", "The complete scope, input hashes, output hashes, and verification status come from"],
    ["；冻结参数来自", "; frozen parameters come from"],
    ["Calibration 到底在问什么", "What question does calibration answer?"],
    ["一个有用的置信度信号，应当让“高置信度”样本平均更容易，让“低置信度”样本平均更难。", "A useful confidence signal should make high-confidence samples easier on average and low-confidence samples harder on average."],
    ["有判别力的置信度", "Discriminative confidence"],
    ["模型说“High”时误差较小，说“Low”时误差明显增大。使用者可以据此决定哪些预测需要实验复核。", "Errors are smaller when the model says High and clearly larger when it says Low. A user can then decide which predictions require experimental review."],
    ["误差较小", "Smaller error"],
    ["误差居中", "Intermediate error"],
    ["误差较大", "Larger error"],
    ["没有判别力的置信度", "Non-discriminative confidence"],
    ["High、Medium、Low 的错误水平没有顺序，甚至模型越自信反而越容易错。这时置信标签不能帮助决策。", "High, Medium, and Low error levels have no meaningful order, or the model may even be more error-prone when it is more confident. In that case, the confidence labels do not support decisions."],
    ["误差不稳定", "Unstable error"],
    ["一句话", "In one sentence"],
    ["预测值回答“材料性能是多少”，置信度回答“这次预测有多值得相信”。本研究检查第二个答案是否真的与实际错误一致。", "The prediction answers 'what is the material property?', while confidence answers 'how much should this prediction be trusted?'. This study tests whether the second answer actually agrees with observed error."],
    ["判定标准", "Decision criteria"],
    ["希望观察到 High-to-Low normalized MAE 单调增加、Low minus High discrimination 为正，以及 tier risk 与样本绝对误差之间的 Spearman 相关为正。", "The desired pattern is a monotonic increase in normalized MAE from High to Low, positive Low-minus-High discrimination, and positive Spearman correlation between tier risk and sample absolute error."],
    ["结果来源", "Result sources"],
    ["所有最终数值均来自", "All final values come from"],
    ["和", "and"],
    ["哪些材料任务进入了比较", "Which material tasks were compared?"],
    ["完整 calibration 重建覆盖 10 个任务。为了让 GPT 与其他方法在完全相同的七个协议上比较，最终主图使用 8 个共同任务。", "The complete calibration rebuild covers 10 tasks. To compare GPT with the other methods under exactly the same seven protocols, the final main figures use 8 common tasks."],
    ["主图纳入的 8 个任务", "Eight tasks included in the main figures"],
    ["没有进入主图的任务", "Tasks excluded from the main figures"],
    ["为什么不能强行补齐", "Why missing cases were not filled in"],
    ["公平比较要求每位参赛者回答同一套题。HEA YS 没有 GPT 结果，Steel YS 的 GPT 只有两个协议，因此把它们放进主图会让不同方法使用不同题目。", "A fair comparison requires every participant to answer the same questions. HEA YS has no GPT result, and GPT covers only two protocols for Steel YS. Including them would make different methods use different test sets."],
    ["共同范围", "Common scope"],
    ["最终范围为 8 tasks x 7 protocols = 56 cases per method，六个方法合计 336 个 task-protocol-method 组合。每个方法均对齐到相同的 9,231 个样本位置。", "The final scope is 8 tasks x 7 protocols = 56 cases per method, or 336 task-protocol-method combinations across six methods. Every method is aligned to the same 9,231 sample positions."],
    ["范围审计", "Scope audit"],
    ["纳入任务、排除原因和每种方法的样本数由最终 manifest 固定。缺失置信层的具体案例记录在", "Included tasks, exclusion reasons, and per-method sample counts are fixed by the final manifest. Cases with missing confidence tiers are recorded in"],
    ["先选择超参数，再选择展示代表", "First select hyperparameters, then select category representatives"],
    ["这是两个完全不同的选择步骤。前者决定模型怎样训练，后者只决定图中用哪一个 Traditional 或 BERT 模型作为该类别代表。", "These are two distinct selection steps. The first determines how each model is trained; the second only determines which Traditional or BERT model represents its category in a figure."],
    ["Gate A：冻结模型参数", "Gate A: Freeze model parameters"],
    ["扫描已有 RandomCV 验证预测，用参数内容的 canonical hash 对齐同一配置。每个 task-model 选择平均验证 MAE 最低的配置，形成 110 条 frozen v1 记录。", "Existing RandomCV validation predictions are scanned, and identical configurations are aligned by a canonical hash of their parameter content. For each task-model pair, the configuration with the lowest mean validation MAE is selected, producing 110 frozen v1 records."],
    ["Gate B：选择类别代表", "Gate B: Select category representatives"],
    ["完成 OOD 预测后，每个 task-protocol 分别在 Traditional 和 BERT 类别中选一个代表。排序依据是 OOD-test MAE、MAE 标准差、R2 和模型名字典序。", "After OOD prediction, one representative is selected separately from the Traditional and BERT categories for each task-protocol pair. Ranking uses OOD-test MAE, MAE standard deviation, R2, and model name in that order."],
    ["类比", "Analogy"],
    ["Gate A 像在考试前确定每位选手使用哪套训练方案；Gate B 像在考试结束后，从同一代表队里选出该题表现最好的选手用于展示。两者不能混为一谈。", "Gate A is like fixing each participant's training plan before the examination. Gate B is like choosing, after the examination, the best-performing participant from the same team to display for that question. The two steps must not be conflated."],
    ["Gate A 的严格规则", "Strict Gate A rule"],
    ["；相同参数哈希在已有 RandomCV 外层折中求", "; identical parameter hashes are averaged across available RandomCV outer folds using"],
    ["。最低均值入选，缺失折不补值，OOD 测试文件不参与超参数选择。", ". The lowest mean is selected, missing folds are not imputed, and OOD test files never participate in hyperparameter selection."],
    ["可追溯文件", "Traceable files"],
    ["110 条 task-model 配置及 snapshot ID 位于", "The 110 task-model configurations and snapshot ID are stored in"],
    ["。112 条类别代表选择位于", ". The 112 category-representative selections are stored in"],
    ["真实 frozen v1 参数浏览器", "Frozen v1 parameter browser"],
    ["选择一个任务和模型，查看它为什么入选、使用了哪些折，以及最终传入模型的有效参数。", "Select a task and model to inspect why it was selected, which folds were observed, and the effective parameters passed to the model."],
    ["任务", "Task"],
    ["模型", "Model"],
    ["查看搜索参数", "Search parameters"],
    ["查看最终有效参数", "Effective parameters"],
    ["七种测试分别模拟什么困难", "What difficulty does each OOD protocol simulate?"],
    ["OOD 表示模型遇到与训练数据分布不同的样本。七个协议从随机基线、跨材料簇、性能外推和低密度区域等角度制造这种变化。", "OOD means that a model encounters samples drawn from a distribution different from its training data. The seven protocols create this shift through a random baseline, held-out material clusters, property extrapolation, and low-density regions."],
    ["随机交叉验证基线", "Random cross-validation baseline"],
    ["随机分成五折。测试样本通常与训练样本相似，是最熟悉的参考场景。", "Randomly split into five folds. Test samples are usually similar to training samples, making this the most familiar reference setting."],
    ["整簇留出", "Leave one cluster out"],
    ["先在成分和工艺特征空间聚类，再把一个完整材料簇留作测试。", "Cluster samples in composition and processing-feature space, then hold out one entire material cluster for testing."],
    ["目标性能外推", "Target-property extrapolation"],
    ["训练主要来自较低目标值，测试集中放入最高约 20% 的性能样本。", "Train mainly on lower target values and place approximately the highest 20% of property values in the test set."],
    ["稀疏特征单点", "Sparse-feature single points"],
    ["从特征空间低密度区域抽取分散的单个稀有样本。", "Select dispersed rare samples from low-density regions of feature space."],
    ["稀疏特征小簇", "Sparse-feature local clusters"],
    ["从稀疏特征区域选择种子，并加入邻近样本形成局部陌生区域。", "Choose seeds in sparse feature regions and add nearby samples to create locally unfamiliar regions."],
    ["稀疏目标单点", "Sparse-target single points"],
    ["根据目标值密度寻找罕见性能区间，并抽取分散单点。", "Find rare property intervals from target-value density and select dispersed single points."],
    ["稀疏目标小簇", "Sparse-target local clusters"],
    ["围绕罕见目标值样本扩展邻域，形成连续的困难测试区域。", "Expand neighborhoods around rare target-value samples to form continuous difficult test regions."],
    ["为什么需要七种", "Why seven protocols?"],
    ["一个模型可能擅长随机测试，却不擅长预测全新的合金区域。七种协议相当于从不同角度追问“离开熟悉数据以后，模型还可靠吗”。", "A model may perform well under random testing but fail on a new alloy region. The seven protocols ask from different angles whether a model remains reliable after leaving familiar data."],
    ["划分实现", "Split implementation"],
    ["RandomCV 使用固定随机种子的五折 KFold；LOCO 在 X 特征空间 KMeans 聚类并逐簇留出；Extrapolation 按目标排序留出高端约 20%；Sparse 协议使用 X 或 Y 密度选择低密度候选，再按 single 或 neighbor-expanded cluster 构建测试集。", "RandomCV uses five-fold KFold with a fixed seed. LOCO applies KMeans in X-feature space and holds out clusters in turn. Extrapolation sorts by target and holds out roughly the upper 20%. Sparse protocols use X or Y density to select low-density candidates, then construct either single-point or neighbor-expanded cluster test sets."],
    ["固定划分", "Fixed splits"],
    ["所有模型读取同一 canonical split，结果目录遵循", "All models read the same canonical split, and result directories follow"],
    ["，避免不同模型使用不同测试样本。", ", preventing different models from using different test samples."],
    ["六种方法如何产生“信心”", "How the six methods produce confidence"],
    ["不同模型没有相同的不确定性输出，因此先按各自原理生成 uncertainty，再统一转换为 High、Medium、Low。", "The models do not expose uncertainty in the same way. Each method first produces uncertainty according to its own mechanism, then maps it to High, Medium, and Low confidence."],
    ["方法", "Method"],
    ["不确定性从哪里来", "Uncertainty source"],
    ["固定设置", "Fixed settings"],
    ["置信层规则", "Confidence-tier rule"],
    ["该案例中 uncertainty 最低的约 1/3", "Approximately the lowest third of uncertainty within the case"],
    ["中间约 1/3", "Approximately the middle third"],
    ["uncertainty 最高的约 1/3", "Approximately the highest third of uncertainty"],
    ["同一模型使用 seeds 42-51 独立训练并预测 10 次。最终预测取 10 个预测值的均值，不确定性取样本标准差。", "The same model is independently trained and evaluated 10 times with seeds 42-51. The final prediction is the mean of the 10 predictions, and uncertainty is their sample standard deviation."],
    ["对选中的 BERT 代表模型，在同一 checkpoint 上保持 dropout 开启，用 seeds 42-71 做 30 次随机前向预测。", "For the selected BERT representative, dropout remains active on the same checkpoint while 30 stochastic forward passes are made with seeds 42-71."],
    ["与 BERT 使用同一类 MC-dropout 方法：同一 checkpoint、30 次随机前向预测、推理时 dropout 保持开启。", "LLM-Prop uses the same MC-dropout mechanism as BERT: one checkpoint, 30 stochastic forward passes, and dropout kept active during inference."],
    ["两个 TabPFN 版本分别处理，不互相平均。API 返回预测均值以及 q10、q50、q90，预测区间越宽表示越不确定。", "The two TabPFN variants are processed separately and are never averaged together. The API returns a predictive mean and q10, q50, and q90; a wider predictive interval indicates greater uncertainty."],
    ["共同的 High / Medium / Low 分组算法：", "Common High / Medium / Low assignment algorithm:"],
    ["对每个 task × protocol × model，先合并该协议的全部 split/fold 样本，仅按 uncertainty_score 从小到大排序；相同 uncertainty 再按 split_id、sample_id 和稳定行号确定顺序。随后把排序后的样本按数量切成三个近似等大的连续组：第一组为 High，中间组为 Medium，最后一组为 Low。分组不使用 y_true 或实际误差，因此不会用测试答案决定置信层。", "For each task × protocol × model, samples from all splits or folds of that protocol are merged and sorted only by uncertainty_score from low to high. Tied uncertainty values are ordered deterministically by split_id, sample_id, and stable row index. The ordered samples are then divided by count into three consecutive, nearly equal groups: the first is High, the middle is Medium, and the last is Low. Neither y_true nor observed error is used, so test answers cannot determine confidence tiers."],
    ["如何理解：", "How to interpret the tiers:"],
    ["High / Medium / Low 表示同一案例内部的相对置信度，不是材料性能值的高低，也不是把实际误差直接分成三档。原始 uncertainty 的量纲和范围不同，因此不能跨任务或跨模型直接比较其绝对数值。", "High, Medium, and Low indicate relative confidence within the same case. They do not mean high or low material-property values, and observed errors are not directly split into three bins. Raw uncertainty has method- and task-specific units and ranges, so its absolute values should not be compared across tasks or models."],
    ["为什么要重复预测", "Why repeat predictions?"],
    ["同一个样本让模型预测多次。如果预测值每次都很接近，模型相对稳定；如果上下波动很大，就说明模型对这个样本不确定。GPT 则直接保留其原始语义置信标签。", "The same sample is predicted repeatedly. Closely grouped predictions indicate stability; large variation indicates uncertainty about that sample. GPT instead retains its original semantic confidence label."],
    ["不确定性定义", "Uncertainty definitions"],
    ["Traditional 使用 seeds 42-51 的 10-member sample SD；BERT 和 LLM-Prop 使用 seeds 42-71 的 30-pass MC-dropout sample SD；TabPFN 使用 predictive q90-q10；非 GPT 方法按每个 task-protocol-model 内 uncertainty 的 1/3 与 2/3 分位点分层。", "Traditional uses the sample standard deviation of 10 ensemble members with seeds 42-51. BERT and LLM-Prop use the sample standard deviation of 30 MC-dropout passes with seeds 42-71. TabPFN uses predictive q90-q10. Non-GPT methods are tiered at the one-third and two-thirds uncertainty quantiles within each task-protocol-model case."],
    ["Traditional 使用 seeds 42-51 的 10-member sample SD；BERT 和 LLM-Prop 使用 seeds 42-71 的 30-pass MC-dropout sample SD；TabPFN 使用 predictive q90-q10。非 GPT 方法在每个 task-protocol-model 内按 uncertainty 稳定排序，再按样本秩切成三个近似等大的组。", "Traditional uses the sample standard deviation of 10 ensemble members with seeds 42-51. BERT and LLM-Prop use the sample standard deviation of 30 MC-dropout passes with seeds 42-71. TabPFN uses predictive q90-q10. Within each task-protocol-model case, non-GPT samples are stably sorted by uncertainty and divided by rank into three nearly equal groups."],
    ["样本合同", "Sample-level contract"],
    ["每条标准化记录包含 task、protocol、selected model、sample ID、y_true、y_pred、abs_error、uncertainty_score、confidence_tier、member_count 和 uncertainty_method。", "Each standardized record contains task, protocol, selected model, sample ID, y_true, y_pred, abs_error, uncertainty_score, confidence_tier, member_count, and uncertainty_method."],
    ["六种方法的真实样本示例", "Real sample examples from all six methods"],
    ["以下每行来自最终 integrated sample table 的一条实际记录。", "Each row below is an actual record from the final integrated sample table."],
    ["任务 / 协议", "Task / protocol"],
    ["真实值", "True value"],
    ["预测值", "Predicted value"],
    ["绝对误差", "Absolute error"],
    ["置信层", "Confidence tier"],
    ["GPT 结果怎样并入同一张表", "How GPT results enter the same table"],
    ["GPT 文件结构与本项目模型不同，因此先从 prediction inventory 发现源文件，再标准化任务名、协议名、fold 和 confidence。", "GPT files use a different structure from the project models. Source files are therefore discovered from the prediction inventory before task names, protocol names, folds, and confidence are standardized."],
    ["发现的 GPT prediction 文件", "GPT prediction files discovered"],
    ["进入共同 8 任务范围的文件", "Files included in the common eight-task scope"],
    ["去重后 GPT 样本记录", "Deduplicated GPT sample records"],
    ["六方法整合后的总记录", "Integrated records across six methods"],
    ["task-protocol-method 汇总案例", "Task-protocol-method summary cases"],
    ["标准化做了什么", "What standardization does"],
    ["把不同文件中的列名翻译成同一种格式，把 Matbench 任务别名统一，把 RandomCV 和 LOCO 的五个 fold 合并，并用 source file 与 sample ID 去除重复记录。", "It maps columns from different files into one schema, normalizes Matbench task aliases, merges the five RandomCV and LOCO folds, and removes duplicates using source file plus sample ID."],
    ["缺失层处理", "Missing-tier handling"],
    ["GPT 原始 High/Medium/Low 不重新三等分。缺少某个层时不生成虚构数据：tier 曲线使用现有层；discrimination 必须同时有 High 和 Low；Spearman 至少需要两个不同风险层。", "GPT's original High, Medium, and Low labels are not re-binned into thirds. Missing tiers are never fabricated: tier curves use available tiers, discrimination requires both High and Low, and Spearman requires at least two distinct risk tiers."],
    ["覆盖数字", "Coverage counts"],
    ["GPT 有 39/56 个完整三层案例、40/56 个 High/Low 可配对案例，以及 54/56 个有效 tier-Spearman 案例。", "GPT has 39/56 cases with all three tiers, 40/56 cases with a paired High and Low tier, and 54/56 cases with a valid tier-Spearman correlation."],
    ["每个缺失原因均记录在 coverage audit。", "Every missing-case reason is recorded in the coverage audit."],
    ["Traditional-best / BERT-best 代表选择浏览器", "Traditional-best / BERT-best representative browser"],
    ["选择一个共同任务和协议，查看这两个类别实际展示了哪个模型。TabPFN Numeric/Text 始终分开，不做 TabPFN-best。", "Select a common task and protocol to see which model represents each category. TabPFN Numeric and Text remain separate; no TabPFN-best selection is made."],
    ["协议", "Protocol"],
    ["每个数字是怎样算出来的", "How every number is calculated"],
    ["所有计算先在一个 task-protocol-method 案例内部完成，再让每个案例以相同权重进入总汇总。", "All calculations are completed within one task-protocol-method case before every case enters the overall summary with equal weight."],
    ["1. 样本绝对误差", "1. Sample absolute error"],
    ["先把高估和低估统一为非负错误大小。", "Convert overestimation and underestimation into a common nonnegative error magnitude."],
    ["2. 标准化 MAE", "2. Normalized MAE"],
    ["小于 1 表示该置信层比该方法自身平均情况更容易。", "A value below 1 means that the confidence tier is easier than the method's own average case."],
    ["3. 判别分数", "3. Discrimination score"],
    ["正值越大，说明 Low 与 High 的错误差异越明显。", "A larger positive value means a clearer error difference between Low and High confidence."],
    ["4. 层级 Spearman", "4. Tier-level Spearman"],
    ["risk 固定映射为 High=1、Medium=2、Low=3。", "Risk is fixed as High=1, Medium=2, and Low=3."],
    ["为什么要标准化", "Why normalize?"],
    ["不同任务的性能单位和难度差别很大。除以同一方法在该案例的 overall MAE 后，1 就成为共同基准，才能把铝合金强度和钛合金延伸率放在同一张图里。", "Property units and task difficulty differ substantially. Dividing by the same method's overall MAE within the case makes 1 a common baseline, allowing aluminum-alloy strength and titanium-alloy elongation to appear in one figure."],
    ["汇总规则", "Aggregation rule"],
    ["每个 task-protocol observation 等权，不按样本数加权。Panel A 的 95% CI 使用 10,000 次有放回 bootstrap，随机种子为 20260715。Spearman 只对有效案例求均值。", "Every task-protocol observation receives equal weight; sample counts do not determine weights. Panel A uses 10,000 bootstrap resamples with replacement for the 95% confidence interval, with random seed 20260715. Spearman values are averaged only across valid cases."],
    ["固定统计合同", "Fixed statistical contract"],
    ["计算定义、bootstrap 次数、种子、risk mapping 和 missing-tier policy 均写入最终 figure manifest，可与汇总 CSV 逐行复算。", "Metric definitions, bootstrap iterations, seed, risk mapping, and missing-tier policy are recorded in the final figure manifest and can be recomputed row by row from the summary CSV files."],
    ["用一个样本亲手算一次", "Calculate one example yourself"],
    ["改变输入值，观察绝对误差、normalized error 和 Low-High 判别分数怎样变化。", "Change the inputs to see how absolute error, normalized error, and the Low-High discrimination score respond."],
    ["真实值 y_true", "True value y_true"],
    ["预测值 y_pred", "Predicted value y_pred"],
    ["该案例 overall MAE", "Case-level overall MAE"],
    ["最终结果应该怎样读", "How to read the final results"],
    ["先看 High-to-Low 曲线是否上升，再看每个协议的 Low-minus-High 热图，最后用 Spearman 和有效覆盖率判断趋势是否稳定、是否有足够数据。", "First check whether the High-to-Low curves rise. Then inspect the Low-minus-High heatmap for each protocol. Finally, use Spearman correlation and valid coverage to judge whether the trend is stable and sufficiently supported."],
    ["“按协议汇总”怎样得到：", "How the protocol summary is produced:"],
    ["先在每个 task-protocol-method 内计算 tier normalized MAE，再对 8 个共同 task 做等权平均，不按各 task 的样本数加权。GPT 某个 tier 缺失时，该 tier 只平均具有有效值的 task，并在数据表中保留 observation count。", "Tier-level normalized MAE is first calculated within each task-protocol-method case. The values are then averaged equally across the eight common tasks, without weighting by task sample size. When GPT lacks a tier, that tier is averaged only across tasks with a valid value, and the observation count is retained in the data table."],
    ["主结果图", "Main result"],
    ["按协议汇总", "By protocol"],
    ["按方法汇总", "By method"],
    ["按任务对比", "By task"],
    ["选择任务", "Select task"],
    ["主图同时展示层级曲线、协议判别分数、Spearman 和有效覆盖率。", "The main figure combines tier curves, protocol-level discrimination, Spearman correlation, and valid coverage."],
    ["重要边界：", "Important boundary:"],
    ["Calibration 衡量的是置信度能否区分易错和不易错样本，不等同于模型预测精度排名。一个方法可以平均误差较低，但置信度判别较弱；也可以平均误差较高，但知道自己何时容易错。", "Calibration measures whether confidence separates error-prone from less error-prone samples. It is not a ranking of prediction accuracy. A method can have low average error but weak confidence discrimination, or higher average error while correctly recognizing when it is likely to fail."],
    ["主要发现", "Main findings"],
    ["GPT-5.4、两个 TabPFN 版本和 Traditional-best 的 High-to-Low 错误总体上升，其中 GPT-5.4 的平均 normalized MAE 约为 0.58、1.12、1.44。BERT-best 的总体曲线不单调，说明其 confidence tier 作为风险提示较不稳定。", "GPT-5.4, both TabPFN variants, and Traditional-best show an overall increase in error from High to Low. GPT-5.4 has mean normalized MAE values of approximately 0.58, 1.12, and 1.44. The overall BERT-best curve is non-monotonic, indicating that its confidence tiers are less stable as risk signals."],
    ["结果解释", "Result interpretation"],
    ["GPT-5.4 的跨协议平均 discrimination 约为 0.92，有效 tier-Spearman 为 54/56；TabPFN Numeric/Text 的平均 discrimination 约为 0.78/0.80，且 56/56 Spearman 有效。BERT-best 的平均 discrimination 接近 0，并在部分 SparseY 协议为负。", "GPT-5.4 has mean discrimination of about 0.92 across protocols and valid tier-Spearman values in 54/56 cases. TabPFN Numeric/Text have mean discrimination of about 0.78/0.80 and valid Spearman values in all 56 cases. BERT-best has mean discrimination close to zero and is negative under some SparseY protocols."],
    ["原始汇总下载", "Download source summaries"],
    ["层级汇总 CSV", "Tier summary CSV"],
    ["协议汇总 CSV", "Protocol summary CSV"],
    ["任务级汇总 CSV", "Task-level summary CSV"],
    ["覆盖审计 CSV", "Coverage audit CSV"],
    ["可直接用于复算图中每个数字。", "These files can be used to recompute every number in the figures."],
    ["六种方法的总体 High / Medium / Low", "Overall High / Medium / Low for all six methods"],
    ["均值由 56 个 task-protocol 案例等权汇总；GPT 缺失层只使用有效案例，因此各层 observation count 不同。", "Means are aggregated with equal weight over 56 task-protocol cases. Missing GPT tiers use valid cases only, so observation counts differ by tier."],
    ["High-Low 观察数", "High / Medium / Low observations"],
    ["怎样证明这张图没有取错数据", "How do we know the figures use the correct data?"],
    ["最终发布不是“代码运行完了”就结束，而是重新核对快照、样本数、分割、结果文件、保护哈希和图片质量。", "Publication does not end when the code finishes. The snapshot, sample counts, splits, result files, protected hashes, and figure quality are checked again."],
    ["最终 full verification 状态", "Final full-verification status"],
    ["通过的完整运行单元", "Complete run units passed"],
    ["整合样本数，六方法各 9,231", "Integrated samples, 9,231 per method"],
    ["最终 GPT 图片文件，11 张逻辑图 x 3 格式", "Final GPT figure files: 11 logical figures x 3 formats"],
    ["复制", "Copy"],
    ["保护原则", "Protection principle"],
    ["aggregate、frozen v1、canonical splits、GPT 原始文件和 failed_attempts 在绘图前后都保持不变。最终 manifest 保存输入与输出 SHA256。", "The aggregate files, frozen v1, canonical splits, original GPT files, and failed_attempts remain unchanged before and after plotting. The final manifest records input and output SHA256 hashes."],
    ["为什么需要哈希", "Why hashes are needed"],
    ["SHA256 像文件的指纹。只要文件内容改变一个字符，指纹就会变化，因此可以证明绘图过程没有悄悄修改训练结果或数据划分。", "SHA256 acts like a file fingerprint. Changing even one character changes the fingerprint, which demonstrates that plotting did not silently alter training results or data splits."],
    ["验证范围", "Verification scope"],
    ["验证包括 snapshot 一致性、完整单元计数、canonical split 对齐、quantile/runtime/prediction 合同、protected hashes、汇总行数、输出数量、PNG DPI 与像素方差、PDF/SVG 可解析性。", "Verification covers snapshot consistency, complete-unit counts, canonical-split alignment, quantile/runtime/prediction contracts, protected hashes, summary row counts, output counts, PNG DPI and pixel variance, and PDF/SVG parseability."],
    ["证据清单", "Evidence inventory"],
    ["下面的文件是本页面复制的只读证据副本。源文件绝对路径与 SHA256 由页面数据包保存。", "The files below are read-only evidence copies included with this page. Their absolute source paths and SHA256 hashes are stored in the page data package."],
    ["下载复现证据", "Download reproducibility evidence"],
    ["这些文件足以追踪参数冻结、类别代表、统计汇总、覆盖缺失和最终图 manifest。", "These files trace parameter freezing, category representatives, statistical summaries, coverage gaps, and the final figure manifest."],
    ["文件", "File"],
    ["用途", "Purpose"],
    ["下载", "Download"],
    ["本说明页由 verified calibration artifacts 自动构建。页面用于解释结果获取过程，不重新训练模型，也不修改任何原始结果文件。", "This explainer is built automatically from verified calibration artifacts. It explains how the results were obtained; it neither retrains models nor modifies original result files."],

    ["页面数据未加载，请确认 data/pipeline_data.js 与 index.html 位于同一交付目录。", "Page data could not be loaded. Confirm that data/pipeline_data.js and index_en.html are in the same delivery directory."],
    ["10 个独立随机种子训练成员", "10 independently seeded training members"],
    ["Seeds 42-51；预测均值作为 y_pred；样本标准差使用 ddof=1", "Seeds 42-51; the prediction mean is y_pred; sample standard deviation uses ddof=1"],
    ["在 task-protocol-model 内稳定排序并按样本秩近似等分三组", "Stable within-case sorting followed by a near-equal three-way rank split"],
    ["同一 checkpoint 的 30 次随机 MC-dropout 预测", "30 stochastic MC-dropout predictions from one checkpoint"],
    ["Seeds 42-71；推理时保持 dropout 开启；样本标准差 ddof=1", "Seeds 42-71; dropout remains active at inference; sample standard deviation uses ddof=1"],
    ["API 预测分位数", "API predictive quantiles"],
    ["Uncertainty = q90 - q10，即 80% 预测区间宽度", "Uncertainty = q90 - q10, the width of the 80% predictive interval"],
    ["原始语义置信度", "Original semantic confidence"],
    ["直接读取 prediction record 中的 High、Medium 或 Low", "Read High, Medium, or Low directly from the prediction record"],
    ["不重新分箱，也不填补缺失置信层", "No re-binning and no missing-tier imputation"],
    ["没有找到该参数记录。", "No parameter record was found."],
    ["模型家族", "Model family"],
    ["平均验证 MAE", "Mean validation MAE"],
    ["观测折", "Observed folds"],
    ["参数哈希", "Parameter hash"],
    ["无选择记录", "No selection record"],
    ["GPT-integrated calibration 主结果图", "GPT-integrated calibration main figure"],
    ["按协议汇总的 calibration 图", "Calibration summary by protocol"],
    ["逐协议比较六种方法的 High、Medium、Low 变化。", "Compare High, Medium, and Low behavior for all six methods within each protocol. Each plotted value is an equal-task average across the eight common tasks."],
    ["按方法汇总的 calibration 热图", "Calibration heatmaps by method"],
    ["比较方法与协议组合的判别分数和层级 Spearman。", "Compare discrimination and tier-level Spearman across method-protocol combinations."],
    ["110 条冻结参数与 snapshot", "110 frozen parameter records and snapshot"],
    ["112 条 Traditional/BERT 代表选择", "112 Traditional/BERT representative selections"],
    ["High/Medium/Low 汇总与 bootstrap CI", "High/Medium/Low summaries and bootstrap confidence intervals"],
    ["协议判别分数与 Spearman", "Protocol-level discrimination and Spearman"],
    ["逐 task-protocol-method 的指标与覆盖", "Metrics and coverage for each task-protocol-method case"],
    ["缺失 tier 与有效案例审计", "Missing-tier and valid-case audit"],
    ["输入输出哈希、数量和图片 QA", "Input/output hashes, counts, and figure QA"],
    ["已复制", "Copied"],
    ["请手动选择", "Select manually"],
    ["放大结果图", "Open enlarged result figure"],
    ["复制 snapshot ID", "Copy snapshot ID"],
    ["关闭", "Close"],
    ["放大的 calibration 结果图", "Enlarged calibration result figure"],
    ["解释深度", "Explanation level"],
    ["页面章节", "Page sections"],
    ["项目范围摘要", "Project scope summary"],
    ["Calibration 全流程", "Complete calibration workflow"],
    ["结果图选择", "Result figure selection"],
    ["四类模型的不确定性计算", "Uncertainty calculations for four model groups"]
  ]);

  const translatePattern = (value) => {
    let match = value.match(/^GPT 有 (\d+)\/(\d+) 个完整三层案例、(\d+)\/(\d+) 个 High\/Low 可配对案例，以及 (\d+)\/(\d+) 个有效 tier-Spearman 案例。$/);
    if (match) {
      return `GPT has ${match[1]}/${match[2]} cases with all three tiers, ${match[3]}/${match[4]} cases with paired High and Low tiers, and ${match[5]}/${match[6]} cases with a valid tier-Spearman correlation.`;
    }
    match = value.match(/^当前展示任务 (.+)：逐协议比较六种方法的 High、Medium、Low normalized MAE。$/);
    if (match) {
      return `Task shown: ${match[1]}. The figure compares High, Medium, and Low normalized MAE across all six methods within each protocol.`;
    }
    match = value.match(/^(.+) calibration 按协议对比图$/);
    if (match) return `${match[1]} calibration comparison by protocol`;
    return null;
  };

  const translateValue = (value) => {
    const compact = String(value ?? "").trim();
    if (!compact) return value;
    const replacement = translations.get(compact) ?? translatePattern(compact);
    return replacement === null || replacement === undefined
      ? value
      : String(value).replace(compact, replacement);
  };

  const translateAttributes = (element) => {
    ["aria-label", "title", "alt"].forEach((name) => {
      if (!element.hasAttribute?.(name)) return;
      const current = element.getAttribute(name);
      const translated = translateValue(current);
      if (translated !== current) element.setAttribute(name, translated);
    });
  };

  const translateNode = (root) => {
    if (root.nodeType === Node.TEXT_NODE) {
      const parentTag = root.parentElement?.tagName;
      if (parentTag === "SCRIPT" || parentTag === "STYLE") return;
      const translated = translateValue(root.nodeValue);
      if (translated !== root.nodeValue) root.nodeValue = translated;
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_NODE) return;
    if (root.nodeType === Node.ELEMENT_NODE) translateAttributes(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      if (node.nodeType === Node.TEXT_NODE) {
        const parentTag = node.parentElement?.tagName;
        if (parentTag !== "SCRIPT" && parentTag !== "STYLE") {
          const translated = translateValue(node.nodeValue);
          if (translated !== node.nodeValue) node.nodeValue = translated;
        }
      } else {
        translateAttributes(node);
      }
      node = walker.nextNode();
    }
  };

  document.title = "How the Calibration Results Were Produced";
  document.body.dataset.language = "en";
  document.querySelectorAll("[data-language-link]").forEach((link) => {
    if (link.dataset.languageLink === "en") link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  translateNode(document.body);

  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === "characterData") translateNode(mutation.target);
      mutation.addedNodes.forEach(translateNode);
    });
  });
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();
