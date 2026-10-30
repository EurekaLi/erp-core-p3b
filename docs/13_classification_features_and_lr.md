# 共享数据清单、紧凑特征与Logistic Regression：执行与解释

## 1. 当前交付与尚未执行的工作

本轮实现三个入口，由使用者在`erp-core-p3b-dl`环境依次执行：

|入口|目的|是否训练|
|---|---|---|
|09_build_classification_manifest.py|建立trial身份、固定分组划分、整理已有证据|否|
|10_extract_compact_features.py|逐trial计算ERP/功率/小波特征|否|
|11_train_logistic_baseline.py|一折smoke或完整五折LR增量对照|是，CPU|

开发验证使用临时合成受试者和CPU模型，没有读取真实全体数据训练或编造项目成绩。
真实结果要在执行命令后产生。程序不使用GPU、不读取CBraMod权重，因而可以与深度学习支线错峰/并行。

本阶段不会重新滤波、ICA、插值或裁决trial。共用基础仍是39人裁决版7207个trial，
Target 1286、Standard 5921。LR使用原生256Hz、257点的短EEG，不使用CBraMod的200Hz候选网格。

## 2. 为什么先清单，再特征，再模型

```text
裁决版FIF＋Phase 5清单
    ↓ 09：trial_manifest＋subject_splits＋证据汇总
    ↓ 10：每trial固定公式 → 40列特征＋字典
    ↓ 11：每折训练内Scaler/LR → 验证／外层测试预测
    ↓ 受试者等权指标、OOF合并指标、CI、成对特征增量
```

三程序独立，不要求一次跑完。文件夹已存在时拒绝覆盖；失败保留run_summary.json诊断。
上游只有status=completed且产物SHA-256一致时才允许继续，避免混用旧划分、新特征或半成品。
每阶段记录软件版本、项目模块hash、配置快照以及数据/划分/特征定义身份。
这些校验用于防误改和追溯，不是对恶意篡改的数字签名认证。

## 3. 环境与前提

无需再建环境。需要MNE、NumPy、SciPy、pandas、scikit-learn、PyWavelets、joblib、PyYAML、
threadpoolctl和matplotlib；这些已属于当前环境准备范围。

先确认当前解释器并进入根目录：

```powershell
Set-Location -LiteralPath 'D:\AILearning\project_for_offer\erp-core-p3b'
python -X utf8 -c "import sys; print(sys.executable)"
```

应指向`D:\LearnWareSoft\CondaEnvs\erp-core-p3b-dl\python.exe`。
`-X utf8`避免Windows默认编码输出中文时出现乱码或编码异常；不改变信号数据。

前提文件：39人adjudicated FIF、Phase 5 analysis_manifest.json、
裁决版subject_preprocessing.csv和subject_measurements.csv。
后两表只作为证据汇总，里面的ERP组结果、ICA数量等不会成为分类特征。

## 4. 09：构造共享清单与固定划分

### 4.1 运行

```powershell
python -X utf8 scripts/09_build_classification_manifest.py
```

默认输出`data/processed/classification/v1/`。
程序逐人读取FIF头和metadata，不载入全体波形。核对事件、通道、单位、采样率、滤波头信息、
既有baseline、逐人类别计数及时间网格，并对每份源FIF计算hash。

### 4.2 trial清单的读法

每行代表一条已经保留的trial：

- trial_id：例如sub-001:trial-0001，在所有模型中保持同一身份；
- subject_id：用于分组，不作为模型输入；
- trial_index：原实验编号，保留原编号缺口；
- epoch_index：该FIF中当前行号，用于读取信号；
- event_sample_256hz：预处理后事件位置，不是原1024Hz sample；
- event_code、condition、label：核对事件到标签的映射；
- source_file：项目相对路径，便于在另一台机器复现。

标签固定Standard=0、Target=1。不能将刺激事件码、正确性、反应时间或原事件sample作为输入。
每人的所有trial绑定同一subject_id，不允许拆开。

### 4.3 分组规则

外层为StratifiedGroupKFold，5折、shuffle=True、seed=42；尽量保持类别比例，但分组优先，
不保证完全平衡。每折测试受试者只来自该折测试组。
在外层训练受试者内用GroupShuffleSplit留出20%受试者作验证，seed=42+外层折号。
比例按受试者数量计算，不是按trial数量计算，实际人数以split_audit.csv为准。

程序验证每折三集合不重叠、合起来覆盖全体，且每条trial恰好被外层测试一次。
不依据模型分数重选随机种子，不因为Target少而额外排除sub-006等受试者。

参考：[分组分层划分文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)。

### 4.4 输出和验收

|文件|用途|
|---|---|
|trial_manifest.csv|7207行共同样本索引|
|dataset_manifest.json|数据hash、源文件hash、时间轴、30通道顺序、头信息协议|
|subject_splits.json|每折三组受试者和trial_id清单|
|split_audit.csv|5×3行人数、trial数、两类计数与比例|
|subject_evidence_summary.csv|39人既有QC/ICA/ERP证据按subject合并，来源列有前缀|
|config_snapshot.json、run_summary.json|参数、版本、状态、文件校验|

dataset_hash表示源信号与trial身份；split_hash单独表示实际划分。
同一数据换种子可能dataset_hash相同，但split_hash不同，不能把两种划分的模型直接当同批对照。

配置复用configs/cbramod_audit.yaml的dataset节以避免重复定义标签/采样率。
CBraMod的适配和权重参数不参与此阶段。证据表缺失/身份重复时停止，不偷偷跳过。

## 5. 10：紧凑特征提取

### 5.1 先pilot，再全体

可选三人pilot，用来检查公式和输出：

```powershell
python -X utf8 scripts/10_extract_compact_features.py --stage pilot
```

默认输出`data/processed/features/compact_v1_pilot/`。pilot仅包含三人，不允许输入LR当作正式全体数据。

随后执行全体：

```powershell
python -X utf8 scripts/10_extract_compact_features.py --stage full
```

默认输出`data/processed/features/compact_v1/`。
依次载入一人的FIF，核对原文件hash、trial身份、标签、通道和时间轴，计算后释放，不同时载入全体连续信号。

### 5.2 ERP幅值：12列

在CPz、Pz、PO3、PO4四通道的三个闭区间200–300、300–600、600–800ms计算平均电位。

```text
mean_uv = mean(single_trial_voltage_v × 1,000,000)
```

每条trial独立计算；绝不把条件平均ERP复制给所有trial。时间掩码使用实际采样时刻。
例如原末点0.80078125秒大于0.800秒，不属于600–800ms特征；字典记录实际端点和样本数。
不以全体分类测试分数重新挑窗或挑通道。

### 5.3 Welch功率：12列

取0–800ms实际205点，Hann窗、nperseg=128、noverlap=64、nfft=128、detrend=constant。
256Hz下频率间距2Hz，共两个完整Welch段；不能将这解释为高精度的0.8秒时间分辨率。

按固定bin矩形积分近似：

```text
band_power_uv2 = sum(selected_PSD_in_uv2_per_hz) × 2Hz
feature = log10(max(band_power_uv2, 1e-12µV²))
```

实际频率bin：theta为4、6Hz；alpha为8、10、12Hz；beta为14至30Hz、步长2Hz。
粗bin规则并不等同于理想连续4–8、8–13、13–30Hz带宽。
log下限只防止log(0)，不是QC阈值。30Hz低通会影响beta上沿。

参考：[SciPy Welch PSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)。

### 5.4 DWT：16列

在完整257点上计算db4、level=5、symmetric边界，检查最大允许层数。
PyWavelets返回A5、D5、D4、D3、D2、D1六块系数。
每通道取D5/D4/D3相对能量，再取六块能量比例的归一化Shannon熵。

```text
E_k = sum(coefficients_k²)
p_k = E_k / sum(all_six_E_blocks)
entropy = -sum(p_k × ln(p_k)) / ln(6)
```

全零能量时相对能量和熵明确设0，并记录zero_energy_channel_trials，不自动删行。
标称D5≈4–8Hz、D4≈8–16Hz、D3≈16–32Hz，不是理想滤波器，也不与Welch频带严格一一对应。
边界延拓使系数能量比例不能无条件当作精确物理能量守恒。

参考：[PyWavelets多层DWT](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html)。

### 5.5 输出和验收

features.csv预期7207行、43列：trial_id/subject_id/label三个索引字段＋40个特征。
ERP 12、功率12、小波16。feature_dictionary.json逐列记录名字、族、单位、公式、窗口、频率bin和边界。
subject_feature_audit.csv记录逐人样本数、Welch段数和小波层数；描述统计仅用于检查，不作为训练Scaler参数。

要求所有值有限、身份与清单一致、所有受试者定义一致。不会自动补缺失、删除异常行或拟合PCA。
固定单trial公式可以提取全体；需要跨样本学习的Scaler/PCA/特征选择必须限制在训练部分。

## 6. 11：先smoke，再正式五折评估

### 6.1 一折ERP试运行

```powershell
python -X utf8 scripts/11_train_logistic_baseline.py --mode smoke --run-name lr-smoke
```

只运行第一折、ERP 12列，只拟合训练受试者并计算验证受试者成绩。
不产生test_predictions.csv，不计算外层测试成绩。验证trial数取决于固定划分，不必是完整7207行。
目的为工程闭环，不根据这一步反复新增窗口或种子；它不是正式性能结论。

### 6.2 正式运行

```powershell
python -X utf8 scripts/11_train_logistic_baseline.py --mode full --run-name lr-full
```

固定三实验：ERP 12列；ERP＋功率24列；ERP＋功率＋小波40列。每组5折，共15个LR模型。
每折仍留出共同验证受试者，即使LR不需要早停也不把验证组重新加入训练，保证与深度学习的训练人数公平。
特征名单只来自数据字典，不能把metadata列自动选入模型。

### 6.3 LR原理和训练范围

标准化后，LR计算`sigmoid(w·z+b)`，学习一个特征空间里的线性判别边界。
C=1、lbfgs、默认L2正则化、max_iter=2000；C更小代表更强的正则约束。
class_weight=balanced使用该折训练标签：每类权重为N_train/(2×该类训练trial数)，减轻多数类主导损失。

Scaler只对训练数组fit。验证和测试只调用已有Pipeline的predict_proba，不更新Scaler或LR。
固定阈值0.5，没有C搜索、阈值搜索或PCA。不收敛就非零退出并保留诊断，不静默输出该折成绩。

加权训练后的sigmoid输出是模型分数，不自动等于原始类别比例下校准良好的概率，
更不代表临床诊断概率。本项目预测的是Target/Standard，不是疾病。

参考：[LogisticRegression接口](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。

### 6.4 正式输出和读法

默认根目录`results/traditional_lr/lr-full/`：

|文件|用途|
|---|---|
|test_predictions.csv|三实验完整OOF，共预期21621行，每组7207条恰好一次|
|test_subject_metrics.csv|39×3=117行，每人每实验一个指标记录|
|subject_macro_ci.csv|主结果：受试者等权平均和bootstrap CI|
|pooled_test_metrics.csv|辅助结果：合并trial指标，不与主结果混称|
|paired_feature_comparisons.csv|相同受试者的特征族增量差值及CI|
|standard_only_reference.csv|恒预测Standard的参考，不是训练出的LR|
|validation_predictions.csv、validation_subject_metrics.csv|验证组检查，不能当OOF测试结果|
|fit_audit.csv|每折训练人数、类权重、特征数、收敛迭代数|
|coefficients.csv|标准化系数、仅训练部分的均值/尺度、截距|
|models/*.joblib|包含训练Scaler的LR Pipeline，仅加载可信来源|
|subject_splits.json、feature_dictionary.json|本轮划分与特征定义快照|
|results_report.md、figures/|阅读报告、逐人BA、三组混淆矩阵|

balanced accuracy为Target召回率与Standard召回率的平均；Standard恒预测参考BA=0.5。
ROC-AUC评价排序，Average Precision用于不平衡类别；AP不与梯形PR-AUC混称。
同时报告Target precision/recall/F1和混淆计数。无预测Target时precision/F1采用0的明确约定；
缺少某类时BA/ROC-AUC/AP缺失而不是硬补0，主数据每人两类均存在。

主CI按39名受试者进行2000次bootstrap，而不是独立重采样相关trial。
这是条件于固定划分与已拟合模型的描述性CI，不包含重训模型的不确定性。
成对差值也是受试者成对bootstrap，不进行多重检验校正、不输出“因果贡献”结论。
LR系数是特征权重，不是脑源强度；相关特征可能相互分配系数。

## 7. 安全重跑、参数文件与排错

三个配置分别为classification_dataset.yaml、compact_features.yaml、modeling.yaml。
旧modeling.yaml的未执行LDA/SVM/置换规划已收敛为当前LR闭环，不改变任何已有ERP结果。

不要手工编辑已完成目录中的CSV/JSON；修改后下游会拒绝使用。
不要通过删除失败记录来隐藏问题。需要重跑时使用新版本/新目录，并让下游显式引用：

```powershell
python -X utf8 scripts/09_build_classification_manifest.py --output-dir data/processed/classification/v2
python -X utf8 scripts/10_extract_compact_features.py `
    --dataset-dir data/processed/classification/v2 `
    --output-dir data/processed/features/compact_v2 --stage full
python -X utf8 scripts/11_train_logistic_baseline.py `
    --dataset-dir data/processed/classification/v2 `
    --features-dir data/processed/features/compact_v2 `
    --mode full --run-name lr-full-v2
```

遇到hash不符先检查上游文件是否被修改；遇到缺类、非有限值、字典不一致先定位数据问题。
不要通过删trial、更换标签或把校验关闭来“跑通”。
配置改动应注明事前定义/事后探索，并另存新版本；后续调参只使用共同验证组或训练组内分组验证。

## 8. 与深度学习并行、以及结论边界

深度学习直接复用09的trial_id与subject_splits；波形缓存可以不同，但数据版本和划分身份必须匹配。
不允许LR使用随机trial划分、神经网络使用受试者划分后直接比较成绩。
CPU仅开少量线程、逐人处理；GPU一次只跑一个训练任务，避免同时写同一产物目录。

上游每人整段记录参与无监督ICA/QC，这是离线清洗后的受试者独立分类，不是严格在线盲测。
既有全体ERP结果已被查看，本阶段不依据全体显著图再选新窗/新电极，不冒称全研究前瞻预注册。
插值通道可能影响局部频谱/小波，解释特征时结合subject_evidence_summary的QC记录。
TFR保护支持复核、预训练数据重叠审计、多随机种子和置换检验继续作为独立工作，不将它们伪称已完成。
