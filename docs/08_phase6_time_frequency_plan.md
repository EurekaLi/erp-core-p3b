# Phase 6：时频分析与特征工程计划

## 1. 本阶段的目标

Phase 5 已经证明 Target 相比 Standard 存在稳定的 P3b 时域效应。Phase 6 不再重复回答“有没有P3b”，而是回答三个新问题：

1. 条件差异在频率维度上主要表现在哪些频带、时间和头皮位置；
2. ERP、时频和离散小波能否形成可审计的单trial特征表；
3. 如何为Phase 7分类准备特征，同时严格避免标准化、特征选择和PCA的数据泄漏。

本阶段仍然使用39名受试者的裁决版数据。`sub-030`只保留为敏感性材料，不进入主要特征表。

## 2. 为什么不能直接对现有epochs运行2 Hz Morlet

当前ERP epochs约为-0.2--0.8 s，总长度约1 s。2 Hz信号的一个周期为0.5 s；如果Morlet使用3个周期，一个小波本身就约1.5 s，比整个epoch更长。此外，当前刺激前只有约200 ms，无法为2--4 Hz功率提供稳定的基线估计。

若直接运行，程序仍可能返回数值，但低频结果会受到严重边缘效应和零填充影响，看起来平滑却不可信。因此Phase 6的第一项工作不是立即画时频图，而是从连续数据复用Phase 4的坏道、ICA和重参考决定，构造专用的长时间窗TFR epochs。

建议的TFR分支为：

- 滤波：0.1--40 Hz；
- 采样率：256 Hz；
- epoch：暂定-1.5--2.0 s；
- baseline：暂定-0.8---0.2 s；
- 正式展示范围：-0.2--0.8 s；
- 不在长epoch上再次拟合ICA，复用Phase 4已经冻结的ICA模型和排除成分。

“暂定”表示必须先检查相邻事件间隔、记录边界和可用epoch数量。若大量trial无法获得完整的-1.5--2.0 s数据，应根据审计结果统一缩短边界，而不能逐受试者使用不同时间窗。

## 3. Phase 6的七个子阶段

### 6.1 长epoch可行性审计

先编写只读审计程序，逐名受试者检查：

- 每个刺激距记录开头和结尾的时间；
- 相邻刺激间隔的分布；
- 使用-1.5--2.0 s时可保留多少Target和Standard trial；
- 长epoch预计磁盘空间和单受试者峰值内存；
- Phase 4裁决版trial能否通过原始事件索引重新对应到连续数据。

这一环节的输出应是审计CSV和分布图，不生成正式TFR结果。验收标准是为所有受试者冻结统一的长epoch边界和基线。

### 6.2 构造TFR专用信号分支

复用Phase 4参数执行：坏道登记、滤波、降采样、已冻结ICA排除、坏道插值和平均参考。随后依据Phase 4保留的trial审计记录构造长epochs。

建议只保存重建长epochs所需的审计信息，或者按受试者保存压缩FIF；不要把39名受试者同时加载到内存。

需要验证：

- 新分支与ERP分支在重叠的-0.2--0.8 s区间波形一致；
- Target/Standard trial身份没有改变；
- 不会把Phase 4已经剔除的trial重新加入；
- 长epoch的边缘没有被误当作生理信号。

### 6.3 Morlet时频与ITC

第一版建议使用2--30 Hz、1 Hz步长的Morlet频率网格。周期数不应对所有频率使用同一个常数，可采用随频率增加的规则，例如：

```text
n_cycles(f) = clip(f / 2, 3, 10)
```

这表示低频至少使用3个周期以保持基本频率分辨率，高频最多10个周期以避免时间分辨率过差。正式规则必须写入YAML并在结果中记录。

每个条件需要区分：

- 总功率：单trial功率后再平均，包含相位锁定和非相位锁定活动；
- ITC：不同trial在某一时间—频率点的相位一致性；
- 条件差：Target - Standard。

第一版不必同时开展复杂的“诱发功率减去evoked”分析。可以先完成总功率和ITC，待其结果和实现稳定后再把诱导功率列为探索性扩展。

功率建议转换为相对于基线的dB：

```text
10 × log10(当前功率 / 基线平均功率)
```

dB比没有单位说明的“logratio”更容易解释。ITC范围为0--1，不做相同的dB基线变换。

### 6.4 传感器级群体时频结果

主要展示位置建议沿用Pz，并增加Phase 5的后部ROI：`CPz、Pz、PO3、PO4`。第一版图形包括：

- Pz的Target、Standard和条件差时频图；
- 后部ROI的条件差时频图；
- delta、theta、alpha、beta频带时间曲线；
- 预先定义时间—频带窗口的头皮图；
- 个体条件差热图。

建议的频带仅作为汇总标签：

- delta：2--4 Hz；
- theta：4--8 Hz；
- alpha：8--13 Hz；
- beta：13--30 Hz。

频带边界在配置中只保存一份。边界频点属于哪个频带必须采用一致规则，避免4 Hz或8 Hz被重复计算。

若进行时间—频率—空间cluster permutation，应明确标记为探索性，并控制数据维度。优先在后部ROI或预设频带上检验，不直接对所有时间、频率和通道构造难以解释的超大搜索空间。

### 6.5 离散小波特征

使用PyWavelets和`db4`对单trial信号进行离散小波分解。256 Hz采样下，5层分解的大致频率对应为：

| 系数 | 近似频段 |
|---|---|
| D3 | 16--32 Hz |
| D4 | 8--16 Hz |
| D5 | 4--8 Hz |
| A5 | 0--4 Hz |

D1和D2主要对应已经被30/40 Hz低通显著衰减的高频，因此不能把它们包装成有意义的神经特征。实际频段还受滤波器响应和边界扩展模式影响，报告中应称为近似对应。

每个有效系数层提取：

- 相对能量；
- 小波熵；
- 系数均值和标准差；
- 必要时增加绝对中位数和四分位距作为稳健统计量。

必须固定小波族、分解层数、边界模式、输入时间窗和通道/ROI。不能为每名受试者单独挑选效果最好的小波参数。

### 6.6 单trial特征表与数据字典

最终特征表应保持一行一个trial，并至少包含：

```text
subject, trial_id, condition, stimulus_onset_s,
target_or_standard, retained_by_qc,
erp_*, bandpower_*, morlet_*, wavelet_*
```

其中`subject`、`trial_id`和`condition`是索引/分组字段，不得作为模型输入特征。建议分成：

- `trial_index.csv`：trial身份与元数据；
- `features_erp.csv.gz`；
- `features_tfr.csv.gz`；
- `features_wavelet.csv.gz`；
- `feature_dictionary.csv`：特征名、单位、通道、时间窗、频带、算法和解释。

不保存完整的trial × channel × frequency × time四维数组。长epoch下这种数组可能达到数十GB。程序应逐受试者计算，并只保存条件平均图和预先定义的紧凑特征。

验收检查包括：

- trial总数仍为7207，除非长epoch边界审计明确记录了统一排除；
- 主键唯一，没有一个trial重复出现；
- 没有NaN或无穷值；
- 条件计数与Phase 5一致；
- 每个特征都有数据字典和单位；
- 输出使用float32降低磁盘占用，但统计汇总使用float64。

### 6.7 PCA与Phase 7接口

Phase 6只生成未经全数据PCA的原始特征表。PCA、标准化、方差过滤和特征选择必须在Phase 7的交叉验证Pipeline内部拟合：

```text
训练受试者
  -> 缺失值处理
  -> 标准化
  -> PCA（仅训练折fit）
  -> 分类器
  -> 对测试受试者transform/predict
```

禁止先对7207个trial整体执行PCA再做受试者分组交叉验证，因为测试受试者的协方差结构会提前泄漏进训练过程。Phase 6应为此编写防泄漏测试，但不报告基于全体数据拟合的PCA分类性能。

## 4. 推荐的实现顺序

1. `scripts/06_audit_tfr_epochs.py`：审计长epoch边界和事件间隔；
2. 冻结`configs/features.yaml`中的TFR epoch、baseline、频率和周期规则；
3. `scripts/07_run_time_frequency.py`：逐人计算Morlet、ITC和群体图；
4. `scripts/08_extract_trial_features.py`：生成ERP、频带功率和DWT特征；
5. 生成`feature_dictionary.csv`和特征QC报告；
6. 添加数值、维度、单位、trial对齐和防泄漏测试；
7. 汇总Phase 6结果和限制，再进入Phase 7。

编号从06开始，是因为Phase 5的正式入口已经是`05_run_erp_analysis.py`。

## 5. 计算和内存策略

- 先用`sub-001、sub-006、sub-025`做小规模试运行，覆盖正常、低Target计数和曾重点复核的情况；
- 一次只处理一名受试者；
- TFR内部并行度先设为1，确认内存后再提高；
- 只保存条件平均TFR、频带曲线和紧凑trial特征；
- 大型中间数组使用float32；
- 每完成一名受试者立即落盘并释放内存；
- 全体运行前先估算单人峰值内存和耗时。

当前Conda环境已经包含MNE、NumPy、SciPy、PyWavelets、pandas和scikit-learn，不需要新建环境。若保持CSV/Gzip输出，也不需要增加PyArrow等依赖。

## 6. Phase 6验收标准

- [ ] 长epoch边界、基线和事件间隔审计完成；
- [ ] 统一的TFR参数在查看条件显著性前冻结；
- [ ] 39人逐人计算，不一次加载完整数据；
- [ ] 输出Target、Standard、条件差和ITC图；
- [ ] 输出ERP、Morlet频带和DWT单trial特征；
- [ ] 特征表主键唯一并与trial审计记录一一对应；
- [ ] 每个特征具有单位和计算规则；
- [ ] PCA未在全体数据上预先拟合；
- [ ] 自动化测试覆盖维度、边界、频带、能量和防泄漏；
- [ ] 生成完整结果文档和一条可复现命令。

## 7. 下一步应该先做什么

Phase 6的第一项实施任务应是编写`06_audit_tfr_epochs.py`。只有确认长epoch的可用率、相邻刺激间隔和内存规模后，才能冻结2 Hz下限、baseline和epoch长度。现在不应直接运行Morlet或DWT全体分析。

