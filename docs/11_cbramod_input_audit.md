# 真实 EEG 的 CBraMod 输入与环境适配审计

## 1. 本阶段做什么、不做什么

本阶段新增 `scripts/08_audit_cbramod_inputs.py`，由使用者在
`erp-core-p3b-dl` 环境执行。开发时只用合成信号、临时FIF和CPU小网络验证逻辑，
不代替使用者运行真实三人GPU审计，也不生成任何正式分类成绩。

目标是确认以下链路的工程正确性：

```text
裁决版FIF → 保留trial身份/30通道/物理单位检查
           → 刺激零点对齐的256→200Hz重采样
           → 幅值除以100µV → (trial,30,1,200) float32
           → 严格加载预训练权重 → 冻结/解冻梯度检查
```

程序不重新滤波、不重新拟合ICA、不重新参考、不重新做电压baseline，
也不剔除任何trial。预训练权重只读取；临时网络会做两种工程更新但不落盘，
临时损失不用于模型比较、早停或超参数选择。

这一阶段不冻结正式输入，不导出全体训练数据、不生成受试者划分。
这些属于人工查看本轮图形之后的独立实施任务。

## 2. 当前可复用的资源

默认读取 `data/processed/epochs/adjudicated`，先对全体文件头和元数据做清单审计：

- 39人，排除sub-030；
- 7207个保留trial，Target 1286、Standard 5921；
- 每人30 EEG、3 EOG；
- 256Hz、257点、0.1–30Hz，既有电压baseline[-0.2,0]秒；
- 与Phase 5 `analysis_manifest.json` 的逐人计数一致；
- EEG通道顺序在受试者之间一致，元数据与事件逐行对应。

随后只对sub-001、sub-006、sub-025载入真实波形、生成图形并测试模型。
三人来自已有pilot约定，不按模型效果选择。sub-006仅5个Target，必须正确处理，
不能为了得到好看的类别平衡而悄悄排除该受试者。

Standard编码为0、Target编码为1。`event_code`、反应时间、正确性和受试者编号
只用于来源追溯，不进入模型张量。trial_id采用`sub-001:trial-0001`形式；
trial_index来自原保留元数据，不能用删掉trial后的行号冒充原trial编号。

## 3. 为什么不是直接把257点缩放成200点

### 3.1 实际采样网格

原始短epoch的实际网格是：

```text
起点 = -51/256 = -0.19921875秒
终点 = 205/256 = 0.80078125秒
刺激0的数组索引 = 51
总点数 = 257
```

0.2秒并不是256Hz的整数个采样间隔。保持旧起点直接改成5毫秒间隔，
刺激0会落在两个采样点之间；使用数组索引随意切片则可能掩盖这个偏移。

本轮候选网格明确设为：

```text
200Hz，200点，[-0.195, 0.800]秒
刺激0的数组索引 = 39
点间距 = 0.005秒
首末点之差 = 0.995秒
```

200点在200Hz构成模型所需的一秒patch；首末采样时刻相差0.995秒不是点数错误。
这些目标点都处于原短epoch的时间范围内，不把缺失的-0.200秒伪造为真实观测。

### 3.2 重采样算法

采用SciPy `resample_poly`，比例为25/32。它包含抗混叠低通，不是简单每隔几个点取一个。

左侧先反射13个旧采样点，使刺激0在补齐数组中的索引变为64。
64可被分母32整除，重采样后的刺激0精确对应索引50；随后根据真实时间轴取200个目标点，
零点落在候选数组索引39。每次记录补齐点数、比例、原/目标端点和零点索引。

参考：[SciPy多相重采样文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)。

反射边界只是重采样的数值假设，不是真实的更早/更晚EEG，也不证明边缘完全无失真。
必须看前后波形和ERP叠加图。若边缘效应确实影响研究问题，再单独建立更充分的连续信号分支；
不能绕过长窗QC或把四秒补零结果解释成四秒脑电。

## 4. 信号单位和预训练分布

MNE读取结果使用物理伏特，程序检查EEG通道的FIFF单位，不重复应用文件校准因子。
作者预训练采用100µV为尺度，本轮变换为：

```text
model_input = EEG_in_volts / 0.0001
例如10µV → 0.00001V → 模型数值0.1
```

转换后检验float32往返误差，容差0.001µV只服务数值精度，不是伪迹阈值。
不按trial重新z-score，不再次减baseline，避免抹掉或改变ERP幅值信息。

现有0.1–30Hz和ERP电压baseline与作者预训练分布不同，程序会记录这一限制。
30Hz以上的信息已经被滤掉，不能用一次重采样或“改成75Hz低通”恢复。
如果以后比较不同滤波分支，需重新从连续数据构建、冻结QC、维护相同trial身份，并作为独立实验。

## 5. 模型资源与安全检查

本轮配置固定已下载资源：

- 作者代码commit：`b9e961003214326972c567eff390e75b0287e32a`；
- 权重revision：`500543c7e30bda1b22bfd51a49301b238dee21fd`；
- 权重SHA-256：`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。

参考：[作者代码](https://github.com/wjq-learning/CBraMod)、
[作者权重](https://huggingface.co/weighting666/CBraMod)、
[论文](https://arxiv.org/html/2412.07236v1)。

不自动联网、不自动拉取新代码，也不自动安装依赖。检查commit、tracked修改、源文件hash、
权重hash以及存在时的HF下载元数据。没有HF元数据时保留null，但权重hash仍必须一致。
原代码MIT许可与权重Hub的Apache-2.0元信息分别登记，不能假定二者许可相同。

先以`weights_only=True`完整且严格加载原网络，再移除重建头并添加临时二分类头。
不使用`strict=False`掩盖缺失/多余参数，不默认放开任意对象反序列化。

本机可能出现Git所有权提示。程序只给本次已明确的作者仓库git读取传入
`-c safe.directory=完整路径`，不修改全局Git配置，也不信任所有目录。

## 6. 实际运行命令

在已激活的新环境中进入项目根目录：

```powershell
Set-Location -LiteralPath 'D:\AILearning\project_for_offer\erp-core-p3b'
python -X utf8 scripts/08_audit_cbramod_inputs.py --help
```

### 6.1 先做信号审计，不使用GPU/模型

```powershell
python -X utf8 scripts/08_audit_cbramod_inputs.py `
    --signal-only `
    --run-name pilot-signal
```

这仍会核对39人的文件头/元数据，但仅载入三人的信号。
不导入torch、不读取模型权重；不能据此说GPU或预训练权重通过。

### 6.2 再做真实EEG的GPU工程检查

```powershell
python -X utf8 scripts/08_audit_cbramod_inputs.py `
    --device cuda `
    --run-name pilot-gpu `
    --save-inputs
```

默认每人取至多2个Standard、2个Target检查：

1. 冻结编码器，确认编码器无梯度，分类头有有限梯度且权重更新；
2. 解冻编码器，确认存在非零编码器梯度、所有可训练参数梯度有限，分类头更新；
3. 记录实际输入/特征/logits维度、工程损失、耗时和CUDA显存。

这不是一个训练epoch，也不保存更新后的参数。三人测试可能依次更新同一个临时编码器，
但结果不是预训练嵌入缓存或可用模型。正式训练必须重新加载固定初始权重。

默认不会静默回退CPU。只在明确需要CPU诊断时使用`--device cpu`，结果会标记为CPU，
`gpu_probe_complete=false`。RSS只记录步骤末的快照，不冒称进程内存峰值。

### 6.3 重跑和失败处理

指定名称已存在时拒绝覆盖，换一个名字：

```powershell
python -X utf8 scripts/08_audit_cbramod_inputs.py --device cuda --run-name pilot-gpu-02
```

也可不传`--run-name`，程序按UTC时间生成独立目录。
任何异常都会非零退出；已建立结果目录时，run_summary.json保存failed和错误信息。
不要删除原结果来掩盖失败，也不要为通过检查随意改变单位、标签或权重。

## 7. 输出和看图方法

结果根目录为`results/cbramod_audit/<run-name>/`：

|文件|含义|
|---|---|
|audit_config.json|此次配置快照|
|input_inventory.json|39人来源、hash、类别计数、通道顺序|
|run_summary.json|逐人检查、资源版本、设备和工程状态|
|audit_report.md|可阅读总结及图形入口|
|sub-XXX_trial_audit.csv|身份、标签、原/适配PTP和复核提示|
|sub-XXX_channel_audit.csv|通道索引、原/适配RMS、PTP中位数、完全恒定trial数|
|figures/sub-XXX_erp_overlay.png|Pz两条件适配前后叠加|
|figures/sub-XXX_trials_and_amplitude.png|示例trial、最大PTP trial、PTP分布|
|figures/sub-XXX_channel_heatmap.png|最大PTP trial的30通道矩阵|
|sub-XXX_audit_inputs.npz|仅在--save-inputs下生成的三人审计缓存|

看图顺序：先Pz条件ERP，再单trial，再通道矩阵。
两种颜色/线型应该在相同时间位置呈现同一主要形状，尤其关注300–600ms。
不要用两条平均曲线“看起来像”代替零点索引和标签逐行检查。
PTP变化与幅值超过模型数值1只提示查看，不自动判坏或删样本。
完全恒定trial计数也只用于定位输入异常，不以临时规则改变保留名单。
通道矩阵不是头皮地形，不能据它解释源位置。

NPZ中原生信号单位是V，模型输入是除以100µV的无量纲数值。
trial_ids和通道名保存为字符串数组，可用`allow_pickle=False`读取。

## 8. 怎么判断结果，以及如何接入下一步

技术通过需要事件身份、单位、网格、有限值、资源校验和所选工程测试全部通过。
GPU通过额外需要`gpu_probe_complete=true`；signal-only或CPU测试不能替代它。

即使技术通过，仍保留：

```text
visual_review_status = pending
eligible_for_formal_training = false
cross_patch_temporal_attention_degenerate = true
```

第三项不是计算错误：一个时间patch的时间注意力不能建模多个patch之间的关系。
若采用这一候选，要把短窗迁移限制写进正式实验；若需改变patch方案或时间窗，必须另行审计。

之后依次进行人工图形复核、候选输入决策、39人数据导出与固定受试者划分、
一折训练闭环、正式分组评估。传统对照共用trial身份和受试者划分，不必使用200Hz适配缓存。

上游ICA在每名受试者全记录上无监督拟合，这是一种离线清洗假设。
不能把最终结果写成严格在线推理、未知受试者实时校准或临床诊断。
