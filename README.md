# ERP CORE P3b：可复现EEG信号处理与跨受试者解码

从40名受试者原始EEG出发，完成数据验证、行为事件匹配、QC、滤波、重参考、坏道插值、ICA与分段；
对裁决后39人的P3b进行统计分析，再比较紧凑特征Logistic Regression、EEGNet及CBraMod冻结/微调。
最终在同一7207个trial和固定受试者五折上评价6组模型，提供配对置信区间和逐受试者误差审计。

本项目分析多通道电生理信号。“视觉oddball”是刺激范式，输入是EEG波形。
方法文档面向尚未系统学习信号与系统的读者，解释单位、时间轴、滤波、特征和评价设计。

## 主要结果

经典P3b主分析固定为Pz、300–600ms。Target−Standard平均差为 **3.486µV**，
95%受试者bootstrap CI为[2.738,4.323]µV，t(38)=8.476，p=2.732×10⁻¹⁰，Cohen's dz=1.357。

![Pz群体波形及差异波](reports/figures/erp_analysis/group_waveforms.png)

分类表采用39人等权macro指标，Standard=0、Target=1。每名受试者在每个模型中只贡献一次outer test结果。

|模型|输入|macro BA|macro ROC-AUC|macro AP|
|---|---|---:|---:|---:|
|LR ERP|12列ERP特征|0.6557|0.7310|0.4249|
|LR ERP+PSD|24列ERP与频带功率|0.6566|0.7373|0.4242|
|LR ERP+PSD+DWT|40列紧凑信号特征|0.6540|0.7353|0.4213|
|EEGNet|30通道原生256Hz短窗|0.6917|0.7744|0.4910|
|CBraMod linear|预训练编码器冻结＋线性头|0.6547|0.7270|0.4129|
|CBraMod finetune|预训练编码器微调＋相同线性头|0.6902|0.7715|0.4832|

![六模型宏平均指标及条件置信区间](reports/figures/final_report/model_comparison.png)

EEGNet−ERP LR的配对BA差为+3.61个百分点，条件95% CI为[1.86,5.37]个百分点。
CBraMod微调−EEGNet为−0.16个百分点，区间[−1.83,1.43]跨0；当前实验不支持稳定排名。
微调相对冻结版改善BA约3.55个百分点。加入功率或DWT，在当前固定LR设置下没有明确增量。

完整公开摘要、配对结果、错误计数和资源来源见[结果材料](reports/release/README.md)。
指标以[聚合表](reports/release/tables/model_overview.csv)为准，区间为固定划分/拟合模型条件下的受试者bootstrap，
模型对照属于描述性分析，没有多重比较校正或跨种子重新训练不确定性。

## 实现内容

- 数据验证：BIDS路径、40人文件、30 EEG/3 EOG、原始1024Hz、物理单位、事件码和刺激—反应配对。
- 原始QC：波形、PSD、方差、平直/噪声候选及行为计数；登记sub-035/PO3、sub-039/Oz坏道。
- 预处理：ERP 0.1–30Hz分支、稳定拟合ICA的临时分支、平均参考、坏道插值、ICA、baseline和200µV epoch峰峰值QC。
- 人工裁决：复核高剔除率个案，主分析排除sub-030，39人保留1286 Target、5921 Standard。
- ERP统计：冻结主通道/窗口，条件平均、配对检验、效应量、受试者CI、敏感性分析与探索性时空置换。
- 紧凑特征：ERP12列、功率12列、DWT16列；三组LR增量对照，训练折内StandardScaler。
- 深度学习：EEGNet PyTorch架构适配、CBraMod严格权重加载、刺激零点锚定重采样、冻结/微调对照与验证早停。
- 评价：固定受试者五折、训练标签类别权重、完整OOF身份校验、39份个体报告和逐trial错误定位。

## 评价边界

同一受试者不跨同一折的train/validation/test。深度学习smoke仅检查训练/验证，正式五折才预测test；
每折重新初始化模型或加载原预训练权重，测试结果不参与早停。
阈值固定0.5，使用BA、ROC-AUC、AP和Target召回/F1，而非仅用类别不平衡下的accuracy。

CBraMod当前为200Hz、一个200点时间patch，重采样有反射边界支持，输入保留ERP滤波和baseline。
该实验不能验证多patch时间建模优势；EEGNet和CBraMod网格不同，对照包含适配因素。
预处理采用每名受试者整段记录的无监督ICA/QC，评价范围是离线受试者分组场景。
只有一个训练seed设置、一个健康受试者数据集；没有外部队列或临床诊断验证。
完整长窗Morlet/ITC仍有候选待复核，是可选扩展；已完成的短窗DWT与它不同。

## 复现入口

新克隆请先读[完整复现路径](REPRODUCING.md)，来源说明见[NOTICE](NOTICE.md)。
只读发布检查无需安装科学计算依赖：

```powershell
python -X utf8 scripts/15_check_release.py
```

原始P3 BIDS数据约1.8GB：[数据下载及目录说明](docs/01_dataset_and_download.md)。
数据源为[ERP CORE](https://erpinfo.org/erp-core)，数据集许可记录为CC-BY-4.0。
EEG/FIF、权重、缓存和完整预测文件不随仓库发布。

Python 3.11信号处理环境：

```powershell
conda env create -f environment.yml
conda activate erp-core-p3b
python -m pip install -e .
python -X utf8 scripts/01_validate_dataset.py --help
```

`environment.yml`提供信号处理依赖，PyTorch GPU训练环境独立准备。
本次DL环境的实测版本见[environment_record.json](reports/release/environment_record.json)，
它是运行记录，不是跨平台完整依赖锁文件。GPU配置、预训练代码/权重来源和输入协议见
[08审计说明](docs/11_cbramod_input_audit.md)及[12–13训练说明](docs/14_deep_learning_export_and_training.md)。

已有裁决版FIF和08审计后，在DL环境依次执行：

```powershell
python -X utf8 scripts/09_build_classification_manifest.py
python -X utf8 scripts/10_extract_compact_features.py --stage full
python -X utf8 scripts/11_train_logistic_baseline.py --mode full --run-name lr-full
python -X utf8 scripts/12_export_deep_learning_dataset.py --stage full --accept-short-window-limitations
python -X utf8 scripts/13_train_deep_learning.py --mode full --models eegnet cbramod_linear cbramod_finetune --device cuda --run-name three-models-full-v1
python -X utf8 scripts/14_build_final_report.py
```

首次训练前应按详细说明运行pilot、看图和smoke。接受参数需建立在已复核输入和理解单patch限制的基础上。
这些命令面向新输出目录；本地已完成者无需重跑。现有输出拒绝覆盖，重跑使用新版本/运行名。
早期预处理及人工裁决按下方方法文档执行，并非跳过01–05直接得到分类数据。

## 文档导航

|主题|文档|
|---|---|
|数据、验证、QC|[下载](docs/01_dataset_and_download.md) · [验证](docs/02_data_validation.md) · [原始QC](docs/03_raw_qc.md)|
|预处理与人工裁决|[滤波/ICA/分段](docs/04_preprocessing.md) · [四人复核](docs/05_four_subject_review.md)|
|经典P3b|[冻结方案](docs/06_p3b_analysis_plan.md) · [正式结果](docs/07_p3b_results.md)|
|长窗可选扩展|[计划](docs/08_phase6_time_frequency_plan.md) · [审计](docs/09_tfr_epoch_audit.md) · [pilot](docs/10_tfr_long_epochs_pilot.md)|
|紧凑特征与LR|[支线计划](docs/12_signal_processing_parallel_plan.md) · [09–11方法和命令](docs/13_classification_features_and_lr.md)|
|CBraMod与训练|[真实输入审计](docs/11_cbramod_input_audit.md) · [12–13导出和训练](docs/14_deep_learning_export_and_training.md)|
|结果与错误解释|[详细解读](docs/15_final_results_and_error_analysis.md) · [公开结果包](reports/release/README.md)|
|GitHub材料管理|[本地核验、提交命令和发布范围](docs/16_github_materials.md)|

详细文档保留阶段当时的操作背景；当前完成状态以本页与[项目计划](docs/00_project_plan.md)为准。
逐受试者/逐trial完整报告由14在本地生成，公开包保留聚合结果与核心图。

## 仓库结构与核验

```text
configs/                 # 冻结参数和人工决策
docs/                    # 方法、结果、执行与解释
scripts/                 # 01–14分析入口，15公开材料核验
src/erp_core_p3b/        # 正式可复用实现
tests/                   # 单位、身份、分组、冻结/梯度及统计契约
reports/release/         # 公开聚合结果和来源记录
reports/figures/         # 选入公开报告的图表
data/、results/          # 本地原始数据、缓存、模型与运行日志
```

核验命令在具有全部测试依赖的DL环境中运行：

```powershell
python -m ruff check src scripts tests
python -m pytest -q
```

原始数据只读，所有删除trial、ICA排除和坏道插值保留来源；
输出保存配置、版本与hash，公共源码与大数据产物分开管理。

## 来源与完成状态

数据论文：[Kappenman et al., ERP CORE](https://doi.org/10.1016/j.neuroimage.2020.117465)。
模型架构：[EEGNet作者实现](https://github.com/vlawhern/arl-eegmodels)、
[CBraMod作者代码](https://github.com/wjq-learning/CBraMod)与[权重](https://huggingface.co/weighting666/CBraMod)。
项目内EEGNet是独立PyTorch适配；CBraMod代码和权重分别记录来源，未重新分发。

- [x] 原始40人数据验证、QC、预处理及人工裁决。
- [x] 39人P3b统计、40列紧凑特征、三组LR。
- [x] 三神经网络smoke及正式受试者五折。
- [x] 六模型对照、39份个体报告、逐trial错误定位、公开结果材料。
- [ ] 可选扩展：多种子、外部验证、完整长窗TFR。

