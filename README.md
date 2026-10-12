# ERP CORE P3b：可复现 EEG/ERP 信号处理项目

本项目以 ERP CORE 的 P3/P3b 主动视觉 oddball 数据为对象，建立一条从原始 EEG 到可复现统计结论和单试次特征的完整流水线。文档面向尚未系统学习信号与系统的读者，会说明采样率、单位、事件、滤波、时频分析等概念为什么需要检查，以及它们对结果有什么影响。

项目的重点不是研究计算机视觉，而是研究多通道神经电生理信号。这里的“视觉 oddball”只描述刺激呈现方式：受试者观看频繁出现的标准刺激和稀少出现的目标刺激。真正分析的对象是刺激发生后各脑电通道随时间产生的电位变化。

## 1. 研究问题

项目回答三个递进问题：

1. Target 与 Standard 条件是否在顶叶区域产生稳定的 P3b 差异？
2. 这种差异在时间、空间和频率维度上如何表现？
3. 经典数字信号处理与特征提取能否得到可解释、可复现的单试次分类结果？

经典 ERP 分析用于建立可信的生理学基线；传统机器学习用于验证单试次信息是否可解码。本阶段不引入图或超图方法。

## 3. 数据集

ERP CORE 包含 40 名健康成年受试者、6 个实验范式和 7 种经典 ERP 成分。本项目只使用 P3 范式的全部 40 名受试者。

关键采集信息：

- 30 个 EEG 通道与 3 个 EOG 通道；
- 原始采样率 1024 Hz；
- 主动视觉 oddball 任务；
- 两个主要类别：Target 与 Standard；
- 原始数据采用 BIDS 组织，便于 MNE-BIDS 读取。

下载方式和数据目录约定见 [docs/01_dataset_and_download.md](docs/01_dataset_and_download.md)。

数据下载后，先运行只读验证，不要直接滤波：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/01_validate_dataset.py
```

验证程序、输出字段和当前数据状态见 [docs/02_data_validation.md](docs/02_data_validation.md)。

完成结构验证后运行原始信号与行为QC：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/02_run_raw_qc.py
```

指标、阈值来源和40人结果见 [docs/03_raw_qc.md](docs/03_raw_qc.md)。

完成坏道复核后运行正式预处理：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/03_run_preprocessing.py --stage final
```

滤波分支、ICA、插值、分段、trial剔除规则及40人结果见
[docs/04_preprocessing.md](docs/04_preprocessing.md)。

四名高剔除率受试者完成个案复核后，主分析应使用裁决版数据：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/03_run_preprocessing.py --stage adjudicated
```

复核仪表板的读法、修正前后对照和逐人决定见
[docs/05_four_subject_review.md](docs/05_four_subject_review.md)。

裁决版数据完成后进入经典 P3b 基线分析。主分析固定为 Pz、300--600 ms
平均幅值，统计单位为受试者；完整的执行顺序、图形读法和验收标准见
[docs/06_p3b_analysis_plan.md](docs/06_p3b_analysis_plan.md)。

运行完整的 Phase 5 分析：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/05_run_erp_analysis.py
```

正式结果、图形解读、敏感性分析和限制见
[docs/07_p3b_results.md](docs/07_p3b_results.md)。39人主分析得到
Target - Standard = 3.486 µV，95% bootstrap CI [2.738, 4.323]，
`t(38)=8.476`，`p=2.732×10⁻¹⁰`，Cohen's `dz=1.357`。

Phase 6不能直接在现有1秒ERP epochs上计算2 Hz Morlet；低频小波长度和基线均不足。
长epoch审计、Morlet/ITC、小波特征、trial特征表及PCA防泄漏方案见
[docs/08_phase6_time_frequency_plan.md](docs/08_phase6_time_frequency_plan.md)。

## 4. 分析路线

项目分为八个阶段：

1. 环境、目录和研究协议固定；
2. 数据下载、BIDS 检查和事件语义确认；
3. 原始数据质量控制；
4. 滤波、重参考、坏道插值、ICA 与分段；
5. P3b 幅值、潜伏期、头皮分布和统计分析；
6. Morlet 时频、离散小波、频带功率与 PCA；
7. 受试者独立的传统机器学习基线；
8. 结果复现、误差分析、报告与简历材料整理。

详细里程碑见 [docs/00_project_plan.md](docs/00_project_plan.md)。

## 5. 环境创建

在 PowerShell 中进入本项目：

```powershell
Set-Location -LiteralPath 'D:\AILearning\project_for_offer\erp-core-p3b'
```

本机常规 `conda.exe` 启动器当前不可执行，但 Anaconda 根目录中的 `_conda.exe` 可正常工作。可以直接运行：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' env create -f environment.yml
```

安装本项目为可编辑包：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python -m pip install -e .
```

验证关键依赖：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python -c "import mne, mne_bids, numpy, sklearn, pywt; print('MNE', mne.__version__)"
```

注册 Jupyter 内核：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python -m ipykernel install --user --name erp-core-p3b --display-name "Python (ERP CORE P3b)"
```

启动 JupyterLab：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b jupyter lab
```

如果日后修复了 Conda shell 初始化，也可以使用常规命令：

```powershell
conda env create -f environment.yml
conda activate erp-core-p3b
python -m pip install -e .
```

## 6. 仓库结构

```text
erp-core-p3b/
├── configs/                 # 可审计的预处理、特征和模型参数
├── data/
│   ├── raw/                 # 下载后保持不变的 BIDS 原始数据
│   ├── interim/             # ICA、坏道记录和临时结果
│   └── processed/           # 可由原始数据重建的 epochs/特征
├── docs/                    # 研究协议、数据说明和决策记录
├── notebooks/               # 探索性分析，不承载正式流水线
├── reports/figures/         # 最终筛选进入报告的图片
├── results/                 # 指标、统计表和模型输出
├── scripts/                 # 命令行入口和批处理脚本
├── src/erp_core_p3b/        # 可测试、可复用的正式实现
├── tests/                   # 事件映射、单位和防泄漏测试
├── environment.yml
└── pyproject.toml
```

## 7. 复现原则

- 原始数据只读，不在原文件上修改；
- 所有处理参数保存在 YAML 中；
- ICA 排除成分、坏道和剔除 trial 必须记录原因；
- PCA、标准化和特征选择只能在训练折上拟合；
- 正式泛化结果按受试者划分训练、验证和测试；
- Notebook 中确认有效的步骤必须迁移到 `src/`；
- 所有最终结论同时报告效应量、置信区间或不确定性信息。

## 8. 当前状态

- [x] 建立项目目录；
- [x] 建立环境与配置骨架；
- [x] 编写数据下载说明；
- [ ] 创建 Conda 环境；
- [x] 下载并确认 P3 全体 40 名受试者数据；
- [x] 编写 BIDS 结构与事件语义验证程序；
- [x] 生成40人的原始信号与行为QC报告；
- [x] 冻结第一版坏道候选阈值；
- [x] 人工复核并登记 sub-035/PO3 与 sub-039/Oz；
- [x] 建立并运行40人的滤波、重参考、坏道插值、ICA与分段流水线；
- [x] 冻结200 µV epoch峰峰值阈值并生成逐trial审计表；
- [x] 完成4名高剔除率受试者复核：保留3人，主分析排除sub-030；
- [x] 生成39人的裁决版数据，共保留7207个epoch。
- [x] 冻结阶段5的主通道、时间窗、统计方法和敏感性分析方案；
- [x] 实现并运行39人的经典P3b分析流水线；
- [x] 完成群体波形、个体图、头皮图、2000次时空置换及敏感性分析；
- [x] Phase 5完成，下一阶段为时频与特征工程。

