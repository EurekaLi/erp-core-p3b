# 数据验证程序：使用方法与概念说明

## 1. 为什么先验证、后滤波

滤波只能改变频率成分，不能补回缺失的受试者、修复错误的事件码或猜出信号单位。如果把错误事件当成 Target，后续即使得到平滑漂亮的 ERP 曲线，结论仍然是错误的。因此本项目的第一段正式代码只读取目录、JSON、TSV 和少量可选信号，不修改原始数据。

入口程序是 `scripts/01_validate_dataset.py`，核心实现位于 `src/erp_core_p3b/validation.py`。

## 2. 当前数据实际状态

当前 BIDS 根目录为：

```text
data/raw/erp-core-p3
```

当前目录已经包含完整的 40 人 P3 数据：

- `participants.tsv` 列出 `sub-001` 到 `sub-040` 共 40 人；
- 磁盘上实际存在 `sub-001` 到 `sub-040` 共 40 个目录；
- 每人均有一组 P3 `.set/.fdt` 文件及相应 JSON/TSV 元数据；
- 当前 NEMAR 发布版本采用无 session 的 `sub-XXX/eeg/` BIDS 布局；
- P3 目录没有个体实测的 `electrodes.tsv` 和 `coordsystem.json`。

验证程序仍将“参与者清单”“实际受试者目录”和“可发现的 P3 记录”分开计数，因为三者相等才说明文件层面的完整性成立。

## 3. 基本运行命令

先进入项目并安装可编辑包：

```powershell
Set-Location -LiteralPath 'D:\AILearning\project_for_offer\erp-core-p3b'
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python -m pip install -e .
```

运行快速元数据验证：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/01_validate_dataset.py
```

显式写出全部参数时等价于：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/01_validate_dataset.py `
  --bids-root data/raw/erp-core-p3 `
  --task P3 `
  --expected-subjects 40 `
  --expected-sfreq 1024 `
  --expected-eeg-channels 30 `
  --expected-eog-channels 3 `
  --output-dir results/data_validation
```

快速验证不加载约 49 MB/人的 `.fdt` 信号，只检查文件配对和元数据，因此适合每次下载后立即运行。

## 4. 深度信号可读性检查

如果要进一步确认 `.set` 与 `.fdt` 能被 MNE 配对读取：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/01_validate_dataset.py --check-signals
```

程序以 `preload=False` 打开每份记录，再读取开头和结尾各约 1 秒。这样既能发现外部 `.fdt` 文件缺失或截断，又不会把全部数据同时放入内存。初次试运行可限制为 2 人：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b python scripts/01_validate_dataset.py --check-signals --signal-limit 2
```

## 5. 程序具体检查什么

### 5.1 BIDS 根目录

程序检查：

- `dataset_description.json`：数据集身份信息；
- `participants.tsv`：参与者清单；
- `task-P3_events.json`：事件码含义；
- 实际 `sub-*` 文件夹和 `eeg` 记录；同时兼容其他版本的 `ses-*/eeg` 布局。

这里将“清单人数”“实际受试者目录数”“P3 记录数”分别报告。三个数字相同，才可能表示数据完整；相同也只是必要条件，不是充分条件。

### 5.2 `.set` 与 `.fdt`

本数据由 EEGLAB 导出：

- `.set` 主要保存头信息、通道和数据索引；
- `.fdt` 保存连续浮点信号；
- 两者必须配套，只有 `.set` 通常无法完成分析。

程序同时检查 `_eeg.json`、`_channels.tsv` 和 `_events.tsv` 是否存在且非空。

### 5.3 通道

预期每人有 30 个 EEG 通道和 3 个 EOG 通道。

- EEG 是头皮电位，是后续 ERP 的主体；
- EOG 是眼电，用于发现眨眼和水平/垂直眼动伪迹；
- EOG 不是需要研究的脑信号，但对 ICA 和质量控制非常重要。

### 5.4 采样率

预期采样率是 1024 Hz，意思是每个通道每秒记录 1024 个数值。因此：

```text
1 个采样点 ≈ 1 / 1024 秒 ≈ 0.9766 毫秒
```

事件表同时保存 `onset`（秒）和 `sample`（采样点）。程序检查：

```text
sample ≈ onset × 1024
```

允许约 1 个采样点的舍入差异。

### 5.5 信号单位

`channels.tsv` 把 EEG/EOG 的原始单位写为 `microV`，即微伏。1 微伏等于百万分之一伏：

```text
1 microV = 1e-6 V
```

MNE 在内存中通常统一使用伏特，因此看到 `5e-6 V` 就是 `5 microV`，不是信号突然变得很小。单位混淆会让阈值相差一百万倍，所以必须在滤波和伪迹阈值设置前确认。

### 5.6 P3 事件码

ERP CORE P3 刺激码使用两位数字：

- 十位：当前 block 中哪个字母是目标；
- 个位：当前 trial 实际呈现哪个字母；
- 两位相同（11、22、33、44、55）：Target；
- 两位不同且都在 1–5（例如 12、25、43）：Standard；
- 201：正确反应；
- 202：错误反应。

程序按受试者统计 Target、Standard、正确反应、错误反应和未知事件。未知事件不会被擅自归类。

### 5.7 电极坐标

当前文件提供标准 10-20 通道名称，但没有每名受试者的实测三维坐标。后续 MNE 可以按通道名套用标准 montage，用于画近似头皮图和插值；报告必须写明它是模板坐标，不能声称是个体电极数字化坐标。

## 6. 输出文件

默认输出到 `results/data_validation/`：

| 文件 | 内容 | 用途 |
|---|---|---|
| `validation_summary.json` | 状态、人数、问题数量、总事件数和关键说明 | 程序读取或快速总览 |
| `subject_task_inventory.csv` | 每人的文件大小、采样率、通道数、单位、参考与时长 | 检查跨受试者一致性 |
| `event_counts.csv` | 每人的 Target/Standard/反应事件计数 | 发现事件缺失或比例异常 |
| `issues.csv` | 每条 ERROR/WARNING、对应文件与通俗解释 | 修复下载和记录限制 |

`PASS` 表示未发现问题；`PASS_WITH_WARNINGS` 表示可以继续调查但要记录限制；`FAIL` 表示存在影响完整性或分析正确性的错误。默认模式即使为 `FAIL` 也会正常写完报告并返回退出码 0，方便初学者查看结果。自动化流水线可添加 `--strict`，让 ERROR 对应退出码 1。

## 7. 运行后的下一步

40 人数据通过结构验证和信号可读性检查后，下一步进入原始信号 QC：查看原始波形、功率谱密度、通道方差、平直通道、异常高频功率与事件计数。仍然先观察数据并记录问题，再冻结滤波、坏道和伪迹阈值。
