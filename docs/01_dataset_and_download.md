# ERP CORE P3 数据与下载说明

## 1. 本项目应该下载哪一份数据

本项目只研究 P3/P3b，因此“全部数据”指 P3 范式全部 40 名受试者，而不是 ERP CORE 的所有六种任务。

优先下载 BIDS 版 P3 数据：

- 体积约 1.8 GB；
- 下载链接：<https://osf.io/download/3zk6n/>；
- 建议保存为 `data/raw/ERP_CORE_P3_BIDS.zip`；
- 解压后把 BIDS 根目录放在 `data/raw/erp-core-p3/`。

如果希望额外保存 ERP CORE 所有实验任务，可下载 NEMAR 固定版本完整归档：

- ZIP 约 14.1 GB，解压后约 17.5 GB；
- 固定版本下载：<https://data.nemar.org/nm000132/v1.0.0.zip>；
- 数据集主页：<https://www.nemar.org/dataset/nm000132>。

完整归档包含 MMN、N170、N2pc、N400、P3 和 flankers。它们不属于当前 P3 项目的必需输入。

## 2. 浏览器下载

网络条件良好时，可以直接在浏览器打开 P3 下载链接。下载完成后不要立即删除 ZIP；先校验文件能否正常列出，再解压。

PowerShell 中可检查 ZIP：

```powershell
Get-Item -LiteralPath '.\data\raw\ERP_CORE_P3_BIDS.zip' |
    Select-Object FullName, Length, LastWriteTime
```

解压示例：

```powershell
Expand-Archive `
    -LiteralPath '.\data\raw\ERP_CORE_P3_BIDS.zip' `
    -DestinationPath '.\data\raw\erp-core-p3' `
    -Force
```

`-Force` 会覆盖同名文件。只有在确认目标目录确实属于本项目且需要重新解压时才使用。

## 3. NEMAR 下载器

创建环境后，可以先查看下载器帮助，确认当前版本参数：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b nemar-py --help
```

下载完整 NEMAR 数据集的基本形式为：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b `
    nemar-py download nm000132 -o '.\data\raw\nm000132'
```

由于下载器参数可能随版本更新，任务和受试者过滤命令应以本地 `--help` 为准，不在脚本中假定尚未验证的参数拼写。

## 4. 下载后必须检查

在任何滤波前记录以下信息：

1. `dataset_description.json` 是否存在；
2. `participants.tsv` 是否包含 40 名受试者；
3. 每个受试者是否存在 P3 EEG、events、channels 和 sidecar 文件；
4. EEG 与 EOG 通道数量；
5. 采样率、单位和线噪频率；
6. Target、Standard、按键和正确性事件的语义；
7. 是否存在缺失受试者或损坏文件。

不能只根据文件名猜测事件含义。必须以任务级 `events.json`、受试者级 `events.tsv` 和原始论文为准。

## 5. Git 与数据许可

原始数据和所有可重建的大文件由 `.gitignore` 排除。GitHub 中只保存：

- 数据来源与引用；
- 下载方法；
- 处理脚本；
- 参数配置；
- 汇总指标；
- 不包含可识别信息的最终图表。

公开仓库中不要重新上传完整 EEG 数据。使用数据时应引用 ERP CORE 原始论文及数据集 DOI。

