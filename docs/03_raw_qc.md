# 原始信号质量控制：方法、阈值与本次结果

## 1. 这一阶段解决什么问题

数据结构验证只能说明文件存在、格式可读和事件码能够解释。原始质量控制进一步回答：

1. 每个刺激能否和行为反应正确配对？
2. 哪些通道可能平直、漂移、噪声过大或与其他通道明显不一致？
3. 某名受试者的问题是否严重到不能靠少量坏道插值解决？
4. 后续滤波、插值和 ICA 应使用什么统一规则？

本阶段不修改 `.set/.fdt`，不执行正式滤波，不插值，不删除 trial，也不拟合 ICA。所有自动结果都只是候选标记，人工结论保存在单独的复核表中。

## 2. 运行方法

完整运行命令：

```powershell
Set-Location -LiteralPath 'D:\AILearning\project_for_offer\erp-core-p3b'

& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b `
  python scripts\02_run_raw_qc.py `
  --output-dir results\raw_qc\final `
  --figures-dir reports\figures\raw_qc\final
```

只测试少数受试者：

```powershell
& 'D:\LearnWareSoft\Aanconda\_conda.exe' run -n erp-core-p3b `
  python scripts\02_run_raw_qc.py `
  --subjects sub-017 sub-006 sub-001 `
  --output-dir results\raw_qc\pilot `
  --figures-dir reports\figures\raw_qc\pilot
```

程序一次只加载一名受试者。即使全部P3数据接近2 GB，也不会把40人同时放入内存。

## 3. 行为刺激—反应配对

每个刺激的反应窗口从刺激 onset 开始，到下一个刺激 onset 或刺激后1.60秒结束，取两者中较早者。窗口中的第一个201/202是主反应：

- 201：正确；
- 202：错误；
- 没有反应：`missing_response`；
- 多于一个反应：`multiple_responses`；
- 主反应时间小于0.15秒：`anticipatory`；
- 不属于任何刺激窗口：`unmatched_response`。

反应时计算为：

```text
RT = response_onset - stimulus_onset
```

0.15秒用于标记需要检查的过快反应，不代表自动删除。原始文件中每人通常有任务正式开始前的反应事件，因此原始201/202数量不能直接作为trial正确率的分母。

## 4. 连续信号指标

MNE把EEG内部单位统一为伏特。程序在生成易读指标时乘以 `1e6` 转成微伏。原始数组不被修改。

### 4.1 时域指标

| 指标 | 含义 |
|---|---|
| `nonfinite_fraction` | NaN或无穷值占比 |
| `std_uv` | 全段标准差，反映总体波动 |
| `variance_uv2` | 全段方差 |
| `peak_to_peak_uv` | 最大值减最小值，容易受单个极端点影响 |
| `robust_peak_to_peak_uv` | 99.9%分位减0.1%分位，更不受单个点影响 |
| `median_abs_diff_uv` | 相邻采样点绝对差的中位数 |
| `flat_longest_s` | 相邻变化不超过0.01微伏的最长时间 |

BioSemi连续记录允许较大的直流偏置。QC图中的波形为了可读性逐通道减去中位数，但标准差、方差、PSD及保存的原始数据均没有使用这个显示变换。

### 4.2 Welch功率谱

程序使用4秒Hann窗、50%重叠和median Welch平均。没有预先做60 Hz陷波，因此图中的60 Hz峰是真实的原始工频噪声表现。

统计的频段包括：

- 0.1–1 Hz：慢漂移；
- 1–30 Hz：主要ERP/脑电频段；
- 30–45 Hz：高频或肌电候选；
- 58–62 Hz：60 Hz工频；
- 54–58 Hz和62–66 Hz：工频峰两侧的参照频段。

功率比使用dB：

```text
ratio_db = 10 × log10(目标频带功率 / 参照频带功率)
```

### 4.3 通道相关性

信号临时降采样到约64 Hz，只用于计算相关性。每个EEG通道与其余29个EEG通道计算绝对相关系数，再取中位数。极低相关性表示该通道的变化模式可能与整体头皮信号不一致，但单独一项低相关不能直接判坏。

## 5. 两轮阈值确定

第一轮在阈值未冻结时处理40人，共得到1200条EEG通道记录。观察到：

- 最长平直段约0.005秒；
- 绝大多数通道的稳健z在±5以内；
- 通道相关性分布偏斜，低相关通道不能单独判坏；
- 两个通道同时表现出极端方差和极低相关性。

冻结后的规则保存在 `configs/qc.yaml`：

```yaml
thresholds:
  frozen: true
  flat_duration_s: 1.0
  robust_z_abs: 5.0
  median_abs_correlation_min: 0.20
  candidate_min_statistical_flags: 2
  max_bad_channel_fraction: 0.20
```

稳健z使用：

```text
robust_z = (x - median) / (1.4826 × MAD)
```

自动坏道候选判定为：

- 出现NaN/无穷值或连续平直至少1秒；或者
- 方差、高频、工频、低相关四类统计标记中至少同时出现两类。

“至少两类证据”的设计用于降低把正常额区活动、枕区alpha或单一60 Hz峰误判为坏道的概率。

## 6. 40人实际结果

### 6.1 行为

| 项目 | 数量 |
|---|---:|
| 刺激trial | 8000 |
| 正确trial | 7609 |
| 错误trial | 352 |
| 漏反应trial | 39 |
| 多次反应trial | 0 |
| RT小于0.15秒的候选 | 35 |
| 未匹配反应 | 82 |

未匹配反应主要来自正式刺激开始前或有效反应窗之外的按键。它们不应强行配到最近的trial。

需要重点关注但不自动排除的行为记录包括：

- `sub-006`：45个错误、2个漏反应、10个过快反应候选；
- `sub-010`：38个错误、7个过快反应候选；
- `sub-008`：21个错误、18个漏反应；
- `sub-035`：25个错误、1个漏反应。

后续ERP平均通常只使用行为正确的trial，但这一规则必须在预处理配置中明确，并报告每人最终保留数量。

### 6.2 通道

自动规则产生两个坏道候选：

| 受试者 | 通道 | 标记 | 标准差 | 方差稳健z | 相关性中位数 |
|---|---|---|---:|---:|---:|
| sub-035 | PO3 | 方差异常、低相关 | 43476.0 µV | 13.88 | 0.157 |
| sub-039 | Oz | 方差异常、低相关 | 8539.3 µV | 12.27 | 0.071 |

这两个通道进入人工复核和后续插值候选。程序没有直接修改通道，也没有把它们从原始数据中删除。

### 6.3 受试者排除

30个EEG通道的20%是6个通道。只有坏道数超过6才触发受试者级自动排除候选。本次没有任何受试者达到该条件，因此：

```text
受试者自动排除候选 = 0
```

这不代表所有trial都无伪迹。trial级峰峰值、眼动和肌电剔除将在分段与预处理阶段完成。

## 7. 输出文件

`results/raw_qc/final/`：

| 文件 | 粒度 | 作用 |
|---|---|---|
| `behavior_trials.csv` | 8000行trial | 刺激、反应、RT、正确性和行为标记 |
| `behavior_summary.csv` | 40行受试者 | 行为汇总 |
| `channel_metrics.csv` | 1320行通道 | 30 EEG + 3 EOG的原始指标 |
| `subject_qc.csv` | 40行受试者 | 信号与行为联合汇总 |
| `bad_channel_candidates.csv` | 自动候选 | 候选的完整数值证据 |
| `bad_channel_review.tsv` | 人工复核 | 人工填写是否判坏及原因 |
| `exclusion_log.tsv` | 40行受试者 | 自动排除候选、人工结论和原因 |
| `run_summary.json` | 单次运行 | 参数和总数审计 |

`reports/figures/raw_qc/final/` 中有40张受试者总览图。红色柱表示自动坏道候选。

人工复核表中的 `manual_decision` 和 `manual_reason` 不会在重复运行自动QC时被覆盖。

## 8. 进入预处理前的结论

原始QC支持继续进入预处理，但需先人工确认 `sub-035/PO3` 和 `sub-039/Oz`。当前没有理由排除整名受试者。预处理阶段应执行：

1. 保留原始数据只读；
2. 依据人工复核结果登记坏道；
3. 建立0.1–30 Hz ERP分支和1–40 Hz ICA拟合分支；
4. 重参考前后分别记录指标；
5. ICA只移除有EOG证据支持的成分；
6. 分段后按统一峰峰值规则标记trial；
7. 报告每名受试者、每个条件最终保留的trial数。

