# 四名高剔除率受试者复核报告

## 1. 为什么不能只按一个百分比删人

初次预处理后，`sub-009、sub-025、sub-030、sub-039` 的epoch剔除率超过20%。高剔除率可能
来自三种完全不同的情况：

1. 一个坏电极反复触发整trial删除，此时应优先插值该电极；
2. 眼动成分未达到保守的自动ICA阈值，但地形和EOG证据足够明确，此时可人工删除成分；
3. 多通道、跨时段持续异常，此时更可能需要排除受试者。

因此本报告先定位异常来源，再用独立的 `reviewed` 分支试算修正，最后冻结决定。保留后的ERP
只用于检查处理是否产生明显错误，不以“P3b是否符合预期”决定删人，避免循环论证。

## 2. 每张九宫格仪表板怎么看

### 第一行：异常来自哪里、是否贯穿实验

- **Epochs >200 µV by channel**：每个通道导致多少epoch超过阈值。一两根柱远高于其他通道，
  优先怀疑局部坏道；许多通道都高，提示弥散噪声。
- **Spatial spread**：每个被拒epoch同时有多少通道超限。1个通道居多通常是局部问题；多个
  通道同时超限更像运动、放大器阶跃或广泛伪迹。
- **Anomalies across recording**：横轴是实验时间，纵轴是每个epoch最大峰峰值，使用对数坐标。
  红点贯穿全程说明问题持续存在；集中成簇则应检查对应时间段。

### 第二行：异常时间块和最严重波形

- **Rejection rate in 10 time bins**：把完整记录等分为10段。只有一两段异常时，可以考虑
  标记坏时间段；后半程逐渐升高可能是电极松动或疲劳运动。
- **Worst epoch: all EEG**：最严重epoch的通道×时间热图。只出现一条强带提示单通道故障；
  多条同步变化提示空间弥散伪迹。显示色阶会裁剪，真实最大值看右侧标题。
- **Worst channel waveform**：检查阶跃、饱和、尖峰还是缓慢眼动。电极阶跃常表现为突然跳变后
  长时间停在新水平；眼动通常在额区更强并有可识别空间分布。

### 第三行：ICA、trial数量和保留后ERP

- **ICA–EOG evidence**：每个成分与三个EOG通道中的最大绝对相关。红色为实际删除成分；相关
  高只是筛选证据，还必须结合头皮地形。
- **Condition trial counts**：同时检查Target和Standard，不只看总trial。Target较少，是最先
  失去统计稳定性的条件。
- **Retained Pz ERP**：灰色为300–600 ms检查窗。主要检查基线、振荡和量级是否明显异常，
  不用曲线是否符合预期来决定排除。

ICA地形图中，额部对称强投影常对应眨眼；左右相反的额颞梯度常对应水平眼动；局限在一个后部
电极附近的极端投影更像坏道，而不是可解释的独立生理源。

## 3. 修正前总览

| 受试者 | 剔除率 | 弥散坏epoch占比 | 时间箱 | 最异常通道 | Target/Standard保留 |
|---|---:|---:|---:|---|---:|
| sub-009 | 52.43% | 1.03% | 10/10 | PO8:61；P10:56 | 14/74 |
| sub-025 | 20.60% | 19.51% | 8/10 | FP2:39；FP1:26 | 33/125 |
| sub-030 | 64.06% | 34.15% | 10/10 | FP1:114；FP2:100；F8:31 | 13/56 |
| sub-039 | 58.08% | 0% | 10/10 | FP2:115 | 21/62 |

“弥散坏epoch”定义为同一epoch至少3个EEG通道超过200 µV。

## 4. 逐人证据和决定

### 4.1 sub-009：保留，补登记PO8/P10坏道

修正前97个epoch超限，但只有1个坏epoch涉及至少3个通道。PO8有61个超限epoch，最严重波形
达到14888 µV，呈单通道巨大阶跃；P10有56个超限epoch。原ICA仅保留3个成分，说明坏道的
极端方差支配了99%方差准则，而不是全脑信号只有3个有效源。

把PO8和P10排除在参考及ICA拟合之外并插值后：

- ICA维数由3恢复到18；
- 185个行为有效epoch全部低于200 µV；
- 保留32个Target、153个Standard；
- 自动识别并删除ICA000、ICA002两个强EOG相关成分。

**决定：保留。** 这是典型的少数坏电极问题，不应因此删除整名受试者。

![sub-009原ICA地形](../reports/figures/subject_review/four_candidates/sub-009_top_ica_eog_topographies.png)

![sub-009修正前](../reports/figures/subject_review/four_candidates/sub-009_review_dashboard.png)

![sub-009修正后](../reports/figures/subject_review/four_candidates_after_intervention/sub-009_review_dashboard.png)

### 4.2 sub-025：保留，并在敏感性分析中重点检查

异常主要集中在实验约第2～2.7分钟，对应第3、4时间箱；后两个时间箱没有超限epoch。最严重
trial是F8超过1300 µV的阶跃。ICA000和ICA002虽然相关只有0.39和0.46，但头皮地形均呈明显
额部眼电模式，因此人工删除。

修正后：

- 剔除率从20.60%降到19.10%；
- 保留33个Target、128个Standard；
- 残余38个坏epoch仍主要位于FP2/FP1及中段时间簇，由trial级200 µV规则删除。

**决定：保留，并做敏感性分析。** 剩余trial数量足够，异常具有时间和空间局限性，没有证据表明
整段记录失效。后续组统计同时报告“包含/不包含sub-025”的结论稳定性。

![sub-025 ICA地形](../reports/figures/subject_review/four_candidates/sub-025_top_ica_eog_topographies.png)

![sub-025修正前](../reports/figures/subject_review/four_candidates/sub-025_review_dashboard.png)

![sub-025修正后](../reports/figures/subject_review/four_candidates_after_intervention/sub-025_review_dashboard.png)

### 4.3 sub-030：主分析排除，保留敏感性分支

ICA000呈明确额部眨眼模式，人工删除后剔除率从64.06%降到30.73%，说明眼电解释了一部分问题，
但不能解释全部：

- 10/10时间箱仍有超限epoch，后半程达到约35%～80%；
- FP1、F8、F4、O2、P10等额、颞、枕多个区域均反复异常；
- 59个坏epoch中33个涉及至少3个通道，即55.93%为空间弥散异常；
- 最严重F8仍超过2173 µV；
- 修正后仅保留26个Target、107个Standard。

**决定：从主分析排除。** 原因是修正后仍存在跨时间、跨通道的持续质量问题，而不是因为P3b
曲线大小。`reviewed`分支仍保存其133个epoch，可用于包含该受试者的敏感性分析。

![sub-030 ICA地形](../reports/figures/subject_review/four_candidates/sub-030_top_ica_eog_topographies.png)

![sub-030修正前](../reports/figures/subject_review/four_candidates/sub-030_review_dashboard.png)

![sub-030修正后](../reports/figures/subject_review/four_candidates_after_intervention/sub-030_review_dashboard.png)

### 4.4 sub-039：保留，Oz插值并人工删除两个眼动成分

修正前115个坏epoch全部由FP2触发，几乎没有多通道同时超过200 µV。ICA000呈额部对称的垂直
眼动地形，ICA007呈左右梯度的水平眼动地形；它们与EOG的最大相关分别约0.60和0.45，低于
为了防止误删而设置的0.70自动阈值，但空间模式清晰。

在原有Oz插值基础上人工删除ICA000和ICA007后：

- 198个行为有效epoch全部低于200 µV；
- 最大epoch峰峰值降到152.0 µV；
- 保留38个Target、160个Standard。

**决定：保留。** 高剔除率由未被保守自动规则捕获的眼动成分解释，修正后没有剩余振幅异常。

![sub-039 ICA地形](../reports/figures/subject_review/four_candidates/sub-039_top_ica_eog_topographies.png)

![sub-039修正前](../reports/figures/subject_review/four_candidates/sub-039_review_dashboard.png)

![sub-039修正后](../reports/figures/subject_review/four_candidates_after_intervention/sub-039_review_dashboard.png)

## 5. 冻结后的主分析数据

正式人工决定保存在 `configs/manual_qc.yaml`。旧的 `final` 结果保留作对照，主分析使用新的
`adjudicated`分支：

| 指标 | 裁决后结果 |
|---|---:|
| 进入主分析的受试者 | 39 |
| 行为有效epoch | 7396 |
| 200 µV规则删除 | 189 |
| 最终保留epoch | 7207 |
| Target / Standard | 1286 / 5921 |
| 未收敛ICA | 0 |
| 仍需自动复核的受试者 | 0 |

Pz在300–600 ms的受试者平均幅值为Target 5.650 µV、Standard 2.165 µV，差值3.486 µV；
39/39人的方向均为Target大于Standard。这只是流水线合理性检查，不替代下一阶段的正式统计。

## 6. 复现命令

生成初始四人复核图：

```powershell
python scripts/04_review_subjects.py --stage final
```

运行独立修正试算并重画：

```powershell
python scripts/03_run_preprocessing.py --stage reviewed `
  --manual-qc configs/review_interventions.yaml `
  --subjects sub-009 sub-025 sub-030 sub-039
python scripts/04_review_subjects.py --stage reviewed
```

生成主分析使用的39人裁决版：

```powershell
python scripts/03_run_preprocessing.py --stage adjudicated
```

原始指标见 `results/subject_review/four_candidates/`，修正后指标见
`results/subject_review/four_candidates_after_intervention/`，逐人最终决定见 `decisions.csv`。
