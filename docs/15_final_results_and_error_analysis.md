# 正式六模型对照、逐受试者误差分析与项目交付

## 1. 当前状态与阅读入口

2026-10-14：使用者完成`three-models-full-v1`；本阶段读取已完成的预测，
核验数据/划分/模型来源及outer test身份，生成六模型对照、39份个体报告和项目最终报告。
没有重新训练，也没有按测试成绩修改阈值、删除受试者或改变既有QC决定。

完整报告：[project_final_report.md](../results/final_report/v1/project_final_report.md)。
这份报告包含QC/ICA/滤波/ERP工作、模型方法、正式对照、统计边界、图形读法、开销和简历表述。
本说明进一步解释当前结果及怎样查个体错误。

报告入口为`14_build_final_report.py`，实现位于`src/erp_core_p3b/final_report.py`。
配置为`configs/final_report.yaml`。
本轮已经生成`results/final_report/v1`，不需要再次运行同一输出目录。

## 2. 为什么这批模型能够比较

所有模型来自同一39名受试者、7207个trial：1286 Target、5921 Standard。
ERP/LR和深度学习共用09的trial身份及固定受试者五折划分。
每个模型每个trial只有一次outer test预测，合并后核验了6×7207＝43242条预测。

报告核验完成状态、文件SHA-256、实际subject_splits.json，以及每条OOF的：

- trial_id是否在共同清单且恰好出现一次；
- subject_id和真实label是否正确；
- fold是否为该trial的outer test折，而不是validation折；
- 分数是否有限且在[0,1]，原有prediction是否与固定0.5阈值一致。

LR包括ERP12列、ERP+功率24列、ERP+功率+DWT40列三个增量对照。
EEGNet与CBraMod的输入采样网格不同，结果同时包含适配差异。
CBraMod linear与finetune输入/分类头一致，二者对照更直接说明更新编码器的效果。
详细训练参数、单patch限制和离线预处理边界见[14说明](14_deep_learning_export_and_training.md)。

## 3. 正式结果及可支持的解释

下表都是先逐人计算、再对39人等权平均的指标；不能与trial混合pooled指标直接混为一谈。

|模型|macro BA|macro ROC-AUC|macro AP|Target召回|Standard特异度|
|---|---:|---:|---:|---:|---:|
|LR ERP12|0.6557|0.7310|0.4249|0.6352|0.6761|
|LR ERP+功率24|0.6566|0.7373|0.4242|0.6279|0.6853|
|LR ERP+功率+DWT40|0.6540|0.7353|0.4213|0.6265|0.6814|
|EEGNet|0.6917|0.7744|0.4910|0.6661|0.7174|
|CBraMod linear|0.6547|0.7270|0.4129|0.5748|0.7346|
|CBraMod finetune|0.6902|0.7715|0.4832|0.6984|0.6820|

EEGNet和微调的表现接近，不能依据0.0016的均值差宣布EEGNet稳定优于微调。
微调−EEGNet的配对BA均值为−0.0016，条件bootstrap 95% CI [−0.0183, 0.0143]；
39人中微调有20人更高、19人更低。区间跨0表明当前数据不能支持稳定差异，
它也不是“已证明等效”的等效性检验。

固定参考为LR ERP12，不按测试最高分选择参考。EEGNet−LR ERP为+0.0361，
95%条件CI [0.0186,0.0537]，31人提高；微调−LR ERP为+0.0345，
区间[0.0199,0.0489]，29人提高。方向在本次拟合模型的受试者重采样中保持较一致。
这些都是描述性条件区间，未做多重比较校正，也不包含重新训练的不确定性。

微调−linear为+0.0355，区间[0.0196,0.0532]，30人提高，
说明当前适配条件下更新预训练编码器比仅训练线性头更有帮助。
不能据此推断换成任意EEG任务、输入窗或数据集也一定有同样收益。

功率加入ERP的平均BA增量为+0.0009，再加入DWT为−0.0026，区间均跨0。
当前40列定义和固定LR下未显示明确增量；这不证明频带/小波方法本身无用，
只限定于本任务、短窗、特征公式、样本和分类器设置。没有为了追求增量继续挑特征。

这是一项只有一个训练seed设置的比较。没有进行标签置换机会水平检验，
不应把描述性分数和bootstrap区间写成已完成的确认性显著性结论。

## 4. 为什么微调召回更高，BA却没超过EEGNet

BA等于Target召回和Standard特异度的平均，两种错误都进入评价。
在全部7207个trial的pooled计数中：

|模型|Target检出TP|Target漏检FN|Standard误报FP|正确Standard TN|
|---|---:|---:|---:|---:|
|EEGNet|850|436|1672|4249|
|CBraMod finetune|892|394|1880|4041|

微调多找出42个Target，但同时多误报208个Standard。
两类数量差异很大，不能直接比较42和208判断谁更好；应分别除以该类总数，
再结合任务对漏检/误报的代价。当前固定0.5阈值下微调偏向较高召回，EEGNet更平衡。
这里没有根据测试组重新选择阈值。

AP体现Target排名质量，EEGNet为0.4910，微调0.4832，均高于linear的0.4129；
AP不是accuracy，也不是ROC-AUC。分数没有经过概率校准，不应解释成临床风险概率。

始终预测Standard的pooled accuracy可以达到5921/7207＝82.16%，
但Target召回为0、BA为0.5。这解释了为何本项目不能只报告总体accuracy。

三神经网络共同错误988个trial，其中237个Target漏检、751个Standard误报；
另有2512个trial的三神经网络预测不一致。这些定位用于解释共同难点和模型差异，
不作为新增剔除伪迹规则，也不自动推断为数据标注错误。

## 5. 如何复核39名受试者

先打开[subject_review.csv](../results/final_report/v1/subject_review.csv)，
再打开对应的`results/final_report/v1/subjects/sub-XXX.md`。
每份报告包含6模型BA/ROC/AP/召回/特异度/FN/FP、原QC和ICA记录、ERP幅值差，
以及三神经网络共同错误的原trial序号和FIF epoch索引。

建议先看五个三神经网络平均BA较低的个案：

|受试者|主要表现|如何解释|
|---|---|---|
|sub-019|Target召回较高，EEGNet/微调特异度约0.38/0.36|误报Standard较多；三网络共同误报68个|
|sub-022|EEGNet/微调Target召回都约0.93，特异度约0.33/0.26|高召回伴随大量误报；共同误报81个|
|sub-036|EEGNet/微调召回约0.24/0.37，特异度约0.92/0.79|主要问题是漏检Target；共同漏检20个|
|sub-018|EEGNet/微调召回约0.33/0.52|两模型均有Target漏检；微调改善召回但损失部分特异度|
|sub-039|EEGNet BA 0.5809，微调0.6808|微调对该人改善明显，但不能将原因直接归于已有Oz插值|

这些受试者当前都保留。低分类成绩不能反向证明原QC决策错误。
如需解释原因，先结合原波形/PSD/ICA审计，再检查错误是否集中在特定时段；
单靠总体BA或ERP差值无法证明残余伪迹、个体生理差异或模型缺陷的因果来源。

另一组需注意的是Target数量少的sub-006、sub-010、sub-035：分别5、8、18个。
每错一个Target，BA变化分别为0.1000、0.0625、0.0278，个体排名可能受少数trial影响。
它们仍属于主分析；本轮只提示解释粒度，没有按事后阈值排除。

查看图形时：

1. `subject_metric_heatmaps.png`左图看BA，中图看Target召回，右图看Standard特异度。
   中图亮、右图暗意味着偏向预测Target；中图暗、右图亮意味着偏保守。
2. `paired_subject_scatter.png`每点一人；微调位于对角线上方表示该人比参考模型BA高。
   重点看收益是否普遍、是否有人明显退化，不只看均值。
3. `confusion_matrices.png`按真实类别逐行归一化；右上是误报率，左下是漏检率。
4. `validation_learning_curves.png`展示15次验证曲线，虚线为实际选中的最佳epoch。
   对照`fit_review.csv`的验证/测试BA和最佳/最后loss，可观察过拟合和折间差异。
5. `error_context.png`只是事后探索性散点。ERP效应和QC指标不是本轮的新预测特征，
   不能把相关外观写成已证实的原因。

进一步定位trial用[trial_error_audit.csv](../results/final_report/v1/trial_error_audit.csv)。
先筛`subject_id`，再找某模型`__error=1`；结合真实`label`区分FN与FP。
`epoch_index`是当前FIF中0-based保留行号，`trial_index`是原trial序号，二者不能互换。
按`source_file`和`epoch_index`读取原波形时不要写回FIF。

## 6. 指标和区间的复现细节

主要表由共同OOF重新计算，模型标签统一映射，macro均值与原训练报告一致。
新报告对单模型bootstrap输入按受试者编号排序；有限的2000次随机重采样中，
顺序与原“按fold后按subject”排序不同，CI端点可能有小幅Monte Carlo差异。
这不代表改了预测或重新训练，均值、混淆计数和身份保持一致。
报告内所有CI以`subject_macro_ci.csv`及`paired_comparisons.csv`为准。

bootstrap保持同人的模型配对，不能按7207个trial独立重采样后当作39人泛化区间。
这些区间条件于现有划分和拟合模型；不覆盖CV训练集重叠的全部依赖和多种子训练方差。
当前没有p值表，不根据区间是否跨0宣称经过多重校正的显著性。

## 7. 输出、执行命令与验收

|产物|用途|
|---|---|
|`project_final_report.md`|完整项目叙述、结果、限制和简历表述|
|`model_overview.csv`、`subject_macro_ci.csv`|6模型主指标和受试者条件CI|
|`paired_comparisons.csv`|固定参考的7组配对差值及提高/降低人数|
|`subject_metrics.csv`、`subject_review.csv`|234个模型×受试者结果及信号证据|
|`subjects/*.md`|39份个体复核报告|
|`trial_error_audit.csv`|7207行共同trial与6模型分数/错误类型|
|`pooled_metrics.csv`|辅助trial混合指标、混淆计数|
|`fit_review.csv`、`compute_summary.csv`|早停、训练/验证loss、验证−测试BA、参数/耗时/显存|
|`figures/*.png`|7张科学图表|
|`source_provenance.json`、`run_summary.json`|上游摘要hash、版本和本报告产物hash|

本次已执行：

```powershell
python scripts/14_build_final_report.py
```

如以后需要独立重现一份新报告：

```powershell
Set-Location 'D:\AILearning\project_for_offer\erp-core-p3b'
conda activate 'D:\LearnWareSoft\CondaEnvs\erp-core-p3b-dl'
python scripts/14_build_final_report.py --output-dir results/final_report/v2
```

14仅做CPU结果汇总，不需要GPU，不加载模型权重，不重新读取全体EEG。
输出目录已存在时拒绝覆盖；输入检查失败时停止，不绕过hash或错折检查。
配置默认锁定0.5阈值、95% CI和2000次bootstrap，与本说明对应。

已验证报告产物hash、7张图的布局，以及关键错误身份/错折拒绝/配对差值/原trial定位测试。

## 8. 项目完成范围和后续选择

本项目的基础闭环已经完成：公开EEG数据→QC和预处理→生理ERP统计→
紧凑信号特征/LR→现代预训练模型适配与微调→受试者分组评估→误差分析与最终报告。
这些内容已能支持信号处理和深度学习相关岗位的项目陈述。

简历可突出：40人数据验证/QC、39人裁决分析、固定受试者五折、训练内标准化、
CBraMod严格权重来源、冻结/微调对照和逐trial审计；模型指标必须写清macro与评价范围。
EEGNet macro BA约0.692、macro ROC-AUC约0.774，微调macro BA约0.690。

后续优先是整理GitHub可公开源码/配置/说明和面试演示，不一定立即增加更多模型。
本阶段未自动提交或推送。大数组、FIF和权重按既有规则不进入版本控制，
最终结果文件位于默认忽略的results目录；如果要在GitHub展示图表，另行选择少量图和聚合结果发布。

若继续研究，优先考虑预先约定的多种子稳定性或外部队列，而不是依据这次OOF反复调参。
长窗Morlet/ITC、完整保护支持复核、外部验证和模型解释属于可选扩展，不能写成已完成。
