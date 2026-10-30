# 深度学习数据导出与训练：操作、方法和验收

## 1. 本次实现与实际执行的边界

2026-10-14：实现两个项目入口，而不是直接调用作者默认训练命令：

|入口|职责|运行位置|
|---|---|---|
|`12_export_deep_learning_dataset.py`|验证既有审计和共同清单，逐人导出两套波形缓存，记录适配提示|现有DL环境，CPU|
|`13_train_deep_learning.py`|EEGNet、CBraMod冻结线性头、CBraMod微调；smoke或完整分组评估|同一环境，默认GPU|

当前真实产物已经包含08三人GPU技术审计、09共同清单、10紧凑特征、11五折LR。
这里完成的是深度学习代码、配置和合成数据测试；**真实12导出及13训练由使用者运行，尚未产生深度学习正式成绩**。
三人08技术通过并不等于自动完成看图复核；原来的`pending`状态会保留在12来源记录里。

本次不重新下载数据、不改原始FIF、不重新拟合ICA、不安装/卸载CUDA、不推送GitHub。
完整Morlet长窗支线不阻塞本轮短窗分类。不要把长窗59个候选与短窗CBraMod适配提示混为一谈。

## 2. 环境与前提

使用已安装项目的环境：`D:\LearnWareSoft\CondaEnvs\erp-core-p3b-dl`。
当前本机检查记录为Python 3.11、PyTorch 2.12.1+cu130、RTX 5070。
本阶段不需要TensorFlow：EEGNet是项目内独立PyTorch架构适配，CBraMod读取已固定的作者代码和权重。
PyTorch自带CUDA运行时、系统驱动与本机开发工具包是不同层次；不要为了本项目卸载你现有CUDA环境。

```powershell
Set-Location 'D:\AILearning\project_for_offer\erp-core-p3b'
conda activate 'D:\LearnWareSoft\CondaEnvs\erp-core-p3b-dl'
python -c "import sys, torch, erp_core_p3b; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python scripts/12_export_deep_learning_dataset.py --help
python scripts/13_train_deep_learning.py --help
```

若Conda激活不可用，直接替换每条命令的`python`为：

```powershell
& 'D:\LearnWareSoft\CondaEnvs\erp-core-p3b-dl\python.exe' scripts/13_train_deep_learning.py --help
```

项目已经editable安装，不需要因为增加Python模块再次安装。

必需的已有文件：

- `data/processed/classification/v1`：09清单和划分；
- `data/processed/epochs/adjudicated`：39人裁决版FIF；
- `results/cbramod_audit/pilot-gpu`：三人真实GPU审计摘要和39人文件头inventory；
- `configs/cbramod_audit.yaml`：08时使用的原配置；
- `third_party/CBraMod`和`models/pretrained/CBraMod/pretrained_weights.pth`：已有模型资源。

12核对08配置hash、pilot名单、全体FIF身份、作者代码commit和权重hash。
13若训练CBraMod，还会再次验证来源代码文件hash。改变资源后必须重新审计，不允许静默加载其他权重。
权重SHA-256：`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`；
代码commit：`b9e961003214326972c567eff390e75b0287e32a`，以08配置记录为准。

## 3. 数据是什么，为什么要两套输入

沿用39人、7207个保留trial：Target 1286、Standard 5921。
标签固定`Standard=0, Target=1`，不是原MNE事件码`Target=1, Standard=2`。
任何一名受试者都不会跨同一折的train/validation/test；09的划分不会在12或13重抽。
`trial_id`、原trial序号、文件hash和epoch行号用于防止把删trial后的第n行错当原始第n次刺激。

|内容|原生缓存／EEGNet|CBraMod适配缓存|
|---|---|---|
|采样率|256 Hz|200 Hz|
|时间点|257|200|
|实际时间范围|-0.19921875～0.80078125 s|-0.195～0.800 s|
|刺激零点索引，0-based|51|39|
|EEG通道|30，保持现有顺序|相同30通道、相同顺序|
|磁盘形状|`(7207,30,257)`|`(7207,30,1,200)`|
|训练单batch形状|`(B,1,30,257)`|`(B,30,1,200)`|
|磁盘单位|物理伏特V|V/0.0001 V，无量纲|
|输入模型时单位|读取batch后除以0.0001 V|不重复缩放|

两者均保留既有0.1–30 Hz滤波、电压基线校正和最终平均参考，只取EEG，不将EOG作为预测输入。
100µV换算为1；10µV换算为0.1。这是固定的物理尺度换算，不是训练折拟合的统计标准化，
不按每个trial做z-score，避免直接抹掉对P3b可能有意义的振幅差异。

256→200 Hz复用08的`resample_poly(25,32)`协议。先将刺激零点锚定，再裁出200点，
边缘计算使用反射支持，不声称这提供了新的真实长时段，也不再做电压baseline。
12保存精确时间轴及每人对齐记录，而不是凭“约1秒”自行reshape或拼接下一trial。

CBraMod当前只有**一个时间patch**。编码器仍可处理通道及patch内部表示，但这不能证明多patch时间关系的优势；
ERP滤波/基线与预训练分布也不同。这是一项短窗离线迁移实验，不是作者ERP CORE结果复现或新SOTA声明。

## 4. 第一步：pilot导出与人工看图

为避免第一次就全体导出，先确认12复用了正确的适配协议：

```powershell
python scripts/12_export_deep_learning_dataset.py --stage pilot
```

输出为`data/processed/deep_learning/pilot_v1`，只含sub-001、sub-006、sub-025。
这个缓存不能用于共同五折训练，13会拒绝它。

看图来源分为两类：

1. 先看已有`results/cbramod_audit/pilot-gpu`中的原生/适配波形与审计报告。
   核对t=0位置是否一致、主要峰形是否保留、纵轴单位是否一致，边缘是否出现明显振铃。
   可通过下面命令列出图片路径，再在文件管理器中打开。
2. 12的`adaptation_review.csv`逐trial列出原生/适配最大通道峰峰值；
   适配超过200µV的trial进入`review_figures`。每图显示该trial适配后峰峰值最大的通道，
   对比原生实线与适配虚线。没有提示时不生成这个目录，不代表省略了数值检查。

```powershell
Get-ChildItem -LiteralPath '.\results\cbramod_audit\pilot-gpu' -Recurse -Filter '*.png' | Select-Object FullName
Get-Content -LiteralPath '.\data\processed\deep_learning\pilot_v1\export_report.md' -Encoding UTF8
```

重点不是看“哪种条件更容易分类”，而是看是否错位、缩放错误、反射边缘异常。
200µV是复核提示，**不自动作为新增删除阈值**；基础trial集合必须与LR保持一致。
若发现不能接受的信号问题，停止训练，先定位原因并形成新数据协议，不能用一句接受说明掩盖明显错误。

## 5. 第二步：确认限制后导出39人

完成上述看图，并接受单patch/短窗边界支持的限制后执行：

```powershell
python scripts/12_export_deep_learning_dataset.py --stage full --accept-short-window-limitations
```

输出为`data/processed/deep_learning/v1`，包含：

- `native_eeg_v.npy`、`cbramod_input.npy`、`labels.npy`；
- `trial_manifest.csv`和`subject_splits.json`：共用身份和划分；
- `export_manifest.json`：输入协议、形状、时间轴、版本hash及模型资源；
- `subject_export_audit.json`：每人对齐、平均参考残差和float32误差；
- `adaptation_review.csv`、可选`review_figures`；
- `export_report.md`、`run_summary.json`和配置快照。

输入依次逐人加载，输出使用float32磁盘memmap；训练也只复制当前batch。
两套波形磁盘体积约212 MiB＋165 MiB＝377 MiB，不包括FIF、权重、模型激活、优化器和检查点。
GPU显存不能用377 MiB直接估算；尤其全量微调要保存梯度和中间激活，以smoke实测为准。
为缓存、15个最佳模型及日志预留至少2 GB额外空间，不能把这个数字当整个项目总大小。

查看摘要：

```powershell
Get-Content -LiteralPath '.\data\processed\deep_learning\v1\export_report.md' -Encoding UTF8
Get-Content -LiteralPath '.\data\processed\deep_learning\v1\run_summary.json' -Encoding UTF8
```

`eligible_for_training=true`表示完整缓存、已接受限制且无新增振幅提示，不等于模型性能合格。
若`adaptation_review_trials>0`，12仍会完成导出，但13默认停止。看完每张候选图后，如结论确为保留全部trial，
训练命令须附实际复核说明，例如：

```powershell
# 示例文字必须替换成你实际看图后的结论，不能照抄为未经复核的授权。
python scripts/13_train_deep_learning.py --mode smoke --models eegnet --device cuda --run-name eegnet-reviewed-smoke --adaptation-review-note '已逐一查看候选图，记录具体数量和边缘波形结论；决定保持共同trial集合'
```

该说明至少10字符，记录到训练摘要，不回写08的pending状态、不篡改12的提示、不自动删除数据。

## 6. 第三步：先一折smoke

先跑小模型，检查DataLoader、训练循环及GPU环境：

```powershell
python scripts/13_train_deep_learning.py --mode smoke --models eegnet --device cuda --run-name eegnet-smoke-v1
```

再检查预训练网络的冻结与全量微调两条路径：

```powershell
python scripts/13_train_deep_learning.py --mode smoke --models cbramod_linear cbramod_finetune --device cuda --run-name cbramod-smoke-v1
```

smoke固定使用09第一折的train/validation，最多5 epoch，早停patience=3。
它**不创建测试DataLoader，不生成OOF，不计算外层测试成绩**。
这是工程检查，不是把5 epoch模型当正式结果。

验收时查看每个`<model>_fold-XX`子目录：

- `history.csv`：每轮训练加权loss、验证加权loss、验证受试者macro BA；
- `learning_curve.png`：左图loss，右图验证BA，虚线为选中epoch；
- `fit_audit.json`：总参数、可训练参数、类别权重、batch、时长、CUDA峰值显存；
- `best_model.pt`：由验证组选择的最佳checkpoint；
- `validation_predictions.csv`：保留trial与受试者身份，便于检查是否始终只预测一种类别。

冻结CBraMod可训练参数应主要是12002个线性头参数（6000×2＋2）；
全量微调显著更多，mask_encoding保持冻结。冻结版本编码器始终eval，dropout关闭。
检查日志没有NaN/Inf，所有配置hash与划分一致。验证BA一两轮不提高不一定是程序错误，
loss能下降也不自动意味着跨受试者泛化良好。不要为了让曲线好看去读取测试组。

## 7. 三种实验具体学习什么

### 7.1 EEGNet：从零训练的轻量对照

原生波形先做时间卷积，再做跨通道depthwise空间滤波，最后做separable卷积并分类。
空间权重可以理解为学习不同电极的组合，不是把EEG当自然图像。
256 Hz下128点时间核约覆盖0.5秒；这是一项明确配置的适配选择，并非自动找到的最优值。
F1=8、D=2、F2=16，池化4和8，dropout=0.5。
每次更新后限制空间核L2范数≤1、分类器行范数≤0.25；BN的PyTorch更新系数0.01对应Keras保留系数0.99。
这不是原Keras实现逐位复现；原作者架构说明见[ARL EEGModels](https://github.com/vlawhern/arl-eegmodels)。

### 7.2 CBraMod linear：预训练表示能否直接区分条件

每折都重新严格载入固定权重，然后移除原重建头`proj_out`。
输出`(B,30,1,200)`表示展平为6000维，只训练一个6000→2线性层。
编码器无梯度、无训练dropout；这样测量“现成表示＋线性头”而不是随机编码器噪声。
作者输入和模型实现依据[CBraMod仓库](https://github.com/wjq-learning/CBraMod)。

### 7.3 CBraMod finetune：允许表示适应当前任务

每折从同一原始预训练权重重新开始，不能接着上一折模型训练。
与linear使用相同线性头结构；编码器学习率1e-5，头1e-3，AdamW weight decay=0.01。
更小编码器学习率是保护预训练表示的初始工程选择，不是经测试组优化后的最优结论。
微调的可训练参数多，39人数据存在过拟合风险；不承诺一定超过LR或EEGNet。

EEGNet为原生网格，CBraMod为适配网格，因此二者差异包含输入适配因素，不能全部归因为架构。
linear与finetune共享CBraMod输入和分类头，更适合讨论“更新编码器”的增量效果。

## 8. 训练与统计协议

默认参数保存在`configs/deep_learning_training.yaml`。
固定五折outer test、每折独立的inner validation，与LR完全共用09划分。
模型训练seed=2026，每折实际seed=2026+fold编号；划分seed仍是09的42，二者职责不同。

训练使用类别加权交叉熵：训练折类别k权重为`w_k=N_train/(2*N_train,k)`。
让少数Target不因数量少而被忽略，不使用测试标签估权重，也不再叠加加权采样。
网络输出两个logits，训练损失内部计算softmax；评价分数是Target概率。
默认阈值0.5，报告BA、ROC-AUC、AP/AUPRC、Target precision/recall/F1和混淆计数。
加权交叉熵下分数未做概率校准，不将其直接解释为临床风险概率。

早停先计算每个验证受试者的BA，再等权平均。
`BA=(Target召回率+Standard召回率)/2`，比总体accuracy更适合当前类别不平衡。
改进须超过1e-6，同分保留较早epoch；正式最多80 epoch、patience=12；不使用学习率搜索或测试组调参。
每轮更新记录CSV，恢复best checkpoint后正式模式才预测该折test一次。

使用float32、无AMP、梯度裁剪L2范数1、Windows下num_workers=0。
严格确定性默认打开，固定Python/NumPy/PyTorch/DataLoader随机性并关闭TF32。
PyTorch官方说明不保证跨版本/硬件完全相同：[Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

full结果按39人等权汇总为主，trial混合pooled为辅。
bootstrap 2000次按受试者重采样，给95% CI；CI条件于本次固定划分和已拟合模型，不涵盖重新训练不确定性。
三模型间配对差值为描述性比较，不自动变成多重比较校正后的显著性结论。
程序验证已有LR目录hash与划分兼容，但不读取LR分数选深度学习参数，也不自动合并成最终报告。

## 9. 第四步：正式五折训练

两种smoke工程检查通过、配置确认后再执行：

```powershell
python scripts/13_train_deep_learning.py --mode full --models eegnet cbramod_linear cbramod_finetune --device cuda --run-name three-models-full-v1
```

这会串行训练3模型×5折＝15个模型；最多80 epoch/模型，不同时占用多个模型的显存。
不能根据数组很小推断训练很快。先看smoke每轮时间和峰值显存，再估计本机正式耗时。

正式输出`results/deep_learning/three-models-full-v1`：

|文件|如何使用|
|---|---|
|`oof_predictions.csv`|每个模型7207行，共21621行；同一trial对每模型只被outer test预测一次|
|`subject_metrics.csv`|每模型39人结果，观察个体差异而不只看总体分数|
|`subject_macro_summary.csv`|每人等权主指标、CI、有效受试者数|
|`pooled_metrics.csv`|全部trial混合的辅助指标及混淆计数|
|`paired_model_comparisons.csv`|同一受试者的第二模型－第一模型差值及bootstrap CI|
|`fit_audit.csv`|15次拟合的早停、参数量、耗时和显存记录|
|`training_report.md`|运行模式、主要BA、方法限制；不是最终简历报告|
|`config_snapshot.json`、`subject_splits.json`、`input_protocol.json`|复现当前实际参数、划分与输入|
|`model_resources.json`、`run_summary.json`|固定权重/代码来源、软件版本、模块hash和产物hash|

正式test查看之后，若还需要调整参数，应标记为探索性，不能继续把同一OOF作为未见测试成绩。
额外种子用于预先约定的稳定性分析，不能训练很多种子只报告最高分。

## 10. 常见错误、重跑与下一步

1. **目录已存在**：所有输出拒绝覆盖。训练换`--run-name eegnet-smoke-v2`；
   导出换`--output-dir data/processed/deep_learning/v2`，训练同步传`--dataset-dir data/processed/deep_learning/v2`。
   不删除或覆盖已完成结果。失败目录保留`status=failed`及错误，不能冒充completed缓存。
2. **CUDA不可用**：默认直接失败，不静默退回CPU。先检查终端是否使用上述DL环境。
   CPU仅适合小型工程排查，可显式`--device cpu`，不推荐用它跑CBraMod正式五折。
3. **CUDA out of memory**：记录错误与显存，再降低配置中CBraMod batch_size（如16→8），
   用新run-name重跑smoke，确认后冻结新配置。改变batch也可能改变优化结果，要如实记录，不能称完全同一实验。
4. **确定性算子不支持**：保留错误，确认具体算子；只有理解限制后才加
   `--allow-nondeterministic`并使用新run-name，摘要会记录deterministic=false，不能声称逐位复现。
5. **缓存hash或模型资源错误**：检查被改的是哪一步，重新生成一致版本，不跳过验证。
6. **新增振幅提示**：先看图，不能靠降低提示阈值/随意删trial凑分数。

下一批工作是正式结果验收与LR同划分对照：比较受试者BA/AP、Target漏检和假阳性，
检查异常受试者、训练/验证曲线与过拟合；然后整理核心图、限制、README和面试表述。
若微调不优于LR，这也是可报告的迁移适配结果；再决定是否值得做长窗、多种子或解释性扩展。
