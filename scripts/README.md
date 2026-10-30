# 命令行脚本

`15_check_release.py`：只用标准库与Git核验公开候选、大小、精选产物hash和首页导航，
可在新克隆安装科学计算环境之前运行；它不暂存、提交或推送文件。

正式脚本按流水线阶段命名：

- `01_validate_dataset.py`；
- `02_run_raw_qc.py`；
- `03_run_preprocessing.py`；
- `04_review_subjects.py`；
- `05_run_erp_analysis.py`；
- `06_audit_tfr_epochs.py`：只读审计长窗、原始事件对齐、小波支持与内存，不生成TFR。
- `07_build_tfr_epochs.py`：三人pilot长窗重建、冻结ICA复用、同参数ERP复现、内存采样与伪迹复核。
- `08_audit_cbramod_inputs.py`：真实短EEG的身份、单位、零点锚定重采样、图形和冻结/解冻模型工程检查。
- `09_build_classification_manifest.py`：共同trial索引、五折受试者划分、已有QC/ICA/ERP证据汇总。
- `10_extract_compact_features.py`：原生256Hz的ERP12、功率12、小波16列，支持pilot/full。
- `11_train_logistic_baseline.py`：smoke只看一折ERP验证组，full做三组五折LR和OOF评价。
- `12_export_deep_learning_dataset.py`：从共同清单逐人导出原生及CBraMod缓存，检查来源、对齐和新增振幅提示。
- `13_train_deep_learning.py`：EEGNet、CBraMod冻结线性头和微调；smoke只看验证组，full正式五折。
- `14_build_final_report.py`：核验两条正式OOF来源，生成六模型配对对照、39份个体报告和项目报告。

06脚本默认审计全体39人；`--subjects sub-001 sub-006`用于局部试运行，其输出进入独立的
`subset_*`子目录，不覆盖完整审计。参数和结果的详细解释见
`docs/09_tfr_epoch_audit.md`。

07脚本默认`--stage pilot`，已有长窗时使用`--verify-existing`校验而不覆盖；
`--review-only`从已生成的pilot长窗输出所有保护支持候选波形。
`--stage final`必须先通过三人门控。本轮有59个候选待复核，因此全体运行暂未放行。
详见`docs/10_tfr_long_epochs_pilot.md`。

脚本只负责解析参数和调用 `src/erp_core_p3b/` 中的实现，避免把核心逻辑复制到多个入口。

08脚本在`erp-core-p3b-dl`执行，默认核对39人文件头并处理三人pilot。
先运行`--signal-only --run-name pilot-signal`，再运行
`--device cuda --run-name pilot-gpu --save-inputs`。结果各自进入独立目录，已有名称拒绝覆盖。
此脚本不是全体训练数据导出或正式训练入口；技术通过后仍需看图，详细步骤见
`docs/11_cbramod_input_audit.md`。信号处理与LR对照支线的实施计划见
`docs/12_signal_processing_parallel_plan.md`。

09默认写入`data/processed/classification/v1`；10默认full写入
`data/processed/features/compact_v1`，pilot为独立目录且禁止当完整训练输入。
11默认smoke；显式`--mode full`才计算全部外层测试成绩。
全部产物目录拒绝覆盖，下游验证上游完成状态与文件hash。
命令、公式、参数、指标和重跑方法见`docs/13_classification_features_and_lr.md`。

12先`--stage pilot`核对；人工看图并接受单patch限制后才运行
`--stage full --accept-short-window-limitations`。新增适配提示不删trial，13要求实际复核说明。
13默认CUDA、smoke和三模型；可用`--models eegnet`先检查轻量训练，CUDA不可用不自动退回CPU。
正式训练用`--mode full`，每折重置模型，保持09的划分。
完整执行命令、两套输入差异、早停和指标解释见`docs/14_deep_learning_export_and_training.md`。

14仅在11和13的正式full产物完成后运行，固定0.5阈值，默认输出`results/final_report/v1`。
本次已生成；重跑传`--output-dir results/final_report/v2`，不会覆盖已有报告。
不重新训练或新增排除受试者；检查每trial的outer test归属，按受试者配对汇总。
详细结果、图形读法及7张图/39份报告的位置见`docs/15_final_results_and_error_analysis.md`。

