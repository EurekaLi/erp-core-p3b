# 命令行脚本

正式脚本按流水线阶段命名：

- `01_validate_dataset.py`；
- `02_run_raw_qc.py`；
- `03_run_preprocessing.py`；
- `04_review_subjects.py`；
- `05_run_erp_analysis.py`。

脚本只负责解析参数和调用 `src/erp_core_p3b/` 中的实现，避免把核心逻辑复制到多个入口。

