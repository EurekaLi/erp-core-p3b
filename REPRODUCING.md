# 从新克隆复现

仓库包含源码、配置、测试与精选结果；数据、模型权重和完整运行输出需要单独获取。
首页可以直接展示已提交的图表，发布核验也不依赖原始数据或科学计算环境。

## 1. 检查源码与公开结果

```powershell
git clone https://github.com/EurekaLi/erp-core-p3b.git
Set-Location erp-core-p3b
python -X utf8 scripts/15_check_release.py
```

15只需要Python 3.11标准库与Git，检查当前工作树的公开候选、大小、精选表/图hash和导航链接。
它不是完整的历史凭据扫描，也不会暂存、提交、联网或加载权重。

## 2. 信号处理环境与数据

```powershell
conda env create -f environment.yml
conda activate erp-core-p3b
python -m pip install -e .
```

数据获取与目录见[下载说明](docs/01_dataset_and_download.md)。按
[验证](docs/02_data_validation.md)、[QC](docs/03_raw_qc.md)、
[预处理](docs/04_preprocessing.md)、[个案裁决](docs/05_four_subject_review.md)和
[ERP分析](docs/06_p3b_analysis_plan.md)顺序执行01–05。
人工裁决已经在配置中留痕；新数据集不可仅照抄本数据的坏道/排除名单。

本项目的主要分类输入是39人裁决版FIF，并依赖Phase 5与预处理证据表。
09不是从原始BIDS直接产生清洗数据的程序，12也不能替代08真实输入审计。

## 3. 独立DL环境与固定资源

可在环境创建时用不同名字建立同一科学计算基础：

```powershell
conda env create -f environment.yml -n erp-core-p3b-dl
conda activate erp-core-p3b-dl
python -m pip install -e .
```

再根据[PyTorch官方安装说明](https://pytorch.org/get-started/locally/)安装与硬件兼容的PyTorch。
实测组合是Python 3.11.17、PyTorch 2.12.1+cu130及RTX 5070，见
[环境记录](reports/release/environment_record.json)。该记录不是全依赖锁文件，不保证跨版本逐位相同。
系统驱动、CUDA工具包与PyTorch运行时分别管理，无需为复现卸载既有CUDA。

固定CBraMod作者代码（在尚不存在目标目录时执行）：

```powershell
git clone https://github.com/wjq-learning/CBraMod.git third_party/CBraMod
git -C third_party/CBraMod checkout b9e961003214326972c567eff390e75b0287e32a
python -m pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='weighting666/CBraMod', filename='pretrained_weights.pth', revision='500543c7e30bda1b22bfd51a49301b238dee21fd', local_dir='models/pretrained/CBraMod'))"
```

权重SHA-256应为`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。
下载接口参数参考[Hugging Face官方文档](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)。
08会校验来源、输入、单位及零点，具体命令与看图要求见
[真实输入审计](docs/11_cbramod_input_audit.md)。作者代码和权重使用各自许可，不随本仓库分发。

## 4. 分类和最终报告

使用[09–11说明](docs/13_classification_features_and_lr.md)建立共同清单、特征和LR；
使用[12–13说明](docs/14_deep_learning_export_and_training.md)进行pilot导出、图形复核、smoke与正式训练；
使用[14说明](docs/15_final_results_and_error_analysis.md)产生完整报告。
首页给出这些入口的简要命令，首次运行仍需按详细说明通过审计。

输出目录拒绝覆盖。重跑应使用新的运行名/版本，并将下游输入目录同步指向相应上游。
完整长窗TFR不是这条分类复现路径的前置条件。

## 5. 核验与发布

在已安装全部测试依赖的DL环境：

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -X utf8 scripts/15_check_release.py
```

Windows限制临时目录权限时，可把pytest的`--basetemp`设为项目内一个尚不存在的
`.pytest-任意唯一标识-tmp`目录；不要复用有重要文件的目录，pytest可能清理它。

提交与推送仍由使用者执行，见[GitHub材料说明](docs/16_github_materials.md)。
