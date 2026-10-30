# 数据、模型和源码来源

本仓库是ERP CORE P3b的独立分析与模型对照项目，与原数据和模型作者的官方实现区分。

- ERP CORE数据与实验来源：[ERP CORE](https://erpinfo.org/erp-core)。
  论文：[Kappenman et al., NeuroImage 225, 117465](https://doi.org/10.1016/j.neuroimage.2020.117465)。
  本地BIDS元数据记录数据许可为CC-BY-4.0。原始EEG未随仓库发布。
- EEGNet架构参考：[ARL EEGModels](https://github.com/vlawhern/arl-eegmodels)。
  本项目为独立PyTorch适配，不宣称逐位复现作者Keras训练。
- CBraMod代码：[wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod)，
  固定commit、权重revision/hash见[实验记录](reports/release/experiment_record.json)。
  代码MIT许可与权重Hub的Apache-2.0元信息分别记录，不互相替代。
  作者源码checkout与权重仅存放在本地忽略目录中。

公开表和精选图片来自本项目已经完成的分析；其来源hash见
[产物清单](reports/release/artifact_manifest.json)。完整EEG、逐trial预测、缓存与训练权重由复现流程生成。

当前没有为自有源码新增统一的开源许可证。将代码上传GitHub与授予第三方某种开源许可是不同决定；
这些上游许可记录不等于本项目全部文件自动采用相同许可。
