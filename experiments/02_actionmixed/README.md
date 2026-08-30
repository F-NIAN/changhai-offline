# ActionMixed 结构化时序实验

本目录记录从数据接入到最佳权重的主线。共同输入是 ActionMixed 的逐帧 YOLO 框和帧级动作标签；共同输出是
FeatureStore-like NPZ、三模型权重、逐帧/片段评估和可接后端的 SegmentFact/FactLedger。

| 顺序 | 内容 | 详情 |
|---:|---|---|
| 0 | 数据集接入和五类动作映射 | [00_dataset_onboarding](00_dataset_onboarding/README.md) |
| 1 | 68 维完整序列三模型 baseline | [01_full_sequence_baseline](01_full_sequence_baseline/README.md) |
| 2 | 113 维对象/遮挡/关系特征 v2 | [02_feature_v2](02_feature_v2/README.md) |
| 3 | 特征组合和训练方式对比 | [03_feature_training_optimization](03_feature_training_optimization/README.md) |
| 4 | 按模型最佳 recipe 训练权重 | [04_best_checkpoints](04_best_checkpoints/README.md) |

主脚本依赖关系：

```text
run_pipeline.py
  ├── data_transfer.py / dataset.py
  ├── segmenter/*.py
  └── segmentfact_ledger.py

run_optimization_experiments.py
  └── 复用 run_pipeline.OfflineSegmenter

train_best_checkpoints.py
  └── 复用优化实验中的特征 recipe 和评估函数
```
