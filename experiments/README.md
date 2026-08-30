# offline-model 实验总索引

本目录按日期和所属关系组织实验说明。源码、原始报告和结果仍保留在历史路径；这里统一回答每次实验的
“当时有什么、做了什么、结果如何、为什么、下一步是什么、怎么复现”。所有路径均相对仓库根目录。

## 文档树

```text
experiments/
├── 01_temporal_baseline/
├── 02_actionmixed/
│   ├── 00_dataset_onboarding/
│   ├── 01_full_sequence_baseline/
│   ├── 02_feature_v2/
│   ├── 03_feature_training_optimization/
│   └── 04_best_checkpoints/
├── 03_rgb_roi/
│   ├── 01_v1_basic_rgb/
│   ├── 02_v2_quality_cache/
│   ├── 03_v3_segment_postprocess/
│   └── 04_v4_proxy_roi/
├── 04_real_yolo_rgb_roi/
│   ├── 01_v1/
│   ├── 02_v2/
│   ├── 03_v3/
│   └── 04_v4/
└── 05_visualization/
```

## 时间顺序

| 日期 | 实验/修改 | 核心结果 | 详细文档 |
|---|---|---|---|
| 2026-07-05 | 离线时序 baseline 与账本闭环 | 数据到 SegmentFact/FactLedger 链路跑通 | [01](01_temporal_baseline/README.md) |
| 2026-07-10 | ActionMixed 数据接入 | 五类动作、逐帧框和视频级划分接入 | [02-00](02_actionmixed/00_dataset_onboarding/README.md) |
| 2026-07-15 | 68 维三模型完整序列 baseline | ASFormer 1 epoch `F1@0.25=0.4607` | [02-01](02_actionmixed/01_full_sequence_baseline/README.md) |
| 2026-07-15 | 113 维 v2 特征 | hand top-2、遮挡补全和关系 delta 完成 | [02-02](02_actionmixed/02_feature_v2/README.md) |
| 2026-07-15 | 特征 × 训练方式优化 | BiGRU 249 维滑窗方案 `F1@0.25=0.1917` | [02-03](02_actionmixed/03_feature_training_optimization/README.md) |
| 2026-07-15 | 三模型最佳 recipe 权重 | 推荐 BiGRU；保留三模型可追溯权重 | [02-04](02_actionmixed/04_best_checkpoints/README.md) |
| 2026-07-21 | RGB/ROI v1 | 低维颜色统计可跑通，BiGRU 最稳 | [03-01](03_rgb_roi/01_v1_basic_rgb/README.md) |
| 2026-07-21 | RGB/ROI v2 | ROI manifest/cache 与质量感知特征 | [03-02](03_rgb_roi/02_v2_quality_cache/README.md) |
| 2026-07-22 | RGB/ROI v3 | val 选段级后处理，test 复用 | [03-03](03_rgb_roi/03_v3_segment_postprocess/README.md) |
| 2026-07-28 | RGB/ROI v4 | proxy ROI、参数集中化和 early stopping | [03-04](03_rgb_roi/04_v4_proxy_roi/README.md) |
| 2026-08-04 | 真实 YOLO 框 v1-v4 复验 | v4 BiGRU test raw ACC `0.7028` | [04](04_real_yolo_rgb_roi/README.md) |
| 2026-08 | 预测时间线可视化 | 生成三模型 GT/Prediction 对比图 | [05](05_visualization/README.md) |

## 指标口径注意

- 早期 `run_pipeline.py` 报告以验证集 Segment F1 为主；后续优化报告同时给帧级和段级指标。
- `image_train/` 使用数据集已有框/理想框；`yolo_image_train/` 使用真实 YOLO v0.3 预测框，二者不可直接混报。
- 多数实验轮数很少，且不同阶段数据序列数从 21 条扩为 train/val/test `14/11/7`，横向比较时必须保留数据来源、split 和 epoch。
- 只有后续图像实验明确列出 test；早期最佳权重报告主要是 validation 结果。
