# RGB / ROI 图像特征实验（标注框来源）

本组实验在结构化 YOLO 框特征之外加入 RGB/ROI 外观信息。这里使用 ActionMixed 数据集已有的框/对齐结果，
更接近“理想框”上限；2026-08-04 的 `yolo_image_train/` 才使用真实 YOLO 预测框，两组指标不能混报。

| 版本 | 日期 | 核心变动 | 详情 |
|---|---|---|---|
| v1 | 2026-07-21 | 基础 ROI 颜色统计 | [01](01_v1_basic_rgb/README.md) |
| v2 | 2026-07-21 | ROI manifest/cache、质量特征和时序增强 | [02](02_v2_quality_cache/README.md) |
| v3 | 2026-07-22 | val 搜索段后处理，test 固定验证 | [03](03_v3_segment_postprocess/README.md) |
| v4 | 2026-07-28 | proxy ROI、训练超参集中化、early stopping | [04](04_v4_proxy_roi/README.md) |

共享代码：

- `image_train/common/roi_cache.py`：ROI 对齐清单与 RGB 缓存。
- `image_train/common/proxy_roi.py`：洗槽、通道、扩手框等 proxy ROI。
- `image_train/common/segment_postprocess.py`：概率平滑、短段删除、间隔合并、阈值和段级指标。
- `image_train/current_feature_inventory.md`：各版本特征维度和语义。

后续方案记录：`image_train/image_train_v4future.md`、`image_train/image_train_v5future.md` 和
`docs/actionmixed_multimodal_future_design.md`。这些是设计文档，不是已完成实验结果。
