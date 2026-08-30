# image_train

本目录保存基于 ActionMixed 已有框/标注框的 RGB/ROI v1-v4 实验代码、原始报告和本地产物。

- v1：`train_image_v1.py`，基础 RGB ROI 统计。
- v2：`v2/train_image_v2.py`，ROI manifest/cache 与质量感知特征。
- v3：`v3/train_image_v3.py`，val 选择段级后处理。
- v4：`v4/train_image_v4.py` + `v4/configs/image_train_v4.json`，proxy ROI、集中配置和 early stopping。
- 共享实现：`common/roi_cache.py`、`common/proxy_roi.py`、`common/segment_postprocess.py`。
- 原始报告：`image_train_v1.md` 至 `image_train_v4.md`。
- 交接版实验文档：`experiments/03_rgb_roi/README.md`（路径相对仓库根目录）。

请勿把这里的标注框结果与 `yolo_image_train/` 的真实 YOLO 预测框结果混报。
