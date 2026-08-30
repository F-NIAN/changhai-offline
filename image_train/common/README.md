# image_train 共享组件

- `roi_cache.py`：建立稳定 ROI manifest，生成/读取 RGB 特征缓存。
- `proxy_roi.py`：检测目标缺失时构造洗槽、通道、扩手框、环形手框和交互 union 等 proxy ROI。
- `segment_postprocess.py`：概率平滑、短段删除、同类段合并、低置信置 idle，以及段级指标/边界误差。

这些模块由 v2-v4 复用，不包含独立训练入口。调用方式和参数由各版本 README/脚本负责说明。
