# 2026-07-22：image_train v3 段级后处理

## 简单总结

v2 的帧级指标提升，但动作段数量和边界仍不稳定。本次训练后在 val 上搜索概率平滑、最短段、同类段间隔合并
和置信度阈值，再把固定参数用于 test。val 上 ASFormer 的 `F1@0.25/0.5=0.1939/0.1758` 最好；test 上
BiGRU 为 `0.1114/0.1114`。后处理改善部分边界，但泛化不稳定，说明应分开报告 raw 与 post 指标。

## 脚本、输入与输出

- 训练：`image_train/v3/train_image_v3.py`。
- 后处理：`image_train/common/segment_postprocess.py`。
- 复用：`image_train/output_v2/roi_manifest_p15/` 和 `roi_rgb_v2_cache/`。
- 报告：`image_train/image_train_v3.md`。
- 结果：`image_train/output_v3/image_train_v3.json`、`image_train/output_v3/predictions/`。

## 命令

```bash
python image_train/v3/train_image_v3.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --out-dir image_train/output_v3 \
  --feature-store-dir image_train/output_v2/feature_store_v2 \
  --roi-manifest-dir image_train/output_v2/roi_manifest_p15 \
  --rgb-cache-dir image_train/output_v2/roi_rgb_v2_cache \
  --epochs 6 \
  --models ms_tcn asformer bigru \
  --report-path image_train/image_train_v3.md
```

## 结果与分析

| 模型 | val ACC / F1@0.25 / F1@0.5 | test ACC / F1@0.25 / F1@0.5 |
|---|---|---|
| MS-TCN | 0.2738 / 0.1194 / 0.0830 | 0.2780 / 0.0333 / 0.0190 |
| ASFormer | 0.6849 / 0.1939 / 0.1758 | 0.4475 / 0.0924 / 0.0448 |
| BiGRU | 0.6460 / 0.1721 / 0.1721 | 0.3768 / 0.1114 / 0.1114 |

val 最优不能稳定迁移到 test，尤其 ASFormer 下降明显。下一步应先以 raw frame accuracy 训练，记录 early stopping，
再单独选择后处理；同时加入检测缺失时可用的 proxy ROI，减少对单一目标框的依赖。
