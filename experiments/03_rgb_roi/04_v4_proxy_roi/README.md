# 2026-07-28：image_train v4 proxy ROI 与集中配置

## 简单总结

v3 把训练和段后处理目标混在一起，且工具漏检时没有图像替代区域。本次训练以 raw frame accuracy 为主，
后处理独立搜索；新增洗槽/通道、扩手框、手部环形和手-控制部位 union 等 proxy ROI，并把训练/后处理参数集中到
JSON。BiGRU test raw `ACC=0.5964`、`Frame-F1=0.5037`，post `F1@0.25/0.5=0.1686/0.1686`，
是标注框 RGB/ROI 主线的推荐版本。

## 配置和代码

- 配置：`image_train/v4/configs/image_train_v4.json`。
- 入口：`image_train/v4/train_image_v4.py`。
- ROI：`image_train/common/roi_cache.py`、`image_train/common/proxy_roi.py`。
- 后处理：`image_train/common/segment_postprocess.py`。
- 文档/结果：`image_train/image_train_v4.md`、`image_train/output_v4/image_train_v4.json`。

## 命令

```bash
python image_train/v4/train_image_v4.py \
  --config image_train/v4/configs/image_train_v4.json \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --out-dir image_train/output_v4 \
  --models ms_tcn asformer bigru
```

常用覆盖：`--epochs`、`--lr`、`--scheduler`、`--patience`、`--train-mode`、`--hand-expand`；
`--post-prob-smooth`、`--post-min-segment`、`--post-merge-gap`、`--post-confidence-threshold` 接受逗号列表。

## 结果

| 模型 | val raw ACC / post F1@0.5 | test raw ACC / post F1@0.5 |
|---|---|---|
| MS-TCN | 0.3079 / 0.0364 | 0.3370 / 0.0143 |
| ASFormer | 0.6647 / 0.1870 | 0.5202 / 0.1400 |
| BiGRU | 0.7022 / 0.1667 | 0.5964 / 0.1686 |

分析：proxy ROI 对 BiGRU/ASFormer 有帮助，MS-TCN 仍不稳定；高维 flat concat 不能显式表示“哪组特征对应哪类
动作”。下一步优先做 group-aware/gated fusion，并用真实 YOLO 预测框复验；设计依据和下一版方案分别在
`image_train/image_train_v4future.md`、`image_train/image_train_v5future.md`。
