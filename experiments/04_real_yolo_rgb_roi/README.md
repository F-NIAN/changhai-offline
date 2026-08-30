# 2026-08-04：用真实 YOLO 预测框复跑 RGB/ROI v1-v4

## 简单总结

`image_train/` 使用数据集已有框，不能反映检测误差。本次解压 `clean-large-v0.3` 与 `clean-small-v0.3`，
对 ActionMixed 图片推理，将两组互补类别映射成统一 8 类 `frames/*.txt`，再原样复跑 image v1-v4。最终 v4
BiGRU test raw `ACC=0.7028`、`Frame-F1=0.6658`，post `F1@0.25/0.5=0.2162/0.2019`，是这条真实检测框复验线的最佳结果。

## 组织结构

| 版本 | 改动 | 详情 |
|---|---|---|
| v1 | 基础 RGB ROI + 真实预测框 | [01](01_v1/README.md) |
| v2 | 质量感知 ROI cache + 真实预测框 | [02](02_v2/README.md) |
| v3 | 段后处理 + 真实预测框 | [03](03_v3/README.md) |
| v4 | proxy ROI + 集中配置 + 真实预测框 | [04](04_v4/README.md) |

## 共同输入与生成流程

```text
ActionMixed images
  -> yolo/clean-large-v0.3.zip（hand / scope control / scope mid）
  -> yolo/clean-small-v0.3.zip（syringe / air gun / distal / short brush / tip）
  -> yolo_image_train/generated/actionmixed_yolo_v03/frames/{split}
  -> 复用 labels/{split}
  -> image_train v1-v4
```

准备脚本：`yolo_image_train/common/prepare_yolo_actionmixed.py`；统一编排：
`yolo_image_train/run_yolo_image_train.py`。

## 完整命令

```bash
python yolo_image_train/run_yolo_image_train.py \
  --source-dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --generated-dataset-root yolo_image_train/generated/actionmixed_yolo_v03 \
  --output-root yolo_image_train \
  --yolo-dir yolo \
  --extract-dir yolo_image_train/weights \
  --yolo-conf 0.25 \
  --yolo-iou 0.7 \
  --imgsz 640 \
  --batch 8 \
  --versions v1 v2 v3 v4 \
  --device 0
```

关键参数：`--skip-prepare` 复用检测结果；`--prepare-only` 只生成数据；`--limit-frames` 做小规模调试；
`--force-extract` 重解权重；`--force-cache` 重建 ROI 缓存；`--dry-run` 只打印命令。

## 共同限制与下一步

- 仍使用切分片段，不是完整手术视频；相邻帧高度相关。
- YOLO 阈值变化必须重建 `frames/`，并记录 conf/IoU/imgsz。
- v4 的高结果可能同时来自检测框分布变化和训练随机性，需多 seed 重复；不能据此声称真实 YOLO 一定优于标注框。
- 下一步应固定 split/fingerprint、增加完整视频评估，并比较 tracker 后的稳定框特征。

详细生成说明：`yolo_image_train/README.md`；运行摘要：`yolo_image_train/yolo_image_train_run_summary.json`。
