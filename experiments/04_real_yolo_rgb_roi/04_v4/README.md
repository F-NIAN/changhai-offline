# 2026-08-04：真实 YOLO 框 image v4

## 简单总结

最终版本把训练目标改为 raw frame accuracy，后处理独立选择；加入洗槽/通道、扩手框和交互 union 等 proxy ROI，
并使用集中配置、scheduler 与 early stopping。BiGRU test raw `ACC=0.7028`、`Frame-F1=0.6658`，post
`F1@0.25/0.5=0.2162/0.2019`，为真实 YOLO 框 v1-v4 中最佳。

## 命令

```bash
python yolo_image_train/run_yolo_image_train.py \
  --skip-prepare \
  --versions v4 \
  --models ms_tcn asformer bigru
```

直接调用底层脚本时：

```bash
python image_train/v4/train_image_v4.py \
  --config image_train/v4/configs/image_train_v4.json \
  --dataset-root yolo_image_train/generated/actionmixed_yolo_v03 \
  --out-dir yolo_image_train/output_v4 \
  --report-path yolo_image_train/yolo_image_train_v4.md
```

## 结果

| 模型 | val raw ACC / post F1@0.5 | test raw ACC / post F1@0.5 |
|---|---|---|
| MS-TCN | 0.4467 / 0.0812 | 0.1517 / 0.0000 |
| ASFormer | 0.4529 / 0.1115 | 0.3109 / 0.0543 |
| BiGRU | 0.7037 / 0.1216 | 0.7028 / 0.2019 |

文件：`image_train/v4/train_image_v4.py`、`image_train/v4/configs/image_train_v4.json`、
`yolo_image_train/yolo_image_train_v4.md`、`yolo_image_train/output_v4/image_train_v4.json`。

分析：BiGRU 明显领先，但单次小数据结果不足以判断泛化。后续至少跑 3 个 seed，保留 val 选参/test 一次验收，
并尝试 track-aware 特征和 group-aware/gated fusion。
