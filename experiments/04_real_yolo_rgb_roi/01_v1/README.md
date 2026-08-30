# 2026-08-04：真实 YOLO 框 image v1

## 简单总结

将 image v1 的基础 RGB ROI 特征直接换成 YOLO v0.3 预测框，用于测量检测噪声下的首个基线。BiGRU test
`ACC=0.5388`、`Frame-F1=0.4410`、`F1@0.25/0.5=0.1657/0.1543`，明显优于同轮另外两模型；
但单一 RGB 统计仍缺少框质量和时间平滑。

## 命令

```bash
python yolo_image_train/run_yolo_image_train.py \
  --skip-prepare \
  --versions v1 \
  --models ms_tcn asformer bigru
```

若尚未生成 `yolo_image_train/generated/actionmixed_yolo_v03/frames/`，去掉 `--skip-prepare` 并显式记录
`--yolo-conf 0.25 --yolo-iou 0.7 --imgsz 640`。

## 结果

| 模型 | val ACC / Frame-F1 / F1@0.5 | test ACC / Frame-F1 / F1@0.5 |
|---|---|---|
| MS-TCN | 0.1475 / 0.1010 / 0.0315 | 0.0439 / 0.0192 / 0.0000 |
| ASFormer | 0.3271 / 0.2771 / 0.0667 | 0.1908 / 0.1700 / 0.0114 |
| BiGRU | 0.6350 / 0.6206 / 0.1182 | 0.5388 / 0.4410 / 0.1543 |

文件：`image_train/train_image_v1.py`、`yolo_image_train/yolo_image_train_v1.md`、
`yolo_image_train/output_v1/image_train_v1.json`。下一步进入 v2 的质量感知 ROI/cache。
