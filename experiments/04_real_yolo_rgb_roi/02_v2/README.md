# 2026-08-04：真实 YOLO 框 image v2

## 简单总结

在真实预测框上加入 ROI valid/conf/area 等质量特征、中心窗口均值和 delta。BiGRU test
`ACC=0.5882`、`Frame-F1=0.6005`、`F1@0.25/0.5=0.1876/0.1543`，比 v1 的帧级表现更稳；
MS-TCN 仍不适应 459 维滑窗输入。

## 命令

```bash
python yolo_image_train/run_yolo_image_train.py \
  --skip-prepare \
  --versions v2 \
  --models ms_tcn asformer bigru
```

## 结果

| 模型 | val ACC / Frame-F1 / F1@0.5 | test ACC / Frame-F1 / F1@0.5 |
|---|---|---|
| MS-TCN | 0.3103 / 0.1019 / 0.0485 | 0.3624 / 0.0928 / 0.0286 |
| ASFormer | 0.4640 / 0.5220 / 0.0719 | 0.3095 / 0.3221 / 0.0667 |
| BiGRU | 0.4284 / 0.3995 / 0.0537 | 0.5882 / 0.6005 / 0.1543 |

文件：`image_train/v2/train_image_v2.py`、`image_train/common/roi_cache.py`、
`yolo_image_train/yolo_image_train_v2.md`、`yolo_image_train/output_v2/image_train_v2.json`。
下一步回退 MS-TCN 维度，并在 val 选择段后处理参数。
