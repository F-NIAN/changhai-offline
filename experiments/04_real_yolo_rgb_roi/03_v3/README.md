# 2026-08-04：真实 YOLO 框 image v3

## 简单总结

在真实预测框上用 val 搜索段级后处理，再把同一参数应用于 test。BiGRU test
`ACC=0.5154`、`F1@0.25/0.5=0.1543/0.1543`；相较 v2 并未全面提升，说明只调平滑/短段/合并不能替代
更好的帧分类和 proxy 特征。

## 命令

```bash
python yolo_image_train/run_yolo_image_train.py \
  --skip-prepare \
  --versions v3 \
  --models ms_tcn asformer bigru
```

## 结果

| 模型 | val ACC / F1@0.25 / F1@0.5 | test ACC / F1@0.25 / F1@0.5 |
|---|---|---|
| MS-TCN | 0.2877 / 0.0939 / 0.0727 | 0.2924 / 0.0514 / 0.0400 |
| ASFormer | 0.4798 / 0.1061 / 0.0818 | 0.3439 / 0.0971 / 0.0686 |
| BiGRU | 0.5778 / 0.1194 / 0.0982 | 0.5154 / 0.1543 / 0.1543 |

文件：`image_train/v3/train_image_v3.py`、`image_train/common/segment_postprocess.py`、
`yolo_image_train/yolo_image_train_v3.md`、`yolo_image_train/output_v3/image_train_v3.json`。
下一步回到 raw accuracy 优先，并加入工具漏检时仍可计算的 proxy ROI。
