# 2026-07-21：image_train v1 基础 RGB ROI

## 简单总结

当时最佳结构化特征仍无法表达工具纹理、亮度和局部外观。本次在每帧对象框上裁 ROI，提取低维 RGB 统计并拼接
到三模型原有最佳特征。流程成功跑通，BiGRU 在 val 上 `ACC=0.5711`、`Frame-F1=0.6104`，test 上
`ACC=0.3782`、`Frame-F1=0.4056`；泛化差距说明简单颜色统计容易受场景/光照影响。

## 设计与文件

- 脚本：`image_train/train_image_v1.py`。
- 报告：`image_train/image_train_v1.md`。
- 结构化结果：`image_train/output_v1/image_train_v1.json`。
- 输出：`image_train/output_v1/{models,predictions,rgb_cache}/`。
- 特征：RGB 均值、亮度、饱和度代理、颜色直方图等；各模型继续使用原结构化最佳 recipe。

## 命令

```bash
python image_train/train_image_v1.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --out-dir image_train/output_v1 \
  --epochs 3 \
  --padding 0.15 \
  --models ms_tcn asformer bigru \
  --report-path image_train/image_train_v1.md
```

`--padding` 控制框外扩比例；`--force-cache` 强制重建 RGB 缓存；调参时先复用缓存，避免重复解码图片。

## 结果与分析

| 模型 | val ACC / Frame-F1 / F1@0.25 | test ACC / Frame-F1 / F1@0.25 |
|---|---|---|
| MS-TCN | 0.2195 / 0.1724 / 0.1072 | 0.2608 / 0.1730 / 0.0471 |
| ASFormer | 0.2286 / 0.1120 / 0.0467 | 0.0885 / 0.0251 / 0.0286 |
| BiGRU | 0.5711 / 0.6104 / 0.1345 | 0.3782 / 0.4056 / 0.1245 |

BiGRU 最稳，但 val/test 落差明显。下一步需要把 ROI 对齐和缓存独立出来，加入 valid/conf/area 等质量信号，
并对图像特征做时间平滑和 delta，而不是只拼接静态颜色统计。
