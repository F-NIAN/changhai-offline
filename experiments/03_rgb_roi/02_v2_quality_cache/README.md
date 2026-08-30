# 2026-07-21：image_train v2 ROI 质量感知与缓存

## 简单总结

v1 每次实验都要重复定位/裁剪 ROI，且模型无法区分真实框和低质量框。本次抽出 ROI manifest 和 RGB cache，
每个关键 ROI 使用 10 维质量/颜色特征，再追加中心 5 帧均值和帧间 delta，共 210 维。BiGRU val
`ACC=0.6191`、`Frame-F1=0.6698`，test `ACC=0.5278`、`Frame-F1=0.4757`；ASFormer val 也明显改善。

## 文件与数据流

```text
image_train/common/roi_cache.py
  -> image_train/output_v2/roi_manifest_p15/
  -> image_train/output_v2/roi_rgb_v2_cache/
  -> image_train/v2/train_image_v2.py
  -> image_train/output_v2/{models,predictions,image_train_v2.json}
```

关键 ROI：两只手、短刷、针筒、气枪、远端口、刷头外露。单 ROI 原始特征包含 valid/conf/area/aspect、RGB
均值、亮度均值/方差和饱和度代理。

## 命令

```bash
python image_train/v2/train_image_v2.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --out-dir image_train/output_v2 \
  --epochs 8 \
  --padding 0.15 \
  --models ms_tcn asformer bigru \
  --report-path image_train/image_train_v2.md
```

通过 `--roi-manifest-dir`、`--rgb-cache-dir` 可复用已有预处理；只有数据/框/padding 改变时才使用
`--force-cache`。

## 结果与分析

| 模型 | val ACC / Frame-F1 / F1@0.5 | test ACC / Frame-F1 / F1@0.5 |
|---|---|---|
| MS-TCN | 0.1614 / 0.0016 / 0.0000 | 0.1764 / 0.0158 / 0.0000 |
| ASFormer | 0.5879 / 0.6575 / 0.1394 | 0.2505 / 0.3072 / 0.1000 |
| BiGRU | 0.6191 / 0.6698 / 0.1667 | 0.5278 / 0.4757 / 0.1352 |

质量感知和缓存提高了复现效率，BiGRU/ASFormer 获益，但 MS-TCN 在 459 维滑窗设置下崩溃。下一步对 MS-TCN
回退低维输入，并把模型输出后的段边界优化单独建模，避免把所有改进都堆在输入维度上。

原始文档：`image_train/image_train_v2.md`；结果 JSON：`image_train/output_v2/image_train_v2.json`。
