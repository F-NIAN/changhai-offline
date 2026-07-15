# ActionMixed 三模型最佳特征组合权重训练报告

- 数据集：`input\modelscope\lhh010__cleansight-ActionMixed`
- 输出目录：`output_actionmixed_best_models`
- 训练轮数：`3`
- 设备：`cpu`

## 总览

| 模型 | 最佳特征组合 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `v2` | `full_sequence` | 113 | 0.2433 | 0.0777 | 0.2153 | 0.1121 | 0.1000 | 0.0750 | `output_actionmixed_best_models\models\best_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors` | `full_sequence` | 121 | 0.6514 | 0.5610 | 0.5763 | 0.5556 | 0.1583 | 0.1250 | `output_actionmixed_best_models\models\best_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors` | `sliding_window` | 249 | 0.7482 | 0.5686 | 0.6404 | 0.5963 | 0.1917 | 0.1750 | `output_actionmixed_best_models\models\best_bigru_offline_segmenter.pt` |

## ms_tcn

- 选择理由：MS-TCN 在当前实验表中使用 v2 + full_sequence 时片段指标最高。
- 特征版本：`clean_bbox_v2_top1_impute`
- 训练样本：`10`，验证序列：`8`

| 类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `long_brush_insert` | 388 | 1272 | 0.2681 | 0.8789 | 0.4108 | 0.3750 | 0.2500 |
| `long_brush_withdraw` | 157 | 257 | 0.1206 | 0.1975 | 0.1498 | 0.1250 | 0.1250 |
| `short_brush_cleaning` | 344 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 318 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## asformer

- 选择理由：ASFormer 在当前实验表中使用 business_priors + full_sequence 时最好。
- 特征版本：`clean_bbox_v2_top1_impute+business_priors`
- 训练样本：`10`，验证序列：`8`

| 类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 207 | 0.3382 | 0.2174 | 0.2647 | 0.0000 | 0.0000 |
| `long_brush_insert` | 388 | 241 | 0.8755 | 0.5438 | 0.6709 | 0.2083 | 0.1250 |
| `long_brush_withdraw` | 157 | 178 | 0.2978 | 0.3376 | 0.3164 | 0.2500 | 0.2500 |
| `short_brush_cleaning` | 344 | 479 | 0.7182 | 1.0000 | 0.8360 | 0.2083 | 0.1250 |
| `flush` | 318 | 348 | 0.9138 | 1.0000 | 0.9550 | 0.1250 | 0.1250 |
| `air_injection` | 0 | 76 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## bigru

- 选择理由：BiGRU 使用 window_stats+business_priors + sliding_window 时为当前整体最优。
- 特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors`
- 训练样本：`83`，验证序列：`8`

| 类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 152 | 0.5000 | 0.2360 | 0.3207 | 0.0000 | 0.0000 |
| `long_brush_insert` | 388 | 448 | 0.8147 | 0.9407 | 0.8732 | 0.3750 | 0.3750 |
| `long_brush_withdraw` | 157 | 106 | 0.3868 | 0.2611 | 0.3118 | 0.2083 | 0.1250 |
| `short_brush_cleaning` | 344 | 478 | 0.7197 | 1.0000 | 0.8370 | 0.2500 | 0.2500 |
| `flush` | 318 | 345 | 0.9217 | 1.0000 | 0.9593 | 0.1250 | 0.1250 |
| `air_injection` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
