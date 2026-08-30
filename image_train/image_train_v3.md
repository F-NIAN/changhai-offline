# image_train_v3：面向动作时间段划分的训练报告

> 交接说明：本文保留当次脚本生成的原始结果；文中的本机绝对路径是运行快照。仓库根相对路径、复现命令和结果分析见 `experiments/03_rgb_roi/03_v3_segment_postprocess/README.md`。

## 1. v3 目标

v3 不再把 ACC 作为主要优化目标，而是优先让预测动作段和标注动作段更接近：段数尽量一致，F1@0.25/F1@0.5 更高，匹配段的起止边界误差更小。ACC 仍保留在报告中，但只作为参考指标。

## 2. 文件夹组织

```text
image_train/
  common/
    roi_cache.py                 # v2/v3 共享 ROI manifest 和 RGB cache
    segment_postprocess.py       # v3 新增，段级后处理和边界评估
  v3/
    train_image_v3.py            # v3 实验入口
  output_v3/
    feature_store_v2/
    roi_manifest_p15/
    roi_rgb_v2_cache/
    models/
    predictions/
    image_train_v3.json
  image_train_v3.md
```

三种时序模型主体仍统一复用根目录 `segmenter/*.py`，版本目录只管理特征组合、缓存和后处理策略。

## 3. v3 做了什么修改

- 数据集：`F:\暑期实习\offline-model\input\modelscope\lhh010__cleansight-ActionMixed\cleansight-ActionMixed`
- 输出目录：`image_train\output_v3`
- 训练轮数：`6`
- 设备：`cpu`
- ROI manifest：`image_train\output_v2\roi_manifest_p15`
- RGB cache：`image_train\output_v2\roi_rgb_v2_cache`

### 3.1 特征策略

| 模型 | 特征组合 | 是否使用 RGB | 训练方式 | dim |
|---|---|---|---|---:|
| `ms_tcn` | `v2+segment_postprocess` | `False` | `full_sequence` | 113 |
| `asformer` | `business_priors+roi_rgb_v2+segment_postprocess` | `True` | `full_sequence` | 331 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+segment_postprocess` | `True` | `full_sequence` | 459 |

### 3.2 段级后处理搜索

每个模型训练后先输出 raw softmax 概率，然后在 val split 上搜索：

- `prob_smooth`：概率时间平滑窗口。
- `min_segment`：最短非 idle 动作段长度，短段会被删除。
- `merge_gap`：同类动作段之间的短 idle 间隔合并阈值。
- `confidence_threshold`：低置信帧置为 idle。

搜索目标函数优先 `F1@0.5`，其次 `F1@0.25`，再惩罚预测段数和真值段数差距、匹配段边界误差。选出的同一套参数再用于 test。

## 4. 数据概况

| split | 序列数 | 帧数 |
|---|---:|---:|
| `test` | 7 | 1457 |
| `train` | 14 | 5993 |
| `val` | 11 | 2082 |

## 5. val 段级整体结果

| 模型 | 特征 | 训练方式 | ACC | Frame-F1 | F1@0.25 | F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | `v2+segment_postprocess` | `full_sequence` | 0.2738 | 0.1749 | 0.1194 | 0.0830 | 22 | 40 | 36 | 19.3750 |
| `asformer` | `business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.6849 | 0.7289 | 0.1939 | 0.1758 | 22 | 22 | 8 | 18.8333 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.6460 | 0.6749 | 0.1721 | 0.1721 | 22 | 25 | 13 | 9.8636 |

## 5. test 段级整体结果

| 模型 | 特征 | 训练方式 | ACC | Frame-F1 | F1@0.25 | F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | `v2+segment_postprocess` | `full_sequence` | 0.2780 | 0.1122 | 0.0333 | 0.0190 | 14 | 34 | 34 | 45.7500 |
| `asformer` | `business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.4475 | 0.2970 | 0.0924 | 0.0448 | 14 | 25 | 15 | 31.6667 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.3768 | 0.3479 | 0.1114 | 0.1114 | 14 | 21 | 13 | 12.7000 |

## 6. 每个模型选出的后处理参数

| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |
|---|---:|---:|---:|---:|
| `ms_tcn` | 5 | 8 | 6 | 0.0 |
| `asformer` | 1 | 8 | 0 | 0.3 |
| `bigru` | 9 | 8 | 6 | 0.45 |

## 7. v2 到 v3 的段级变化

| 模型 | val F1@0.25变化 | val F1@0.5变化 | test F1@0.25变化 | test F1@0.5变化 |
|---|---:|---:|---:|---:|
| `ms_tcn` | 0.1194 | 0.0830 | 0.0333 | 0.0190 |
| `asformer` | 0.0303 | 0.0364 | -0.0076 | -0.0552 |
| `bigru` | -0.0006 | 0.0055 | -0.0238 | -0.0238 |

## 8. 每个模型逐动作段识别情况

### ms_tcn

- 设计理由：MS-TCN 在 v2 的高维 RGB+滑窗配置退化明显，v3 回到低维 v2 bbox 特征，把改进重点放在段级后处理。
- 特征版本：`clean_bbox_v2_top1_impute+segment_postprocess_v3`
- 训练样本：`14`，最后一轮 loss：`1.4034`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 283 | 0.5300 | 0.3061 | 0.3881 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 1044 | 0.2193 | 0.5011 | 0.3051 | 0.2879 | 0.1061 | 4 | 11 | 7 | 30.7500 |
| `long_brush_withdraw` | 217 | 446 | 0.2735 | 0.5622 | 0.3680 | 0.2182 | 0.2182 | 5 | 14 | 9 | 10.0000 |
| `short_brush_cleaning` | 376 | 309 | 0.2233 | 0.1835 | 0.2015 | 0.0909 | 0.0909 | 4 | 15 | 11 | 2.0000 |
| `flush` | 333 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 0 | 8 | - |
| `air_injection` | 209 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1 | - |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 181 | 0.5138 | 0.2337 | 0.3212 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 736 | 0.3886 | 0.5000 | 0.4373 | 0.1667 | 0.0952 | 6 | 9 | 3 | 45.7500 |
| `long_brush_withdraw` | 0 | 231 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 12 | 12 | - |
| `short_brush_cleaning` | 111 | 309 | 0.0841 | 0.2342 | 0.1238 | 0.0000 | 0.0000 | 1 | 13 | 12 | - |
| `flush` | 261 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6 | 0 | 6 | - |
| `air_injection` | 115 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1 | - |

### asformer

- 设计理由：ASFormer 保留 v2 中较有效的 business_priors+ROI RGB v2，使用段级后处理控制预测段数量和边界。
- 特征版本：`clean_bbox_v2_top1_impute+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，最后一轮 loss：`1.2947`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 149 | 0.6040 | 0.1837 | 0.2817 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 530 | 0.7566 | 0.8775 | 0.8126 | 0.3636 | 0.2727 | 4 | 5 | 1 | 17.6250 |
| `long_brush_withdraw` | 217 | 252 | 0.3690 | 0.4286 | 0.3966 | 0.1818 | 0.1818 | 5 | 5 | 0 | 8.5000 |
| `short_brush_cleaning` | 376 | 611 | 0.6154 | 1.0000 | 0.7619 | 0.2727 | 0.2727 | 4 | 6 | 2 | 20.8750 |
| `flush` | 333 | 403 | 0.8263 | 1.0000 | 0.9049 | 0.0909 | 0.0909 | 8 | 4 | 4 | 2.0000 |
| `air_injection` | 209 | 137 | 0.9708 | 0.6364 | 0.7688 | 0.0606 | 0.0606 | 1 | 2 | 1 | 53.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 205 | 0.6537 | 0.3367 | 0.4444 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 387 | 0.8527 | 0.5769 | 0.6882 | 0.2381 | 0.0000 | 6 | 7 | 1 | 56.1667 |
| `long_brush_withdraw` | 0 | 293 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 4 | 4 | - |
| `short_brush_cleaning` | 111 | 402 | 0.2537 | 0.9189 | 0.3977 | 0.0714 | 0.0714 | 1 | 9 | 8 | 4.5000 |
| `flush` | 261 | 170 | 0.5059 | 0.3295 | 0.3991 | 0.1524 | 0.1524 | 6 | 5 | 1 | 8.5000 |
| `air_injection` | 115 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1 | - |

### bigru

- 设计理由：BiGRU 是 v1/v2 中最稳的主线；v3 为了快速验证段级后处理，先使用 full_sequence 训练，滑窗长训放到后续单独实验。
- 特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，最后一轮 loss：`0.6715`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 291 | 0.6942 | 0.4122 | 0.5173 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 233 | 0.7124 | 0.3632 | 0.4812 | 0.1818 | 0.1818 | 4 | 5 | 1 | 6.7500 |
| `long_brush_withdraw` | 217 | 225 | 0.3511 | 0.3641 | 0.3575 | 0.1818 | 0.1818 | 5 | 8 | 3 | 11.0000 |
| `short_brush_cleaning` | 376 | 742 | 0.4906 | 0.9681 | 0.6512 | 0.2545 | 0.2545 | 4 | 8 | 4 | 14.3750 |
| `flush` | 333 | 356 | 0.9129 | 0.9760 | 0.9434 | 0.1515 | 0.1515 | 8 | 3 | 5 | 1.2500 |
| `air_injection` | 209 | 235 | 0.8894 | 1.0000 | 0.9414 | 0.0909 | 0.0909 | 1 | 1 | 0 | 13.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 321 | 0.5016 | 0.4045 | 0.4478 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 90 | 1.0000 | 0.1573 | 0.2719 | 0.1429 | 0.1429 | 6 | 3 | 3 | 6.5000 |
| `long_brush_withdraw` | 0 | 181 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 5 | 5 | - |
| `short_brush_cleaning` | 111 | 484 | 0.2004 | 0.8739 | 0.3261 | 0.0714 | 0.0714 | 1 | 6 | 5 | 7.0000 |
| `flush` | 261 | 195 | 0.4410 | 0.3295 | 0.3772 | 0.2000 | 0.2000 | 6 | 6 | 0 | 7.2500 |
| `air_injection` | 115 | 186 | 0.6183 | 1.0000 | 0.7641 | 0.1429 | 0.1429 | 1 | 1 | 0 | 35.5000 |

## 9. 结果分析

- val 上按段级目标最好的模型是 `asformer`：F1@0.5=0.1758，F1@0.25=0.1939，预测段数/真值段数=22/22，段数绝对误差=8。
- test 上按段级目标最好的模型是 `bigru`：F1@0.5=0.1114，F1@0.25=0.1114，预测段数/真值段数=21/14，段数绝对误差=13。
- 相对 v2，test F1@0.5 提升最大的是 `ms_tcn`，变化为 0.0190。
- 但 test 上存在泛化退化：`asformer`(-0.0552)、`bigru`(-0.0238)。这说明只在单一 val split 搜索后处理参数容易过拟合，后续应使用交叉验证或更保守的参数选择。
- v3 的改进重点不是改变网络结构，而是在验证集上选择段级后处理参数；这直接服务于“段数更接近、边界更接近”的目标。
- 当前数据集较小，val 上搜索后处理参数可能过拟合，因此报告同时列出 test 结果作为外部参考。

## 10. 后续建议

- v3 已证明后处理参数对段级结果影响很大；下一步应把后处理搜索从单一 val split 扩展到交叉验证，降低过拟合。
- 如果目标继续是边界接近，建议新增 boundary head：从标注段起止点生成边界标签，让模型显式学习动作起止。
- 当前后处理仍是全类别共享参数，后续可做每类 min_segment/merge_gap，特别是 long_brush 与 flush 的持续时长差异很大。
- 保留 ROI manifest 作为稳定输入层；视觉特征下一版可换成冻结 DINOv2 embedding，但仍应以段级 F1 和边界误差作为主指标。
- 报告中的 test 结果比 val 更重要；如果某模型 val 提升但 test 退化，应优先认为后处理搜索过拟合。
