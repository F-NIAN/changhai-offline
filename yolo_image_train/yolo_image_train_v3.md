# image_train_v3：面向动作时间段划分的训练报告

> 交接说明：本文是使用真实 YOLO v0.3 预测框生成的原始报告；本机绝对路径是运行快照。复现说明见 `experiments/04_real_yolo_rgb_roi/03_v3/README.md`。

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

- 数据集：`F:\暑期实习\offline-model\yolo_image_train\generated\actionmixed_yolo_v03`
- 输出目录：`F:\暑期实习\offline-model\yolo_image_train\output_v3`
- 训练轮数：`8`
- 设备：`cpu`
- ROI manifest：`F:\暑期实习\offline-model\yolo_image_train\output_v3\roi_manifest_p15`
- RGB cache：`F:\暑期实习\offline-model\yolo_image_train\output_v3\roi_rgb_v2_cache`

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
| `ms_tcn` | `v2+segment_postprocess` | `full_sequence` | 0.2877 | 0.1603 | 0.0939 | 0.0727 | 22 | 31 | 27 | 26.5000 |
| `asformer` | `business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.4798 | 0.5444 | 0.1061 | 0.0818 | 22 | 18 | 8 | 23.3571 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.5778 | 0.6237 | 0.1194 | 0.0982 | 22 | 21 | 5 | 7.7500 |

## 5. test 段级整体结果

| 模型 | 特征 | 训练方式 | ACC | Frame-F1 | F1@0.25 | F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | `v2+segment_postprocess` | `full_sequence` | 0.2924 | 0.0928 | 0.0514 | 0.0400 | 14 | 28 | 28 | 29.6667 |
| `asformer` | `business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.3439 | 0.3145 | 0.0971 | 0.0686 | 14 | 15 | 17 | 24.5000 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+segment_postprocess` | `full_sequence` | 0.5154 | 0.4533 | 0.1543 | 0.1543 | 14 | 15 | 11 | 13.9167 |

## 6. 每个模型选出的后处理参数

| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |
|---|---:|---:|---:|---:|
| `ms_tcn` | 5 | 12 | 6 | 0.0 |
| `asformer` | 5 | 12 | 10 | 0.45 |
| `bigru` | 1 | 8 | 10 | 0.45 |

## 7. v2 到 v3 的段级变化

| 模型 | val F1@0.25变化 | val F1@0.5变化 | test F1@0.25变化 | test F1@0.5变化 |
|---|---:|---:|---:|---:|
| `ms_tcn` | 0.0939 | 0.0727 | 0.0514 | 0.0400 |
| `asformer` | -0.0576 | -0.0576 | -0.0029 | -0.0314 |
| `bigru` | -0.0533 | -0.0685 | 0.0190 | 0.0190 |

## 8. 每个模型逐动作段识别情况

### ms_tcn

- 设计理由：MS-TCN 在 v2 的高维 RGB+滑窗配置退化明显，v3 回到低维 v2 bbox 特征，把改进重点放在段级后处理。
- 特征版本：`clean_bbox_v2_top1_impute+segment_postprocess_v3`
- 训练样本：`14`，最后一轮 loss：`1.2717`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 545 | 0.3780 | 0.4204 | 0.3981 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 991 | 0.2714 | 0.5886 | 0.3715 | 0.2727 | 0.2727 | 4 | 8 | 4 | 33.8333 |
| `long_brush_withdraw` | 217 | 215 | 0.2047 | 0.2028 | 0.2037 | 0.1515 | 0.0909 | 5 | 10 | 5 | 8.0000 |
| `short_brush_cleaning` | 376 | 331 | 0.2417 | 0.2128 | 0.2263 | 0.0455 | 0.0000 | 4 | 13 | 9 | 41.5000 |
| `flush` | 333 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 0 | 8 | - |
| `air_injection` | 209 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1 | - |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 297 | 0.5926 | 0.4422 | 0.5065 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 731 | 0.3187 | 0.4073 | 0.3576 | 0.2571 | 0.2000 | 6 | 8 | 2 | 29.6667 |
| `long_brush_withdraw` | 0 | 220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 11 | 11 | - |
| `short_brush_cleaning` | 111 | 209 | 0.0813 | 0.1532 | 0.1062 | 0.0000 | 0.0000 | 1 | 9 | 8 | - |
| `flush` | 261 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6 | 0 | 6 | - |
| `air_injection` | 115 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 | 0 | 1 | - |

### asformer

- 设计理由：ASFormer 保留 v2 中较有效的 business_priors+ROI RGB v2，使用段级后处理控制预测段数量和边界。
- 特征版本：`clean_bbox_v2_top1_impute+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，最后一轮 loss：`0.7293`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 710 | 0.2746 | 0.3980 | 0.3250 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 99 | 0.2929 | 0.0635 | 0.1043 | 0.0909 | 0.0000 | 4 | 3 | 1 | 20.0000 |
| `long_brush_withdraw` | 217 | 224 | 0.2679 | 0.2765 | 0.2721 | 0.0909 | 0.0909 | 5 | 5 | 0 | 14.5000 |
| `short_brush_cleaning` | 376 | 526 | 0.4525 | 0.6330 | 0.5277 | 0.1364 | 0.1364 | 4 | 6 | 2 | 19.7500 |
| `flush` | 333 | 304 | 0.8816 | 0.8048 | 0.8414 | 0.1212 | 0.0909 | 8 | 3 | 5 | 42.2500 |
| `air_injection` | 209 | 219 | 0.9543 | 1.0000 | 0.9766 | 0.0909 | 0.0909 | 1 | 1 | 0 | 5.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 383 | 0.4125 | 0.3970 | 0.4046 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 57 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6 | 1 | 5 | - |
| `long_brush_withdraw` | 0 | 127 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 5 | 5 | - |
| `short_brush_cleaning` | 111 | 387 | 0.0336 | 0.1171 | 0.0522 | 0.0000 | 0.0000 | 1 | 5 | 4 | - |
| `flush` | 261 | 326 | 0.6595 | 0.8238 | 0.7325 | 0.3429 | 0.2000 | 6 | 3 | 3 | 22.3333 |
| `air_injection` | 115 | 177 | 0.6497 | 1.0000 | 0.7877 | 0.1429 | 0.1429 | 1 | 1 | 0 | 31.0000 |

### bigru

- 设计理由：BiGRU 是 v1/v2 中最稳的主线；v3 为了快速验证段级后处理，先使用 full_sequence 训练，滑窗长训放到后续单独实验。
- 特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，最后一轮 loss：`0.7837`

#### val

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 629 | 0.4404 | 0.5653 | 0.4951 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 457 | 121 | 0.6694 | 0.1772 | 0.2803 | 0.0909 | 0.0909 | 4 | 4 | 0 | 5.5000 |
| `long_brush_withdraw` | 217 | 441 | 0.1769 | 0.3594 | 0.2371 | 0.1515 | 0.0909 | 5 | 7 | 2 | 11.2500 |
| `short_brush_cleaning` | 376 | 314 | 0.7771 | 0.6489 | 0.7072 | 0.1273 | 0.1273 | 4 | 4 | 0 | 12.5000 |
| `flush` | 333 | 366 | 0.8579 | 0.9429 | 0.8984 | 0.1364 | 0.0909 | 8 | 5 | 3 | 4.0000 |
| `air_injection` | 209 | 211 | 0.9905 | 1.0000 | 0.9952 | 0.0909 | 0.0909 | 1 | 1 | 0 | 1.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 475 | 0.4505 | 0.5377 | 0.4903 | 0.0000 | 0.0000 | 0 | 0 | 0 | - |
| `long_brush_insert` | 572 | 240 | 0.8042 | 0.3374 | 0.4754 | 0.2857 | 0.2857 | 6 | 4 | 2 | 11.5000 |
| `long_brush_withdraw` | 0 | 256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 5 | 5 | - |
| `short_brush_cleaning` | 111 | 58 | 0.2414 | 0.1261 | 0.1657 | 0.0000 | 0.0000 | 1 | 2 | 1 | - |
| `flush` | 261 | 249 | 0.8635 | 0.8238 | 0.8431 | 0.3429 | 0.3429 | 6 | 3 | 3 | 9.5000 |
| `air_injection` | 115 | 179 | 0.6425 | 1.0000 | 0.7823 | 0.1429 | 0.1429 | 1 | 1 | 0 | 32.0000 |

## 9. 结果分析

- val 上按段级目标最好的模型是 `bigru`：F1@0.5=0.0982，F1@0.25=0.1194，预测段数/真值段数=21/22，段数绝对误差=5。
- test 上按段级目标最好的模型是 `bigru`：F1@0.5=0.1543，F1@0.25=0.1543，预测段数/真值段数=15/14，段数绝对误差=11。
- 相对 v2，test F1@0.5 提升最大的是 `ms_tcn`，变化为 0.0400。
- 但 test 上存在泛化退化：`asformer`(-0.0314)。这说明只在单一 val split 搜索后处理参数容易过拟合，后续应使用交叉验证或更保守的参数选择。
- v3 的改进重点不是改变网络结构，而是在验证集上选择段级后处理参数；这直接服务于“段数更接近、边界更接近”的目标。
- 当前数据集较小，val 上搜索后处理参数可能过拟合，因此报告同时列出 test 结果作为外部参考。

## 10. 后续建议

- v3 已证明后处理参数对段级结果影响很大；下一步应把后处理搜索从单一 val split 扩展到交叉验证，降低过拟合。
- 如果目标继续是边界接近，建议新增 boundary head：从标注段起止点生成边界标签，让模型显式学习动作起止。
- 当前后处理仍是全类别共享参数，后续可做每类 min_segment/merge_gap，特别是 long_brush 与 flush 的持续时长差异很大。
- 保留 ROI manifest 作为稳定输入层；视觉特征下一版可换成冻结 DINOv2 embedding，但仍应以段级 F1 和边界误差作为主指标。
- 报告中的 test 结果比 val 更重要；如果某模型 val 提升但 test 退化，应优先认为后处理搜索过拟合。
