# image_train_v1：ActionMixed RGB 图像特征首版改进训练报告

> 交接说明：本文是使用真实 YOLO v0.3 预测框生成的原始报告；本机绝对路径是运行快照。复现说明见 `experiments/04_real_yolo_rgb_roi/01_v1/README.md`。

## 1. 本次改进目标

本次在不改动三种时序模型主体结构的前提下，新增 `image_train` 实验目录，把 ActionMixed 已下载的 JPEG 帧图像 RGB 信息加入训练。实现方式是复用当前三种模型各自最好的检测框特征 recipe，再拼接轻量 RGB ROI 外观特征，验证第一版图像信息是否能给动作分割带来收益。

## 2. 改了什么

- 新增训练脚本：`F:\暑期实习\offline-model\image_train\train_image_v1.py`
- 数据集：`F:\暑期实习\offline-model\yolo_image_train\generated\actionmixed_yolo_v03`
- 输出目录：`F:\暑期实习\offline-model\yolo_image_train\output_v1`
- 训练轮数：`3`
- 设备：`cpu`
- RGB ROI 槽位：`10` 个，分别为 `hand_top1, hand_top2, short_brush, long_brush, syringe, air_gun, scope_control_body, scope_mid_section, scope_distal_end, brush_tip_out`。
- 每个 ROI 槽位 RGB 特征：`19` 维，总 RGB 维度：`190`。

### 2.1 RGB ROI 特征如何提取

1. 对每个视频片段读取 `frames/{split}/{video}.mp4-{frame_id}.txt` 中的 YOLO bbox，并找到同名 `images/{split}/{video}.mp4-{frame_id}.jpg`。
2. 每帧按对象选择 ROI：`hand` 保留 top-2 两个独立槽位，其它对象保留 top-1。排序分数为 `confidence * sqrt(width * height)`。
3. 对 bbox 加 `15%` padding 后在原图数组上直接切片 ROI，不做 resize，直接统计 ROI 内 RGB 分布。
4. 每个 ROI 只提取 RGB 信息：`valid`、RGB 三通道均值、RGB 三通道标准差、RGB 三通道各 4 个归一化直方图 bin。
5. 当前帧某对象缺失或图片缺失时，该 ROI 槽位全 0；是否真实检测到、missing_age、imputed 仍由原 v2 检测框特征表达。

### 2.2 三个模型现在使用的特征

| 模型 | 原最佳特征 | 新增图像特征 | 当前总特征 | 训练方式 |
|---|---|---|---:|---|
| `ms_tcn` | `v2` | `rgb_roi_stats_v1` (190维) | 303 | `full_sequence` |
| `asformer` | `business_priors` | `rgb_roi_stats_v1` (190维) | 311 | `full_sequence` |
| `bigru` | `window_stats+business_priors` | `rgb_roi_stats_v1` (190维) | 439 | `sliding_window` |

特征分类说明：

- v2 检测框结构特征：对象 candidate_count、present、confidence、bbox 中心、面积、speed、missing_age、imputed、对象间 distance/delta、时间位置编码。
- window_stats：对动作相关列追加中心窗口统计，离线模型可以使用当前帧前后的上下文。
- business_priors：基于业务对象关系追加弱先验分数，例如短刷靠近控制部、针筒/气枪靠近远端口、长刷刷头相对远端口的运动方向。
- rgb_roi_stats_v1：本次新增的图片 RGB 外观特征，描述手、刷子、针筒、气枪、内镜部位等 ROI 的局部颜色分布。

## 3. 数据概况

| split | 序列数 | 帧数 |
|---|---:|---:|
| `test` | 7 | 1457 |
| `train` | 14 | 5993 |
| `val` | 11 | 2082 |

## 4. val 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `v2+rgb_roi` | `full_sequence` | 303 | 0.1475 | 0.1114 | 0.1509 | 0.1010 | 0.0497 | 0.0315 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+rgb_roi` | `full_sequence` | 311 | 0.3271 | 0.2891 | 0.4126 | 0.2771 | 0.0758 | 0.0667 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+rgb_roi` | `sliding_window` | 439 | 0.6350 | 0.6923 | 0.6548 | 0.6206 | 0.1273 | 0.1182 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_bigru_offline_segmenter.pt` |

## 4. test 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `v2+rgb_roi` | `full_sequence` | 303 | 0.0439 | 0.0838 | 0.0108 | 0.0192 | 0.0000 | 0.0000 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+rgb_roi` | `full_sequence` | 311 | 0.1908 | 0.2274 | 0.2700 | 0.1700 | 0.0495 | 0.0114 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+rgb_roi` | `sliding_window` | 439 | 0.5388 | 0.4682 | 0.4483 | 0.4410 | 0.1657 | 0.1543 | `F:\暑期实习\offline-model\yolo_image_train\output_v1\models\image_v1_bigru_offline_segmenter.pt` |

## 5. 每个模型逐动作识别情况（以帧数为单位）

## ms_tcn

- 使用特征版本：`clean_bbox_v2_top1_impute+rgb_roi_stats_v1`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.6029`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 149 | 75 | 74 | 415 | 0.5034 | 0.1531 | 0.2347 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 169 | 43 | 126 | 414 | 0.2544 | 0.0941 | 0.1374 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 217 | 1316 | 81 | 1235 | 136 | 0.0616 | 0.3733 | 0.1057 | 0.1273 | 0.1273 |
| `short_brush_cleaning` | 376 | 448 | 108 | 340 | 268 | 0.2411 | 0.2872 | 0.2621 | 0.1212 | 0.0303 |
| `flush` | 333 | 0 | 0 | 0 | 333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 79 | 33 | 46 | 365 | 0.4177 | 0.0829 | 0.1384 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 74 | 31 | 43 | 541 | 0.4189 | 0.0542 | 0.0960 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 0 | 1099 | 0 | 1099 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 205 | 0 | 205 | 111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 0 | 0 | 0 | 261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## asformer

- 使用特征版本：`clean_bbox_v2_top1_impute+business_priors+rgb_roi_stats_v1`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.6367`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 174 | 93 | 81 | 397 | 0.5345 | 0.1898 | 0.2801 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 149 | 0 | 149 | 457 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 217 | 1051 | 215 | 836 | 2 | 0.2046 | 0.9908 | 0.3391 | 0.2424 | 0.2424 |
| `short_brush_cleaning` | 376 | 174 | 140 | 34 | 236 | 0.8046 | 0.3723 | 0.5091 | 0.0909 | 0.0909 |
| `flush` | 333 | 534 | 233 | 301 | 100 | 0.4363 | 0.6997 | 0.5375 | 0.0455 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 26 | 24 | 2 | 374 | 0.9231 | 0.0603 | 0.1132 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 151 | 95 | 56 | 477 | 0.6291 | 0.1661 | 0.2628 | 0.0476 | 0.0000 |
| `long_brush_withdraw` | 0 | 612 | 0 | 612 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 500 | 111 | 389 | 0 | 0.2220 | 1.0000 | 0.3633 | 0.0571 | 0.0571 |
| `flush` | 261 | 168 | 48 | 120 | 213 | 0.2857 | 0.1839 | 0.2238 | 0.1429 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## bigru

- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+rgb_roi_stats_v1`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.8480`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 501 | 225 | 276 | 265 | 0.4491 | 0.4592 | 0.4541 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 799 | 434 | 365 | 23 | 0.5432 | 0.9497 | 0.6911 | 0.3182 | 0.3182 |
| `long_brush_withdraw` | 217 | 13 | 5 | 8 | 212 | 0.3846 | 0.0230 | 0.0435 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 376 | 201 | 137 | 64 | 239 | 0.6816 | 0.3644 | 0.4749 | 0.0909 | 0.0909 |
| `flush` | 333 | 351 | 312 | 39 | 21 | 0.8889 | 0.9369 | 0.9123 | 0.1364 | 0.0909 |
| `air_injection` | 209 | 217 | 209 | 8 | 0 | 0.9631 | 1.0000 | 0.9812 | 0.0909 | 0.0909 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 472 | 184 | 288 | 214 | 0.3898 | 0.4623 | 0.4230 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 359 | 298 | 61 | 274 | 0.8301 | 0.5210 | 0.6402 | 0.3429 | 0.3429 |
| `long_brush_withdraw` | 0 | 49 | 0 | 49 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 181 | 0 | 181 | 111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 213 | 188 | 25 | 73 | 0.8826 | 0.7203 | 0.7932 | 0.3429 | 0.2857 |
| `air_injection` | 115 | 183 | 115 | 68 | 0 | 0.6284 | 1.0000 | 0.7718 | 0.1429 | 0.1429 |

## 6. 结果分析

- val 上按 F1@0.25 排名最高的是 `bigru`，F1@0.25=0.1273，F1@0.5=0.1182，Frame-F1=0.6206。
- test 上按 F1@0.25 排名最高的是 `bigru`，F1@0.25=0.1657，F1@0.5=0.1543，Frame-F1=0.4410。
- 本版 RGB 特征是轻量统计特征，不训练图像 backbone，也不引入端到端视觉模型；它主要验证“ROI 局部外观是否有信号”，工程风险低，但表达能力明显弱于 DINOv2/VideoMAE。
- 由于 train/val/test 序列数量较少，单次 3 epoch 结果波动会比较大，应主要看 per-class recall 和 segment F1 是否出现稳定方向，而不是只看单一 ACC。

## 7. 后续修改建议

- 把 `rgb_roi_stats_v1` 与 baseline 最佳报告做横向对比，重点看 F1@0.25/F1@0.5 和每类 recall，判断 RGB 是否对边界和漏检有帮助。
- 如果 RGB 统计特征有收益，下一版改为冻结 DINOv2 ROI embedding，并用 PCA/Linear 压到每槽 32 或 64 维。
- 加入质量感知融合：用 present/conf/missing_age/imputed/candidate_count 学一个 gate，而不是简单 concat。
- 对 `flush` 和 `air_injection` 单独增强 syringe/air_gun ROI，并检查两类误检混淆；这两类最依赖外观区分。
- 正式结论前至少跑 20-100 epoch，并重复 3 个随机种子；当前 3 epoch 只能视为首版工程验证。
