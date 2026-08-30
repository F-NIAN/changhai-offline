# image_train_v2：ActionMixed ROI RGB 质量感知特征训练报告

> 交接说明：本文是使用真实 YOLO v0.3 预测框生成的原始报告；本机绝对路径是运行快照。复现说明见 `experiments/04_real_yolo_rgb_roi/02_v2/README.md`。

## 1. v2 目标

v2 的主目标是提高帧级准确率，同时把 ROI 预处理抽成可复用层，方便后续 v3/v4 继续使用同一套 ROI 对齐结果。模型架构不在每个版本里复制，三种模型仍统一复用仓库根目录下的 `segmenter/*.py`。

## 2. 文件夹组织设计

当前建议的 `image_train` 组织方式如下：

```text
image_train/
  common/
    roi_cache.py              # 通用 ROI manifest / RGB feature cache
  v2/
    train_image_v2.py         # v2 实验入口，只放本版本 feature recipe 和报告逻辑
  output_v2/
    feature_store_v2/         # 复用原 bbox FeatureStore
    roi_manifest_p15/         # 所有帧的 ROI 槽位、bbox、图片路径、置信度
    roi_rgb_v2_cache/         # 基于 manifest 的 v2 RGB 特征缓存
    models/                   # 三种模型权重
    predictions/              # val/test soft label 输出
    image_train_v2.json       # 结构化实验结果
  image_train_v2.md           # 人读报告
```

这样后续版本只需要新增 `v3/train_image_v3.py` 和新的 feature cache，不需要复制三种模型实现。ROI manifest 是稳定层，DINOv2、VideoMAE、gated fusion 都可以继续基于它生成不同版本缓存。

## 3. v2 具体改进

- 数据集：`F:\暑期实习\offline-model\yolo_image_train\generated\actionmixed_yolo_v03`
- 输出目录：`F:\暑期实习\offline-model\yolo_image_train\output_v2`
- 训练轮数：`8`
- 设备：`cpu`
- ROI padding：`0.15`
- ROI manifest：`F:\暑期实习\offline-model\yolo_image_train\output_v2\roi_manifest_p15`
- RGB v2 cache：`F:\暑期实习\offline-model\yolo_image_train\output_v2\roi_rgb_v2_cache`

### 3.1 ROI manifest

v2 先对所有 train/val/test 帧预处理 ROI manifest。每个视频片段一个 `.npz`，包含：`frame_numbers`、`image_paths`、`yolo_paths`、`slots`、`boxes[T,S,4]`、`valid[T,S]`、`conf[T,S]`。slot 规则仍是 `hand_top1/hand_top2` 和其它对象 top-1。

### 3.2 ROI RGB v2 特征

v2 只对关键 ROI 槽位提取图像特征：`hand_top1, hand_top2, short_brush, syringe, air_gun, scope_distal_end, brush_tip_out`。

每个 ROI 原始特征 10 维：`valid`、`conf`、`area`、`aspect`、`r_mean`、`g_mean`、`b_mean`、`brightness_mean`、`brightness_std`、`saturation_proxy`。

随后追加两类时序增强：中心窗口均值 `center_mean_w5` 和相邻帧差分 `delta`。因此 RGB v2 维度为 `7 slots * 10 dims * 3 blocks = 210`。

## 4. 三个模型当前特征

| 模型 | base 特征 | 新增图像特征 | 总 dim | 训练方式 |
|---|---|---|---:|---|
| `ms_tcn` | `window_stats+business_priors` | `roi_rgb_v2_quality_smooth` (210维) | 459 | `sliding_window` |
| `asformer` | `business_priors` | `roi_rgb_v2_quality_smooth` (210维) | 331 | `full_sequence` |
| `bigru` | `window_stats+business_priors` | `roi_rgb_v2_quality_smooth` (210维) | 459 | `sliding_window` |

特征分类：

- bbox/检测结构特征：present、confidence、center、area、speed、missing_age、imputed、对象距离和距离变化。
- 离线窗口统计：对关键检测列做中心窗口均值，利用前后帧上下文提高稳定性。
- 业务先验：根据短刷/针筒/气枪/长刷与内镜部位的关系构造弱先验分数。
- ROI RGB v2：关键 ROI 的低维颜色、亮度、饱和度和质量特征，并追加局部时间平滑与变化量。

## 5. 数据概况

| split | 序列数 | 帧数 |
|---|---:|---:|
| `test` | 7 | 1457 |
| `train` | 14 | 5993 |
| `val` | 11 | 2082 |

## 6.1. val 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.3103 | 0.0792 | 0.1673 | 0.1019 | 0.0697 | 0.0485 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+roi_rgb_v2` | `full_sequence` | 331 | 0.4640 | 0.5152 | 0.5636 | 0.5220 | 0.0892 | 0.0719 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.4284 | 0.4396 | 0.4641 | 0.3995 | 0.0719 | 0.0537 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_bigru_offline_segmenter.pt` |

## 6.2. test 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.3624 | 0.0863 | 0.1003 | 0.0928 | 0.0400 | 0.0286 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+roi_rgb_v2` | `full_sequence` | 331 | 0.3095 | 0.2985 | 0.4029 | 0.3221 | 0.0952 | 0.0667 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.5882 | 0.6777 | 0.6048 | 0.6005 | 0.1876 | 0.1543 | `F:\暑期实习\offline-model\yolo_image_train\output_v2\models\image_v2_bigru_offline_segmenter.pt` |

## 7. v1 到 v2 的准确率变化

| 模型 | v1 val ACC | v2 val ACC | val 变化 | v1 test ACC | v2 test ACC | test 变化 |
|---|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | 0.2195 | 0.3103 | 0.0908 | 0.2608 | 0.3624 | 0.1016 |
| `asformer` | 0.2286 | 0.4640 | 0.2354 | 0.0885 | 0.3095 | 0.2210 |
| `bigru` | 0.5711 | 0.4284 | -0.1427 | 0.3782 | 0.5882 | 0.2100 |

从准确率目标看，v2 对 `bigru` 和 `asformer` 有明显收益；`ms_tcn` 对当前高维窗口增强更敏感，本版不建议作为主线。

## 8. 每个模型逐动作识别情况（以帧数为单位）

### ms_tcn

- 设计理由：v2 以准确率为目标，给 ms_tcn 增加中心窗口统计和滑窗样本数，缓解原 full_sequence 样本过少。
- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.5754`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 767 | 278 | 489 | 212 | 0.3625 | 0.5673 | 0.4423 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 1168 | 355 | 813 | 102 | 0.3039 | 0.7768 | 0.4369 | 0.2879 | 0.2424 |
| `long_brush_withdraw` | 217 | 141 | 13 | 128 | 204 | 0.0922 | 0.0599 | 0.0726 | 0.0606 | 0.0000 |
| `short_brush_cleaning` | 376 | 0 | 0 | 0 | 376 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 333 | 6 | 0 | 6 | 333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 666 | 241 | 425 | 157 | 0.3619 | 0.6055 | 0.4530 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 665 | 287 | 378 | 285 | 0.4316 | 0.5017 | 0.4640 | 0.2000 | 0.1429 |
| `long_brush_withdraw` | 0 | 126 | 0 | 126 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 0 | 0 | 0 | 111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 0 | 0 | 0 | 261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### asformer

- 设计理由：ASFormer 保留当前更稳的 full_sequence + business_priors，只加入低维质量感知 ROI RGB 特征。
- 使用特征版本：`clean_bbox_v2_top1_impute+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.7293`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 426 | 149 | 277 | 341 | 0.3498 | 0.3041 | 0.3253 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 126 | 29 | 97 | 428 | 0.2302 | 0.0635 | 0.0995 | 0.0606 | 0.0000 |
| `long_brush_withdraw` | 217 | 428 | 60 | 368 | 157 | 0.1402 | 0.2765 | 0.1860 | 0.0606 | 0.0606 |
| `short_brush_cleaning` | 376 | 535 | 235 | 300 | 141 | 0.4393 | 0.6250 | 0.5159 | 0.1169 | 0.1169 |
| `flush` | 333 | 346 | 284 | 62 | 49 | 0.8208 | 0.8529 | 0.8365 | 0.1169 | 0.0909 |
| `air_injection` | 209 | 221 | 209 | 12 | 0 | 0.9457 | 1.0000 | 0.9721 | 0.0909 | 0.0909 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 186 | 83 | 103 | 315 | 0.4462 | 0.2085 | 0.2842 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 93 | 18 | 75 | 554 | 0.1935 | 0.0315 | 0.0541 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 0 | 251 | 0 | 251 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 391 | 16 | 375 | 95 | 0.0409 | 0.1441 | 0.0637 | 0.0000 | 0.0000 |
| `flush` | 261 | 358 | 219 | 139 | 42 | 0.6117 | 0.8391 | 0.7076 | 0.3333 | 0.1905 |
| `air_injection` | 115 | 178 | 115 | 63 | 0 | 0.6461 | 1.0000 | 0.7850 | 0.1429 | 0.1429 |

### bigru

- 设计理由：BiGRU 延续当前准确率最高的滑窗训练和增强 bbox 特征，再加入 v2 ROI RGB 特征。
- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.3984`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 651 | 280 | 371 | 210 | 0.4301 | 0.5714 | 0.4908 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 133 | 8 | 125 | 449 | 0.0602 | 0.0175 | 0.0271 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 217 | 550 | 75 | 475 | 142 | 0.1364 | 0.3456 | 0.1956 | 0.1775 | 0.0866 |
| `short_brush_cleaning` | 376 | 26 | 11 | 15 | 365 | 0.4231 | 0.0293 | 0.0547 | 0.0000 | 0.0000 |
| `flush` | 333 | 506 | 309 | 197 | 24 | 0.6107 | 0.9279 | 0.7366 | 0.0909 | 0.0909 |
| `air_injection` | 209 | 216 | 209 | 7 | 0 | 0.9676 | 1.0000 | 0.9835 | 0.0909 | 0.0909 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 499 | 238 | 261 | 160 | 0.4770 | 0.5980 | 0.5307 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 194 | 194 | 0 | 378 | 1.0000 | 0.3392 | 0.5065 | 0.3095 | 0.1429 |
| `long_brush_withdraw` | 0 | 230 | 0 | 230 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 110 | 96 | 14 | 15 | 0.8727 | 0.8649 | 0.8688 | 0.1429 | 0.1429 |
| `flush` | 261 | 244 | 214 | 30 | 47 | 0.8770 | 0.8199 | 0.8475 | 0.3429 | 0.3429 |
| `air_injection` | 115 | 180 | 115 | 65 | 0 | 0.6389 | 1.0000 | 0.7797 | 0.1429 | 0.1429 |

## 9. 结果分析

- val 上按准确率最高的是 `asformer`：ACC=0.4640，Frame-F1=0.5220，F1@0.25=0.0892。
- test 上按准确率最高的是 `bigru`：ACC=0.5882，Frame-F1=0.6005，F1@0.25=0.1876。
- `ms_tcn` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.2433，v2=0.3103，差值=0.0670。注意历史报告对应当时的数据划分/样本数，主要作参考。
- `asformer` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.6514，v2=0.4640，差值=-0.1874。注意历史报告对应当时的数据划分/样本数，主要作参考。
- `bigru` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.7482，v2=0.4284，差值=-0.3198。注意历史报告对应当时的数据划分/样本数，主要作参考。
- v2 把 v1 的 190 维原始 RGB 直方图替换为 7 个关键 ROI 的质量感知低维特征，并追加中心窗口均值与 delta，目标是降低 RGB 噪声对小数据训练的干扰。
- 三种时序模型主体没有复制到 image_train；v2 继续通过 `OfflineSegmenter` 复用 `segmenter/ms_tcn.py`、`segmenter/asformer.py`、`segmenter/bigru.py`。
- 相对 v1，test ACC 提升最大的是 `asformer`，变化为 0.2210。

## 10. 后续建议

- 继续保留 `common/roi_cache.py` 的 ROI manifest，v3 可以直接在此基础上预提取 DINOv2 ROI embedding，不再重复解析 YOLO 和图片路径。
- 如果以准确率为主，可继续尝试更长 epoch、多个 seed 和按 val ACC 选择 checkpoint；当前 `OfflineSegmenter.fit` 只保存最后一轮。
- 对 ASFormer 单独调小输入维度或增加 dropout；它对小数据和额外 RGB 特征更敏感。
- 对 `flush`/`air_injection` 增加 syringe 与 air_gun 的专门二分类外观差异特征，减少两类互相误报。
- 如果后续目标转回动作段边界，建议以 F1@0.25/F1@0.5 为主，并加入 boundary head 或边界后处理。
