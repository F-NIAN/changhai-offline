# image_train_v2：ActionMixed ROI RGB 质量感知特征训练报告

> 交接说明：本文保留当次脚本生成的原始结果；文中的本机绝对路径是运行快照。仓库根相对路径、复现命令和结果分析见 `experiments/03_rgb_roi/02_v2_quality_cache/README.md`。

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

- 数据集：`F:\暑期实习\offline-model\input\modelscope\lhh010__cleansight-ActionMixed\cleansight-ActionMixed`
- 输出目录：`image_train\output_v2`
- 训练轮数：`8`
- 设备：`cpu`
- ROI padding：`0.15`
- ROI manifest：`image_train\output_v2\roi_manifest_p15`
- RGB v2 cache：`image_train\output_v2\roi_rgb_v2_cache`

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
| `ms_tcn` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.1614 | 0.0065 | 0.0009 | 0.0016 | 0.0000 | 0.0000 | `image_train\output_v2\models\image_v2_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+roi_rgb_v2` | `full_sequence` | 331 | 0.5879 | 0.7157 | 0.8265 | 0.6575 | 0.1636 | 0.1394 | `image_train\output_v2\models\image_v2_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.6191 | 0.7789 | 0.7348 | 0.6698 | 0.1727 | 0.1667 | `image_train\output_v2\models\image_v2_bigru_offline_segmenter.pt` |

## 6.2. test 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.1764 | 0.0193 | 0.0133 | 0.0158 | 0.0000 | 0.0000 | `image_train\output_v2\models\image_v2_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+roi_rgb_v2` | `full_sequence` | 331 | 0.2505 | 0.4619 | 0.4487 | 0.3072 | 0.1000 | 0.1000 | `image_train\output_v2\models\image_v2_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+roi_rgb_v2` | `sliding_window` | 459 | 0.5278 | 0.5679 | 0.5362 | 0.4757 | 0.1352 | 0.1352 | `image_train\output_v2\models\image_v2_bigru_offline_segmenter.pt` |

## 7. v1 到 v2 的准确率变化

| 模型 | v1 val ACC | v2 val ACC | val 变化 | v1 test ACC | v2 test ACC | test 变化 |
|---|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | 0.2195 | 0.1614 | -0.0581 | 0.2608 | 0.1764 | -0.0844 |
| `asformer` | 0.2286 | 0.5879 | 0.3593 | 0.0885 | 0.2505 | 0.1620 |
| `bigru` | 0.5711 | 0.6191 | 0.0480 | 0.3782 | 0.5278 | 0.1496 |

从准确率目标看，v2 对 `bigru` 和 `asformer` 有明显收益；`ms_tcn` 对当前高维窗口增强更敏感，本版不建议作为主线。

## 8. 每个模型逐动作识别情况（以帧数为单位）

### ms_tcn

- 设计理由：v2 以准确率为目标，给 ms_tcn 增加中心窗口统计和滑窗样本数，缓解原 full_sequence 样本过少。
- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.4700`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 1320 | 335 | 985 | 155 | 0.2538 | 0.6837 | 0.3702 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 731 | 0 | 731 | 457 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 217 | 31 | 1 | 30 | 216 | 0.0323 | 0.0046 | 0.0081 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 376 | 0 | 0 | 0 | 376 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 333 | 0 | 0 | 0 | 333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 1034 | 219 | 815 | 179 | 0.2118 | 0.5503 | 0.3059 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 393 | 38 | 355 | 534 | 0.0967 | 0.0664 | 0.0788 | 0.0000 | 0.0000 |
| `long_brush_withdraw` | 0 | 30 | 0 | 30 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 0 | 0 | 0 | 111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 0 | 0 | 0 | 261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### asformer

- 设计理由：ASFormer 保留当前更稳的 full_sequence + business_priors，只加入低维质量感知 ROI RGB 特征。
- 使用特征版本：`clean_bbox_v2_top1_impute+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.8916`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 35 | 28 | 7 | 462 | 0.8000 | 0.0571 | 0.1067 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 64 | 62 | 2 | 395 | 0.9688 | 0.1357 | 0.2380 | 0.0909 | 0.0909 |
| `long_brush_withdraw` | 217 | 781 | 217 | 564 | 0 | 0.2778 | 1.0000 | 0.4349 | 0.2727 | 0.2727 |
| `short_brush_cleaning` | 376 | 556 | 376 | 180 | 0 | 0.6763 | 1.0000 | 0.8069 | 0.2424 | 0.1818 |
| `flush` | 333 | 373 | 332 | 41 | 1 | 0.8901 | 0.9970 | 0.9405 | 0.1515 | 0.0909 |
| `air_injection` | 209 | 273 | 209 | 64 | 0 | 0.7656 | 1.0000 | 0.8672 | 0.0606 | 0.0606 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 41 | 32 | 9 | 366 | 0.7805 | 0.0804 | 0.1458 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 65 | 65 | 0 | 507 | 1.0000 | 0.1136 | 0.2041 | 0.1429 | 0.1429 |
| `long_brush_withdraw` | 0 | 525 | 0 | 525 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 538 | 105 | 433 | 6 | 0.1952 | 0.9459 | 0.3236 | 0.0714 | 0.0714 |
| `flush` | 261 | 90 | 48 | 42 | 213 | 0.5333 | 0.1839 | 0.2735 | 0.1429 | 0.1429 |
| `air_injection` | 115 | 198 | 115 | 83 | 0 | 0.5808 | 1.0000 | 0.7348 | 0.1429 | 0.1429 |

### bigru

- 设计理由：BiGRU 延续当前准确率最高的滑窗训练和增强 bbox 特征，再加入 v2 ROI RGB 特征。
- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+roi_rgb_v2_quality_smooth`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.2913`

#### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 477 | 217 | 260 | 273 | 0.4549 | 0.4429 | 0.4488 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 56 | 56 | 0 | 401 | 1.0000 | 0.1225 | 0.2183 | 0.0909 | 0.0909 |
| `long_brush_withdraw` | 217 | 491 | 152 | 339 | 65 | 0.3096 | 0.7005 | 0.4294 | 0.2879 | 0.2879 |
| `short_brush_cleaning` | 376 | 503 | 337 | 166 | 39 | 0.6700 | 0.8963 | 0.7668 | 0.2424 | 0.2121 |
| `flush` | 333 | 344 | 318 | 26 | 15 | 0.9244 | 0.9550 | 0.9394 | 0.1515 | 0.1515 |
| `air_injection` | 209 | 211 | 209 | 2 | 0 | 0.9905 | 1.0000 | 0.9952 | 0.0909 | 0.0909 |

#### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 494 | 237 | 257 | 161 | 0.4798 | 0.5955 | 0.5314 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 230 | 226 | 4 | 346 | 0.9826 | 0.3951 | 0.5636 | 0.2381 | 0.2381 |
| `long_brush_withdraw` | 0 | 187 | 0 | 187 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 263 | 107 | 156 | 4 | 0.4068 | 0.9640 | 0.5722 | 0.0571 | 0.0571 |
| `flush` | 261 | 104 | 84 | 20 | 177 | 0.8077 | 0.3218 | 0.4603 | 0.2381 | 0.2381 |
| `air_injection` | 115 | 179 | 115 | 64 | 0 | 0.6425 | 1.0000 | 0.7823 | 0.1429 | 0.1429 |

## 9. 结果分析

- val 上按准确率最高的是 `bigru`：ACC=0.6191，Frame-F1=0.6698，F1@0.25=0.1727。
- test 上按准确率最高的是 `bigru`：ACC=0.5278，Frame-F1=0.4757，F1@0.25=0.1352。
- `ms_tcn` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.2433，v2=0.1614，差值=-0.0819。注意历史报告对应当时的数据划分/样本数，主要作参考。
- `asformer` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.6514，v2=0.5879，差值=-0.0635。注意历史报告对应当时的数据划分/样本数，主要作参考。
- `bigru` 与历史 best_model_report 的 val ACC 粗略对比：baseline=0.7482，v2=0.6191，差值=-0.1291。注意历史报告对应当时的数据划分/样本数，主要作参考。
- v2 把 v1 的 190 维原始 RGB 直方图替换为 7 个关键 ROI 的质量感知低维特征，并追加中心窗口均值与 delta，目标是降低 RGB 噪声对小数据训练的干扰。
- 三种时序模型主体没有复制到 image_train；v2 继续通过 `OfflineSegmenter` 复用 `segmenter/ms_tcn.py`、`segmenter/asformer.py`、`segmenter/bigru.py`。
- 相对 v1，test ACC 提升最大的是 `asformer`，变化为 0.1620。

## 10. 后续建议

- 继续保留 `common/roi_cache.py` 的 ROI manifest，v3 可以直接在此基础上预提取 DINOv2 ROI embedding，不再重复解析 YOLO 和图片路径。
- 如果以准确率为主，可继续尝试更长 epoch、多个 seed 和按 val ACC 选择 checkpoint；当前 `OfflineSegmenter.fit` 只保存最后一轮。
- 对 ASFormer 单独调小输入维度或增加 dropout；它对小数据和额外 RGB 特征更敏感。
- 对 `flush`/`air_injection` 增加 syringe 与 air_gun 的专门二分类外观差异特征，减少两类互相误报。
- 如果后续目标转回动作段边界，建议以 F1@0.25/F1@0.5 为主，并加入 boundary head 或边界后处理。
