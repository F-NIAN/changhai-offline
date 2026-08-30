# image_train_v1：ActionMixed RGB 图像特征首版改进训练报告

> 交接说明：本文保留当次脚本生成的原始结果；文中的本机绝对路径是运行快照。仓库根相对路径、复现命令和结果分析见 `experiments/03_rgb_roi/01_v1_basic_rgb/README.md`。

## 1. 本次改进目标

本次在不改动三种时序模型主体结构的前提下，新增 `image_train` 实验目录，把 ActionMixed 已下载的 JPEG 帧图像 RGB 信息加入训练。实现方式是复用当前三种模型各自最好的检测框特征 recipe，再拼接轻量 RGB ROI 外观特征，验证第一版图像信息是否能给动作分割带来收益。

## 2. 改了什么

- 新增训练脚本：`F:\暑期实习\offline-model\image_train\train_image_v1.py`
- 数据集：`F:\暑期实习\offline-model\input\modelscope\lhh010__cleansight-ActionMixed\cleansight-ActionMixed`
- 输出目录：`image_train\output_v1`
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
| `ms_tcn` | `v2+rgb_roi` | `full_sequence` | 303 | 0.2195 | 0.1583 | 0.2537 | 0.1724 | 0.1072 | 0.0628 | `image_train\output_v1\models\image_v1_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+rgb_roi` | `full_sequence` | 311 | 0.2286 | 0.0825 | 0.1862 | 0.1120 | 0.0467 | 0.0467 | `image_train\output_v1\models\image_v1_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+rgb_roi` | `sliding_window` | 439 | 0.5711 | 0.7348 | 0.6677 | 0.6104 | 0.1345 | 0.0739 | `image_train\output_v1\models\image_v1_bigru_offline_segmenter.pt` |

## 4. test 整体结果

| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ms_tcn` | `v2+rgb_roi` | `full_sequence` | 303 | 0.2608 | 0.1567 | 0.2016 | 0.1730 | 0.0471 | 0.0286 | `image_train\output_v1\models\image_v1_ms_tcn_offline_segmenter.pt` |
| `asformer` | `business_priors+rgb_roi` | `full_sequence` | 311 | 0.0885 | 0.0250 | 0.0252 | 0.0251 | 0.0286 | 0.0286 | `image_train\output_v1\models\image_v1_asformer_offline_segmenter.pt` |
| `bigru` | `window_stats+business_priors+rgb_roi` | `sliding_window` | 439 | 0.3782 | 0.5546 | 0.4950 | 0.4056 | 0.1245 | 0.1163 | `image_train\output_v1\models\image_v1_bigru_offline_segmenter.pt` |

## 5. 每个模型逐动作识别情况（以帧数为单位）

## ms_tcn

- 使用特征版本：`clean_bbox_v2_top1_impute+rgb_roi_stats_v1`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.5467`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 139 | 57 | 82 | 433 | 0.4101 | 0.1163 | 0.1812 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 417 | 111 | 306 | 346 | 0.2662 | 0.2429 | 0.2540 | 0.1212 | 0.0606 |
| `long_brush_withdraw` | 217 | 1143 | 132 | 1011 | 85 | 0.1155 | 0.6083 | 0.1941 | 0.2374 | 0.1364 |
| `short_brush_cleaning` | 376 | 383 | 157 | 226 | 219 | 0.4099 | 0.4176 | 0.4137 | 0.1775 | 0.1169 |
| `flush` | 333 | 0 | 0 | 0 | 333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 149 | 36 | 113 | 362 | 0.2416 | 0.0905 | 0.1316 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 577 | 288 | 289 | 284 | 0.4991 | 0.5035 | 0.5013 | 0.2000 | 0.1429 |
| `long_brush_withdraw` | 0 | 534 | 0 | 534 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 197 | 56 | 141 | 55 | 0.2843 | 0.5045 | 0.3636 | 0.0357 | 0.0000 |
| `flush` | 261 | 0 | 0 | 0 | 261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## asformer

- 使用特征版本：`clean_bbox_v2_top1_impute+business_priors+rgb_roi_stats_v1`
- 训练样本：`14`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`1.9518`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 450 | 98 | 352 | 392 | 0.2178 | 0.2000 | 0.2085 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 1247 | 335 | 912 | 122 | 0.2686 | 0.7330 | 0.3932 | 0.1879 | 0.1879 |
| `long_brush_withdraw` | 217 | 299 | 43 | 256 | 174 | 0.1438 | 0.1982 | 0.1667 | 0.0455 | 0.0455 |
| `short_brush_cleaning` | 376 | 86 | 0 | 86 | 376 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 333 | 0 | 0 | 0 | 333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0 | 0 | 209 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 315 | 57 | 258 | 341 | 0.1810 | 0.1432 | 0.1599 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 577 | 72 | 505 | 500 | 0.1248 | 0.1259 | 0.1253 | 0.1429 | 0.1429 |
| `long_brush_withdraw` | 0 | 456 | 0 | 456 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 109 | 0 | 109 | 111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 0 | 0 | 0 | 261 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 115 | 0 | 0 | 0 | 115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## bigru

- 使用特征版本：`clean_bbox_v2_top1_impute+center_window+business_priors+rgb_roi_stats_v1`
- 训练样本：`157`，train 序列：`14`，val 序列：`11`，test 序列：`7`
- 最后一轮 loss：`0.6467`

### val

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 490 | 498 | 196 | 302 | 294 | 0.3936 | 0.4000 | 0.3968 | 0.0000 | 0.0000 |
| `long_brush_insert` | 457 | 34 | 34 | 0 | 423 | 1.0000 | 0.0744 | 0.1385 | 0.0909 | 0.0000 |
| `long_brush_withdraw` | 217 | 457 | 81 | 376 | 136 | 0.1772 | 0.3733 | 0.2404 | 0.1879 | 0.0364 |
| `short_brush_cleaning` | 376 | 504 | 344 | 160 | 32 | 0.6825 | 0.9149 | 0.7818 | 0.1515 | 0.0909 |
| `flush` | 333 | 359 | 325 | 34 | 8 | 0.9053 | 0.9760 | 0.9393 | 0.1515 | 0.1515 |
| `air_injection` | 209 | 230 | 209 | 21 | 0 | 0.9087 | 1.0000 | 0.9522 | 0.0909 | 0.0909 |

### test

| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 398 | 371 | 163 | 208 | 235 | 0.4394 | 0.4095 | 0.4239 | 0.0000 | 0.0000 |
| `long_brush_insert` | 572 | 60 | 60 | 0 | 512 | 1.0000 | 0.1049 | 0.1899 | 0.1429 | 0.1429 |
| `long_brush_withdraw` | 0 | 414 | 0 | 414 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 294 | 107 | 187 | 4 | 0.3639 | 0.9640 | 0.5284 | 0.0714 | 0.0714 |
| `flush` | 261 | 137 | 106 | 31 | 155 | 0.7737 | 0.4061 | 0.5327 | 0.2653 | 0.2245 |
| `air_injection` | 115 | 181 | 115 | 66 | 0 | 0.6354 | 1.0000 | 0.7770 | 0.1429 | 0.1429 |

## 6. 结果分析

- val 上按 F1@0.25 排名最高的是 `bigru`，F1@0.25=0.1345，F1@0.5=0.0739，Frame-F1=0.6104。
- test 上按 F1@0.25 排名最高的是 `bigru`，F1@0.25=0.1245，F1@0.5=0.1163，Frame-F1=0.4056。
- 本版 RGB 特征是轻量统计特征，不训练图像 backbone，也不引入端到端视觉模型；它主要验证“ROI 局部外观是否有信号”，工程风险低，但表达能力明显弱于 DINOv2/VideoMAE。
- 由于 train/val/test 序列数量较少，单次 3 epoch 结果波动会比较大，应主要看 per-class recall 和 segment F1 是否出现稳定方向，而不是只看单一 ACC。

## 7. 后续修改建议

- 把 `rgb_roi_stats_v1` 与 baseline 最佳报告做横向对比，重点看 F1@0.25/F1@0.5 和每类 recall，判断 RGB 是否对边界和漏检有帮助。
- 如果 RGB 统计特征有收益，下一版改为冻结 DINOv2 ROI embedding，并用 PCA/Linear 压到每槽 32 或 64 维。
- 加入质量感知融合：用 present/conf/missing_age/imputed/candidate_count 学一个 gate，而不是简单 concat。
- 对 `flush` 和 `air_injection` 单独增强 syringe/air_gun ROI，并检查两类误检混淆；这两类最依赖外观区分。
- 正式结论前至少跑 20-100 epoch，并重复 3 个随机种子；当前 3 epoch 只能视为首版工程验证。
