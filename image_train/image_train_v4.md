# image_train_v4：逐帧准确率优先的 RGB/proxy ROI 训练报告

> 交接说明：本文保留当次脚本生成的原始结果；文中的本机绝对路径是运行快照。仓库根相对路径、复现命令和结果分析见 `experiments/03_rgb_roi/04_v4_proxy_roi/README.md`。

## 1. v4 目标

v4 按 2026-07-28 讨论后的策略执行：训练阶段以逐帧动作分类准确率为主要目标，动作段切分、段内平滑、短段删除和相邻同类段合并作为模型输出后的后处理。报告同时列出 raw 输出和 postprocessed 输出。

## 2. 参考方法依据

- 本地 `F:/暑期实习/参考文献/CS-TCN.pdf`：MS-TCN 以逐帧交叉熵为核心分类损失，平滑损失用于减少过分割，因此 v4 把帧分类和段平滑拆开处理。
- 本地 `Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf`：第一人称手部动作识别中，手部和手-物交互线索对细粒度动作有效，因此 v4 新增 expanded/ring hand proxy ROI。
- 本地 `s11263-022-01594-9.pdf`：综述指出真实视频动作识别受背景、光照、相似动作和标注不足影响，因此 v4 把 ROI、训练超参和后处理参数全部参数化。

## 3. 文件组织

```text
image_train/
  common/
    roi_cache.py
    proxy_roi.py
    segment_postprocess.py
  v4/
    configs/image_train_v4.json
    train_image_v4.py
  output_v4/
    feature_store_v2/
    roi_manifest_p15/
    roi_rgb_v2_cache/
    proxy_rgb_v4_cache/
    models/
    predictions/
    image_train_v4.json
  image_train_v4.md
```

## 4. 训练和调参参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `epochs` | `12` | 命令行参数，可在 v4 config 中集中修改 |
| `lr` | `0.002` | 命令行参数，可在 v4 config 中集中修改 |
| `min_lr` | `1e-05` | 命令行参数，可在 v4 config 中集中修改 |
| `weight_decay` | `0.0001` | 命令行参数，可在 v4 config 中集中修改 |
| `label_smoothing` | `0.03` | 命令行参数，可在 v4 config 中集中修改 |
| `grad_clip` | `5.0` | 命令行参数，可在 v4 config 中集中修改 |
| `scheduler` | `cosine` | 命令行参数，可在 v4 config 中集中修改 |
| `early_stopping` | `True` | 命令行参数，可在 v4 config 中集中修改 |
| `patience` | `4` | 命令行参数，可在 v4 config 中集中修改 |
| `early_metric` | `val_accuracy` | 命令行参数，可在 v4 config 中集中修改 |
| `train_mode` | `full_sequence` | 命令行参数，可在 v4 config 中集中修改 |
| `proxy_smooth_window` | `5` | 命令行参数，可在 v4 config 中集中修改 |
| `hand_expand` | `1.6` | 命令行参数，可在 v4 config 中集中修改 |
| `postprocess_objective` | `accuracy` | 命令行参数，可在 v4 config 中集中修改 |

## 5. 数据概况

| split | 序列数 | 帧数 |
|---|---:|---:|
| `test` | 7 | 1457 |
| `train` | 14 | 5993 |
| `val` | 11 | 2082 |

## 6. 特征设计

| 模型 | 特征组合 | dim | 说明 |
|---|---|---:|---|
| `ms_tcn` | `business_priors+roi_rgb_v2+proxy_rgb_v4` | 691 | MS-TCN 保留多阶段逐帧 refinement，v4 给它补充 object ROI 与 proxy ROI，但训练选择以 raw frame accuracy 为主。 |
| `asformer` | `business_priors+roi_rgb_v2+proxy_rgb_v4` | 691 | ASFormer 继续使用完整序列 attention 建模长上下文，输入与 MS-TCN 保持同一套 v4 低维视觉特征。 |
| `bigru` | `window_stats+business_priors+roi_rgb_v2+proxy_rgb_v4` | 819 | BiGRU 保留窗口统计特征以增强局部稳定性，同时使用 ROI/proxy RGB 弥补短刷和长刷弱检测。 |

新增 proxy ROI 包括 `wash_tank`、三段 `wash_tank_strip`、`scope_channel`、`hand_top1/2_expanded`、`hand_top1/2_ring`、`hand_control_union`。每个 ROI 提取 `valid/conf/area/aspect/RGB mean/brightness mean/std/saturation/edge_energy/motion_energy`，再追加中心 5 帧均值和 delta。

## 7. val 整体结果

| 模型 | dim | Raw ACC | Raw P | Raw R | Raw Frame-F1 | Post ACC | Post Frame-F1 | Post F1@0.25 | Post F1@0.5 | Pred/GT段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | 691 | 0.3079 | 0.1591 | 0.2824 | 0.2014 | 0.3295 | 0.2015 | 0.0788 | 0.0364 | 33/22 | 29 | 36.3000 |
| `asformer` | 691 | 0.6647 | 0.7444 | 0.8408 | 0.7446 | 0.6777 | 0.7493 | 0.2052 | 0.1870 | 27/22 | 13 | 11.3438 |
| `bigru` | 819 | 0.7022 | 0.7731 | 0.7822 | 0.7495 | 0.7051 | 0.7512 | 0.2030 | 0.1667 | 20/22 | 10 | 13.3846 |

## 8. test 整体结果

| 模型 | dim | Raw ACC | Raw P | Raw R | Raw Frame-F1 | Post ACC | Post Frame-F1 | Post F1@0.25 | Post F1@0.5 | Pred/GT段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | 691 | 0.3370 | 0.2391 | 0.2936 | 0.1746 | 0.3727 | 0.1708 | 0.0571 | 0.0143 | 22/14 | 24 | 45.8333 |
| `asformer` | 691 | 0.5202 | 0.5464 | 0.5454 | 0.4687 | 0.5285 | 0.4710 | 0.1400 | 0.1400 | 20/14 | 16 | 9.3333 |
| `bigru` | 819 | 0.5964 | 0.5557 | 0.5676 | 0.5037 | 0.5951 | 0.5021 | 0.1686 | 0.1686 | 10/14 | 10 | 8.5714 |

## 9. 后处理参数

| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |
|---|---:|---:|---:|---:|
| `ms_tcn` | 1 | 8 | 0 | 0.25 |
| `asformer` | 1 | 8 | 0 | 0.4 |
| `bigru` | 1 | 8 | 0 | 0.0 |

## 10. 逐模型逐动作结果

### ms_tcn

- 最佳 epoch：`4`；停止 epoch：`8`；验证最佳分数：`0.3079`。
- 权重文件：`image_train\output_v4\models\image_v4_ms_tcn_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 318 | 0.3208 | 0.2082 | 0.2525 |
| `long_brush_insert` | 457 | 671 | 0.2653 | 0.3895 | 0.3156 |
| `long_brush_withdraw` | 217 | 273 | 0.1172 | 0.1475 | 0.1306 |
| `short_brush_cleaning` | 376 | 797 | 0.4128 | 0.8750 | 0.5610 |
| `flush` | 333 | 23 | 0.0000 | 0.0000 | 0.0000 |
| `air_injection` | 209 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 281 | 0.4164 | 0.2940 | 0.3446 |
| `long_brush_insert` | 572 | 377 | 0.6870 | 0.4528 | 0.5458 |
| `long_brush_withdraw` | 0 | 153 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 634 | 0.1751 | 1.0000 | 0.2980 |
| `flush` | 261 | 12 | 0.3333 | 0.0153 | 0.0293 |
| `air_injection` | 115 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 5 | 1 | 0.0909 | 0.0909 | 64.5000 |
| `long_brush_withdraw` | 5 | 17 | 12 | 0.1515 | 0.0000 | 11.0000 |
| `short_brush_cleaning` | 4 | 11 | 7 | 0.1515 | 0.0909 | 47.5000 |
| `flush` | 8 | 0 | 8 | 0.0000 | 0.0000 | - |
| `air_injection` | 1 | 0 | 1 | 0.0000 | 0.0000 | - |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 5 | 1 | 0.2143 | 0.0000 | 64.5000 |
| `long_brush_withdraw` | 0 | 7 | 7 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 10 | 9 | 0.0714 | 0.0714 | 8.5000 |
| `flush` | 6 | 0 | 6 | 0.0000 | 0.0000 | - |
| `air_injection` | 1 | 0 | 1 | 0.0000 | 0.0000 | - |

### asformer

- 最佳 epoch：`12`；停止 epoch：`12`；验证最佳分数：`0.6647`。
- 权重文件：`image_train\output_v4\models\image_v4_asformer_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 181 | 0.6243 | 0.2306 | 0.3368 |
| `long_brush_insert` | 457 | 211 | 0.9005 | 0.4158 | 0.5689 |
| `long_brush_withdraw` | 217 | 537 | 0.3464 | 0.8571 | 0.4934 |
| `short_brush_cleaning` | 376 | 589 | 0.6384 | 1.0000 | 0.7793 |
| `flush` | 333 | 334 | 0.9281 | 0.9309 | 0.9295 |
| `air_injection` | 209 | 230 | 0.9087 | 1.0000 | 0.9522 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 233 | 0.8670 | 0.5075 | 0.6403 |
| `long_brush_insert` | 572 | 264 | 0.9394 | 0.4336 | 0.5933 |
| `long_brush_withdraw` | 0 | 362 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 310 | 0.3452 | 0.9640 | 0.5083 |
| `flush` | 261 | 105 | 0.8190 | 0.3295 | 0.4699 |
| `air_injection` | 115 | 183 | 0.6284 | 1.0000 | 0.7718 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 3 | 1 | 0.1818 | 0.1818 | 15.0000 |
| `long_brush_withdraw` | 5 | 9 | 4 | 0.3091 | 0.2182 | 8.7500 |
| `short_brush_cleaning` | 4 | 8 | 4 | 0.2545 | 0.2545 | 15.1250 |
| `flush` | 8 | 5 | 3 | 0.2197 | 0.2197 | 10.2000 |
| `air_injection` | 1 | 2 | 1 | 0.0606 | 0.0606 | 5.0000 |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 5 | 1 | 0.2857 | 0.2857 | 5.7500 |
| `long_brush_withdraw` | 0 | 7 | 7 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 5 | 4 | 0.0714 | 0.0714 | 2.0000 |
| `flush` | 6 | 2 | 4 | 0.2000 | 0.2000 | 4.2500 |
| `air_injection` | 1 | 1 | 0 | 0.1429 | 0.1429 | 34.0000 |

### bigru

- 最佳 epoch：`12`；停止 epoch：`12`；验证最佳分数：`0.7022`。
- 权重文件：`image_train\output_v4\models\image_v4_bigru_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 417 | 0.5875 | 0.5000 | 0.5402 |
| `long_brush_insert` | 457 | 218 | 1.0000 | 0.4770 | 0.6459 |
| `long_brush_withdraw` | 217 | 395 | 0.2886 | 0.5253 | 0.3725 |
| `short_brush_cleaning` | 376 | 470 | 0.7532 | 0.9415 | 0.8369 |
| `flush` | 333 | 352 | 0.9148 | 0.9670 | 0.9401 |
| `air_injection` | 209 | 230 | 0.9087 | 1.0000 | 0.9522 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 273 | 0.8315 | 0.5704 | 0.6766 |
| `long_brush_insert` | 572 | 343 | 0.9854 | 0.5909 | 0.7388 |
| `long_brush_withdraw` | 0 | 281 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 264 | 0.3826 | 0.9099 | 0.5387 |
| `flush` | 261 | 112 | 0.7857 | 0.3372 | 0.4718 |
| `air_injection` | 115 | 184 | 0.6250 | 1.0000 | 0.7692 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 3 | 1 | 0.2727 | 0.1818 | 26.1667 |
| `long_brush_withdraw` | 5 | 8 | 3 | 0.2273 | 0.2273 | 7.0000 |
| `short_brush_cleaning` | 4 | 5 | 1 | 0.2727 | 0.1818 | 15.1250 |
| `flush` | 8 | 3 | 5 | 0.1515 | 0.1515 | 1.7500 |
| `air_injection` | 1 | 1 | 0 | 0.0909 | 0.0909 | 10.5000 |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 3 | 3 | 0.4286 | 0.4286 | 2.8333 |
| `long_brush_withdraw` | 0 | 1 | 1 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 3 | 2 | 0.0714 | 0.0714 | 5.0000 |
| `flush` | 6 | 2 | 4 | 0.2000 | 0.2000 | 6.0000 |
| `air_injection` | 1 | 1 | 0 | 0.1429 | 0.1429 | 34.5000 |

## 11. 结果分析

- val raw frame accuracy 最好的模型是 `bigru`，ACC=0.7022。
- val post frame accuracy 最好的模型是 `bigru`，ACC=0.7051；post 结果用于后续动作段输出。
- test raw frame accuracy 最好的模型是 `bigru`，ACC=0.5964。
- test post frame accuracy 最好的模型是 `bigru`，ACC=0.5951；post 结果用于后续动作段输出。
- 按本轮主目标 test raw frame accuracy 排序，`bigru` 最好，ACC=0.5964。
- `long_brush_withdraw` 在当前 test split 中 GT 帧数为 0，但模型仍有预测：`ms_tcn` 误报 153 帧、`asformer` 误报 362 帧、`bigru` 误报 281 帧；这说明当前 split 类别分布不均衡，且长刷插入/拔出方向证据仍不稳定。
- v4 训练选择 raw frame accuracy 做主目标，避免在切分片段数据上过度追逐段级后处理参数。
- postprocess 仍会影响 ACC 和段级 F1；本轮默认按验证集 post frame accuracy 选择参数，segment F1/段数/边界误差作为附带约束。
- 新增 proxy RGB 特征维度较多，如果 test 明显低于 val，需要优先做特征组消融，而不是继续加复杂模型。

## 12. 后续建议

- 先做 proxy 特征组消融：只开 object ROI、只开 hand proxy、只开 wash/channel proxy、全部开启，确认哪些组真实提升 test raw ACC。
- 当前 postprocess 默认按验证集 frame accuracy 搜索；若后续再次切回动作段目标，可把 `--postprocess-objective segment` 打开。
- 若完整视频级人工标注准备好，应以整段视频作为 sequence 重新训练和测试；当前切分片段数据仍不适合最终判断真实动作段划分能力。
- 如果 proxy ROI 带来 val 高但 test 低，应先收紧固定 ROI 或提高正则，而不是继续加 DINOv2/VideoMAE 高维特征。
