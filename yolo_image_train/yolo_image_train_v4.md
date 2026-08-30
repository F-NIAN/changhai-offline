# image_train_v4：逐帧准确率优先的 RGB/proxy ROI 训练报告

> 交接说明：本文是使用真实 YOLO v0.3 预测框生成的原始报告；本机绝对路径是运行快照。复现说明见 `experiments/04_real_yolo_rgb_roi/04_v4/README.md`。

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
| `ms_tcn` | 691 | 0.4467 | 0.2695 | 0.4452 | 0.3352 | 0.4621 | 0.3452 | 0.0964 | 0.0812 | 27/22 | 17 | 27.6875 |
| `asformer` | 691 | 0.4529 | 0.3898 | 0.5347 | 0.3866 | 0.4673 | 0.3980 | 0.1115 | 0.1115 | 25/22 | 19 | 14.1875 |
| `bigru` | 819 | 0.7037 | 0.7640 | 0.7610 | 0.7503 | 0.7027 | 0.7467 | 0.1510 | 0.1216 | 22/22 | 10 | 14.6364 |

## 8. test 整体结果

| 模型 | dim | Raw ACC | Raw P | Raw R | Raw Frame-F1 | Post ACC | Post Frame-F1 | Post F1@0.25 | Post F1@0.5 | Pred/GT段数 | 段数误差 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_tcn` | 691 | 0.1517 | 0.0802 | 0.0526 | 0.0578 | 0.1517 | 0.0557 | 0.0229 | 0.0000 | 21/14 | 19 | 26.5000 |
| `asformer` | 691 | 0.3109 | 0.3407 | 0.2557 | 0.2474 | 0.3157 | 0.2561 | 0.0733 | 0.0543 | 23/14 | 11 | 13.1250 |
| `bigru` | 819 | 0.7028 | 0.6950 | 0.6736 | 0.6658 | 0.6987 | 0.6667 | 0.2162 | 0.2019 | 16/14 | 10 | 14.0556 |

## 9. 后处理参数

| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |
|---|---:|---:|---:|---:|
| `ms_tcn` | 1 | 8 | 0 | 0.0 |
| `asformer` | 7 | 8 | 0 | 0.0 |
| `bigru` | 7 | 8 | 4 | 0.4 |

## 10. 逐模型逐动作结果

### ms_tcn

- 最佳 epoch：`10`；停止 epoch：`12`；验证最佳分数：`0.4467`。
- 权重文件：`F:\暑期实习\offline-model\yolo_image_train\output_v4\models\image_v4_ms_tcn_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 277 | 0.5776 | 0.3265 | 0.4172 |
| `long_brush_insert` | 457 | 676 | 0.5325 | 0.7877 | 0.6355 |
| `long_brush_withdraw` | 217 | 367 | 0.3515 | 0.5945 | 0.4418 |
| `short_brush_cleaning` | 376 | 156 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 333 | 606 | 0.4637 | 0.8438 | 0.5985 |
| `air_injection` | 209 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 194 | 0.6031 | 0.2940 | 0.3953 |
| `long_brush_insert` | 572 | 222 | 0.2928 | 0.1136 | 0.1637 |
| `long_brush_withdraw` | 0 | 452 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 228 | 0.0000 | 0.0000 | 0.0000 |
| `flush` | 261 | 361 | 0.1080 | 0.1494 | 0.1254 |
| `air_injection` | 115 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 6 | 2 | 0.2424 | 0.2424 | 22.1667 |
| `long_brush_withdraw` | 5 | 14 | 9 | 0.1182 | 0.0727 | 13.6667 |
| `short_brush_cleaning` | 4 | 3 | 1 | 0.0000 | 0.0000 | - |
| `flush` | 8 | 4 | 4 | 0.1212 | 0.0909 | 57.0000 |
| `air_injection` | 1 | 0 | 1 | 0.0000 | 0.0000 | - |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 4 | 2 | 0.1143 | 0.0000 | 26.5000 |
| `long_brush_withdraw` | 0 | 9 | 9 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 5 | 4 | 0.0000 | 0.0000 | - |
| `flush` | 6 | 3 | 3 | 0.0000 | 0.0000 | - |
| `air_injection` | 1 | 0 | 1 | 0.0000 | 0.0000 | - |

### asformer

- 最佳 epoch：`9`；停止 epoch：`12`；验证最佳分数：`0.4529`。
- 权重文件：`F:\暑期实习\offline-model\yolo_image_train\output_v4\models\image_v4_asformer_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 152 | 0.6974 | 0.2163 | 0.3302 |
| `long_brush_insert` | 457 | 99 | 0.5556 | 0.1204 | 0.1978 |
| `long_brush_withdraw` | 217 | 802 | 0.2332 | 0.8618 | 0.3670 |
| `short_brush_cleaning` | 376 | 464 | 0.5991 | 0.7394 | 0.6619 |
| `flush` | 333 | 565 | 0.5611 | 0.9520 | 0.7060 |
| `air_injection` | 209 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 134 | 0.7687 | 0.2588 | 0.3872 |
| `long_brush_insert` | 572 | 114 | 0.8509 | 0.1696 | 0.2828 |
| `long_brush_withdraw` | 0 | 652 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 227 | 0.1189 | 0.2432 | 0.1598 |
| `flush` | 261 | 308 | 0.7338 | 0.8659 | 0.7944 |
| `air_injection` | 115 | 22 | 0.0000 | 0.0000 | 0.0000 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 2 | 2 | 0.0909 | 0.0909 | 4.0000 |
| `long_brush_withdraw` | 5 | 12 | 7 | 0.2545 | 0.2545 | 16.0000 |
| `short_brush_cleaning` | 4 | 8 | 4 | 0.1212 | 0.1212 | 22.2500 |
| `flush` | 8 | 3 | 5 | 0.0909 | 0.0909 | 1.0000 |
| `air_injection` | 1 | 0 | 1 | 0.0000 | 0.0000 | - |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 5 | 1 | 0.0952 | 0.0000 | 39.0000 |
| `long_brush_withdraw` | 0 | 8 | 8 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 3 | 2 | 0.0000 | 0.0000 | - |
| `flush` | 6 | 6 | 0 | 0.2714 | 0.2714 | 4.5000 |
| `air_injection` | 1 | 1 | 0 | 0.0000 | 0.0000 | - |

### bigru

- 最佳 epoch：`12`；停止 epoch：`12`；验证最佳分数：`0.7036`。
- 权重文件：`F:\暑期实习\offline-model\yolo_image_train\output_v4\models\image_v4_bigru_offline_segmenter.pt`。

#### val raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 490 | 478 | 0.5565 | 0.5429 | 0.5496 |
| `long_brush_insert` | 457 | 281 | 0.9217 | 0.5667 | 0.7019 |
| `long_brush_withdraw` | 217 | 321 | 0.2928 | 0.4332 | 0.3494 |
| `short_brush_cleaning` | 376 | 422 | 0.7441 | 0.8351 | 0.7870 |
| `flush` | 333 | 367 | 0.8801 | 0.9700 | 0.9229 |
| `air_injection` | 209 | 213 | 0.9812 | 1.0000 | 0.9905 |

#### test raw 逐动作帧级识别

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |
|---|---:|---:|---:|---:|---:|
| `idle` | 398 | 330 | 0.7515 | 0.6231 | 0.6813 |
| `long_brush_insert` | 572 | 357 | 0.9804 | 0.6119 | 0.7535 |
| `long_brush_withdraw` | 0 | 253 | 0.0000 | 0.0000 | 0.0000 |
| `short_brush_cleaning` | 111 | 116 | 0.9397 | 0.9820 | 0.9604 |
| `flush` | 261 | 218 | 0.9266 | 0.7739 | 0.8434 |
| `air_injection` | 115 | 183 | 0.6284 | 1.0000 | 0.7718 |

#### val post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 4 | 5 | 1 | 0.2424 | 0.1818 | 22.3333 |
| `long_brush_withdraw` | 5 | 7 | 2 | 0.1273 | 0.1273 | 6.2500 |
| `short_brush_cleaning` | 4 | 6 | 2 | 0.1429 | 0.1169 | 24.8333 |
| `flush` | 8 | 3 | 5 | 0.1515 | 0.0909 | 2.2500 |
| `air_injection` | 1 | 1 | 0 | 0.0909 | 0.0909 | 2.5000 |

#### test post 动作段识别

| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |
|---|---:|---:|---:|---:|---:|---:|
| `long_brush_insert` | 6 | 5 | 1 | 0.4524 | 0.3810 | 15.7500 |
| `long_brush_withdraw` | 0 | 6 | 6 | 0.0000 | 0.0000 | - |
| `short_brush_cleaning` | 1 | 1 | 0 | 0.1429 | 0.1429 | 2.0000 |
| `flush` | 6 | 3 | 3 | 0.3429 | 0.3429 | 9.1667 |
| `air_injection` | 1 | 1 | 0 | 0.1429 | 0.1429 | 34.0000 |

## 11. 结果分析

- val raw frame accuracy 最好的模型是 `bigru`，ACC=0.7037。
- val post frame accuracy 最好的模型是 `bigru`，ACC=0.7027；post 结果用于后续动作段输出。
- test raw frame accuracy 最好的模型是 `bigru`，ACC=0.7028。
- test post frame accuracy 最好的模型是 `bigru`，ACC=0.6987；post 结果用于后续动作段输出。
- 按本轮主目标 test raw frame accuracy 排序，`bigru` 最好，ACC=0.7028。
- `long_brush_withdraw` 在当前 test split 中 GT 帧数为 0，但模型仍有预测：`ms_tcn` 误报 452 帧、`asformer` 误报 652 帧、`bigru` 误报 253 帧；这说明当前 split 类别分布不均衡，且长刷插入/拔出方向证据仍不稳定。
- v4 训练选择 raw frame accuracy 做主目标，避免在切分片段数据上过度追逐段级后处理参数。
- postprocess 仍会影响 ACC 和段级 F1；本轮默认按验证集 post frame accuracy 选择参数，segment F1/段数/边界误差作为附带约束。
- 新增 proxy RGB 特征维度较多，如果 test 明显低于 val，需要优先做特征组消融，而不是继续加复杂模型。

## 12. 后续建议

- 先做 proxy 特征组消融：只开 object ROI、只开 hand proxy、只开 wash/channel proxy、全部开启，确认哪些组真实提升 test raw ACC。
- 当前 postprocess 默认按验证集 frame accuracy 搜索；若后续再次切回动作段目标，可把 `--postprocess-objective segment` 打开。
- 若完整视频级人工标注准备好，应以整段视频作为 sequence 重新训练和测试；当前切分片段数据仍不适合最终判断真实动作段划分能力。
- 如果 proxy ROI 带来 val 高但 test 低，应先收紧固定 ROI 或提高正则，而不是继续加 DINOv2/VideoMAE 高维特征。
