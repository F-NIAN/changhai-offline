# 离线模型特征与训练策略实验报告

- 数据源：`input\modelscope\lhh010__cleansight-ActionMixed`
- epoch：`3`
- 实验数量：`8`

## 总体排名

| 排名 | 特征方法 | 训练方式 | 模型 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `window_stats+business_priors` | `sliding_window` | `bigru` | 249 | 0.7482 | 0.5686 | 0.6404 | 0.5963 | 0.1917 | 0.1750 |
| 2 | `v2` | `sliding_window` | `bigru` | 113 | 0.7449 | 0.5731 | 0.6486 | 0.5996 | 0.1625 | 0.1333 |
| 3 | `business_priors` | `sliding_window` | `bigru` | 121 | 0.7070 | 0.5784 | 0.5973 | 0.5765 | 0.1167 | 0.1000 |
| 4 | `window_stats` | `sliding_window` | `ms_tcn` | 241 | 0.3702 | 0.2048 | 0.1914 | 0.1908 | 0.1083 | 0.0583 |
| 5 | `window_stats` | `sliding_window` | `bigru` | 241 | 0.5441 | 0.5055 | 0.4153 | 0.3920 | 0.0917 | 0.0500 |
| 6 | `window_stats+business_priors` | `sliding_window` | `ms_tcn` | 249 | 0.3172 | 0.2547 | 0.1930 | 0.1390 | 0.0488 | 0.0167 |
| 7 | `business_priors` | `sliding_window` | `ms_tcn` | 121 | 0.3375 | 0.1596 | 0.1238 | 0.1393 | 0.0437 | 0.0312 |
| 8 | `v2` | `sliding_window` | `ms_tcn` | 113 | 0.2747 | 0.1514 | 0.0658 | 0.0911 | 0.0306 | 0.0306 |

## 最优实验逐类结果

最优配置：`window_stats+business_priors` + `sliding_window` + `bigru`。

| 动作类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-P@0.25 | Seg-R@0.25 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 152 | 0.5000 | 0.2360 | 0.3207 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `long_brush_insert` | 388 | 448 | 0.8147 | 0.9407 | 0.8732 | 0.3750 | 0.3750 | 0.3750 | 0.3750 |
| `long_brush_withdraw` | 157 | 106 | 0.3868 | 0.2611 | 0.3118 | 0.2500 | 0.1875 | 0.2083 | 0.1250 |
| `short_brush_cleaning` | 344 | 478 | 0.7197 | 1.0000 | 0.8370 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| `flush` | 318 | 345 | 0.9217 | 1.0000 | 0.9593 | 0.1250 | 0.1250 | 0.1250 | 0.1250 |
| `air_injection` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 方法说明

- `v2`：hand top-2，非 hand top-1，遮挡补全，关系 delta。
- `window_stats`：在 v2 上增加中心窗口统计，离线模型可以利用未来帧。
- `business_priors`：在 v2 上增加短刷/推流/注气/长刷插拔相关弱先验分数。
- `window_stats+business_priors`：组合以上两类增强特征。
- `sliding_window`：训练时切 128 帧窗口、stride=32；验证仍使用完整序列。

## 建议

- 优先看 target macro Segment F1@0.25/@0.5，而不是只看 frame ACC；动作边界和片段命中才是离线分割目标。
- 若 window_stats 组合优于 v2，说明离线模型确实受益于未来帧，应把中心窗口统计作为后续主线。
- 若 sliding_window 明显优于 full_sequence，说明当前 21 条序列样本太少，应固定使用滑窗训练、全序列验证。
- 若某类 recall 长期为 0，优先检查该动作段关键检测目标召回，而不是继续加深模型。
- 当前实验仍是小数据快速验证，正式结论建议把 epoch 提高到 20-100 并重复 3 个随机种子。
