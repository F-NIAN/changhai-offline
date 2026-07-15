# 离线模型特征与训练策略实验报告

- 数据源：`input\modelscope\lhh010__cleansight-ActionMixed`
- epoch：`3`
- 实验数量：`12`

## 总体排名

| 排名 | 特征方法 | 训练方式 | 模型 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `window_stats` | `full_sequence` | `bigru` | 241 | 0.6952 | 0.4885 | 0.6108 | 0.5283 | 0.1810 | 0.1071 |
| 2 | `business_priors` | `full_sequence` | `asformer` | 121 | 0.6514 | 0.5610 | 0.5763 | 0.5556 | 0.1583 | 0.1250 |
| 3 | `business_priors` | `full_sequence` | `bigru` | 121 | 0.5088 | 0.4255 | 0.5161 | 0.4199 | 0.1435 | 0.0821 |
| 4 | `window_stats` | `full_sequence` | `asformer` | 241 | 0.6684 | 0.5141 | 0.6031 | 0.5521 | 0.1410 | 0.1171 |
| 5 | `v2` | `full_sequence` | `bigru` | 113 | 0.6763 | 0.4445 | 0.5901 | 0.5021 | 0.1351 | 0.0988 |
| 6 | `window_stats+business_priors` | `full_sequence` | `asformer` | 249 | 0.6828 | 0.5187 | 0.6308 | 0.5673 | 0.1338 | 0.1171 |
| 7 | `v2` | `full_sequence` | `ms_tcn` | 113 | 0.2466 | 0.0906 | 0.2580 | 0.1339 | 0.1292 | 0.1167 |
| 8 | `v2` | `full_sequence` | `asformer` | 113 | 0.6540 | 0.4726 | 0.5845 | 0.5205 | 0.1292 | 0.0875 |
| 9 | `window_stats+business_priors` | `full_sequence` | `bigru` | 249 | 0.5337 | 0.4882 | 0.5459 | 0.4376 | 0.1226 | 0.0821 |
| 10 | `business_priors` | `full_sequence` | `ms_tcn` | 121 | 0.2466 | 0.0622 | 0.1989 | 0.0914 | 0.0750 | 0.0500 |
| 11 | `window_stats` | `full_sequence` | `ms_tcn` | 241 | 0.1027 | 0.0205 | 0.2000 | 0.0372 | 0.0500 | 0.0250 |
| 12 | `window_stats+business_priors` | `full_sequence` | `ms_tcn` | 249 | 0.1027 | 0.0205 | 0.2000 | 0.0372 | 0.0500 | 0.0250 |

## 最优实验逐类结果

最优配置：`window_stats` + `full_sequence` + `bigru`。

| 动作类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-P@0.25 | Seg-R@0.25 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 8 | 1.0000 | 0.0248 | 0.0485 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `long_brush_insert` | 388 | 643 | 0.5956 | 0.9871 | 0.7430 | 0.3750 | 0.3750 | 0.3750 | 0.2500 |
| `long_brush_withdraw` | 157 | 54 | 0.2037 | 0.0701 | 0.1043 | 0.1250 | 0.1250 | 0.1250 | 0.0000 |
| `short_brush_cleaning` | 344 | 484 | 0.7107 | 1.0000 | 0.8309 | 0.2500 | 0.1875 | 0.2083 | 0.1250 |
| `flush` | 318 | 340 | 0.9324 | 0.9969 | 0.9635 | 0.2500 | 0.1750 | 0.1964 | 0.1607 |
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
