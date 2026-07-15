# ActionMixed 离线模型优化实验报告

## 实验目标

当前任务是离线动作分割，不受实时约束，可以使用当前帧之后的上下文。本轮实验比较几类特征输入和训练方式，
观察它们对五类动作的片段级分割效果影响：

```text
long_brush_insert
long_brush_withdraw
short_brush_cleaning
flush
air_injection
```

评估重点是片段级 `Segment F1@0.25` 和 `Segment F1@0.5`。`ACC/Precision/Recall/Frame-F1`
是逐帧分类指标，用于辅助判断模型是否偏向某些类别。

注意：当前 ActionMixed 的验证集里 `air_injection` support 为 0，因此本轮无法真实评估
`air_injection` 的召回和片段 F1；表中该类为 0，主要反映验证集划分覆盖不足。

## 已尝试的方法

### 特征方法

- `v2`：113 维特征。hand top-2，非 hand top-1，短遮挡补全，`missing_age/imputed`，关系 `delta`。
- `window_stats`：在 v2 基础上增加中心窗口统计。因为是离线模型，窗口可以使用未来帧。
- `business_priors`：在 v2 基础上增加短刷刷洗、推流、注气、长刷插拔相关弱先验分数。
- `window_stats+business_priors`：组合中心窗口统计和业务先验。

### 训练方式

- `full_sequence`：每条视频序列作为一个训练样本。
- `sliding_window`：训练时切 128 帧窗口、stride=32；验证仍使用完整序列。

### 模型

- `ms_tcn`：MS-TCN + BiLSTM。
- `asformer`：ASFormer 风格时序 attention。
- `bigru`：3 层 BiGRU。

完整全序列训练覆盖三种模型；滑窗训练覆盖 `ms_tcn/bigru`，因为 ASFormer 滑窗组合耗时明显更高，
本轮先用全序列结果代表其架构表现。

## 实验设置

```text
数据集：input/modelscope/lhh010__cleansight-ActionMixed
训练轮数：3 epoch
训练 split：10 条序列
验证 split：8 条序列
总实验数：20
```

输出目录：

```text
output_actionmixed_optim_fullseq/
output_actionmixed_optim_sliding/
```

## 总体结果

表中：

- `ACC`：逐帧总体准确率。
- `Precision/Recall/Frame-F1`：五个动作类的 macro frame 指标，不含 idle。
- `F1@0.25/F1@0.5`：五个动作类的 macro segment F1。

| 排名 | 特征方法 | 训练方式 | 模型 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `window_stats+business_priors` | `sliding_window` | `bigru` | 249 | 0.7482 | 0.5686 | 0.6404 | 0.5963 | 0.1917 | 0.1750 |
| 2 | `window_stats` | `full_sequence` | `bigru` | 241 | 0.6952 | 0.4885 | 0.6108 | 0.5283 | 0.1810 | 0.1071 |
| 3 | `v2` | `sliding_window` | `bigru` | 113 | 0.7449 | 0.5731 | 0.6486 | 0.5996 | 0.1625 | 0.1333 |
| 4 | `business_priors` | `full_sequence` | `asformer` | 121 | 0.6514 | 0.5610 | 0.5763 | 0.5556 | 0.1583 | 0.1250 |
| 5 | `business_priors` | `full_sequence` | `bigru` | 121 | 0.5088 | 0.4255 | 0.5161 | 0.4199 | 0.1435 | 0.0821 |
| 6 | `window_stats` | `full_sequence` | `asformer` | 241 | 0.6684 | 0.5141 | 0.6031 | 0.5521 | 0.1410 | 0.1171 |
| 7 | `v2` | `full_sequence` | `bigru` | 113 | 0.6763 | 0.4445 | 0.5901 | 0.5021 | 0.1351 | 0.0988 |
| 8 | `window_stats+business_priors` | `full_sequence` | `asformer` | 249 | 0.6828 | 0.5187 | 0.6308 | 0.5673 | 0.1338 | 0.1171 |
| 9 | `v2` | `full_sequence` | `ms_tcn` | 113 | 0.2466 | 0.0906 | 0.2580 | 0.1339 | 0.1292 | 0.1167 |
| 10 | `v2` | `full_sequence` | `asformer` | 113 | 0.6540 | 0.4726 | 0.5845 | 0.5205 | 0.1292 | 0.0875 |
| 11 | `window_stats+business_priors` | `full_sequence` | `bigru` | 249 | 0.5337 | 0.4882 | 0.5459 | 0.4376 | 0.1226 | 0.0821 |
| 12 | `business_priors` | `sliding_window` | `bigru` | 121 | 0.7070 | 0.5784 | 0.5973 | 0.5765 | 0.1167 | 0.1000 |
| 13 | `window_stats` | `sliding_window` | `ms_tcn` | 241 | 0.3702 | 0.2048 | 0.1914 | 0.1908 | 0.1083 | 0.0583 |
| 14 | `window_stats` | `sliding_window` | `bigru` | 241 | 0.5441 | 0.5055 | 0.4153 | 0.3920 | 0.0917 | 0.0500 |
| 15 | `business_priors` | `full_sequence` | `ms_tcn` | 121 | 0.2466 | 0.0622 | 0.1989 | 0.0914 | 0.0750 | 0.0500 |
| 16 | `window_stats` | `full_sequence` | `ms_tcn` | 241 | 0.1027 | 0.0205 | 0.2000 | 0.0372 | 0.0500 | 0.0250 |
| 17 | `window_stats+business_priors` | `full_sequence` | `ms_tcn` | 249 | 0.1027 | 0.0205 | 0.2000 | 0.0372 | 0.0500 | 0.0250 |
| 18 | `window_stats+business_priors` | `sliding_window` | `ms_tcn` | 249 | 0.3172 | 0.2547 | 0.1930 | 0.1390 | 0.0488 | 0.0167 |
| 19 | `business_priors` | `sliding_window` | `ms_tcn` | 121 | 0.3375 | 0.1596 | 0.1238 | 0.1393 | 0.0437 | 0.0312 |
| 20 | `v2` | `sliding_window` | `ms_tcn` | 113 | 0.2747 | 0.1514 | 0.0658 | 0.0911 | 0.0306 | 0.0306 |

## 最优配置逐类结果

最优配置：

```text
feature = window_stats+business_priors
train = sliding_window
model = bigru
feature_dim = 249
```

逐类结果：

| 动作类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-P@0.25 | Seg-R@0.25 | Seg-F1@0.25 | Seg-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idle` | 322 | 152 | 0.5000 | 0.2360 | 0.3207 | - | - | - | - |
| `long_brush_insert` | 388 | 448 | 0.8147 | 0.9407 | 0.8732 | 0.3750 | 0.3750 | 0.3750 | 0.3750 |
| `long_brush_withdraw` | 157 | 106 | 0.3868 | 0.2611 | 0.3118 | 0.2500 | 0.1875 | 0.2083 | 0.1250 |
| `short_brush_cleaning` | 344 | 478 | 0.7197 | 1.0000 | 0.8370 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| `flush` | 318 | 345 | 0.9217 | 1.0000 | 0.9593 | 0.1250 | 0.1250 | 0.1250 | 0.1250 |
| `air_injection` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 结果分析

### 1. BiGRU 当前最稳

在前 8 名中，BiGRU 占 5 个。当前数据量只有 21 条序列，BiGRU 比 ASFormer 更稳定，
也明显优于当前 MS-TCN 实现。MS-TCN 在多数组合里出现类别坍缩，说明它需要更长训练、
更强正则或重新调整多阶段结构。

### 2. 中心窗口统计有效

`window_stats + full_sequence + bigru` 排名第 2，`window_stats+business_priors + sliding_window + bigru`
排名第 1。说明离线模型确实受益于未来帧上下文。对这个任务，单帧 bbox 特征不够，
需要显式加入短时间窗口内的存在率、速度、距离变化和遮挡趋势。

### 3. 滑窗训练有帮助，但不是单独充分条件

`v2 + sliding_window + bigru` 的 frame 指标较高，Segment F1@0.25 也高于 `v2 + full_sequence + bigru`。
但 `window_stats + sliding_window + bigru` 反而下降，说明滑窗训练和特征增强之间存在交互，
不是所有增强都能叠加变好。当前最好的组合是滑窗 + 窗口统计 + 业务先验。

### 4. 业务先验单独使用提升有限

`business_priors` 单独使用时，ASFormer 和 BiGRU 有一定收益，但不如窗口统计稳定。
这说明业务先验有帮助，但不能替代真实时序上下文。更好的做法是把业务先验作为辅助特征，
而不是主要判断依据。

### 5. 各动作表现差异明显

- `flush`：帧级效果最好，Precision 0.9217、Recall 1.0、Frame-F1 0.9593，但 Segment F1 只有 0.125，
  表示模型能识别推流帧，但边界和片段合并仍不准。
- `short_brush_cleaning`：帧级 Recall 1.0，Frame-F1 0.8370，说明短刷动作较容易被召回；
  但 predicted 多于 support，存在过检。
- `long_brush_insert`：帧级表现较好，Frame-F1 0.8732，Segment F1@0.5 也达到 0.375，
  是当前最可靠的长刷相关动作。
- `long_brush_withdraw`：仍然较差，Frame-F1 0.3118，Segment F1@0.5 只有 0.125。
  这说明拔出动作的方向性和边界还没被充分建模。
- `air_injection`：当前验证集没有样本，不能下结论。必须调整 split 或单独构造含注气动作的验证集。

## 建议

1. 下一轮主线建议使用 `window_stats+business_priors + sliding_window + BiGRU`。
2. 训练轮数从 3 epoch 提高到 20-100 epoch，并至少跑 3 个随机种子，避免小数据偶然性。
3. 重新划分验证集，确保五类动作都有 support，尤其是 `air_injection`。
4. 对 `long_brush_withdraw` 增加方向性特征：手/刷头远离 `scope_distal_end` 的速度、趋势和持续时间。
5. 对片段边界增加后处理或模型头：boundary head、最短/最长持续时长约束、Viterbi/CRF 解码。
6. 暂不建议继续加大 Transformer；当前数据规模下 ASFormer 没有超过 BiGRU。
7. 在后端接入前，必须同步 `feature_version` 和 `feature_names`，当前最优实验是 249 维，不兼容后端 68/113 维输入。
