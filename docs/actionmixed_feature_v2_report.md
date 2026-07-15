# ActionMixed 特征转换 v2 测试报告

## 背景

医院摄像头下的胃肠镜洗刷动作存在几个典型问题：

- 长毛刷细、遮挡多，单独检测框不稳定；
- 短毛刷和注射器目标较小，容易漏检或被手遮挡；
- 胃肠镜控制部、中段、末端等通常是单实例目标；
- 同类多框做加权聚合会把中心点平均到现实中不存在的位置。

因此本次只在 `offline-model` 中验证特征转换 v2，不改后端仓库。后续若确认有效，再把同名同序的
`FeatureVectorizer` 同步到后端。

## 特征转换 v2 设计

### 1. 手部 top-2 保留

`hand` 仍保留两个独立槽位：

```text
hand_top1_*
hand_top2_*
```

原因是两只手可能同时参与操作，加权成一个中心点会丢失左右手差异。

### 2. 非手目标改为 top-1

以下对象不再做同类多框加权平均：

```text
short_brush
long_brush
syringe
air_gun
scope_control_body
scope_mid_section
scope_distal_end
brush_tip_out
```

每帧按 `confidence * sqrt(area)` 和跨帧位置稳定性选择一个 top-1 候选框，同时保留
`candidate_count`。这样可以避免多个误检框被平均成错误位置。

### 3. 遮挡与漏检特征

每个对象槽位增加：

```text
present
conf
cx
cy
area
speed
missing_age
imputed
```

其中：

- `present`：当前帧真实检测是否存在；
- `missing_age`：连续缺失帧数归一化；
- `imputed`：是否由短时遮挡补全得到。

短缺失段会做轻量线性插值；序列尾部短缺失会做短时前向填充。补全帧不会伪装成真实检测，
即 `present=0`、`imputed=1`。

### 4. 关系特征增加 delta

关键对象对从旧版：

```text
valid
dist
```

扩展为：

```text
valid
dist
delta
```

`delta` 表示对象距离变化，帮助模型感知靠近/远离趋势，尤其对长毛刷插入和拔出更有意义。

## 输入输出变化

旧版特征：

```text
features[T, 68]
```

新版特征：

```text
features[T, 113]
```

三种模型仍使用统一输入：

```text
torch tensor [1, T, 113]
```

输出仍为：

```text
logits [1, 6, T]
```

动作类别不变：

```text
idle
long_brush_insert
long_brush_withdraw
short_brush_cleaning
flush
air_injection
```

## 测试命令

```bash
python run_pipeline.py \
  --input-source actionmixed \
  --actionmixed-root input\modelscope\lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 1 \
  --out-dir output_actionmixed_feature_v2
```

## 测试结果

本轮共解析：

```text
21 条序列
5655 个采样帧
feature_dim = 113
```

三模型验证集 `Segment F1@0.25`：

| 模型 | Segment F1@0.25 | 输出片段数 |
|---|---:|---:|
| `ms_tcn` | 0.3125 | 24 |
| `bigru` | 0.3125 | 71 |
| `asformer` | 0.1708 | 97 |

当前 1 epoch 快速测试下选择 `ms_tcn` 作为本轮默认模型。该结果主要用于验证特征转换和训练链路，
不代表最终精度。

## 输出目录

```text
output_actionmixed_feature_v2/
  feature_store/
  models/
  predictions/
  pipeline_report.json
  training_summary_report.md
  ms_tcn_segment_facts.jsonl
  ms_tcn_fact_ledger.jsonl
```

## 后续建议

- 用 20-100 epoch 重跑，观察 v2 特征是否稳定优于旧版 68 维特征；
- 对长毛刷单独增加 `brush_tip_out -> scope_distal_end` 的方向性和持续性特征；
- 本轮 checkpoint 已写入 `feature_version=clean_bbox_v2_top1_impute` 和 113 个 `feature_names`；
- 如果后续接入后端，必须把 `feature_version`、`feature_names`、`normalizer_mean/std`
  一起写入 checkpoint，并在后端严格校验；
- 当前 v2 checkpoint 不能直接给后端现有 68 维 `clean.py` 使用，必须同步后端特征转换后才能接入。
