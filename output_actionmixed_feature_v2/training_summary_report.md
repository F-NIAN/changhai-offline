# ActionMixed 数据接入与离线时序分割训练汇报

## 1. 本轮任务

本轮将 ActionMixed 数据集转换为离线动作分割模型可直接读取的序列样本，并重新训练 `ms_tcn`、`asformer`、`bigru` 三个 baseline。训练轮数为 `1`，本轮输出目录为 `output_actionmixed_feature_v2`。

## 2. 动作标签体系

模型端固定使用 6 个逐帧类别：`idle` 表示无动作，其余 5 类为动作分割标签。

| 模型标签ID | 动作标签 | 帧数 | 占比 |
|---:|---|---:|---:|
| 0 | `idle` | 1242 | 21.96% |
| 1 | `long_brush_insert` | 1325 | 23.43% |
| 2 | `long_brush_withdraw` | 792 | 14.01% |
| 3 | `short_brush_cleaning` | 669 | 11.83% |
| 4 | `flush` | 1261 | 22.30% |
| 5 | `air_injection` | 366 | 6.47% |

ActionMixed 原始 `labels/data.yaml` 的动作 ID 顺序和模型内部类别顺序不同，因此转换时按名称做显式映射：

| 原始ActionMixed ID | 原始标签 | 模型标签ID | 模型标签 |
|---:|---|---:|---|
| 0 | `idle` | 0 | `idle` |
| 1 | `air_injection` | 5 | `air_injection` |
| 2 | `flush` | 4 | `flush` |
| 3 | `long_brush_insert` | 1 | `long_brush_insert` |
| 4 | `long_brush_withdraw` | 2 | `long_brush_withdraw` |
| 5 | `short_brush_cleaning` | 3 | `short_brush_cleaning` |

## 3. 数据如何转换为模型输入

### 3.1 原始文件组织

已下载数据位于 `input/modelscope/lhh010__cleansight-ActionMixed`。其中 `labels/{train,val,test}/{video}.txt` 存放动作真值，`frames/{train,val,test}/{video}.mp4-{frame_id}.txt` 存放同一采样帧的 YOLO 检测框。

动作标签文件每行按 `frame_id action_id` 解析；帧号与 `frames` 目录中的文件名对齐。检测框文件每行按 `class_id cx cy w h` 解析，坐标为 0-1 归一化中心点和宽高。

检测类别为：0:hand, 1:scope_control_body, 2:scope_mid_section, 3:scope_distal_end, 4:syringe, 5:air_gun, 6:short_brush, 7:brush_tip_out。

### 3.2 逐帧标签对齐

转换脚本先读取每个视频片段实际存在的采样帧号，再把动作标签按帧号写入 `labels[T]`。未被任何动作覆盖的帧保持为 `idle`。原始动作 ID 不直接作为模型 ID 使用，而是先转成动作名称，再映射到模型内部的 `CLASSES` 顺序。

### 3.3 逐帧检测框特征

每一帧的 YOLO 检测框先按业务对象转换为 v2 特征。`hand` 使用 top-2 独立槽位，避免两只手被加权合并；其它对象按 `confidence * sqrt(area)` 和跨帧位置稳定性选择 top-1，不再把同类多框加权平均成一个可能不存在的中心点。

单目标特征包含候选数量、真实可见性、置信度、中心点、面积、速度、连续缺失时长 `missing_age` 和短遮挡补全标记 `imputed`。短缺失段会做轻量线性插值或短尾部前向填充，但补全帧的 `present` 仍为 0，只通过 `imputed=1` 告诉模型这是推测值。

随后为关键对象对补充 `valid/dist/delta` 关系特征，例如 `hand` 到 `short_brush`、`air_gun` 到 `scope_distal_end`、`syringe` 到 `scope_distal_end` 的可用性、距离和距离变化。最后加入 `t_norm/t_sin/t_cos` 三个时间位置特征。

因此每个样本最终保存为 `features[T, 113] float32` 和 `labels[T] int64`。训练时只用训练集统计均值和标准差，对 `features` 做 `(x - mean) / std` 标准化，再扩展为 `[1, T, 113]` 输入模型；模型输出为 `[1, 6, T]`，逐帧做交叉熵监督。

### 3.4 FeatureStore-like 落盘

每条视频/片段序列写为 `output/feature_store/task_<task_id>_step_1.npz`，其中包含 `features`、`labels`、`fps`、`frames`、`duration_s`、`feature_names`、`task_id`、`step_id`、`split`、`video_ref`。

## 4. 数据统计

本轮共生成 `21` 条序列样本、`5655` 个采样帧，特征维度为 `113`。

| split | 样本数 |
|---|---:|
| `test` | 3 |
| `train` | 10 |
| `val` | 8 |

全量逐帧标签分布：

| 模型标签ID | 动作标签 | 帧数 | 占比 |
|---:|---|---:|---:|
| 0 | `idle` | 1242 | 21.96% |
| 1 | `long_brush_insert` | 1325 | 23.43% |
| 2 | `long_brush_withdraw` | 792 | 14.01% |
| 3 | `short_brush_cleaning` | 669 | 11.83% |
| 4 | `flush` | 1261 | 22.30% |
| 5 | `air_injection` | 366 | 6.47% |

训练集逐帧标签分布：

| 模型训练标签 | 训练帧数 |
|---|---:|
| `idle` | 765 |
| `long_brush_insert` | 578 |
| `long_brush_withdraw` | 635 |
| `short_brush_cleaning` | 214 |
| `flush` | 743 |
| `air_injection` | 366 |

## 5. 三个模型训练结果

验证指标使用片段级 `Segment F1@0.25`：同类别预测片段与真值片段 IoU 达到 0.25 即视为命中。该指标用于快速比较 baseline，不代表最终上线阈值。

| 模型 | 最后一轮训练loss | 验证集Segment F1@0.25 | 输出片段数 | 权重文件 |
|---|---:|---:|---:|---|
| `ms_tcn` | 1.9483 | 0.3125 | 24 | `output_actionmixed_feature_v2\models\ms_tcn_offline_segmenter.pt` |
| `asformer` | 2.0151 | 0.1708 | 97 | `output_actionmixed_feature_v2\models\asformer_offline_segmenter.pt` |
| `bigru` | 1.7776 | 0.3125 | 71 | `output_actionmixed_feature_v2\models\bigru_offline_segmenter.pt` |

本轮按验证集 `Segment F1@0.25` 选择的默认下游模型为 `ms_tcn`。

## 6. 输出产物

- 结构化训练报告：`output_actionmixed_feature_v2\pipeline_report.json`
- 本汇报文件：`output_actionmixed_feature_v2\training_summary_report.md`
- 特征缓存：`output_actionmixed_feature_v2\feature_store`
- 预测片段 CSV/soft labels：`output_actionmixed_feature_v2\predictions`
- 下游推荐 SegmentFact：`output_actionmixed_feature_v2\ms_tcn_segment_facts.jsonl`
- 下游推荐 FactLedger：`output_actionmixed_feature_v2\ms_tcn_fact_ledger.jsonl`

## 7. 结论

本轮已经把动作分割标签扩展并固定为 `long_brush_insert`、`long_brush_withdraw`、`short_brush_cleaning`、`flush`、`air_injection` 五类，非动作帧统一为 `idle`。数据转换链路从原始逐帧检测框和动作标签开始，最终形成固定维度时序特征、逐帧监督标签、模型权重、预测片段和 FactLedger，可继续接入后端离线复核流程。
