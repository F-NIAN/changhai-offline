# ActionMixed 数据接入与离线时序分割训练汇报

## 1. 本轮任务

本轮将 ActionMixed 数据集转换为离线动作分割模型可直接读取的序列样本，并重新训练 `ms_tcn`、`asformer`、`bigru` 三个 baseline。训练轮数为 `1`，本轮输出目录为 `output`。

## 2. 动作标签体系

模型端固定使用 6 个逐帧类别：`idle` 表示无动作，其余 5 类为动作分割标签。

| 模型标签ID | 动作标签 | 帧数 | 占比 |
|---:|---|---:|---:|
| 0 | `idle` | 906 | 20.13% |
| 1 | `long_brush_insert` | 1105 | 24.55% |
| 2 | `long_brush_withdraw` | 763 | 16.95% |
| 3 | `short_brush_cleaning` | 252 | 5.60% |
| 4 | `flush` | 1109 | 24.64% |
| 5 | `air_injection` | 366 | 8.13% |

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

每一帧的 YOLO 检测框先按业务对象聚合。对每个对象生成 `count/cx/cy/area/speed` 5 个特征，`count` 表示该对象检测数量，`cx/cy/area` 是检测框中心和面积的聚合值，`speed` 是相邻采样帧中心点位移按 fps 归一化后的运动量。

随后为关键对象对补充 `valid/dist` 关系特征，例如 `hand` 到 `short_brush`、`air_gun` 到 `scope_distal_end`、`syringe` 到 `scope_distal_end` 的可见性和距离。最后加入 `t_norm/t_sin/t_cos` 三个时间位置特征。

因此每个样本最终保存为 `features[T, 62] float32` 和 `labels[T] int64`。训练时只用训练集统计均值和标准差，对 `features` 做 `(x - mean) / std` 标准化，再扩展为 `[1, T, 62]` 输入模型；模型输出为 `[1, 6, T]`，逐帧做交叉熵监督。

### 3.4 FeatureStore-like 落盘

每条视频/片段序列写为 `output/feature_store/task_<task_id>_step_1.npz`，其中包含 `features`、`labels`、`fps`、`frames`、`duration_s`、`feature_names`、`task_id`、`step_id`、`split`、`video_ref`。

## 4. 数据统计

本轮共生成 `16` 条序列样本、`4501` 个采样帧，特征维度为 `62`。

| split | 样本数 |
|---|---:|
| `test` | 2 |
| `train` | 8 |
| `val` | 6 |

全量逐帧标签分布：

| 模型标签ID | 动作标签 | 帧数 | 占比 |
|---:|---|---:|---:|
| 0 | `idle` | 906 | 20.13% |
| 1 | `long_brush_insert` | 1105 | 24.55% |
| 2 | `long_brush_withdraw` | 763 | 16.95% |
| 3 | `short_brush_cleaning` | 252 | 5.60% |
| 4 | `flush` | 1109 | 24.64% |
| 5 | `air_injection` | 366 | 8.13% |

训练集逐帧标签分布：

| 模型训练标签 | 训练帧数 |
|---|---:|
| `idle` | 638 |
| `long_brush_insert` | 466 |
| `long_brush_withdraw` | 635 |
| `short_brush_cleaning` | 115 |
| `flush` | 743 |
| `air_injection` | 366 |

## 5. 三个模型训练结果

验证指标使用片段级 `Segment F1@0.25`：同类别预测片段与真值片段 IoU 达到 0.25 即视为命中。该指标用于快速比较 baseline，不代表最终上线阈值。

| 模型 | 最后一轮训练loss | 验证集Segment F1@0.25 | 输出片段数 | 权重文件 |
|---|---:|---:|---:|---|
| `ms_tcn` | 1.7901 | 0.1111 | 16 | `output\models\ms_tcn_offline_segmenter.pt` |
| `asformer` | 1.6878 | 0.1736 | 132 | `output\models\asformer_offline_segmenter.pt` |
| `bigru` | 1.7199 | 0.1111 | 25 | `output\models\bigru_offline_segmenter.pt` |

本轮按验证集 `Segment F1@0.25` 选择的默认下游模型为 `asformer`。

## 6. 输出产物

- 结构化训练报告：`output\pipeline_report.json`
- 本汇报文件：`output\training_summary_report.md`
- 特征缓存：`output\feature_store`
- 预测片段 CSV/soft labels：`output\predictions`
- 下游推荐 SegmentFact：`output\asformer_segment_facts.jsonl`
- 下游推荐 FactLedger：`output\asformer_fact_ledger.jsonl`

## 7. 结论

本轮已经把动作分割标签扩展并固定为 `long_brush_insert`、`long_brush_withdraw`、`short_brush_cleaning`、`flush`、`air_injection` 五类，非动作帧统一为 `idle`。数据转换链路从原始逐帧检测框和动作标签开始，最终形成固定维度时序特征、逐帧监督标签、模型权重、预测片段和 FactLedger，可继续接入后端离线复核流程。
