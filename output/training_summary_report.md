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

每一帧的 YOLO 检测框按业务对象聚合后，转换为模型可直接消费的时序特征。当前特征版本为 `clean_bbox_v2_top1_impute`，实际落盘特征维度为 `113`。

#### 3.3.1 目标对象经营方式

- `hand`：保留 top-2 检测结果，支持双手交互场景。每个手槽生成 `present/conf/cx/cy/area/speed/missing_age/imputed` 8 维特征。
- 其它对象：按类别保留 top-1 检测结果，不再对同类多框做加权平均。每类生成 `candidate_count/present/conf/cx/cy/area/speed/missing_age/imputed` 9 维特征。

对象类别包括：`short_brush`、`long_brush`、`syringe`、`air_gun`、`scope_control_body`、`scope_mid_section`、`scope_distal_end`、`brush_tip_out`。

#### 3.3.2 每个对象槽的特征含义

- `count` / `candidate_count`：当前帧该类别检测数量，剪裁到 0～3 后归一化。这个特征反映检测器对目标的候选数量和遮挡/误检情况。
- `present`：是否存在有效目标框。用来区分真实检测缺失与空帧。
- `conf`：检测置信度。反映目标定位可信度，辅助模型避免过度依赖低置信度框。
- `cx`, `cy`：检测框中心归一化坐标。表示目标在画面中的位置。
- `area`：检测框面积。表示目标尺寸、缩放变化和近远距离信息。
- `speed`：相邻帧中心点位移乘以 fps 后归一化。描述目标运动量和速度变化。
- `missing_age`：连续缺失帧数归一化值，最大分辨短时遮挡与长期消失。
- `imputed`：短时缺失补全标记。补全帧 `present` 仍为 0，避免把插值结果混淆为真实检测。

#### 3.3.3 短时遮挡补全策略

对于每个对象槽，当前设计对短时缺失做轻量补全：

- 如果两个真实检测间的缺失长度不超过 6 帧，则对 `cx/cy/area/conf` 线性插值，补全帧标记 `imputed=1`。
- 如果序列尾部出现短缺失，用最后一次已检测到的坐标前向填充，置信度折半。
- `active` 标记表示真实检测或补全帧；仅对 active 帧保留坐标和速度信息，对非 active 帧把 `conf/cx/cy/area` 置 0。

该策略的设计意义在于：保持局部动作连续性，减少短时遮挡带来的轨迹断裂，同时让模型区分“真实出现”和“短时补全”。

#### 3.3.4 关系特征

为关键对象对补充关系特征，当前包含 7 组对象关系：

- `hand` ↔ `short_brush`
- `hand` ↔ `long_brush`
- `brush_tip_out` ↔ `scope_distal_end`
- `short_brush` ↔ `scope_control_body`
- `long_brush` ↔ `scope_mid_section`
- `air_gun` ↔ `scope_distal_end`
- `syringe` ↔ `scope_distal_end`

每组关系特征包括：

- `valid`：当前帧两个对象均可用时为 1，否则为 0。
- `dist`：两者中心距离归一化后值，距离越近表示目标间语义关系越强。
- `delta`：相邻帧距离变化量，帮助捕捉夹持、接近、远离等动态关系。

其中涉及 `hand` 的关系取两只手距离的最小值，避免因手切换导致距离特征抖动。

#### 3.3.5 时间位置编码

在序列末尾补入三维时间特征：

- `t_norm`：当前帧在序列中的归一化位置。
- `t_sin` / `t_cos`：周期性位置编码，提供时间顺序信息，帮助模型区分序列前中后的动作模式。

#### 3.3.6 特征维度汇总

当前特征组合构成：

- `hand_count`：1 维
- `hand_top1` / `hand_top2`：2 × 8 = 16 维
- 8 个非手对象槽：8 × 9 = 72 维
- 7 组关系特征：7 × 3 = 21 维
- 时间编码：3 维

总计 `113` 维。

因此每个样本最终保存为 `features[T, 113] float32` 和 `labels[T] int64`。训练时只用训练集统计均值和标准差，对 `features` 做 `(x - mean) / std` 标准化，再扩展为 `[1, T, 113]` 输入模型；模型输出为 `[1, 6, T]`，逐帧做交叉熵监督。

### 3.4 FeatureStore-like 落盘

每条视频/片段序列写为 `output/feature_store/task_<task_id>_step_1.npz`，其中包含 `features`、`labels`、`fps`、`frames`、`duration_s`、`feature_names`、`task_id`、`step_id`、`split`、`video_ref`。

### 3.5 模型间特征使用差异

本轮基础训练流程中，`ms_tcn`、`asformer` 和 `bigru` 三个 baseline 均使用同一套 `clean_bbox_v2_top1_impute` 结构化特征输入（113 维）。三者的差异主要体现在模型结构与时序建模方式上，而不是基础输入特征本身。

不过在后续的模型优化与最佳权重实验中，不同模型对特征增强的偏好不同：

- `ms_tcn`：通常在原始 `v2` 特征上表现最佳，说明它对结构化几何+关系特征的直接时序建模能力更强。
- `asformer`：更适合加入 `business_priors` 弱先验特征，说明注意力机制在结合业务语义关系时能获得更多增益。
- `bigru`：在 `window_stats+business_priors` 组合上效果最好，表明双向循环网络可以从中心窗口统计和先验信息中更好地提取跨帧动态规律。

这意味着当前报告中的三种 baseline 采用相同的基础特征输入，但在后续实验中可针对模型类型选择更适合的特征扩展策略。

## 4. 数据统计

本轮共生成 `16` 条序列样本、`4501` 个采样帧，特征维度为 `113`。

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

## 7. 后续可行改进方案

### 7.1 当前特征设计优势

- 目标级几何特征强，适合动作分割中的工具-手协同、物体接近和运动变化判断。
- 短时缺失补全与 `missing_age/imputed` 标记减少检测遮挡带来的间断。
- 关系特征直接建模关键对象对的空间关系，增强对 `hand`、`brush`、`scope`、`air_gun` 之间交互的感知。
- 时间编码补充了序列进度信息，帮助模型区分动作阶段而不只依赖瞬时几何。

### 7.2 可行的改进方向

- 增加视觉表征：当前仅用 YOLO bbox 的几何信息，若后续能稳定获取原始 RGB，可考虑引入轻量视觉嵌入（如 ResNet/ViT 预训练特征）或显著性图，帮助区分形态接近但语义不同的动作。
- 引入光流/运动特征：对于 `flush`、`air_injection` 等动作，目标几何变化有限但流体运动/喷射方向不同，光流信息可以补强运动模式。
- 增加姿态/手势特征：若能进一步检测手部关键点或工具朝向，可提升 `hand` 与 `brush`、`scope` 之间交互的判别能力。
- 融合对象语义：目前只保留 top-1/2 框，后续可考虑利用检测类别分布、目标遮挡关系、多个候选框的置信度梯度等更细粒度不确定性信息。
- 加入窗口统计与业务先验：使用滑动窗口统计、动作持续时间先验或状态转移特征，可以改善动作边界判别和时序一致性。

### 7.3 是否需要原始 RGB 图像？

- 目前的离线基线设计不依赖原始 RGB，而是通过检测框几何、速度、关系特征完成建模，适合仅有检测结果时的轻量部署。
- 但如果目标是进一步提升精度，原始 RGB 是有价值的：
  - 对于同一目标位置下动作类别区分（如 `flush` vs `air_injection`、`long_brush_insert` vs `long_brush_withdraw`）时，视觉纹理、工具状态和液体/气流外观可能更直接。
  - 还可从 RGB 提取手部姿态、工具朝向、遮挡细节、表面反光等当前 bbox 特征难以捕捉的线索。
- 取舍建议：若数据量与算力允许，优先引入“视觉嵌入 + 光流”而不是原始全帧像素，以降低存储和训练成本；保持当前 bbox 特征作为结构化先验，作为多模态融合的基础。

## 8. 结论

本轮已经把动作分割标签扩展并固定为 `long_brush_insert`、`long_brush_withdraw`、`short_brush_cleaning`、`flush`、`air_injection` 五类，非动作帧统一为 `idle`。数据转换链路从原始逐帧检测框和动作标签开始，最终形成固定维度时序特征、逐帧监督标签、模型权重、预测片段和 FactLedger，可继续接入后端离线复核流程。
