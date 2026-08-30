# ActionMixed 多模态离线时序分割后续改进参考方案

## 1. 文档目标

本文基于当前 `offline-model` 已完成的三种离线时序模型 baseline：

- `segmenter/ms_tcn.py`：MS-TCN + BiLSTM；
- `segmenter/asformer.py`：ASFormer 风格时序 attention；
- `segmenter/bigru.py`：BiGRU；

结合当前 ActionMixed 数据接入和 v2 检测特征，整理一个后续可逐步实现的多模态时序分割方案。

目标不是立即替换现有 baseline，而是给出一条从“YOLO 结构化特征”升级到“RGB 外观特征 + YOLO 结构化特征”的工程路线，用于提升以下五类动作的离线分割效果：

```text
long_brush_insert
long_brush_withdraw
short_brush_cleaning
flush
air_injection
```

后续设计仍保持离线模型定位：

- 不受实时约束；
- 可以使用当前帧之后的上下文；
- 重点追求动作段边界准确；
- 输出仍然服务于 `SegmentFact -> FactLedger` 或后端离线链路。

## 2. 当前 baseline 已经完成什么

当前 `offline-model` 的基本数据流是：

```text
ActionMixed / Label Studio / YOLO 输出
  -> data_transfer.py / dataset.py
  -> FeatureStore-like npz
  -> dataset.py 构造完整序列数据集
  -> segmenter/*.py 三种模型训练和验证
  -> SegmentFact / FactLedger 风格输出
```

当前核心输入主要来自 YOLO 检测框，已经做了比较完整的结构化特征工程。

### 2.1 当前 v2 特征

当前 v2 特征大致包含：

- `hand` 使用 top-2 槽位，分别保留两只手；
- 非 `hand` 目标使用 top-1，不做同类多框加权平均；
- 每个目标包含：
  - 是否真实检测到；
  - 置信度；
  - bbox 中心点；
  - bbox 面积；
  - 运动速度；
  - 连续缺失时长；
  - 短遮挡补全标记；
- 关键目标之间的关系特征，例如距离、距离变化；
- 时间位置编码，例如当前帧在整段视频中的相对位置。

这些特征已经能把“检测框序列”转换成 `[T, F]` 的时序模型输入，其中：

- `T` 是视频抽帧后的时间步数量；
- `F` 是每帧特征维度；
- 每一行对应一帧；
- 每一列对应一个具体特征。

### 2.2 当前三种模型

#### MS-TCN + BiLSTM

当前 `ms_tcn.py` 的输入输出：

```text
输入:  x [batch, time, feature_dim]
输出: logits [batch, class_count, time]
```

结构大致是：

```text
LayerNorm
  -> Linear projection
  -> BiLSTM
  -> 多阶段 TCN refinement
  -> frame-wise logits
```

特点：

- BiLSTM 可以同时使用过去和未来帧，符合离线模型定位；
- TCN 通过膨胀卷积扩大时间感受野；
- 多阶段 refinement 可以逐步修正逐帧预测和边界；
- 适合作为第一主线模型。

#### ASFormer 风格模型

当前 `asformer.py` 的输入输出：

```text
输入:  x [batch, time, feature_dim]
输出: logits [batch, class_count, time]
```

结构大致是：

```text
LayerNorm
  -> Linear projection
  -> 局部膨胀卷积
  -> 多头自注意力
  -> 前馈网络
  -> frame-wise logits
```

特点：

- attention 能建模较长时间范围内的依赖；
- 局部卷积给模型加入“邻近帧更相关”的归纳偏置；
- 对长序列有潜力，但小数据下更容易过拟合；
- 适合作为强对照模型。

#### BiGRU

当前 `bigru.py` 的输入输出：

```text
输入:  x [batch, time, feature_dim]
输出: logits [batch, class_count, time]
```

结构大致是：

```text
LayerNorm
  -> Linear projection
  -> 多层双向 GRU
  -> 时序卷积分类头
  -> frame-wise logits
```

特点：

- 结构比 ASFormer 简单；
- 对小数据更稳；
- 训练速度通常更快；
- 当前实验中已经表现出较好的实用性。

## 3. 术语解释

### 3.1 ROI 是什么

ROI 是 `Region of Interest`，意思是“感兴趣区域”。

在本任务中，YOLO 检测框给出的每一个 bbox 都可以看作一个 ROI。例如：

```text
hand bbox
short_brush bbox
syringe bbox
air_gun bbox
scope_distal_end bbox
```

如果直接把整张视频帧送入视觉模型，背景、桌面、无关器械都会引入噪声。ROI 的思想是只裁剪出和动作相关的局部区域，让视觉模型重点看：

- 手是否抓住刷子；
- 短刷是否接触阀门口；
- 针筒是否正在推注；
- 气枪是否靠近远端口；
- 长刷/刷头是否出现在内镜远端。

### 3.2 ROI Backbone 是什么

Backbone 可以理解为“特征提取主干网络”。

ROI Backbone 就是对 ROI 图像块提取视觉特征的模型。例如：

```text
原始视频帧
  -> 根据 YOLO bbox 裁剪 ROI
  -> resize 到固定尺寸
  -> 送入 DINOv2 / ResNet / VideoMAE
  -> 得到 ROI appearance feature
```

ROI Backbone 的输出不是动作标签，而是一段向量，例如：

```text
hand_roi_feature: [384]
short_brush_roi_feature: [384]
syringe_roi_feature: [384]
```

这些向量描述 ROI 里的外观信息，比如纹理、颜色、形状、局部姿态、器械可见程度等。

### 3.3 DINOv2 是什么

DINOv2 是一种自监督视觉基础模型，常用于提取通用图像特征。

在这里可以把它理解成：

```text
单帧 ROI 图像 -> DINOv2 -> 外观向量
```

优点：

- 不需要你从零训练视觉 backbone；
- 对小数据任务比较友好；
- 对图像外观、物体形状、局部结构有较强表达能力；
- 适合作为第一版 ROI appearance feature 提取器。

局限：

- DINOv2 本身是图像模型，不直接理解一段视频里的运动；
- 它更适合补充“当前帧看起来像什么”，而不是单独判断“动作正在发生什么”。

因此建议第一版使用“冻结 DINOv2”：

```text
不训练 DINOv2 参数
只用它离线提取 ROI 特征
再把 ROI 特征交给时序模型学习动作
```

### 3.4 VideoMAE 是什么

VideoMAE 是一种视频 masked autoencoder。可以理解为专门处理短视频 clip 的视觉 backbone。

DINOv2 看的是单张图像，VideoMAE 看的是一小段连续帧：

```text
一段 ROI clip，例如 16 帧
  -> VideoMAE
  -> clip-level video feature
```

它通过遮挡视频中的大量 tube patch，让模型学习从剩余视频块重建被遮挡内容，从而学到视频时空特征。

优点：

- 能直接建模局部运动；
- 对刷洗、推注、插拔这类动态动作更有表达力；
- 理论上比单帧 DINOv2 更适合动作识别。

局限：

- 计算量更大；
- 训练和微调更复杂；
- 如果数据量不大，直接微调容易过拟合；
- ROI clip 的裁剪、对齐、缺失补全都比单帧 ROI 更麻烦。

因此建议不要第一步就上 VideoMAE 微调。更稳的路线是：

```text
第一阶段：DINOv2 单帧 ROI 特征
第二阶段：VideoMAE 冻结特征
第三阶段：必要时再尝试小规模微调
```

### 3.5 Missing 特征是什么

Missing 特征描述“某个目标多久没有被检测到了”。

例如短刷在某几帧被手遮挡，YOLO 没检测到：

```text
t=10 short_brush detected
t=11 short_brush missing
t=12 short_brush missing
t=13 short_brush detected
```

这时可以构造：

```text
short_brush_present:
  [1, 0, 0, 1]

short_brush_missing_age:
  [0, 1, 2, 0]
```

实际实现里通常会把 missing_age 归一化到 0-1。

它的意义是告诉模型：

- 当前没有检测框；
- 但这个目标刚刚出现过；
- 这可能是短遮挡，不一定代表目标真的消失。

### 3.6 Imputed 特征是什么

Imputed 是“补全/插补”的意思。

当目标短暂消失，但前后帧都有检测结果时，可以估计中间几帧的位置。例如：

```text
t=10 检测到 short_brush 在左侧
t=11 没检测到
t=12 没检测到
t=13 检测到 short_brush 在右侧
```

可以对 `t=11, t=12` 做线性插值，估计短刷位置。但要注意：

```text
present 仍然是 0
imputed 设为 1
```

这样模型能区分：

- `present=1`：真实检测到了；
- `imputed=1`：这是根据前后帧推测出来的；
- 两者不能混为一谈。

这对遮挡频繁的医院清洗场景很重要，因为手、刷子、内镜端口经常互相遮挡。

### 3.7 Track 特征是什么

Track 特征描述同一个目标在时间上的轨迹一致性。

YOLO 每帧独立检测，不天然知道“这一帧的短刷”和“上一帧的短刷”是不是同一个实体。Track 特征可以补充这种时间连续性。

常见 track 特征包括：

- `track_id`：目标轨迹编号；
- `track_age`：这个轨迹已经持续多少帧；
- `time_since_update`：这个轨迹多久没被新检测框更新；
- `track_confidence`：轨迹可靠程度；
- `velocity_x / velocity_y`：目标中心点移动速度；
- `acceleration_x / acceleration_y`：目标运动加速度；
- `track_switch_flag`：当前目标是否发生了轨迹切换。

在本任务中，track 特征尤其适合：

- 区分两只手；
- 稳定短刷/针筒/气枪这类小目标；
- 判断长刷插入和拔出方向；
- 降低短时漏检对动作边界的影响。

### 3.8 Box / Motion / Interaction / State Feature 分别是什么

#### Box Feature

描述单个目标 bbox 的空间信息：

```text
cx, cy, width, height, area, aspect_ratio, confidence
```

例如：

```text
short_brush_cx
short_brush_cy
short_brush_area
short_brush_conf
```

#### Motion Feature

描述目标随时间怎么移动：

```text
speed
velocity_x
velocity_y
acceleration
direction
distance_delta
```

例如长刷插入和拔出，关键不只是有没有长刷，而是它相对远端口的距离变化方向。

#### Interaction Feature

描述多个目标之间的关系：

```text
hand_to_short_brush_dist
syringe_to_scope_distal_end_dist
brush_tip_to_scope_distal_end_dist
short_brush_to_scope_control_body_dist
```

例如：

- 短刷靠近阀门口且手存在，更像 `short_brush_cleaning`；
- 针筒靠近远端口且手存在，更像 `flush`；
- 气枪靠近远端口，更像 `air_injection`；
- 刷头从远端冒出，更像长刷插入阶段结束。

#### State Feature

描述目标当前状态和检测质量：

```text
present
candidate_count
missing_age
imputed
track_age
time_since_update
occlusion_score
```

这些特征帮助模型理解：

- 当前检测是否可靠；
- 是否可能发生遮挡；
- 是否有多个候选框；
- 是否需要依赖前后文判断。

## 4. 原始方案可行性分析

你提出的方案是：

```text
RGB Video
      │
      ├──────────────► ROI Backbone（DINOv2 / VideoMAE）
      │                     │
      │                ROI Appearance Feature
      │
YOLO Detection
      │
      ├──► Box Feature
      ├──► Motion Feature
      ├──► Interaction Feature
      ├──► State Feature
      │
      ▼
Feature Fusion（MLP / Cross-Attention）
      │
      ▼
Temporal Encoder
(MS-TCN++ / ASFormer)
      │
      ▼
Frame-wise Action Prediction
      │
      ▼
Temporal Smoothing + Boundary Refinement
      │
      ▼
动作时序分割结果
```

整体判断：**可行，并且是当前 bbox-only baseline 的合理升级方向。**

原因：

1. 当前任务不是纯物体检测，而是动作分割。只知道“目标出现在哪里”不够，还需要知道“局部外观和运动状态”。
2. 医院清洗场景存在遮挡、小目标、长刷弱检测问题，RGB ROI 外观能补充 YOLO 的漏检和误检。
3. 离线模型可以使用完整视频前后文，适合使用 MS-TCN++ / ASFormer / BiGRU 这类时序模型。
4. 当前代码已经有三种时序模型和 FeatureStore-like 输入，只需要扩展特征生成，不需要推翻主流程。

但不建议一次性全部上：

```text
VideoMAE + Cross-Attention + ASFormer + Boundary Head
```

因为这会同时增加四类变量，很难判断效果提升来自哪里，也容易在小数据下过拟合。

## 5. 推荐的分阶段实现路线

### 阶段 0：保留当前 YOLO v2 baseline

当前已经完成：

```text
YOLO detections
  -> v2 structured feature
  -> ms_tcn / asformer / bigru
  -> frame-wise prediction
  -> SegmentFact
```

这阶段的目标是作为所有后续实验的对照组。

建议固定以下内容：

- 数据集划分；
- 类别映射；
- 抽帧 fps；
- 评价指标；
- 当前三模型最佳特征组合。

这样后续每次新增模块，都能和当前 baseline 公平比较。

### 阶段 1：加入 DINOv2 ROI Appearance Feature

第一步建议只加入单帧 ROI 外观特征。

数据流：

```text
RGB frame + YOLO bbox
  -> crop ROI
  -> DINOv2 frozen backbone
  -> ROI appearance feature
  -> object-level pooling
  -> concat 到当前 v2 feature
  -> 三种时序模型训练
```

关键设计：

#### ROI 裁剪

对每个检测框裁剪 ROI：

```text
frame[y1:y2, x1:x2]
```

建议加一点 padding，例如 bbox 宽高的 10%-20%，因为动作通常发生在物体和周围结构之间。

例如短刷刷洗阀门口时，如果只裁剪短刷本身，可能看不到阀门口；加 padding 后能看到接触关系。

#### ROI resize

统一 resize 到 backbone 输入尺寸，例如：

```text
224 x 224
```

#### ROI feature pooling

每帧每类目标可能有多个 ROI。建议：

- `hand`：保留 top-2 两只手的 ROI feature；
- 其它目标：保留 top-1 ROI feature；
- 如果某类目标缺失：ROI feature 全 0，并用 `present/missing/imputed` 告诉模型这是缺失。

不要直接对同类多框做加权平均，尤其是两只手不能平均成“一只手”。

#### 特征维度控制

DINOv2 输出维度可能较大，例如 384/768/1024。不要直接把所有 ROI feature 原样拼到时序模型里，否则 `[T, F]` 的 F 会暴涨。

建议加一个投影层或离线 PCA：

```text
roi_feature_raw: [384]
  -> Linear / PCA
roi_feature_small: [32 或 64]
```

然后再拼接：

```text
fused_frame_feature = concat(v2_feature, roi_feature_small)
```

### 阶段 2：加入质量感知融合

当同时有 YOLO 结构化特征和 ROI 外观特征时，不能默认两者同等可靠。

例如：

- YOLO 置信度很高、目标没遮挡时，结构化特征更可信；
- YOLO 漏检或低置信时，ROI 外观和前后文更重要；
- 长刷难检测时，可能更依赖 brush_tip_out、远端口附近 ROI 外观、手部运动。

建议第二阶段使用 gated fusion。

输入：

```text
det_feat: 当前 v2 / window_stats / business_priors 特征
roi_feat: DINOv2 ROI 外观特征
quality_feat: 置信度、missing_age、imputed、candidate_count、track_age
```

融合方式：

```text
gate = MLP(quality_feat)
fused = gate * roi_feat + (1 - gate) * det_feat_projection
```

也可以先做简单版本：

```text
fused = MLP(concat(det_feat, roi_feat, quality_feat))
```

建议先做 concat + MLP，再做 gated fusion。不要一开始就上 Cross-Attention。

### 阶段 3：尝试 Cross-Attention

Cross-Attention 可以让一个模态主动查询另一个模态。

在本任务中可以设计为：

```text
Query: YOLO structured feature
Key/Value: ROI appearance feature
```

含义：

```text
模型先知道当前有哪些物体、位置关系如何，
再去 RGB ROI 特征里查询哪些外观信息对当前动作有帮助。
```

也可以反过来：

```text
Query: ROI appearance feature
Key/Value: YOLO structured feature
```

但建议优先用 YOLO 特征做 Query，因为当前任务强依赖业务对象关系。

Cross-Attention 的风险：

- 参数更多；
- 数据少时容易过拟合；
- debug 成本更高；
- 如果 ROI 特征质量不稳定，attention 可能学到噪声。

因此建议放在 DINOv2 + MLP/gated fusion 之后再尝试。

### 阶段 4：尝试 VideoMAE clip feature

当单帧 ROI 外观特征仍不足时，可以尝试 VideoMAE。

输入不再是单帧 ROI，而是短 clip：

```text
t-7 到 t+8 的 ROI clip
  -> VideoMAE
  -> 当前帧对应的 video appearance feature
```

适合提升：

- 短刷往复刷洗；
- 针筒推注；
- 气枪注气；
- 长刷插入/拔出的动态趋势。

但实现复杂度更高，需要处理：

- ROI clip 的跨帧对齐；
- 目标缺失时的 ROI 位置补全；
- clip 长度选择；
- 显存和训练时间；
- 是否冻结 backbone。

建议第一版只冻结 VideoMAE，离线预提取 clip feature，不直接端到端训练。

### 阶段 5：边界 refinement

当前时序模型输出是 frame-wise action prediction：

```text
每一帧 -> 一个动作类别
```

然后通过连续相同类别合并成动作段。

如果想提升边界精度，可以加入 boundary refinement。

可选方案：

#### 方案 A：后处理平滑

规则：

- 去掉长度太短的孤立片段；
- 对低置信度短段合并到邻近高置信片段；
- 对动作段起止点附近做小范围搜索。

优点：简单、稳定。

缺点：上限有限，容易变成手工规则。

#### 方案 B：增加 boundary head

模型同时输出：

```text
action_logits: [T, class_count]
boundary_logits: [T, 2]  # 是否为动作开始/结束附近
```

训练时从标注段生成边界标签。

推理时：

```text
frame-wise action prediction
  -> 初始 segment
  -> 在 segment 起止点附近找 boundary score 高的位置
  -> 修正 start/end
```

优点：更贴合“精确边界”的目标。

缺点：需要标注边界质量较高，否则会学习标注噪声。

## 6. 对三种现有模型分别怎么改

### 6.1 MS-TCN + BiLSTM

建议作为第一主线模型。

当前：

```text
x [B, T, F]
  -> BiLSTM
  -> MS-TCN stages
  -> logits [B, C, T]
```

后续改法：

```text
det_feature [B, T, F_det]
roi_feature [B, T, F_roi]
quality_feature [B, T, F_q]
  -> fusion MLP / gate
  -> fused_feature [B, T, F_new]
  -> BiLSTM
  -> MS-TCN++
  -> logits
```

建议优先级：

1. 加 DINOv2 ROI feature；
2. 加 gated fusion；
3. 改 MS-TCN 为更标准 MS-TCN++；
4. 加 boundary head。

原因：

- MS-TCN 类模型本身适合 action segmentation；
- 多阶段 refinement 对边界有帮助；
- BiLSTM 可以补充完整序列上下文。

### 6.2 ASFormer

适合作为强对照模型。

当前：

```text
x [B, T, F]
  -> local conv
  -> multi-head self-attention
  -> logits
```

后续改法：

```text
det_feature + roi_feature
  -> fusion
  -> ASFormer encoder
  -> decoder/refinement
  -> logits
```

建议：

- 不要一开始把 Cross-Attention 和 ASFormer 同时加满；
- 先让 ASFormer 吃 MLP 融合后的统一特征；
- 如果效果稳定，再把 Cross-Attention 放在 ASFormer 前面或内部。

注意：

ASFormer 对小数据更敏感，需要重点控制：

- hidden dim；
- attention heads；
- dropout；
- weight decay；
- early stopping；
- 类别不平衡 loss。

### 6.3 BiGRU

适合作为轻量强 baseline。

当前：

```text
x [B, T, F]
  -> BiGRU
  -> temporal conv head
  -> logits
```

后续改法：

```text
det_feature + roi_feature + quality_feature
  -> fusion
  -> BiGRU
  -> temporal conv head
  -> logits
```

建议：

- 第一批多模态实验可以先从 BiGRU 开始；
- 它训练快、稳定、方便判断 ROI feature 是否真的有用；
- 如果 BiGRU 加 ROI feature 都没有提升，说明问题可能在 ROI 提取或数据标注，而不是复杂模型不够强。

## 7. 推荐的数据格式扩展

当前 FeatureStore-like npz 可以继续保留。

建议新增字段时，不破坏原字段：

```text
features:        [T, F_total]
labels:          [T]
timestamps:      [T]
feature_names:   [F_total]
metadata:        dict/json
```

新增多模态后，可以考虑：

```text
det_features:    [T, F_det]
roi_features:    [T, F_roi]
quality_features:[T, F_quality]
features:        [T, F_total]  # 训练时实际使用的拼接/融合输入
feature_names:   [F_total]
```

如果想保持最小改动，也可以仍然只保存一个 `features`：

```text
features = concat(det_features, roi_features, quality_features)
```

但建议至少在 metadata 中记录：

```text
feature_version
feature_method
roi_backbone
roi_feature_dim
roi_pooling_method
```

例如：

```json
{
  "feature_version": "clean_bbox_v3_roi_dinov2",
  "feature_method": "v2+dinov2_roi+quality",
  "roi_backbone": "dinov2_vits14_frozen",
  "roi_feature_dim": 64,
  "roi_pooling": "hand_top2_other_top1"
}
```

这样后续权重加载时可以检查：

```text
checkpoint.feature_version == input.feature_version
checkpoint.feature_names == input.feature_names
```

避免训练和推理特征不一致。

## 8. 五类动作的特征设计建议

### 8.1 long_brush_insert

问题：

- 长刷本体难检测；
- 有时只能看到刷头或远端冒出；
- 插入动作持续时间可能长，中间存在遮挡。

建议特征：

- `brush_tip_out_present / imputed`；
- `brush_tip_out_to_scope_distal_end_dist`；
- `brush_tip_out_to_scope_distal_end_delta`；
- `hand_to_scope_mid_section_dist`；
- 长刷或刷头 ROI appearance；
- 远端口附近 ROI appearance；
- track 方向特征。

关键思想：

```text
不要只依赖 long_brush bbox。
要结合 brush_tip_out、scope_distal_end、hand、运动方向和远端 ROI 外观。
```

### 8.2 long_brush_withdraw

与 insert 共享很多特征，但方向相反。

建议重点使用：

- 刷头/长刷相对远端口的距离变化方向；
- 手和管口的相对运动；
- 刷头出现后逐渐远离远端口的趋势；
- 中心窗口统计，利用未来帧确认“这是拔出而不是插入中间遮挡”。

### 8.3 short_brush_cleaning

问题：

- 短刷较小；
- 手容易遮挡；
- 需要判断是否真的在刷阀门口，而不是拿着短刷移动。

建议特征：

- `short_brush_present / imputed`；
- `hand_top1/top2`；
- `short_brush_to_scope_control_body_dist`；
- `short_brush_speed`；
- `short_brush_to_scope_control_body_delta`；
- 短刷 ROI appearance；
- 阀门口/控制部附近 ROI appearance；
- 局部周期性运动特征。

关键思想：

```text
短刷刷洗 = 短刷出现 + 靠近阀门口 + 手参与 + 局部往复运动。
```

### 8.4 flush

问题：

- 针筒小，容易和气枪/手混淆；
- 推注动作不一定有明显大幅移动。

建议特征：

- `syringe_present / conf / imputed`；
- `syringe_to_scope_distal_end_dist`；
- `hand_to_syringe_dist`；
- syringe ROI appearance；
- 远端口附近 ROI appearance；
- 针筒活塞区域外观变化，如果画面分辨率允许。

关键思想：

```text
flush 更像“针筒靠近端口并保持一段时间”，运动幅度可能不大。
```

### 8.5 air_injection

问题：

- 气枪和针筒可能交替出现；
- 外观相似时，YOLO 误检会影响动作判断。

建议特征：

- `air_gun_present / conf / imputed`；
- `air_gun_to_scope_distal_end_dist`；
- air_gun ROI appearance；
- 与 syringe ROI appearance 做区分；
- `candidate_count` 和 `track_consistency`，避免单帧误检造成短假段。

关键思想：

```text
air_injection 依赖气枪外观和端口接近关系，建议加入 ROI 外观帮助区分 syringe。
```

## 9. 训练策略建议

### 9.1 先冻结视觉 backbone

第一版不要训练 DINOv2 / VideoMAE，只预提取特征。

原因：

- 数据量小；
- 端到端训练显存压力大；
- debug 难度高；
- 先证明 ROI feature 有用更重要。

### 9.2 控制特征维度

建议：

```text
DINOv2 raw dim 384/768
  -> PCA 或 Linear projection
  -> 32/64 dim per object slot
```

避免时序模型输入维度过大导致：

- 训练变慢；
- 小数据过拟合；
- checkpoint 和后端部署复杂。

### 9.3 保留逐步消融实验

每次只改一类变量：

```text
baseline v2
v2 + DINOv2 ROI
v2 + DINOv2 ROI + quality feature
v2 + DINOv2 ROI + gated fusion
v2 + DINOv2 ROI + boundary head
```

不要一次性加入所有模块。

### 9.4 类别不平衡处理

ActionMixed 中不同动作数量可能不均衡，尤其 `air_injection` 在某些划分中可能 support 很少。

建议：

- 使用 class-weighted CE；
- 或 focal loss；
- 保证 train/val/test 都覆盖五类动作；
- 报告每类 Precision / Recall / F1，而不是只看总体 ACC。

### 9.5 评价指标

继续保留：

```text
ACC
Precision
Recall
Frame-F1
F1@0.25
F1@0.5
```

其中：

- ACC 容易被 idle 或长动作主导；
- Recall 能看出某类动作是否漏检；
- Precision 能看出是否误报；
- F1@0.25 / F1@0.5 更贴近动作段边界质量。

建议新增：

```text
Boundary Error Mean
Boundary Error Median
Per-class Segment F1
```

用于回答：

```text
模型到底是分类错了，还是边界不准？
```

## 10. 工程实现建议

### 10.1 新增 ROI 特征提取脚本

建议新增：

```text
scripts/extract_roi_features.py
```

职责：

```text
读取 ActionMixed 图像帧 + YOLO 标签
  -> 裁剪 ROI
  -> DINOv2 提特征
  -> 保存 roi feature cache
```

输出可以是：

```text
input/cache/roi_features/<video_id>.npz
```

每个文件包含：

```text
timestamps
object_slots
roi_features
roi_feature_names
```

### 10.2 修改 data_transfer / dataset

在 `data_transfer.py` 或 `dataset.py` 中增加可选参数：

```text
--roi-feature-root input/cache/roi_features
--feature-method v2+dinov2_roi
```

逻辑：

```text
基础 v2 特征
  + 读取对应视频的 ROI feature
  + 对齐时间戳 / frame_id
  + 拼接成新 features
```

不要破坏原 `v2` 流程。

### 10.3 修改训练配置

训练脚本需要记录：

```text
feature_method
feature_version
feature_dim
roi_backbone
roi_feature_dim
```

checkpoint 中也要保存这些字段，方便后端接入时校验。

### 10.4 后端接入注意

后端 `CleanSightBackend` 目前的离线模型接入方式更偏向：

```text
FrameDetections -> clean.py 内部特征转换 -> torch model -> SegmentFact
```

如果后续使用 ROI appearance feature，后端需要能拿到视频帧或 ROI cache。否则只有 `features.jsonl` 里的检测框，不足以运行 DINOv2/VideoMAE。

可选路线：

#### 路线 A：offline-model 预提取 ROI feature，后端只加载融合后的 features

适合研究阶段。

```text
offline-model
  -> 生成完整 features npz
  -> 训练和验证
```

后端暂不接 RGB backbone。

#### 路线 B：后端离线 worker 读取视频帧并提 ROI feature

适合正式接入。

```text
FeatureStore detections
  + 原始视频/帧路径
  -> ROI Backbone
  -> fused features
  -> CleanSegmenter
```

这需要后端离线链路能访问：

- 原视频；
- 抽帧结果；
- bbox；
- 模型权重；
- ROI backbone 权重。

复杂度明显更高。

建议先走路线 A，把模型效果证明清楚，再决定是否后端接 ROI backbone。

## 11. 推荐优先级

### 第一优先级

```text
YOLO v2 + DINOv2 ROI feature + concat MLP + BiGRU / MS-TCN
```

原因：

- 改动最小；
- 最容易验证 ROI 外观是否有增益；
- BiGRU/MS-TCN 已经在当前仓库内可训练；
- 不需要复杂 attention 融合。

### 第二优先级

```text
YOLO v2 + DINOv2 ROI feature + quality-aware gated fusion + MS-TCN++
```

原因：

- 解决 YOLO 漏检/低置信时的特征信任问题；
- MS-TCN++ 更贴近 action segmentation 主流 baseline；
- 对边界 refinement 更友好。

### 第三优先级

```text
VideoMAE ROI clip feature + ASFormer / Cross-Attention
```

原因：

- 理论上更强；
- 但实现成本和过拟合风险更高；
- 应该放在前两阶段稳定之后。

## 12. 预期收益与风险

### 预期收益

- 提升遮挡场景下的动作识别；
- 提升小目标动作，如 `flush`、`air_injection`；
- 提升长刷弱检测场景的上下文判断；
- 提升动作段边界精度；
- 减少单帧 YOLO 漏检造成的断段。

### 主要风险

- ROI 特征提取耗时大；
- ROI 裁剪质量依赖 YOLO bbox；
- 数据量小，复杂融合模型容易过拟合；
- VideoMAE 需要更大显存和更复杂的数据对齐；
- 后端正式接入 RGB backbone 需要额外视频帧访问能力。

## 13. 推荐实验表

建议后续按下面顺序跑实验：

| 阶段 | 特征 | 融合 | 模型 | 目的 |
|---|---|---|---|---|
| 0 | v2 | 无 | BiGRU / MS-TCN / ASFormer | 当前 baseline |
| 1 | v2 + DINOv2 ROI | concat + MLP | BiGRU | 验证 ROI 外观是否有效 |
| 2 | v2 + DINOv2 ROI | concat + MLP | MS-TCN + BiLSTM | 验证强时序模型收益 |
| 3 | v2 + DINOv2 ROI + quality | gated fusion | MS-TCN++ | 处理检测可靠性变化 |
| 4 | v2 + DINOv2 ROI + boundary | gated fusion | MS-TCN++ | 优化边界 |
| 5 | v2 + VideoMAE ROI clip | concat / gate | ASFormer | 验证视频外观特征 |
| 6 | v2 + DINOv2/VideoMAE | Cross-Attention | ASFormer | 尝试复杂多模态融合 |

每一组都需要记录：

```text
ACC
Precision
Recall
Frame-F1
F1@0.25
F1@0.5
每类动作 Precision/Recall/F1
每类动作 Segment F1
边界误差
```

## 14. 结论

推荐后续不要直接推翻当前三模型 baseline，而是在当前 FeatureStore-like 输入和三种时序模型基础上逐步增加特征来源：

```text
当前阶段:
YOLO structured feature
  -> MS-TCN / ASFormer / BiGRU

下一阶段:
YOLO structured feature
  + DINOv2 ROI appearance feature
  -> MLP / gated fusion
  -> MS-TCN / ASFormer / BiGRU

再下一阶段:
YOLO structured feature
  + VideoMAE ROI clip feature
  + boundary refinement
  -> 更高边界精度的离线动作分割模型
```

最稳的第一步是：

```text
冻结 DINOv2
离线预提取 ROI appearance feature
拼接到当前 v2 特征
先用 BiGRU 和 MS-TCN + BiLSTM 做消融
```

如果这一阶段能提升 `F1@0.25 / F1@0.5`，再继续尝试 gated fusion、MS-TCN++、ASFormer 和 VideoMAE。

## 15. 参考资料

- DINOv2: Learning Robust Visual Features without Supervision
  https://arxiv.org/abs/2304.07193
- VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training
  https://arxiv.org/abs/2203.12602
- MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
  https://arxiv.org/abs/1903.01945
- MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation
  https://arxiv.org/abs/2006.09220
- ASFormer: Transformer for Action Segmentation
  https://arxiv.org/abs/2110.08568
