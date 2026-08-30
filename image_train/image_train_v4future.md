# image_train v4 future 设计分析记录

生成时间：2026-07-23

本文记录面向 `image_train` 后续 v4 改进的设计分析。2026-07-28 讨论后，v4 目标做一次明确调整：训练阶段暂时以逐帧分类准确率为主要目标，动作段划分、动作段内平滑、短段删除和相邻同类段合并放到模型输出后的后处理阶段处理。段级 F1、段数误差和边界误差仍然保留为重要评估指标，但不再作为本轮训练的主优化目标。

## 1. 背景和目标

当前 `image_train` 已经完成 v1/v2/v3 三轮实验：

- v1：在 YOLO 结构化特征基础上加入轻量 RGB ROI 统计。
- v2：加入可复用 ROI cache，并把 ROI RGB 特征改为质量感知、局部平滑和 delta 形式。
- v3：以动作段划分为主目标，重点评估 segment F1、预测段数误差和边界误差，而不是只看 frame-level ACC。

v4 当前更核心的目标是：先让每一帧的动作类别尽可能判准，再通过参数化后处理把逐帧 logits/probabilities 转成更稳定的动作段。原因是当前数据集仍以切分片段为主，每段通常只有一种动作加前后少量 idle，直接把段级目标作为主训练目标容易让验证集后处理过拟合；逐帧准确率更适合作为当前阶段的主训练信号。

用户提出的三个问题是：

1. 长毛刷因为形状问题没有进行稳定 YOLO 检测，是否有必要对整个洗槽位置提取 RGB 特征，以获得可能的长毛刷信息。
2. 当前 BiGRU 把所有特征直接拼进一个特征向量里，没有显式体现“哪些特征和哪些动作更相关”。
3. 短毛刷在实际视频中难以识别，基本是缺位状态，是否可以通过 RGB 特征弥补，比如把手的检测框适当扩大作为另一个 RGB 特征。

本文的结论是：三个方向都值得做，但不建议简单粗暴地增加更多 RGB 均值维度。更合理的 v4 方向是：

- 对长毛刷增加 `wash_tank_roi` / `scope_channel_roi` 这类场景固定 ROI 的视觉 proxy。
- 对短毛刷增加 `expanded_hand_roi` / `hand_ring_roi` / `hand_target_union_roi` 这类手部周边 proxy。
- 对 BiGRU 和其它模型输入增加分组、门控和动作先验，让模型知道不同特征组更倾向服务于不同动作类别。

## 2. 相关论文和方法启发

### 2.1 时序动作分割关注的是长视频逐帧标注和动作段边界

MS-TCN 这类方法把 temporal action segmentation 定义为对长视频中的每一帧预测动作类别，并通过多阶段时序卷积逐步 refine 预测结果。用户提供的本地参考文献 `F:/暑期实习/参考文献/CS-TCN.pdf` 实际内容是 MS-TCN 论文。该论文的训练主体仍是逐帧交叉熵分类损失，同时额外加入 truncated MSE smoothing loss 来减少过分割。论文实验也说明，不同 stage 的 frame-wise accuracy 可能接近，但 segmental F1 和 edit score 差异很大，因此“帧分类”和“动作段质量”需要同时报告，但可以在工程上分成训练目标与后处理目标。

参考：

- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/1903.01945)
- [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/2006.09220)

ASFormer 则强调 action segmentation 常见于分钟级长视频，模型输入通常是预提取的 frame-wise feature sequence，而不是直接端到端处理原始 RGB。它还指出小数据场景下需要局部连接、层级结构等归纳偏置，避免 Transformer 在小数据上学到不稳定关系。

参考：

- [ASFormer: Transformer for Action Segmentation](https://arxiv.org/abs/2110.08568)

这些论文给当前任务的启发是：

- 当前 v4 训练主目标可以先优化单帧分类 ACC，但报告不能只看 ACC；动作段边界、段数和 segment F1 仍作为后处理质量与上线风险指标。
- 输入特征必须能表达动作持续过程中的稳定证据和动作切换处的变化证据。
- 小数据场景下，不应完全依赖模型自己从大而扁平的向量里学习业务关系，应该在特征组织和先验上给模型更强约束。
- 对当前 v4，本轮先把逐帧交叉熵和验证集 frame accuracy 作为训练主目标；平滑、短段删除、合并短 idle gap 等放在输出后处理中做参数搜索。

### 2.2 Object-centric / ROI 表示适合人-工具-对象交互动作

当前动作本质不是普通人体姿态分类，而是人手、工具和内镜部位之间的交互。例如：

- 手 + 长毛刷/刷头 + 远端口/管腔方向：长刷插入或拔出。
- 手 + 短毛刷 + 控制部/阀门附近：短刷清洗。
- 手 + 针筒 + 远端口：冲洗。
- 手 + 气枪 + 远端口：注气。

Object-centric video representation 相关研究强调，物体是理解人-物交互和预测动作的重要线索。尤其在新环境、小数据、工具变化或遮挡较多时，围绕关键对象提取表示通常比整帧全局特征更稳。

用户提供的本地参考文献 `F:/暑期实习/参考文献/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf` 也支持这一判断。该论文研究第一人称手部动作识别，数据包含 RGB-D、手部 3D pose 和物体交互，实验结论强调手部姿态/手-物交互线索对细粒度动作识别有明显价值。这和当前内镜清洗任务高度相似：动作并不只由工具类别决定，还由手是否握持、手在什么位置、手是否靠近目标部位、手周围是否出现工具外观共同决定。

参考：

- [Object-Centric Video Representation for Long-Term Action Anticipation](https://arxiv.org/abs/2311.00180)
- [Is an Object-Centric Video Representation Beneficial for Transfer?](https://arxiv.org/abs/2207.10075)
- [Symbiotic Attention for Egocentric Action Recognition with Object-centric Alignment](https://yu-wu.net/pdf/TPAMI2020_Egocentric-Action.pdf)

这些论文给当前任务的启发是：

- RGB 特征不应只取整帧，也不应只依赖 YOLO 检测到的对象框。
- 对稳定可检测对象，可以继续使用 YOLO bbox ROI。
- 对难检测对象，例如长毛刷、短毛刷，应该设计 proxy ROI，即不直接检测对象本身，而从对象最可能出现的位置、交互对象周边、手周边和固定工作区域中提取证据。
- 对当前 v4，优先使用低维 proxy RGB/motion/edge 统计，而不是直接引入 DINOv2/VideoMAE 高维 embedding；这样更容易在小数据上控制过拟合，并保持三模型通用输入。

### 2.3 动作识别综述对当前风险的提示

用户提供的本地参考文献 `F:/暑期实习/参考文献/s11263-022-01594-9.pdf` 是 Human Action Recognition and Prediction 综述。该综述指出，真实视频动作识别常见困难包括相似动作之间的混淆、复杂背景、摄像机运动、光照变化和标注数据不足。

这些问题在当前任务中对应为：

- `flush` 和 `air_injection` 都可能表现为工具靠近远端口并保持相对稳定。
- `short_brush_cleaning` 和长毛刷插拔都可能表现为一只手稳定、另一只手移动。
- 洗槽水面反光可能污染 RGB 特征。
- 短毛刷和长毛刷检测缺失会造成结构化特征长期为 0。
- 当前切分片段数据规模较小，复杂模型或过强后处理都容易过拟合。

因此 v4 应该采用可参数化、可消融、可回退的工程设计：新增特征组可以通过命令行开关启用/关闭，训练调参也应集中成参数，方便后续系统搜索。

### 2.4 边界检测和动作段级建模作为后处理与后续目标

一些新的 temporal action segmentation 方法会引入 boundary-aware、segment-level 或 query-based 结构，目的不是单纯提高每帧分类，而是减少过分割、修正动作段边界、对动作段整体做 denoising。

参考：

- [Temporal Segment Transformer for Action Segmentation](https://arxiv.org/abs/2302.13074)
- [Efficient Temporal Action Segmentation via Boundary-aware Query](https://arxiv.org/abs/2405.15995)

这些论文给当前任务的启发是：

- v4 不应只扩展 frame feature，还应考虑动作边界特征。
- RGB proxy 的 `delta`、局部运动能量、工具出现/消失变化，比静态均值更可能帮助判断动作起止。
- 如果暂时不大改模型结构，也可以先把 boundary-like features 加入输入，或在后处理阶段使用动作类别先验和边界证据。

## 3. 问题一：长毛刷未检测时，是否提取整个洗槽 RGB

### 3.1 判断

有必要提取，但不建议只提取“整个洗槽 RGB 均值”。

长毛刷的难点是：

- 形状细长，YOLO bbox 容易不稳定。
- 可能被手、内镜管道、槽体边缘遮挡。
- 颜色和背景可能接近，单帧外观不明显。
- 插入和拔出动作的区别更依赖方向、轨迹和时序变化，而不是某一帧是否检测到完整长毛刷。

因此，长毛刷更适合通过 proxy features 表达：

- 洗槽/管腔区域内是否出现细长结构。
- 该区域是否有沿管腔方向的运动。
- 远端口或刷头附近是否出现明显纹理、亮度、颜色变化。
- 手部运动是否和这些区域变化同步。

### 3.2 不建议只用整个洗槽 ROI 均值的原因

如果只对整个洗槽区域计算 RGB 均值、亮度均值、饱和度均值，可能会有几个问题：

- 洗槽面积较大，长毛刷占比小，平均值会被背景、水面、反光淹没。
- 光照变化、水面反射、器械阴影可能比长毛刷本身更强。
- 均值特征缺少方向信息，难以区分插入和拔出。
- 均值特征缺少局部变化信息，难以判断动作段边界。

所以“整个洗槽 ROI”可以作为一个粗粒度背景变化特征，但不能作为长毛刷信息的唯一来源。

### 3.3 推荐的长毛刷 proxy ROI 设计

#### 3.3.1 `wash_tank_roi`

固定洗槽大区域 ROI。可以人工配置一个相对坐标框，或者从视频标定信息中读取。

建议提取：

- `wash_tank_brightness_mean`：洗槽区域亮度均值，用于感知整体照明和大面积变化。
- `wash_tank_brightness_std`：亮度标准差，用于感知纹理、边缘和反光变化。
- `wash_tank_saturation_proxy`：颜色差异均值，用于感知工具、手套、液体或背景颜色变化。
- `wash_tank_rgb_mean`：RGB 三通道均值，作为基础外观特征。
- `wash_tank_rgb_delta`：当前帧与上一帧 RGB 均值差，用于动作切换或遮挡变化。
- `wash_tank_frame_diff`：当前 ROI 和上一帧 ROI 的像素差平均值，用于整体运动能量。

用途：

- 判断洗槽区域是否有操作活动。
- 给长刷动作、短刷动作提供背景视觉上下文。
- 辅助识别 idle 和真实操作段边界。

局限：

- 对长毛刷本身不够敏感。
- 容易受水面和光照影响。

#### 3.3.2 `wash_tank_strip_roi`

把洗槽区域按长毛刷常经过的位置划分成若干条带。例如：

- 左/中/右三条纵向 strip。
- 上/中/下三条横向 strip。
- 沿内镜管道方向的一条主通道 strip。
- 远端口附近 strip。

建议提取：

- 每个 strip 的 RGB/亮度/饱和度统计。
- 每个 strip 的亮度标准差。
- 每个 strip 的 frame difference。
- 相邻 strip 之间的差值。
- 主通道 strip 与洗槽整体 ROI 的差值。

用途：

- 捕捉长毛刷在洗槽中某条路径上的局部变化。
- 如果长毛刷只占小区域，strip 特征比整体均值更容易保留信号。
- 可用于判断长毛刷是否沿固定路径移动。

#### 3.3.3 `scope_channel_roi`

如果能人工标定内镜管道或刷子进入路径，可以定义一个更窄的通道 ROI。

建议提取：

- `channel_edge_density`：边缘密度，使用 Sobel/Canny 或轻量梯度计算。
- `channel_dominant_orientation`：主方向边缘比例，用于捕捉细长工具方向。
- `channel_motion_energy`：通道内帧差运动能量。
- `channel_texture_energy`：局部纹理强度。
- `channel_line_like_ratio`：近似线状结构占比，可由边缘连通域、Hough-like 简化特征或方向梯度统计得到。

用途：

- 作为 `long_brush` 缺失时的主要视觉 proxy。
- 帮助区分长毛刷插入/拔出和普通手部移动。
- 对动作段边界更敏感，因为刷子进入/离开通道时通道局部变化会明显。

### 3.4 和当前长毛刷特征的关系

当前 v3 已经有以下长毛刷相关线索：

- `long_brush_*`：但当前检测不稳定，甚至可能长期缺失。
- `brush_tip_out_*`：刷头外露检测，比完整长毛刷更关键。
- `brush_tip_out_to_scope_distal_end_*`：刷头到远端口的距离和变化。
- `prior_long_signal_near_scope`
- `prior_long_towards_distal`
- `prior_long_away_distal`
- `prior_hand_long_contact`
- `roi_rgb_v2_brush_tip_out_*`
- `roi_rgb_v2_scope_distal_end_*`

v4 可以新增：

- `wash_tank_roi_*`
- `wash_tank_strip_roi_*`
- `scope_channel_roi_*`
- `scope_channel_motion_*`
- `scope_channel_edge_*`

这些不是替代现有 `brush_tip_out`，而是在 `brush_tip_out` 缺失、长毛刷无法被 YOLO 检出时补充弱证据。

### 3.5 预期收益

对 `long_brush_insert`：

- 更容易捕捉刷子进入通道前后的视觉变化。
- 若结合方向 delta，有助于判断插入趋势。

对 `long_brush_withdraw`：

- 更容易捕捉刷子退出时通道和远端附近的变化。
- 如果结合刷头出现/消失和通道运动方向，可能改善 insert/withdraw 混淆。

对 idle：

- 当手或检测框存在但洗槽/通道区域没有明显动作变化时，可降低误报动作段的概率。

### 3.6 风险

- 固定 ROI 对摄像机位置敏感。若不同视频视角差异大，需要做视频级 ROI 标定。
- 洗槽反光和水面运动可能造成误判。
- 如果只用统计均值，新增特征维度增加但信号弱，可能导致过拟合。
- 如果使用复杂视觉特征，如 DINOv2/VideoMAE，需要更多算力和更严格缓存管理。

## 4. 问题二：BiGRU 扁平特征没有体现特征-动作关联性

### 4.1 判断

这个问题是成立的。

当前 `bigru` 输入是一个逐帧向量 `[T, F]`。v3 中它包含：

- YOLO 基础结构化特征。
- 非 RGB 的窗口统计特征。
- 业务先验特征。
- ROI RGB v2 特征。

这些特征被直接拼接后输入 BiGRU。理论上 BiGRU 可以学习特征之间的关联，但当前任务存在几个不利条件：

- 数据量小。
- 很多关键工具检测缺失。
- 动作之间存在相似手部运动。
- 当前数据集是切分片段，不是完整真实使用视频，动作上下文不完整。
- RGB 特征是低维统计，不是强语义视觉 embedding。

在这种情况下，模型可能学到一些偶然相关，而不是稳定的动作机制。例如：

- “一只手握控制端，另一只手来回移动”可能对应长毛刷插入/拔出，也可能对应短毛刷刷洗。
- 但这种模式通常不像注气或冲洗。
- 如果模型没有显式的动作候选约束，可能会把局部相似运动误分到错误类别。

### 4.2 当前 flat vector 的主要问题

#### 4.2.1 特征语义被混在一起

手部、工具、位置关系、RGB 外观、业务先验、窗口统计都拼成同一个向量。模型知道数值，不知道这些特征天然属于不同证据组。

例如：

- `hand_top1_speed`
- `short_brush_speed`
- `syringe_to_scope_distal_end_dist`
- `roi_rgb_v2_air_gun_brightness_std`
- `prior_long_towards_distal`

这些特征在输入里只是不同维度，但它们对动作类别的意义完全不同。

#### 4.2.2 没有动作条件化

某些特征天然只对少数动作有意义：

- `air_gun_*` 主要服务于 `air_injection`。
- `syringe_*` 主要服务于 `flush`。
- `short_brush_*` 主要服务于 `short_brush_cleaning`。
- `brush_tip_out_*` 和通道 ROI 更服务于 `long_brush_insert/withdraw`。

flat vector 没有显式告诉模型这些对应关系。

#### 4.2.3 检测缺失时信任度不明确

虽然当前有 `present/conf/missing_age/imputed`，但模型仍需要自己学习“某个对象缺失时该对象相关特征可信度下降”。小数据下，这种学习并不稳定。

#### 4.2.4 对动作段边界缺乏专门建模

BiGRU 做逐帧分类，最终再通过后处理合并动作段。若输入特征没有明确的边界变化信号，模型容易产生短段抖动或边界延迟。

### 4.3 推荐改成分组特征结构

v4 可以保持三种主模型架构不变，但在输入前增加 feature grouping。也就是仍然输出 `[T, F]` 给模型，但生成过程和中间融合要按组组织。

建议分组：

#### 4.3.1 `hand_interaction_group`

包含：

- 手数量。
- hand top-1/top-2 present/conf/cx/cy/area/speed。
- 双手距离。
- 双手相对速度。
- 一手稳定、一手移动的模式。
- 手到控制端、远端、短刷、刷头、针筒、气枪的最近距离。

主要服务动作：

- 所有动作的操作状态判断。
- 长刷插入/拔出中的双手协作。
- 短刷刷洗中的局部往复运动。
- idle 和动作段边界区分。

#### 4.3.2 `tool_detection_group`

包含：

- `short_brush_*`
- `long_brush_*`
- `brush_tip_out_*`
- `syringe_*`
- `air_gun_*`
- 各工具的 present/conf/speed/missing_age/imputed。

主要服务动作：

- `short_brush_cleaning`
- `long_brush_insert`
- `long_brush_withdraw`
- `flush`
- `air_injection`

#### 4.3.3 `object_relation_group`

包含：

- `hand_to_short_brush_*`
- `hand_to_long_brush_*`
- `brush_tip_out_to_scope_distal_end_*`
- `short_brush_to_scope_control_body_*`
- `long_brush_to_scope_mid_section_*`
- `air_gun_to_scope_distal_end_*`
- `syringe_to_scope_distal_end_*`

主要服务动作：

- 判断“工具是否在正确位置被使用”。
- 判断动作起止时工具是否靠近或远离目标区域。
- 区分工具存在但未操作和真实动作。

#### 4.3.4 `rgb_object_roi_group`

包含当前 v2/v3 已经使用的：

- `roi_rgb_v2_hand_top1_*`
- `roi_rgb_v2_hand_top2_*`
- `roi_rgb_v2_short_brush_*`
- `roi_rgb_v2_syringe_*`
- `roi_rgb_v2_air_gun_*`
- `roi_rgb_v2_scope_distal_end_*`
- `roi_rgb_v2_brush_tip_out_*`

主要服务动作：

- 工具外观补充。
- YOLO 漏检或低置信时的弱证据。
- 远端口附近的视觉变化。

#### 4.3.5 `rgb_proxy_roi_group`

v4 新增，包含：

- 洗槽整体 ROI。
- 洗槽 strip ROI。
- 通道 ROI。
- 扩大手框 ROI。
- 手周围 ring ROI。
- 手-目标 union ROI。

主要服务动作：

- 长毛刷检测缺失时的 proxy。
- 短毛刷检测缺失时的 proxy。
- 动作段边界变化。

#### 4.3.6 `business_prior_group`

包含当前已有的：

- `prior_short_clean_near`
- `prior_short_clean_motion`
- `prior_flush_stable`
- `prior_air_stable`
- `prior_long_signal_near_scope`
- `prior_long_towards_distal`
- `prior_long_away_distal`
- `prior_hand_long_contact`

v4 可新增：

- `prior_two_hand_long_brush_pattern`
- `prior_hand_ring_short_brush_pattern`
- `prior_channel_long_motion`
- `prior_control_body_local_brushing`
- `prior_distal_stable_tool_contact`

主要作用：

- 不做硬规则判定，只作为 soft evidence。
- 给模型一个动作候选倾向。
- 在数据少时减少明显不合理类别。

#### 4.3.7 `boundary_group`

v4 建议新增：

- YOLO 对象出现/消失 delta。
- RGB ROI frame difference。
- 手速度突变。
- 工具到目标距离变化突变。
- prior score 的变化。
- 通道 ROI 运动能量变化。

主要服务：

- 动作起点。
- 动作终点。
- 减少短段抖动。
- 帮助后处理合并或切分动作段。

### 4.4 推荐的融合方式

#### 4.4.1 最小改动：分组标准化 + concat

先按组分别标准化，再拼接：

```text
hand_group
tool_group
relation_group
rgb_object_group
rgb_proxy_group
prior_group
boundary_group
  -> concat
  -> BiGRU / ASFormer / MS-TCN
```

优点：

- 改动小。
- 能保留当前三模型通用架构。
- 方便做消融实验。

缺点：

- 模型仍然要自己学习各组对动作的权重。

#### 4.4.2 中等改动：group MLP + concat

每组先过一个小 MLP 压缩，再拼接：

```text
hand_group -> MLP -> hand_embed
tool_group -> MLP -> tool_embed
relation_group -> MLP -> relation_embed
rgb_proxy_group -> MLP -> proxy_embed
prior_group -> MLP -> prior_embed
boundary_group -> MLP -> boundary_embed
  -> concat
  -> temporal model
```

优点：

- 每组内部先学习局部组合。
- 输入主时序模型的维度可控。
- 可以更清楚地做特征组 ablation。

缺点：

- 需要修改模型输入模块。
- 训练样本少时 MLP 也可能过拟合。

#### 4.4.3 更推荐的 v4 中期方向：动作条件化 gate

为每个动作类别生成一个 soft score 或 gate：

```text
hand/tool/relation/rgb/prior features
  -> class-wise evidence scorer
  -> action evidence vector
  -> temporal model
```

例如：

- `evidence_long_insert`
- `evidence_long_withdraw`
- `evidence_short_clean`
- `evidence_flush`
- `evidence_air_injection`
- `evidence_idle`

这些不是最终分类结果，而是输入给时序模型的辅助证据。

好处：

- 显式体现特征和动作的关联性。
- 便于人工分析每类动作为什么被激活。
- 对小数据更友好。

风险：

- 如果先验设计过强，可能把模型限制死。
- 因此 gate 应该是 soft feature，而不是硬规则。

## 5. 问题三：短毛刷难检测时，是否用扩大手框 RGB 弥补

### 5.1 判断

有必要，而且这是短毛刷当前最值得尝试的补偿方向之一。

短毛刷在实际视频中难以识别，原因可能包括：

- 目标小。
- 被手遮挡。
- 露出部分短。
- 和背景或其它器械颜色相近。
- 动作快，单帧模糊。
- YOLO 训练数据不足，检测框不稳定。

短毛刷虽然本体难检测，但它一般和手高度绑定。因此手框附近的 RGB/纹理/运动变化可能比 `short_brush` YOLO 框更可靠。

### 5.2 不建议只简单放大手框

如果只把 hand bbox 等比例扩大并计算 RGB 均值，会有几个问题：

- 手和手套颜色会主导特征。
- 背景被引入，稀释短毛刷信号。
- 对不同视频视角和手大小敏感。
- 无法区分“手在移动但没拿短刷”和“手握短刷刷洗”。

因此，扩大手框是必要起点，但应进一步设计为多尺度和差分 ROI。

### 5.3 推荐的短毛刷 proxy ROI

#### 5.3.1 `expanded_hand_roi`

对 `hand_top1` 和 `hand_top2` 分别扩框：

- `1.3x`
- `1.6x`
- `2.0x`

提取：

- RGB/亮度/饱和度统计。
- 亮度标准差。
- edge density。
- frame difference。
- 当前帧和局部 5 帧均值。
- delta。

用途：

- 捕捉手中露出的小工具。
- 捕捉手周围刷洗动作引起的局部变化。

#### 5.3.2 `hand_ring_roi`

用扩大框减去原始 hand bbox，只看手周围一圈区域。

```text
hand_ring_roi = expanded_hand_bbox - original_hand_bbox
```

这是比单纯扩大手框更关键的设计。原因是短毛刷常常只在手边缘露出，直接看整个扩大框会被手本体淹没，而 ring ROI 可以相对减少皮肤/手套颜色干扰。

建议特征：

- `hand_ring_brightness_std`
- `hand_ring_saturation_proxy`
- `hand_ring_edge_density`
- `hand_ring_motion_energy`
- `hand_ring_texture_energy`
- `hand_ring_color_delta`

用途：

- 弥补 `short_brush_present` 长期为 0 的情况。
- 判断手周围是否有工具露出或局部刷洗运动。

#### 5.3.3 `directional_hand_roi`

如果当前帧有 `scope_control_body`、`scope_distal_end` 或其它目标位置，可以不做四周等比例扩展，而是向目标方向扩展。

例如：

```text
hand bbox -> 朝 scope_control_body 方向扩展
hand bbox -> 朝 scope_distal_end 方向扩展
```

用途：

- 短毛刷清洗通常发生在控制部或阀门附近。
- 如果手靠近控制部，朝控制部方向的手外区域更可能包含短刷或刷洗接触区域。

#### 5.3.4 `hand_target_union_roi`

把手框和目标区域框合并成一个 union ROI。例如：

- hand + `scope_control_body`
- hand + `scope_distal_end`
- hand + 人工配置的控制部局部 ROI

提取：

- union ROI 的 RGB/纹理/运动。
- hand 与目标之间区域的 edge/motion。
- union ROI 内局部 frame difference。

用途：

- 判断手是否正在对某个内镜部位进行操作。
- 对短刷清洗的动作段边界可能更有帮助。

### 5.4 和当前短毛刷特征的关系

当前已有短毛刷相关线索：

- `short_brush_present`
- `short_brush_conf`
- `short_brush_speed`
- `short_brush_missing_age`
- `short_brush_to_scope_control_body_dist`
- `prior_short_clean_near`
- `prior_short_clean_motion`
- `roi_rgb_v2_short_brush_*`

但如果 `short_brush` 基本缺位，那么这些特征会长期无效。v4 可以新增：

- `expanded_hand_roi_*`
- `hand_ring_roi_*`
- `directional_hand_roi_to_control_body_*`
- `hand_control_union_roi_*`
- `hand_near_control_motion_*`
- `hand_ring_edge_density_*`

这些特征的目标不是证明短毛刷存在，而是提供“手正在用小工具在控制部附近做局部刷洗”的 proxy evidence。

### 5.5 对短毛刷动作的预期收益

对 `short_brush_cleaning`：

- 能弥补 YOLO 检不到短刷的问题。
- 能捕捉手周围的小范围高频运动。
- 能捕捉控制部附近局部纹理变化。
- 如果结合 `hand_to_scope_control_body_dist`，可减少把普通手部移动误判为短刷刷洗。

对非短刷动作：

- 当手部移动但不靠近控制部，或者 hand ring 没有明显纹理/运动时，可降低短刷误报。

### 5.6 风险

- 手框扩大过大时，背景噪声增加。
- ring ROI 实现需要注意遮罩和边界裁剪。
- 手框跟踪不稳定时，frame difference 可能包含检测框抖动而非真实运动。
- 如果控制部位置也不稳定，directional ROI 可能引入错误方向。

## 6. v4 推荐设计

### 6.1 数据层建议

v4 如果继续使用当前切分片段数据，最多只能验证特征是否有局部增益；但它不能充分验证完整视频动作段划分能力。

更推荐的数据组织是：

```text
whole_video/
  video_001.mp4
  video_002.mp4

yolo_outputs/
  video_001.jsonl
  video_002.jsonl

annotations/
  video_001_segments.json
  video_002_segments.json

image_train/
  common/
  v4/
    configs/
    roi_definitions/
    cache/
    outputs/
    train_image_v4.py
```

每个完整视频作为一条 sequence：

```text
video_id
frames[0:T]
yolo_detections[0:T]
manual_segments
features[0:T, F]
labels[0:T]
```

这样训练和测试才更接近真实使用场景。

### 6.2 ROI 预处理建议

后续改进如果使用相同 ROI，可以对所有帧提前预处理。建议缓存内容包括：

- `video_id`
- `frame_index`
- `roi_version`
- `roi_name`
- `bbox`
- `bbox_source`
- `frame_width`
- `frame_height`
- `feature_names`
- `feature_values`
- `normalization_config`

固定 ROI 和动态 ROI 要分开管理：

- 固定 ROI：`wash_tank_roi`、`scope_channel_roi`、控制部局部 ROI。
- YOLO 动态 ROI：hand、syringe、air_gun、scope_distal_end、brush_tip_out。
- proxy 动态 ROI：expanded hand、hand ring、directional hand、hand-target union。

如果未来 ROI 定义变化，应提升 `roi_version`，避免新旧缓存混用。

### 6.3 特征层建议

v4 最小可行新增特征：

#### 长毛刷 proxy

- `wash_tank_brightness_mean`
- `wash_tank_brightness_std`
- `wash_tank_motion_energy`
- `wash_tank_strip_motion_energy`
- `scope_channel_edge_density`
- `scope_channel_dominant_orientation`
- `scope_channel_motion_energy`
- `scope_channel_line_like_ratio`

#### 短毛刷 proxy

- `hand_expanded_brightness_std`
- `hand_expanded_motion_energy`
- `hand_ring_edge_density`
- `hand_ring_motion_energy`
- `hand_ring_saturation_proxy`
- `hand_to_control_body_union_motion_energy`
- `hand_near_control_body_motion_score`

#### 动作先验

- `prior_channel_long_motion`
- `prior_two_hand_long_pattern`
- `prior_hand_ring_short_pattern`
- `prior_control_body_local_brushing`
- `prior_distal_tool_stable_contact`

#### 边界特征

- `motion_energy_delta`
- `roi_valid_delta`
- `tool_target_distance_delta_abs`
- `prior_score_delta`
- `hand_speed_peak`
- `rgb_proxy_change_peak`

### 6.4 模型层建议

#### 保持三模型主架构

为了版本管理，v4 不建议推翻当前三种模型。可以继续保留：

- MS-TCN
- ASFormer
- BiGRU

但把通用特征提取和 ROI cache 放在 `image_train/common`，v4 只新增配置和特征组合策略。

#### BiGRU 建议先做 group-aware input

优先级：

1. 分组标准化 + concat。
2. group MLP + concat。
3. action-conditioned evidence gate。

如果只做 v4 第一版，建议先做第 1 或第 2 步，不要一开始把结构改太复杂。

#### ASFormer / MS-TCN 使用方式

ASFormer 和 MS-TCN 仍然可以使用同一套最终 feature vector。区别在于：

- ASFormer 更适合建模长距离上下文，但小数据下可能过拟合。
- MS-TCN 有更强局部时序平滑和抗过分割倾向。
- BiGRU 训练快，适合先验证新特征是否有效。

### 6.5 后处理建议

继续沿用 v3 的 segment postprocess，但增加类别相关参数：

- 长刷动作允许更长段、更强平滑。
- 短刷清洗允许局部高频运动，但不应切成很多短段。
- 注气和冲洗需要工具稳定接触，短暂误报应被抑制。
- idle 到动作、动作到 idle 的边界可参考 boundary_group。

## 7. 实验优先级

建议 v4 按以下顺序做消融，不要一次加入所有东西：

### 实验 A：短毛刷 hand proxy

新增：

- expanded hand ROI
- hand ring ROI
- hand-control union ROI

目标：

- 看 `short_brush_cleaning` 的召回率、segment F1 和边界误差是否改善。
- 检查是否导致普通手部移动被误判为短刷清洗。

### 实验 B：长毛刷 wash/channel proxy

新增：

- wash tank ROI
- wash tank strip ROI
- scope channel ROI
- edge/motion/line-like features

目标：

- 看 `long_brush_insert` / `long_brush_withdraw` 的 segment F1 是否改善。
- 检查 insert/withdraw 是否仍然混淆。
- 检查 idle 是否被洗槽反光误报。

### 实验 C：group-aware BiGRU

新增：

- 按组标准化。
- 可选 group MLP。
- 输出中保留每组 embedding 或 evidence score 便于分析。

目标：

- 验证特征-动作关联性是否比 flat vector 更稳定。
- 看小数据下是否减少明显不合理类别。

### 实验 D：动作条件化 soft prior

新增：

- class-wise evidence score。
- 每类动作对应的 soft gate。

目标：

- 看段数误差和动作类别混淆是否降低。
- 避免把“手移动”误判为注气/冲洗等工具强依赖动作。

### 实验 E：完整视频级训练/测试

如果条件允许，这是最关键实验。

新增：

- 整段视频 YOLO 输出。
- 人工标注完整视频动作段。
- 以完整视频作为 sequence 训练和测试。

目标：

- 真实评估动作段数量、边界误差和 segment F1。
- 避免当前切分片段数据带来的边界泄漏和 idle 分布不真实问题。

## 8. 预期动作级影响

| 动作 | 当前主要问题 | v4 推荐补充 | 预期收益 |
|---|---|---|---|
| `long_brush_insert` | 长毛刷 YOLO 不稳定，刷头缺失时证据弱 | `scope_channel_roi`、`wash_tank_strip_roi`、方向/运动/线状特征、`prior_channel_long_motion` | 改善长刷动作召回和起点判断 |
| `long_brush_withdraw` | 和 insert 共享对象，方向难区分 | 刷头/通道 motion delta、远端附近变化、方向趋势 | 改善 insert/withdraw 混淆和终点判断 |
| `short_brush_cleaning` | 短毛刷基本缺位 | expanded hand、hand ring、hand-control union、局部高频 motion/edge | 弥补短刷缺检，提高召回 |
| `flush` | 和 air injection 都可能是工具靠近远端且相对稳定 | syringe ROI、远端口稳定接触、液体/亮度变化 | 需要谨慎，RGB 可能辅助但不能替代 syringe 检测 |
| `air_injection` | 和 flush 外观/位置相似 | air_gun ROI、喷嘴附近 ROI、与 syringe 的外观区分 | 降低 flush/air 混淆 |
| `idle` | 手存在但未操作时容易误报动作 | boundary_group、proxy motion 低值、工具/目标关系弱 | 降低短动作误报，减少过分割 |

## 9. 主要风险和控制方式

### 9.1 ROI 固定位置风险

如果不同视频角度差异大，固定洗槽 ROI 可能不准。

控制方式：

- 先人工检查每个视频固定 ROI 是否覆盖正确。
- 为每个视频保存 ROI 配置。
- 增加 `roi_valid` 和 `roi_version`。

### 9.2 RGB 受光照和反光影响

洗槽区域可能有强反光和水面波动。

控制方式：

- 不只用 RGB 均值。
- 加局部纹理、边缘、运动和 delta。
- 用局部窗口平滑降低单帧噪声。

### 9.3 特征维度膨胀

新增 ROI 很容易让维度暴涨。

控制方式：

- v4 第一版只加低维统计，不直接加高维 DINOv2。
- 每组做 PCA 或 MLP 压缩。
- 做严格消融，保留有收益的特征组。

### 9.4 先验过强

如果动作先验设计成硬规则，可能误杀真实异常情况。

控制方式：

- 所有 prior 都作为 soft feature。
- 不直接覆写模型输出。
- 后处理参数通过验证集搜索。

### 9.5 当前切分数据无法真实评估完整动作段

当前 ActionMixed 数据是切分片段，每段基本只有一种动作加前后 idle，这与真实完整视频使用场景不同。

控制方式：

- v4 特征消融可以继续用当前数据做初筛。
- 最终必须引入完整视频级人工标注做训练/测试。

## 10. 建议的 v4 文件组织

为了方便版本管理，建议如下：

```text
image_train/
  common/
    roi_cache.py
    segment_postprocess.py
    feature_groups.py
    proxy_roi.py
    roi_geometry.py
  v4/
    configs/
      image_train_v4.yaml
      roi_definitions_template.yaml
    train_image_v4.py
    evaluate_image_v4.py
    README.md
  output_v4/
    cache/
    models/
    metrics/
    predictions/
  image_train_v4.md
  image_train_v4future.md
```

其中：

- `common/roi_cache.py`：继续负责通用 ROI 缓存。
- `common/proxy_roi.py`：新增 expanded hand、hand ring、wash tank、channel ROI 计算。
- `common/feature_groups.py`：新增分组特征组织。
- `common/roi_geometry.py`：放 bbox expand、clip、union、ring mask 等几何工具。
- `v4/configs/roi_definitions_template.yaml`：保存固定 ROI 配置模板。
- `output_v4/cache`：保存按 `roi_version` 生成的 RGB/proxy 特征缓存。

这样三种模型的主实现仍可共用，v4 主要改变输入特征和融合方式。

## 11. v4 最小可行方案

如果只做一次初版 v4，不建议同时引入 DINOv2、VideoMAE 和复杂模型结构。建议先做：

1. 新增短毛刷 proxy：
   - `expanded_hand_roi`
   - `hand_ring_roi`
   - `hand_control_union_roi`
2. 新增长毛刷 proxy：
   - `wash_tank_roi`
   - `wash_tank_strip_roi`
   - `scope_channel_roi`
3. 新增低维 proxy 特征：
   - RGB 均值
   - 亮度均值/标准差
   - 饱和度 proxy
   - frame difference
   - edge density
   - motion delta
4. 对 BiGRU 做分组标准化 + concat。
5. 保持 MS-TCN、ASFormer、BiGRU 三个主模型结构基本不变。
6. 用 v3 的 segment 指标评估：
   - frame ACC
   - precision
   - recall
   - F1@0.25
   - F1@0.5
   - 预测段数/真实段数
   - count error
   - boundary error
   - 每类动作帧级识别情况

## 12. 2026-07-28 实施版 v4 记录

本次已经按“逐帧准确率优先、动作段作为后处理”的策略完成 v4 初版代码和训练。实际落地文件为：

```text
image_train/
  common/
    proxy_roi.py
  v4/
    configs/
      image_train_v4.json
    train_image_v4.py
  output_v4/
    models/
    predictions/
    image_train_v4.json
  image_train_v4.md
```

实际训练主目标：

- 训练损失：逐帧 `CrossEntropyLoss`。
- 主验证指标：`val_raw` frame accuracy。
- 早停依据：验证集 raw frame accuracy。
- 动作段平滑：训练后对 softmax probabilities 做后处理参数搜索。
- 后处理默认选择目标：验证集 post frame accuracy，同时报告 segment F1、段数误差和边界误差。

实际参数集中在 `image_train/v4/configs/image_train_v4.json`，也都可以通过命令行覆盖。当前主要参数包括：

| 参数 | 当前默认值 | 作用 |
|---|---:|---|
| `epochs` | 12 | 最大训练轮数 |
| `lr` | 0.002 | AdamW 初始学习率 |
| `min_lr` | 0.00001 | 余弦退火最低学习率 |
| `weight_decay` | 0.0001 | AdamW 权重衰减 |
| `label_smoothing` | 0.03 | 交叉熵标签平滑 |
| `scheduler` | `cosine` | 学习率调度器 |
| `early_stopping` | `true` | 是否启用早停 |
| `patience` | 4 | 早停等待轮数 |
| `early_metric` | `val_accuracy` | 早停监控指标 |
| `use_class_weights` | `true` | 是否使用类别权重 |
| `train_mode` | `full_sequence` | 默认完整序列训练 |
| `rgb_smooth_window` | 5 | object ROI RGB 局部平滑窗口 |
| `proxy_smooth_window` | 5 | proxy ROI RGB 局部平滑窗口 |
| `hand_expand` | 1.6 | 手框扩大倍率 |
| `postprocess_objective` | `accuracy` | 后处理参数选择目标 |
| `post_prob_smooth` | `1,3,5,7` | 后处理概率平滑候选 |
| `post_min_segment` | `1,3,5,8` | 后处理最短段候选 |
| `post_merge_gap` | `0,2,4,8` | 后处理短 gap 合并候选 |
| `post_confidence_threshold` | `0.0,0.25,0.4` | 低置信置 idle 候选 |

实际 v4 特征：

- `ms_tcn`：`business_priors + roi_rgb_v2 + proxy_rgb_v4`，输入维度 691。
- `asformer`：`business_priors + roi_rgb_v2 + proxy_rgb_v4`，输入维度 691。
- `bigru`：`window_stats + business_priors + roi_rgb_v2 + proxy_rgb_v4`，输入维度 819。

`proxy_rgb_v4` 新增 10 个 proxy slot：

- `wash_tank`
- `wash_tank_left`
- `wash_tank_center`
- `wash_tank_right`
- `scope_channel`
- `hand_top1_expanded`
- `hand_top1_ring`
- `hand_top2_expanded`
- `hand_top2_ring`
- `hand_control_union`

每个 proxy slot 提取 12 个 raw 特征：

- `valid`
- `conf`
- `area`
- `aspect`
- `r_mean`
- `g_mean`
- `b_mean`
- `brightness_mean`
- `brightness_std`
- `saturation_proxy`
- `edge_energy`
- `motion_energy`

然后追加：

- 5 帧 centered mean。
- 相邻帧 delta。

因此 proxy RGB 总维度为 `10 * 12 * 3 = 360`。

本轮正式训练结果记录在 `image_train/image_train_v4.md` 和 `image_train/output_v4/image_train_v4.json`。按 test raw frame accuracy，当前最好模型是 `bigru`。

## 13. 结论

对于当前任务，v4 的关键不是“再多加一些 RGB 特征”，而是把 RGB 变成有业务含义的动作证据：

- 长毛刷不能稳定 YOLO 检测时，应使用洗槽、通道、远端附近的 proxy ROI，重点提取方向、线状结构和运动变化。
- 短毛刷缺位时，应使用扩大手框、手周围 ring ROI、手-控制部 union ROI，重点提取手边缘周围的小工具和刷洗运动证据。
- BiGRU 的 flat vector 输入需要改成 group-aware 表示，至少在特征生成和标准化层面分组；更进一步可以使用 action-conditioned soft evidence。
- 所有新增 prior 都应作为软特征，不应作为硬规则。
- 最终要解决动作段划分准确性，仍然需要完整视频级数据和人工动作段标注；当前切分片段数据只能作为特征消融初筛。

因此，推荐 v4 的主线是：

```text
YOLO structured features
  + object ROI RGB
  + proxy ROI RGB/motion/edge
  + action prior
  + boundary features
  -> group-aware fusion
  -> MS-TCN / ASFormer / BiGRU
  -> segment-aware evaluation
```
