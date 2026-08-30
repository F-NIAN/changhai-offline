# image_train v5 future 更新设计草案

生成时间：2026-07-28

本文是 v5 改进设计草案，只做问题分析和方案拟定，不包含代码实现和训练结果。后续需要先人工审阅、删改、确认实验优先级，再进入 v5 代码和训练阶段。

## 1. 当前结论概览

v4 已经按“逐帧准确率优先、动作段作为后处理”的策略完成一次训练。整体看，v4 的方向是有效的：

- proxy RGB 特征加入后，`asformer` 和 `bigru` 的 test raw ACC 明显高于 v2/v3。
- 当前主目标下，`bigru` 是最好的模型。
- 低维 RGB/proxy ROI 特征对 `short_brush_cleaning`、`air_injection`、`long_brush_insert` 有帮助。

但 v4 也暴露出几个结构性问题：

- test split 中 `long_brush_withdraw` 没有 GT 帧，但三个模型都预测了该类，说明 split 分布不均衡，也说明长刷插入/拔出方向证据不稳定。
- `flush` 召回率仍偏低，模型倾向把一部分冲洗帧判成其它动作或 idle。
- `short_brush_cleaning` 召回率高，但精确率偏低，说明 hand proxy 可能把“手部移动”扩大成短刷清洗。
- `ms_tcn` 在 v4 中仍明显弱于 `asformer` 和 `bigru`，说明当前 MS-TCN 架构/训练参数对高维 proxy 特征不适配。
- 当前 v4 是一次 all-in 特征实验，缺少系统消融，无法确认 object ROI、hand proxy、wash/channel proxy 各自真实贡献。

因此 v5 不建议继续无差别加特征。v5 更应该做：

- 特征组消融。
- 类别不均衡与 split 诊断。
- 长刷方向判别增强。
- 短刷 proxy 误报抑制。
- flush/air 区分和召回补强。
- 后处理从统一参数改为类别相关参数。
- 训练/评估输出更便于做错误分析。

## 2. 参考资料对 v5 的启发

### 2.1 MS-TCN / CS-TCN 参考

本地参考文献：

- `F:/暑期实习/参考文献/CS-TCN.pdf`

该 PDF 实际内容是 MS-TCN 论文。核心启发：

- temporal action segmentation 的基础任务仍然是 frame-wise action prediction。
- 训练主体是逐帧分类损失。
- smoothing loss 用来减少过分割，而不是替代分类目标。
- frame-wise accuracy 接近时，segmental F1 和 edit score 仍可能差异很大。

对 v5 的意义：

- 当前“逐帧准确率为训练主目标，段平滑作为后处理”的策略可以继续保留。
- 但 v5 的后处理不能只按统一参数搜索，应考虑不同动作的最短段、merge gap 和低置信策略不同。
- 如果后续改 MS-TCN，可加入轻量 smoothing loss，但不应让它成为主目标。

### 2.2 第一人称手部动作参考

本地参考文献：

- `F:/暑期实习/参考文献/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf`

核心启发：

- 第一人称细粒度动作很依赖手部姿态、手部运动和手-物交互。
- 单纯 RGB 外观不够，手与对象的相对关系很关键。
- 手部被物体遮挡时，物体/手周围局部外观仍然有价值。

对 v5 的意义：

- v4 的 expanded hand 和 hand ring proxy 是合理的。
- 但短刷误报说明 hand proxy 还需要绑定目标位置，不能只看手周围图像。
- v5 应补充“手-控制部/阀门附近局部交互”的条件化特征，降低普通手部移动被误判为短刷。

### 2.3 动作识别综述参考

本地参考文献：

- `F:/暑期实习/参考文献/s11263-022-01594-9.pdf`

核心启发：

- 实际视频动作识别受背景、光照、摄像机变化、相似动作和标注不足影响。
- human-object interaction 类动作需要关注“谁、做什么、对哪个对象、在什么位置、什么时候变化”。

对 v5 的意义：

- 固定洗槽 ROI 可能受视角和反光影响，v5 应加入 ROI 质量检查和消融。
- 类别相似动作要靠更明确的对象关系、稳定接触、方向和时序模式区分。
- 数据 split 分布不均衡会严重误导评估，因此 v5 应先做 split 诊断。

## 3. v4 结果复盘

### 3.1 整体指标

| 版本 | 模型 | Test Raw ACC | Test Raw Frame-F1 | Test Segment F1@0.25 | 输入维度 |
|---|---|---:|---:|---:|---:|
| v2 | `ms_tcn` | 0.1764 | 0.0158 | 0.0000 | 459 |
| v2 | `asformer` | 0.2505 | 0.3072 | 0.1000 | 331 |
| v2 | `bigru` | 0.5278 | 0.4757 | 0.1352 | 459 |
| v3 | `ms_tcn` | 0.2780 | 0.1122 | 0.0333 | 113 |
| v3 | `asformer` | 0.4475 | 0.2970 | 0.0924 | 331 |
| v3 | `bigru` | 0.3768 | 0.3479 | 0.1114 | 459 |
| v4 | `ms_tcn` | 0.3370 | 0.1746 | 0.0571 | 691 |
| v4 | `asformer` | 0.5202 | 0.4687 | 0.1400 | 691 |
| v4 | `bigru` | 0.5964 | 0.5037 | 0.1686 | 819 |

观察：

- v4 相比 v2/v3 整体有提升，说明 proxy RGB/motion/edge 特征是有用的。
- `bigru` 依旧最稳，说明当前数据规模下，BiGRU 的序列建模和窗口统计更适配。
- `asformer` 也明显提升，但低于 `bigru`，可能需要更强正则或更小模型。
- `ms_tcn` 仍弱，说明它不是当前 v5 的优先主线，除非专门调结构/损失。

### 3.2 v4 test raw 每类动作问题

#### `bigru`

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 | 问题 |
|---|---:|---:|---:|---:|---:|---|
| `idle` | 398 | 273 | 0.832 | 0.570 | 0.677 | idle 召回不足，一部分 idle 被动作误报 |
| `long_brush_insert` | 572 | 343 | 0.985 | 0.591 | 0.739 | 精确率高但召回不足，漏掉部分插入帧 |
| `long_brush_withdraw` | 0 | 281 | 0.000 | 0.000 | 0.000 | test 无 GT 但大量误报 |
| `short_brush_cleaning` | 111 | 264 | 0.383 | 0.910 | 0.539 | 召回高但精确率低，短刷 proxy 过宽 |
| `flush` | 261 | 112 | 0.786 | 0.337 | 0.472 | 召回偏低 |
| `air_injection` | 115 | 184 | 0.625 | 1.000 | 0.769 | 召回好但预测偏多 |

#### `asformer`

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 | 问题 |
|---|---:|---:|---:|---:|---:|---|
| `idle` | 398 | 233 | 0.867 | 0.508 | 0.640 | idle 召回不足 |
| `long_brush_insert` | 572 | 264 | 0.939 | 0.434 | 0.593 | 插入召回不足 |
| `long_brush_withdraw` | 0 | 362 | 0.000 | 0.000 | 0.000 | withdraw 误报更严重 |
| `short_brush_cleaning` | 111 | 310 | 0.345 | 0.964 | 0.508 | 短刷误报明显 |
| `flush` | 261 | 105 | 0.819 | 0.330 | 0.470 | 召回偏低 |
| `air_injection` | 115 | 183 | 0.628 | 1.000 | 0.772 | air 召回好但预测偏多 |

#### `ms_tcn`

| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 | 问题 |
|---|---:|---:|---:|---:|---:|---|
| `idle` | 398 | 281 | 0.416 | 0.294 | 0.345 | idle 很差 |
| `long_brush_insert` | 572 | 377 | 0.687 | 0.453 | 0.546 | 插入一般 |
| `long_brush_withdraw` | 0 | 153 | 0.000 | 0.000 | 0.000 | withdraw 误报 |
| `short_brush_cleaning` | 111 | 634 | 0.175 | 1.000 | 0.298 | 严重短刷误报 |
| `flush` | 261 | 12 | 0.333 | 0.015 | 0.029 | 几乎识别不出 flush |
| `air_injection` | 115 | 0 | 0.000 | 0.000 | 0.000 | 未识别 air |

## 4. 当前主要问题分析

### 4.1 test split 类别分布不均衡

最明显的问题是：test split 中 `long_brush_withdraw` 的 GT 帧数为 0，但模型仍然预测了该类别。

这会造成两个后果：

- `long_brush_withdraw` 的 test precision/recall/F1 无法正常反映真实能力。
- 模型是否真的学会 withdraw，不能通过当前 test split 判断。

这不是单纯模型问题，也是评估设计问题。v5 如果继续用当前 split，必须在报告里明确标出每类 GT 帧数和 GT 段数；否则某些类别的指标会误导判断。

建议：

- v5 先做 split 诊断，不急着增加复杂模型。
- 对每个 split 输出每类帧数、段数、视频数。
- 如果某类在 test 中为 0，应在报告里标记为 `not evaluable`，不要把 F1=0 简单理解为模型不会识别。
- 若允许，可重新做 stratified split，让每个动作在 train/val/test 都至少有若干段。

### 4.2 长刷 insert/withdraw 方向证据不足

`long_brush_insert` 和 `long_brush_withdraw` 共享很多外观和对象关系：

- 手。
- 长刷/刷头。
- 远端口。
- 洗槽/通道 ROI。
- 手部移动。

区别主要在方向和阶段：

- insert 更像刷头/手/通道信号朝远端或通道内部推进。
- withdraw 更像刷头/手/通道信号离开远端或通道内部退出。

v4 的 proxy ROI 主要提取均值、边缘、运动能量和 delta，但没有显式表示“沿通道方向的投影位移”和“多帧轨迹斜率”。因此模型容易把长刷相关活动混成 insert 或 withdraw。

建议：

- 新增 `channel_axis_motion`：沿 `scope_channel_roi` 主轴方向的运动投影。
- 新增 `brush_tip_distal_slope_w15/w31`：刷头到远端口距离在 15/31 帧窗口内的线性斜率。
- 新增 `hand_distal_slope_w15/w31`：手到远端口/通道入口的距离斜率。
- 新增 `long_direction_consistency`：窗口内方向变化是否稳定，而不是单帧 delta。
- 对 withdraw 增加可评估数据，否则不要把 test 指标作为有效结论。

### 4.3 短刷 proxy 召回高但精确率低

v4 中 `short_brush_cleaning` 的 recall 高，但 precision 低：

- `bigru`：precision 0.383，recall 0.910。
- `asformer`：precision 0.345，recall 0.964。
- `ms_tcn`：precision 0.175，recall 1.000。

这说明 expanded/ring hand ROI 捕捉到了“手部局部活动”，但不够确认“短刷清洗”。也就是说，hand proxy 过于宽泛。

短刷动作应该至少满足：

- 手在控制部/阀门/短刷工作区域附近。
- 手周围 ring ROI 有局部纹理或运动。
- 运动是局部高频往复，而不是大范围移动。
- 不是 syringe/air_gun 明确出现并靠近远端。

建议：

- hand proxy 必须与 `scope_control_body` 或人工控制部 ROI 绑定。
- 新增 `hand_control_dist`、`hand_control_union_motion`、`hand_ring_motion_when_near_control`。
- 加入短刷候选抑制特征：当 syringe/air_gun 高置信靠近远端时，降低 short_brush_cleaning 证据。
- 对短刷 proxy 做 ablation，比较 expanded hand、ring hand、hand-control union 哪个贡献最大。

### 4.4 flush 召回偏低

v4 中 `flush` precision 尚可，但 recall 偏低：

- `bigru`：precision 0.786，recall 0.337。
- `asformer`：precision 0.819，recall 0.330。
- `ms_tcn`：precision 0.333，recall 0.015。

这说明模型只在很确定时预测 flush，漏掉大量真实 flush 帧。

可能原因：

- syringe 检测不稳定或被手遮挡。
- flush 和 air injection 都是工具靠近远端并相对稳定，外观/位置相似。
- 当前 RGB 特征没有表达“推注过程”或液体/远端口变化。
- 类别权重/后处理参数没有针对 flush 做召回补偿。

建议：

- 新增 `syringe_hand_union_roi`，捕捉手握针筒区域。
- 新增 `syringe_distal_contact_duration`，表达针筒靠近远端持续时间。
- 新增 `distal_brightness_delta_when_syringe_near`，表达远端口附近视觉变化。
- 对 flush 使用类别相关阈值或 logit bias，适度提高召回。
- 后处理给 flush 使用较小 confidence threshold 或较长 contact duration 条件。

### 4.5 air_injection 召回高但预测偏多

v4 中 `air_injection` recall 很高，但 predicted frames 多于 GT：

- `bigru`：GT 115，Pred 184。
- `asformer`：GT 115，Pred 183。

这说明模型能够抓住 air 线索，但边界或结束点可能偏宽，也可能把部分 flush/idle 判成 air。

建议：

- 对 air 加 `air_gun_distal_contact_duration` 和 `air_gun_stability`。
- 区分 syringe/air_gun 外观：保留两者 ROI RGB 差异特征，并增加 `tool_type_competition`。
- 后处理对 air 使用较高 confidence threshold 或更严格的工具存在条件，减少过宽预测。

### 4.6 idle 召回不足

`bigru` 的 idle recall 为 0.570，说明大量 idle 被动作类别吞掉。

可能原因：

- 当前训练目标和类别权重可能偏向动作类，降低 idle 权重。
- 切分片段中 idle 多出现在动作前后，模型容易把边界附近 idle 吸收到动作段里。
- hand/proxy motion 特征只要有手部活动就容易触发动作。

建议：

- v5 增加 idle hard negative 分析：手存在但没有动作、工具存在但未接触、洗槽反光但无操作。
- 后处理增加 idle recovery：低置信动作帧、无目标接触帧、proxy motion 低帧回退 idle。
- 训练中尝试不同 class weight 策略，避免动作类权重过强。

### 4.7 v4 缺少系统消融

v4 是一次整合实验，新增了：

- object ROI RGB。
- wash/channel proxy。
- expanded/ring hand proxy。
- hand-control union。
- edge/motion 特征。

但目前无法判断：

- 是 hand proxy 提升最大，还是 wash/channel proxy 提升最大。
- 哪些 proxy 带来了误报。
- object ROI 和 proxy ROI 是否重复。
- MS-TCN 变差是否因为维度太高、归一化不适配，还是结构本身不适配。

v5 必须把消融作为主线，否则继续迭代会不可控。

## 5. v5 总体目标

v5 暂定目标：

1. 保持训练主目标为逐帧准确率。
2. 系统做特征组消融，找出真正有效的 RGB/proxy 特征。
3. 针对当前主要错误做定向改进：
   - 减少 `short_brush_cleaning` 误报。
   - 提升 `flush` 召回。
   - 降低 `long_brush_withdraw` 无 GT split 上的误报，并在报告中标记不可评估类别。
   - 提升 idle 召回。
4. 后处理从全类别统一参数改为类别相关参数。
5. 报告输出更详细的错误分析和消融结果，供后续人工判断。

v5 不建议做的事：

- 不建议直接上 DINOv2/VideoMAE 高维特征。
- 不建议在当前切分数据上大改复杂模型。
- 不建议只看 ACC 选方案；ACC 是训练主目标，但类别召回、误报和 split 可评估性必须同时看。
- 不建议继续一次性加入大量新特征。

## 6. v5 推荐实验主线

### 6.1 主线 A：数据与 split 诊断

先新增一个 v5 诊断输出，不训练也可以跑：

- 每个 split 的每类帧数。
- 每个 split 的每类动作段数。
- 每个 split 的每类视频数。
- 每个类别是否在 val/test 中可评估。
- 每个类别的平均段长、最短段长、最长段长。

报告中增加：

```text
evaluable_class = gt_frames > 0 and gt_segments > 0
```

如果不可评估：

- frame F1 可以列出，但必须标记 `not evaluable`。
- segment F1 不作为主结论。
- 误报帧数仍然保留，因为它反映 false positive 风险。

可选改进：

- 如果数据允许，新增 `--split-strategy stratified_by_action`。
- 保证每个动作在 train/val/test 都至少有 1 个视频或若干动作段。
- 当前 declared split 仍保留，方便和 v1-v4 对比。

### 6.2 主线 B：特征组消融矩阵

v5 应把特征组开关参数化：

| 开关 | 含义 | 默认建议 |
|---|---|---|
| `--use-object-rgb` | 使用 v2 object ROI RGB | 开 |
| `--use-wash-proxy` | 使用 wash tank / strip ROI | 开 |
| `--use-channel-proxy` | 使用 scope channel ROI | 开 |
| `--use-hand-expanded-proxy` | 使用 expanded hand ROI | 先开，消融验证 |
| `--use-hand-ring-proxy` | 使用 hand ring ROI | 先开，消融验证 |
| `--use-hand-control-union` | 使用 hand-control union ROI | 开 |
| `--use-proxy-edge` | 使用 edge energy | 开 |
| `--use-proxy-motion` | 使用 motion energy | 开 |
| `--use-window-stats` | 使用非 RGB 窗口统计 | BiGRU 开，其它模型可选 |
| `--use-business-priors` | 使用业务先验 | 开 |

推荐消融顺序：

| 实验ID | 特征组合 | 目的 |
|---|---|---|
| `v5_a0_bbox_priors` | bbox v2 + business priors | 低维基线 |
| `v5_a1_object_rgb` | a0 + object ROI RGB | 验证 YOLO 对象 ROI 的贡献 |
| `v5_a2_hand_proxy` | a1 + expanded/ring hand | 验证短刷 proxy 收益和误报 |
| `v5_a3_wash_channel` | a1 + wash/channel proxy | 验证长刷 proxy 收益 |
| `v5_a4_all_proxy` | a1 + hand + wash/channel | 对照 v4 |
| `v5_a5_no_edge` | a4 - edge energy | 判断 edge 是否有用 |
| `v5_a6_no_motion` | a4 - motion energy | 判断 motion 是否有用 |
| `v5_a7_no_window_stats` | bigru 去掉窗口统计 | 判断窗口统计贡献 |

主模型建议：

- 第一轮只跑 `bigru`，因为它是 v4 最好且训练相对稳定。
- 消融确认后再跑 `asformer`。
- `ms_tcn` 暂不作为 v5 主线，除非单独做结构/损失调参。

### 6.3 主线 C：长刷方向特征

为 `long_brush_insert` / `long_brush_withdraw` 新增方向类特征。

#### 6.3.1 距离斜率

对以下距离特征做窗口线性拟合斜率：

- `brush_tip_out_to_scope_distal_end_dist`
- `hand_to_long_brush_dist`
- `long_brush_to_scope_mid_section_dist`
- `hand_to_scope_distal_end_dist`，如果当前没有需新增
- `hand_to_scope_channel_entry_dist`，如果定义 channel entry

新增特征：

```text
{distance_feature}_slope_w15
{distance_feature}_slope_w31
{distance_feature}_slope_sign_stability_w31
```

用途：

- 斜率为负，更偏靠近远端或插入。
- 斜率为正，更偏远离远端或拔出。
- sign stability 表示方向是否持续一致，减少单帧抖动。

#### 6.3.2 通道方向运动

在 `scope_channel_roi` 中增加方向运动：

- `channel_motion_x`
- `channel_motion_y`
- `channel_motion_axis_projection`
- `channel_motion_axis_abs`
- `channel_motion_axis_sign_stability`

如果暂时没有精确通道轴，可先用人工配置：

```json
{
  "scope_channel_axis": {
    "start": [0.10, 0.50],
    "end": [0.90, 0.50]
  }
}
```

后续再按视频单独配置。

#### 6.3.3 长刷候选证据

新增 soft evidence：

- `evidence_long_insert_direction`
- `evidence_long_withdraw_direction`
- `evidence_long_channel_motion`
- `evidence_long_tip_near_distal`

这些只作为输入特征，不直接覆盖模型输出。

### 6.4 主线 D：短刷误报抑制

短刷 v5 的重点不是继续提高 recall，而是提高 precision。

#### 6.4.1 手-控制部条件化

新增：

- `hand_to_scope_control_body_dist`
- `hand_control_union_motion_energy`
- `hand_ring_motion_when_near_control`
- `hand_ring_edge_when_near_control`
- `short_clean_control_contact_score`

其中：

```text
short_clean_control_contact_score =
  hand_present
  * near(hand_to_scope_control_body_dist)
  * hand_ring_motion_energy
  * hand_control_union_edge_energy
```

#### 6.4.2 工具竞争抑制

短刷清洗通常不应与 syringe/air_gun 强接触远端同时成立。

新增：

- `not_flush_air_context`
- `short_vs_flush_competition`
- `short_vs_air_competition`

例：

```text
not_flush_air_context =
  1 - max(prior_flush_stable, prior_air_stable)
```

短刷 evidence 可以乘以该 soft factor，但仍作为特征，不硬规则判定。

### 6.5 主线 E：flush 召回补强

新增 flush 相关特征：

- `syringe_hand_union_roi_*`
- `syringe_to_hand_dist`
- `syringe_distal_contact_duration_w15`
- `syringe_distal_contact_duration_w31`
- `distal_rgb_delta_when_syringe_near`
- `distal_motion_when_syringe_near`
- `evidence_flush_contact`

flush 的后处理建议：

- 使用类别相关 confidence threshold，flush 阈值可低于 air。
- 对靠近远端且稳定持续的 flush 段，允许短暂低置信帧补齐。
- 但如果 air_gun 高置信，flush 需要被竞争抑制。

### 6.6 主线 F：类别相关后处理

v4 后处理参数是全类别共享：

- `prob_smooth`
- `min_segment`
- `merge_gap`
- `confidence_threshold`

v5 建议改为类别相关：

```json
{
  "default": {
    "prob_smooth": 3,
    "min_segment": 5,
    "merge_gap": 2,
    "confidence_threshold": 0.25
  },
  "short_brush_cleaning": {
    "min_segment": 8,
    "confidence_threshold": 0.35
  },
  "flush": {
    "min_segment": 5,
    "confidence_threshold": 0.15,
    "merge_gap": 4
  },
  "air_injection": {
    "min_segment": 5,
    "confidence_threshold": 0.35
  },
  "long_brush_insert": {
    "min_segment": 8,
    "merge_gap": 4
  },
  "long_brush_withdraw": {
    "min_segment": 8,
    "merge_gap": 4
  }
}
```

目的：

- 短刷误报多，所以提高阈值和最短段。
- flush 召回低，所以降低阈值并允许 gap 合并。
- air 预测偏多，所以阈值略高。
- 长刷动作持续性更强，所以最短段更长、允许中间短 gap。

后处理参数搜索要避免组合爆炸：

- 第一版只允许每类覆盖 `min_segment` 和 `confidence_threshold`。
- `prob_smooth` 先全局共享。
- 每次只搜索少量候选。

## 7. v5 推荐模型策略

### 7.1 主模型

第一阶段：

- 只跑 `bigru`。
- 原因：v4 最好，训练稳定，便于特征消融。

第二阶段：

- 对最优特征组合再跑 `asformer`。
- 原因：ASFormer 有长上下文潜力，但需要避免过拟合。

第三阶段：

- 是否继续 `ms_tcn` 取决于前两阶段结果。
- 如果继续，建议单独调小模型或加入 smoothing loss。

### 7.2 训练目标

继续以逐帧分类为主：

- `CrossEntropyLoss`
- class weights 可配置
- label smoothing 可配置
- early stopping by val raw ACC

新增可选项：

- `--loss focal`：对难样本强化，尝试改善 flush 召回。
- `--class-weight-mode balanced|sqrt|none|manual`
- `--manual-class-weights '{"flush":1.5,"idle":1.2,"short_brush_cleaning":0.9}'`

不建议默认使用复杂联合损失。先保持训练目标简单，避免无法解释。

### 7.3 校准策略

v5 可以加入 logits/probability 校准：

- temperature scaling。
- per-class threshold。
- validation 上搜索阈值。

目标：

- 减少短刷、air、withdraw 误报。
- 提升 flush 召回。
- 让后处理不只依赖 argmax。

## 8. v5 参数化建议

建议新增配置文件：

```text
image_train/v5/configs/image_train_v5.json
```

建议参数：

```json
{
  "objective": "frame_accuracy",
  "models": ["bigru"],
  "ablation_id": "v5_a4_all_proxy",
  "epochs": 12,
  "lr": 0.002,
  "min_lr": 0.00001,
  "weight_decay": 0.0001,
  "label_smoothing": 0.03,
  "scheduler": "cosine",
  "early_stopping": true,
  "patience": 4,
  "early_metric": "val_accuracy",
  "class_weight_mode": "balanced",
  "loss": "cross_entropy",
  "use_object_rgb": true,
  "use_wash_proxy": true,
  "use_channel_proxy": true,
  "use_hand_expanded_proxy": true,
  "use_hand_ring_proxy": true,
  "use_hand_control_union": true,
  "use_proxy_edge": true,
  "use_proxy_motion": true,
  "use_window_stats": true,
  "use_business_priors": true,
  "use_direction_features": true,
  "use_classwise_postprocess": true,
  "postprocess_objective": "accuracy_with_fp_penalty",
  "report_evaluable_classes": true
}
```

命令行示例：

```powershell
python .\image_train\v5\train_image_v5.py `
  --models bigru `
  --ablation-id v5_a2_hand_proxy `
  --use-object-rgb `
  --use-hand-expanded-proxy `
  --use-hand-ring-proxy `
  --no-use-wash-proxy `
  --no-use-channel-proxy
```

## 9. v5 文件组织建议

建议：

```text
image_train/
  common/
    roi_cache.py
    proxy_roi.py
    segment_postprocess.py
    feature_ablation.py          # v5 新增：按开关组合特征组
    direction_features.py        # v5 新增：长刷方向/斜率特征
    classwise_postprocess.py     # v5 新增：类别相关后处理
    split_diagnostics.py         # v5 新增：split 可评估性诊断
  v5/
    configs/
      image_train_v5.json
      ablations.json
      classwise_postprocess.json
    train_image_v5.py
    diagnose_image_v5.py
  output_v5/
    diagnostics/
    ablations/
    cache/
    models/
    predictions/
    image_train_v5.json
  image_train_v5future.md
  image_train_v5.md
```

说明：

- `train_image_v5.py` 不应复制大量 v4 代码，优先复用 common。
- v5 的主要变化应该是“特征组合可配置”和“报告更强”，不是重写三种模型。
- 消融结果可以放在 `output_v5/ablations/{ablation_id}/` 下，避免所有实验互相覆盖。

## 10. v5 评估报告应新增的内容

v5 报告除了 v4 已有表格，还应新增：

### 10.1 split 可评估性

| split | 动作 | GT帧数 | GT段数 | 视频数 | 是否可评估 |
|---|---|---:|---:|---:|---|

### 10.2 false positive 表

对每类动作列出：

- GT 帧数。
- Pred 帧数。
- FP 帧数。
- FP 主要来自哪些 GT 类别。

这比只看 precision 更有用。例如 `long_brush_withdraw` 在 test 中 GT=0，但 pred 很多，应该明确列为 false positive 风险。

### 10.3 消融对照表

| 实验ID | 模型 | 特征组 | Test ACC | Test Macro-F1 | short P/R | flush P/R | air P/R | withdraw FP |
|---|---|---|---:|---:|---:|---:|---:|---:|

### 10.4 后处理前后对照

对每个模型/消融输出：

- raw ACC。
- post ACC。
- raw 每类 P/R/F1。
- post 每类 P/R/F1。
- post 段数误差。

如果 post 提高段级指标但明显降低 ACC，要明确写出来。

## 11. v5 优先级建议

### 必做

1. split 诊断和可评估类别标记。
2. v4 特征组消融。
3. 类别相关后处理。
4. 短刷误报抑制。
5. flush 召回补强。

### 建议做

1. 长刷方向斜率特征。
2. per-class threshold / calibration。
3. false positive 来源分析。

### 暂缓

1. DINOv2/VideoMAE 高维特征。
2. 大幅重写模型结构。
3. 完整视频级训练代码。

暂缓原因不是这些方向没价值，而是当前 v4 结果显示还有基础问题没有拆清楚。先做可解释消融和错误分析更稳。

## 12. v5 预期结果判断标准

v5 是否成功，不只看总 ACC。

建议成功标准：

- `bigru` test raw ACC 不低于 v4 的 0.5964。
- test target macro frame F1 高于 v4 的 0.5037。
- `short_brush_cleaning` precision 明显提高，同时 recall 不大幅下降。
- `flush` recall 明显提高。
- `air_injection` predicted frames 更接近 GT。
- `long_brush_withdraw` 在无 GT test split 上 false positive 明显减少，或报告明确标记不可评估。
- postprocess 后 ACC 不明显低于 raw，且段数误差不恶化。

## 13. 建议的 v5 第一版实验

如果只做 v5 第一版，建议不要一次做太多，按以下顺序：

1. `diagnose_only`：只输出 split 分布和可评估性，不训练。
2. `v5_a0_bbox_priors_bigru`：bbox + priors 基线。
3. `v5_a1_object_rgb_bigru`：加 object ROI RGB。
4. `v5_a2_hand_proxy_bigru`：加 hand proxy，观察 short precision/recall。
5. `v5_a3_wash_channel_bigru`：加 wash/channel proxy，观察 long insert/withdraw。
6. `v5_a4_all_proxy_bigru`：接近 v4，但使用类别相关后处理。
7. 对最佳特征组合跑 `asformer`。

这样可以回答几个关键问题：

- v4 的提升到底来自哪组特征。
- 短刷误报是否由 hand proxy 引起。
- 长刷误报是否由 wash/channel proxy 引起。
- 类别相关后处理能否提高 flush recall、降低 short/air/withdraw 误报。

## 14. 当前建议结论

v5 的主线不应该是“继续加更多视觉特征”，而应该是“把 v4 已经有效但混杂的特征拆清楚，并针对每类动作错误做可解释修正”。

最推荐的 v5 方向是：

```text
split diagnostics
  -> feature group ablation
  -> direction/contact evidence features
  -> classwise postprocess
  -> detailed FP/FN report
```

在你确认这个设计前，不建议开始写 v5 训练代码或跑训练。
