# 当前模型特征清单

本文对应当前 `image_train/output_v3/models` 中的三种模型 checkpoint。当前所谓“模型所用特征”指输入到时序模型的逐帧向量 `features[T, F]`，不是模型内部隐藏层。

## 1. 三个模型实际使用的特征组

| 模型 | 总维度 | base v2 检测结构特征 | 非 RGB 窗口统计 | 业务先验 | ROI RGB v2 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `ms_tcn` | 113 | 113 | 0 | 0 | 0 | v3 中回退到低维 bbox-only，用于避免 v2 高维增强后退化 |
| `asformer` | 331 | 113 | 0 | 8 | 210 | `business_priors + roi_rgb_v2 + segment_postprocess` |
| `bigru` | 459 | 113 | 128 | 8 | 210 | `window_stats + business_priors + roi_rgb_v2 + segment_postprocess` |

所有模型的后处理参数不属于训练输入特征，而是推理后的动作段平滑、短段删除、gap 合并等规则。

## 2. 原始检测对象与槽位

YOLO 检测类别先映射到业务对象，再被组织为固定槽位。

| 对象/槽位 | 来源 | 选择规则 | 主要可能相关动作 |
|---|---|---|---|
| `hand_top1` | YOLO `hand` | 当前帧 hand 候选按 `confidence * sqrt(area)` 排序，取第 1 个 | 所有动作，尤其判断是否有人正在操作 |
| `hand_top2` | YOLO `hand` | 同上，取第 2 个，避免两只手被平均成一个点 | 所有动作，尤其遮挡、双手协作 |
| `short_brush` | YOLO `short_brush` | 非 hand 对象取 top-1 | `short_brush_cleaning` |
| `long_brush` | YOLO/历史对象槽位 | 非 hand 对象取 top-1；ActionMixed 当前检测类里未必稳定出现 | `long_brush_insert`, `long_brush_withdraw` |
| `syringe` | YOLO `syringe` | 非 hand 对象取 top-1 | `flush` |
| `air_gun` | YOLO `air_gun` | 非 hand 对象取 top-1 | `air_injection` |
| `scope_control_body` | YOLO `scope_control_body` | 非 hand 对象取 top-1 | 短刷清洗控制部/阀门附近 |
| `scope_mid_section` | YOLO `scope_mid_section` | 非 hand 对象取 top-1 | 长刷与镜身中段关系 |
| `scope_distal_end` | YOLO `scope_distal_end` | 非 hand 对象取 top-1 | `flush`, `air_injection`, 长刷刷头出入 |
| `brush_tip_out` | YOLO `brush_tip_out` | 非 hand 对象取 top-1 | `long_brush_insert`, `long_brush_withdraw` |

## 3. base v2 检测结构特征

base v2 共 113 维，所有三个 v3 模型都至少使用这一组。

### 3.1 hand 数量特征

| 特征名 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `hand_count` | 当前帧所有 hand 候选数量，裁剪到最大 3 后除以 3 | 0-1 归一化手数量 | 判断是否处于人工操作状态；对所有动作都有背景意义 |

### 3.2 hand top-1/top-2 单槽位特征

下面每个后缀都会分别生成在 `hand_top1_*` 和 `hand_top2_*` 上。

| 特征名 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `hand_top1_present` / `hand_top2_present` | 该 hand 槽位当前帧是否有真实 YOLO 检测 | 1 表示真实检测到，0 表示未检测到 | 区分真实操作与缺失；所有动作的操作前提 |
| `hand_top1_conf` / `hand_top2_conf` | YOLO 置信度；若缺失但被短 gap 补全，则使用衰减置信度 | 检测可靠性 | 给模型判断当前 hand 特征是否可信 |
| `hand_top1_cx` / `hand_top2_cx` | YOLO bbox 中心点 x，归一化到 0-1 | 手在画面横向位置 | 判断手是否靠近刷子、针筒、气枪、内镜部位 |
| `hand_top1_cy` / `hand_top2_cy` | YOLO bbox 中心点 y，归一化到 0-1 | 手在画面纵向位置 | 同上 |
| `hand_top1_area` / `hand_top2_area` | YOLO bbox 宽高相乘，归一化面积 | 手可见面积 | 遮挡、靠近镜头、操作强度的弱信号 |
| `hand_top1_speed` / `hand_top2_speed` | 当前中心点与上一帧中心点欧氏距离乘 fps，再裁剪归一化 | 手部运动速度 | 区分静止持物、刷洗、插拔等动态差异 |
| `hand_top1_missing_age` / `hand_top2_missing_age` | 连续未真实检测到 hand 的帧数，按最大 gap 归一化 | 缺失持续时间 | 判断短遮挡还是长期不存在 |
| `hand_top1_imputed` / `hand_top2_imputed` | 短 gap 线性插值或尾部短填充产生的坐标标记 | 1 表示该帧坐标是补全值而非真实检测 | 防止模型把补全坐标当成真实检测 |

### 3.3 非 hand 对象单槽位特征

以下对象都使用同一套 9 个特征：

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

每个对象都会生成：

| 特征名模板 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `{object}_candidate_count` | 当前帧该对象所有候选数，裁剪到最大 3 后除以 3 | 同类候选数量 | 判断检测是否混乱；候选多时可能有误检/遮挡 |
| `{object}_present` | top-1 槽位当前帧是否有真实 YOLO 检测 | 1/0 可见性 | 对应动作是否可能发生的基础证据 |
| `{object}_conf` | top-1 YOLO 置信度；补全帧使用衰减值 | 检测可靠性 | 质量感知；低置信时更多依赖上下文 |
| `{object}_cx` | top-1 bbox 中心 x | 横向位置 | 与 hand、内镜部位的关系 |
| `{object}_cy` | top-1 bbox 中心 y | 纵向位置 | 同上 |
| `{object}_area` | top-1 bbox 面积 | 可见大小 | 小目标是否出现、是否被遮挡 |
| `{object}_speed` | 中心点相邻帧移动速度 | 运动强度 | 刷洗、插入、拔出、接近远端口等动态线索 |
| `{object}_missing_age` | 连续缺失帧数归一化 | 缺失持续时间 | 区分短遮挡和真正不在场 |
| `{object}_imputed` | 当前帧是否来自短 gap 插值/填充 | 补全标记 | 告诉模型该对象状态不完全可信 |

对象与动作的直观关系：

| 对象 | 主要动作意义 |
|---|---|
| `short_brush_*` | `short_brush_cleaning` 的核心对象；位置、速度、靠近控制部关系很重要 |
| `long_brush_*` | 长刷插入/拔出；但当前检测稳定性可能弱 |
| `syringe_*` | `flush` 核心对象；靠近远端口且有手操作时更有意义 |
| `air_gun_*` | `air_injection` 核心对象；与 `syringe` 外观和位置容易混淆 |
| `scope_control_body_*` | 短刷清洗控制部附近的锚点 |
| `scope_mid_section_*` | 长刷与镜身中段关系的锚点 |
| `scope_distal_end_*` | flush、air injection、长刷刷头出入的关键锚点 |
| `brush_tip_out_*` | 长刷插入/拔出最关键的弱检测信号 |

### 3.4 对象关系特征

以下每个对象对都会生成 3 个特征：`valid`、`dist`、`delta`。

| 对象对 | 特征名 | 怎么获得 | 值含义 | 主要用于 |
|---|---|---|---|---|
| hand 到 short_brush | `hand_to_short_brush_valid` | hand 任一槽位和 short_brush 都 active | 关系是否可用 | 判断手是否拿/操作短刷 |
| hand 到 short_brush | `hand_to_short_brush_dist` | 两只手到 short_brush 中心距离取最小，除以 `sqrt(2)` | 距离越小越接近 | `short_brush_cleaning` |
| hand 到 short_brush | `hand_to_short_brush_delta` | 当前距离减上一帧距离 | 正负表示远离/接近趋势 | 短刷被拿起、靠近控制部前后 |
| hand 到 long_brush | `hand_to_long_brush_valid` | hand 与 long_brush active | 关系是否可用 | 长刷操作 |
| hand 到 long_brush | `hand_to_long_brush_dist` | hand 到 long_brush 最近距离 | 手-长刷接触程度 | `long_brush_insert/withdraw` |
| hand 到 long_brush | `hand_to_long_brush_delta` | 距离变化 | 接近/远离趋势 | 长刷操作动态 |
| brush_tip_out 到 scope_distal_end | `brush_tip_out_to_scope_distal_end_valid` | 刷头和远端口 active | 关系是否可用 | 长刷插拔 |
| brush_tip_out 到 scope_distal_end | `brush_tip_out_to_scope_distal_end_dist` | 刷头到远端口距离 | 刷头是否靠近远端口 | `long_brush_insert/withdraw` |
| brush_tip_out 到 scope_distal_end | `brush_tip_out_to_scope_distal_end_delta` | 距离变化 | 方向趋势；变小/变大可能对应插入/拔出阶段 | 长刷边界和方向 |
| short_brush 到 scope_control_body | `short_brush_to_scope_control_body_valid` | 短刷和控制部 active | 关系是否可用 | 短刷清洗 |
| short_brush 到 scope_control_body | `short_brush_to_scope_control_body_dist` | 短刷到控制部距离 | 是否在阀门/控制部附近 | `short_brush_cleaning` |
| short_brush 到 scope_control_body | `short_brush_to_scope_control_body_delta` | 距离变化 | 靠近/离开控制部 | 清洗动作起止 |
| long_brush 到 scope_mid_section | `long_brush_to_scope_mid_section_valid` | 长刷和镜身中段 active | 关系是否可用 | 长刷动作 |
| long_brush 到 scope_mid_section | `long_brush_to_scope_mid_section_dist` | 长刷到镜身中段距离 | 长刷是否沿镜身操作 | `long_brush_insert/withdraw` |
| long_brush 到 scope_mid_section | `long_brush_to_scope_mid_section_delta` | 距离变化 | 操作趋势 | 长刷动作 |
| air_gun 到 scope_distal_end | `air_gun_to_scope_distal_end_valid` | 气枪和远端口 active | 关系是否可用 | 注气 |
| air_gun 到 scope_distal_end | `air_gun_to_scope_distal_end_dist` | 气枪到远端口距离 | 是否靠近远端口 | `air_injection` |
| air_gun 到 scope_distal_end | `air_gun_to_scope_distal_end_delta` | 距离变化 | 接近/离开趋势 | 注气起止 |
| syringe 到 scope_distal_end | `syringe_to_scope_distal_end_valid` | 针筒和远端口 active | 关系是否可用 | 冲洗 |
| syringe 到 scope_distal_end | `syringe_to_scope_distal_end_dist` | 针筒到远端口距离 | 是否靠近远端口 | `flush` |
| syringe 到 scope_distal_end | `syringe_to_scope_distal_end_delta` | 距离变化 | 接近/离开趋势 | 冲洗起止 |

### 3.5 时间位置特征

| 特征名 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `t_norm` | 当前帧索引在线性序列中的相对位置，0 到 1 | 视频片段内部位置 | 学习动作在片段中常见位置；当前短片段数据里可能有切片偏置 |
| `t_sin` | `sin(2π * t_norm)` | 周期位置编码 | 给模型时间顺序信息 |
| `t_cos` | `cos(2π * t_norm)` | 周期位置编码 | 同上 |

注意：因为当前数据是动作短片段，时间位置特征可能学习到“动作通常在片段中间”的偏置。完整视频训练时要重新评估是否保留。

## 4. 非 RGB 窗口统计特征

只有当前 v3 的 `bigru` 使用这一组，共 128 维。

窗口统计来自 `add_centered_window_stats`，对关键列做中心窗口均值。窗口大小有两个：

```text
w5: 当前帧前后各约 2 帧
w15: 当前帧前后各约 7 帧
```

### 4.1 会被做窗口均值的特征

只要原特征名以后缀结尾，就会生成窗口统计：

```text
_present
_conf
_speed
_dist
_delta
_missing_age
_imputed
```

生成命名规则：

```text
{base_feature}_center_mean_w5
{base_feature}_center_mean_w15
```

### 4.2 窗口统计具体含义

| 特征名模板 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `{object}_present_center_mean_w5/w15` | 对 `{object}_present` 做中心窗口平均 | 该对象在局部时间范围内出现比例 | 减少单帧漏检影响；判断动作段持续性 |
| `{object}_conf_center_mean_w5/w15` | 对 `{object}_conf` 做中心窗口平均 | 局部检测可信度 | 判断该对象在一段时间内是否稳定存在 |
| `{object}_speed_center_mean_w5/w15` | 对 `{object}_speed` 做中心窗口平均 | 局部运动强度 | 区分静止拿着 vs 正在刷洗/插拔 |
| `{object}_missing_age_center_mean_w5/w15` | 对 `{object}_missing_age` 做中心窗口平均 | 局部缺失程度 | 判断遮挡持续性 |
| `{object}_imputed_center_mean_w5/w15` | 对 `{object}_imputed` 做中心窗口平均 | 局部补全比例 | 判断特征可靠性 |
| `{pair}_dist_center_mean_w5/w15` | 对对象距离做中心窗口平均 | 局部稳定接近关系 | 动作段内部通常比边界更稳定 |
| `{pair}_delta_center_mean_w5/w15` | 对距离变化做中心窗口平均 | 局部接近/远离趋势 | 长刷插拔方向、靠近远端口/控制部 |

窗口统计对动作意义较大：

| 动作 | 可能关键窗口统计 |
|---|---|
| `short_brush_cleaning` | `short_brush_present_center_mean_*`, `short_brush_speed_center_mean_*`, `short_brush_to_scope_control_body_dist_center_mean_*` |
| `flush` | `syringe_present_center_mean_*`, `syringe_to_scope_distal_end_dist_center_mean_*`, `syringe_conf_center_mean_*` |
| `air_injection` | `air_gun_present_center_mean_*`, `air_gun_to_scope_distal_end_dist_center_mean_*`, `air_gun_conf_center_mean_*` |
| `long_brush_insert/withdraw` | `brush_tip_out_present_center_mean_*`, `brush_tip_out_to_scope_distal_end_delta_center_mean_*`, `hand_to_long_brush_dist_center_mean_*` |

## 5. 业务先验特征

当前 v3 的 `asformer` 和 `bigru` 使用这 8 维；`ms_tcn` 不使用。

| 特征名 | 怎么获得 | 值含义 | 主要用于 |
|---|---|---|---|
| `prior_short_clean_near` | `hand * short_brush * near(short_brush_to_scope_control_body_dist)` | 手存在、短刷存在、短刷靠近控制部 | `short_brush_cleaning` |
| `prior_short_clean_motion` | `hand * short_brush * max(short_brush_speed, abs(short_brush_to_scope_control_body_delta))` | 手+短刷存在，且短刷有局部运动或相对控制部距离变化 | `short_brush_cleaning` |
| `prior_flush_stable` | `hand * syringe * near(syringe_to_scope_distal_end_dist) * (1 - syringe_speed)` | 手和针筒存在，针筒靠近远端口且相对稳定 | `flush`，因为冲洗可能位移不大 |
| `prior_air_stable` | `hand * air_gun * near(air_gun_to_scope_distal_end_dist) * (1 - air_gun_speed)` | 手和气枪存在，气枪靠近远端口且相对稳定 | `air_injection` |
| `prior_long_signal_near_scope` | `hand * max(long_brush, brush_tip_out, brush_tip_out_imputed) * max(tip_near, long_near)` | 手存在，长刷/刷头信号存在，且靠近内镜相关位置 | 长刷插入/拔出 |
| `prior_long_towards_distal` | `hand * long_signal * clip(-brush_tip_out_to_scope_distal_end_delta, 0, 1)` | 刷头到远端口距离在变小 | 更偏 `long_brush_insert` |
| `prior_long_away_distal` | `hand * long_signal * clip(brush_tip_out_to_scope_distal_end_delta, 0, 1)` | 刷头到远端口距离在变大 | 更偏 `long_brush_withdraw` |
| `prior_hand_long_contact` | `near(hand_to_long_brush_dist) * long_signal` | 手靠近长刷/刷头信号 | 长刷操作成立的辅助证据 |

其中 `near(dist) = clip(1 - dist, 0, 1)`。

## 6. ROI RGB v2 图像特征

当前 v3 的 `asformer` 和 `bigru` 使用 ROI RGB v2，共 210 维；`ms_tcn` 不使用。RGB 特征来自 `image_train/common/roi_cache.py`。

### 6.1 RGB ROI 槽位

v2/v3 只对 7 个关键 ROI 提取 RGB：

```text
hand_top1
hand_top2
short_brush
syringe
air_gun
scope_distal_end
brush_tip_out
```

没有对 `long_brush`、`scope_control_body`、`scope_mid_section` 提取 RGB v2。原因是 v2/v3 先控制图像特征维度，只保留和五类动作最直接相关的 ROI。

### 6.2 单个 ROI 的 10 个原始 RGB 特征

每个槽位都生成以下 10 个原始特征：

| 特征名模板 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `roi_rgb_v2_{slot}_valid` | ROI manifest 中该槽位当前帧是否有 bbox | 1/0 ROI 是否存在 | 告诉模型 RGB 特征是否有效 |
| `roi_rgb_v2_{slot}_conf` | 对应 bbox 的 YOLO 置信度 | ROI 可靠性 | 图像外观和检测质量联合判断 |
| `roi_rgb_v2_{slot}_area` | padding 后 ROI bbox 面积，归一化 | ROI 可见大小 | 小目标是否清晰；遮挡程度 |
| `roi_rgb_v2_{slot}_aspect` | ROI 宽高比，裁剪到最大 4 后除以 4 | ROI 形状比例 | 区分细长工具、手、远端口等外形 |
| `roi_rgb_v2_{slot}_r_mean` | ROI 内 R 通道平均值，除以 255 | 红色平均强度 | 外观/材质/光照线索 |
| `roi_rgb_v2_{slot}_g_mean` | ROI 内 G 通道平均值，除以 255 | 绿色平均强度 | 同上 |
| `roi_rgb_v2_{slot}_b_mean` | ROI 内 B 通道平均值，除以 255 | 蓝色平均强度 | 同上 |
| `roi_rgb_v2_{slot}_brightness_mean` | ROI 内 `(R+G+B)/3` 的平均值 | 亮度 | 区分明亮器械、暗背景、遮挡 |
| `roi_rgb_v2_{slot}_brightness_std` | ROI 内亮度标准差 | 纹理/明暗变化程度 | 工具边缘、刷头、管口区域复杂度 |
| `roi_rgb_v2_{slot}_saturation_proxy` | ROI 内 `max(R,G,B)-min(R,G,B)` 的平均值 | 近似饱和度/颜色差异 | 区分不同器械外观 |

具体完整原始特征名示例：

```text
roi_rgb_v2_hand_top1_valid
roi_rgb_v2_hand_top1_conf
...
roi_rgb_v2_brush_tip_out_saturation_proxy
```

### 6.3 RGB 中心窗口均值特征

每个原始 RGB 特征都会生成一个 5 帧中心窗口均值：

```text
roi_rgb_v2_{slot}_{feature}_center_mean_w5
```

| 特征名模板 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `roi_rgb_v2_{slot}_valid_center_mean_w5` | 5 帧内 valid 平均 | ROI 在局部时间内出现比例 | 降低单帧漏检/误检影响 |
| `roi_rgb_v2_{slot}_conf_center_mean_w5` | 5 帧内 conf 平均 | 局部检测可靠性 | 判断稳定工具出现 |
| `roi_rgb_v2_{slot}_area_center_mean_w5` | 5 帧内 ROI 面积平均 | 局部可见大小 | 动作段内部稳定性 |
| `roi_rgb_v2_{slot}_aspect_center_mean_w5` | 5 帧内宽高比平均 | 局部形状稳定性 | 区分工具与手/背景 |
| `roi_rgb_v2_{slot}_r/g/b_mean_center_mean_w5` | 5 帧内 RGB 均值的平均 | 局部颜色外观 | 减少光照抖动 |
| `roi_rgb_v2_{slot}_brightness_mean_center_mean_w5` | 5 帧内亮度均值的平均 | 局部亮度 | 减少单帧曝光变化 |
| `roi_rgb_v2_{slot}_brightness_std_center_mean_w5` | 5 帧内亮度纹理平均 | 局部纹理复杂度 | 工具/刷头纹理 |
| `roi_rgb_v2_{slot}_saturation_proxy_center_mean_w5` | 5 帧内颜色差异平均 | 局部颜色稳定性 | syringe vs air_gun 等外观差异 |

### 6.4 RGB 帧间差分特征

每个原始 RGB 特征都会生成一个相邻帧差分：

```text
roi_rgb_v2_{slot}_{feature}_delta
```

| 特征名模板 | 怎么获得 | 值含义 | 用在什么地方 |
|---|---|---|---|
| `roi_rgb_v2_{slot}_valid_delta` | 当前 valid 减上一帧 valid | ROI 出现/消失变化 | 动作起止边界 |
| `roi_rgb_v2_{slot}_conf_delta` | 当前 conf 减上一帧 conf | 检测质量变化 | 遮挡、进入视野 |
| `roi_rgb_v2_{slot}_area_delta` | 当前面积减上一帧面积 | ROI 大小变化 | 工具靠近/远离、遮挡变化 |
| `roi_rgb_v2_{slot}_aspect_delta` | 当前宽高比减上一帧宽高比 | 形状变化 | 工具姿态变化 |
| `roi_rgb_v2_{slot}_r/g/b_mean_delta` | 当前 RGB 均值减上一帧 | 颜色变化 | 工具进入/离开 ROI、局部遮挡 |
| `roi_rgb_v2_{slot}_brightness_mean_delta` | 当前亮度均值减上一帧 | 亮度变化 | 动作边界、遮挡 |
| `roi_rgb_v2_{slot}_brightness_std_delta` | 当前亮度纹理减上一帧 | 纹理变化 | 刷头、气枪/针筒位置变化 |
| `roi_rgb_v2_{slot}_saturation_proxy_delta` | 当前近似饱和度减上一帧 | 颜色差异变化 | 外观变化、工具替换 |

### 6.5 RGB 特征对动作的直观意义

| ROI 槽位 | 可能帮助的动作 | 直观意义 |
|---|---|---|
| `hand_top1/hand_top2` | 全部动作 | 是否有手在稳定操作；手部外观变化通常伴随拿取/遮挡 |
| `short_brush` | `short_brush_cleaning` | 短刷是否真实出现，局部颜色/亮度/纹理是否像短刷 |
| `syringe` | `flush` | 针筒外观与靠近远端口关系结合，区分冲洗 |
| `air_gun` | `air_injection` | 气枪外观与针筒外观区分 |
| `scope_distal_end` | `flush`, `air_injection`, 长刷插拔 | 远端口附近局部变化，工具接近时 RGB/亮度/纹理会变化 |
| `brush_tip_out` | `long_brush_insert`, `long_brush_withdraw` | 刷头是否出现、消失，以及与远端口附近变化是否同步 |

## 7. v1 RGB 与 v2/v3 RGB 的区别

v1 使用的是 `rgb_roi_stats_v1`，v2/v3 当前使用的是 `roi_rgb_v2_quality_smooth`。

| 版本 | RGB 槽位 | 每槽原始特征 | 是否有窗口/差分 | 总 RGB 维度 | 当前是否用于 v3 |
|---|---|---:|---|---:|---|
| v1 | 10 个：hand_top1/top2 + 8 个非 hand 对象 | 19 维：valid、RGB 均值、RGB 标准差、RGB 4-bin 直方图 | 否 | 190 | 否 |
| v2/v3 | 7 个关键 ROI | 10 维：valid、conf、area、aspect、RGB/亮度/饱和度 | 是：w5 均值 + delta | 210 | `asformer`, `bigru` |

v1 的 19 维每槽特征具体为：

```text
rgb_roi_{slot}_valid
rgb_roi_{slot}_r_mean
rgb_roi_{slot}_g_mean
rgb_roi_{slot}_b_mean
rgb_roi_{slot}_r_std
rgb_roi_{slot}_g_std
rgb_roi_{slot}_b_std
rgb_roi_{slot}_r_hist_0_25
rgb_roi_{slot}_r_hist_25_50
rgb_roi_{slot}_r_hist_50_75
rgb_roi_{slot}_r_hist_75_100
rgb_roi_{slot}_g_hist_0_25
rgb_roi_{slot}_g_hist_25_50
rgb_roi_{slot}_g_hist_50_75
rgb_roi_{slot}_g_hist_75_100
rgb_roi_{slot}_b_hist_0_25
rgb_roi_{slot}_b_hist_25_50
rgb_roi_{slot}_b_hist_50_75
rgb_roi_{slot}_b_hist_75_100
```

v1 槽位包括：

```text
hand_top1
hand_top2
short_brush
long_brush
syringe
air_gun
scope_control_body
scope_mid_section
scope_distal_end
brush_tip_out
```

## 8. 对五类动作的特征重要性初步判断

| 动作 | 当前最有意义的特征组 | 可能还缺什么 |
|---|---|---|
| `long_brush_insert` | `brush_tip_out_*`, `brush_tip_out_to_scope_distal_end_delta`, `prior_long_towards_distal`, `roi_rgb_v2_brush_tip_out_*`, `roi_rgb_v2_scope_distal_end_*` | 更稳定的长刷/刷头跟踪；完整视频中插入方向的长窗口趋势 |
| `long_brush_withdraw` | `brush_tip_out_to_scope_distal_end_delta`, `prior_long_away_distal`, `hand_to_long_brush_dist`, `brush_tip_out_present/missing_age` | 需要和 insert 共享对象但区分方向，可能要加速度方向/轨迹拟合 |
| `short_brush_cleaning` | `short_brush_present/speed`, `short_brush_to_scope_control_body_dist`, `prior_short_clean_near`, `prior_short_clean_motion`, `roi_rgb_v2_short_brush_*` | 往复刷洗频率/周期性特征；控制部局部 ROI 外观 |
| `flush` | `syringe_present/conf`, `syringe_to_scope_distal_end_dist`, `prior_flush_stable`, `roi_rgb_v2_syringe_*`, `roi_rgb_v2_scope_distal_end_*` | 针筒推注动作细节；手-针筒关系；液体/管口变化 |
| `air_injection` | `air_gun_present/conf`, `air_gun_to_scope_distal_end_dist`, `prior_air_stable`, `roi_rgb_v2_air_gun_*` | 气枪与针筒的外观区分特征；气枪喷嘴区域局部 ROI |

## 9. 可能需要补充的特征方向

| 方向 | 说明 | 优先级 |
|---|---|---|
| 完整视频级样本 | 当前短片段数据会造成切片偏置，完整视频 + 多动作段标注更符合真实目标 | 最高 |
| 轨迹级特征 | 对每个对象维护 track_id、track_age、time_since_update、速度方向、加速度 | 高 |
| 边界特征 | 从标签生成动作开始/结束附近的 boundary target，训练 boundary head | 高 |
| 类别专属 ROI | 给 `syringe`/`air_gun`、`short_brush`/控制部、`brush_tip_out`/远端口做更专门的局部 ROI | 中高 |
| DINOv2 ROI embedding | 用冻结视觉 backbone 替代手工 RGB 统计 | 中高 |
| 长窗口趋势 | 例如 1-3 秒内距离趋势斜率，而不只是相邻帧 delta | 中 |
| 周期/频率 | 对短刷清洗提取局部往复运动频率 | 中 |
| 去除短片段时间偏置 | 完整视频训练时重新评估 `t_norm/t_sin/t_cos` 是否保留 | 中 |
