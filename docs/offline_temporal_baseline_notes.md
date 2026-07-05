# CleanSight 离线时序分割 baseline 建设说明

## 1. 任务定位

本次 baseline 的目标不是实时识别，也不是直接从单帧图像判断动作，而是做离线时序分割。

也就是说，模型拿到一整段视频对应的完整时间序列后，输出每一帧属于哪个动作类别，然后再把连续的动作帧合并成动作片段。

当前目标类别是：

```text
background
long_brush_cleaning
short_brush_cleaning
```

最终希望形成的数据流是：

```text
FeatureStore.load(task_id, step_id)
-> OfflineSegmenter
-> SegmentFact
-> FactLedger
```

这条链路对应三个用途：

```text
1. 送标/质检的高精度参考真值
2. 在线因果模型的 teacher，用离线软标签做蒸馏或对齐
3. 离线复算事实，幂等写入 FactLedger
```

## 2. 三种模型的原理和流程

### 2.1 MS-TCN / MS-TCN-style

MS-TCN 是 action segmentation 里常用的离线时序模型。它的核心思想是：不要只看当前帧，而是用时序卷积在时间轴上看一段上下文。

#### 输入

输入是一整段视频的逐帧特征序列：

```text
T 帧 x F 维特征
```

例如：

```text
第 1 帧: [hand_present, short_brush_present, hand_x, hand_y, short_brush_x, short_brush_y, ...]
第 2 帧: [hand_present, short_brush_present, hand_x, hand_y, short_brush_x, short_brush_y, ...]
...
第 T 帧: [...]
```

这里的输入不是原始 RGB 图像，而是从检测框、关键点、时间序列变化中提取出来的结构化特征。

#### 中间处理

MS-TCN-style 会沿着时间轴做一维卷积。卷积核不是在图像平面上滑动，而是在时间轴上滑动。

可以理解为它在问：

```text
当前帧附近几十帧里，手、刷子、内镜口的位置关系和运动趋势是什么？
这个模式更像 background、长毛刷清洗，还是短毛刷清洗？
```

MS-TCN 的一个优势是可以使用 dilated convolution，也就是带空洞的时序卷积。这样模型不需要非常深，也能看到比较长的时间范围。

在离线场景里，模型可以同时利用当前帧之前和之后的信息。例如一个动作边界点，单看这一帧可能不清楚，但看前后几秒就更容易判断。

#### 输出

模型先输出每一帧的类别概率：

```text
frame_001 -> background: 0.95, long: 0.03, short: 0.02
frame_002 -> background: 0.94, long: 0.04, short: 0.02
...
frame_150 -> background: 0.12, long: 0.03, short: 0.85
```

然后把连续的非 background 帧合并成动作段：

```json
{
  "task_id": 58,
  "step_id": 1,
  "label": "short_brush_cleaning",
  "start_ms": 6208,
  "end_ms": 11167,
  "confidence": 0.978,
  "source": "ms_tcn_final"
}
```

这就是后续可以写入 `SegmentFact` 和 `FactLedger` 的结构。

#### 本次结论

本次最终选定的是 MS-TCN-style。原因是它最符合离线 action segmentation 的定位：

```text
1. 能利用完整序列上下文
2. 对动作边界建模比较直接
3. 是离线动作分割里常见的参考 baseline
4. 相比 Transformer，在小数据场景下更稳一些
```

但需要强调：这次只是流程 baseline，动作段 F1 仍然是 0，不能作为有效精度结论。

### 2.2 ASFormer-lite

ASFormer 是基于 Transformer 的动作分割模型。它的核心思想是用 attention 在时间轴上建立远距离帧之间的关系。

#### 输入

输入同样是：

```text
T 帧 x F 维特征
```

也就是说，ASFormer-lite 不直接吃原视频帧，而是吃已经抽取好的逐帧结构化特征。

#### 中间处理

Transformer 的 attention 会让每一帧去参考其他帧的信息。

例如对于某一帧，模型可能会综合判断：

```text
1. 当前帧有没有手
2. 当前帧有没有短毛刷
3. 当前帧前后几秒内短毛刷是否持续出现
4. 手和刷子的相对位置是否稳定在某个区域
5. 这个模式是否和一段完整的清洗动作一致
```

相比 MS-TCN，ASFormer 更容易显式建模长距离依赖。比如动作开始和结束相隔很久，Transformer 理论上可以更直接地关联远处时间点。

#### 输出

输出仍然是逐帧类别概率：

```text
每帧 -> background / long_brush_cleaning / short_brush_cleaning
```

再经过后处理合并成动作段。

#### 本次结论

这次 ASFormer-lite 的 frame accuracy 看起来不低，但主要原因是 background 占比太高。模型基本倾向于预测 background，没有有效切出长毛刷或短毛刷动作边界。

因此它没有被选为当前 baseline。

### 2.3 BiGRU

BiGRU 是双向循环神经网络，属于更传统的时序模型。

#### 输入

输入仍然是：

```text
T 帧 x F 维特征
```

#### 中间处理

GRU 会按时间顺序读取序列。BiGRU 有两个方向：

```text
正向: 从视频开头看到视频结尾
反向: 从视频结尾看到视频开头
```

这样每一帧的判断既包含过去信息，也包含未来信息。

可以理解为模型在做：

```text
当前帧的状态 = 当前帧特征 + 前面发生过什么 + 后面会发生什么
```

这也符合离线模型的条件，因为离线推理时整段视频都已经可用。

#### 输出

BiGRU 最后也输出每一帧的分类概率，然后合并成动作段。

#### 本次结论

BiGRU 作为简单时序 baseline 可以保留，但本次没有有效识别动作段。它更适合作为对照模型，而不是主 baseline。

## 3. 多维特征具体是什么

本次模型没有直接使用原始视频帧，也没有直接跑 YOLO。它使用的是从 Label Studio 标注中构造出来的逐帧结构化特征。

这些特征可以理解为未来生产 FeatureStore 的模拟版本。

### 3.1 原始标注来源

Label Studio 里有两类标注：

#### 检测框 bbox

例如：

```text
hand
short_brush
syringe
air_gun
scope_control_body
scope_mid_section
scope_distal_end
brush_tip_out
```

这些标注告诉我们某些目标在某些帧上出现在哪里。

#### 时间段 timeline

例如：

```text
long_brush_insert
long_brush_withdraw
short_brush_cleaning
flush
air_injection
```

这些标注告诉我们某个动作从什么时候开始，到什么时候结束。

### 3.2 bbox 如何变成逐帧特征

对于每一类目标，脚本会把 Label Studio 的框序列展开到每一帧。

例如某一帧检测到：

```text
hand: x=0.40, y=0.52, w=0.18, h=0.22
short_brush: x=0.55, y=0.50, w=0.10, h=0.08
scope_control_body: x=0.30, y=0.60, w=0.25, h=0.20
```

那么可以形成类似这样的特征：

```text
hand_present = 1
hand_count = 1
hand_cx = 0.49
hand_cy = 0.63
hand_area = 0.0396

short_brush_present = 1
short_brush_count = 1
short_brush_cx = 0.60
short_brush_cy = 0.54
short_brush_area = 0.0080

scope_control_body_present = 1
scope_control_body_cx = 0.425
scope_control_body_cy = 0.70
scope_control_body_area = 0.0500
```

其中：

```text
cx = x + w / 2
cy = y + h / 2
area = w * h
```

如果某一帧没有对应目标，则该目标相关特征会变成 0。

### 3.3 每类目标的基础特征

对每一种目标，通常会生成这几类基础特征：

```text
present: 当前帧是否出现
count: 当前帧出现几个
cx: 框中心点 x
cy: 框中心点 y
area: 框面积
```

举例：

```text
hand_present
hand_count
hand_cx
hand_cy
hand_area

short_brush_present
short_brush_count
short_brush_cx
short_brush_cy
short_brush_area

air_gun_present
air_gun_count
air_gun_cx
air_gun_cy
air_gun_area
```

### 3.4 运动特征

除了当前帧的位置，还会构造相邻帧变化。

例如：

```text
hand_dx = hand_cx[t] - hand_cx[t-1]
hand_dy = hand_cy[t] - hand_cy[t-1]
hand_speed = sqrt(dx^2 + dy^2)
```

不需要记公式，可以理解为：

```text
hand_dx: 手在水平方向移动了多少
hand_dy: 手在垂直方向移动了多少
hand_speed: 手移动得快不快
```

对刷洗动作来说，运动特征很重要。因为很多时候单帧看起来只是“手和刷子出现了”，但动作本身需要从连续运动中判断。

例如：

```text
短毛刷清洗:
  short_brush 持续出现
  hand 和 short_brush 距离较近
  二者在小区域内有来回移动

background:
  hand 可能出现
  scope 可能出现
  但 short_brush 不出现，或者没有持续运动模式
```

### 3.5 目标之间的相对关系特征

除了单个目标的位置，还会计算目标之间的关系。

例如：

```text
dist_hand_to_short_brush
dist_hand_to_scope_control_body
dist_short_brush_to_scope_control_body
dist_hand_to_scope_distal_end
```

这些特征用来表达：

```text
手是否靠近短毛刷
短毛刷是否靠近内镜控制部
刷头是否靠近远端口
注射器/气枪是否靠近内镜相关部位
```

举例：

```text
frame 100:
hand_present = 1
short_brush_present = 1
scope_control_body_present = 1
dist_hand_to_short_brush = 0.04
dist_short_brush_to_scope_control_body = 0.08
short_brush_speed = 0.02
```

这类模式可能更像短毛刷清洗。

再比如：

```text
frame 200:
hand_present = 1
short_brush_present = 0
air_gun_present = 1
dist_hand_to_air_gun = 0.03
```

这可能更像注气或 background，而不是刷洗动作。

### 3.6 时间上下文特征

模型本身会处理时间上下文，但输入特征里也可以包含一些简单时间信息。

例如：

```text
frame_index_normalized
time_ratio
```

它们表达当前帧大概处于视频的前段、中段还是后段。

这类特征不能单独判断动作，但在短样本 baseline 里有时能提供弱提示。

### 3.7 当前实际使用的特征规模

本次 id50-59 实验中，每一帧构造了 62 维特征。

可以概括为：

```text
目标是否存在
目标数量
目标中心点
目标面积
目标运动变化
目标之间的距离/相对关系
简单时间上下文
```

最终每个 task 会保存成一个 FeatureStore NPZ 文件：

```text
task_50_step_1.npz
task_51_step_1.npz
...
task_59_step_1.npz
```

每个 NPZ 里主要包含：

```text
features: T x 62 的逐帧特征
labels: T 的逐帧类别标签
fps: 帧率
frames: 总帧数
duration_s: 视频时长
feature_names: 62 个特征名称
```

## 4. timeline 如何变成训练标签

timeline 标注用于生成监督标签，也就是告诉模型每一帧的正确类别。

### 4.1 类别映射

当前任务只关心长毛刷和短毛刷刷洗行为，所以做了简化映射。

```text
long_brush_insert
long_brush_withdraw
long_brush_cleaning
长毛刷清洗
-> long_brush_cleaning
```

```text
short_brush_cleaning
-> short_brush_cleaning
```

```text
flush
air_injection
无动作段
-> background
```

### 4.2 时间段到逐帧标签

假设某段标注是：

```text
short_brush_cleaning: 6.2s 到 11.1s
```

视频 fps 是 24，那么大概会映射成：

```text
第 149 帧到第 266 帧 = short_brush_cleaning
其他帧 = background
```

模型训练时看到的是：

```text
features[149:266] -> label = short_brush_cleaning
features[其他帧] -> label = background
```

## 5. 当前 baseline 的完整接入流程

### 5.1 当前实验输入

当前使用的是 Label Studio 导出的 JSON。

可用 task：

```text
50, 51, 52, 53, 54, 55, 56, 58, 59
```

缺失 task：

```text
57
```

其中：

```text
50, 52, 53, 54, 55, 56 是 background-only
51 有长毛刷样本
58, 59 有短毛刷样本
```

### 5.2 标注解析

脚本先解析 Label Studio JSON：

```text
读取每个 task
读取 annotations.result
区分 videorectangle 和 timelinelabels
```

其中：

```text
videorectangle -> 用于构造逐帧输入特征
timelinelabels -> 用于构造逐帧训练标签
```

### 5.3 构造 FeatureStore

对每个 task 生成：

```text
features: T x 62
labels: T
metadata: fps / frames / duration / feature_names
```

保存到：

```text
/root/shared-nvme/cleansight_real_baseline/outputs_50_59/feature_store
```

这一步相当于模拟：

```text
FeatureStore.load(task_id, step_id)
```

只不过当前 FeatureStore 的来源是 Label Studio 标注，而不是生产检测模型。

### 5.4 训练和验证

训练时模型输入：

```text
features: T x 62
labels: T
```

验证方式是 LOSO，也就是 leave-one-sequence-out：

```text
每次留一个 task 做测试
其余 task 做训练
```

这样可以在样本非常少的情况下，至少检查模型有没有跨视频泛化能力。

### 5.5 多模型对比

本次尝试了：

```text
1. MS-TCN-style
2. ASFormer-lite
3. BiGRU
```

对比指标包括：

```text
frame accuracy
target macro IoU
segment F1@0.25
boundary MAE
final_segments_count
```

当前结果：

```text
MS-TCN-style:
  F1@0.25 = 0.0
  target_mIoU = 0.0033
  frame_acc = 0.9382
  final_segments = 5

ASFormer-lite:
  F1@0.25 = 0.0
  target_mIoU = 0.0
  frame_acc = 0.9444
  final_segments = 7

BiGRU:
  F1@0.25 = 0.0
  target_mIoU = 0.0
  frame_acc = 0.9444
  final_segments = 7
```

虽然 ASFormer-lite 和 BiGRU 的 frame accuracy 略高，但主要是因为预测 background 的比例高。动作分割更重要的是 segment F1 和边界，所以最终选择 MS-TCN-style 作为 baseline。

### 5.6 推理输出

最终推理输出有三类。

#### 逐帧 soft labels

每一帧的类别概率：

```text
frame_t:
  background: 0.10
  long_brush_cleaning: 0.05
  short_brush_cleaning: 0.85
```

它可以用于后续 teacher 蒸馏。

#### SegmentFact

把连续动作帧合并成动作事实：

```json
{
  "task_id": 59,
  "step_id": 1,
  "label": "short_brush_cleaning",
  "start_ms": 56253,
  "end_ms": 63004,
  "confidence": 0.75817,
  "source": "ms_tcn_final"
}
```

#### FactLedger

把 SegmentFact 包装成幂等写入格式：

```json
{
  "ledger_op": "upsert",
  "idem_key": "59:1:ms_tcn_final:1351:1512:short_brush_cleaning:ms_tcn_v0",
  "model_version": "ms_tcn_v0",
  "fact": {
    "task_id": 59,
    "label": "short_brush_cleaning",
    "start_ms": 56253,
    "end_ms": 63004
  }
}
```

幂等 key 的意义是：同一次离线复算重复执行时，不应该插入重复事实，而应该 upsert。

## 6. 是否使用了 YOLO 框、时间段、原视频帧

### 6.1 本次使用了什么

本次使用了：

```text
1. Label Studio 的 bbox 标注
2. Label Studio 的 timeline 时间段标注
```

bbox 用于构造模型输入特征。

timeline 用于构造训练标签。

### 6.2 本次没有使用什么

本次没有使用：

```text
1. 原视频 RGB 帧作为模型输入
2. YOLO 真实推理输出
3. 光流模型
4. 图像 backbone 提取的视觉 embedding
```

所以当前 baseline 的准确定位是：

```text
基于检测序列特征的离线时序分割 baseline
```

而不是：

```text
端到端视频理解模型
```

## 7. 后续如何接入真正推理流程

未来正式接入时，Label Studio bbox 应该被 YOLO 或其他检测模型输出替换。

目标流程应该是：

```text
原视频
-> 抽帧
-> YOLO / 检测模型
-> 每帧 bbox / keypoint / confidence
-> FeatureStore.save(task_id, step_id, source='detector')
-> FeatureStore.load(task_id, step_id)
-> OfflineSegmenter.predict(...)
-> SegmentFact
-> FactLedger upsert
```

### 7.1 检测模型需要提供什么

检测模型至少需要提供：

```text
frame_index
timestamp_ms
label
bbox_x
bbox_y
bbox_w
bbox_h
confidence
track_id 可选
```

例如：

```json
{
  "frame_index": 150,
  "timestamp_ms": 6250,
  "label": "short_brush",
  "bbox": [0.55, 0.50, 0.10, 0.08],
  "confidence": 0.91,
  "track_id": 3
}
```

这些检测结果再被转换成和当前 baseline 一样的 62 维特征。

### 7.2 FeatureStore 需要稳定的 schema

后续最关键的是对齐 FeatureStore schema。

需要明确：

```text
1. 每一帧怎么编号
2. fps 和 timestamp 怎么存
3. bbox 坐标是像素值还是归一化值
4. 缺失检测怎么填
5. 多个同类目标怎么聚合
6. detector confidence 是否进入特征
7. track_id 是否用于运动特征
```

如果这些不统一，离线模型训练和线上/离线推理会出现输入分布不一致。

### 7.3 OfflineSegmenter 接口建议

后续可以把模型包装成类似接口：

```python
class OfflineSegmenter:
    def load_model(self, model_path):
        ...

    def predict(self, sequence):
        ...
        return {
            "frame_probs": probs,
            "frame_labels": labels,
            "segments": segment_facts,
        }
```

输入：

```text
FeatureStore.load(task_id, step_id) 返回的完整序列
```

输出：

```text
逐帧概率
逐帧预测标签
SegmentFact 列表
```

### 7.4 后处理建议

模型输出逐帧标签后，还需要后处理：

```text
1. 合并连续同类动作帧
2. 过滤过短片段
3. 平滑短暂抖动
4. 计算片段 confidence
5. 转换 frame index 到 timestamp
6. 生成幂等 fact_id
```

例如：

```text
short, short, background, short, short
```

如果中间 background 只有 1 到 2 帧，可能应该平滑成一个连续的 short_brush_cleaning 段。

## 8. 当前结果如何汇报

可以这样对组长说：

```text
我这边已经把离线时序分割 baseline 跑通了。当前不是直接用原视频帧训练，而是把 Label Studio 的 bbox 标注转换成逐帧检测序列特征，把 timeline 时间段转换成逐帧监督标签，然后训练 OfflineSegmenter。

我尝试了 MS-TCN-style、ASFormer-lite 和 BiGRU 三种模型。三者都能完成 FeatureStore -> OfflineSegmenter -> SegmentFact -> FactLedger 的流程。最终选择 MS-TCN-style 作为 baseline，因为它更符合离线 action segmentation 的建模方式。

但是当前数据量严重不足，id50-59 中 id57 缺失，50/52/53/54/55/56 是 background-only，长毛刷只有 task 51，短毛刷主要只有 58/59。因此 frame accuracy 被 background 拉高，segment F1 仍为 0。这个结果主要证明链路接通，并暴露了后续真正模型接入需要解决的数据和 schema 问题。
```

## 9. 当前主要坑点

### 9.1 数据不足

长毛刷样本只有 task 51，短毛刷样本主要只有 task 58 和 59。

这会导致模型很难学到稳定边界。

### 9.2 类别不均衡

大量帧是 background。

所以 frame accuracy 很容易虚高。

更应该看：

```text
segment F1
boundary error
target action IoU
```

### 9.3 长毛刷标注不规范

当前 task 51 使用的是中文 `长毛刷清洗` sequence-style 标注。

但规范里期望的是：

```text
long_brush_insert
long_brush_withdraw
```

这会影响模型学习“插入”和“拔出”的边界。

### 9.4 缺少 id57

Label Studio 页面需要登录态，当前无法从 URL 自动拉取 id57。

如果要完整覆盖 id50-59，需要提供：

```text
1. Label Studio 导出 JSON
2. 或 API token
3. 或登录 cookie/session
```

### 9.5 当前输入不是真正 YOLO 输出

本次用 Label Studio bbox 模拟检测序列。

真正上线时需要替换为：

```text
YOLO 推理框 -> FeatureStore -> OfflineSegmenter
```

因此后续必须和检测模型同学对齐 bbox schema 和类别名称。

### 9.6 时间戳和帧率需要统一

时间段标注是秒级或毫秒级，模型内部是帧级。

必须统一：

```text
fps
frame_index
timestamp_ms
start_ms
end_ms
```

否则边界误差可能不是模型问题，而是时间换算问题。

## 10. 下一步建议

### 10.1 数据侧

优先补齐：

```text
1. id57 标注
2. 更多 long_brush_insert / long_brush_withdraw 样本
3. 更多 short_brush_cleaning 样本
4. 明确 flush / air_injection 是否永远作为 background
```

### 10.2 模型侧

继续保留 MS-TCN-style 作为主 baseline。

可以继续尝试：

```text
1. class weight / focal loss，缓解 background 过多
2. 边界平滑后处理
3. 最短动作段过滤
4. 加入 detector confidence
5. 加入 track_id 运动轨迹特征
6. 使用真实 MS-TCN 或成熟 action segmentation repo
```

### 10.3 工程侧

下一步最重要的是把临时脚本整理成项目内标准接口：

```text
FeatureStore.load(task_id, step_id)
OfflineSegmenter.predict(sequence)
SegmentFact writer
FactLedger upsert
```

这样模型同学只需要替换 OfflineSegmenter 的内部实现，不需要改数据流和业务接口。

