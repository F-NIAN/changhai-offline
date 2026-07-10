# CleanSight 离线时序分割 Baseline 说明

本文档说明当前 `changhai-offline` 仓库中的离线时序分割 baseline，包括完整流程、三种模型、输入输出数据格式，以及本轮接入 ModelScope ActionMixed 数据集后的更新内容。

## 1. 任务目标

本 baseline 面向内镜清洗视频的离线动作分割任务，当前实验版已经从原来的三分类扩展为六类逐帧动作分类：

```text
0 idle
1 long_brush_insert
2 long_brush_withdraw
3 short_brush_cleaning
4 flush
5 air_injection
```

目标不是实时告警，而是离线处理完整序列，尽量得到更稳定的动作片段边界。整体定位对应原需求：

```text
FeatureStore 读完整序列
  -> OfflineSegmenter
  -> SegmentFact
  -> FactLedger
```

三类用途：

```text
1. 给标注/质检提供高精度参考结果。
2. 作为未来在线因果模型的 teacher，输出 soft label 做蒸馏或对齐。
3. 离线复算动作事实，并通过 FactLedger 幂等写入。
```

## 2. 仓库结构

```text
changhai-offline/
  data_transfer.py
  dataset.py
  run_pipeline.py
  segmentfact_ledger.py
  segmenter/
    ms_tcn.py
    asformer.py
    bigru.py
  input/
    labelstudio/
      id58.json
      id59.json
      video_addresses.txt
    test/
      test.mp4
  output/
    feature_store/
    models/
    predictions/
    pipeline_report.json
  docs/
    baseline_overview.md
```

各文件职责：

```text
data_transfer.py
  负责数据转换。把 Label Studio 标注或 YOLO 检测结果转成 FeatureStore-like .npz。

dataset.py
  负责构造完整序列数据集，按 task 划分 train/val/test，并计算归一化参数和类别权重。

run_pipeline.py
  总入口。串起数据转换、数据划分、模型训练、验证、SegmentFact 输出和 FactLedger 输出。

segmentfact_ledger.py
  把逐帧预测结果合并成动作片段 SegmentFact，再包装成 FactLedger upsert 行。

segmenter/
  放三种时序模型：MS-TCN 简化版、ASFormer-lite、BiGRU。
```

## 3. 完整流程

### 3.1 数据转换

入口函数在 `data_transfer.py`：

```python
labelstudio_to_feature_store(...)
yolo_csv_to_feature_store(...)
FeatureStore.load(...)
```

当前主要使用 Label Studio 导出的 JSON。转换逻辑是：

```text
Label Studio JSON
  -> 解析 videorectangle 目标框
  -> 解析 timelinelabels 时间段
  -> 逐帧展开 bbox / 时间段
  -> 构造多维时序特征 features
  -> 构造逐帧动作标签 labels
  -> 保存为 output/feature_store/task_<task_id>_step_<step_id>.npz
```

其中：

```text
videorectangle
  用于生成模型输入特征，比如手、短刷、长刷、针筒、气枪、内镜部位的位置和运动。

timelinelabels
  用于生成逐帧监督标签，比如 long_brush_cleaning / short_brush_cleaning。
```

### 3.2 FeatureStore 读取完整序列

当前 `FeatureStore` 是文件版最小实现：

```python
feature_store = FeatureStore(feature_dir)
item = feature_store.load(task_id, step_id)
```

读取出来的一条样本是一个字典：

```python
{
    "task_id": int,
    "step_id": int,
    "features": np.ndarray,      # [time, feature_dim]
    "labels": np.ndarray,        # [time]
    "fps": float,
    "frames": int,
    "duration_s": float,
    "feature_names": list[str],
    "file_upload": str,
    "video_ref": str,
}
```

后续接业务后端时，可以把这个文件版 `FeatureStore.load` 替换成真实数据库、对象存储或特征服务读取，后面的模型接口不需要大改。

### 3.3 数据集划分

数据划分在 `dataset.py`：

```python
split_by_task(items, val_ratio=0.2, test_ratio=0.0)
```

当前按 `task_id` 划分，而不是把同一个视频切碎后随机划分。这样可以避免同一视频的相似帧同时出现在训练集和验证集，导致验证结果虚高。

训练时还会计算：

```text
normalizer mean/std
  对 features 做标准化。

class weights
  缓解 background 帧远多于动作帧的问题。
```

### 3.4 模型训练和验证

模型统一由 `OfflineSegmenter` 包装：

```python
segmenter = OfflineSegmenter(model_name, in_dim, class_count, device)
segmenter.fit(train_items, epochs)
pred, probs = segmenter.predict(item)
```

输入：

```text
features: [time, feature_dim]
```

模型内部会加 batch 维：

```text
[1, time, feature_dim]
```

输出：

```text
pred:  [time]                 # 每帧预测类别 id
probs: [time, class_count]    # 每帧类别概率，可作为 soft label
```

预测后会做两个简单后处理：

```text
1. 7 帧多数投票平滑，减少孤立帧抖动。
2. 删除短于约 0.25 秒的非 background 片段。
```

### 3.5 SegmentFact

`SegmentFact` 是模型输出的业务事实，表示一个连续动作片段。

由 `segmentfact_ledger.py` 生成：

```python
labels_to_segment_facts(labels, fps, task_id, step_id, source, probs)
```

输入：

```text
labels: [time]，逐帧类别
fps: 视频帧率
task_id / step_id: 业务标识
probs: [time, class_count]，可选，用于计算 confidence
```

输出示例：

```json
{
  "fact_id": "58:1:ms_tcn_offline_segmenter:151:268:short_brush_cleaning",
  "task_id": 58,
  "step_id": 1,
  "label": "short_brush_cleaning",
  "start_frame": 151,
  "end_frame": 268,
  "start_ms": 6250,
  "end_ms": 11167,
  "confidence": 0.93699,
  "source": "ms_tcn_offline_segmenter"
}
```

### 3.6 FactLedger

`FactLedger` 是对 `SegmentFact` 的幂等写入包装，用于离线复算。

由下面函数生成：

```python
segment_facts_to_fact_ledger(facts, model_version)
```

输出示例：

```json
{
  "ledger_op": "upsert",
  "idem_key": "58:1:ms_tcn_offline_segmenter:151:268:short_brush_cleaning:ms_tcn_v0",
  "run_id": "offline-segmenter-...",
  "model_version": "ms_tcn_v0",
  "fact": {
    "...": "SegmentFact 内容"
  }
}
```

这样同一模型版本、同一动作片段重复复算时，可以根据 `idem_key` 做 upsert，避免重复插入。

## 3.7 接入 ActionMixed 数据集

本轮已经把数据源从原先的 Label Studio 结果拓展到 ModelScope 的 ActionMixed 数据集，数据目录位于：

```text
input/modelscope/lhh010__cleansight-ActionMixed/
```

数据集的组织形式如下：

```text
labels/{train,val,test}/{video}.txt
  每行格式为 frame_id action_id
  记录每个采样帧对应的动作真值。

frames/{train,val,test}/{video}.mp4-{frame_id}.txt
  每行格式为 class_id cx cy w h
  记录同一帧上的 YOLO 检测框信息。
```

转换流程如下：

```text
1. 读取每个视频片段实际存在的采样帧号。
2. 按 frame_id 读取动作标签，未覆盖到的帧默认记为 idle。
3. 将原始 action_id 映射为统一的动作名称，再映射到模型内部类别顺序。
4. 对每一帧的检测框做聚合，生成 count / cx / cy / area / speed 等特征。
5. 补充 hand-to-short_brush、air_gun-to-scope_distal_end、syringe-to-scope_distal_end 等关系特征。
6. 加入 t_norm / t_sin / t_cos 时间位置特征。
7. 最终形成 features [T, 62] 和 labels [T]，写成 FeatureStore-like .npz 文件。
```

本轮生成的样本统计如下：

```text
16 条序列样本
4501 个采样帧
特征维度 62
训练/验证/测试划分分别为 8 / 6 / 2
```

### 3.8 初步训练结果分析

本轮分别训练了 `ms_tcn`、`asformer`、`bigru` 三个 baseline，评估使用片段级 `Segment F1@0.25`。当前结果显示：

```text
ms_tcn   val F1@0.25 = 0.1111
asformer val F1@0.25 = 0.1736
bigru    val F1@0.25 = 0.1111
```

从结果看，`asformer` 在当前条件下表现最好，但整体指标仍然偏低，这说明：

```text
1. 当前训练轮数较少，仍属于初步验证阶段，尚未充分收敛。
2. 新增的动作类别带来更明显的类别不平衡，尤其 short_brush_cleaning 样本相对较少。
3. 现有特征主要来自 YOLO 检测框，信息密度有限，仍有提升空间。
4. 片段级评估对边界误差敏感，容易放大轻微错分和过分割现象。
```

因此，这轮结果更像是验证数据链路和训练流程是否通畅，而不是最终上线版本的性能上限。后续可以继续优化特征质量、类别权重、训练轮数和后处理规则。

## 4. 输入数据格式

### 4.1 Label Studio JSON

默认输入目录：

```text
input/labelstudio/
```

当前随仓库保留：

```text
input/labelstudio/id58.json
input/labelstudio/id59.json
input/labelstudio/video_addresses.txt
```

Label Studio 中使用两类标注：

```text
videorectangle
  目标框标注，例如 hand、short_brush、syringe、air_gun、scope_control_body 等。

timelinelabels
  时间段标注，例如 long_brush_insert、long_brush_withdraw、short_brush_cleaning。
```

当前动作标签映射：

```text
long_brush_insert     -> long_brush_cleaning
long_brush_withdraw   -> long_brush_cleaning
long_brush_cleaning   -> long_brush_cleaning
short_brush_cleaning  -> short_brush_cleaning
flush                 -> background
air_injection         -> background
background            -> background
```

### 4.2 YOLO CSV

`data_transfer.py` 里保留了 YOLO CSV 转换入口：

```python
yolo_csv_to_feature_store(yolo_csv, feature_dir)
```

期望字段：

```text
task_id
frame
fps
label
x1
y1
x2
y2
confidence
```

可选字段：

```text
width
height
track_id
instance_id
```

说明：

```text
YOLO CSV 当前只生成检测特征，labels 默认为 background。
如果要用 YOLO 输出训练时序模型，还需要合并人工时间段标注或其它真值来源。
```

### 4.3 FeatureStore .npz

转换后的模型输入文件：

```text
output/feature_store/task_<task_id>_step_<step_id>.npz
```

`.npz` 是 NumPy 压缩包，里面包含多个数组。

主要字段：

```text
features       float32 [time, feature_dim]
labels         int64   [time]
fps            float
frames         int
duration_s     float
feature_names  str[feature_dim]
task_id        int
step_id        int
file_upload    str
video_ref      str
```

当前特征维度：

```text
feature_dim = 62
```

特征大致分为三类：

```text
1. 单目标特征
   例如 hand_count、hand_cx、hand_cy、hand_area、hand_speed。

2. 目标关系特征
   例如 hand_to_short_brush_valid、hand_to_short_brush_dist。

3. 时间位置特征
   例如 t_norm、t_sin、t_cos。
```

## 5. 输出数据格式

### 5.1 模型文件

```text
output/models/<model>_offline_segmenter.pt
```

保存内容：

```text
state_dict
model_name
class_names
feature_names
normalizer_mean
normalizer_std
```

### 5.2 每任务预测片段 CSV

```text
output/predictions/<model>_task_<task_id>_segments.csv
```

字段：

```text
fact_id
task_id
step_id
label
start_frame
end_frame
start_ms
end_ms
confidence
source
```

### 5.3 Soft Labels

```text
output/predictions/<model>_task_<task_id>_soft_labels.npz
```

字段：

```text
task_id
predicted_labels: [time]
probabilities: [time, class_count]
class_names
```

用途：

```text
1. 给未来在线模型做 teacher signal。
2. 分析模型不确定性。
3. 对齐离线模型和在线模型输出。
```

### 5.4 SegmentFact JSONL

```text
output/<model>_segment_facts.jsonl
```

每一行是一条动作片段事实。

### 5.5 FactLedger JSONL

```text
output/<model>_fact_ledger.jsonl
```

每一行是一条幂等 upsert 记录。

### 5.6 总报告

```text
output/pipeline_report.json
```

包含：

```text
status
device
task_ids
feature_dim
classes
split
models
ranking
selected_model
outputs_for_downstream
notes
```

## 6. 三种模型说明

### 6.1 MS-TCN 简化版

文件：

```text
segmenter/ms_tcn.py
```

输入：

```text
[batch, time, feature_dim]
```

中间处理：

```text
1. 先用 1x1 Conv1d 把 feature_dim 投影到 hidden channel。
2. 再通过多层残差膨胀卷积块。
3. 膨胀卷积在时间轴上扩大感受野，让每一帧预测能参考前后较长时间范围。
4. 最后用 1x1 Conv1d 输出每一帧的类别 logits。
```

输出：

```text
[batch, class_count, time]
```

特点：

```text
优点：
  结构简单，训练快，适合做 action segmentation baseline。
  对局部边界和中等长度上下文比较友好。

限制：
  当前是简化实现，不是完整论文版 MS-TCN。
  感受野由卷积层数和 dilation 决定，不像 Transformer 那样天然全局。
```

当前代码的 dilation：

```text
[1, 2, 4, 8, 16, 1, 2, 4]
```

卷积核大小为 3，因此理论感受野约：

```text
1 + 2 * (1 + 2 + 4 + 8 + 16 + 1 + 2 + 4) = 77 帧
```

如果视频约 24 FPS，则大约是 3.2 秒上下文。

### 6.2 ASFormer-lite

文件：

```text
segmenter/asformer.py
```

输入：

```text
[batch, time, feature_dim]
```

中间处理：

```text
1. Linear 投影到 hidden 维。
2. 加入正弦位置编码，让模型知道帧顺序。
3. 使用 TransformerEncoder 做 self-attention。
4. 每一帧可以和同一条序列里的其它帧交互。
5. 最后 Linear 输出每一帧类别 logits。
```

输出：

```text
[batch, class_count, time]
```

特点：

```text
优点：
  理论上能利用整条视频的全局上下文。
  适合离线场景，不受实时因果约束。

限制：
  当前是 ASFormer-lite，不是完整 ASFormer 论文实现。
  小数据下容易过拟合或产生过多碎片段。
  序列很长时显存和计算量会明显增加。
```

### 6.3 BiGRU

文件：

```text
segmenter/bigru.py
```

输入：

```text
[batch, time, feature_dim]
```

中间处理：

```text
1. Linear 投影到 hidden 维。
2. 双向 GRU 从正向和反向各读一遍完整序列。
3. 每一帧的 hidden state 同时包含过去和未来上下文。
4. 最后 Linear 输出每帧类别 logits。
```

输出：

```text
[batch, class_count, time]
```

特点：

```text
优点：
  结构直观，参数量相对可控。
  能利用双向上下文，适合离线分割。

限制：
  长序列训练可能比卷积慢。
  对很长距离依赖不一定比 Transformer 稳定。
  不是实时因果模型，因为它使用未来帧。
```

## 7. 当前运行和结果说明

运行命令：

```bash
python run_pipeline.py
```

默认会尝试训练：

```text
ms_tcn
asformer
bigru
```

当前仓库中 `output/` 随附的是一次轻量运行产物，主要用于说明数据格式和链路结构。容器中曾完整跑通三模型流程，结果显示：

```text
status: completed
selected_model: ms_tcn
feature_dim: 62
task_ids: 50, 51, 52, 53, 54, 55, 56, 58, 59
```

需要注意：

```text
1. 当前可用标注数据较少。
2. 验证集可能被划到 background-only 任务，导致 segment_f1@0.25 为 0。
3. 这说明流程跑通，但不能把当前指标当作可靠模型精度。
4. 真正评估边界精度，需要更多包含长刷/短刷动作的验证样本。
```

## 8. 如何运行

安装好依赖后，在仓库根目录执行：

```bash
python run_pipeline.py
```

只快速测试 MS-TCN：

```bash
python run_pipeline.py --models ms_tcn --epochs 1
```

指定任务：

```bash
python run_pipeline.py --task-ids 51,58,59 --epochs 5
```

指定 YOLO CSV：

```bash
python run_pipeline.py --yolo-csv path/to/yolo_output.csv
```

## 9. 后续改进方案

### 9.1 数据层改进

```text
1. 补充更多包含 long_brush_cleaning 和 short_brush_cleaning 的正样本。
2. 保证验证集和测试集都有动作段，避免 background-only 验证。
3. 区分 long_brush_insert 和 long_brush_withdraw，不一定都合并成 long_brush_cleaning。
4. 统一 Label Studio 标注规范，减少同一动作多种名称。
5. 增加标注质量检查，比如时间段是否越界、是否重叠、是否缺少必要 bbox。
```

### 9.2 特征层改进

```text
1. 用真实 YOLO/检测模型输出替换 Label Studio bbox，模拟生产输入。
2. 增加 track_id，构造更稳定的目标轨迹特征。
3. 增加刷头与内镜远端、手与刷子的相对速度/角度等动作特征。
4. 引入视觉 backbone 特征，而不只依赖 bbox 几何特征。
5. 对缺失检测框做插值或置信度建模。
```

### 9.3 模型层改进

```text
1. 接入完整 MS-TCN / MS-TCN++ 实现。
2. 接入完整 ASFormer，并使用论文中的 decoder/refinement 结构。
3. 尝试 ActionFormer、Temporal Convolution + Transformer 混合结构。
4. 增加边界损失或 segment-level loss，而不只做 frame-wise CrossEntropy。
5. 使用类别平衡采样或 focal loss 处理 background 占比过高问题。
```

### 9.4 评估层改进

```text
1. 固定 train/val/test split，不要每次随机划分。
2. 汇报 segment F1@0.1 / 0.25 / 0.5。
3. 汇报 boundary MAE，即起止边界平均误差。
4. 区分 frame accuracy 和 action IoU，避免 background 掩盖问题。
5. 加入人工可视化检查，把预测片段叠到视频时间轴上。
```

### 9.5 工程接入改进

```text
1. 把文件版 FeatureStore 替换成业务后端的真实 FeatureStore。
2. 把 SegmentFact / FactLedger 写入数据库，而不是只写 JSONL。
3. 给 run_pipeline 增加配置文件，管理模型、路径、类别和训练参数。
4. 增加单元测试，覆盖 Label Studio 转换、SegmentFact 合并和 FactLedger 幂等键。
5. 增加独立 inference 脚本，只加载 checkpoint 做离线复算，不重新训练。
```

## 10. 和业务后端的接口建议

推荐后续服务化接口：

```text
POST /offline-segmentation/tasks/{task_id}/steps/{step_id}/run
GET  /offline-segmentation/tasks/{task_id}/steps/{step_id}/facts
```

后端内部流程：

```text
FeatureStore.load(task_id, step_id)
  -> OfflineSegmenter.predict(...)
  -> labels_to_segment_facts(...)
  -> segment_facts_to_fact_ledger(...)
  -> FactLedger upsert
```

当前代码已经把这些步骤拆开，后续接入时主要替换数据读取和事实写入两端即可。

