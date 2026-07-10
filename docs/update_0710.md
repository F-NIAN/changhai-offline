# 2026-07-10 更新说明

## 1. 本次更新背景

本次更新基于现有离线时序分割 baseline，在原有流程上补齐了数据集接入说明，并把模型训练部分的结果整理成更易阅读的汇报内容。

## 2. 数据集接入

### 2.1 数据集来源

本轮已接入 ModelScope 的 ActionMixed 数据集，路径为：

```text
input/modelscope/lhh010__cleansight-ActionMixed/
```

### 2.2 数据组织形式

数据集主要包含两个核心部分：

```text
labels/{train,val,test}/{video}.txt
  存放逐帧动作真值，格式为 frame_id action_id。

frames/{train,val,test}/{video}.mp4-{frame_id}.txt
  存放对应采样帧的检测框信息，格式为 class_id cx cy w h。
```

这种组织方式便于把每个视频片段按时间轴展开成连续序列样本，适合做离线动作分割任务。

### 2.3 数据如何转为模型输入

数据转换流程如下：

```text
1. 读取每个视频片段的采样帧和动作标签。
2. 将 action_id 映射为统一动作名称，再映射为模型内部类别。
3. 未被动作标注覆盖的帧记为 idle。
4. 聚合检测框信息，生成对象级特征，如 count / cx / cy / area / speed。
5. 增加目标关系特征与时间位置特征。
6. 最终生成 features [T, 62] 和 labels [T]。
7. 以 .npz 形式落盘到 output/feature_store。
```

## 3. 模型与标签体系更新

本轮模型端的逐帧类别从原先的三类扩展为六类：

```text
idle
long_brush_insert
long_brush_withdraw
short_brush_cleaning
flush
air_injection
```

这使得模型不仅能识别“是否在刷”，还可以进一步区分具体动作阶段与清洗动作。

## 4. 初步训练结果

本轮训练了三个 baseline：`ms_tcn`、`asformer`、`bigru`。验证指标使用片段级 `Segment F1@0.25`：

```text
ms_tcn   : 0.1111
asformer : 0.1736
bigru    : 0.1111
```

从结果看：

```text
- asformer 当前表现最好。
- 整体指标仍偏低，说明训练还属于初步验证阶段。
- 未来可继续优化类别平衡、特征质量、训练轮数和后处理策略。
```

## 5. 结论

本次更新完成了以下几点：

```text
- 补齐数据集接入说明；
- 说明数据集如何组织以及如何转成模型输入；
- 更新动作类别体系；
- 汇总初步训练结果并给出分析结论。
```
