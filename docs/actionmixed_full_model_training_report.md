# ActionMixed 三模型完整 baseline 训练报告

## 数据集与命令

- 数据集：`input/modelscope/lhh010__cleansight-ActionMixed`
- 数据集提交：`9361ebd3 Fix snapshot_download example in README`
- 训练命令：

```bash
python run_pipeline.py --input-source actionmixed \
  --actionmixed-root input\modelscope\lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 1 \
  --out-dir output_actionmixed_full
```

## 数据如何转成模型输入

ActionMixed 里有两类文本文件：

- `labels/{train,val,test}/{video}.txt`：动作真值，每行按 `frame_id action_id` 解析。
- `frames/{train,val,test}/{video}.mp4-{frame_id}.txt`：同一采样帧的 YOLO 检测框，每行按 `class_id cx cy w h` 解析，坐标是 0-1 归一化中心点和宽高。

转换流程：

1. 按视频名把 `frames` 目录中的逐帧检测文件分组，并按 `frame_id` 排序。
2. 读取对应 `labels` 文件，将原始动作 ID 先映射为动作名称，再映射到模型内部 6 类：`idle`、`long_brush_insert`、`long_brush_withdraw`、`short_brush_cleaning`、`flush`、`air_injection`。
3. 对每一帧 YOLO 框按业务对象聚合。`hand` 使用 top-2 独立槽位，避免两只手被加权合并；其它对象生成 `count/cx/cy/area/speed`；再补充关键对象对 `valid/dist` 和时间位置编码 `t_norm/t_sin/t_cos`。
4. 保存为 FeatureStore-like `.npz`：`features[T, 68] float32`、`labels[T] int64`、`fps`、`frames`、`task_id`、`step_id`、`split`、`video_ref`。
5. 训练时用训练集统计 `mean/std`，把每条序列标准化后扩展为 `[1, T, 68]` 输入模型；模型输出 `[1, 6, T]`，逐帧交叉熵监督。

本轮共解析 `21` 条序列、`5655` 个采样帧。

## 三种模型实现

### ms_tcn

当前 `ms_tcn` 已改为 `MS-TCN + BiLSTM`：

- 输入 `[B,T,F]` 先做 LayerNorm 和线性投影。
- BiLSTM 双向读取完整视频序列，利用未来帧和过去帧上下文。
- 多阶段 TCN 使用膨胀残差卷积扩大感受野，并逐阶段细化边界。
- 输出 `[B,C,T]`，用于逐帧动作分类。

### asformer

当前 `asformer` 使用 ASFormer 风格结构：

- 输入 `[B,T,F]` 做投影和正弦位置编码。
- 每层包含局部时序卷积、多头 self-attention 和前馈网络。
- 局部卷积关注边界附近细节，attention 负责跨长时间范围建模。
- 输出 `[B,C,T]`。

### bigru

当前 `bigru` 使用更完整的双向 GRU baseline：

- 输入 `[B,T,F]` 做 LayerNorm 和线性投影。
- 3 层 BiGRU 双向读取完整序列。
- 时序卷积 head 做局部平滑和逐帧分类。
- 输出 `[B,C,T]`。

## 训练结果

指标：验证集 `Segment F1@0.25`，同类别预测片段与真值片段 IoU 达到 0.25 视为命中。

| 模型 | 训练 loss | 验证 Segment F1@0.25 | 输出片段数 | 本地权重 |
|---|---:|---:|---:|---|
| `asformer` | 2.1737 | 0.4607 | 105 | `output_actionmixed_full/models/asformer_offline_segmenter.pt` |
| `bigru` | 1.8156 | 0.3125 | 23 | `output_actionmixed_full/models/bigru_offline_segmenter.pt` |
| `ms_tcn` | 2.1078 | 0.1146 | 93 | `output_actionmixed_full/models/ms_tcn_offline_segmenter.pt` |

本轮按验证集片段指标选择 `asformer` 作为当前推荐 baseline。

## 输出产物

以下产物均保留在本地 `output_actionmixed_full/`，没有随仓库提交：

- 结构化报告：`pipeline_report.json`
- 自动训练汇报：`training_summary_report.md`
- FeatureStore-like 输入：`feature_store/*.npz`
- 三模型权重：`models/*_offline_segmenter.pt`
- 预测片段与软标签：`predictions/`
- 推荐下游 SegmentFact：`asformer_segment_facts.jsonl`
- 推荐下游 FactLedger：`asformer_fact_ledger.jsonl`

## 结论与注意点

- 三种模型均已用最新 ActionMixed 数据完成一次完整训练、验证和结果导出。
- 由于只跑 1 epoch，结果主要用于验证流程和模型接入，不代表最终精度上限。
- 当前 `asformer` 的验证片段指标最好，可作为下一步后端离线推理权重接入候选。
- 本轮已把 offline-model 数据转换同步到后端 `clean.py` 的 68 维 hand top-2 特征，后续接入时仍需确认 checkpoint 的 `feature_names` 与后端输出列名逐列一致。
