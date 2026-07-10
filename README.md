# CleanSight Offline Model Baseline

```text
数据转换 -> 数据集划分 -> 模型训练 -> 验证 -> SegmentFact -> FactLedger
```

```text
FeatureStore.load(task_id, step_id)
  -> OfflineSegmenter
  -> SegmentFact
  -> FactLedger
```

## 目录

```text
offline_model_baseline/
  data_transfer.py          # Label Studio / YOLO 输出 -> FeatureStore-like npz
  dataset.py                # 构造完整序列数据集，按 task 划分 train/val/test
  run_pipeline.py           # 总入口，跑完整 baseline 流程
  segmentfact_ledger.py     # 逐帧预测 -> SegmentFact -> FactLedger upsert 行
  input/                    # 测试输入，包含 labelstudio/ 和 test/
  output/                   # 运行输出
  docs/                     # 说明文档
  segmenter/
    ms_tcn.py               # 简化 MS-TCN baseline
    asformer.py             # 简化 ASFormer/Transformer baseline
    bigru.py                # 简化 BiGRU baseline
```

## 运行

进入目录后直接运行：

```bash
python run_pipeline.py --models ms_tcn asformer bigru --epochs 5
```

如果想直接接入 ModelScope 数据集并把原始文件下载到工作区：

```bash
python run_pipeline.py --models ms_tcn asformer bigru --epochs 3 --dataset-root input/modelscope --dataset-name lhh010/cleansight-ActionMixed
```

此时脚本会：
- 把数据集下载到 input/modelscope/lhh010__cleansight-ActionMixed；
- 将其转换为 FeatureStore-like npz，写入 output/feature_store；
- 继续走训练、验证与结果输出流程。

如果只想快速 smoke test：

```bash
python run_pipeline.py --models ms_tcn --epochs 1
```

也可以指定任务：

```bash
python run_pipeline.py --task-ids 51,58,59 --epochs 5
```

默认读取：

```text
input/labelstudio/*.json
```

默认输出：

```text
output/
```

关键输出：

```text
output/feature_store/task_<task_id>_step_<step_id>.npz
output/models/<model>_offline_segmenter.pt
output/predictions/<model>_task_<task_id>_segments.csv
output/<model>_segment_facts.jsonl
output/<model>_fact_ledger.jsonl
output/pipeline_report.json
```

## 接入说明

当前 `FeatureStore` 是最小文件版实现。后续接业务后端时，把 `FeatureStore.load(task_id, step_id)` 替换成真实 FeatureStore 读取即可，后半段保持：

```text
完整检测序列 -> OfflineSegmenter.predict -> SegmentFact -> FactLedger upsert
```

