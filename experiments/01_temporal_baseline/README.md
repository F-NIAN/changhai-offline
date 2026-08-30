# 2026-07-05：基础离线时序分割 baseline

## 简单总结

当时仓库只有离线模型任务设想，没有统一数据入口、三模型接口或后端可消费产物。本次建立
`数据转换 -> 数据划分 -> 训练/验证 -> SegmentFact -> FactLedger` 全链路，并实现 MS-TCN、
ASFormer-lite、BiGRU 三个可替换 baseline。结果是工程闭环可运行，但早期小样本指标只用于验证接口，
后续需要接入真实 ActionMixed 数据并固定特征契约。

## 已有内容与本次变动

- 已有：Label Studio JSON、示例视频和后端 `FeatureStore.load(task_id, step_id)` 设想。
- 新增：`data_transfer.py`、`dataset.py`、`run_pipeline.py`、`segmentfact_ledger.py`。
- 新增：`segmenter/ms_tcn.py`、`segmenter/asformer.py`、`segmenter/bigru.py`。
- 约定：按 task/video 划分 split，避免同源片段泄漏；checkpoint 保存归一化和特征信息。

## 输入、输出与脚本

- 输入：`input/labelstudio/*.json`，或通过 `--yolo-csv` 传入检测 CSV。
- 主脚本：`run_pipeline.py`。
- 数据脚本：`data_transfer.py`、`dataset.py`。
- 模型脚本：`segmenter/*.py`。
- 账本脚本：`segmentfact_ledger.py`。
- 默认输出：`output/feature_store/`、`output/models/`、`output/predictions/`、
  `output/*_segment_facts.jsonl`、`output/*_fact_ledger.jsonl`、`output/pipeline_report.json`。

## 复现命令

```bash
python run_pipeline.py \
  --input-source labelstudio \
  --labelstudio-dir input/labelstudio \
  --models ms_tcn asformer bigru \
  --epochs 5 \
  --out-dir output
```

可选参数：`--task-ids 51,58,59` 限定任务；`--models` 选择模型；`--reuse-feature-store` 复用 NPZ。

## 结果、分析和下一步

基础产物和模型文件均成功生成，证明数据契约与下游写出可行。限制是示例数据少、标签体系和真实检测分布
尚未固定，因此不应引用早期数值做模型优劣判断。下一步是接入 ActionMixed、按视频切分、扩展到五类动作，
并验证三模型在统一输入上的表现。

## 对应文档

- `docs/baseline_overview.md`：完整原理、输入输出格式和模型说明。
- `output/training_summary_report.md`：后续更新后的基础训练汇总。
- `output/pipeline_report.json`：结构化运行结果。
