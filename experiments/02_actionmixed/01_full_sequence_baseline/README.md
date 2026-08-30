# 2026-07-15：ActionMixed 三模型完整序列 baseline

## 简单总结

当时 ActionMixed 已接入，但模型还没有在同一套完整序列输入上系统比较。本次用 68 维 hand top-2/对象几何/
关系/时间编码，分别训练 MS-TCN+BiLSTM、ASFormer 和 3 层 BiGRU。1 epoch 验证中 ASFormer 的
`Segment F1@0.25=0.4607` 最高，因此被选为阶段候选；由于训练轮数极少，这一结论只说明当时配置下的相对表现。

## 实验设计

- 数据：21 条序列、5655 个采样帧。
- 输入：`features[T,68]`，完整视频序列训练。
- 监督：六类逐帧交叉熵。
- 评估：验证集 `Segment F1@0.25`。

## 命令

```bash
python run_pipeline.py \
  --input-source actionmixed \
  --actionmixed-root input/modelscope/lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 1 \
  --out-dir output_actionmixed_full
```

## 结果

| 模型 | train loss | val F1@0.25 | 预测段数 |
|---|---:|---:|---:|
| ASFormer | 2.1737 | 0.4607 | 105 |
| BiGRU | 1.8156 | 0.3125 | 23 |
| MS-TCN | 2.1078 | 0.1146 | 93 |

分析：ASFormer 对长上下文有优势，但预测段偏多；BiGRU 段数更保守；MS-TCN 在 1 epoch 下未收敛。下一步不应
直接增加模型规模，而应先提高检测结构特征质量并固定 checkpoint 的特征契约。

## 文件

- 脚本：`run_pipeline.py`、`dataset.py`、`segmenter/*.py`。
- 实验文档：`docs/actionmixed_full_model_training_report.md`。
- 结果：`output_actionmixed_full/pipeline_report.json`、`output_actionmixed_full/training_summary_report.md`。
- 权重：`output_actionmixed_full/models/*_offline_segmenter.pt`。
