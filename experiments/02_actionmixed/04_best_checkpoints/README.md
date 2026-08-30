# 2026-07-15：三模型最佳 recipe 权重

## 简单总结

优化实验表明不同模型偏好的特征不同。本次不再强制共用输入，而是按各自最佳 recipe 训练并保存权重：
MS-TCN 用 113 维 v2 全序列，ASFormer 用 121 维先验全序列，BiGRU 用 249 维组合滑窗。BiGRU 综合最佳，
MS-TCN 出现明显类别坍缩；交付时推荐 BiGRU，同时保留三模型用于追溯。

## 配置与命令

配置集中在 `train_best_checkpoints.py` 顶部 `BEST_RECIPES`。

```bash
python train_best_checkpoints.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed \
  --out-dir output_actionmixed_best_models \
  --epochs 3 \
  --models ms_tcn asformer bigru
```

## 结果

| 模型 | recipe | dim | ACC | Frame-F1 | F1@0.25 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|
| MS-TCN | v2 + full | 113 | 0.2433 | 0.1121 | 0.1000 | 0.0750 |
| ASFormer | priors + full | 121 | 0.6514 | 0.5556 | 0.1583 | 0.1250 |
| BiGRU | window+priors + sliding | 249 | 0.7482 | 0.5963 | 0.1917 | 0.1750 |

## 产物与接入

- 脚本：`train_best_checkpoints.py`。
- 报告：`output_actionmixed_best_models/best_model_report.{md,json}`。
- 权重：`output_actionmixed_best_models/models/best_*_offline_segmenter.pt`。
- 可视化：`output_actionmixed_best_models/visualizations/`。

checkpoint 内含 recipe、feature names/version、normalizer 和验证元数据。推理端必须按相同 recipe 重建特征，
不能把 249 维 BiGRU 权重直接接到 68/113 维后端转换器。下一步先完善数据划分，再考虑增加 epoch；不建议在
当前小数据上仅凭一次 3 epoch 结果直接发布。
