# 预测动作时间线可视化

## 简单总结

数值指标不能直观看出过分割、边界偏移和类别混淆。本实验加载三模型最佳权重，对 test split 画
Ground Truth/Prediction 时间线。结果图可用于汇报和定位长刷插入/拔出、短刷、推流等动作的边界问题；
它不计算新指标，也不会改变 checkpoint。

## 脚本和命令

- 脚本：`scripts/visualize_results.py`。
- 使用说明：`docs/visualization_readme.md`。

```bash
python scripts/visualize_results.py \
  --feature-dir output_actionmixed_best_models/feature_store_v2 \
  --model-dir output_actionmixed_best_models/models \
  --out-dir output_actionmixed_best_models/visualizations \
  --models ms_tcn asformer bigru \
  --device cpu \
  --all-tests
```

参数：`--all-tests` 把全部 test 序列堆叠到一张图；不传时选择代表样本；`--device` 可设 `cpu` 或 CUDA 设备。

## 输出与观察

```text
output_actionmixed_best_models/visualizations/
├── ms_tcn_vs_gt_alltests.png
├── asformer_vs_gt_alltests.png
└── bigru_vs_gt_alltests.png
```

从定量报告和时间线共同判断：BiGRU 的主要动作帧覆盖较好，但 withdraw 和段边界仍弱；MS-TCN 有类别坍缩；
ASFormer 比 MS-TCN 稳但碎片仍多。下一步可在图中增加置信度曲线、边界误差标记和视频/任务 ID，便于逐例复核。
