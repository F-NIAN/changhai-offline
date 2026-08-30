# 2026-07-15：结构化特征与训练方式优化

## 简单总结

113 维 v2 已可用，但未知哪类离线上下文和训练粒度更有效。本次比较 4 种特征、full sequence/sliding window
两种训练方式和三种模型，共 20 个实验。最佳为 `BiGRU + window_stats+business_priors + sliding_window`：
`ACC=0.7482`、`Frame-F1=0.5963`、`F1@0.25=0.1917`、`F1@0.5=0.1750`。帧级效果明显好于段级，说明主要瓶颈已转向边界和片段合并。

## 实验设计

- `v2`：113 维。
- `window_stats`：对动作相关列增加中心窗口均值，241 维；离线模型允许看未来帧。
- `business_priors`：增加 8 个弱业务先验，121 维。
- `window_stats+business_priors`：249 维。
- `sliding_window`：128 帧、stride 32；验证仍使用完整序列。
- 训练：3 epoch；train 10 条、val 8 条。

## 命令

```bash
python run_optimization_experiments.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed \
  --out-dir output_actionmixed_optim_experiments \
  --epochs 3 \
  --models ms_tcn asformer bigru \
  --feature-methods v2 window_stats business_priors window_stats+business_priors \
  --train-modes full_sequence sliding_window
```

## 主要结果

| 配置 | ACC | Frame-F1 | F1@0.25 | F1@0.5 |
|---|---:|---:|---:|---:|
| BiGRU + 249 维 + sliding | 0.7482 | 0.5963 | 0.1917 | 0.1750 |
| BiGRU + window stats + full | 0.6952 | 0.5283 | 0.1810 | 0.1071 |
| BiGRU + v2 + sliding | 0.7449 | 0.5996 | 0.1625 | 0.1333 |
| ASFormer + priors + full | 0.6514 | 0.5556 | 0.1583 | 0.1250 |

逐类分析：`flush` 帧级 F1 `0.9593`、`short_brush_cleaning` `0.8370`、`long_brush_insert` `0.8732`；
`long_brush_withdraw` 仅 `0.3118`；val 中 `air_injection` support 为 0。MS-TCN 多组实验出现类别坍缩。

下一步：重做覆盖全部动作的视频级 split；增加训练轮数和随机种子；为 withdraw 加入远离方向/持续时间特征；
用后处理或 boundary head 优化段边界。

## 文件

- 脚本：`run_optimization_experiments.py`。
- 总结：`docs/actionmixed_optimization_experiments_report.md`。
- 原始报告：`output_actionmixed_optim_fullseq/experiment_report.{md,json}`、
  `output_actionmixed_optim_sliding/experiment_report.{md,json}`。
