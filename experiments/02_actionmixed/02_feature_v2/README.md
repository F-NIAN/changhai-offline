# 2026-07-15：113 维结构化特征 v2

## 简单总结

68 维版本对双手、同类多框和短时漏检表达不足。本次改为 hand top-2 独立槽、其它对象 top-1，加入置信度、
missing age、插值标志、关键对象关系及距离变化，形成 `clean_bbox_v2_top1_impute` 113 维特征。1 epoch 快测中
MS-TCN 和 BiGRU 均达到 `F1@0.25=0.3125`；结果主要验证转换正确，不能证明最终模型排名。

## 变动

- hand：`hand_count + 2 × 8` 维槽位。
- 8 个非手对象：每类 9 维，按置信度保留 top-1。
- 7 组关系：`valid/dist/delta`。
- 时间编码：`t_norm/t_sin/t_cos`。
- 最长 6 帧短缺失插值，并保留 `present` 与 `imputed` 区别。

## 命令和参数

```bash
python run_pipeline.py \
  --input-source actionmixed \
  --actionmixed-root input/modelscope/lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 1 \
  --out-dir output_actionmixed_feature_v2
```

## 结果

| 模型 | val F1@0.25 | 预测段数 |
|---|---:|---:|
| MS-TCN | 0.3125 | 24 |
| BiGRU | 0.3125 | 71 |
| ASFormer | 0.1708 | 97 |

分析：转换链路和 113 维 checkpoint 契约已跑通，但 1 epoch 的模型差异波动大。下一步需要在统一 split 上比较
显式离线窗口统计、业务先验和滑窗训练，并至少记录帧级与段级两套指标。

## 文件

- 代码：`data_transfer.py`、`dataset.py`、`run_pipeline.py`。
- 文档：`docs/actionmixed_feature_v2_report.md`、`output_actionmixed_feature_v2/training_summary_report.md`。
- 结果：`output_actionmixed_feature_v2/pipeline_report.json`。
