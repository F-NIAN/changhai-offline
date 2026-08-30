可视化说明：生成测试集上模型预测与真值对比图

用途：对比 `ms_tcn`、`asformer`、`bigru` 三个最佳权重在同一代表性测试样本上的预测差异。

运行示例：

```bash
python scripts/visualize_results.py \
  --feature-dir output_actionmixed_best_models/feature_store_v2 \
  --model-dir output_actionmixed_best_models/models \
  --out-dir output_actionmixed_best_models/visualizations
```

默认行为：
- 从 `--feature-dir` 读取 FeatureStore .npz，优先选择有 `split==test` 的样本；若无，则选择最长序列作为代表样本。
- 对每个模型加载 `best_{model}_offline_segmenter.pt`（位于 `--model-dir`），并使用 checkpoint 中的归一化参数执行预测。
- 为每个模型生成一张 PNG，文件名格式：`{model}_vs_gt_{video_ref}.png`，存放在 `--out-dir`。

注意事项：
- 脚本假定 checkpoint 包含 `state_dict`、`normalizer_mean`、`normalizer_std`、`class_names` 或可回退到默认类名。
- 若你的特征文件或模型不在默认路径，请通过命令行参数调整。
- 本脚本在 CPU 上运行即可；若需要 GPU，请传 `--device cuda` 并且本机有可用 GPU。

如需扩展：
- 可把脚本改为对所有 test 样本都生成图，或并排绘制多模型在同一图片上对比。欢迎告诉我你希望的展示形式，我可以替你改写脚本。
