# offline-model 完整交接文档（2026-08-28）

本文由原 handoff 更新整理而成。所有路径均以仓库根目录为基准。仓库级导航和命令见
[`README.md`](../README.md)，逐实验详细说明见 [`experiments/`](../experiments/README.md)。

## 1. 当前交付结论

- 已形成 `检测框/标注 -> FeatureStore -> 离线时序模型 -> SegmentFact -> FactLedger` 完整链路。
- 当前结构化特征主候选：`BiGRU + window_stats+business_priors + sliding_window`。
- 当前验证结果：`ACC=0.7482`、`Frame-F1=0.5963`、`F1@0.25=0.1917`、`F1@0.5=0.1750`。
- 当前候选权重：`output_actionmixed_best_models/models/best_bigru_offline_segmenter.pt`。
- RGB/ROI 已完成 v1-v4，并用真实 YOLO v0.3 预测框复验；该线 v4 BiGRU test raw
  `ACC=0.7028`、post `F1@0.5=0.2019`。
- 上述实验均为小数据、少 epoch 结果；结构化实验的 val 不含 `air_injection`，不能作为上线验收。

## 2. 仓库状态与交付边界

- 分支：`main`，对齐 `origin/main`；本次整理没有 commit 或 push。
- 已有跟踪文件修改包括训练总结、优化报告和最佳权重脚本注释/参数整理。
- 新增内容包括 `experiments/` 文档树、RGB/ROI 实验、真实 YOLO 框复验、可视化和设计文档。
- `input/modelscope/lhh010__cleansight-ActionMixed` 是本地数据快照/嵌套仓库，不应作为普通源码提交。
- `yolo/*.zip`、`output*/models/*.pt`、图片和生成数据可能体积较大，应按组内资产策略单独交付。
- 本次整理保留正式数据、正式结果和模型权重，只清理过 smoke 临时目录和 Python 缓存。

建议源码提交范围：

```text
README.md
experiments/
docs/
*.py
segmenter/
image_train/common/
image_train/v2/
image_train/v3/
image_train/v4/
yolo_image_train/common/
yolo_image_train/run_yolo_image_train.py
scripts/
```

需要单独评估是否入库：

```text
input/
output*/
image_train/output*/
yolo_image_train/generated/
yolo_image_train/output*/
yolo_image_train/weights/
yolo/*.zip
```

## 3. 时间顺序工作记录

| 日期 | 当时状况与工作 | 结果 | 分析与后续 | 详细文档 |
|---|---|---|---|---|
| 07-05 | 从零建立数据转换、三模型和账本闭环 | 全链路跑通 | 工程验证数据少，需接真实数据 | `experiments/01_temporal_baseline/README.md` |
| 07-10 | 接入 ActionMixed 五类动作与逐帧框 | 16 序列/4501 帧初训 | 简单对象聚合不足 | `experiments/02_actionmixed/00_dataset_onboarding/README.md` |
| 07-15 | 68 维完整序列三模型比较 | ASFormer 1 epoch `F1@0.25=0.4607` | 只作阶段候选 | `experiments/02_actionmixed/01_full_sequence_baseline/README.md` |
| 07-15 | 升级 113 维 v2 | hand top-2、遮挡补全、关系 delta 跑通 | 需系统比较上下文和训练方式 | `experiments/02_actionmixed/02_feature_v2/README.md` |
| 07-15 | 20 组特征/训练实验 | BiGRU 249 维滑窗最佳 | 段边界、withdraw、split 是主要问题 | `experiments/02_actionmixed/03_feature_training_optimization/README.md` |
| 07-15 | 按模型最佳 recipe 训练 | 保留三模型权重，推荐 BiGRU | checkpoint 输入契约必须严格匹配 | `experiments/02_actionmixed/04_best_checkpoints/README.md` |
| 07-21~28 | RGB/ROI v1-v4 | v4 BiGRU 标注框 test ACC `0.5964` | proxy ROI 有效，MS-TCN 仍不稳 | `experiments/03_rgb_roi/README.md` |
| 08-04 | 真实 YOLO 框复验 v1-v4 | v4 BiGRU test ACC `0.7028` | 需多 seed/完整视频确认 | `experiments/04_real_yolo_rgb_roi/README.md` |
| 08 | GT/预测时间线可视化 | 三模型 all-test PNG | 便于看过分割和边界偏移 | `experiments/05_visualization/README.md` |

## 4. 核心脚本与用途

| 路径 | 用途 | 主要配置位置 |
|---|---|---|
| `run_pipeline.py` | 数据转换、训练、验证、预测、账本总入口 | `make_parser()` 与文件顶部 `SEED` |
| `run_optimization_experiments.py` | 特征方法 × 训练模式 × 模型矩阵 | 文件顶部 `SEED`，CLI 列表参数 |
| `train_best_checkpoints.py` | 按模型最佳 recipe 训练候选权重 | 顶部 `DEFAULT_*`、`BEST_RECIPES` |
| `image_train/v4/train_image_v4.py` | RGB/proxy ROI v4 | `image_train/v4/configs/image_train_v4.json` |
| `yolo_image_train/common/prepare_yolo_actionmixed.py` | 两组 YOLO 权重生成统一 8 类框 | CLI conf/IoU/imgsz/batch |
| `yolo_image_train/run_yolo_image_train.py` | 统一复跑 v1-v4 | 顶部 `DEFAULT_*`、`VERSION_SCRIPTS` |
| `scripts/visualize_results.py` | GT/预测动作时间线图 | 顶部 `DEFAULT_*`、CLI |

完整命令和参数解释见根 `README.md` 及各叶子实验 README。

## 5. 关键数据和模型契约

ActionMixed 输入：

```text
frames/{train,val,test}/{video}.mp4-{frame_id}.txt  # 检测框
labels/{train,val,test}/{video}.mp4.txt             # 动作标签
```

FeatureStore-like 输出：

```text
features: [T,D] float32
labels: [T] int64
feature_names / feature_version / task_id / split / fps / frames
```

checkpoint 必须绑定：模型结构、输入维度、`feature_names`、`feature_version`、normalizer、recipe。当前三候选
输入维度分别为 MS-TCN 113、ASFormer 121、BiGRU 249，不能互换。

## 6. 结果解读

- `flush`、`short_brush_cleaning`、`long_brush_insert` 的帧级表现相对较好。
- `long_brush_withdraw` 的方向与边界弱，容易和 insert/idle 混淆。
- 结构化实验 val 中 `air_injection` support 为 0，必须重做 split 后再评估。
- MS-TCN 在当前数据/超参下频繁类别坍缩；增加维度或滑窗并未稳定解决。
- RGB/ROI 结果说明外观和 proxy 区域有价值，但高维 flat concat 不是最终结构，建议 group-aware/gated fusion。
- 帧级 F1 明显高于 segment F1，表明后续重点是边界、段数和状态转移，而不只是继续堆输入特征。

## 7. 下一步优先级

1. 按完整视频重做 train/val/test，保证五类动作都有 support，并固定 split 清单和 fingerprint。
2. 最佳方案至少跑 3 个随机种子、20~100 epoch，报告均值/方差和独立 test。
3. 为 `long_brush_withdraw` 增加轨迹方向、远离速度、持续时间和 brush tip 状态。
4. 引入 tracker/插值质量，避免单帧 top-1 跳变；保留 missing/imputed/track 状态。
5. 把结构化、对象 ROI、proxy ROI 分组编码，尝试 gate/cross-attention，而非继续直接 concat。
6. 将边界优化独立为后处理或 boundary head，并同时报告 raw/post 指标。
7. 后端接入前确认能否访问 RGB 帧、track id、统一框格式和 checkpoint YAML/metadata。

## 8. 交接验收建议

接收方先运行：

```bash
python run_pipeline.py --help
python run_optimization_experiments.py --help
python train_best_checkpoints.py --help
python image_train/v4/train_image_v4.py --help
python yolo_image_train/run_yolo_image_train.py --help
```

再用最小模型/少 epoch 输出到新的临时目录验证环境，不要覆盖历史 `output*`。确认文档、代码、数据和权重的
交付渠道后，再决定 Git 提交范围。
