# CleanSight 离线动作时序模型

本仓库完成了从检测框/标注到离线动作时间线的完整链路：

```text
ActionMixed / Label Studio / YOLO CSV
  -> FeatureStore-like NPZ
  -> MS-TCN / ASFormer / BiGRU
  -> 逐帧预测与动作片段
  -> SegmentFact
  -> FactLedger JSONL
```

当前建议基线是 `BiGRU + window_stats+business_priors + sliding_window`。在 2026-07-15 的验证集实验中，
其 `ACC=0.7482`、`Frame-F1=0.5963`、`F1@0.25=0.1917`、`F1@0.5=0.1750`。这些数值来自
仅 21 条序列、3 epoch 的小数据实验；验证集没有 `air_injection`，不能作为最终上线精度。

## 从哪里开始看

- [实验总索引](experiments/README.md)：按日期列出每次实验、改动、结果、分析、脚本和结果文档。
- [基础链路说明](docs/baseline_overview.md)：数据格式、模型、SegmentFact 和 FactLedger 契约。
- [完整交接文档](docs/handoff_update_20260828.md)：工作区状态、交付范围和后续注意事项。
- [当前特征清单](image_train/current_feature_inventory.md)：113/121/241/249 维结构化特征及 RGB/ROI 特征。
- [多模态后续设计](docs/actionmixed_multimodal_future_design.md)：DINOv2、VideoMAE、track 和边界优化路线。

## 仓库结构

```text
offline-model/
├── README.md                         # 仓库入口、快速命令和交付结论
├── experiments/                     # 按时间和所属关系组织的实验文档树
├── docs/                            # 原理、阶段报告、handoff 和设计文档
├── data_transfer.py                 # Label Studio/YOLO -> FeatureStore-like NPZ
├── dataset.py                       # ActionMixed 转换、划分、归一化与数据集工具
├── run_pipeline.py                  # 基础训练、验证、预测和账本输出总入口
├── run_optimization_experiments.py  # 特征方法 × 训练方式 × 模型的对比实验
├── train_best_checkpoints.py        # 按模型最佳 recipe 训练最终候选权重
├── segmentfact_ledger.py            # 帧标签 -> SegmentFact -> FactLedger
├── segmenter/                       # MS-TCN、ASFormer、BiGRU 实现
├── image_train/                     # 基于标注框的 RGB/ROI v1-v4 实验
├── yolo_image_train/                # 用真实 YOLO 预测框复跑 RGB/ROI v1-v4
├── scripts/visualize_results.py     # GT/预测动作时间线可视化
├── input/                           # 本地数据；提交前需检查体积和嵌套仓库状态
├── output*/                         # 历史结构化特征、权重、预测与报告
└── yolo/                            # 本地 YOLO 模型包；大文件不建议直接提交
```

`experiments/` 只负责组织实验说明，源码和历史结果仍保留在原路径，避免破坏已有命令、checkpoint 元数据和
报告链接。所有文档中的路径均以本仓库根目录为基准。

## 时间线摘要

| 日期 | 阶段 | 当时状况与动作 | 结果与结论 |
|---|---|---|---|
| 2026-07-05 | 基础链路 | 建立数据转换、三模型、SegmentFact/FactLedger 闭环 | 链路跑通；早期结果只用于工程验证 |
| 2026-07-10 | ActionMixed 接入 | 接入五类动作和逐帧 YOLO 框 | 完成 16 条序列初训；确认需升级对象特征 |
| 2026-07-15 | 三模型 68 维 baseline | 用完整序列比较三模型 | 1 epoch 下 ASFormer `F1@0.25=0.4607`，仅作链路候选 |
| 2026-07-15 | 113 维 v2 | hand top-2、非 hand top-1、遮挡补全、关系 delta | 1 epoch 下 MS-TCN/BiGRU `F1@0.25=0.3125` |
| 2026-07-15 | 特征与训练优化 | 比较 4 类特征、2 种训练方式、3 种模型 | BiGRU 249 维滑窗方案整体最佳 |
| 2026-07-15 | 最佳权重 | 按每个模型各自最优 recipe 重训 | 保留三权重；推荐 BiGRU，MS-TCN 有类别坍缩 |
| 2026-07-21~28 | RGB/ROI v1-v4 | 逐步加入 ROI 缓存、质量特征、段后处理和 proxy ROI | 标注框实验 v4 的 BiGRU test raw ACC `0.5964` |
| 2026-08-04 | 真实 YOLO 框复验 | 用 YOLO v0.3 预测框替代理想标注框重跑 v1-v4 | v4 BiGRU test raw ACC `0.7028`，post `F1@0.5=0.2019` |
| 2026-08-28 | 交接整理 | 清理 smoke/cache，补齐文档树和使用说明 | 不改动数据和正式实验产物，不执行 commit/push |

完整逐次记录见 [experiments/README.md](experiments/README.md)。

## 环境与数据

建议在仓库根目录执行命令。核心依赖包括 Python、NumPy、PyTorch；RGB/ROI 实验还需要 Pillow、OpenCV、
Matplotlib，真实 YOLO 框生成需要 Ultralytics。

ActionMixed 默认目录：

```text
input/modelscope/lhh010__cleansight-ActionMixed/
```

部分脚本使用其内部的：

```text
input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed/
```

数据目录是本地 ModelScope/Git LFS 数据快照，提交代码前不要把嵌套仓库状态和大文件误当作源码提交。

## 常用命令

查看任一入口的完整参数：

```bash
python run_pipeline.py --help
python run_optimization_experiments.py --help
python train_best_checkpoints.py --help
python image_train/v4/train_image_v4.py --help
python yolo_image_train/run_yolo_image_train.py --help
```

### 1. 跑基础 ActionMixed 三模型链路

```bash
python run_pipeline.py \
  --input-source actionmixed \
  --actionmixed-root input/modelscope/lhh010__cleansight-ActionMixed \
  --models ms_tcn asformer bigru \
  --epochs 3 \
  --out-dir output_actionmixed
```

关键参数：

- `--input-source`：`actionmixed`、`labelstudio` 或 `yolo_csv`。
- `--actionmixed-root`：已下载的数据集根目录；传入后不会使用默认缓存位置猜测数据。
- `--models`：要训练的模型子集。
- `--epochs`：每个模型训练轮数。
- `--out-dir`：FeatureStore、权重、预测、报告和账本的输出根目录。
- `--reuse-feature-store`：复用输出目录中已有 NPZ，跳过数据转换。

### 2. 跑结构化特征/训练方式对比

```bash
python run_optimization_experiments.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed \
  --out-dir output_actionmixed_optim_experiments \
  --epochs 3 \
  --models ms_tcn asformer bigru \
  --feature-methods v2 window_stats business_priors window_stats+business_priors \
  --train-modes full_sequence sliding_window
```

### 3. 训练三模型各自最佳权重

```bash
python train_best_checkpoints.py \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed \
  --out-dir output_actionmixed_best_models \
  --epochs 3 \
  --models ms_tcn asformer bigru
```

recipe 集中在 `train_best_checkpoints.py` 顶部的 `BEST_RECIPES`。checkpoint 会保存特征名、特征版本、
归一化参数和 recipe；推理端必须按同一 recipe 构造输入。

### 4. 跑 RGB/ROI v4

```bash
python image_train/v4/train_image_v4.py \
  --config image_train/v4/configs/image_train_v4.json \
  --dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --out-dir image_train/output_v4 \
  --models ms_tcn asformer bigru
```

v4 参数优先集中在 `image_train/v4/configs/image_train_v4.json`；命令行参数可覆盖配置。

### 5. 用真实 YOLO 框复跑 v1-v4

```bash
python yolo_image_train/run_yolo_image_train.py \
  --source-dataset-root input/modelscope/lhh010__cleansight-ActionMixed/cleansight-ActionMixed \
  --generated-dataset-root yolo_image_train/generated/actionmixed_yolo_v03 \
  --output-root yolo_image_train \
  --yolo-dir yolo \
  --versions v1 v2 v3 v4 \
  --device 0
```

若已经生成检测结果，可加 `--skip-prepare`；只检查命令不执行可加 `--dry-run`。

### 6. 输出 GT/预测时间线图

```bash
python scripts/visualize_results.py \
  --feature-dir output_actionmixed_best_models/feature_store_v2 \
  --model-dir output_actionmixed_best_models/models \
  --out-dir output_actionmixed_best_models/visualizations \
  --models ms_tcn asformer bigru \
  --all-tests
```

## 输入输出契约

FeatureStore-like NPZ 的核心数组：

```text
features:   float32[T, D]
labels:     int64[T]
timestamps: float32[T] 或由 frame/fps 推导
meta:       task_id、step_id、split、feature_names、feature_version
```

模型输入/输出：

```text
input:         [B, T, D]
logits:        [B, C, T]
probabilities: [B, T, C]
pred_labels:   [B, T]
```

下游关键产物：

- `output*/models/*.pt`：模型、归一化参数和特征契约。
- `output*/predictions/`：逐任务动作段 CSV 和 soft labels。
- `output*/*_segment_facts.jsonl`：连续帧合并后的动作事实。
- `output*/*_fact_ledger.jsonl`：供后端幂等 upsert 的账本记录。
- `output*/pipeline_report.json`、`*.md`：结构化/人读结果。

## 交付结论与限制

- 当前可交付主基线是 `output_actionmixed_best_models/models/best_bigru_offline_segmenter.pt`。
- 249 维最佳权重不兼容后端早期 68/113 维输入；接入时必须同步 `feature_names` 和 `feature_version`。
- 现有验证集不覆盖 `air_injection`，并且总体序列数很少；下一轮必须重做视频级划分并跑多随机种子。
- `long_brush_withdraw` 的方向和边界仍弱，应加入轨迹、远离趋势和边界约束。
- RGB/ROI 结果受检测框来源影响，`image_train/`（标注框）和 `yolo_image_train/`（真实预测框）不可混报。
- `input/`、`output*/`、`yolo/*.zip` 可能包含大文件；提交前按组内模型资产策略筛选。
