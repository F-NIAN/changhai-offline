"""
离线时序分割 baseline 总入口。

完整链路:
    数据转换 -> FeatureStore.load 完整序列 -> OfflineSegmenter 训练/验证
    -> SegmentFact -> FactLedger upsert 行

支持输入源:
    1. ActionMixed: ModelScope 数据集 lhh010/cleansight-ActionMixed
    2. Label Studio: 早期人工标注 JSON
    3. YOLO CSV: 生产检测输出占位格式
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data_transfer import CLASS_TO_ID, CLASSES, FeatureStore, labelstudio_to_feature_store, yolo_csv_to_feature_store
from dataset import (
    ACTIONMIXED_DATASET,
    ACTIONMIXED_ACTION_CLASSES,
    ACTIONMIXED_DETECTION_CLASSES,
    actionmixed_default_dir,
    actionmixed_to_feature_store,
    class_weights,
    ensure_actionmixed_dataset,
    make_normalizer,
    split_by_declared_split,
    split_by_task,
)
from segmenter import MODEL_PRIORITY, MODEL_REGISTRY
from segmentfact_ledger import labels_to_segment_facts, segment_facts_to_fact_ledger, write_jsonl, write_segment_csv

SEED = 20260701
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def smooth_labels(labels: np.ndarray, window: int = 7) -> np.ndarray:
    """多数投票平滑，减少单帧抖动。"""
    if len(labels) == 0:
        return labels
    output = labels.copy()
    radius = window // 2
    for idx in range(len(labels)):
        values = labels[max(0, idx - radius) : min(len(labels), idx + radius + 1)]
        output[idx] = Counter(values).most_common(1)[0][0]
    return output


def drop_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    """去掉过短的非 background 片段，避免把孤立抖动写成事实。"""
    if len(labels) == 0:
        return labels
    output = labels.copy()
    start = 0
    current = int(output[0])
    for idx in range(1, len(output) + 1):
        next_value = int(output[idx]) if idx < len(output) else None
        if next_value != current:
            if current != 0 and idx - start < min_len:
                left = int(output[start - 1]) if start > 0 else 0
                right = int(next_value) if next_value is not None else left
                output[start:idx] = left if left == right else 0
            start = idx
            current = next_value if next_value is not None else 0
    return output


class OfflineSegmenter:
    """统一离线分割器封装。

    输入:
        features [time, feature_dim]，来自 FeatureStore。
    输出:
        predicted_labels [time] 和 probabilities [time, class_count]。
    """

    def __init__(self, model_name: str, in_dim: int, class_count: int, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = MODEL_REGISTRY[model_name](in_dim, class_count).to(device)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_items: list[dict[str, Any]], epochs: int) -> dict[str, Any]:
        """训练模型，并保存训练集归一化参数。"""
        self.mean, self.std = make_normalizer(train_items)
        weights, support = class_weights(train_items, self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        history = []

        for epoch in range(epochs):
            random.shuffle(train_items)
            total_loss = 0.0
            self.model.train()
            for item in train_items:
                x = torch.tensor(
                    ((item["features"] - self.mean) / self.std)[None, :, :],
                    dtype=torch.float32,
                    device=self.device,
                )
                y = torch.tensor(item["labels"][None, :], dtype=torch.long, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = loss_fn(logits.transpose(1, 2).reshape(-1, len(CLASSES)), y.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            history.append({"epoch": epoch + 1, "loss": round(total_loss / max(1, len(train_items)), 6)})

        return {"history": history, "class_support_frames": support}

    def predict(self, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """对一条完整 FeatureStore 序列做离线推理。"""
        if self.mean is None or self.std is None:
            raise RuntimeError("OfflineSegmenter 必须先 fit 再 predict")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                ((item["features"] - self.mean) / self.std)[None, :, :],
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model(x)[0].transpose(0, 1)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = probs.argmax(axis=1).astype(np.int64)
        pred = drop_short_segments(smooth_labels(pred), max(3, int(round(item["fps"] * 0.25))))
        return pred, probs

    def save(self, path: Path, feature_names: list[str]) -> None:
        """保存模型权重、类别名、特征名和归一化参数。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "class_names": CLASSES,
                "feature_names": feature_names,
                "normalizer_mean": self.mean,
                "normalizer_std": self.std,
            },
            path,
        )


def frame_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """帧级指标；只作为快速 sanity check。"""
    metrics: dict[str, Any] = {"frame_accuracy": float((gt == pred).mean()), "per_class": {}}
    target_ious = []
    for idx, name in enumerate(CLASSES):
        gt_mask = gt == idx
        pred_mask = pred == idx
        union = np.logical_or(gt_mask, pred_mask).sum()
        inter = np.logical_and(gt_mask, pred_mask).sum()
        iou = None if union == 0 else float(inter / union)
        if idx != 0 and iou is not None:
            target_ious.append(iou)
        metrics["per_class"][name] = {
            "support_frames": int(gt_mask.sum()),
            "pred_frames": int(pred_mask.sum()),
            "iou": iou,
        }
    metrics["target_macro_iou"] = float(np.mean(target_ious)) if target_ious else None
    return metrics


def segment_f1(gt_facts: list[dict[str, Any]], pred_facts: list[dict[str, Any]], threshold: float = 0.25) -> float:
    """基于同类片段 IoU 的简化 Segment F1。"""
    used = set()
    matched = 0
    for pred in pred_facts:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_facts):
            if idx in used or pred["label"] != gt["label"]:
                continue
            inter = max(0, min(pred["end_frame"], gt["end_frame"]) - max(pred["start_frame"], gt["start_frame"]) + 1)
            union = max(pred["end_frame"], gt["end_frame"]) - min(pred["start_frame"], gt["start_frame"]) + 1
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used.add(best_idx)
            matched += 1
    precision = matched / len(pred_facts) if pred_facts else 0.0
    recall = matched / len(gt_facts) if gt_facts else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def label_distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize frame-level labels by split and by model class name."""
    total: Counter[int] = Counter()
    by_split: dict[str, Counter[int]] = {}
    for item in items:
        counts = Counter(int(x) for x in item["labels"].tolist())
        total.update(counts)
        split_name = str(item.get("split") or "unspecified")
        by_split.setdefault(split_name, Counter()).update(counts)

    def named_counts(counter: Counter[int]) -> dict[str, int]:
        return {CLASSES[idx]: int(counter.get(idx, 0)) for idx in range(len(CLASSES))}

    return {
        "total_frames": int(sum(total.values())),
        "total": named_counts(total),
        "by_split": {
            split_name: named_counts(counter)
            for split_name, counter in sorted(by_split.items(), key=lambda row: row[0])
        },
    }


def _count_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{count / total:.2%}"


def _metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _class_distribution_table(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    lines = ["| 模型标签ID | 动作标签 | 帧数 | 占比 |", "|---:|---|---:|---:|"]
    for idx, name in enumerate(CLASSES):
        count = int(counts.get(name, 0))
        lines.append(f"| {idx} | `{name}` | {count} | {_count_pct(count, total)} |")
    return lines


def write_training_summary_report(
    report: dict[str, Any],
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    path: Path,
) -> None:
    """Write the human-facing training summary used for project reporting."""
    distribution = report["label_distribution"]
    total_counts = distribution["total"]
    split_counts = Counter(str(item.get("split") or "unspecified") for item in items)
    frame_count = int(distribution["total_frames"])
    feature_dim = int(report["feature_dim"])
    epoch_count = int(args.epochs)

    raw_mapping_lines = ["| 原始ActionMixed ID | 原始标签 | 模型标签ID | 模型标签 |", "|---:|---|---:|---|"]
    for raw_id, raw_name in sorted(ACTIONMIXED_ACTION_CLASSES.items()):
        model_id = CLASS_TO_ID.get(raw_name, CLASS_TO_ID["idle"])
        raw_mapping_lines.append(f"| {raw_id} | `{raw_name}` | {model_id} | `{CLASSES[model_id]}` |")

    detection_names = ", ".join(
        f"{idx}:{name}" for idx, name in sorted(ACTIONMIXED_DETECTION_CLASSES.items())
    )

    model_lines = [
        "| 模型 | 最后一轮训练loss | 验证集Segment F1@0.25 | 输出片段数 | 权重文件 |",
        "|---|---:|---:|---:|---|",
    ]
    for model_name in args.models:
        data = report["models"][model_name]
        history = data.get("train", {}).get("history") or []
        last_loss = history[-1]["loss"] if history else None
        model_lines.append(
            f"| `{model_name}` | {_metric(last_loss)} | {_metric(data.get('mean_segment_f1@0.25'))} | "
            f"{data.get('segment_count', 0)} | `{data.get('model_path')}` |"
        )

    split_lines = ["| split | 样本数 |", "|---|---:|"]
    for split_name, count in sorted(split_counts.items()):
        split_lines.append(f"| `{split_name}` | {count} |")

    support_lines = ["| 模型训练标签 | 训练帧数 |", "|---|---:|"]
    first_model = args.models[0]
    support = report["models"][first_model]["train"].get("class_support_frames", {})
    for name in CLASSES:
        support_lines.append(f"| `{name}` | {int(support.get(name, 0))} |")

    lines = [
        "# ActionMixed 数据接入与离线时序分割训练汇报",
        "",
        "## 1. 本轮任务",
        "",
        (
            f"本轮将 ActionMixed 数据集转换为离线动作分割模型可直接读取的序列样本，"
            f"并重新训练 `ms_tcn`、`asformer`、`bigru` 三个 baseline。训练轮数为 `{epoch_count}`，"
            f"本轮输出目录为 `{args.out_dir}`。"
        ),
        "",
        "## 2. 动作标签体系",
        "",
        "模型端固定使用 6 个逐帧类别：`idle` 表示无动作，其余 5 类为动作分割标签。",
        "",
        *_class_distribution_table(total_counts),
        "",
        "ActionMixed 原始 `labels/data.yaml` 的动作 ID 顺序和模型内部类别顺序不同，因此转换时按名称做显式映射：",
        "",
        *raw_mapping_lines,
        "",
        "## 3. 数据如何转换为模型输入",
        "",
        "### 3.1 原始文件组织",
        "",
        (
            "已下载数据位于 `input/modelscope/lhh010__cleansight-ActionMixed`。"
            "其中 `labels/{train,val,test}/{video}.txt` 存放动作真值，"
            "`frames/{train,val,test}/{video}.mp4-{frame_id}.txt` 存放同一采样帧的 YOLO 检测框。"
        ),
        "",
        "动作标签文件每行按 `frame_id action_id` 解析；帧号与 `frames` 目录中的文件名对齐。检测框文件每行按 `class_id cx cy w h` 解析，坐标为 0-1 归一化中心点和宽高。",
        "",
        f"检测类别为：{detection_names}。",
        "",
        "### 3.2 逐帧标签对齐",
        "",
        (
            "转换脚本先读取每个视频片段实际存在的采样帧号，再把动作标签按帧号写入 `labels[T]`。"
            "未被任何动作覆盖的帧保持为 `idle`。原始动作 ID 不直接作为模型 ID 使用，而是先转成动作名称，再映射到模型内部的 `CLASSES` 顺序。"
        ),
        "",
        "### 3.3 逐帧检测框特征",
        "",
        (
            "每一帧的 YOLO 检测框先按业务对象聚合。对每个对象生成 `count/cx/cy/area/speed` 5 个特征，"
            "`count` 表示该对象检测数量，`cx/cy/area` 是检测框中心和面积的聚合值，`speed` 是相邻采样帧中心点位移按 fps 归一化后的运动量。"
        ),
        "",
        (
            "随后为关键对象对补充 `valid/dist` 关系特征，例如 `hand` 到 `short_brush`、`air_gun` 到 `scope_distal_end`、"
            "`syringe` 到 `scope_distal_end` 的可见性和距离。最后加入 `t_norm/t_sin/t_cos` 三个时间位置特征。"
        ),
        "",
        (
            f"因此每个样本最终保存为 `features[T, {feature_dim}] float32` 和 `labels[T] int64`。"
            "训练时只用训练集统计均值和标准差，对 `features` 做 `(x - mean) / std` 标准化，"
            f"再扩展为 `[1, T, {feature_dim}]` 输入模型；模型输出为 `[1, {len(CLASSES)}, T]`，逐帧做交叉熵监督。"
        ),
        "",
        "### 3.4 FeatureStore-like 落盘",
        "",
        (
            "每条视频/片段序列写为 `output/feature_store/task_<task_id>_step_1.npz`，"
            "其中包含 `features`、`labels`、`fps`、`frames`、`duration_s`、`feature_names`、`task_id`、`step_id`、`split`、`video_ref`。"
        ),
        "",
        "## 4. 数据统计",
        "",
        f"本轮共生成 `{len(items)}` 条序列样本、`{frame_count}` 个采样帧，特征维度为 `{feature_dim}`。",
        "",
        *split_lines,
        "",
        "全量逐帧标签分布：",
        "",
        *_class_distribution_table(total_counts),
        "",
        "训练集逐帧标签分布：",
        "",
        *support_lines,
        "",
        "## 5. 三个模型训练结果",
        "",
        "验证指标使用片段级 `Segment F1@0.25`：同类别预测片段与真值片段 IoU 达到 0.25 即视为命中。该指标用于快速比较 baseline，不代表最终上线阈值。",
        "",
        *model_lines,
        "",
        f"本轮按验证集 `Segment F1@0.25` 选择的默认下游模型为 `{report['selected_model']}`。",
        "",
        "## 6. 输出产物",
        "",
        f"- 结构化训练报告：`{report['report_path']}`",
        f"- 本汇报文件：`{path}`",
        f"- 特征缓存：`{report['feature_store']}`",
        f"- 预测片段 CSV/soft labels：`{report['outputs_for_downstream']['prediction_dir']}`",
        f"- 下游推荐 SegmentFact：`{report['outputs_for_downstream']['qa_reference']}`",
        f"- 下游推荐 FactLedger：`{report['outputs_for_downstream']['fact_ledger']}`",
        "",
        "## 7. 结论",
        "",
        (
            "本轮已经把动作分割标签扩展并固定为 `long_brush_insert`、`long_brush_withdraw`、"
            "`short_brush_cleaning`、`flush`、`air_injection` 五类，非动作帧统一为 `idle`。"
            "数据转换链路从原始逐帧检测框和动作标签开始，最终形成固定维度时序特征、逐帧监督标签、模型权重、预测片段和 FactLedger，可继续接入后端离线复核流程。"
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_task_ids(raw: str | None) -> list[int] | None:
    """解析逗号分隔的 task_id 列表。"""
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def clear_feature_store(feature_dir: Path) -> None:
    """清理本次输出目录下旧的 FeatureStore npz，避免不同输入源混在一起。"""
    if not feature_dir.exists():
        return
    for path in feature_dir.glob("task_*_step_*.npz"):
        path.unlink()


def clear_generated_outputs(out_dir: Path) -> None:
    """Remove stale files generated by previous pipeline runs in the same output dir."""
    patterns = [
        "pipeline_report.json",
        "training_summary_report.md",
        "*_segment_facts.jsonl",
        "*_fact_ledger.jsonl",
        "models/*_offline_segmenter.pt",
        "predictions/*_segments.csv",
        "predictions/*_soft_labels.npz",
    ]
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def build_feature_store(args: argparse.Namespace, feature_dir: Path) -> str:
    """按输入源生成 FeatureStore，并返回数据源说明。"""
    if not args.reuse_feature_store:
        clear_feature_store(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    if args.input_source == "actionmixed":
        if args.actionmixed_root is not None:
            dataset_root = args.actionmixed_root
        else:
            dataset_root = ensure_actionmixed_dataset(
                args.actionmixed_cache,
                dataset_name=args.actionmixed_dataset_name,
                force_clone=args.actionmixed_force_clone,
                refresh_lfs=args.actionmixed_refresh_lfs,
                include_images=args.actionmixed_include_images,
            )
        actionmixed_to_feature_store(dataset_root, feature_dir, fps=args.actionmixed_fps)
        return f"actionmixed:{dataset_root}"

    if args.input_source == "labelstudio":
        task_ids = parse_task_ids(args.task_ids)
        labelstudio_to_feature_store(args.labelstudio_dir, feature_dir, task_ids=task_ids)
        return f"labelstudio:{args.labelstudio_dir}"

    if args.input_source == "yolo_csv":
        if args.yolo_csv is None:
            raise ValueError("--input-source yolo_csv 需要同时传 --yolo-csv")
        yolo_csv_to_feature_store(args.yolo_csv, feature_dir)
        return f"yolo_csv:{args.yolo_csv}"

    raise ValueError(f"不支持的输入源: {args.input_source}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CleanSight 离线时序分割 baseline 全流程")
    parser.add_argument("--input-source", choices=["actionmixed", "labelstudio", "yolo_csv"], default="actionmixed")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output_actionmixed")
    parser.add_argument("--reuse-feature-store", action="store_true", help="复用 out-dir/feature_store 中已有 npz")

    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--actionmixed-root", type=Path, default=None, help="已下载的 ActionMixed 根目录")
    parser.add_argument("--actionmixed-cache", type=Path, default=Path(__file__).resolve().parent / "input" / "modelscope")
    parser.add_argument("--actionmixed-dataset-name", type=str, default=ACTIONMIXED_DATASET)
    parser.add_argument("--actionmixed-refresh-lfs", action="store_true", help="运行 git lfs pull 拉取 frames/labels")
    parser.add_argument("--actionmixed-force-clone", action="store_true", help="删除旧目录后重新 clone")
    parser.add_argument("--actionmixed-include-images", action="store_true", help="下载 images；baseline 训练默认不需要")
    parser.add_argument("--actionmixed-fps", type=float, default=7.5, help="ActionMixed 抽帧后的近似 fps")

    parser.add_argument("--labelstudio-dir", type=Path, default=Path(__file__).resolve().parent / "input" / "labelstudio")
    parser.add_argument("--task-ids", type=str, default=None, help="Label Studio 输入时可选，例如 51,58,59")
    parser.add_argument("--yolo-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_feature_store:
        clear_generated_outputs(args.out_dir)
    feature_dir = args.out_dir / "feature_store"
    input_desc = build_feature_store(args, feature_dir)

    # FeatureStore 读完整序列。
    feature_store = FeatureStore(feature_dir)
    items = feature_store.load_all()
    if not items:
        raise RuntimeError(f"没有可训练的 FeatureStore 数据: {feature_dir}")

    # ActionMixed 自带 train/val/test；其它输入源按样本随机划分。
    if args.input_source == "actionmixed":
        split = split_by_declared_split(items, seed=SEED)
    else:
        split = split_by_task(items, val_ratio=0.2, test_ratio=0.0, seed=SEED)

    in_dim = int(items[0]["features"].shape[1])
    results = {}
    ranking = []

    for model_name in args.models:
        segmenter = OfflineSegmenter(model_name, in_dim, len(CLASSES), device)
        train_info = segmenter.fit(split.train, epochs=args.epochs)

        val_reports = []
        val_scores = []
        for item in split.val:
            pred, probs = segmenter.predict(item)
            gt_facts = labels_to_segment_facts(
                item["labels"],
                item["fps"],
                item["task_id"],
                item["step_id"],
                "actionmixed_ground_truth" if args.input_source == "actionmixed" else "label_studio_ground_truth",
            )
            pred_facts = labels_to_segment_facts(
                pred,
                item["fps"],
                item["task_id"],
                item["step_id"],
                f"{model_name}_validation",
                probs,
            )
            score = segment_f1(gt_facts, pred_facts, threshold=0.25)
            val_scores.append(score)
            val_reports.append(
                {
                    "task_id": item["task_id"],
                    "split": item.get("split", ""),
                    "video_ref": item.get("video_ref", ""),
                    "frame_metrics": frame_metrics(item["labels"], pred),
                    "segment_f1@0.25": score,
                    "gt_segments": gt_facts,
                    "pred_segments": pred_facts,
                }
            )

        all_facts = []
        pred_dir = args.out_dir / "predictions"
        for item in items:
            pred, probs = segmenter.predict(item)
            facts = labels_to_segment_facts(
                pred,
                item["fps"],
                item["task_id"],
                item["step_id"],
                f"{model_name}_offline_segmenter",
                probs,
            )
            all_facts.extend(facts)
            write_segment_csv(pred_dir / f"{model_name}_task_{item['task_id']}_segments.csv", facts)
            np.savez_compressed(
                pred_dir / f"{model_name}_task_{item['task_id']}_soft_labels.npz",
                task_id=np.array([item["task_id"]]),
                predicted_labels=pred.astype(np.int64),
                probabilities=probs.astype(np.float32),
                class_names=np.array(CLASSES),
            )

        model_path = args.out_dir / "models" / f"{model_name}_offline_segmenter.pt"
        segmenter.save(model_path, items[0]["feature_names"])
        facts_path = args.out_dir / f"{model_name}_segment_facts.jsonl"
        ledger_path = args.out_dir / f"{model_name}_fact_ledger.jsonl"
        write_jsonl(facts_path, all_facts)
        ledger_rows = segment_facts_to_fact_ledger(all_facts, f"{model_name}_v0")
        write_jsonl(ledger_path, ledger_rows)

        mean_f1 = float(np.mean(val_scores)) if val_scores else None
        results[model_name] = {
            "train": train_info,
            "validation": val_reports,
            "mean_segment_f1@0.25": mean_f1,
            "model_path": str(model_path),
            "segment_facts": str(facts_path),
            "fact_ledger": str(ledger_path),
            "segment_count": len(all_facts),
            "ledger_rows": len(ledger_rows),
        }
        ranking.append({"model": model_name, "mean_segment_f1@0.25": mean_f1, "segment_count": len(all_facts)})

    ranking.sort(
        key=lambda row: (
            row["mean_segment_f1@0.25"] if row["mean_segment_f1@0.25"] is not None else -1.0,
            MODEL_PRIORITY[row["model"]],
        ),
        reverse=True,
    )
    selected = ranking[0]["model"]

    report_path = args.out_dir / "pipeline_report.json"
    summary_path = args.out_dir / "training_summary_report.md"
    report = {
        "status": "completed",
        "device": str(device),
        "epochs": args.epochs,
        "input_source": args.input_source,
        "input_desc": input_desc,
        "pipeline": "data_transfer -> FeatureStore.load -> OfflineSegmenter -> SegmentFact -> FactLedger",
        "feature_store": str(feature_dir),
        "report_path": str(report_path),
        "training_summary_report": str(summary_path),
        "task_ids": [item["task_id"] for item in items],
        "feature_dim": in_dim,
        "classes": CLASSES,
        "model_class_to_id": CLASS_TO_ID,
        "raw_action_classes": ACTIONMIXED_ACTION_CLASSES,
        "raw_to_model_action_mapping": {
            str(raw_id): {
                "raw_label": raw_name,
                "model_label": CLASSES[CLASS_TO_ID.get(raw_name, CLASS_TO_ID["idle"])],
                "model_id": CLASS_TO_ID.get(raw_name, CLASS_TO_ID["idle"]),
            }
            for raw_id, raw_name in sorted(ACTIONMIXED_ACTION_CLASSES.items())
        },
        "label_distribution": label_distribution(items),
        "split": {
            "train_task_ids": [item["task_id"] for item in split.train],
            "val_task_ids": [item["task_id"] for item in split.val],
            "test_task_ids": [item["task_id"] for item in split.test],
        },
        "models": results,
        "ranking": ranking,
        "selected_model": selected,
        "outputs_for_downstream": {
            "qa_reference": results[selected]["segment_facts"],
            "fact_ledger": results[selected]["fact_ledger"],
            "model_path": results[selected]["model_path"],
            "prediction_dir": str(args.out_dir / "predictions"),
        },
        "notes": [
            "当前实现是用于打通离线时序分割链路的 baseline，不是论文级 MS-TCN/ASFormer 完整复现。",
            "ActionMixed 原始动作 ID 与模型内部类别 ID 不同，本流程按动作名称显式映射到 idle + 五类动作标签。",
            "模型输入由 YOLO bbox 聚合出的 62 维逐帧几何/运动/关系特征和逐帧动作标签组成。",
            "SegmentFact 是动作片段事实，FactLedger 是用于离线复核与幂等 upsert 的账本行。",
        ],
    }
    report_path = args.out_dir / "pipeline_report.json"
    write_training_summary_report(report, items, args, summary_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"status": "completed", "selected_model": selected, "ranking": ranking, "report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
