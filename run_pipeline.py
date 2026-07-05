"""
离线模型 baseline 总运行脚本。

完整流程：
    数据转换 -> 数据集划分 -> 模型训练 -> 验证 -> SegmentFact -> FactLedger

对应任务要求：
    FeatureStore.load(task_id, step_id)
        -> OfflineSegmenter
        -> SegmentFact
        -> FactLedger

默认输入：
    ../input/labelstudio/*.json

默认输出：
    ./output/
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

from data_transfer import CLASSES, FeatureStore, labelstudio_to_feature_store, yolo_csv_to_feature_store
from dataset import class_weights, make_normalizer, split_by_task
from segmenter import MODEL_PRIORITY, MODEL_REGISTRY
from segmentfact_ledger import labels_to_segment_facts, segment_facts_to_fact_ledger, write_jsonl, write_segment_csv

SEED = 20260701
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def smooth_labels(labels: np.ndarray, window: int = 7) -> np.ndarray:
    """多数投票平滑，减少孤立帧抖动。"""
    if len(labels) == 0:
        return labels
    output = labels.copy()
    radius = window // 2
    for idx in range(len(labels)):
        window_values = labels[max(0, idx - radius): min(len(labels), idx + radius + 1)]
        output[idx] = Counter(window_values).most_common(1)[0][0]
    return output


def drop_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    """删除过短的非 background 片段，避免把抖动写成事实。"""
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
    """统一的离线时序分割器包装。

    输入：
        一条完整序列 features [time, feature_dim]

    输出：
        predicted_labels [time]
        probabilities [time, class_count]
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
                x = torch.tensor(((item["features"] - self.mean) / self.std)[None, :, :], dtype=torch.float32, device=self.device)
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
            x = torch.tensor(((item["features"] - self.mean) / self.std)[None, :, :], dtype=torch.float32, device=self.device)
            logits = self.model(x)[0].transpose(0, 1)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = probs.argmax(axis=1).astype(np.int64)
        pred = drop_short_segments(smooth_labels(pred), max(3, int(round(item["fps"] * 0.25))))
        return pred, probs

    def save(self, path: Path, feature_names: list[str]) -> None:
        """保存权重、类别名、特征名和归一化参数。"""
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
    """帧级指标。背景占比高时，这个指标只作为 sanity check。"""
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


def parse_task_ids(raw: str | None) -> list[int] | None:
    """解析逗号分隔的 task_id 列表。"""
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CleanSight 离线时序分割 baseline 全流程")
    parser.add_argument("--labelstudio-dir", type=Path, default=Path(__file__).resolve().parent / "input" / "labelstudio")
    parser.add_argument("--yolo-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--task-ids", type=str, default=None, help="可选，例如 51,58,59；为空时使用全部任务")
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dir = args.out_dir / "feature_store"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据转换：Label Studio / YOLO 输出 -> FeatureStore-like npz。
    task_ids = parse_task_ids(args.task_ids)
    if args.labelstudio_dir.exists():
        labelstudio_to_feature_store(args.labelstudio_dir, feature_dir, task_ids=task_ids)
    if args.yolo_csv is not None and args.yolo_csv.exists():
        yolo_csv_to_feature_store(args.yolo_csv, feature_dir)

    # 2. FeatureStore 读完整序列。
    feature_store = FeatureStore(feature_dir)
    items = feature_store.load_all()
    if not items:
        raise RuntimeError(f"没有可训练的 FeatureStore 数据：{feature_dir}")

    # 3. 数据集划分。
    split = split_by_task(items, val_ratio=0.2, test_ratio=0.0, seed=SEED)
    in_dim = int(items[0]["features"].shape[1])
    results = {}
    ranking = []

    for model_name in args.models:
        # 4. 模型训练。
        segmenter = OfflineSegmenter(model_name, in_dim, len(CLASSES), device)
        train_info = segmenter.fit(split.train, epochs=args.epochs)

        # 5. 验证。
        val_reports = []
        val_scores = []
        for item in split.val:
            pred, probs = segmenter.predict(item)
            gt_facts = labels_to_segment_facts(item["labels"], item["fps"], item["task_id"], item["step_id"], "label_studio_ground_truth")
            pred_facts = labels_to_segment_facts(pred, item["fps"], item["task_id"], item["step_id"], f"{model_name}_validation", probs)
            score = segment_f1(gt_facts, pred_facts, threshold=0.25)
            val_scores.append(score)
            val_reports.append(
                {
                    "task_id": item["task_id"],
                    "frame_metrics": frame_metrics(item["labels"], pred),
                    "segment_f1@0.25": score,
                    "gt_segments": gt_facts,
                    "pred_segments": pred_facts,
                }
            )

        # 6. 全量输出 SegmentFact 和 FactLedger。
        all_facts = []
        pred_dir = args.out_dir / "predictions"
        for item in items:
            pred, probs = segmenter.predict(item)
            facts = labels_to_segment_facts(pred, item["fps"], item["task_id"], item["step_id"], f"{model_name}_offline_segmenter", probs)
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
    report = {
        "status": "completed",
        "device": str(device),
        "pipeline": "data_transfer -> dataset split -> OfflineSegmenter train/validate -> SegmentFact -> FactLedger",
        "feature_store": str(feature_dir),
        "task_ids": [item["task_id"] for item in items],
        "feature_dim": in_dim,
        "classes": CLASSES,
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
            "当前实现是能跑通的 baseline 仓本，不是完整论文级 MS-TCN/ASFormer 复现。",
            "Label Studio 转换会同时使用 bbox 特征和 timeline 标签；YOLO CSV 转换当前只提供检测特征，标签需要另行合并。",
            "SegmentFact 是动作片段事实，FactLedger 是用于离线复算幂等 upsert 的账本行。",
        ],
    }
    report_path = args.out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "selected_model": selected, "ranking": ranking, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

