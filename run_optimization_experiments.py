"""离线特征/训练策略对比实验。

本脚本只用于 offline-model 仓内快速探索，不接入后端。它复用现有 ActionMixed
转换与三种时序模型，在同一验证集上比较：

1. 当前 v2 特征；
2. 利用离线场景可看未来帧的中心窗口统计特征；
3. 基于业务关系构造的动作先验分数；
4. 中心窗口统计 + 业务先验组合；
5. 全序列训练 vs 滑窗训练。

输出:
    output_actionmixed_optim_experiments/experiment_report.json
    output_actionmixed_optim_experiments/experiment_report.md
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_transfer import CLASSES, FeatureStore
from dataset import actionmixed_to_feature_store, split_by_declared_split
from run_pipeline import OfflineSegmenter
from segmenter import MODEL_REGISTRY


SEED = 20260715
TARGET_CLASS_IDS = [idx for idx, name in enumerate(CLASSES) if name != "idle"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # run_pipeline 导入时可能已经设置过 interop threads。
        pass


def clone_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["features"] = np.asarray(item["features"], dtype=np.float32).copy()
    out["labels"] = np.asarray(item["labels"], dtype=np.int64).copy()
    out["feature_names"] = list(item["feature_names"])
    return out


def centered_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """中心窗口均值；离线模型允许使用当前帧之后的观测。"""
    if radius <= 0:
        return values.astype(np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        out[idx] = values[lo:hi].mean(axis=0)
    return out


def add_centered_window_stats(item: dict[str, Any], windows: tuple[int, ...] = (5, 15)) -> dict[str, Any]:
    """为关键列添加中心窗口统计。

    只对动作相关的列做窗口统计，避免把全部 113 维暴力扩到过大。
    """
    out = clone_item(item)
    names = out["feature_names"]
    feature = out["features"]
    selected = [
        idx
        for idx, name in enumerate(names)
        if name.endswith(("_present", "_conf", "_speed", "_dist", "_delta", "_missing_age", "_imputed"))
    ]
    if not selected:
        return out

    extra_blocks: list[np.ndarray] = []
    extra_names: list[str] = []
    base = feature[:, selected]
    for window in windows:
        radius = max(1, window // 2)
        mean = centered_mean(base, radius)
        extra_blocks.append(mean)
        extra_names.extend([f"{names[idx]}_center_mean_w{window}" for idx in selected])

    out["features"] = np.concatenate([feature, *extra_blocks], axis=1).astype(np.float32)
    out["feature_names"] = names + extra_names
    out["feature_version"] = f"{out.get('feature_version', 'unknown')}+center_window"
    return out


def col(features: np.ndarray, name_to_idx: dict[str, int], name: str) -> np.ndarray:
    idx = name_to_idx.get(name)
    if idx is None:
        return np.zeros(features.shape[0], dtype=np.float32)
    return features[:, idx].astype(np.float32)


def near_score(dist: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)


def add_business_priors(item: dict[str, Any]) -> dict[str, Any]:
    """增加动作相关弱先验分数，作为模型输入而非规则输出。"""
    out = clone_item(item)
    x = out["features"]
    n = {name: idx for idx, name in enumerate(out["feature_names"])}

    hand = np.maximum(col(x, n, "hand_top1_present"), col(x, n, "hand_top2_present"))
    short_brush = col(x, n, "short_brush_present")
    syringe = col(x, n, "syringe_present")
    air_gun = col(x, n, "air_gun_present")
    brush_tip = col(x, n, "brush_tip_out_present")
    long_brush = col(x, n, "long_brush_present")

    short_near = near_score(col(x, n, "short_brush_to_scope_control_body_dist"))
    syringe_near = near_score(col(x, n, "syringe_to_scope_distal_end_dist"))
    air_near = near_score(col(x, n, "air_gun_to_scope_distal_end_dist"))
    tip_near = near_score(col(x, n, "brush_tip_out_to_scope_distal_end_dist"))
    long_near = near_score(col(x, n, "long_brush_to_scope_mid_section_dist"))

    short_motion = np.maximum(
        col(x, n, "short_brush_speed"),
        np.abs(col(x, n, "short_brush_to_scope_control_body_delta")),
    )
    syringe_stable = syringe * syringe_near * (1.0 - np.clip(col(x, n, "syringe_speed"), 0.0, 1.0))
    air_stable = air_gun * air_near * (1.0 - np.clip(col(x, n, "air_gun_speed"), 0.0, 1.0))
    long_signal = np.maximum.reduce([long_brush, brush_tip, col(x, n, "brush_tip_out_imputed")])
    long_delta = col(x, n, "brush_tip_out_to_scope_distal_end_delta")
    hand_to_long = near_score(col(x, n, "hand_to_long_brush_dist"))

    priors = np.stack(
        [
            hand * short_brush * short_near,
            hand * short_brush * short_motion,
            hand * syringe_stable,
            hand * air_stable,
            hand * long_signal * np.maximum(tip_near, long_near),
            hand * long_signal * np.clip(-long_delta, 0.0, 1.0),
            hand * long_signal * np.clip(long_delta, 0.0, 1.0),
            hand_to_long * long_signal,
        ],
        axis=1,
    ).astype(np.float32)

    prior_names = [
        "prior_short_clean_near",
        "prior_short_clean_motion",
        "prior_flush_stable",
        "prior_air_stable",
        "prior_long_signal_near_scope",
        "prior_long_towards_distal",
        "prior_long_away_distal",
        "prior_hand_long_contact",
    ]
    out["features"] = np.concatenate([x, priors], axis=1).astype(np.float32)
    out["feature_names"] = out["feature_names"] + prior_names
    out["feature_version"] = f"{out.get('feature_version', 'unknown')}+business_priors"
    return out


def apply_feature_method(items: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    out = [clone_item(item) for item in items]
    if method == "v2":
        return out
    if method == "window_stats":
        return [add_centered_window_stats(item) for item in out]
    if method == "business_priors":
        return [add_business_priors(item) for item in out]
    if method == "window_stats+business_priors":
        return [add_business_priors(add_centered_window_stats(item)) for item in out]
    raise ValueError(f"unknown feature method: {method}")


def make_windows(items: list[dict[str, Any]], window: int = 128, stride: int = 32) -> list[dict[str, Any]]:
    """训练时切滑窗，验证/推理仍使用完整序列。"""
    windows: list[dict[str, Any]] = []
    for item in items:
        length = int(item["features"].shape[0])
        if length <= window:
            windows.append(clone_item(item))
            continue
        starts = list(range(0, max(1, length - window + 1), stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        for seq_idx, start in enumerate(starts):
            end = start + window
            sub = clone_item(item)
            sub["features"] = item["features"][start:end].astype(np.float32)
            sub["labels"] = item["labels"][start:end].astype(np.int64)
            sub["frames"] = int(end - start)
            sub["duration_s"] = float((end - start) / item["fps"])
            sub["window_start"] = int(start)
            sub["window_end"] = int(end)
            sub["window_index"] = int(seq_idx)
            windows.append(sub)
    return windows


def labels_to_spans(labels: np.ndarray, class_id: int | None = None) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    if len(labels) == 0:
        return spans
    start = 0
    cur = int(labels[0])
    for idx in range(1, len(labels) + 1):
        nxt = int(labels[idx]) if idx < len(labels) else None
        if nxt != cur:
            if cur != 0 and (class_id is None or cur == class_id):
                spans.append((start, idx - 1, cur))
            start = idx
            cur = nxt if nxt is not None else 0
    return spans


def segment_f1_for_class(gt: np.ndarray, pred: np.ndarray, class_id: int, threshold: float) -> tuple[float, float, float]:
    gt_spans = labels_to_spans(gt, class_id)
    pred_spans = labels_to_spans(pred, class_id)
    used: set[int] = set()
    matched = 0
    for ps, pe, _ in pred_spans:
        best_idx = None
        best_iou = 0.0
        for idx, (gs, ge, _) in enumerate(gt_spans):
            if idx in used:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs) + 1)
            union = max(pe, ge) - min(ps, gs) + 1
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used.add(best_idx)
            matched += 1
    precision = matched / len(pred_spans) if pred_spans else 0.0
    recall = matched / len(gt_spans) if gt_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def frame_classification_report(gt_all: np.ndarray, pred_all: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "accuracy": float((gt_all == pred_all).mean()) if len(gt_all) else 0.0,
        "per_class": {},
    }
    macro_p = []
    macro_r = []
    macro_f1 = []
    for idx, name in enumerate(CLASSES):
        tp = int(((gt_all == idx) & (pred_all == idx)).sum())
        fp = int(((gt_all != idx) & (pred_all == idx)).sum())
        fn = int(((gt_all == idx) & (pred_all != idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["per_class"][name] = {
            "support": int((gt_all == idx).sum()),
            "predicted": int((pred_all == idx).sum()),
            "precision": precision,
            "recall": recall,
            "frame_f1": f1,
        }
        if idx != 0:
            macro_p.append(precision)
            macro_r.append(recall)
            macro_f1.append(f1)
    result["target_macro_precision"] = float(np.mean(macro_p)) if macro_p else 0.0
    result["target_macro_recall"] = float(np.mean(macro_r)) if macro_r else 0.0
    result["target_macro_frame_f1"] = float(np.mean(macro_f1)) if macro_f1 else 0.0
    return result


def evaluate_predictions(records: list[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
    gt_all = np.concatenate([item["labels"] for item, _ in records])
    pred_all = np.concatenate([pred for _, pred in records])
    frame_report = frame_classification_report(gt_all, pred_all)

    segment_report: dict[str, Any] = {"per_class": {}}
    for threshold in (0.25, 0.5):
        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        for class_id in TARGET_CLASS_IDS:
            p_values = []
            r_values = []
            f_values = []
            for item, pred in records:
                p, r, f = segment_f1_for_class(item["labels"], pred, class_id, threshold)
                p_values.append(p)
                r_values.append(r)
                f_values.append(f)
            name = CLASSES[class_id]
            cls_metrics = segment_report["per_class"].setdefault(name, {})
            cls_metrics[f"segment_precision@{threshold}"] = float(np.mean(p_values))
            cls_metrics[f"segment_recall@{threshold}"] = float(np.mean(r_values))
            cls_metrics[f"segment_f1@{threshold}"] = float(np.mean(f_values))
            per_class_precision.append(cls_metrics[f"segment_precision@{threshold}"])
            per_class_recall.append(cls_metrics[f"segment_recall@{threshold}"])
            per_class_f1.append(cls_metrics[f"segment_f1@{threshold}"])
        segment_report[f"target_macro_segment_precision@{threshold}"] = float(np.mean(per_class_precision))
        segment_report[f"target_macro_segment_recall@{threshold}"] = float(np.mean(per_class_recall))
        segment_report[f"target_macro_segment_f1@{threshold}"] = float(np.mean(per_class_f1))

    return {"frame": frame_report, "segment": segment_report}


def run_one_experiment(
    base_items: list[dict[str, Any]],
    feature_method: str,
    train_mode: str,
    model_name: str,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed()
    items = apply_feature_method(base_items, feature_method)
    split = split_by_declared_split(items, seed=SEED)
    train_items = split.train if train_mode == "full_sequence" else make_windows(split.train)
    in_dim = int(items[0]["features"].shape[1])

    segmenter = OfflineSegmenter(model_name, in_dim, len(CLASSES), device)
    train_info = segmenter.fit(train_items, epochs=epochs)

    records = []
    for item in split.val:
        pred, _ = segmenter.predict(item)
        records.append((item, pred))
    metrics = evaluate_predictions(records)
    return {
        "feature_method": feature_method,
        "train_mode": train_mode,
        "model": model_name,
        "epochs": epochs,
        "feature_dim": in_dim,
        "train_sequences": len(split.train),
        "train_samples": len(train_items),
        "val_sequences": len(split.val),
        "train": train_info,
        "metrics": metrics,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = sorted(
        report["experiments"],
        key=lambda r: (
            r["metrics"]["segment"]["target_macro_segment_f1@0.25"],
            r["metrics"]["segment"]["target_macro_segment_f1@0.5"],
            r["metrics"]["frame"]["target_macro_frame_f1"],
        ),
        reverse=True,
    )
    lines = [
        "# 离线模型特征与训练策略实验报告",
        "",
        f"- 数据源：`{report['dataset_root']}`",
        f"- epoch：`{report['epochs']}`",
        f"- 实验数量：`{len(rows)}`",
        "",
        "## 总体排名",
        "",
        "| 排名 | 特征方法 | 训练方式 | 模型 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        frame = row["metrics"]["frame"]
        seg = row["metrics"]["segment"]
        lines.append(
            "| "
            f"{rank} | `{row['feature_method']}` | `{row['train_mode']}` | `{row['model']}` | {row['feature_dim']} | "
            f"{fmt(frame['accuracy'])} | {fmt(frame['target_macro_precision'])} | "
            f"{fmt(frame['target_macro_recall'])} | {fmt(frame['target_macro_frame_f1'])} | "
            f"{fmt(seg['target_macro_segment_f1@0.25'])} | {fmt(seg['target_macro_segment_f1@0.5'])} |"
        )

    best = rows[0]
    lines += [
        "",
        "## 最优实验逐类结果",
        "",
        f"最优配置：`{best['feature_method']}` + `{best['train_mode']}` + `{best['model']}`。",
        "",
        "| 动作类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-P@0.25 | Seg-R@0.25 | Seg-F1@0.25 | Seg-F1@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        frame_cls = best["metrics"]["frame"]["per_class"][name]
        seg_cls = best["metrics"]["segment"]["per_class"].get(name, {})
        lines.append(
            "| "
            f"`{name}` | {frame_cls['support']} | {frame_cls['predicted']} | "
            f"{fmt(frame_cls['precision'])} | {fmt(frame_cls['recall'])} | {fmt(frame_cls['frame_f1'])} | "
            f"{fmt(seg_cls.get('segment_precision@0.25', 0.0))} | "
            f"{fmt(seg_cls.get('segment_recall@0.25', 0.0))} | "
            f"{fmt(seg_cls.get('segment_f1@0.25', 0.0))} | "
            f"{fmt(seg_cls.get('segment_f1@0.5', 0.0))} |"
        )

    lines += [
        "",
        "## 方法说明",
        "",
        "- `v2`：hand top-2，非 hand top-1，遮挡补全，关系 delta。",
        "- `window_stats`：在 v2 上增加中心窗口统计，离线模型可以利用未来帧。",
        "- `business_priors`：在 v2 上增加短刷/推流/注气/长刷插拔相关弱先验分数。",
        "- `window_stats+business_priors`：组合以上两类增强特征。",
        "- `sliding_window`：训练时切 128 帧窗口、stride=32；验证仍使用完整序列。",
        "",
        "## 建议",
        "",
    ]
    lines.extend([f"- {item}" for item in report["recommendations"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ActionMixed 特征/训练策略对比实验")
    parser.add_argument("--dataset-root", type=Path, default=Path("input/modelscope/lhh010__cleansight-ActionMixed"))
    parser.add_argument("--out-dir", type=Path, default=Path("output_actionmixed_optim_experiments"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--feature-methods",
        nargs="+",
        default=["v2", "window_stats", "business_priors", "window_stats+business_priors"],
    )
    parser.add_argument("--train-modes", nargs="+", default=["full_sequence", "sliding_window"])
    args = parser.parse_args()

    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.out_dir / "feature_store_v2"
    for path in feature_dir.glob("*.npz") if feature_dir.exists() else []:
        path.unlink()
    actionmixed_to_feature_store(args.dataset_root, feature_dir)
    base_items = FeatureStore(feature_dir).load_all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = []
    for feature_method in args.feature_methods:
        for train_mode in args.train_modes:
            for model_name in args.models:
                print(f"running feature={feature_method} train={train_mode} model={model_name}", flush=True)
                experiments.append(
                    run_one_experiment(base_items, feature_method, train_mode, model_name, args.epochs, device)
                )

    ranked = sorted(
        experiments,
        key=lambda r: (
            r["metrics"]["segment"]["target_macro_segment_f1@0.25"],
            r["metrics"]["segment"]["target_macro_segment_f1@0.5"],
            r["metrics"]["frame"]["target_macro_frame_f1"],
        ),
        reverse=True,
    )
    recommendations = [
        "优先看 target macro Segment F1@0.25/@0.5，而不是只看 frame ACC；动作边界和片段命中才是离线分割目标。",
        "若 window_stats 组合优于 v2，说明离线模型确实受益于未来帧，应把中心窗口统计作为后续主线。",
        "若 sliding_window 明显优于 full_sequence，说明当前 21 条序列样本太少，应固定使用滑窗训练、全序列验证。",
        "若某类 recall 长期为 0，优先检查该动作段关键检测目标召回，而不是继续加深模型。",
        "当前实验仍是小数据快速验证，正式结论建议把 epoch 提高到 20-100 并重复 3 个随机种子。",
    ]
    report = {
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "epochs": args.epochs,
        "models": args.models,
        "feature_methods": args.feature_methods,
        "train_modes": args.train_modes,
        "class_names": CLASSES,
        "experiments": experiments,
        "best": ranked[0],
        "recommendations": recommendations,
    }
    json_path = args.out_dir / "experiment_report.json"
    md_path = args.out_dir / "experiment_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "best": ranked[0], "report": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
