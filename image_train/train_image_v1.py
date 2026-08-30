"""ActionMixed image-feature v1 training experiment.

This file intentionally lives under image_train/ so the first RGB experiment is
isolated from the current baseline code.  It reuses the existing dataset,
feature recipes, temporal models, and metrics, then adds per-frame RGB ROI
features extracted from ActionMixed JPEG frames.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_transfer import CLASSES, FeatureStore, OBJECT_MAP, OBJECTS
from dataset import (
    ACTIONMIXED_DETECTION_CLASSES,
    actionmixed_to_feature_store,
    split_by_declared_split,
    _read_simple_yaml_names,
)
from run_optimization_experiments import (
    SEED,
    add_business_priors,
    add_centered_window_stats,
    evaluate_predictions,
    make_windows,
)
from run_pipeline import OfflineSegmenter


BEST_RECIPES: dict[str, dict[str, str]] = {
    "ms_tcn": {
        "feature_method": "v2+rgb_roi",
        "train_mode": "full_sequence",
        "base_feature_method": "v2",
        "reason": "沿用当前 ms_tcn 最优的 v2 + full_sequence，在末尾拼接 RGB ROI 外观特征。",
    },
    "asformer": {
        "feature_method": "business_priors+rgb_roi",
        "train_mode": "full_sequence",
        "base_feature_method": "business_priors",
        "reason": "沿用当前 asformer 最优的 business_priors + full_sequence，在末尾拼接 RGB ROI 外观特征。",
    },
    "bigru": {
        "feature_method": "window_stats+business_priors+rgb_roi",
        "train_mode": "sliding_window",
        "base_feature_method": "window_stats+business_priors",
        "reason": "沿用当前 bigru 最优的 window_stats+business_priors + sliding_window，在末尾拼接 RGB ROI 外观特征。",
    },
}

FRAME_RE = re.compile(r"^(?P<video>.+\.mp4)-(?P<frame>\d+)\.txt$")
RGB_BINS = (0.0, 0.25, 0.5, 0.75, 1.000001)
SLOTS = ["hand_top1", "hand_top2"] + [obj for obj in OBJECTS if obj != "hand"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def clone_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["features"] = np.asarray(item["features"], dtype=np.float32).copy()
    out["labels"] = np.asarray(item["labels"], dtype=np.int64).copy()
    out["feature_names"] = list(item["feature_names"])
    return out


def clear_npz_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*.npz"):
        file.unlink()


def parse_frame_name(path: Path) -> tuple[str, int] | None:
    match = FRAME_RE.match(path.name)
    if not match:
        return None
    return match.group("video"), int(match.group("frame"))


def grouped_frame_files(dataset_root: Path, split: str) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted((dataset_root / "frames" / split).glob("*.txt")):
        parsed = parse_frame_name(path)
        if parsed is None:
            continue
        video_id, frame_no = parsed
        groups.setdefault(video_id, []).append((frame_no, path))
    for video_id in groups:
        groups[video_id].sort(key=lambda row: row[0])
    return groups


def detection_score(row: tuple[str, float, float, float, float, float]) -> float:
    _, _, _, width, height, conf = row
    return conf * math.sqrt(max(width * height, 1e-8))


def parse_detections(path: Path, detection_names: dict[int, str]) -> dict[str, list[tuple[str, float, float, float, float, float]]]:
    detections: dict[str, list[tuple[str, float, float, float, float, float]]] = {}
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
        except ValueError:
            continue
        obj = OBJECT_MAP.get(detection_names.get(class_id, ""))
        if obj is None:
            continue
        detections.setdefault(obj, []).append((obj, cx, cy, width, height, max(0.0, min(conf, 1.0))))
    for obj in detections:
        detections[obj].sort(key=detection_score, reverse=True)
    return detections


def crop_box_array(
    image_array: np.ndarray,
    row: tuple[str, float, float, float, float, float],
    padding: float,
) -> np.ndarray | None:
    _, cx, cy, width, height, _ = row
    img_h, img_w = image_array.shape[:2]
    pad_w = width * padding
    pad_h = height * padding
    x1 = int(round((cx - width / 2.0 - pad_w) * img_w))
    y1 = int(round((cy - height / 2.0 - pad_h) * img_h))
    x2 = int(round((cx + width / 2.0 + pad_w) * img_w))
    y2 = int(round((cy + height / 2.0 + pad_h) * img_h))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_array[y1:y2, x1:x2]
    return crop if crop.size else None


def rgb_feature(crop: np.ndarray | None) -> np.ndarray:
    """Return 19 RGB-only features: valid + mean/std + 4-bin hist per channel."""
    if crop is None:
        return np.zeros(19, dtype=np.float32)
    flat = crop.reshape(-1, 3).astype(np.float32) / 255.0
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=RGB_BINS)
        hist_parts.append(hist.astype(np.float32) / max(1, flat.shape[0]))
    return np.concatenate([[1.0], mean, std, *hist_parts]).astype(np.float32)


def rgb_feature_names() -> list[str]:
    names: list[str] = []
    suffixes = [
        "valid",
        "r_mean",
        "g_mean",
        "b_mean",
        "r_std",
        "g_std",
        "b_std",
        "r_hist_0_25",
        "r_hist_25_50",
        "r_hist_50_75",
        "r_hist_75_100",
        "g_hist_0_25",
        "g_hist_25_50",
        "g_hist_50_75",
        "g_hist_75_100",
        "b_hist_0_25",
        "b_hist_25_50",
        "b_hist_50_75",
        "b_hist_75_100",
    ]
    for slot in SLOTS:
        names.extend([f"rgb_roi_{slot}_{suffix}" for suffix in suffixes])
    return names


def frame_rgb_features(
    image_path: Path,
    yolo_path: Path,
    detection_names: dict[int, str],
    padding: float,
) -> tuple[np.ndarray, int]:
    detections = parse_detections(yolo_path, detection_names)
    if not image_path.exists():
        return np.zeros(len(SLOTS) * 19, dtype=np.float32), 1

    missing_image = 0
    try:
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
    except OSError:
        return np.zeros(len(SLOTS) * 19, dtype=np.float32), 1

    features: list[np.ndarray] = []
    hand_rows = detections.get("hand", [])
    for slot_idx in range(2):
        row = hand_rows[slot_idx] if slot_idx < len(hand_rows) else None
        features.append(rgb_feature(crop_box_array(image_array, row, padding) if row else None))

    for obj in OBJECTS:
        if obj == "hand":
            continue
        rows = detections.get(obj, [])
        row = rows[0] if rows else None
        features.append(rgb_feature(crop_box_array(image_array, row, padding) if row else None))
    return np.concatenate(features).astype(np.float32), missing_image


def make_rgb_cache(
    dataset_root: Path,
    cache_dir: Path,
    padding: float,
) -> dict[tuple[str, str], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    detection_names = _read_simple_yaml_names(dataset_root / "frames" / "data.yaml", ACTIONMIXED_DETECTION_CLASSES)
    cache_index: dict[tuple[str, str], Path] = {}

    for split in ("train", "val", "test"):
        for video_id, frames in grouped_frame_files(dataset_root, split).items():
            cache_path = cache_dir / f"{split}__{safe_name(video_id)}.npz"
            if cache_path.exists():
                cache_index[(split, video_id)] = cache_path
                continue
            rows = []
            frame_numbers = []
            missing_images = 0
            for frame_no, yolo_path in frames:
                image_path = dataset_root / "images" / split / f"{video_id}-{frame_no:06d}.jpg"
                feat, missing = frame_rgb_features(image_path, yolo_path, detection_names, padding)
                rows.append(feat)
                frame_numbers.append(frame_no)
                missing_images += missing
            matrix = np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, len(SLOTS) * 19), dtype=np.float32)
            np.savez_compressed(
                cache_path,
                split=np.array([split]),
                video_ref=np.array([video_id]),
                frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
                rgb_features=matrix,
                feature_names=np.asarray(rgb_feature_names()),
                missing_images=np.array([missing_images], dtype=np.int64),
                padding=np.array([padding], dtype=np.float32),
            )
            cache_index[(split, video_id)] = cache_path
    return cache_index


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_rgb_features(cache_index: dict[tuple[str, str], Path], item: dict[str, Any]) -> np.ndarray:
    key = (str(item.get("split", "")), str(item.get("video_ref", "")))
    path = cache_index.get(key)
    if path is None:
        return np.zeros((int(item["features"].shape[0]), len(SLOTS) * 19), dtype=np.float32)
    data = np.load(path, allow_pickle=True)
    rgb = data["rgb_features"].astype(np.float32)
    expected = int(item["features"].shape[0])
    if rgb.shape[0] == expected:
        return rgb
    out = np.zeros((expected, len(SLOTS) * 19), dtype=np.float32)
    n = min(expected, rgb.shape[0])
    out[:n] = rgb[:n]
    return out


def add_rgb_to_item(item: dict[str, Any], rgb: np.ndarray) -> dict[str, Any]:
    out = clone_item(item)
    out["features"] = np.concatenate([out["features"], rgb], axis=1).astype(np.float32)
    out["feature_names"] = list(out["feature_names"]) + rgb_feature_names()
    out["feature_version"] = f"{out.get('feature_version', 'unknown')}+rgb_roi_stats_v1"
    out["sources"] = list(out.get("sources", ["bbox", "geometry", "motion"])) + ["rgb_roi_stats_v1"]
    return out


def apply_base_feature_method(item: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "v2":
        return clone_item(item)
    if method == "business_priors":
        return add_business_priors(item)
    if method == "window_stats+business_priors":
        return add_business_priors(add_centered_window_stats(item))
    raise ValueError(f"unknown base feature method: {method}")


def make_model_items(base_items: list[dict[str, Any]], cache_index: dict[tuple[str, str], Path], model_name: str) -> list[dict[str, Any]]:
    recipe = BEST_RECIPES[model_name]
    items = []
    for item in base_items:
        with_base = apply_base_feature_method(item, recipe["base_feature_method"])
        items.append(add_rgb_to_item(with_base, load_rgb_features(cache_index, item)))
    return items


def train_one(
    model_name: str,
    base_items: list[dict[str, Any]],
    cache_index: dict[tuple[str, str], Path],
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed()
    recipe = BEST_RECIPES[model_name]
    items = make_model_items(base_items, cache_index, model_name)
    split = split_by_declared_split(items, seed=SEED)
    train_items = split.train if recipe["train_mode"] == "full_sequence" else make_windows(split.train)

    feature_dim = int(items[0]["features"].shape[1])
    segmenter = OfflineSegmenter(model_name, feature_dim, len(CLASSES), device)
    train_info = segmenter.fit(train_items, epochs=epochs)

    eval_sets = {
        "val": split.val,
        "test": split.test,
    }
    metrics: dict[str, Any] = {}
    records_by_split: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_name, eval_items in eval_sets.items():
        records = []
        for item in eval_items:
            pred, probs = segmenter.predict(item)
            records.append((item, pred))
            np.savez_compressed(
                pred_dir / f"{model_name}_{split_name}_task_{item['task_id']}_soft_labels.npz",
                task_id=np.array([item["task_id"]]),
                split=np.array([split_name]),
                video_ref=np.array([item.get("video_ref", "")]),
                predicted_labels=pred.astype(np.int64),
                probabilities=probs.astype(np.float32),
                class_names=np.array(CLASSES),
            )
        records_by_split[split_name] = records
        metrics[split_name] = evaluate_predictions(records) if records else {}

    model_path = out_dir / "models" / f"image_v1_{model_name}_offline_segmenter.pt"
    segmenter.save(model_path, items[0]["feature_names"], items[0].get("feature_version", ""))
    enrich_checkpoint(
        model_path,
        {
            "image_train_version": "v1",
            "best_recipe": copy.deepcopy(recipe),
            "epochs": epochs,
            "feature_dim": feature_dim,
            "rgb_slots": SLOTS,
            "rgb_feature_per_slot": 19,
            "metrics": metrics,
            "train_sequences": len(split.train),
            "train_samples": len(train_items),
            "val_sequences": len(split.val),
            "test_sequences": len(split.test),
        },
    )

    return {
        "model": model_name,
        "model_path": str(model_path),
        "feature_method": recipe["feature_method"],
        "base_feature_method": recipe["base_feature_method"],
        "train_mode": recipe["train_mode"],
        "reason": recipe["reason"],
        "feature_version": items[0].get("feature_version", ""),
        "feature_dim": feature_dim,
        "rgb_feature_dim": len(SLOTS) * 19,
        "epochs": epochs,
        "train_sequences": len(split.train),
        "train_samples": len(train_items),
        "val_sequences": len(split.val),
        "test_sequences": len(split.test),
        "last_loss": train_info["history"][-1]["loss"] if train_info.get("history") else None,
        "train": train_info,
        "metrics": metrics,
    }


def enrich_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_row(row: dict[str, Any], split_name: str) -> str:
    metric = row["metrics"].get(split_name, {})
    frame = metric.get("frame", {})
    segment = metric.get("segment", {})
    return (
        "| "
        f"`{row['model']}` | `{row['feature_method']}` | `{row['train_mode']}` | {row['feature_dim']} | "
        f"{fmt(frame.get('accuracy'))} | {fmt(frame.get('target_macro_precision'))} | "
        f"{fmt(frame.get('target_macro_recall'))} | {fmt(frame.get('target_macro_frame_f1'))} | "
        f"{fmt(segment.get('target_macro_segment_f1@0.25'))} | "
        f"{fmt(segment.get('target_macro_segment_f1@0.5'))} | `{row['model_path']}` |"
    )


def add_per_class_table(lines: list[str], row: dict[str, Any], split_name: str) -> None:
    metric = row["metrics"].get(split_name, {})
    if not metric:
        lines.append(f"{split_name} split 没有样本。")
        return
    lines += [
        f"### {split_name}",
        "",
        "| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for class_id, name in enumerate(CLASSES):
        frame_cls = metric["frame"]["per_class"][name]
        support = int(frame_cls["support"])
        predicted = int(frame_cls["predicted"])
        precision = float(frame_cls["precision"])
        recall = float(frame_cls["recall"])
        tp = int(round(recall * support)) if support else 0
        fp = int(round(tp / precision - tp)) if precision > 0 else max(0, predicted - tp)
        fn = max(0, support - tp)
        seg_cls = metric["segment"]["per_class"].get(name, {})
        lines.append(
            "| "
            f"`{name}` | {support} | {predicted} | {tp} | {fp} | {fn} | "
            f"{fmt(precision)} | {fmt(recall)} | {fmt(frame_cls['frame_f1'])} | "
            f"{fmt(seg_cls.get('segment_f1@0.25', 0.0))} | {fmt(seg_cls.get('segment_f1@0.5', 0.0))} |"
        )


def dataset_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    for item in items:
        split = str(item.get("split", "unknown"))
        entry = by_split.setdefault(split, {"sequences": 0, "frames": 0})
        entry["sequences"] += 1
        entry["frames"] += int(item["features"].shape[0])
    return by_split


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["results"]
    lines = [
        "# image_train_v1：ActionMixed RGB 图像特征首版改进训练报告",
        "",
        "## 1. 本次改进目标",
        "",
        "本次在不改动三种时序模型主体结构的前提下，新增 `image_train` 实验目录，把 ActionMixed 已下载的 JPEG 帧图像 RGB 信息加入训练。实现方式是复用当前三种模型各自最好的检测框特征 recipe，再拼接轻量 RGB ROI 外观特征，验证第一版图像信息是否能给动作分割带来收益。",
        "",
        "## 2. 改了什么",
        "",
        f"- 新增训练脚本：`{report['script']}`",
        f"- 数据集：`{report['dataset_root']}`",
        f"- 输出目录：`{report['out_dir']}`",
        f"- 训练轮数：`{report['epochs']}`",
        f"- 设备：`{report['device']}`",
        f"- RGB ROI 槽位：`{len(SLOTS)}` 个，分别为 `{', '.join(SLOTS)}`。",
        f"- 每个 ROI 槽位 RGB 特征：`19` 维，总 RGB 维度：`{len(SLOTS) * 19}`。",
        "",
        "### 2.1 RGB ROI 特征如何提取",
        "",
        "1. 对每个视频片段读取 `frames/{split}/{video}.mp4-{frame_id}.txt` 中的 YOLO bbox，并找到同名 `images/{split}/{video}.mp4-{frame_id}.jpg`。",
        "2. 每帧按对象选择 ROI：`hand` 保留 top-2 两个独立槽位，其它对象保留 top-1。排序分数为 `confidence * sqrt(width * height)`。",
        "3. 对 bbox 加 `15%` padding 后在原图数组上直接切片 ROI，不做 resize，直接统计 ROI 内 RGB 分布。",
        "4. 每个 ROI 只提取 RGB 信息：`valid`、RGB 三通道均值、RGB 三通道标准差、RGB 三通道各 4 个归一化直方图 bin。",
        "5. 当前帧某对象缺失或图片缺失时，该 ROI 槽位全 0；是否真实检测到、missing_age、imputed 仍由原 v2 检测框特征表达。",
        "",
        "### 2.2 三个模型现在使用的特征",
        "",
        "| 模型 | 原最佳特征 | 新增图像特征 | 当前总特征 | 训练方式 |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | `{row['base_feature_method']}` | `rgb_roi_stats_v1` ({row['rgb_feature_dim']}维) | {row['feature_dim']} | `{row['train_mode']}` |"
        )

    lines += [
        "",
        "特征分类说明：",
        "",
        "- v2 检测框结构特征：对象 candidate_count、present、confidence、bbox 中心、面积、speed、missing_age、imputed、对象间 distance/delta、时间位置编码。",
        "- window_stats：对动作相关列追加中心窗口统计，离线模型可以使用当前帧前后的上下文。",
        "- business_priors：基于业务对象关系追加弱先验分数，例如短刷靠近控制部、针筒/气枪靠近远端口、长刷刷头相对远端口的运动方向。",
        "- rgb_roi_stats_v1：本次新增的图片 RGB 外观特征，描述手、刷子、针筒、气枪、内镜部位等 ROI 的局部颜色分布。",
        "",
        "## 3. 数据概况",
        "",
        "| split | 序列数 | 帧数 |",
        "|---|---:|---:|",
    ]
    for split, stat in sorted(report["dataset_stats"].items()):
        lines.append(f"| `{split}` | {stat['sequences']} | {stat['frames']} |")

    for split_name in ("val", "test"):
        lines += [
            "",
            f"## 4. {split_name} 整体结果",
            "",
            "| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(metric_row(row, split_name))

    lines += ["", "## 5. 每个模型逐动作识别情况（以帧数为单位）", ""]
    for row in rows:
        lines += [
            f"## {row['model']}",
            "",
            f"- 使用特征版本：`{row['feature_version']}`",
            f"- 训练样本：`{row['train_samples']}`，train 序列：`{row['train_sequences']}`，val 序列：`{row['val_sequences']}`，test 序列：`{row['test_sequences']}`",
            f"- 最后一轮 loss：`{fmt(row['last_loss'])}`",
            "",
        ]
        add_per_class_table(lines, row, "val")
        lines.append("")
        add_per_class_table(lines, row, "test")
        lines.append("")

    lines += [
        "## 6. 结果分析",
        "",
        *[f"- {item}" for item in report["analysis"]],
        "",
        "## 7. 后续修改建议",
        "",
        *[f"- {item}" for item in report["next_steps"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_analysis(results: list[dict[str, Any]]) -> list[str]:
    analysis = []
    for split_name in ("val", "test"):
        ranked = sorted(
            results,
            key=lambda row: (
                row["metrics"].get(split_name, {}).get("segment", {}).get("target_macro_segment_f1@0.25", -1.0),
                row["metrics"].get(split_name, {}).get("frame", {}).get("target_macro_frame_f1", -1.0),
            ),
            reverse=True,
        )
        if ranked:
            best = ranked[0]
            seg = best["metrics"].get(split_name, {}).get("segment", {})
            frame = best["metrics"].get(split_name, {}).get("frame", {})
            analysis.append(
                f"{split_name} 上按 F1@0.25 排名最高的是 `{best['model']}`，F1@0.25={fmt(seg.get('target_macro_segment_f1@0.25'))}，F1@0.5={fmt(seg.get('target_macro_segment_f1@0.5'))}，Frame-F1={fmt(frame.get('target_macro_frame_f1'))}。"
            )
    analysis.append("本版 RGB 特征是轻量统计特征，不训练图像 backbone，也不引入端到端视觉模型；它主要验证“ROI 局部外观是否有信号”，工程风险低，但表达能力明显弱于 DINOv2/VideoMAE。")
    analysis.append("由于 train/val/test 序列数量较少，单次 3 epoch 结果波动会比较大，应主要看 per-class recall 和 segment F1 是否出现稳定方向，而不是只看单一 ACC。")
    return analysis


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ActionMixed image feature v1 models.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "image_train" / "output_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(BEST_RECIPES))
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = args.out_dir / "feature_store_v2"
    clear_npz_dir(feature_dir)
    actionmixed_to_feature_store(args.dataset_root, feature_dir)
    base_items = FeatureStore(feature_dir).load_all()
    if not base_items:
        raise RuntimeError(f"No feature sequences found in {feature_dir}")

    cache_index = make_rgb_cache(args.dataset_root, args.out_dir / "rgb_roi_cache", args.padding)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in args.models:
        print(f"training image v1 model: {model_name}", flush=True)
        results.append(train_one(model_name, base_items, cache_index, args.epochs, device, args.out_dir))

    report = {
        "status": "completed",
        "script": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "epochs": args.epochs,
        "padding": args.padding,
        "classes": CLASSES,
        "rgb_slots": SLOTS,
        "dataset_stats": dataset_stats(base_items),
        "results": results,
        "analysis": build_analysis(results),
        "next_steps": [
            "把 `rgb_roi_stats_v1` 与 baseline 最佳报告做横向对比，重点看 F1@0.25/F1@0.5 和每类 recall，判断 RGB 是否对边界和漏检有帮助。",
            "如果 RGB 统计特征有收益，下一版改为冻结 DINOv2 ROI embedding，并用 PCA/Linear 压到每槽 32 或 64 维。",
            "加入质量感知融合：用 present/conf/missing_age/imputed/candidate_count 学一个 gate，而不是简单 concat。",
            "对 `flush` 和 `air_injection` 单独增强 syringe/air_gun ROI，并检查两类误检混淆；这两类最依赖外观区分。",
            "正式结论前至少跑 20-100 epoch，并重复 3 个随机种子；当前 3 epoch 只能视为首版工程验证。",
        ],
    }
    json_path = args.out_dir / "image_train_v1.json"
    md_path = args.report_path or (ROOT / "image_train" / "image_train_v1.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "report": str(md_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
