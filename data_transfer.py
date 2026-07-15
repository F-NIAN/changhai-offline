"""
数据转换模块。

作用：
    把不同来源的数据转成离线时序分割模型统一使用的 FeatureStore-like npz。

输入：
    1. Label Studio JSON：包含 videorectangle 目标框和 timelinelabels 时间段。
    2. YOLO CSV：逐帧检测框，作为后续生产检测输出的占位转换方式。

输出：
    feature_store/task_<task_id>_step_<step_id>.npz

npz 字段：
    features: float32 [time, feature_dim]
    labels: int64 [time]，0 表示 idle / 无动作
    fps, frames, duration_s, feature_names, task_id, step_id
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CLASSES = [
    "idle",
    "long_brush_insert",
    "long_brush_withdraw",
    "short_brush_cleaning",
    "flush",
    "air_injection",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

ACTION_MAP = {
    "long_brush_insert": "long_brush_insert",
    "long_brush_withdraw": "long_brush_withdraw",
    "long_brush_cleaning": "long_brush_insert",
    "长毛刷清洗": "long_brush_insert",
    "short_brush_cleaning": "short_brush_cleaning",
    "短毛刷清洗": "short_brush_cleaning",
    "flush": "flush",
    "air_injection": "air_injection",
    "idle": "idle",
    "background": "idle",
    "背景/空闲": "idle",
    "空闲": "idle",
    "无动作": "idle",
}

OBJECT_MAP = {
    "hand": "hand",
    "short_brush": "short_brush",
    "long_brush": "long_brush",
    "长毛刷清洗": "long_brush",
    "syringe": "syringe",
    "air_gun": "air_gun",
    "scope_control_body": "scope_control_body",
    "scope_mid_section": "scope_mid_section",
    "scope_distal_end": "scope_distal_end",
    "brush_tip_out": "brush_tip_out",
}

OBJECTS = [
    "hand",
    "short_brush",
    "long_brush",
    "syringe",
    "air_gun",
    "scope_control_body",
    "scope_mid_section",
    "scope_distal_end",
    "brush_tip_out",
]

PAIR_FEATURES = [
    ("hand", "short_brush"),
    ("hand", "long_brush"),
    ("brush_tip_out", "scope_distal_end"),
    ("short_brush", "scope_control_body"),
    ("long_brush", "scope_mid_section"),
    ("air_gun", "scope_distal_end"),
    ("syringe", "scope_distal_end"),
]


def _annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    """取出未取消的 Label Studio 标注结果。"""
    out: list[dict[str, Any]] = []
    for ann in task.get("annotations") or []:
        if not ann.get("was_cancelled"):
            out.extend(ann.get("result") or [])
    return out


def _infer_shape(results: list[dict[str, Any]]) -> tuple[int, float, float]:
    """从标注字段推断帧数、时长和 fps。"""
    frames = 0
    durations: list[float] = []
    for result in results:
        value = result.get("value") or {}
        if value.get("framesCount"):
            frames = max(frames, int(round(float(value["framesCount"]))))
        if value.get("duration"):
            durations.append(float(value["duration"]))
        for point in value.get("sequence") or []:
            if point.get("frame") is not None:
                frames = max(frames, int(round(float(point["frame"]))))
            if point.get("time") is not None:
                durations.append(float(point["time"]))
        for item in value.get("ranges") or []:
            if item.get("end") is not None:
                frames = max(frames, int(round(float(item["end"]))))
    duration = max(durations) if durations else max(frames / 24.0, 1.0)
    fps = frames / duration if duration > 0 else 24.0
    return max(frames, 1), float(duration), float(fps)


def _fill_mask_from_sequence(sequence: list[dict[str, Any]], frames: int) -> np.ndarray:
    """把 Label Studio 的 sequence 开关标注展开成逐帧 mask。"""
    mask = np.zeros(frames, dtype=bool)
    points = sorted(sequence, key=lambda p: int(round(float(p.get("frame", 1)))))
    for idx, point in enumerate(points):
        start = max(0, int(round(float(point.get("frame", 1)))) - 1)
        end = frames
        if idx + 1 < len(points):
            end = max(start, int(round(float(points[idx + 1].get("frame", frames)))) - 1)
        if bool(point.get("enabled", True)):
            mask[start:end] = True
    return mask


def _fill_box_sequence(sequence: list[dict[str, Any]], frames: int) -> np.ndarray:
    """把视频 bbox 展开成 [present, cx, cy, area] 的逐帧数组。"""
    arr = np.zeros((frames, 4), dtype=np.float32)
    points = sorted(sequence, key=lambda p: int(round(float(p.get("frame", 1)))))
    for idx, point in enumerate(points):
        start = max(0, int(round(float(point.get("frame", 1)))) - 1)
        end = frames
        if idx + 1 < len(points):
            end = max(start, int(round(float(points[idx + 1].get("frame", frames)))) - 1)
        if not bool(point.get("enabled", True)):
            continue
        x = float(point.get("x", 0.0)) / 100.0
        y = float(point.get("y", 0.0)) / 100.0
        w = float(point.get("width", 0.0)) / 100.0
        h = float(point.get("height", 0.0)) / 100.0
        arr[start:end] = (1.0, x + w / 2.0, y + h / 2.0, max(0.0, w * h))
    return arr


def _mark_action_range(labels: np.ndarray, start: float, end: float, cls: str, frames: int, duration: float) -> None:
    """把时间段标注写入逐帧标签；范围可能是秒，也可能是帧号。"""
    if end <= duration + 1 and start <= duration + 1:
        s = int(round(start / duration * frames))
        e = int(round(end / duration * frames))
    else:
        s = int(round(start)) - 1
        e = int(round(end))
    labels[max(0, s) : max(0, min(frames, e))] = CLASS_TO_ID[cls]


def _build_feature_matrix(object_arrays: dict[str, list[np.ndarray]], frames: int, fps: float) -> tuple[np.ndarray, list[str]]:
    """把目标框序列汇总为多维时序特征。

    与后端 `segmenters/clean.py` 对齐：
    - hand 使用 top-2 独立槽位，不把两只手加权合成一个中心点；
    - 其它对象仍按存在框加权聚合为 count/cx/cy/area/speed；
    - 输出维度为 68。
    """
    blocks: list[np.ndarray] = []
    names: list[str] = []
    centers: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}

    hand_arrs = object_arrays.get("hand", [])
    hand_count = np.zeros(frames, dtype=np.float32)
    hand_slots = [np.zeros((frames, 4), dtype=np.float32), np.zeros((frames, 4), dtype=np.float32)]
    if hand_arrs:
        hand_stack = np.stack(hand_arrs, axis=0)
        hand_count = hand_stack[:, :, 0].sum(axis=0).astype(np.float32)
        for t in range(frames):
            candidates = []
            for arr in hand_arrs:
                if arr[t, 0] > 0:
                    # arr[t] = [present, cx, cy, area]；按面积选 top-2。
                    candidates.append(arr[t].copy())
            candidates.sort(key=lambda row: float(row[3]), reverse=True)
            for slot_idx, row in enumerate(candidates[:2]):
                hand_slots[slot_idx][t] = row

    blocks.append((np.clip(hand_count, 0, 3) / 3.0)[:, None].astype(np.float32))
    names.append("hand_count")
    hand_centers = []
    for slot_idx, slot in enumerate(hand_slots, start=1):
        coords = slot[:, 1:3]
        speed = np.zeros(frames, dtype=np.float32)
        if frames > 1:
            speed[1:] = np.clip(np.linalg.norm(np.diff(coords, axis=0), axis=1) * fps, 0.0, 5.0) / 5.0
            speed[slot[:, 0] <= 0] = 0.0
        blocks.append(np.stack([slot[:, 0], slot[:, 1], slot[:, 2], slot[:, 3], speed], axis=1).astype(np.float32))
        names += [
            f"hand_top{slot_idx}_present",
            f"hand_top{slot_idx}_cx",
            f"hand_top{slot_idx}_cy",
            f"hand_top{slot_idx}_area",
            f"hand_top{slot_idx}_speed",
        ]
        hand_centers.append(coords)
    centers["hand"] = np.stack(hand_centers, axis=0)  # [2, T, 2]
    present["hand"] = hand_count > 0

    for obj in OBJECTS:
        if obj == "hand":
            continue
        arrs = object_arrays.get(obj, [])
        if arrs:
            stack = np.stack(arrs, axis=0)
            count = stack[:, :, 0].sum(axis=0)
            denom = np.maximum(count, 1.0)
            cx = (stack[:, :, 1] * stack[:, :, 0]).sum(axis=0) / denom
            cy = (stack[:, :, 2] * stack[:, :, 0]).sum(axis=0) / denom
            area = (stack[:, :, 3] * stack[:, :, 0]).sum(axis=0) / denom
        else:
            count = np.zeros(frames, dtype=np.float32)
            cx = np.zeros(frames, dtype=np.float32)
            cy = np.zeros(frames, dtype=np.float32)
            area = np.zeros(frames, dtype=np.float32)

        coords = np.stack([cx, cy], axis=1)
        speed = np.zeros(frames, dtype=np.float32)
        if frames > 1:
            speed[1:] = np.clip(np.linalg.norm(np.diff(coords, axis=0), axis=1) * fps, 0.0, 5.0) / 5.0
            speed[count <= 0] = 0.0
        blocks.append(np.stack([np.clip(count, 0, 3) / 3.0, cx, cy, area, speed], axis=1).astype(np.float32))
        names += [f"{obj}_count", f"{obj}_cx", f"{obj}_cy", f"{obj}_area", f"{obj}_speed"]
        centers[obj] = coords
        present[obj] = count > 0

    for a, b in PAIR_FEATURES:
        valid = (present[a] & present[b]).astype(np.float32)
        dist = np.zeros(frames, dtype=np.float32)
        if a == "hand":
            right = centers[b]
            d0 = np.linalg.norm(centers["hand"][0] - right, axis=1)
            d1 = np.linalg.norm(centers["hand"][1] - right, axis=1)
            dist = np.minimum(d0, d1).astype(np.float32)
        elif b == "hand":
            left = centers[a]
            d0 = np.linalg.norm(left - centers["hand"][0], axis=1)
            d1 = np.linalg.norm(left - centers["hand"][1], axis=1)
            dist = np.minimum(d0, d1).astype(np.float32)
        else:
            dist = np.linalg.norm(centers[a] - centers[b], axis=1).astype(np.float32)
        dist = np.where(valid > 0, np.clip(dist, 0.0, math.sqrt(2.0)) / math.sqrt(2.0), 0.0)
        blocks.append(np.stack([valid, dist], axis=1).astype(np.float32))
        names += [f"{a}_to_{b}_valid", f"{a}_to_{b}_dist"]

    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    blocks.append(np.stack([t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)], axis=1).astype(np.float32))
    names += ["t_norm", "t_sin", "t_cos"]
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def build_sequence_from_labelstudio_task(task: dict[str, Any], step_id: int = 1) -> dict[str, Any] | None:
    """把单个 Label Studio task 转成一条完整时序样本。"""
    results = _annotation_results(task)
    if not results:
        return None
    frames, duration, fps = _infer_shape(results)
    labels = np.zeros(frames, dtype=np.int64)
    object_arrays: dict[str, list[np.ndarray]] = defaultdict(list)

    for result in results:
        value = result.get("value") or {}
        if result.get("type") == "videorectangle":
            for label in value.get("labels") or []:
                obj = OBJECT_MAP.get(str(label))
                if obj:
                    object_arrays[obj].append(_fill_box_sequence(value.get("sequence") or [], frames))
        if result.get("type") == "timelinelabels":
            for label in value.get("timelinelabels") or []:
                mapped = ACTION_MAP.get(str(label))
                if not mapped:
                    continue
                for item in value.get("ranges") or []:
                    _mark_action_range(labels, float(item["start"]), float(item["end"]), mapped, frames, duration)
                if value.get("sequence"):
                    mask = _fill_mask_from_sequence(value.get("sequence") or [], frames)
                    labels[mask] = CLASS_TO_ID[mapped]

    features, feature_names = _build_feature_matrix(object_arrays, frames, fps)
    return {
        "task_id": int(task.get("id")),
        "step_id": int(step_id),
        "features": features,
        "labels": labels,
        "fps": float(fps),
        "frames": int(frames),
        "duration_s": float(duration),
        "feature_names": feature_names,
        "file_upload": str(task.get("file_upload") or ""),
        "video_ref": str((task.get("data") or {}).get("video") or ""),
    }


def save_feature_sequence(item: dict[str, Any], feature_dir: Path) -> Path:
    """保存一条 FeatureStore-like npz。"""
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"task_{item['task_id']}_step_{item['step_id']}.npz"
    np.savez_compressed(
        path,
        task_id=np.array([item["task_id"]]),
        step_id=np.array([item["step_id"]]),
        features=item["features"].astype(np.float32),
        labels=item["labels"].astype(np.int64),
        fps=np.array([item["fps"]], dtype=np.float32),
        frames=np.array([item["frames"]], dtype=np.int64),
        duration_s=np.array([item["duration_s"]], dtype=np.float32),
        feature_names=np.array(item["feature_names"]),
        file_upload=np.array([item.get("file_upload", "")]),
        video_ref=np.array([item.get("video_ref", "")]),
        source=np.array([item.get("source", "")]),
        split=np.array([item.get("split", "")]),
    )
    return path


def labelstudio_to_feature_store(labelstudio_dir: Path, feature_dir: Path, task_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """批量转换 Label Studio JSON；不再硬编码 id 范围。"""
    selected = set(task_ids) if task_ids is not None else None
    tasks: dict[int, dict[str, Any]] = {}
    for path in sorted(labelstudio_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = [data]
        for task in data:
            task_id = int(task["id"])
            if selected is None or task_id in selected:
                tasks[task_id] = task

    items: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        item = build_sequence_from_labelstudio_task(tasks[task_id])
        if item is not None:
            save_feature_sequence(item, feature_dir)
            items.append(item)
    return items


def yolo_csv_to_feature_store(yolo_csv: Path, feature_dir: Path, step_id: int = 1) -> list[dict[str, Any]]:
    """YOLO CSV 转换占位实现。

    期望字段：task_id, frame, fps, label, x1, y1, x2, y2，可选 width/height/track_id。
    这里只生成检测特征，labels 默认为 idle；真实训练标签需再合并人工时间段。
    """
    rows = list(csv.DictReader(yolo_csv.open("r", encoding="utf-8-sig")))
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)

    items: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        frames = max(int(float(row["frame"])) for row in task_rows)
        fps = float(task_rows[0].get("fps") or 24.0)
        object_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        by_instance: dict[tuple[str, str], np.ndarray] = {}
        for row in task_rows:
            obj = OBJECT_MAP.get(str(row["label"]))
            if not obj:
                continue
            key = (obj, row.get("track_id") or "default")
            arr = by_instance.setdefault(key, np.zeros((frames, 4), dtype=np.float32))
            idx = max(0, min(frames - 1, int(float(row["frame"])) - 1))
            width = float(row.get("width") or 1.0)
            height = float(row.get("height") or 1.0)
            x1 = float(row["x1"]) / width if width > 1 and float(row["x1"]) > 1 else float(row["x1"])
            y1 = float(row["y1"]) / height if height > 1 and float(row["y1"]) > 1 else float(row["y1"])
            x2 = float(row["x2"]) / width if width > 1 and float(row["x2"]) > 1 else float(row["x2"])
            y2 = float(row["y2"]) / height if height > 1 and float(row["y2"]) > 1 else float(row["y2"])
            arr[idx] = (1.0, (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1) * max(0.0, y2 - y1))
        for (obj, _), arr in by_instance.items():
            object_arrays[obj].append(arr)
        features, feature_names = _build_feature_matrix(object_arrays, frames, fps)
        item = {
            "task_id": task_id,
            "step_id": step_id,
            "features": features,
            "labels": np.zeros(frames, dtype=np.int64),
            "fps": fps,
            "frames": frames,
            "duration_s": frames / fps,
            "feature_names": feature_names,
            "file_upload": "",
            "video_ref": "",
        }
        save_feature_sequence(item, feature_dir)
        items.append(item)
    return items


class FeatureStore:
    """最小 FeatureStore 文件实现，模拟 FeatureStore.load(task_id, step_id)。"""

    def __init__(self, root: Path):
        self.root = Path(root)

    def list_task_steps(self) -> list[tuple[int, int]]:
        pairs = []
        for path in sorted(self.root.glob("task_*_step_*.npz")):
            parts = path.stem.split("_")
            pairs.append((int(parts[1]), int(parts[3])))
        return pairs

    def load(self, task_id: int, step_id: int = 1, sources: list[str] | None = None) -> dict[str, Any]:
        data = np.load(self.root / f"task_{task_id}_step_{step_id}.npz", allow_pickle=True)
        return {
            "task_id": int(data["task_id"][0]),
            "step_id": int(data["step_id"][0]),
            "features": data["features"].astype(np.float32),
            "labels": data["labels"].astype(np.int64),
            "fps": float(data["fps"][0]),
            "frames": int(data["frames"][0]),
            "duration_s": float(data["duration_s"][0]),
            "feature_names": [str(x) for x in data["feature_names"]],
            "file_upload": str(data["file_upload"][0]) if "file_upload" in data else "",
            "video_ref": str(data["video_ref"][0]) if "video_ref" in data else "",
            "source": str(data["source"][0]) if "source" in data else "",
            "split": str(data["split"][0]) if "split" in data else "",
            "sources": sources or ["bbox", "geometry", "motion"],
        }

    def load_all(self) -> list[dict[str, Any]]:
        return [self.load(task_id, step_id) for task_id, step_id in self.list_task_steps()]
