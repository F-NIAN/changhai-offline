"""Reusable ROI manifest and RGB feature cache utilities for image_train.

The temporal models are version-independent.  What changes across image_train
versions is the feature recipe, so ROI alignment is cached here once and reused
by v2+ experiments.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_transfer import OBJECT_MAP, OBJECTS
from dataset import ACTIONMIXED_DETECTION_CLASSES, _read_simple_yaml_names

FRAME_RE = re.compile(r"^(?P<video>.+\.mp4)-(?P<frame>\d+)\.txt$")
SLOTS = ["hand_top1", "hand_top2"] + [obj for obj in OBJECTS if obj != "hand"]
CRITICAL_SLOTS = [
    "hand_top1",
    "hand_top2",
    "short_brush",
    "syringe",
    "air_gun",
    "scope_distal_end",
    "brush_tip_out",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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


def normalized_xyxy(row: tuple[str, float, float, float, float, float], padding: float) -> tuple[float, float, float, float]:
    _, cx, cy, width, height, _ = row
    pad_w = width * padding
    pad_h = height * padding
    x1 = max(0.0, cx - width / 2.0 - pad_w)
    y1 = max(0.0, cy - height / 2.0 - pad_h)
    x2 = min(1.0, cx + width / 2.0 + pad_w)
    y2 = min(1.0, cy + height / 2.0 + pad_h)
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return x1, y1, x2, y2


def build_frame_slots(
    yolo_path: Path,
    detection_names: dict[int, str],
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detections = parse_detections(yolo_path, detection_names)
    boxes = np.zeros((len(SLOTS), 4), dtype=np.float32)
    valid = np.zeros(len(SLOTS), dtype=np.float32)
    conf = np.zeros(len(SLOTS), dtype=np.float32)

    hand_rows = detections.get("hand", [])
    for slot_idx in range(2):
        if slot_idx < len(hand_rows):
            row = hand_rows[slot_idx]
            boxes[slot_idx] = normalized_xyxy(row, padding)
            valid[slot_idx] = 1.0
            conf[slot_idx] = row[5]

    for obj in OBJECTS:
        if obj == "hand":
            continue
        slot_idx = SLOTS.index(obj)
        rows = detections.get(obj, [])
        if not rows:
            continue
        row = rows[0]
        boxes[slot_idx] = normalized_xyxy(row, padding)
        valid[slot_idx] = 1.0
        conf[slot_idx] = row[5]
    return boxes, valid, conf


def make_roi_manifest(
    dataset_root: Path,
    manifest_dir: Path,
    padding: float = 0.15,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    detection_names = _read_simple_yaml_names(dataset_root / "frames" / "data.yaml", ACTIONMIXED_DETECTION_CLASSES)
    index: dict[tuple[str, str], Path] = {}
    manifest_meta: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        for video_id, frames in grouped_frame_files(dataset_root, split).items():
            path = manifest_dir / f"{split}__{safe_name(video_id)}.npz"
            if path.exists() and not force:
                index[(split, video_id)] = path
                continue
            boxes_rows = []
            valid_rows = []
            conf_rows = []
            frame_numbers = []
            image_paths = []
            yolo_paths = []
            for frame_no, yolo_path in frames:
                boxes, valid, conf = build_frame_slots(yolo_path, detection_names, padding)
                boxes_rows.append(boxes)
                valid_rows.append(valid)
                conf_rows.append(conf)
                frame_numbers.append(frame_no)
                image_paths.append(str(dataset_root / "images" / split / f"{video_id}-{frame_no:06d}.jpg"))
                yolo_paths.append(str(yolo_path))
            np.savez_compressed(
                path,
                split=np.array([split]),
                video_ref=np.array([video_id]),
                frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
                image_paths=np.asarray(image_paths),
                yolo_paths=np.asarray(yolo_paths),
                slots=np.asarray(SLOTS),
                boxes=np.asarray(boxes_rows, dtype=np.float32),
                valid=np.asarray(valid_rows, dtype=np.float32),
                conf=np.asarray(conf_rows, dtype=np.float32),
                padding=np.array([padding], dtype=np.float32),
            )
            index[(split, video_id)] = path
            manifest_meta.append({"split": split, "video_ref": video_id, "frames": len(frame_numbers)})

    meta_path = manifest_dir / "manifest_index.json"
    meta_path.write_text(
        json.dumps(
            {
                "version": "roi_manifest_v1",
                "padding": padding,
                "slots": SLOTS,
                "videos": manifest_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index


def crop_from_box(image_array: np.ndarray, box: np.ndarray, valid: float) -> np.ndarray | None:
    if valid <= 0:
        return None
    h, w = image_array.shape[:2]
    x1 = int(round(float(box[0]) * w))
    y1 = int(round(float(box[1]) * h))
    x2 = int(round(float(box[2]) * w))
    y2 = int(round(float(box[3]) * h))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    crop = image_array[y1:y2, x1:x2]
    return crop if crop.size else None


def roi_rgb_v2_feature(crop: np.ndarray | None, valid: float, conf: float, box: np.ndarray) -> np.ndarray:
    """Low-dimensional quality-aware ROI RGB feature.

    10 dims:
    valid, conf, area, aspect, r_mean, g_mean, b_mean, brightness_mean,
    brightness_std, saturation_proxy.
    """
    if valid <= 0 or crop is None:
        return np.zeros(10, dtype=np.float32)
    flat = crop.reshape(-1, 3).astype(np.float32) / 255.0
    mean_rgb = flat.mean(axis=0)
    brightness = flat.mean(axis=1)
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    area = min(1.0, width * height)
    aspect = min(4.0, width / max(height, 1e-6)) / 4.0
    saturation = float((flat.max(axis=1) - flat.min(axis=1)).mean())
    return np.asarray(
        [
            1.0,
            conf,
            area,
            aspect,
            mean_rgb[0],
            mean_rgb[1],
            mean_rgb[2],
            float(brightness.mean()),
            float(brightness.std()),
            saturation,
        ],
        dtype=np.float32,
    )


def roi_rgb_v2_names(slots: list[str] | None = None) -> list[str]:
    suffixes = [
        "valid",
        "conf",
        "area",
        "aspect",
        "r_mean",
        "g_mean",
        "b_mean",
        "brightness_mean",
        "brightness_std",
        "saturation_proxy",
    ]
    active_slots = slots or CRITICAL_SLOTS
    return [f"roi_rgb_v2_{slot}_{suffix}" for slot in active_slots for suffix in suffixes]


def centered_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values.astype(np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        out[idx] = values[lo:hi].mean(axis=0)
    return out


def make_roi_rgb_v2_cache(
    manifest_index: dict[tuple[str, str], Path],
    cache_dir: Path,
    slots: list[str] | None = None,
    smooth_window: int = 5,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    active_slots = slots or CRITICAL_SLOTS
    slot_indices = [SLOTS.index(slot) for slot in active_slots]
    index: dict[tuple[str, str], Path] = {}

    for key, manifest_path in manifest_index.items():
        split, video_id = key
        out_path = cache_dir / f"{split}__{safe_name(video_id)}.npz"
        if out_path.exists() and not force:
            index[key] = out_path
            continue
        data = np.load(manifest_path, allow_pickle=True)
        image_paths = [Path(str(x)) for x in data["image_paths"]]
        boxes = data["boxes"].astype(np.float32)
        valid = data["valid"].astype(np.float32)
        conf = data["conf"].astype(np.float32)

        rows = []
        missing_images = 0
        for t, image_path in enumerate(image_paths):
            try:
                image_array = np.asarray(Image.open(image_path).convert("RGB"))
            except OSError:
                image_array = None
                missing_images += 1
            frame_parts = []
            for slot_idx in slot_indices:
                crop = crop_from_box(image_array, boxes[t, slot_idx], valid[t, slot_idx]) if image_array is not None else None
                frame_parts.append(roi_rgb_v2_feature(crop, valid[t, slot_idx], conf[t, slot_idx], boxes[t, slot_idx]))
            rows.append(np.concatenate(frame_parts).astype(np.float32))

        raw = np.stack(rows, axis=0).astype(np.float32)
        radius = max(1, smooth_window // 2)
        smooth = centered_mean(raw, radius)
        delta = np.zeros_like(raw, dtype=np.float32)
        if len(raw) > 1:
            delta[1:] = np.clip(raw[1:] - raw[:-1], -1.0, 1.0)
        features = np.concatenate([raw, smooth, delta], axis=1).astype(np.float32)
        names = roi_rgb_v2_names(active_slots)
        feature_names = names + [f"{name}_center_mean_w{smooth_window}" for name in names] + [f"{name}_delta" for name in names]
        np.savez_compressed(
            out_path,
            split=np.array([split]),
            video_ref=np.array([video_id]),
            frame_numbers=data["frame_numbers"].astype(np.int64),
            feature_names=np.asarray(feature_names),
            rgb_features=features,
            slots=np.asarray(active_slots),
            smooth_window=np.array([smooth_window], dtype=np.int64),
            missing_images=np.array([missing_images], dtype=np.int64),
        )
        index[key] = out_path
    return index


def load_cached_rgb_features(
    cache_index: dict[tuple[str, str], Path],
    item: dict[str, Any],
    expected_dim: int,
) -> tuple[np.ndarray, list[str]]:
    key = (str(item.get("split", "")), str(item.get("video_ref", "")))
    path = cache_index.get(key)
    expected_len = int(item["features"].shape[0])
    if path is None:
        return np.zeros((expected_len, expected_dim), dtype=np.float32), []
    data = np.load(path, allow_pickle=True)
    rgb = data["rgb_features"].astype(np.float32)
    names = [str(x) for x in data["feature_names"]]
    if rgb.shape[0] == expected_len:
        return rgb, names
    out = np.zeros((expected_len, rgb.shape[1]), dtype=np.float32)
    n = min(expected_len, rgb.shape[0])
    out[:n] = rgb[:n]
    return out, names
