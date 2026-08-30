"""Proxy ROI RGB/motion feature cache for image_train v4.

These features target tools that are hard to detect directly:
- wash/channel fixed ROIs for long-brush proxy evidence
- expanded/ring hand ROIs for short-brush proxy evidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image_train.common.roi_cache import SLOTS, centered_mean, crop_from_box, safe_name

PROXY_SLOTS = [
    "wash_tank",
    "wash_tank_left",
    "wash_tank_center",
    "wash_tank_right",
    "scope_channel",
    "hand_top1_expanded",
    "hand_top1_ring",
    "hand_top2_expanded",
    "hand_top2_ring",
    "hand_control_union",
]

PROXY_SUFFIXES = [
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
    "edge_energy",
    "motion_energy",
]

DEFAULT_FIXED_BOXES: dict[str, tuple[float, float, float, float]] = {
    "wash_tank": (0.05, 0.08, 0.95, 0.92),
    "wash_tank_left": (0.05, 0.08, 0.35, 0.92),
    "wash_tank_center": (0.35, 0.08, 0.65, 0.92),
    "wash_tank_right": (0.65, 0.08, 0.95, 0.92),
    "scope_channel": (0.10, 0.35, 0.90, 0.65),
}


def clip_box(box: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float32).copy()
    arr[[0, 2]] = np.clip(arr[[0, 2]], 0.0, 1.0)
    arr[[1, 3]] = np.clip(arr[[1, 3]], 0.0, 1.0)
    if arr[2] <= arr[0]:
        arr[2] = min(1.0, arr[0] + 1e-4)
    if arr[3] <= arr[1]:
        arr[3] = min(1.0, arr[1] + 1e-4)
    return arr


def expand_box(box: np.ndarray, factor: float) -> np.ndarray:
    cx = float((box[0] + box[2]) / 2.0)
    cy = float((box[1] + box[3]) / 2.0)
    width = float(box[2] - box[0]) * factor
    height = float(box[3] - box[1]) * factor
    return clip_box((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0))


def union_box(*boxes: np.ndarray) -> np.ndarray:
    stacked = np.stack([clip_box(box) for box in boxes], axis=0)
    return clip_box((stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 2].max(), stacked[:, 3].max()))


def normalized_inner_mask(outer: np.ndarray, inner: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    height, width = image_shape
    ox1 = int(round(float(outer[0]) * width))
    oy1 = int(round(float(outer[1]) * height))
    ox2 = int(round(float(outer[2]) * width))
    oy2 = int(round(float(outer[3]) * height))
    ix1 = int(round(float(inner[0]) * width)) - ox1
    iy1 = int(round(float(inner[1]) * height)) - oy1
    ix2 = int(round(float(inner[2]) * width)) - ox1
    iy2 = int(round(float(inner[3]) * height)) - oy1
    crop_h = max(1, oy2 - oy1)
    crop_w = max(1, ox2 - ox1)
    mask = np.ones((crop_h, crop_w), dtype=bool)
    ix1 = max(0, min(crop_w, ix1))
    ix2 = max(0, min(crop_w, ix2))
    iy1 = max(0, min(crop_h, iy1))
    iy2 = max(0, min(crop_h, iy2))
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = False
    if mask.sum() < 8:
        return None
    return mask


def sample_crop(crop: np.ndarray, mask: np.ndarray | None = None, max_side: int = 40) -> tuple[np.ndarray, np.ndarray | None]:
    if crop is None or crop.size == 0:
        return crop, mask
    step = max(1, int(np.ceil(max(crop.shape[0], crop.shape[1]) / max_side)))
    sampled = crop[::step, ::step]
    sampled_mask = mask[::step, ::step] if mask is not None and mask.shape[:2] == crop.shape[:2] else None
    return sampled, sampled_mask


def edge_energy(crop: np.ndarray, mask: np.ndarray | None = None) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = crop.astype(np.float32).mean(axis=2) / 255.0
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    if mask is not None and mask.shape == gray.shape:
        gx_mask = mask[:, 1:] & mask[:, :-1]
        gy_mask = mask[1:, :] & mask[:-1, :]
        gx_value = float(gx[gx_mask].mean()) if gx_mask.any() else 0.0
        gy_value = float(gy[gy_mask].mean()) if gy_mask.any() else 0.0
        return min(1.0, gx_value + gy_value)
    return min(1.0, float(gx.mean() + gy.mean()))


def proxy_roi_feature(
    crop: np.ndarray | None,
    prev_crop: np.ndarray | None,
    valid: float,
    conf: float,
    box: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if valid <= 0 or crop is None or crop.size == 0:
        return np.zeros(len(PROXY_SUFFIXES), dtype=np.float32)

    sampled_crop, sampled_mask = sample_crop(crop, mask)
    if sampled_mask is not None and sampled_mask.shape[:2] == sampled_crop.shape[:2] and sampled_mask.any():
        pixels = sampled_crop[sampled_mask].reshape(-1, 3).astype(np.float32) / 255.0
    else:
        pixels = sampled_crop.reshape(-1, 3).astype(np.float32) / 255.0
    mean_rgb = pixels.mean(axis=0)
    brightness = pixels.mean(axis=1)
    saturation = float((pixels.max(axis=1) - pixels.min(axis=1)).mean())
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    area = min(1.0, width * height)
    aspect = min(4.0, width / max(height, 1e-6)) / 4.0

    motion = 0.0
    if prev_crop is not None and prev_crop.size:
        sampled_prev, _ = sample_crop(prev_crop, None, max_side=40)
        min_h = min(sampled_crop.shape[0], sampled_prev.shape[0])
        min_w = min(sampled_crop.shape[1], sampled_prev.shape[1])
        if min_h > 0 and min_w > 0:
            cur_small = sampled_crop[:min_h, :min_w].astype(np.float32) / 255.0
            prev_small = sampled_prev[:min_h, :min_w].astype(np.float32) / 255.0
            motion = float(np.abs(cur_small - prev_small).mean())

    return np.asarray(
        [
            1.0,
            float(np.clip(conf, 0.0, 1.0)),
            area,
            aspect,
            mean_rgb[0],
            mean_rgb[1],
            mean_rgb[2],
            float(brightness.mean()),
            float(brightness.std()),
            saturation,
            edge_energy(sampled_crop, sampled_mask),
            min(1.0, motion),
        ],
        dtype=np.float32,
    )


def proxy_rgb_v4_names(slots: list[str] | None = None) -> list[str]:
    active_slots = slots or PROXY_SLOTS
    return [f"proxy_rgb_v4_{slot}_{suffix}" for slot in active_slots for suffix in PROXY_SUFFIXES]


def frame_proxy_boxes(
    boxes: np.ndarray,
    valid: np.ndarray,
    conf: np.ndarray,
    hand_expand: float,
    fixed_boxes: dict[str, tuple[float, float, float, float]],
) -> list[tuple[str, np.ndarray, float, float, np.ndarray | None]]:
    hand1 = SLOTS.index("hand_top1")
    hand2 = SLOTS.index("hand_top2")
    control = SLOTS.index("scope_control_body") if "scope_control_body" in SLOTS else -1

    rows: list[tuple[str, np.ndarray, float, float, np.ndarray | None]] = []
    for name in ("wash_tank", "wash_tank_left", "wash_tank_center", "wash_tank_right", "scope_channel"):
        rows.append((name, clip_box(fixed_boxes[name]), 1.0, 1.0, None))

    for slot_name, slot_idx in (("hand_top1", hand1), ("hand_top2", hand2)):
        if valid[slot_idx] > 0:
            base = clip_box(boxes[slot_idx])
            expanded = expand_box(base, hand_expand)
            rows.append((f"{slot_name}_expanded", expanded, 1.0, float(conf[slot_idx]), None))
            rows.append((f"{slot_name}_ring", expanded, 1.0, float(conf[slot_idx]), base))
        else:
            zero = np.zeros(4, dtype=np.float32)
            rows.append((f"{slot_name}_expanded", zero, 0.0, 0.0, None))
            rows.append((f"{slot_name}_ring", zero, 0.0, 0.0, None))

    if valid[hand1] > 0 and control >= 0 and valid[control] > 0:
        hand_control = expand_box(union_box(boxes[hand1], boxes[control]), 1.15)
        rows.append(("hand_control_union", hand_control, 1.0, float(min(conf[hand1], conf[control])), None))
    elif valid[hand1] > 0:
        rows.append(("hand_control_union", expand_box(boxes[hand1], hand_expand), 1.0, float(conf[hand1]), None))
    else:
        rows.append(("hand_control_union", np.zeros(4, dtype=np.float32), 0.0, 0.0, None))

    return rows


def make_proxy_rgb_v4_cache(
    manifest_index: dict[tuple[str, str], Path],
    cache_dir: Path,
    smooth_window: int = 5,
    hand_expand: float = 1.6,
    fixed_boxes: dict[str, tuple[float, float, float, float]] | None = None,
    force: bool = False,
) -> dict[tuple[str, str], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixed = dict(DEFAULT_FIXED_BOXES)
    if fixed_boxes:
        fixed.update({name: tuple(value) for name, value in fixed_boxes.items()})
    index: dict[tuple[str, str], Path] = {}
    meta_rows: list[dict[str, Any]] = []

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
        prev_image: np.ndarray | None = None
        for t, image_path in enumerate(image_paths):
            try:
                image = np.asarray(Image.open(image_path).convert("RGB"))
            except OSError:
                image = None
                missing_images += 1

            frame_parts = []
            for _, box, box_valid, box_conf, inner_box in frame_proxy_boxes(
                boxes[t], valid[t], conf[t], hand_expand, fixed
            ):
                crop = crop_from_box(image, box, box_valid) if image is not None else None
                prev_crop = crop_from_box(prev_image, box, box_valid) if prev_image is not None else None
                mask = (
                    normalized_inner_mask(box, inner_box, image.shape[:2])
                    if image is not None and inner_box is not None and box_valid > 0
                    else None
                )
                frame_parts.append(proxy_roi_feature(crop, prev_crop, box_valid, box_conf, box, mask))
            rows.append(np.concatenate(frame_parts).astype(np.float32))
            prev_image = image

        raw = np.stack(rows, axis=0).astype(np.float32)
        radius = max(1, smooth_window // 2)
        smooth = centered_mean(raw, radius)
        delta = np.zeros_like(raw, dtype=np.float32)
        if len(raw) > 1:
            delta[1:] = np.clip(raw[1:] - raw[:-1], -1.0, 1.0)
        features = np.concatenate([raw, smooth, delta], axis=1).astype(np.float32)
        names = proxy_rgb_v4_names(PROXY_SLOTS)
        feature_names = names + [f"{name}_center_mean_w{smooth_window}" for name in names] + [f"{name}_delta" for name in names]
        np.savez_compressed(
            out_path,
            split=np.array([split]),
            video_ref=np.array([video_id]),
            frame_numbers=data["frame_numbers"].astype(np.int64),
            feature_names=np.asarray(feature_names),
            proxy_features=features,
            proxy_slots=np.asarray(PROXY_SLOTS),
            smooth_window=np.array([smooth_window], dtype=np.int64),
            hand_expand=np.array([hand_expand], dtype=np.float32),
            fixed_boxes_json=np.array([json.dumps(fixed, ensure_ascii=False)]),
            missing_images=np.array([missing_images], dtype=np.int64),
        )
        index[key] = out_path
        meta_rows.append({"split": split, "video_ref": video_id, "frames": int(features.shape[0]), "missing_images": missing_images})

    (cache_dir / "proxy_rgb_v4_index.json").write_text(
        json.dumps(
            {
                "version": "proxy_rgb_v4",
                "slots": PROXY_SLOTS,
                "suffixes": PROXY_SUFFIXES,
                "smooth_window": smooth_window,
                "hand_expand": hand_expand,
                "fixed_boxes": fixed,
                "videos": meta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index


def load_cached_proxy_features(
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
    features = data["proxy_features"].astype(np.float32)
    names = [str(x) for x in data["feature_names"]]
    if features.shape[0] == expected_len:
        return features, names
    out = np.zeros((expected_len, features.shape[1]), dtype=np.float32)
    n = min(expected_len, features.shape[0])
    out[:n] = features[:n]
    return out, names
