"""
数据集构造、划分与 ActionMixed 接入模块。

通用输入:
    FeatureStore.load_all() 读出的完整序列列表，每条序列包含:
    features [time, feature_dim]、labels [time]、fps、task_id、step_id。

通用输出:
    train/val/test 划分、训练归一化参数、类别权重。

ActionMixed 输入:
    ModelScope 数据集 lhh010/cleansight-ActionMixed，本地目录结构:
        frames/{train,val,test}/{video_id}.mp4-{frame_id:06d}.txt  # 单帧 YOLO bbox
        labels/{train,val,test}/{video_id}.mp4.txt                # 动作标签

ActionMixed 输出:
    FeatureStore-like npz:
        features: 由逐帧 YOLO bbox 聚合出的几何/数量/速度/关系特征
        labels:   idle / long_brush_insert / long_brush_withdraw /
                  short_brush_cleaning / flush / air_injection
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from data_transfer import (
    CLASS_TO_ID,
    CLASSES,
    OBJECT_MAP,
    OBJECTS,
    _build_feature_matrix,
    save_feature_sequence,
)


@dataclass
class SplitResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]


class SequenceDataset(Dataset):
    """PyTorch Dataset 封装；每个样本是一整条视频/片段时间序列。"""

    def __init__(self, items: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray):
        self.items = items
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        item = self.items[idx]
        x = ((item["features"] - self.mean) / self.std).astype(np.float32)
        y = item["labels"].astype(np.int64)
        meta = {
            key: item[key]
            for key in ["task_id", "step_id", "fps", "frames", "duration_s"]
            if key in item
        }
        if "split" in item:
            meta["split"] = item["split"]
        if "video_ref" in item:
            meta["video_ref"] = item["video_ref"]
        return torch.tensor(x), torch.tensor(y), meta


def split_by_task(
    items: list[dict[str, Any]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 20260701,
) -> SplitResult:
    """按样本划分数据集；小数据下至少保留 1 条验证样本。"""
    if not items:
        raise ValueError("没有可划分的数据")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    test_n = int(round(n * test_ratio)) if n >= 5 else 0
    val_n = max(1, int(round(n * val_ratio))) if n >= 2 else 0
    test = shuffled[:test_n]
    val = shuffled[test_n : test_n + val_n]
    train = shuffled[test_n + val_n :]
    if not train:
        train, val = val, []
    return SplitResult(train=train, val=val, test=test)


def split_by_declared_split(items: list[dict[str, Any]], seed: int = 20260701) -> SplitResult:
    """优先使用数据源自带的 train/val/test 字段；缺失时回退到随机划分。"""
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for item in items:
        split = str(item.get("split", "")).lower()
        if split in buckets:
            buckets[split].append(item)

    if not any(buckets.values()):
        return split_by_task(items, seed=seed)

    # 极端情况下数据源只给 train，没有 val；这里从 train 中切出一小部分做 sanity check。
    if not buckets["val"] and len(buckets["train"]) >= 2:
        fallback = split_by_task(buckets["train"], val_ratio=0.2, seed=seed)
        buckets["train"], buckets["val"] = fallback.train, fallback.val

    if not buckets["train"]:
        fallback = split_by_task(items, seed=seed)
        return fallback

    return SplitResult(train=buckets["train"], val=buckets["val"], test=buckets["test"])


def make_normalizer(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """只用训练集计算均值和方差，避免验证集信息泄漏。"""
    x = np.concatenate([item["features"] for item in items], axis=0)
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-4] = 1.0
    return mean, std


def class_weights(items: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    """按帧数反比生成类别权重，缓解 idle 或单一动作占比过高的问题。"""
    y = np.concatenate([item["labels"] for item in items])
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / counts[present]
        weights[present] = weights[present] / np.mean(weights[present])
    support = {CLASSES[idx]: int(count) for idx, count in enumerate(counts)}
    return torch.tensor(weights, dtype=torch.float32, device=device), support


ACTIONMIXED_DATASET = "lhh010/cleansight-ActionMixed"

ACTIONMIXED_ACTION_CLASSES = {
    0: "idle",
    1: "air_injection",
    2: "flush",
    3: "long_brush_insert",
    4: "long_brush_withdraw",
    5: "short_brush_cleaning",
}

ACTIONMIXED_DETECTION_CLASSES = {
    0: "hand",
    1: "scope_control_body",
    2: "scope_mid_section",
    3: "scope_distal_end",
    4: "syringe",
    5: "air_gun",
    6: "short_brush",
    7: "brush_tip_out",
}

ACTIONMIXED_TO_TARGET_CLASS = {
    "idle": "idle",
    "air_injection": "air_injection",
    "flush": "flush",
    "long_brush_insert": "long_brush_insert",
    "long_brush_withdraw": "long_brush_withdraw",
    "short_brush_cleaning": "short_brush_cleaning",
}

FRAME_FILE_RE = re.compile(r"^(?P<video>.+\.mp4)-(?P<frame>\d+)\.txt$")


@dataclass
class ParsedActionLabels:
    """动作标签解析结果，同时兼容逐行标签、frame->label 和 segment range 三种格式。"""

    ordered: list[int]
    by_frame: dict[int, int]
    ranges: list[tuple[int, int, int]]


def actionmixed_default_dir(base_dir: Path) -> Path:
    """ModelScope 数据集默认落盘目录名，避免路径里出现斜杠。"""
    return Path(base_dir) / ACTIONMIXED_DATASET.replace("/", "__")


def ensure_actionmixed_dataset(
    cache_dir: Path,
    dataset_name: str = ACTIONMIXED_DATASET,
    force_clone: bool = False,
    refresh_lfs: bool = False,
    include_images: bool = False,
) -> Path:
    """通过 Git LFS 获取 ActionMixed 数据集。

    默认只拉取 frames/labels/说明文件，训练 baseline 不读取 images，可以显著减少下载量。
    如果已经手动 clone 到目标目录，本函数会直接复用；传 refresh_lfs=True 时会补拉 LFS 文本物料。
    """
    cache_dir = Path(cache_dir)
    target = cache_dir / dataset_name.replace("/", "__")
    clone_url = f"https://www.modelscope.cn/datasets/{dataset_name}.git"

    if force_clone and target.exists():
        shutil.rmtree(target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(["git", "lfs", "install"], check=True, cwd=str(cache_dir))
        subprocess.run(["git", "clone", clone_url, str(target)], check=True, cwd=str(cache_dir), env=env)

    if refresh_lfs:
        include = ["frames/**", "labels/**", "README.md", "tracking_actionmixed.md"]
        if include_images:
            include.append("images/**")
        exclude = "" if include_images else "images/**"
        subprocess.run(
            ["git", "lfs", "pull", "--include", ",".join(include), "--exclude", exclude],
            check=True,
            cwd=str(target),
        )

    return target


def _stable_task_id(text: str) -> int:
    """把视频名/切分名变成稳定整数 id，避免依赖 Label Studio task id。"""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 100000000


def _is_lfs_pointer(path: Path) -> bool:
    """判断文件是否还只是 Git LFS 指针；指针文件不能用于训练。"""
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:128]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec")


def _read_simple_yaml_names(path: Path, fallback: dict[int, str]) -> dict[int, str]:
    """轻量解析 data.yaml 的 names，不额外引入 PyYAML 依赖。"""
    if not path.exists() or _is_lfs_pointer(path):
        return dict(fallback)
    names = dict(fallback)
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def _target_id_from_action_name(name: str) -> int:
    target_name = ACTIONMIXED_TO_TARGET_CLASS.get(name, "idle")
    return CLASS_TO_ID[target_name]


def _target_id_from_action_id(action_id: int, action_names: dict[int, str]) -> int:
    return _target_id_from_action_name(action_names.get(action_id, "idle"))


def _target_id_from_action_text(text: str, action_names: dict[int, str]) -> int | None:
    normalized = text.lower()
    for name in sorted(action_names.values(), key=len, reverse=True):
        if name.lower() in normalized:
            return _target_id_from_action_name(name)
    return None


def _numeric_tokens(line: str) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[\s,;]+", line.strip()):
        if not token:
            continue
        try:
            values.append(int(round(float(token))))
        except ValueError:
            continue
    return values


def _parse_action_label_file(path: Path, action_names: dict[int, str]) -> ParsedActionLabels:
    """解析 ActionMixed 动作标签。

    兼容几类常见写法:
    - 每行一个 class_id，表示按 frames 文件排序的一帧标签；
    - frame_id class_id 或 class_id frame_id；
    - start_frame end_frame class_id 或 class_id start_frame end_frame；
    - 行内带动作名称文本，如 long_brush_insert。
    """
    if _is_lfs_pointer(path):
        raise RuntimeError(f"{path} 仍是 Git LFS 指针，请先执行 git lfs pull --include labels/**,frames/**")

    ordered: list[int] = []
    by_frame: dict[int, int] = {}
    ranges: list[tuple[int, int, int]] = []
    max_action_id = max(action_names)

    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        text_label = _target_id_from_action_text(line, action_names)
        nums = _numeric_tokens(line)

        if len(nums) >= 3:
            a, b, c = nums[:3]
            if 0 <= a <= max_action_id and b <= c:
                ranges.append((b, c, _target_id_from_action_id(a, action_names)))
                continue
            if 0 <= c <= max_action_id and a <= b:
                ranges.append((a, b, _target_id_from_action_id(c, action_names)))
                continue

        if len(nums) >= 2:
            a, b = nums[:2]
            # ActionMixed labels are exported as "frame_id action_id".  The
            # earliest sampled frames can be 1/5, which are also valid class
            # ids, so this format must be checked before the legacy
            # "action_id frame_id" fallback.
            if 0 <= b <= max_action_id:
                by_frame[a] = _target_id_from_action_id(b, action_names)
                continue
            if 0 <= a <= max_action_id:
                by_frame[b] = _target_id_from_action_id(a, action_names)
                continue

        if len(nums) == 1 and 0 <= nums[0] <= max_action_id:
            ordered.append(_target_id_from_action_id(nums[0], action_names))
            continue

        if text_label is not None:
            ordered.append(text_label)

    return ParsedActionLabels(ordered=ordered, by_frame=by_frame, ranges=ranges)


def _labels_for_sampled_frames(parsed: ParsedActionLabels, frame_numbers: list[int]) -> np.ndarray:
    """把动作标签对齐到本次训练实际使用的采样帧序列。"""
    labels = np.zeros(len(frame_numbers), dtype=np.int64)

    if parsed.ordered:
        n = min(len(parsed.ordered), len(frame_numbers))
        labels[:n] = np.asarray(parsed.ordered[:n], dtype=np.int64)
        if n == len(frame_numbers):
            return labels

    for idx, frame_no in enumerate(frame_numbers):
        if frame_no in parsed.by_frame:
            labels[idx] = parsed.by_frame[frame_no]

    for start, end, target_id in parsed.ranges:
        lo, hi = min(start, end), max(start, end)
        for idx, frame_no in enumerate(frame_numbers):
            if lo <= frame_no <= hi:
                labels[idx] = target_id

    return labels


def _parse_frame_file_name(path: Path) -> tuple[str, int] | None:
    match = FRAME_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("video"), int(match.group("frame"))


def _frame_groups(frame_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(frame_dir.glob("*.txt")):
        parsed = _parse_frame_file_name(path)
        if parsed is None:
            continue
        video_id, frame_no = parsed
        groups.setdefault(video_id, []).append((frame_no, path))
    for video_id in groups:
        groups[video_id].sort(key=lambda item: item[0])
    return groups


def _object_arrays_from_yolo_frames(
    frame_paths: list[Path],
    detection_names: dict[int, str],
    max_instances_per_object: int = 4,
) -> dict[str, list[np.ndarray]]:
    """把一组 YOLO txt 转成 _build_feature_matrix 需要的 object_arrays。

    YOLO 行格式:
        class_id cx cy w h
    坐标是 0-1 归一化中心点和宽高。没有 track id 时，按面积从大到小放入固定槽位。
    """
    per_frame: list[dict[str, list[tuple[float, float, float]]]] = []

    for path in frame_paths:
        if _is_lfs_pointer(path):
            raise RuntimeError(f"{path} 仍是 Git LFS 指针，请先执行 git lfs pull --include labels/**,frames/**")

        objects: dict[str, list[tuple[float, float, float]]] = {}
        for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, width, height = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                continue
            label = detection_names.get(class_id)
            obj = OBJECT_MAP.get(str(label))
            if obj is None:
                continue
            area = max(0.0, width) * max(0.0, height)
            objects.setdefault(obj, []).append((cx, cy, area))
        per_frame.append(objects)

    object_arrays: dict[str, list[np.ndarray]] = {}
    time_len = len(frame_paths)
    for obj in OBJECTS:
        max_slots = min(
            max((len(frame_objects.get(obj, [])) for frame_objects in per_frame), default=0),
            max_instances_per_object,
        )
        for slot_idx in range(max_slots):
            arr = np.zeros((time_len, 4), dtype=np.float32)
            for t, frame_objects in enumerate(per_frame):
                detections = sorted(frame_objects.get(obj, []), key=lambda row: row[2], reverse=True)
                if slot_idx < len(detections):
                    cx, cy, area = detections[slot_idx]
                    arr[t] = (1.0, cx, cy, area)
            object_arrays.setdefault(obj, []).append(arr)

    return object_arrays


def actionmixed_to_feature_store(
    dataset_root: Path,
    feature_dir: Path,
    fps: float = 7.5,
    splits: Iterable[str] = ("train", "val", "test"),
) -> list[dict[str, Any]]:
    """把 ActionMixed 转成模型可用的 FeatureStore-like npz。

    数据变化:
    1. labels/split/video.txt 解析为每个采样帧的动作真值；
    2. frames/split/video-frame.txt 解析为每个采样帧的 YOLO bbox；
    3. bbox 按对象类别聚合为 hand top-2、count/cx/cy/area/speed/对象距离等 68 维特征；
    4. 动作类映射到 idle + 五种动作标签；
    5. 保存为 task_<task_id>_step_1.npz。
    """
    dataset_root = Path(dataset_root)
    feature_dir = Path(feature_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"ActionMixed 目录不存在: {dataset_root}")

    action_names = _read_simple_yaml_names(dataset_root / "labels" / "data.yaml", ACTIONMIXED_ACTION_CLASSES)
    detection_names = _read_simple_yaml_names(dataset_root / "frames" / "data.yaml", ACTIONMIXED_DETECTION_CLASSES)

    items: list[dict[str, Any]] = []
    for split in splits:
        label_dir = dataset_root / "labels" / split
        frame_dir = dataset_root / "frames" / split
        if not label_dir.exists() or not frame_dir.exists():
            continue

        grouped_frames = _frame_groups(frame_dir)
        for label_path in sorted(label_dir.glob("*.txt")):
            video_id = label_path.name[:-4] if label_path.name.endswith(".txt") else label_path.stem
            frames = grouped_frames.get(video_id, [])
            if not frames:
                continue

            frame_numbers = [frame_no for frame_no, _ in frames]
            frame_paths = [path for _, path in frames]
            parsed_labels = _parse_action_label_file(label_path, action_names)
            labels = _labels_for_sampled_frames(parsed_labels, frame_numbers)
            object_arrays = _object_arrays_from_yolo_frames(frame_paths, detection_names)
            features, feature_names = _build_feature_matrix(object_arrays, len(frame_paths), fps)

            item = {
                "task_id": _stable_task_id(f"{split}:{video_id}"),
                "step_id": 1,
                "features": features,
                "labels": labels,
                "fps": float(fps),
                "frames": int(len(frame_paths)),
                "duration_s": float(len(frame_paths) / fps),
                "feature_names": feature_names,
                "file_upload": label_path.name,
                "video_ref": video_id,
                "source": "modelscope_actionmixed",
                "split": split,
            }
            save_feature_sequence(item, feature_dir)
            items.append(item)

    if not items:
        raise RuntimeError(
            f"未从 {dataset_root} 解析到 ActionMixed 样本；请确认 labels/ 和 frames/ 已通过 Git LFS 下载完成。"
        )

    return items
