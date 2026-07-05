"""
数据集构造与划分模块。

输入：
    FeatureStore.load_all() 得到的完整序列列表。

输出：
    train/val/test 划分、训练归一化参数、类别权重。

说明：
    这里按 task 划分，避免同一视频同时出现在训练和验证中。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_transfer import CLASSES


@dataclass
class SplitResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]


class SequenceDataset(Dataset):
    """每个样本是一整条完整视频/任务序列。"""

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
        meta = {key: item[key] for key in ["task_id", "step_id", "fps", "frames", "duration_s"]}
        return torch.tensor(x), torch.tensor(y), meta


def split_by_task(items: list[dict[str, Any]], val_ratio: float = 0.2, test_ratio: float = 0.0, seed: int = 20260701) -> SplitResult:
    """按 task_id 划分数据集；小数据下至少留 1 条验证样本。"""
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


def make_normalizer(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """只用训练集计算均值和方差，避免验证集信息泄漏。"""
    x = np.concatenate([item["features"] for item in items], axis=0)
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-4] = 1.0
    return mean, std


def class_weights(items: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, dict[str, int]]:
    """生成类别权重，缓解 background 帧远多于动作帧的问题。"""
    y = np.concatenate([item["labels"] for item in items])
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / counts[present]
        weights[present] = weights[present] / np.mean(weights[present])
    support = {CLASSES[idx]: int(count) for idx, count in enumerate(counts)}
    return torch.tensor(weights, dtype=torch.float32, device=device), support

