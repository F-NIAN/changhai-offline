"""
MS-TCN 简化版。

输入：
    x [batch, time, feature_dim]

输出：
    logits [batch, class_count, time]

说明：
    用多层膨胀一维卷积模拟 MS-TCN 的大时间感受野。这里不是论文完整复现，
    重点是作为离线 action segmentation baseline 跑通完整链路。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    """残差膨胀卷积块：在时间轴上扩大上下文。"""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class MSTCN(nn.Module):
    family = "MS-TCN simplified baseline"

    def __init__(self, in_dim: int, classes: int, hidden: int = 32):
        super().__init__()
        self.input_projection = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualTemporalBlock(hidden, dilation) for dilation in [1, 2, 4, 8, 16, 1, 2, 4])
        )
        self.classifier = nn.Conv1d(hidden, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d 需要 [batch, channels, time]，所以先把 feature_dim 转到 channel 维。
        z = self.input_projection(x.transpose(1, 2))
        z = self.blocks(z)
        return self.classifier(z)

