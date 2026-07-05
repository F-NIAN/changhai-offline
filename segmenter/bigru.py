"""
BiGRU 时序分割 baseline。

输入：
    x [batch, time, feature_dim]

输出：
    logits [batch, class_count, time]

说明：
    双向 GRU 会从正向和反向各读一遍完整序列，因此适合离线场景；
    它不是实时因果模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    family = "BiGRU temporal baseline"

    def __init__(self, in_dim: int, classes: int, hidden: int = 48):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.classifier = nn.Linear(hidden * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.projection(x))
        z, _ = self.gru(z)
        return self.classifier(z).transpose(1, 2)

