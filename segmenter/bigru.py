"""
BiGRU 离线时序分割模型。

输入:
    x: FloatTensor [batch, time, feature_dim]

输出:
    logits: FloatTensor [batch, class_count, time]

实现说明:
    双向 GRU 同时读取过去和未来帧，适合不受实时约束的离线分割。
    本版本加入输入归一化、残差投影和时序卷积平滑头，比此前 smoke-test 版本更接近可训练 baseline。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    family = "BiGRU full temporal baseline"

    def __init__(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.projection = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.temporal_head = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.projection(self.input_norm(x)))
        z, _ = self.gru(z)
        return self.temporal_head(z.transpose(1, 2))
