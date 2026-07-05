"""
ASFormer-lite 简化版。

输入：
    x [batch, time, feature_dim]

输出：
    logits [batch, class_count, time]

说明：
    用 TransformerEncoder 表达 ASFormer 类模型的核心思想：每一帧通过
    self-attention 读取同一视频里的其它帧信息。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """生成正弦位置编码，让 Transformer 知道帧顺序。"""
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    idx = torch.arange(dim, device=device).float().unsqueeze(0)
    div = torch.exp(torch.floor(idx / 2) * (-math.log(10000.0) / max(dim, 1)))
    enc = pos * div
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2] = torch.sin(enc[:, 0::2])
    out[:, 1::2] = torch.cos(enc[:, 1::2])
    return out


class ASFormerLite(nn.Module):
    family = "ASFormer-lite transformer baseline"

    def __init__(self, in_dim: int, classes: int, hidden: int = 48, heads: int = 4, layers: int = 2):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 3,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        z = self.projection(x)
        z = z + sinusoidal_position(time, z.shape[-1], x.device).unsqueeze(0)
        z = self.encoder(z)
        return self.classifier(z).transpose(1, 2)

