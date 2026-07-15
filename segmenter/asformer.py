"""
ASFormer 风格离线时序分割模型。

输入:
    x: FloatTensor [batch, time, feature_dim]

输出:
    logits: FloatTensor [batch, class_count, time]

实现说明:
    ASFormer 的核心是用注意力在完整视频序列中建立长程依赖，再结合局部时序卷积稳定边界。
    这里实现为“局部膨胀卷积 + 多头自注意力 + 前馈网络”的多层编码器，
    不是只用一层轻量 Transformer 的 smoke-test 版本。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """生成正弦位置编码，让 attention 能区分帧顺序。"""

    pos = torch.arange(length, device=device).float().unsqueeze(1)
    idx = torch.arange(dim, device=device).float().unsqueeze(0)
    div = torch.exp(torch.floor(idx / 2) * (-math.log(10000.0) / max(dim, 1)))
    enc = pos * div
    out = torch.zeros(length, dim, device=device)
    out[:, 0::2] = torch.sin(enc[:, 0::2])
    out[:, 1::2] = torch.cos(enc[:, 1::2])
    return out


class TemporalAttentionBlock(nn.Module):
    """ASFormer 风格 block：局部卷积负责边界细节，attention 负责长程上下文。"""

    def __init__(self, hidden: int, heads: int, dilation: int, dropout: float):
        super().__init__()
        self.local = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=1,
        )
        self.local_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        local = self.local(x.transpose(1, 2)).transpose(1, 2)
        x = self.local_norm(x + self.dropout(torch.relu(local)))

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.dropout(attn_out))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class ASFormer(nn.Module):
    """多层时序 attention 分割模型。"""

    family = "ASFormer temporal attention baseline"

    def __init__(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.projection = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            TemporalAttentionBlock(hidden, heads, dilation=2 ** (i % 4), dropout=dropout)
            for i in range(layers)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        z = self.projection(self.input_norm(x))
        z = z + sinusoidal_position(time, z.shape[-1], x.device).unsqueeze(0)
        for block in self.blocks:
            z = block(z)
        return self.classifier(z).transpose(1, 2)


# 保留旧类名，避免注册表和历史脚本改动过大。
ASFormerLite = ASFormer
