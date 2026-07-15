"""
MS-TCN + BiLSTM 离线时序分割模型。

输入:
    x: FloatTensor [batch, time, feature_dim]

输出:
    logits: FloatTensor [batch, class_count, time]

实现说明:
    1. 先用线性层把多维检测特征投影到 hidden 维；
    2. BiLSTM 双向读取完整序列，补足离线模型需要的前后文；
    3. 多个 TCN stage 使用膨胀卷积扩大时间感受野，并逐 stage 细化逐帧分类结果。

这里保留 MS-TCN 的“多阶段逐帧 refinement”思想，并显式加入 BiLSTM。
训练和推理接口仍与其它模型一致，方便 run_pipeline.py 统一调用。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DilatedResidualLayer(nn.Module):
    """单个膨胀残差层，在时间轴聚合局部上下文。"""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dilated(x)
        out = self.act(self.norm(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return self.act(x + out)


class SingleStageTCN(nn.Module):
    """一个 MS-TCN stage：输入时序特征，输出逐帧 logits。"""

    def __init__(
        self,
        in_channels: int,
        classes: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(hidden, dilation=2 ** i, dropout=dropout)
            for i in range(layers)
        )
        self.classifier = nn.Conv1d(hidden, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x)
        for layer in self.layers:
            z = layer(z)
        return self.classifier(z)


class MSTCNBiLSTM(nn.Module):
    """BiLSTM 编码 + 多阶段 TCN refinement 的离线分割模型。"""

    family = "MS-TCN + BiLSTM full baseline"

    def __init__(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 64,
        lstm_layers: int = 2,
        tcn_layers: int = 6,
        stages: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_projection = nn.Linear(in_dim, hidden)
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_projection = nn.Conv1d(hidden * 2, hidden, kernel_size=1)
        self.first_stage = SingleStageTCN(hidden, classes, hidden, tcn_layers, dropout)
        self.refine_stages = nn.ModuleList(
            SingleStageTCN(classes, classes, hidden, tcn_layers, dropout)
            for _ in range(max(0, stages - 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = self.input_norm(x)
        z = torch.relu(self.input_projection(z))
        z, _ = self.bilstm(z)       # [B, T, 2H]，双向使用完整序列上下文
        z = self.lstm_projection(z.transpose(1, 2))

        logits = self.first_stage(z)
        for stage in self.refine_stages:
            # 后续 stage 消费上一 stage 的逐帧类别分布，逐步修正边界。
            logits = stage(torch.softmax(logits, dim=1))
        return logits


# 保持旧注册名兼容 run_pipeline.py / 已保存报告里的 ms_tcn。
MSTCN = MSTCNBiLSTM
