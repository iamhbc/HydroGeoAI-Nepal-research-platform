"""Deep sequence baselines: LSTM, GRU, temporal CNN (TCN), Transformer.

All share the interface  forward(x_dyn (B,L,C), static: dict[str, (B,F)]) -> logits (B,)
so they can be compared on identical inputs.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class StaticFusionHead(nn.Module):
    def __init__(self, d_seq: int, static_dims: dict[str, int], hidden: int, dropout: float):
        super().__init__()
        self.groups = list(static_dims)
        d_static = sum(static_dims.values())
        self.static_mlp = nn.Sequential(nn.Linear(d_static, hidden), nn.GELU()) if d_static else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_seq + (hidden if d_static else 0), hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h, static):
        if self.static_mlp is not None:
            h = torch.cat([h, self.static_mlp(torch.cat([static[g] for g in self.groups], -1))], -1)
        return self.head(h).squeeze(-1)


class RNNModel(nn.Module):
    def __init__(self, n_in, static_dims, hidden=64, dropout=0.2, cell="lstm", layers=2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = StaticFusionHead(hidden, static_dims, hidden, dropout)

    def forward(self, x, static):
        out, _ = self.rnn(x)
        return self.head(out[:, -1], static)


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return torch.relu(self.drop(torch.relu(y)) + self.res(x))


class TCNModel(nn.Module):
    def __init__(self, n_in, static_dims, hidden=64, dropout=0.2, k=3, levels=5):
        super().__init__()
        layers, c = [], n_in
        for i in range(levels):                       # receptive field = 1 + 2*(2^levels - 1) = 63 days
            layers.append(CausalConv(c, hidden, k, 2 ** i, dropout))
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = StaticFusionHead(hidden, static_dims, hidden, dropout)

    def forward(self, x, static):
        return self.head(self.tcn(x.transpose(1, 2))[:, :, -1], static)


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_in, static_dims, hidden=64, dropout=0.2, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_in, hidden)
        self.pe = SinusoidalPE(hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = StaticFusionHead(2 * hidden, static_dims, hidden, dropout)

    def forward(self, x, static):
        h = self.enc(self.pe(self.proj(x)))
        return self.head(torch.cat([h[:, -1], h.mean(1)], -1), static)


def build_deep(name: str, n_in: int, static_dims: dict[str, int], hidden=64, dropout=0.2) -> nn.Module:
    if name in ("lstm", "gru"):
        return RNNModel(n_in, static_dims, hidden, dropout, cell=name)
    if name == "tcn":
        return TCNModel(n_in, static_dims, hidden, dropout)
    if name == "transformer":
        return TransformerModel(n_in, static_dims, hidden, dropout)
    raise ValueError(name)
