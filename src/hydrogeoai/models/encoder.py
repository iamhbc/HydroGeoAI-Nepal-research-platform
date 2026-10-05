"""Hydroclimatic Encoder: self-supervised representation learning for daily station time series.

Pretraining objectives (Section 11):
1. Masked span reconstruction: contiguous spans (~30% of days) are hidden; the encoder reconstructs
   the standardised values, loss only on positions that were masked AND originally observed.
   Masked positions look like missing data, so the encoder also learns gap-robust representations.
2. Temporal contrastive learning (NT-Xent): two stochastic views (different masks + jitter) of the
   same window should map close together; other windows in the batch are negatives.
3. Multi-scale targets: predict 7-day and 30-day means at the end of the window from the pooled
   representation (weekly/monthly structure).

Leakage rule: pretraining windows come only from the training stations and training period of the
split being evaluated (no transductive use of test stations).

Artifacts follow the Hugging Face layout: config.json + model.safetensors (or pytorch_model.bin).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class EncoderConfig:
    n_in: int = 9
    n_values: int = 3
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    max_len: int = 366
    model_type: str = "hydroclimatic-encoder"
    version: str = "1.0"


class HydroclimaticEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.config = cfg
        self.proj = nn.Linear(cfg.n_in, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 4 * cfg.d_model, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pool = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, L, n_in) -> (token states (B, L, d), pooled representation (B, d))."""
        L = x.size(1)
        h = self.proj(x) + self.pos(torch.arange(L, device=x.device))[None]
        h = self.norm(self.enc(h))
        return h, self.pool(torch.cat([h[:, -1], h.mean(1)], -1))

    # ---- Hugging Face style persistence ---------------------------------------------------------
    def save_pretrained(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({**asdict(self.config), **(extra or {})}, indent=2))
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(state, str(path / "model.safetensors"))
        except ImportError:
            torch.save(state, path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HydroclimaticEncoder":
        path = Path(path)
        cfgd = json.loads((path / "config.json").read_text())
        cfg = EncoderConfig(**{k: v for k, v in cfgd.items() if k in EncoderConfig.__dataclass_fields__})
        m = cls(cfg)
        if (path / "model.safetensors").exists():
            from safetensors.torch import load_file
            m.load_state_dict(load_file(str(path / "model.safetensors")))
        else:
            m.load_state_dict(torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True))
        return m


def span_mask(B: int, L: int, ratio: float, mean_span: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean (B, L) mask of contiguous spans covering ~ratio of positions."""
    m = np.zeros((B, L), bool)
    target = int(round(ratio * L))
    for b in range(B):
        while m[b].sum() < target:
            ln = min(int(rng.geometric(1 / mean_span)), L)
            st = int(rng.integers(0, L - ln + 1))
            m[b, st: st + ln] = True
    return m


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    z = nn.functional.normalize(torch.cat([z1, z2]), dim=-1)
    sim = z @ z.T / tau
    n = z1.size(0)
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return nn.functional.cross_entropy(sim, targets)


class Pretrainer(nn.Module):
    def __init__(self, encoder: HydroclimaticEncoder, value_idx: list[int], mask_idx: list[int],
                 contrastive_weight=0.1, multiscale_weight=0.2, scales=(7, 30)):
        super().__init__()
        d = encoder.config.d_model
        nv = len(value_idx)
        self.encoder, self.value_idx, self.mask_idx = encoder, value_idx, mask_idx
        self.recon = nn.Linear(d, nv)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ms = nn.Linear(d, nv * len(scales))
        self.cw, self.mw, self.scales = contrastive_weight, multiscale_weight, scales

    def _corrupt(self, x, m, jitter=0.05):
        x = x.clone()
        mt = torch.as_tensor(m, device=x.device)
        for vi, mi in zip(self.value_idx, self.mask_idx):
            x[..., vi] = torch.where(mt, torch.zeros_like(x[..., vi]), x[..., vi] + jitter * torch.randn_like(x[..., vi]))
            x[..., mi] = torch.where(mt, torch.zeros_like(x[..., mi]), x[..., mi])
        return x

    def forward(self, x: torch.Tensor, rng: np.random.Generator, ratio=0.3, mean_span=5.0) -> dict:
        B, L, _ = x.shape
        vals = x[..., self.value_idx]
        observed = x[..., self.mask_idx] > 0.5
        m1, m2 = span_mask(B, L, ratio, mean_span, rng), span_mask(B, L, ratio, mean_span, rng)
        h1, p1 = self.encoder(self._corrupt(x, m1))
        rec = self.recon(h1)
        mm = torch.as_tensor(m1, device=x.device)[..., None] & observed
        l_rec = ((rec - vals) ** 2 * mm).sum() / mm.sum().clamp(min=1)
        out = {"recon": l_rec}
        loss = l_rec
        if self.cw > 0:
            _, p2 = self.encoder(self._corrupt(x, m2))
            l_con = nt_xent(self.proj(p1), self.proj(p2))
            out["contrastive"] = l_con
            loss = loss + self.cw * l_con
        if self.mw > 0:
            tgt, wts = [], []
            for s in self.scales:
                ob = observed[:, -s:].float()
                tgt.append((vals[:, -s:] * ob).sum(1) / ob.sum(1).clamp(min=1))
                wts.append((ob.sum(1) > 0).float())
            tgt, wts = torch.cat(tgt, -1), torch.cat(wts, -1)
            l_ms = (((self.ms(p1) - tgt) ** 2) * wts).sum() / wts.sum().clamp(min=1)
            out["multiscale"] = l_ms
            loss = loss + self.mw * l_ms
        out["loss"] = loss
        return out
