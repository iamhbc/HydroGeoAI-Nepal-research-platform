"""HydroGeoAI multimodal model (Section 12).

    meteorological window --> Hydroclimatic Encoder (pretrained or scratch) --> token m
    GIS / terrain features --> Geospatial encoder (MLP)                     --> token g
    remote-sensing summary --> Remote-sensing encoder (MLP)                 --> token r
    station metadata       --> Metadata encoder (MLP)                       --> token s
    [m, g, r, s] + modality-type embeddings --> cross-modal attention layer --> gated pooling
        --> heads: extreme-event logit | regime (dry/normal/wet SPI-30 class at t+lead)
Uncertainty is estimated around the model (deep ensembles, MC dropout, conformal prediction).

Modality dropout randomly removes non-meteorological modalities during training so the model
degrades gracefully when a modality is unavailable; the gate weights are reported as *model
attention*, not as physical importance.
"""
from __future__ import annotations

import torch
from torch import nn

from .encoder import EncoderConfig, HydroclimaticEncoder


class HydroGeoAIModel(nn.Module):
    def __init__(self, n_dyn: int, static_dims: dict[str, int], d_model: int = 64, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.2, modality_dropout: float = 0.1,
                 encoder: HydroclimaticEncoder | None = None, regime_head: bool = True):
        super().__init__()
        self.encoder = encoder or HydroclimaticEncoder(EncoderConfig(n_in=n_dyn, d_model=d_model,
                                                                     n_layers=n_layers, n_heads=n_heads))
        d = self.encoder.config.d_model
        self.groups = list(static_dims)
        self.static_enc = nn.ModuleDict({
            g: nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
            for g, n in static_dims.items()})
        self.type_emb = nn.Embedding(1 + len(self.groups), d)
        if self.groups:
            self.fusion = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.gate = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)
        self.event_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.regime_head = nn.Linear(d, 3) if regime_head else None
        self.modality_dropout = modality_dropout

    def forward(self, x: torch.Tensor, static: dict[str, torch.Tensor],
                drop_modalities: list[str] | None = None) -> dict:
        _, m = self.encoder(x)
        tokens = [m + self.type_emb.weight[0]]
        keep = [torch.ones(x.size(0), device=x.device)]
        for i, g in enumerate(self.groups):
            t = self.static_enc[g](static[g]) + self.type_emb.weight[i + 1]
            k = torch.ones(x.size(0), device=x.device)
            if self.training and self.modality_dropout > 0:
                k = (torch.rand(x.size(0), device=x.device) > self.modality_dropout).float()
            if drop_modalities and g in drop_modalities:
                k = torch.zeros(x.size(0), device=x.device)
            tokens.append(t * k[:, None])
            keep.append(k)
        H = torch.stack(tokens, 1)
        K = torch.stack(keep, 1)
        if self.groups:
            H = self.fusion(H, src_key_padding_mask=(K < 0.5))
        logits = self.gate(H).squeeze(-1).masked_fill(K < 0.5, -1e9)
        w = torch.softmax(logits, -1)
        fused = self.drop((w[..., None] * H).sum(1))
        out = {"logit": self.event_head(fused).squeeze(-1), "modality_weights": w, "rep": fused}
        if self.regime_head is not None:
            out["regime_logits"] = self.regime_head(fused)
        return out
