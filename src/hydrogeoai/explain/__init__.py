"""Explainability (Section 17).

IMPORTANT: every output of this module describes what the *model* relies on (attribution), which is
not physical causation. See hydrogeoai.ATTRIBUTION_DISCLAIMER; it is attached to every result.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .. import ATTRIBUTION_DISCLAIMER
from ..models.data import ModelData
from ..models.training import _out, predict


def integrated_gradients(model, md: ModelData, split: str, sel: np.ndarray, channels, static_groups,
                         steps: int = 32, device="cpu") -> dict:
    """Integrated gradients (Sundararajan et al. 2017) w.r.t. dynamic and static inputs.

    Baseline = all-zeros input, i.e. the training mean for standardised channels with every
    observation marked missing. Returns mean |attribution| per channel, per lag, per static feature.
    """
    model.eval()
    x, st, _, _ = md.batch(split, sel, channels, static_groups, device)
    alphas = torch.linspace(0, 1, steps, device=device)
    gx = torch.zeros_like(x)
    gs = {g: torch.zeros_like(v) for g, v in st.items()}
    for a in alphas:
        xi = (a * x).requires_grad_(True)
        si = {g: (a * v).requires_grad_(True) for g, v in st.items()}
        logit = _out(model(xi, si))["logit"].sum()
        grads = torch.autograd.grad(logit, [xi, *si.values()])
        gx += grads[0] / steps
        for g, gr in zip(si, grads[1:]):
            gs[g] += gr / steps
    attr_x = (x * gx).detach().abs().cpu().numpy()       # (B, L, C)
    names = [md.channel_names[c] for c in channels]
    out = {"disclaimer": ATTRIBUTION_DISCLAIMER,
           "channel_importance": dict(zip(names, attr_x.sum(1).mean(0).round(5).tolist())),
           "lag_importance": attr_x.sum(2).mean(0)[::-1].round(5).tolist(),   # index 0 = day t
           "static_importance": {}}
    for g in static_groups:
        a = (st[g] * gs[g]).detach().abs().cpu().numpy().mean(0)
        out["static_importance"].update({f"{g}:{n}": float(v) for n, v in zip(md.static_names[g], a)})
    return out


def channel_permutation_importance(model, md: ModelData, split: str, channels, static_groups, seed=0,
                                   device="cpu") -> dict:
    """Drop in AUPRC when one dynamic channel is shuffled across stations/time (model reliance)."""
    rng = np.random.default_rng(seed)
    y = md.splits[split].y
    base = average_precision_score(y, predict(model, md, split, channels, static_groups, device=device))
    res = {}
    for c in channels:
        dyn = md.dyn.clone()
        flat = dyn[..., c].reshape(-1)
        dyn[..., c] = flat[torch.as_tensor(rng.permutation(flat.numel()))].reshape(dyn.shape[:2])
        p = predict(model, md, split, channels, static_groups, device=device, dyn=dyn)
        res[md.channel_names[c]] = float(base - average_precision_score(y, p))
    return {"disclaimer": ATTRIBUTION_DISCLAIMER, "baseline_auprc": float(base), "auprc_drop": res}


def modality_ablation(model, md: ModelData, split: str, channels, static_groups, device="cpu") -> dict:
    """Remove one modality at inference time (model trained with modality dropout)."""
    y = md.splits[split].y
    base = average_precision_score(y, predict(model, md, split, channels, static_groups, device=device))
    res = {}
    for g in static_groups:
        p = predict(model, md, split, channels, static_groups, device=device, drop_modalities=[g])
        res[g] = float(base - average_precision_score(y, p))
    return {"disclaimer": ATTRIBUTION_DISCLAIMER, "baseline_auprc": float(base), "auprc_drop_without": res}


def counterfactual_precip(model, md: ModelData, split: str, sel, channels, static_groups,
                          scale_days: int = 7, factors=(0.5, 1.0, 1.5, 2.0), device="cpu") -> dict:
    """What-if: rescale antecedent precipitation over the last `scale_days` days (in mm space)."""
    pi = md.channel_names.index("precip_log1p")
    mu, sd = md.norm["dynamic"]["mean"][0], md.norm["dynamic"]["std"][0]
    res = {}
    for f in factors:
        dyn = md.dyn.clone()
        v = dyn[..., pi] * sd + mu                       # log1p(mm)
        mm = torch.expm1(v).clamp(min=0) * f
        new = (torch.log1p(mm) - mu) / sd
        T = dyn.shape[1]
        sd_idx = md.splits[split].idx[sel]
        for s, t in sd_idx:
            lo = max(0, t - scale_days + 1)
            dyn[s, lo: t + 1, pi] = torch.where(dyn[s, lo: t + 1, pi + 3] > 0.5, new[s, lo: t + 1], dyn[s, lo: t + 1, pi])
        _ = T
        sub = md.splits[split]
        p = predict(model, _SubsetView(md, split, sel), split, channels, static_groups, device=device, dyn=dyn)
        res[str(f)] = float(np.mean(p))
        _ = sub
    return {"disclaimer": ATTRIBUTION_DISCLAIMER, "mean_probability_by_factor": res, "days_scaled": scale_days}


class _SubsetView:
    """Lightweight view of ModelData restricted to selected rows of one split."""

    def __init__(self, md: ModelData, split: str, sel):
        from ..models.data import SplitData
        self._md = md
        sd = md.splits[split]
        self.splits = {split: SplitData(sd.idx[sel], sd.y[sel], sd.regime[sel], sd.dates[sel], sd.stations[sel])}

    def __getattr__(self, k):
        return getattr(self._md, k)

    def batch(self, split, sel, channels, static_groups, device="cpu", dyn=None):
        from ..models.data import ModelData as _MD
        return _MD.batch(self, split, sel, channels, static_groups, device, dyn)
