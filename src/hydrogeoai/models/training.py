"""Training / inference loops for deep models, multimodal model and SSL pretraining."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from ..utils import get_logger
from .data import MASK_CHANNELS, VALUE_CHANNELS, ModelData
from .encoder import EncoderConfig, HydroclimaticEncoder, Pretrainer

log = get_logger(__name__)


def _out(o):
    return o if isinstance(o, dict) else {"logit": o}


def predict(model: nn.Module, md: ModelData, split: str, channels, static_groups, batch_size=1024,
            device="cpu", mc_dropout: bool = False, drop_modalities=None, dyn=None,
            return_extra: bool = False):
    model.eval()
    if mc_dropout:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    n = len(md.splits[split].y)
    probs, extras = [], {"modality_weights": [], "regime": []}
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sel = np.arange(i, min(i + batch_size, n))
            x, st, _, _ = md.batch(split, sel, channels, static_groups, device, dyn=dyn)
            kw = {"drop_modalities": drop_modalities} if drop_modalities else {}
            o = _out(model(x, st, **kw))
            probs.append(torch.sigmoid(o["logit"]).cpu().numpy())
            if return_extra and "modality_weights" in o:
                extras["modality_weights"].append(o["modality_weights"].cpu().numpy())
            if return_extra and "regime_logits" in o:
                extras["regime"].append(torch.softmax(o["regime_logits"], -1).cpu().numpy())
    p = np.concatenate(probs) if probs else np.array([])
    if return_extra:
        return p, {k: np.concatenate(v) if v else None for k, v in extras.items()}
    return p


def train_classifier(model: nn.Module, md: ModelData, channels, static_groups, cfg: dict, seed: int = 0,
                     device="cpu", regime_weight: float = 0.2, encoder_lr_scale: float = 1.0,
                     freeze_encoder: bool = False) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    params = model.parameters()
    if hasattr(model, "encoder") and (encoder_lr_scale != 1.0 or freeze_encoder):
        enc = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        rest = [p for p in model.parameters() if id(p) not in enc_ids]
        if freeze_encoder:
            for p in enc:
                p.requires_grad_(False)
            params = [{"params": rest}]
        else:
            params = [{"params": rest}, {"params": enc, "lr": cfg["lr"] * encoder_lr_scale}]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    tr = md.splits["train"]
    n = len(tr.y)
    best, best_score, bad, hist = None, -np.inf, 0, []
    bs = cfg.get("batch_size", 256)
    for ep in range(cfg.get("epochs", 10)):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            sel = perm[i: i + bs]
            x, st, y, r = md.batch("train", sel, channels, static_groups, device)
            o = _out(model(x, st))
            loss = bce(o["logit"], y)
            if "regime_logits" in o and regime_weight > 0 and (r >= 0).any():
                loss = loss + regime_weight * ce(o["regime_logits"], r)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sel)
        pv = predict(model, md, "val", channels, static_groups, device=device)
        yv = md.splits["val"].y
        score = average_precision_score(yv, pv) if yv.sum() > 0 else -np.mean((pv - yv) ** 2)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_auprc": float(score), "sec": time.time() - t0})
        log.debug("epoch %d loss %.4f val AUPRC %.4f", ep, tot / n, score)
        if score > best_score + 1e-4:
            best_score, best, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.get("patience", 3):
                break
    if best is not None:
        model.load_state_dict(best)
    return model, {"history": hist, "best_val_auprc": float(best_score)}


def pretrain_encoder(md: ModelData, channels, cfg: dict, seed: int = 0, device="cpu") -> tuple[HydroclimaticEncoder, dict]:
    """Self-supervised pretraining on training-split windows only (labels are not used)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = [md.channel_names[c] for c in channels]
    v_idx = [names.index(c) for c in VALUE_CHANNELS]
    m_idx = [names.index(c) for c in MASK_CHANNELS]
    enc = HydroclimaticEncoder(EncoderConfig(n_in=len(channels), n_values=len(v_idx), d_model=cfg.get("d_model", 64),
                                             n_layers=cfg.get("n_layers", 3), n_heads=cfg.get("n_heads", 4),
                                             max_len=max(366, md.window)))
    pt = Pretrainer(enc, v_idx, m_idx, cfg.get("contrastive_weight", 0.1), cfg.get("multiscale_weight", 0.2)).to(device)
    opt = torch.optim.AdamW(pt.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4)
    n = len(md.splits["train"].y)
    n_use = min(n, cfg.get("max_samples", n))
    bs = cfg.get("batch_size", 256)
    hist = []
    for ep in range(cfg.get("epochs", 5)):
        pt.train()
        sel_all = rng.permutation(n)[:n_use]
        agg: dict[str, float] = {}
        for i in range(0, n_use, bs):
            sel = sel_all[i: i + bs]
            x, _, _, _ = md.batch("train", sel, channels, [], device)
            out = pt(x, rng, cfg.get("mask_ratio", 0.3), cfg.get("mean_span", 5))
            opt.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_norm_(pt.parameters(), 1.0)
            opt.step()
            for k, v in out.items():
                agg[k] = agg.get(k, 0.0) + v.item() * len(sel)
        hist.append({"epoch": ep, **{k: v / n_use for k, v in agg.items()}})
        log.info("pretrain epoch %d: %s", ep, {k: round(v, 4) for k, v in hist[-1].items() if k != "epoch"})
    # held-out reconstruction error on validation windows (does SSL generalise?)
    pt.eval()
    with torch.no_grad():
        nv = min(len(md.splits["val"].y), 4096)
        x, _, _, _ = md.batch("val", np.arange(nv), channels, [], device)
        val = pt(x, np.random.default_rng(seed + 1), cfg.get("mask_ratio", 0.3), cfg.get("mean_span", 5))
    return enc.cpu(), {"history": hist, "val_recon_mse": float(val["recon"]),
                       # reference: predicting the (standardised) training mean everywhere
                       "val_naive_mse_zero": float((x[..., v_idx] ** 2).mean())}
