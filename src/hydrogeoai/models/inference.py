"""Inference with a saved HydroGeoAI model directory (used by the API and the Hugging Face Space).

Every result carries model version, dataset version and parameters so that it is reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import ATTRIBUTION_DISCLAIMER
from ..features.temporal import calendar_features
from ..uncertainty import EPS, decompose, entropy_bits
from .data import TEMPORAL_CHANNELS
from .encoder import EncoderConfig, HydroclimaticEncoder
from .multimodal import HydroGeoAIModel


class Predictor:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        enc_cfg = json.loads((self.dir / "encoder" / "config.json").read_text())
        ecfg = EncoderConfig(**{k: v for k, v in enc_cfg.items() if k in EncoderConfig.__dataclass_fields__})
        self.members = []
        for i in range(self.cfg["n_members"]):
            m = HydroGeoAIModel(len(self.cfg["channels"]), self.cfg["static_dims"], d_model=self.cfg["d_model"],
                                dropout=self.cfg["dropout"], encoder=HydroclimaticEncoder(ecfg))
            m.load_state_dict(torch.load(self.dir / f"member_{i}.pt", map_location="cpu", weights_only=True))
            m.eval()
            self.members.append(m)
        self.version = self.cfg["version"]

    # ---------------------------------------------------------------------------------------------
    def _arrays(self, obs: pd.DataFrame, stations: pd.DataFrame, station_ids: list[str]):
        n = self.cfg["normalization"]
        dates = pd.date_range(obs.date.min(), obs.date.max(), freq="D")
        full = (pd.MultiIndex.from_product([station_ids, dates], names=["station_id", "date"]).to_frame(index=False)
                .merge(obs, on=["station_id", "date"], how="left"))
        S, T = len(station_ids), len(dates)
        vals = np.stack([np.log1p(full.precip_qc.clip(lower=0).to_numpy()), full.tmax_qc.to_numpy(),
                         full.tmin_qc.to_numpy()], -1).reshape(S, T, 3)
        mask = np.isfinite(vals).astype(np.float32)
        vals = np.nan_to_num((vals - np.array(n["dynamic"]["mean"])) / np.array(n["dynamic"]["std"]), nan=0.0)
        cal = calendar_features(pd.Series(dates))[TEMPORAL_CHANNELS].to_numpy()
        dyn = np.concatenate([vals, mask, np.broadcast_to(cal[None], (S, T, 3))], -1).astype(np.float32)
        all_names = ["precip_log1p", "tmax", "tmin", "precip_obs", "tmax_obs", "tmin_obs"] + TEMPORAL_CHANNELS
        dyn = np.ascontiguousarray(dyn[..., [all_names.index(c) for c in self.cfg["channels"]]])
        stx = stations.set_index("station_id").loc[station_ids]
        static = {}
        for g in self.cfg["static_groups"]:
            a = stx[self.cfg["static_features"][g]].to_numpy(dtype=float)
            static[g] = torch.tensor(np.nan_to_num((a - np.array(n[g]["mean"])) / np.array(n[g]["std"])),
                                     dtype=torch.float32)
        target = full[self.cfg["target"]].to_numpy().reshape(S, T) if self.cfg["target"] in full else None
        return torch.from_numpy(dyn).contiguous(), static, dates, target, mask[..., 0]

    def _forward(self, x, st, drop=None):
        probs, weights = [], []
        with torch.no_grad():
            for m in self.members:
                o = m(x, st, drop_modalities=drop)
                probs.append(torch.sigmoid(o["logit"]).numpy())
                weights.append(o["modality_weights"].numpy())
        return np.stack(probs), np.mean(weights, 0)

    def predict(self, obs: pd.DataFrame, stations: pd.DataFrame, station_ids: list[str], start: str, end: str,
                calibrated: bool = True) -> pd.DataFrame:
        W, lead = self.cfg["window_days"], self.cfg["lead_days"]
        dyn, static, dates, target, pmask = self._arrays(obs, stations, station_ids)
        t_idx = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
        t_idx = t_idx[t_idx >= W - 1]
        rows = []
        offs = torch.arange(-W + 1, 1)
        for si, sid in enumerate(station_ids):
            for b in range(0, len(t_idx), 2048):
                tt = torch.as_tensor(t_idx[b: b + 2048])
                x = dyn[si][tt[:, None] + offs[None, :]]
                st = {g: v[si].expand(len(tt), -1) for g, v in static.items()}
                mp, w = self._forward(x, st)
                if calibrated:
                    T = self.cfg.get("temperature", 1.0)
                    z = np.log(np.clip(mp, EPS, 1 - EPS) / (1 - np.clip(mp, EPS, 1 - EPS)))
                    mp = 1 / (1 + np.exp(-z / T))
                d = decompose(mp)
                q = self.cfg["conformal"]["q"]
                set0, set1 = d["probability"] <= q["0"], (1 - d["probability"]) <= q["1"]
                q50, q90 = self.cfg["val_epistemic"]
                cat = np.where(d["epistemic"] > q50, "moderate", "low").astype(object)
                cat[(d["epistemic"] > q90) | (~set0 & ~set1)] = "high"   # see uncertainty.categorise
                for k, t in enumerate(tt.numpy()):
                    obs_frac = float(pmask[si, max(0, t - W + 1): t + 1].mean())
                    rows.append({
                        "station_id": sid, "issued": dates[t].date().isoformat(),
                        "target_date": (dates[t] + pd.Timedelta(days=lead)).date().isoformat(),
                        "probability": float(d["probability"][k]), "ensemble_std": float(d["std"][k]),
                        "confidence": float(d["confidence"][k]), "epistemic": float(d["epistemic"][k]),
                        "conformal_set": "{0,1}" if set0[k] and set1[k] else ("{1}" if set1[k] else ("{0}" if set0[k] else "{}")),
                        "uncertainty": cat[k], "predicted_event": int(d["probability"][k] >= self.cfg["decision_threshold"]),
                        "observed_event": (None if target is None or t + lead - 1 >= target.shape[1] or
                                           not np.isfinite(target[si, t]) else int(target[si, t])),
                        "window_observed_fraction": obs_frac,
                        **{f"weight_{n}": float(v) for n, v in zip(["met"] + self.cfg["static_groups"], w[k])},
                    })
        return pd.DataFrame(rows)

    def explain(self, obs, stations, station_id: str, issued: str, steps: int = 32) -> dict:
        W = self.cfg["window_days"]
        dyn, static, dates, _, _ = self._arrays(obs, stations, [station_id])
        t = int(np.flatnonzero(dates == pd.Timestamp(issued))[0])
        x = dyn[0, t - W + 1: t + 1][None]
        st = {g: v[:1] for g, v in static.items()}
        m = self.members[0]
        gx = torch.zeros_like(x)
        gs = {g: torch.zeros_like(v) for g, v in st.items()}
        for a in torch.linspace(0, 1, steps):
            xi = (a * x).requires_grad_(True)
            si = {g: (a * v).requires_grad_(True) for g, v in st.items()}
            out = m(xi, si)["logit"].sum()
            grads = torch.autograd.grad(out, [xi, *si.values()])
            gx += grads[0] / steps
            for g, gr in zip(si, grads[1:]):
                gs[g] += gr / steps
        ax = (x * gx)[0].detach().numpy()
        ch = self.cfg["channels"]
        lag = {f"t-{i}": float(np.abs(ax[W - 1 - i]).sum()) for i in range(min(W, 30))}
        statics = {f"{g}:{n}": float((st[g] * gs[g])[0, j]) for g in st for j, n in enumerate(self.cfg["static_features"][g])}
        drop = {}
        base = self._forward(x, st)[0].mean()
        for g in self.cfg["static_groups"]:
            drop[g] = float(base - self._forward(x, st, drop=[g])[0].mean())
        top_static = dict(sorted(statics.items(), key=lambda kv: -abs(kv[1]))[:8])
        precip_obs = (ch.index("precip_obs") if "precip_obs" in ch else None)
        return {
            "station_id": station_id, "issued": issued, "probability": float(base),
            "channel_attribution": {c: float(ax[:, i].sum()) for i, c in enumerate(ch)},
            "lag_attribution_abs": lag, "top_static_attribution": top_static,
            "modality_ablation_delta_p": drop,
            "window_missing_fraction": float(1 - x[0, :, precip_obs].mean()) if precip_obs is not None else None,
            "entropy_bits": float(entropy_bits(base)),
            "disclaimer": ATTRIBUTION_DISCLAIMER, "model_version": self.version,
        }
