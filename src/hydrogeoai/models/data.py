"""Model-ready arrays: dynamic windows, static modality features, tabular features, targets.

Normalisation statistics are fitted on TRAINING stations and TRAINING period only.
Dynamic inputs use raw (globally standardised) values, NOT station-climatology anomalies, so that the
model must use geospatial context to interpret values at unseen stations (this is what RQ4/H3 test).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..events.taxonomy import add_target
from ..experiments.splits import Split
from ..features.spatial import FEATURE_GROUPS
from ..features.temporal import calendar_features, tabular_features

VALUE_CHANNELS = ["precip_log1p", "tmax", "tmin"]
MASK_CHANNELS = ["precip_obs", "tmax_obs", "tmin_obs"]
TEMPORAL_CHANNELS = ["doy_sin", "doy_cos", "monsoon"]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x):
        return (x - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class SplitData:
    idx: np.ndarray          # (N, 2): station index, time index
    y: np.ndarray            # (N,)
    regime: np.ndarray       # (N,) 0 dry / 1 normal / 2 wet at target date (-1 unknown)
    dates: np.ndarray
    stations: np.ndarray


@dataclass
class ModelData:
    dyn: torch.Tensor                    # (S, T, C) float32, NaN-free
    channel_names: list[str]
    static: dict[str, torch.Tensor]      # group -> (S, F)
    static_names: dict[str, list[str]]
    station_ids: list[str]
    dates: pd.DatetimeIndex
    splits: dict[str, SplitData]
    window: int
    target: str
    tabular: pd.DataFrame | None = None  # rows aligned to (station, time) flat index
    tabular_groups: dict[str, list[str]] = field(default_factory=dict)
    norm: dict = field(default_factory=dict)
    station_meta: pd.DataFrame | None = None
    y_full: np.ndarray | None = None     # (S, T) target for every station-day (for persistence etc.)

    def dyn_channels(self, kind: str) -> list[int]:
        names = VALUE_CHANNELS + MASK_CHANNELS + (TEMPORAL_CHANNELS if kind == "met_temporal" else [])
        return [self.channel_names.index(n) for n in names]

    def batch(self, split: str, sel: np.ndarray, channels: list[int], static_groups: list[str],
              device="cpu", dyn: torch.Tensor | None = None):
        sd = self.splits[split]
        s = torch.as_tensor(sd.idx[sel, 0])
        t = torch.as_tensor(sd.idx[sel, 1])
        offs = torch.arange(-self.window + 1, 1)
        D = self.dyn if dyn is None else dyn
        x = D[s[:, None], t[:, None] + offs[None, :]][..., channels]
        st = {g: self.static[g][s] for g in static_groups}
        y = torch.as_tensor(sd.y[sel], dtype=torch.float32)
        r = torch.as_tensor(sd.regime[sel], dtype=torch.long)
        return x.to(device), {k: v.to(device) for k, v in st.items()}, y.to(device), r.to(device)

    def tabular_matrix(self, split: str, groups: list[str], static_groups: list[str]) -> tuple[np.ndarray, list[str]]:
        sd = self.splits[split]
        flat = sd.idx[:, 0] * len(self.dates) + sd.idx[:, 1]
        cols = [c for g in groups for c in self.tabular_groups[g]]
        X = self.tabular.iloc[flat][cols].to_numpy(dtype=np.float32)
        names = list(cols)
        for g in static_groups:
            X = np.hstack([X, self.static[g][sd.idx[:, 0]].numpy()])
            names += [f"{g}:{n}" for n in self.static_names[g]]
        return X, names


def _regime(spi):
    r = np.full(spi.shape, -1, dtype=np.int64)
    ok = np.isfinite(spi)
    r[ok] = 1
    r[ok & (spi <= -1)] = 0
    r[ok & (spi >= 1)] = 2
    return r


def build_model_data(obs: pd.DataFrame, stations: pd.DataFrame, split: Split, target: str = "extreme_wet_day",
                     window: int = 60, lead: int = 1, horizon: int = 1, lags=(1, 3, 7, 15, 30),
                     rolling=(3, 7, 15, 30), max_samples: dict | None = None, seed: int = 0,
                     with_tabular: bool = True, min_window_obs: float = 0.5,
                     label_fraction: float = 1.0) -> ModelData:
    rng = np.random.default_rng(seed)
    max_samples = max_samples or {}
    ids = sorted(stations.station_id)
    dates = pd.date_range(obs.date.min(), obs.date.max(), freq="D")
    full = (pd.MultiIndex.from_product([ids, dates], names=["station_id", "date"]).to_frame(index=False)
            .merge(obs, on=["station_id", "date"], how="left"))
    full = add_target(full, target, lead, horizon)
    S, T = len(ids), len(dates)

    # ---- dynamic channels -----------------------------------------------------------------------
    p = np.log1p(full["precip_qc"].clip(lower=0).to_numpy()).reshape(S, T)
    tx = full["tmax_qc"].to_numpy().reshape(S, T)
    tn = full["tmin_qc"].to_numpy().reshape(S, T)
    vals = np.stack([p, tx, tn], -1)
    sid_index = {s: i for i, s in enumerate(ids)}
    tr_s = [sid_index[s] for s in split.stations["train"]]
    t0, t1 = (dates.get_loc(pd.Timestamp(d)) for d in split.periods["train"])
    train_block = vals[tr_s, t0: t1 + 1].reshape(-1, 3)
    vnorm = Normalizer(np.nanmean(train_block, 0), np.nanstd(train_block, 0) + 1e-6)
    obs_mask = np.isfinite(vals).astype(np.float32)
    vals = np.nan_to_num(vnorm(vals), nan=0.0)
    cal = calendar_features(pd.Series(dates))
    temporal = np.broadcast_to(cal[TEMPORAL_CHANNELS].to_numpy()[None], (S, T, 3))
    dyn = np.ascontiguousarray(np.concatenate([vals, obs_mask, temporal], -1), dtype=np.float32)
    channel_names = VALUE_CHANNELS + MASK_CHANNELS + TEMPORAL_CHANNELS

    # ---- static modality features ---------------------------------------------------------------
    stx = stations.set_index("station_id").loc[ids]
    static, static_names, norm = {}, {}, {"dynamic": vnorm.to_dict()}
    for g, cols in FEATURE_GROUPS.items():
        a = stx[cols].to_numpy(dtype=float)
        tr = a[tr_s]
        mu, sd = np.nanmean(tr, 0), np.nanstd(tr, 0) + 1e-6
        mu, sd = np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-5), sd, 1.0)
        static[g] = torch.tensor(np.nan_to_num((a - mu) / sd), dtype=torch.float32)
        static_names[g] = cols
        norm[g] = {"mean": mu.tolist(), "std": sd.tolist()}

    # ---- samples per split ----------------------------------------------------------------------
    y_all = full["target"].to_numpy().reshape(S, T)
    spi_next = full.groupby("station_id", sort=False)["spi30"].shift(-lead).to_numpy().reshape(S, T) \
        if "spi30" in full else np.full((S, T), np.nan)
    reg_all = _regime(spi_next)
    obs_frac = pd.DataFrame(obs_mask[..., 0].T).rolling(window, min_periods=1).mean().to_numpy().T
    splits = {}
    for name in ("train", "val", "test"):
        si = np.array([sid_index[s] for s in split.stations[name]])
        a, b = (dates.get_loc(pd.Timestamp(d)) for d in split.periods[name])
        a = max(a, window - 1)
        ss, tt = np.meshgrid(si, np.arange(a, b + 1), indexing="ij")
        ss, tt = ss.ravel(), tt.ravel()
        ok = np.isfinite(y_all[ss, tt]) & (obs_frac[ss, tt] >= min_window_obs)
        ss, tt = ss[ok], tt[ok]
        if name == "train" and label_fraction < 1.0:
            keep = rng.random(len(ss)) < label_fraction
            ss, tt = ss[keep], tt[keep]
        cap = max_samples.get(name)
        if cap and len(ss) > cap:
            keep = np.sort(rng.choice(len(ss), cap, replace=False))
            ss, tt = ss[keep], tt[keep]
        splits[name] = SplitData(np.stack([ss, tt], 1), y_all[ss, tt].astype(np.float32), reg_all[ss, tt],
                                 dates.values[tt], np.array(ids, dtype=object)[ss])

    md = ModelData(torch.from_numpy(dyn), channel_names, static, static_names, ids, dates, splits, window,
                   target, norm=norm, station_meta=stx.reset_index(), y_full=y_all)
    if with_tabular:
        tab, groups = tabular_features(full, lags, rolling)
        md.tabular, md.tabular_groups = tab.reset_index(drop=True), groups
    return md
