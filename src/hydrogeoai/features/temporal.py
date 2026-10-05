"""Tabular temporal features (lags, rolling statistics, spells, calendar) for baseline models.

Every feature at day t uses only observations up to and including day t (causal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONSOON_MONTHS = (6, 7, 8, 9)


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    doy = dates.dt.dayofyear.to_numpy()
    return pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "monsoon": dates.dt.month.isin(MONSOON_MONTHS).astype(float).to_numpy(),
        "month": dates.dt.month.to_numpy(),
    }, index=dates.index)


def tabular_features(df: pd.DataFrame, lags=(1, 3, 7, 15, 30), windows=(3, 7, 15, 30),
                     variables=("precip", "tmax", "tmin")) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (features, groups). Groups: met (current values), temporal (lags/rolling/spells/calendar)."""
    groups: dict[str, list[str]] = {"met": [], "temporal": []}
    feats = {}
    for v in variables:
        col = f"{v}_qc" if f"{v}_qc" in df else v
        g = df.groupby("station_id", sort=False)[col]
        x = df[col]
        if v == "precip":
            x = np.log1p(x)
            g = np.log1p(df[col]).groupby(df.station_id, sort=False)
        feats[f"{v}_t0"] = x
        groups["met"].append(f"{v}_t0")
        for L in lags:
            feats[f"{v}_lag{L}"] = g.shift(L)
            groups["temporal"].append(f"{v}_lag{L}")
        for w in windows:
            r = g.rolling(w, min_periods=max(1, int(0.7 * w)))
            for stat in ("mean", "std", "max", "min"):
                if v != "precip" and stat in ("max", "min") and w < 7:
                    continue
                feats[f"{v}_roll{w}_{stat}"] = getattr(r, stat)().reset_index(level=0, drop=True)
                groups["temporal"].append(f"{v}_roll{w}_{stat}")
    for c in ("wet_spell_len", "dry_spell_len", "warm_spell_len"):
        if c in df:
            feats[c] = df[c]
            groups["temporal"].append(c)
    cal = calendar_features(df.date)
    for c in ("doy_sin", "doy_cos", "monsoon"):
        feats[c] = cal[c]
        groups["temporal"].append(c)
    out = pd.DataFrame(feats, index=df.index)
    return out, groups
