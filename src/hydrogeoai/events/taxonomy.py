"""Hydroclimatic event taxonomy (Section 4): precipitation, dry, thermal and compound extremes.

All thresholds are station-specific and fitted on the reference period only (`fit_thresholds`).
Labels are NaN where the underlying observation is missing (never silently treated as "no event").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indices import spi

EVENT_COLUMNS = [
    "extreme_wet_day", "very_extreme_wet_day", "extreme_rainfall_event", "consecutive_wet_days",
    "rainfall_persistence", "dry_spell", "precipitation_deficit", "meteorological_drought",
    "hot_day", "warm_spell", "cold_night", "hot_dry", "wet_hot", "rain_after_dryness",
]

EVENT_GROUPS = {
    "precipitation": ["extreme_wet_day", "very_extreme_wet_day", "extreme_rainfall_event",
                      "consecutive_wet_days", "rainfall_persistence"],
    "dry": ["dry_spell", "precipitation_deficit", "meteorological_drought"],
    "thermal": ["hot_day", "warm_spell", "cold_night"],
    "compound": ["hot_dry", "wet_hot", "rain_after_dryness"],
}


def _col(obs, v):
    return f"{v}_qc" if f"{v}_qc" in obs else v


def run_length(mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Length of the current run of True ending at each position (NaN/invalid resets the run)."""
    out = np.zeros(len(mask), dtype=float)
    cur = 0
    for i, m in enumerate(mask):
        if valid is not None and not valid[i]:
            cur = 0
            out[i] = np.nan
            continue
        cur = cur + 1 if m else 0
        out[i] = cur
    return out


def _calendar_percentile(values: pd.Series, doy: np.ndarray, in_ref: np.ndarray, q: float,
                         half_window: int) -> np.ndarray:
    """Percentile for each calendar day using a +/- half_window-day moving window (ETCCDI style)."""
    v = values.to_numpy()
    thr = np.full(367, np.nan)
    ref_v, ref_d = v[in_ref], doy[in_ref]
    for d in range(1, 367):
        dist = np.minimum(np.abs(ref_d - d), 366 - np.abs(ref_d - d))
        sel = (dist <= half_window) & np.isfinite(ref_v)
        if sel.sum() > 20:
            thr[d] = np.percentile(ref_v[sel], q)
    return thr[doy]


def fit_thresholds(obs: pd.DataFrame, ref: tuple[str, str], cfg: dict) -> pd.DataFrame:
    """Station-level precipitation thresholds from the reference period."""
    wet_mm = cfg.get("wet_day_mm", 1.0)
    p = _col(obs, "precip")
    rows = []
    for sid, g in obs.groupby("station_id"):
        r = g[(g.date >= ref[0]) & (g.date <= ref[1])]
        wet = r[p][r[p] >= wet_mm].dropna()
        acc3 = r[p].rolling(3, min_periods=3).sum().dropna()
        rows.append({
            "station_id": sid, "n_ref_days": int(r[p].notna().sum()), "n_ref_wet_days": int(len(wet)),
            "p95_wet": float(np.percentile(wet, 95)) if len(wet) > 50 else np.nan,
            "p99_wet": float(np.percentile(wet, 99)) if len(wet) > 50 else np.nan,
            "p99_3day": float(np.percentile(acc3, 99)) if len(acc3) > 365 else np.nan,
        })
    return pd.DataFrame(rows)


def label_events(obs: pd.DataFrame, thresholds: pd.DataFrame, ref: tuple[str, str], cfg: dict) -> pd.DataFrame:
    """Return obs with event label columns, SPI-30/SPI-90 and spell lengths appended."""
    wet_mm = cfg.get("wet_day_mm", 1.0)
    pc, tx, tn = _col(obs, "precip"), _col(obs, "tmax"), _col(obs, "tmin")
    thr = thresholds.set_index("station_id")
    c_pr, c_dry, c_th, c_cp = cfg.get("precipitation", {}), cfg.get("dry", {}), cfg.get("thermal", {}), cfg.get("compound", {})
    parts = []
    for sid, g in obs.groupby("station_id", sort=True):
        g = g.sort_values("date").copy()
        P = g[pc]
        valid_p = P.notna().to_numpy()
        doy = g.date.dt.dayofyear.to_numpy()
        in_ref = ((g.date >= ref[0]) & (g.date <= ref[1])).to_numpy()
        t = thr.loc[sid] if sid in thr.index else None

        def lab(cond, valid):
            return np.where(valid, cond.astype(float), np.nan)

        if t is not None:
            g["extreme_wet_day"] = lab(P.to_numpy() > t.p95_wet, valid_p)
            g["very_extreme_wet_day"] = lab(P.to_numpy() > t.p99_wet, valid_p)
            acc = P.rolling(c_pr.get("extreme_rainfall_event", {}).get("window_days", 3), min_periods=3).sum()
            g["extreme_rainfall_event"] = lab(acc.to_numpy() > t.p99_3day, acc.notna().to_numpy())
        wet = (P >= wet_mm).to_numpy()
        g["wet_spell_len"] = run_length(wet, valid_p)
        g["dry_spell_len"] = run_length(~wet, valid_p)
        g["consecutive_wet_days"] = lab(g.wet_spell_len >= c_pr.get("consecutive_wet_days", {}).get("min_days", 5), valid_p)
        pers = c_pr.get("persistence", {"window_days": 10, "min_wet_days": 8})
        wc = pd.Series(np.where(valid_p, wet, np.nan)).rolling(pers["window_days"], min_periods=pers["window_days"]).sum()
        g["rainfall_persistence"] = lab(wc.to_numpy() >= pers["min_wet_days"], wc.notna().to_numpy())
        g["dry_spell"] = lab(g.dry_spell_len >= c_dry.get("dry_spell", {}).get("min_days", 15), valid_p)

        g["spi30"] = spi(P.reset_index(drop=True), g.date.reset_index(drop=True), 30, ref)
        g["spi90"] = spi(P.reset_index(drop=True), g.date.reset_index(drop=True), 90, ref)
        thr_def = c_dry.get("precipitation_deficit", {}).get("spi_threshold", -1.0)
        g["precipitation_deficit"] = lab(g.spi30 <= thr_def, g.spi30.notna().to_numpy())
        g["meteorological_drought"] = lab(g.spi90 <= c_dry.get("meteorological_drought", {}).get("spi_threshold", -1.0),
                                          g.spi90.notna().to_numpy())

        hd = c_th.get("hot_day", {"percentile": 90, "window_days": 15})
        cn = c_th.get("cold_night", {"percentile": 10, "window_days": 15})
        g["tx90_thr"] = _calendar_percentile(g[tx], doy, in_ref, hd["percentile"], hd["window_days"] // 2)
        g["tn10_thr"] = _calendar_percentile(g[tn], doy, in_ref, cn["percentile"], cn["window_days"] // 2)
        vt, vn = g[tx].notna().to_numpy(), g[tn].notna().to_numpy()
        hot = (g[tx] > g.tx90_thr).to_numpy()
        g["hot_day"] = lab(hot, vt)
        g["warm_spell_len"] = run_length(hot, vt)
        # label every day of a spell of >= min_days (forward and backward)
        k = c_th.get("warm_spell", {}).get("min_days", 3)
        hs = pd.Series(hot & vt).astype(int)
        grp = (hs != hs.shift()).cumsum()
        spell_len = hs.groupby(grp).transform("sum")
        g["warm_spell"] = lab(((hs == 1) & (spell_len >= k)).to_numpy(), vt)
        g["cold_night"] = lab((g[tn] < g.tn10_thr).to_numpy(), vn)

        if t is not None:
            dry_thr = c_cp.get("hot_dry", {}).get("spi_threshold", -1.0)
            g["hot_dry"] = lab(hot & (g.spi30 <= dry_thr).to_numpy(), vt & g.spi30.notna().to_numpy())
            g["wet_hot"] = lab((g.extreme_wet_day == 1).to_numpy() & hot, vt & valid_p)
            antecedent = g.dry_spell_len.shift(1).to_numpy()
            nd = c_cp.get("rain_after_dryness", {}).get("antecedent_dry_days", 20)
            g["rain_after_dryness"] = lab((g.extreme_wet_day == 1).to_numpy() & (antecedent >= nd),
                                          valid_p & np.isfinite(antecedent))
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_target(df: pd.DataFrame, event: str, lead: int = 1, horizon: int = 1) -> pd.DataFrame:
    """Target y(t) = 1 if `event` occurs in (t+lead-1, t+lead-1+horizon]; NaN if unknown.

    Features at row t may only use information up to day t.
    """
    out = []
    for _, g in df.groupby("station_id", sort=False):
        e = g[event]
        fut = [e.shift(-(lead + h)) for h in range(horizon)]
        F = pd.concat(fut, axis=1)
        y = F.max(axis=1, skipna=True)
        y[F.isna().all(axis=1)] = np.nan
        out.append(y)
    return df.assign(target=pd.concat(out).reindex(df.index))


def event_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EVENT_COLUMNS if c in labeled]
    s = labeled.groupby("station_id")[cols].mean().mul(100).round(3)
    return s
