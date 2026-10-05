"""Standardised climate indices.

Why these indices (Section 9 asks to justify, not just compute):
* SPI (McKee et al. 1993) needs only precipitation, which is the best-observed variable in Nepal's
  station network; it is multi-scalar (30 d ~ soil-moisture/agricultural deficits, 90 d ~ meteorological
  drought) and removes seasonality, which is essential under a monsoon regime where dry pre-monsoon
  months are climatologically normal.
* SPEI requires potential evapotranspiration. With only Tmax/Tmin available we can use Hargreaves PET,
  which is acceptable for relative anomalies but uncertain at high elevation; SPEI is therefore
  provided as an *optional* index and should be reported with this caveat.
* ETCCDI indices (Zhang et al. 2011) are the community standard for station-based extremes, enabling
  comparison with published Himalayan studies.
Distribution parameters are fitted on the reference period only (no leakage from evaluation years).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _thom_gamma(x: np.ndarray) -> tuple[float, float]:
    """Thom (1958) maximum-likelihood approximation for gamma shape/scale (standard in SPI)."""
    x = x[x > 0]
    if len(x) < 5:
        return np.nan, np.nan
    m = x.mean()
    A = np.log(m) - np.mean(np.log(x))
    if A <= 0:
        return np.nan, np.nan
    a = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
    return a, m / a


def spi(precip: pd.Series, dates: pd.Series, window: int, ref: tuple[str, str],
        min_frac: float = 0.9) -> np.ndarray:
    """Daily-resolution SPI on a rolling `window`-day accumulation, fitted per calendar month."""
    acc = precip.rolling(window, min_periods=int(window * min_frac)).sum()
    acc = acc * window / precip.notna().rolling(window, min_periods=1).sum()   # completeness-scale
    months = dates.dt.month.to_numpy()
    in_ref = ((dates >= ref[0]) & (dates <= ref[1])).to_numpy()
    out = np.full(len(acc), np.nan)
    a_vals = acc.to_numpy()
    for m in range(1, 13):
        sel = months == m
        fit = a_vals[sel & in_ref]
        fit = fit[np.isfinite(fit)]
        if len(fit) < 30:
            continue
        q = np.mean(fit <= 0)
        shape, scale = _thom_gamma(fit)
        if not np.isfinite(shape):
            continue
        v = a_vals[sel]
        H = q + (1 - q) * stats.gamma.cdf(np.where(v > 0, v, 0), shape, scale=scale)
        H = np.where(v <= 0, q / 2 if q > 0 else 1e-3, H)
        out[sel] = stats.norm.ppf(np.clip(H, 1e-4, 1 - 1e-4))
        out[sel & ~np.isfinite(a_vals)] = np.nan
    return out


def hargreaves_pet(tmax, tmin, doy, lat_deg) -> np.ndarray:
    """Hargreaves-Samani PET (mm/day) using extraterrestrial radiation from latitude & day-of-year."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    tmean = (tmax + tmin) / 2
    return 0.0023 * 0.408 * Ra * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))


def spei(precip: pd.Series, pet: pd.Series, dates: pd.Series, window: int, ref: tuple[str, str]) -> np.ndarray:
    """SPEI with a log-logistic (Fisk) distribution fitted per calendar month (Vicente-Serrano 2010)."""
    bal = (precip - pet).rolling(window, min_periods=int(0.9 * window)).sum()
    months = dates.dt.month.to_numpy()
    in_ref = ((dates >= ref[0]) & (dates <= ref[1])).to_numpy()
    out = np.full(len(bal), np.nan)
    b = bal.to_numpy()
    for m in range(1, 13):
        sel = months == m
        fit = b[sel & in_ref]
        fit = fit[np.isfinite(fit)]
        if len(fit) < 30:
            continue
        c, loc, scale = stats.fisk.fit(fit)
        out[sel] = stats.norm.ppf(np.clip(stats.fisk.cdf(b[sel], c, loc, scale), 1e-4, 1 - 1e-4))
    return out


def _max_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def etccdi_annual(obs: pd.DataFrame, thresholds: pd.DataFrame | None = None, wet_mm: float = 1.0,
                  min_completeness: float = 0.8) -> pd.DataFrame:
    """Annual ETCCDI-style indices per station.

    PRCPTOT, SDII, R10mm, R20mm, Rx1day, Rx5day, CDD, CWD, R95pTOT (needs thresholds),
    TXx, TNn, TX90p/TN10p (fraction of days, needs thresholds via `hot_day`/`cold_night` labels if present).
    """
    p = "precip_qc" if "precip_qc" in obs else "precip"
    tx = "tmax_qc" if "tmax_qc" in obs else "tmax"
    tn = "tmin_qc" if "tmin_qc" in obs else "tmin"
    rows = []
    o = obs.assign(year=obs.date.dt.year)
    thr = thresholds.set_index("station_id") if thresholds is not None else None
    for (sid, yr), g in o.groupby(["station_id", "year"]):
        pr = g[p].to_numpy()
        comp = np.isfinite(pr).mean()
        if comp < min_completeness:
            continue
        wet = pr >= wet_mm
        r = {"station_id": sid, "year": yr, "completeness": comp,
             "PRCPTOT": np.nansum(pr[wet]), "SDII": np.nanmean(pr[wet]) if wet.any() else 0.0,
             "R10mm": int(np.nansum(pr >= 10)), "R20mm": int(np.nansum(pr >= 20)),
             "Rx1day": np.nanmax(pr), "Rx5day": np.nanmax(pd.Series(pr).rolling(5).sum()),
             "CDD": _max_run(np.nan_to_num(pr, nan=np.inf) < wet_mm), "CWD": _max_run(wet),
             "TXx": np.nanmax(g[tx]) if g[tx].notna().any() else np.nan,
             "TNn": np.nanmin(g[tn]) if g[tn].notna().any() else np.nan}
        if thr is not None and sid in thr.index:
            r["R95pTOT"] = float(np.nansum(pr[pr > thr.loc[sid, "p95_wet"]]))
        if "hot_day" in g:
            r["TX90p"] = float(g.hot_day.mean() * 100)
        if "cold_night" in g:
            r["TN10p"] = float(g.cold_night.mean() * 100)
        rows.append(r)
    return pd.DataFrame(rows)
