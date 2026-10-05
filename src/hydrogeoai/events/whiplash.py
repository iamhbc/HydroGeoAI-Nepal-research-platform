"""Hydroclimatic Whiplash Engine (Section 18).

Definition (anomaly-based, so the climatological monsoon onset is NOT counted as whiplash):
    A *dry-to-wet whiplash* occurs when a dry episode (SPI-n <= dry_threshold) is followed by a wet
    episode (SPI-n >= wet_threshold) whose onset lies within `max_transition_days` of the dry episode's
    end. *Wet-to-dry* is the mirror case. SPI is fitted per calendar month on the reference period.

Measured per event: duration, intensity (SPI swing = peak wet SPI - trough dry SPI), transition time,
transition speed (swing / transition days) and spatial extent (fraction of reporting stations that
experience a same-direction event within +/- `extent_window` days).

Trend analysis is hypothesis-neutral: decadal comparisons use non-parametric tests with Benjamini-
Hochberg correction, and results are reported whether or not they are significant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..qc.homogeneity import benjamini_hochberg, mann_kendall, sens_slope


def _episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def detect_station(dates: pd.Series, spi_vals: np.ndarray, dry: float = -1.0, wet: float = 1.0,
                   max_transition: int = 30, min_separation: int = 30) -> pd.DataFrame:
    s = np.asarray(spi_vals, float)
    dry_eps = _episodes(np.nan_to_num(s, nan=0) <= dry)
    wet_eps = _episodes(np.nan_to_num(s, nan=0) >= wet)
    dts = pd.to_datetime(dates).to_numpy()
    rows = []
    for direction, first, second in (("dry_to_wet", dry_eps, wet_eps), ("wet_to_dry", wet_eps, dry_eps)):
        starts2 = np.array([a for a, _ in second]) if second else np.array([], int)
        last_end = -10 ** 9
        for a1, b1 in first:
            nxt = np.flatnonzero((starts2 > b1) & (starts2 - b1 <= max_transition))
            if not len(nxt):
                continue
            a2, b2 = second[nxt[0]]
            gap = s[b1 + 1: a2]
            if (gap.size and np.isnan(gap).mean() > 0.5) or a2 - last_end < min_separation:
                continue
            seg1, seg2 = s[a1: b1 + 1], s[a2: b2 + 1]
            ext1 = np.nanmin(seg1) if direction == "dry_to_wet" else np.nanmax(seg1)
            ext2 = np.nanmax(seg2) if direction == "dry_to_wet" else np.nanmin(seg2)
            i1 = a1 + int(np.nanargmin(seg1) if direction == "dry_to_wet" else np.nanargmax(seg1))
            i2 = a2 + int(np.nanargmax(seg2) if direction == "dry_to_wet" else np.nanargmin(seg2))
            swing = abs(ext2 - ext1)
            rows.append({
                "direction": direction, "start": dts[a1], "transition_date": dts[a2], "end": dts[b2],
                "duration_days": int(b2 - a1 + 1), "transition_days": int(a2 - b1),
                "intensity_spi_swing": float(swing), "first_extreme": float(ext1), "second_extreme": float(ext2),
                "speed_spi_per_day": float(swing / max(i2 - i1, 1)),
            })
            last_end = b2
    return pd.DataFrame(rows)


def detect(labeled: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    w = cfg.get("whiplash", cfg)
    col = f"spi{w.get('spi_window_days', 30)}"
    out = []
    for sid, g in labeled.groupby("station_id"):
        ev = detect_station(g.date, g[col].to_numpy(), w.get("dry_threshold", -1.0), w.get("wet_threshold", 1.0),
                            w.get("max_transition_days", 30), w.get("min_separation_days", 30))
        if len(ev):
            out.append(ev.assign(station_id=sid))
    if not out:
        return pd.DataFrame(columns=["station_id", "direction", "start", "transition_date", "end",
                                     "duration_days", "transition_days", "intensity_spi_swing", "speed_spi_per_day"])
    ev = pd.concat(out, ignore_index=True)
    return add_spatial_extent(ev, labeled, col)


def add_spatial_extent(ev: pd.DataFrame, labeled: pd.DataFrame, spi_col: str, window: int = 15) -> pd.DataFrame:
    reporting = labeled[labeled[spi_col].notna()].groupby("date").station_id.nunique()
    extents = []
    for r in ev.itertuples():
        same = ev[(ev.direction == r.direction) &
                  ((ev.transition_date - r.transition_date).abs() <= pd.Timedelta(days=window))]
        n_rep = reporting.get(r.transition_date, np.nan)
        extents.append(same.station_id.nunique() / n_rep if n_rep and n_rep > 0 else np.nan)
    return ev.assign(spatial_extent=extents)


def station_years(labeled: pd.DataFrame, spi_col: str = "spi30", min_valid: float = 0.8) -> pd.DataFrame:
    v = labeled.assign(year=labeled.date.dt.year).groupby(["station_id", "year"])[spi_col].apply(
        lambda s: s.notna().mean())
    return v[v >= min_valid].reset_index()[["station_id", "year"]]


def annual_metrics(ev: pd.DataFrame, labeled: pd.DataFrame, spi_col: str = "spi30") -> pd.DataFrame:
    """Per-year network-level frequency (events per station-year), intensity and extent."""
    sy = station_years(labeled, spi_col)
    n_sy = sy.groupby("year").size().rename("station_years")
    e = ev.assign(year=pd.to_datetime(ev.transition_date).dt.year)
    e = e.merge(sy, on=["station_id", "year"])  # only count events in adequately observed station-years
    agg = e.groupby("year").agg(n_events=("direction", "size"), mean_intensity=("intensity_spi_swing", "mean"),
                                mean_speed=("speed_spi_per_day", "mean"), mean_extent=("spatial_extent", "mean"),
                                mean_transition_days=("transition_days", "mean"))
    out = pd.concat([n_sy, agg], axis=1).fillna({"n_events": 0})
    out["frequency_per_station_year"] = out.n_events / out.station_years
    return out.reset_index().rename(columns={"index": "year"})


def decadal_comparison(ev: pd.DataFrame, labeled: pd.DataFrame, decades: list[list[int]],
                       spi_col: str = "spi30", n_perm: int = 2000, seed: int = 0) -> dict:
    """Compare whiplash characteristics across periods without presupposing a trend."""
    sy = station_years(labeled, spi_col)
    e = ev.assign(year=pd.to_datetime(ev.transition_date).dt.year).merge(sy, on=["station_id", "year"])
    rows, rates = [], {}
    for d0, d1 in decades:
        lbl = f"{d0}-{d1}"
        syd = sy[(sy.year >= d0) & (sy.year <= d1)]
        ed = e[(e.year >= d0) & (e.year <= d1)]
        per_station = (ed.groupby("station_id").size() / syd.groupby("station_id").size()).fillna(0)
        per_station = per_station.reindex(syd.station_id.unique()).fillna(0)
        rates[lbl] = per_station
        rows.append({"period": lbl, "station_years": len(syd), "n_events": len(ed),
                     "events_per_station_year": len(ed) / max(len(syd), 1),
                     "mean_intensity": ed.intensity_spi_swing.mean(), "mean_speed": ed.speed_spi_per_day.mean(),
                     "mean_extent": ed.spatial_extent.mean(), "mean_transition_days": ed.transition_days.mean(),
                     "dry_to_wet": int((ed.direction == "dry_to_wet").sum()),
                     "wet_to_dry": int((ed.direction == "wet_to_dry").sum())})
    table = pd.DataFrame(rows)
    groups = [r.values for r in rates.values() if len(r) > 2]
    kw = stats.kruskal(*groups) if len(groups) >= 2 else None
    # permutation test on first vs last period mean intensity
    first, last = decades[0], decades[-1]
    a = e[(e.year >= first[0]) & (e.year <= first[1])].intensity_spi_swing.to_numpy()
    b = e[(e.year >= last[0]) & (e.year <= last[1])].intensity_spi_swing.to_numpy()
    perm_p = np.nan
    if len(a) > 5 and len(b) > 5:
        rng = np.random.default_rng(seed)
        obs_d = b.mean() - a.mean()
        pool = np.r_[a, b]
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(pool)
            cnt += abs(pool[len(a):].mean() - pool[: len(a)].mean()) >= abs(obs_d)
        perm_p = (cnt + 1) / (n_perm + 1)
    return {"table": table,
            "kruskal_frequency": {"H": float(kw.statistic), "p_value": float(kw.pvalue)} if kw else None,
            "intensity_first_vs_last_permutation_p": float(perm_p),
            "note": "Periods of unequal length/coverage; frequencies are normalised per station-year."}


def trend_tests(annual: pd.DataFrame, ev: pd.DataFrame, labeled: pd.DataFrame, spi_col: str = "spi30") -> dict:
    out = {}
    for col in ("frequency_per_station_year", "mean_intensity", "mean_speed", "mean_extent"):
        s = annual.dropna(subset=[col])
        if len(s) >= 10:
            mk = mann_kendall(s[col].to_numpy())
            slope, _ = sens_slope(s[col].to_numpy(), s.year.to_numpy())
            out[col] = {"mk_z": mk["z"], "p_value": mk["p_value"], "sens_slope_per_year": slope, "n_years": len(s)}
    # station-level frequency trends with FDR control (field significance)
    sy = station_years(labeled, spi_col)
    e = ev.assign(year=pd.to_datetime(ev.transition_date).dt.year)
    cnt = e.groupby(["station_id", "year"]).size().rename("n")
    st = sy.merge(cnt.reset_index(), on=["station_id", "year"], how="left").fillna({"n": 0})
    rows = []
    for sid, g in st.groupby("station_id"):
        if len(g) >= 15:
            mk = mann_kendall(g.n.to_numpy(), prewhiten=False)
            rows.append({"station_id": sid, "p_value": mk["p_value"], "sens_slope": mk["sens_slope"], "z": mk["z"]})
    sdf = pd.DataFrame(rows)
    if len(sdf):
        sdf["significant_fdr"] = benjamini_hochberg(sdf.p_value.to_numpy())
    out["station_trends"] = sdf
    out["n_stations_significant_fdr"] = int(sdf.significant_fdr.sum()) if len(sdf) else 0
    return out
