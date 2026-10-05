"""Failure analysis (Section 31): where and when is the model confidently wrong?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..gis.spatial import haversine_km


def quadrants(y, p, threshold, confident_mask) -> pd.Series:
    correct = (np.asarray(p) >= threshold).astype(int) == np.asarray(y).astype(int)
    conf = np.asarray(confident_mask, bool)
    lab = np.where(conf & correct, "confident_correct",
                   np.where(conf & ~correct, "confident_wrong",
                            np.where(~conf & correct, "uncertain_correct", "uncertain_wrong")))
    return pd.Series(lab)


def build_frame(y, p, threshold, uncertainty_cat, dates, station_ids, stations: pd.DataFrame,
                train_station_ids, window_obs_frac=None, event_types: pd.DataFrame | None = None) -> pd.DataFrame:
    st = stations.set_index("station_id")
    tr = st.loc[list(train_station_ids)]
    dist = {s: float(np.min(haversine_km(st.loc[s, "lat"], st.loc[s, "lon"], tr.lat.values, tr.lon.values)
                            + np.where(tr.index == s, np.inf, 0)))
            for s in np.unique(station_ids)}
    d = pd.DataFrame({"y": y, "p": p, "station_id": station_ids, "date": pd.to_datetime(dates),
                      "uncertainty": uncertainty_cat})
    d["predicted"] = (d.p >= threshold).astype(int)
    d["error"] = (d.predicted != d.y).astype(int)
    d["quadrant"] = quadrants(d.y, d.p, threshold, d.uncertainty == "low").values
    d["elevation_band"] = pd.cut(d.station_id.map(st.elevation_m), [-1, 500, 1500, 2500, 9000],
                                 labels=["<500 m", "500-1500 m", "1500-2500 m", ">2500 m"])
    d["season"] = d.date.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                       6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON"})
    d["basin"] = d.station_id.map(st.basin)
    d["nearest_train_station_km"] = d.station_id.map(dist)
    d["station_density"] = pd.cut(d.nearest_train_station_km, [-1, 1, 50, 100, 1e9],
                                  labels=["in-training", "<50 km", "50-100 km", ">100 km"])
    if window_obs_frac is not None:
        d["window_missing"] = pd.cut(1 - np.asarray(window_obs_frac), [-0.01, 0.0, 0.1, 0.3, 1.0],
                                     labels=["0%", "0-10%", "10-30%", ">30%"])
    if event_types is not None:
        for c in event_types:
            d[c] = event_types[c].values
    return d


def error_clustering(frame: pd.DataFrame, factors=("elevation_band", "season", "basin", "station_density",
                                                    "window_missing")) -> pd.DataFrame:
    """Error rates (overall, among events = miss rate, among non-events = false-alarm rate) per factor
    level, with a chi-square test of independence between error and factor."""
    rows = []
    for f in factors:
        if f not in frame:
            continue
        sub = frame.dropna(subset=[f])
        ct = pd.crosstab(sub[f], sub.error)
        chi = stats.chi2_contingency(ct) if ct.shape[0] > 1 and ct.shape[1] > 1 else None
        for lvl, g in sub.groupby(f, observed=True):
            ev = g[g.y == 1]
            ne = g[g.y == 0]
            rows.append({"factor": f, "level": str(lvl), "n": len(g), "n_events": int(len(ev)),
                         "error_rate": g.error.mean(), "miss_rate": ev.error.mean() if len(ev) else np.nan,
                         "false_alarm_rate": ne.error.mean() if len(ne) else np.nan,
                         "frac_confident_wrong": (g.quadrant == "confident_wrong").mean(),
                         "chi2_p_value": float(chi[1]) if chi else np.nan})
    return pd.DataFrame(rows)


def quadrant_table(frame: pd.DataFrame) -> pd.DataFrame:
    t = frame.groupby("quadrant").agg(n=("y", "size"), event_rate=("y", "mean"), mean_p=("p", "mean"))
    t["fraction"] = t.n / t.n.sum()
    return t.reset_index()
