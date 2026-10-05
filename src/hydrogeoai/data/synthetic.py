"""Synthetic Nepal-like hydroclimatic dataset for pipeline development and testing.

WARNING — SYNTHETIC DATA. Station names refer to approximate real locations purely to give the
generator realistic geography (elevation gradients, monsoon, rain shadows). All values are simulated.
They must never be used for scientific conclusions about Nepal's climate. Replace with real
observations (DHM Nepal, APHRODITE, CHIRPS, ERA5-Land, ...) via `hydrogeoai.data.ingest`.

Design (so the benchmark is non-trivial and the QC/homogeneity modules have ground truth):
* spatially correlated, temporally persistent weather via a Gaussian copula with national, basin
  and local AR(1) latent fields (gives realistic wet/dry spells and regional coherence);
* orographic precipitation climatology, monsoon seasonality, western-disturbance winter rain;
* interannual monsoon variability and modest trends in intensity/dry-season occurrence;
* lapse-rate temperature, elevation-dependent warming, rain-temperature coupling;
* heterogeneous record lengths, missing blocks, injected errors and inhomogeneities, all recorded
  in `ground_truth.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..gis.raster import GeoRaster

# (name, lat, lon, elevation_m, approx mean annual precip mm, rain_shadow)
_STATIONS = [
    ("Mahendranagar", 28.96, 80.18, 176, 1650, 0), ("Dhangadhi", 28.69, 80.59, 180, 1750, 0),
    ("Dadeldhura", 29.30, 80.58, 1848, 1350, 0), ("Baitadi", 29.55, 80.42, 1600, 1250, 0),
    ("Darchula", 29.83, 80.57, 1100, 2000, 0),
    ("Chisapani", 28.64, 81.27, 225, 2100, 0), ("Surkhet", 28.60, 81.62, 720, 1650, 0),
    ("Dailekh", 28.85, 81.72, 1400, 1700, 0), ("Jumla", 29.28, 82.17, 2300, 800, 1),
    ("Simikot", 29.97, 81.83, 2800, 850, 1), ("Dunai", 28.93, 82.90, 2050, 500, 1),
    ("Nepalgunj", 28.06, 81.62, 165, 1400, 0), ("Ghorahi", 28.05, 82.48, 700, 1550, 0),
    ("Tulsipur", 28.13, 82.30, 725, 1500, 0),
    ("Pokhara", 28.21, 83.98, 827, 3900, 0), ("Lumle", 28.30, 83.80, 1740, 5000, 0),
    ("Jomsom", 28.78, 83.72, 2744, 280, 1), ("Gorkha", 28.00, 84.62, 1100, 1800, 0),
    ("Bharatpur", 27.68, 84.43, 205, 2000, 0), ("Bhairahawa", 27.51, 83.42, 109, 1650, 0),
    ("Kathmandu", 27.70, 85.36, 1337, 1500, 0), ("Godavari", 27.59, 85.38, 1400, 1900, 0),
    ("Nagarkot", 27.70, 85.52, 2150, 1850, 0), ("Hetauda", 27.42, 85.03, 474, 2300, 0),
    ("Dhankuta", 26.98, 87.35, 1200, 1000, 0), ("Biratnagar", 26.48, 87.26, 72, 1850, 0),
    ("Okhaldhunga", 27.31, 86.50, 1720, 1750, 0), ("Taplejung", 27.35, 87.67, 1730, 2000, 0),
    ("Namche", 27.80, 86.71, 3450, 950, 1), ("Dharan", 26.81, 87.28, 444, 2400, 0),
]

NEPAL_OUTLINE = [
    (80.06, 28.84), (80.5, 28.6), (81.2, 28.35), (81.6, 28.0), (81.9, 27.9), (82.7, 27.5),
    (83.3, 27.35), (84.1, 27.35), (84.7, 27.05), (85.2, 26.85), (85.9, 26.6), (86.5, 26.45),
    (87.3, 26.35), (88.1, 26.4), (88.2, 26.9), (88.0, 27.4), (88.15, 27.85), (87.8, 27.9),
    (87.0, 27.95), (86.4, 28.0), (86.0, 28.1), (85.6, 28.3), (85.1, 28.5), (84.6, 28.6),
    (84.1, 28.9), (83.5, 29.2), (83.0, 29.4), (82.5, 29.7), (82.0, 30.0), (81.5, 30.4),
    (81.0, 30.2), (80.6, 29.9), (80.3, 29.5), (80.06, 29.1), (80.06, 28.84),
]

RIVERS = {
    "Mahakali": [(80.9, 30.2), (80.6, 29.6), (80.4, 29.2), (80.2, 28.9)],
    "Karnali": [(81.9, 30.2), (81.6, 29.6), (81.5, 29.0), (81.3, 28.65), (81.1, 28.4)],
    "West Rapti": [(82.9, 28.3), (82.3, 28.15), (81.7, 28.0), (81.6, 27.95)],
    "Kali Gandaki-Narayani": [(83.9, 29.1), (83.7, 28.7), (83.6, 28.1), (84.1, 27.75), (84.3, 27.4)],
    "Bagmati": [(85.4, 27.75), (85.3, 27.6), (85.5, 27.1), (85.5, 26.8)],
    "Koshi": [(87.3, 27.9), (87.2, 27.3), (87.15, 26.9), (87.0, 26.5)],
}

# Basin regions as half-plane clips of the outline: (name, lon_min, lon_max, lat_min, lat_max)
_BASIN_BOXES = [
    ("Mahakali", 79.0, 80.95, 25.0, 32.0),
    ("Karnali", 80.95, 81.3, 25.0, 32.0),
    ("Karnali", 81.3, 83.2, 28.35, 32.0),
    ("West Rapti", 81.3, 83.2, 25.0, 28.35),
    ("Gandaki", 83.2, 85.0, 25.0, 32.0),
    ("Bagmati", 85.0, 85.75, 25.0, 32.0),
    ("Koshi", 85.75, 89.0, 25.0, 32.0),
]
PROVINCE_BY_BASIN = {"Mahakali": "Sudurpashchim", "Karnali": "Karnali", "West Rapti": "Lumbini",
                     "Gandaki": "Gandaki", "Bagmati": "Bagmati", "Koshi": "Koshi"}


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------
def _clip_halfplane(poly, a, b, c):
    """Sutherland–Hodgman: keep a*x + b*y + c >= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _clip_box(poly, x0, x1, y0, y1):
    for a, b, c in [(1, 0, -x0), (-1, 0, x1), (0, 1, -y0), (0, -1, y1)]:
        poly = _clip_halfplane(poly, a, b, c)
        if not poly:
            break
    return poly


def basins_geojson() -> dict:
    outline = NEPAL_OUTLINE[:-1]
    parts: dict[str, list] = {}
    for name, x0, x1, y0, y1 in _BASIN_BOXES:
        ring = _clip_box(outline, x0, x1, y0, y1)
        if len(ring) >= 3:
            ring = [(round(x, 4), round(y, 4)) for x, y in ring]
            parts.setdefault(name, []).append([ring + [ring[0]]])
    feats = []
    for name, polys in parts.items():
        geom = ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
                else {"type": "MultiPolygon", "coordinates": polys})
        feats.append({"type": "Feature", "properties": {"basin": name, "province": PROVINCE_BY_BASIN[name]},
                      "geometry": geom})
    return {"type": "FeatureCollection", "name": "basins_synthetic",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": feats}


def rivers_geojson() -> dict:
    feats = [{"type": "Feature", "properties": {"river": k},
              "geometry": {"type": "LineString", "coordinates": v}} for k, v in RIVERS.items()]
    return {"type": "FeatureCollection", "name": "rivers_synthetic",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": feats}


def outline_geojson() -> dict:
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [{"type": "Feature", "properties": {"name": "Nepal (simplified)"},
                          "geometry": {"type": "Polygon", "coordinates": [NEPAL_OUTLINE]}}]}


def _border_lat(lon: np.ndarray, which: str) -> np.ndarray:
    pts = np.array(NEPAL_OUTLINE[:-1])
    seg = pts[:14] if which == "south" else pts[16:34]
    order = np.argsort(seg[:, 0])
    return np.interp(lon, seg[order, 0], seg[order, 1])


def synthetic_dem(resolution: float = 0.05, seed: int = 0, stations: pd.DataFrame | None = None) -> GeoRaster:
    """Physiographic-zone DEM (Terai → Siwalik → Middle hills → High Himal → Trans-Himalaya)."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = 80.0, 88.3, 26.3, 30.5
    cols, rows = int(round((x1 - x0) / resolution)), int(round((y1 - y0) / resolution))
    xs = x0 + resolution * (np.arange(cols) + 0.5)
    ys = y1 - resolution * (np.arange(rows) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    south, north = _border_lat(X, "south"), _border_lat(X, "north")
    t = np.clip((Y - south) / np.maximum(north - south, 0.3), -0.2, 1.3)
    prof = np.interp(t, [-0.2, 0.0, 0.12, 0.2, 0.27, 0.45, 0.6, 0.72, 0.82, 0.95, 1.3],
                     [60, 90, 250, 1100, 700, 1700, 2600, 4800, 6200, 4800, 4600])
    noise = np.zeros_like(X)
    for k, amp in [(2, 450), (5, 300), (11, 180), (23, 90)]:
        ph = rng.uniform(0, 2 * np.pi, 4)
        noise += amp * np.sin(k * X + ph[0]) * np.cos(k * 1.3 * Y + ph[1])
    dem = np.maximum(prof + noise * np.clip(t + 0.2, 0, 1), 30.0)
    if stations is not None:  # honour station elevations locally (kernel correction)
        r = GeoRaster(dem, "EPSG:4326", (x0, resolution, 0.0, y1, 0.0, -resolution))
        resid = stations.elevation_m.values - r.sample(stations.lon.values, stations.lat.values, "EPSG:4326")
        for (lat, lon), d in zip(stations[["lat", "lon"]].values, resid):
            w = np.exp(-(((X - lon) ** 2 + (Y - lat) ** 2) / (2 * 0.06 ** 2)))
            dem += w * d
        dem = np.maximum(dem, 30.0)
    return GeoRaster(dem.astype("float32"), "EPSG:4326", (x0, resolution, 0.0, y1, 0.0, -resolution),
                     source="synthetic:hydrogeoai.data.synthetic", name="dem", units="m",
                     history=["synthetic physiographic DEM"])


# --------------------------------------------------------------------------------------------
# Weather generator
# --------------------------------------------------------------------------------------------
def _ar1(rng, n, phi, size=None):
    shape = (n,) if size is None else (n, size)
    e = rng.standard_normal(shape) * np.sqrt(1 - phi ** 2)
    x = np.empty(shape)
    x[0] = rng.standard_normal(shape[1:] if size else ())
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _monthly_weights(lon: float, elev: float) -> np.ndarray:
    base = np.array([1.2, 1.6, 2.0, 3.0, 5.5, 14.0, 25.0, 23.0, 14.5, 4.0, 0.8, 1.0])
    west = np.clip((83.5 - lon) / 3.5, 0, 1)                  # western disturbances (winter rain)
    base = base + west * np.array([4.0, 4.5, 3.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5, 2.5]) * (1 + elev / 3000)
    east = np.clip((lon - 85.5) / 2.5, 0, 1)                  # pre-monsoon thunderstorms in the east
    base = base + east * np.array([0, 0, 1.5, 3.5, 4.0, 0, 0, 0, 0, 0, 0, 0])
    return base / base.sum()


def generate(cfg: dict | None = None) -> dict:
    """Generate the synthetic dataset. Returns dict of DataFrames/objects (nothing written)."""
    cfg = cfg or {}
    seed = cfg.get("seed", 42)
    n_st = min(cfg.get("n_stations", 30), len(_STATIONS))
    rng = np.random.default_rng(seed)
    dates = pd.date_range(cfg.get("start", "1990-01-01"), cfg.get("end", "2024-12-31"), freq="D")
    T = len(dates)

    st = pd.DataFrame(_STATIONS[:n_st], columns=["name", "lat", "lon", "elevation_m", "map_mm", "rain_shadow"])
    st.insert(0, "station_id", [f"NP{idx:04d}" for idx in range(1, n_st + 1)])
    from ..gis.spatial import spatial_join_points_polygons
    basins = basins_geojson()
    st["basin"] = spatial_join_points_polygons(st, basins, "basin").fillna("Unknown").values
    st["province"] = st.basin.map(PROVINCE_BY_BASIN).fillna("Unknown")
    S = n_st
    basin_codes, basin_idx = np.unique(st.basin.to_numpy(dtype=object).astype(str), return_inverse=True)

    years = dates.year.values
    month = dates.month.values - 1
    doy = dates.dayofyear.values
    yrs = np.arange(years.min(), years.max() + 1)

    # ---- latent weather fields (Gaussian copula) --------------------------------------------
    g_nat = _ar1(rng, T, 0.65)
    g_bas = _ar1(rng, T, 0.6, len(basin_codes))
    g_loc = _ar1(rng, T, 0.35, S)
    z = np.sqrt(0.35) * g_nat[:, None] + np.sqrt(0.35) * g_bas[:, basin_idx] + np.sqrt(0.30) * g_loc
    u = stats.norm.cdf(z)

    # interannual monsoon strength (national + basin), trend in intensity & dry-season occurrence
    mon_nat = _ar1(rng, len(yrs), 0.3)
    mon_bas = _ar1(rng, len(yrs), 0.3, len(basin_codes))
    yi = years - yrs.min()
    monsoon_factor = np.exp(0.18 * (0.7 * mon_nat[yi, None] + 0.5 * mon_bas[yi][:, basin_idx]))
    is_monsoon = np.isin(month, [5, 6, 7, 8])[:, None]
    yr_c = (years - 2005)[:, None]
    intensity_trend = 1 + 0.006 * yr_c
    dry_occ_trend = np.where(is_monsoon, 1.0, 1 - 0.006 * yr_c)

    precip = np.zeros((T, S))
    for s in range(S):
        w = _monthly_weights(st.lon[s], st.elevation_m[s])
        mean_daily = st.map_mm[s] * w[month] / (365.25 / 12)
        mean_daily = mean_daily * np.where(is_monsoon[:, 0], monsoon_factor[:, s], 1.0)
        p_wet = np.clip(0.04 + 0.86 * (1 - np.exp(-mean_daily / 7.0)), 0.02, 0.92) * dry_occ_trend[:, 0]
        k = 0.65 + 0.15 * (1 - st.rain_shadow[s])
        mean_wet = mean_daily / p_wet
        scale = mean_wet / k * intensity_trend[:, 0]
        wet = u[:, s] > (1 - p_wet)
        q = np.clip((u[:, s] - (1 - p_wet)) / p_wet, 1e-6, 1 - 1e-6)
        amt = stats.gamma.ppf(q, k, scale=scale)
        precip[:, s] = np.where(wet, np.round(np.maximum(amt, 0.1), 1), 0.0)

    # ---- temperature ------------------------------------------------------------------------
    elev = st.elevation_m.values / 1000.0
    lat = st.lat.values
    t_mean = 25.5 - 6.0 * elev - 0.6 * (lat - 27.5)
    amp = 6.5 + 1.2 * elev + 0.8 * (lat - 27.5)
    peak = 165 + 15 * (elev > 2.5)
    clim = t_mean[None, :] + amp[None, :] * np.cos(2 * np.pi * (doy[:, None] - peak[None, :]) / 365.25)
    monsoon_bump = np.exp(-0.5 * ((doy - 210) / 40.0) ** 2)[:, None]
    dtr = 12.0 - 5.0 * monsoon_bump + 0.6 * elev[None, :]
    a_nat, a_bas, a_loc = _ar1(rng, T, 0.75), _ar1(rng, T, 0.7, len(basin_codes)), _ar1(rng, T, 0.5, S)
    anom = 1.9 * (np.sqrt(0.5) * a_nat[:, None] + np.sqrt(0.3) * a_bas[:, basin_idx] + np.sqrt(0.2) * a_loc)
    wet_flag = (precip > 1.0).astype(float)
    dryness = np.clip(-z, 0, None) * (1 - monsoon_bump)          # pre-monsoon heat under dry spells
    warming = (years - 1990)[:, None] * (0.025 * (1 + 0.15 * elev))[None, :]
    tmax = clim + dtr / 2 + anom - 1.4 * wet_flag * (1 + q_term(u)) + 0.6 * dryness + warming
    tmin = clim - dtr / 2 + 0.8 * anom + 0.5 * wet_flag + (years - 1990)[:, None] * 0.03
    tmax, tmin = np.round(tmax, 1), np.round(np.minimum(tmin, tmax - 0.5), 1)
    rh = np.clip(np.round(50 + 35 * monsoon_bump + 12 * wet_flag - 0.6 * anom + rng.normal(0, 4, (T, S))), 5, 100)

    obs = pd.DataFrame({
        "station_id": np.repeat(st.station_id.to_numpy(dtype=object), T),
        "date": np.tile(dates.values, S),
        "precip": precip.ravel(order="F"), "tmax": tmax.ravel(order="F"),
        "tmin": tmin.ravel(order="F"), "rh": rh.ravel(order="F"),
    })
    truth: dict = {"warning": "SYNTHETIC DATA", "seed": seed, "missing_blocks": 0, "errors": [],
                   "inhomogeneities": [], "metadata_errors": [],
                   "trends": {"wet_intensity_per_year": 0.006, "dry_season_occurrence_per_year": -0.006,
                              "tmax_warming_c_per_year": "0.025*(1+0.15*elev_km)",
                              "tmin_warming_c_per_year": 0.03}}

    # rh only available at ~70% stations (heterogeneous variable availability)
    no_rh = rng.choice(st.station_id.to_numpy(dtype=object), size=int(0.3 * S), replace=False)
    obs.loc[obs.station_id.isin(no_rh), "rh"] = np.nan

    # ---- heterogeneous record lengths + missing blocks ----------------------------------------
    for s, sid in enumerate(st.station_id):
        start_year = 2000 if st.elevation_m[s] > 2700 else rng.choice([1990, 1990, 1990, 1993, 1997])
        obs.loc[(obs.station_id == sid) & (obs.date.dt.year < start_year), ["precip", "tmax", "tmin", "rh"]] = np.nan
    rate = cfg.get("missing_block_rate", 0.002)
    vals = obs[["precip", "tmax", "tmin", "rh"]].to_numpy(dtype=float, copy=True)
    for s in range(S):
        base = s * T
        starts = np.flatnonzero(rng.random(T) < rate)
        for t0 in starts:
            L = int(rng.geometric(1 / 15))
            cols = [[0, 1, 2, 3], [0], [1, 2]][rng.choice(3, p=[0.6, 0.2, 0.2])]
            vals[base + t0: base + min(T, t0 + L), cols] = np.nan
            truth["missing_blocks"] += 1
    obs[["precip", "tmax", "tmin", "rh"]] = vals

    # ---- inhomogeneities (ground truth) -----------------------------------------------------------
    n_inh = cfg.get("inject_inhomogeneities", 4)
    st["relocation_events"] = "[]"
    cand = st.station_id[st.elevation_m < 2700].to_numpy(dtype=object)
    for j, sid in enumerate(rng.choice(cand, size=min(n_inh, len(cand)), replace=False)):
        year = int(rng.integers(2002, 2015))
        kind = ["tmax_shift", "tmin_shift", "precip_ratio", "tmax_shift"][j % 4]
        m = (obs.station_id == sid) & (obs.date.dt.year >= year)
        if kind == "tmax_shift":
            mag = float(np.round(rng.uniform(1.0, 1.6), 2)); obs.loc[m, "tmax"] += mag
        elif kind == "tmin_shift":
            mag = -1.2; obs.loc[m, "tmin"] += mag
        else:
            mag = 0.8; obs.loc[m, "precip"] = np.round(obs.loc[m, "precip"] * mag, 1)
        documented = j % 2 == 0
        if documented:
            st.loc[st.station_id == sid, "relocation_events"] = json.dumps(
                [{"date": f"{year}-01-01", "type": "instrument/site change", "detail": kind}])
        truth["inhomogeneities"].append({"station_id": sid, "year": year, "type": kind,
                                         "magnitude": mag, "documented": documented})

    # ---- injected errors (ground truth for QC) ------------------------------------------------
    if cfg.get("inject_errors", True):
        valid = obs.index[obs.precip.notna() & obs.tmax.notna()].values
        def pick(n):
            return rng.choice(valid, size=n, replace=False)
        for i in pick(3):
            obs.loc[i, "precip"] = -5.0; truth["errors"].append(("negative_precip", int(i)))
        for i in pick(2):
            obs.loc[i, "precip"] = 999.9; truth["errors"].append(("precip_sentinel", int(i)))
        for i in pick(2):
            obs.loc[i, "tmax"] = 65.0; truth["errors"].append(("tmax_spike", int(i)))
        for i in pick(5):
            obs.loc[i, ["tmax", "tmin"]] = obs.loc[i, ["tmin", "tmax"]].values
            truth["errors"].append(("tmax_tmin_swap", int(i)))
        dup = obs.loc[pick(6)].copy()
        dup["precip"] = dup["precip"].fillna(0) + 0.3
        obs = pd.concat([obs, dup])
        truth["errors"] += [("duplicate", int(i)) for i in dup.index]
        sid_u = st.station_id.iloc[int(rng.integers(0, S))]
        m = (obs.station_id == sid_u) & (obs.date.dt.year == 2008)
        obs.loc[m, "precip"] = obs.loc[m, "precip"] * 10
        truth["errors"].append(("unit_error_x10", sid_u, 2008))
        k = int(rng.integers(0, S))
        true_e = int(st.elevation_m.iloc[k])
        st.loc[k, "elevation_m"] = true_e + 1800
        truth["metadata_errors"].append({"station_id": st.station_id.iloc[k], "field": "elevation_m",
                                         "recorded": true_e + 1800, "true": true_e})

    obs = obs.sort_values(["station_id", "date"], kind="stable").reset_index(drop=True)

    true_st = st.copy()
    for e in truth["metadata_errors"]:
        true_st.loc[true_st.station_id == e["station_id"], "elevation_m"] = e["true"]
    dem = synthetic_dem(cfg.get("dem_resolution_deg", 0.05), seed, true_st)
    rs = synthetic_remote_sensing(true_st, rng)
    stations = st.drop(columns=["map_mm", "rain_shadow"])
    stations["instrument"] = rng.choice(["manual", "automatic"], size=S, p=[0.7, 0.3])
    stations["source"] = "synthetic"
    return {"stations": stations, "observations": obs, "dem": dem, "basins": basins,
            "rivers": rivers_geojson(), "outline": outline_geojson(), "remote_sensing": rs,
            "ground_truth": truth}


def synthetic_remote_sensing(st: pd.DataFrame, rng) -> pd.DataFrame:
    """Station-level remote-sensing summaries mimicking MODIS-type products (2001-2010 climatology).

    Real counterparts: MOD13Q1 NDVI, MOD10A1 snow cover, MOD11A2 LST, ESA WorldCover land cover.
    """
    e = st.elevation_m.values.astype(float)
    wet = st.map_mm.values.astype(float)
    ndvi = np.clip(0.30 + 0.35 * np.exp(-((e - 1200) / 1600) ** 2) + 0.05 * np.log(wet / 1500)
                   - 0.25 * (e > 3800) + rng.normal(0, 0.03, len(e)), 0.05, 0.85)
    amp = np.clip(0.10 + 0.15 * (1 - np.exp(-wet / 2000)) + rng.normal(0, 0.02, len(e)), 0.02, 0.5)
    snow = 1 / (1 + np.exp(-(e - 3300) / 350))
    lst = 31.0 - 6.3 * e / 1000 + rng.normal(0, 0.7, len(e))
    urban = st.name.isin(["Kathmandu", "Pokhara", "Biratnagar", "Nepalgunj", "Bharatpur", "Dhangadhi"])
    lc = np.select([urban, e < 300, e < 1000, e < 3200, e < 4500],
                   ["urban", "cropland", "cropland_forest_mosaic", "forest", "shrub_grassland"], "bare_snow")
    return pd.DataFrame({"station_id": st.station_id.values, "ndvi_mean": ndvi.round(3),
                         "ndvi_seasonal_amplitude": amp.round(3), "snow_cover_fraction": snow.round(3),
                         "lst_day_mean_c": lst.round(2), "landcover": lc,
                         "source": "synthetic (MODIS/WorldCover-like)"})


def q_term(u: np.ndarray) -> np.ndarray:
    """Heavier rain -> stronger daytime cooling (0..1)."""
    return np.clip((u - 0.7) / 0.3, 0, 1)


def write(out_dir: str | Path, cfg: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = generate(cfg)
    d["stations"].to_csv(out / "stations.csv", index=False)
    d["observations"].to_parquet(out / "observations.parquet", index=False)
    d["dem"].save(out / "dem")
    d["remote_sensing"].to_csv(out / "remote_sensing.csv", index=False)
    for k in ("basins", "rivers", "outline"):
        (out / f"{k}.geojson").write_text(json.dumps(d[k]))
    (out / "ground_truth.json").write_text(json.dumps(d["ground_truth"], indent=2, default=str))
    (out / "README.txt").write_text(
        "SYNTHETIC DATA generated by hydrogeoai.data.synthetic for pipeline testing only.\n"
        "Do not use for scientific conclusions. See docs/02_data.md for real data sources.\n")
    return d
