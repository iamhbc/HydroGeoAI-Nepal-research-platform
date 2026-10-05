"""Automated data quality control (Section 7 of the specification).

Outputs never overwrite observations. For each variable `v` the QC adds:
    v_flag : 0 = ok, 1 = suspicious (kept), 2 = error (excluded), 3 = missing
    v_qc   : the value used downstream (NaN where v_flag == 2)
Every flag is written to an `AuditLog` with reason, original value and correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.provenance import AuditLog, ProvenanceLog
from ..gis.raster import GeoRaster
from ..gis.spatial import haversine_km, point_in_polygon
from ..utils import get_logger

log = get_logger(__name__)

OK, SUSPICIOUS, ERROR, MISSING = 0, 1, 2, 3
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON"}   # Nepal: monsoon = JJAS
DEFAULTS = dict(precip_max_mm=500.0, temp_min_c=-45.0, temp_max_c=48.0, max_daily_temp_jump_c=15.0,
                robust_z_spike=6.0, min_annual_completeness=0.8, nepal_bbox=[80.0, 26.3, 88.3, 30.5],
                dem_elevation_tolerance_m=300.0, stuck_run_days=7, unit_error_ratio=4.0)


@dataclass
class QCResult:
    observations: pd.DataFrame
    flags: pd.DataFrame
    audit: AuditLog
    completeness: dict[str, pd.DataFrame]
    gaps: pd.DataFrame
    station_issues: pd.DataFrame
    provenance: ProvenanceLog
    summary: dict = field(default_factory=dict)


def _flag(df, flags, audit, mask, var, level, code, reason):
    mask = mask.fillna(False) & (df[f"{var}_flag"] < level)
    n = int(mask.sum())
    if n == 0:
        return 0
    df.loc[mask, f"{var}_flag"] = level
    audit.extend_from_flags(df, mask, var, code, reason, corrected=None if level == ERROR else "unchanged")
    sub = df.loc[mask, ["station_id", "date"]].copy()
    sub["variable"], sub["flag"], sub["severity"], sub["reason"] = var, code, \
        {SUSPICIOUS: "suspicious", ERROR: "error"}[level], reason
    flags.append(sub)
    return n


def resolve_duplicates(obs: pd.DataFrame, audit: AuditLog, variables) -> tuple[pd.DataFrame, int, int]:
    """Exact duplicates are dropped; conflicting duplicates keep one row with conflicting values NaN."""
    key = ["station_id", "date"]
    exact = obs.duplicated(keep="first")
    n_exact = int(exact.sum())
    obs = obs[~exact]
    dup = obs.duplicated(key, keep=False)
    n_conf = 0
    if dup.any():
        conf = obs[dup]
        resolved = []
        for (sid, d), g in conf.groupby(key):
            row = g.iloc[0].copy()
            for v in variables:
                if v in g and g[v].nunique(dropna=True) > 1:
                    audit.add(station_id=str(sid), date=str(pd.Timestamp(d).date()), variable=v,
                              original=float(g[v].iloc[0]), corrected=None, flag="duplicate_conflict",
                              reason=f"{len(g)} rows with differing values {g[v].tolist()}")
                    row[v] = np.nan
                    n_conf += 1
            resolved.append(row)
        obs = pd.concat([obs[~dup], pd.DataFrame(resolved)]).sort_values(key)
    return obs.reset_index(drop=True), n_exact, n_conf


def completeness_tables(obs: pd.DataFrame, variables) -> dict[str, pd.DataFrame]:
    o = obs.assign(year=obs.date.dt.year, month=obs.date.dt.month, season=obs.date.dt.month.map(SEASONS))
    vs = [v for v in variables if v in o]
    out = {"station": o.groupby("station_id")[vs].apply(lambda g: g.notna().mean())}
    for name, keys in {"annual": ["station_id", "year"], "seasonal": ["station_id", "year", "season"],
                       "monthly": ["station_id", "year", "month"]}.items():
        out[name] = o.groupby(keys)[vs].apply(lambda g: g.notna().mean()).reset_index()
    return out


def find_gaps(obs: pd.DataFrame, var: str, min_len: int = 1) -> pd.DataFrame:
    rows = []
    for sid, g in obs.groupby("station_id"):
        miss = g[var].isna().to_numpy()
        if not miss.any():
            continue
        d = np.diff(np.r_[0, miss.astype(int), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        dates = g.date.to_numpy()
        for s, e in zip(starts, ends):
            if e - s >= min_len:
                rows.append({"station_id": sid, "variable": var, "start": dates[s], "end": dates[e - 1],
                             "length_days": int(e - s)})
    return pd.DataFrame(rows, columns=["station_id", "variable", "start", "end", "length_days"])


def station_consistency(stations: pd.DataFrame, cfg: dict, dem: GeoRaster | None = None,
                        outline: list | None = None) -> pd.DataFrame:
    issues = []
    lon0, lat0, lon1, lat1 = cfg["nepal_bbox"]
    for r in stations.itertuples():
        if not (lon0 <= r.lon <= lon1 and lat0 <= r.lat <= lat1):
            issues.append((r.station_id, "coordinates_outside_bbox", f"({r.lat}, {r.lon})"))
        if outline is not None and not point_in_polygon(np.array([r.lon]), np.array([r.lat]), outline)[0]:
            issues.append((r.station_id, "coordinates_outside_outline", f"({r.lat}, {r.lon})"))
    if dem is not None:
        z = dem.sample(stations.lon.values, stations.lat.values, "EPSG:4326")
        diff = stations.elevation_m.values - z
        for sid, d, zz in zip(stations.station_id, diff, z):
            if np.isfinite(d) and abs(d) > cfg["dem_elevation_tolerance_m"]:
                issues.append((sid, "elevation_metadata_vs_dem",
                               f"metadata - DEM = {d:+.0f} m (DEM {zz:.0f} m); check metadata or relocation"))
    lat, lon = stations.lat.values, stations.lon.values
    for i in range(len(stations)):
        d = haversine_km(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        if d.min() < 0.5:
            issues.append((stations.station_id.iloc[i], "near_duplicate_location", f"{d.min():.2f} km to another station"))
    if "relocation_events" in stations:
        for sid, ev in zip(stations.station_id, stations.relocation_events):
            if isinstance(ev, str) and ev not in ("", "[]"):
                issues.append((sid, "documented_station_history", ev))
    return pd.DataFrame(issues, columns=["station_id", "issue", "detail"])


def _deseasonalised_robust_z(g: pd.DataFrame, var: str) -> pd.Series:
    doy = g.date.dt.dayofyear
    clim = g.groupby(doy)[var].transform("median")
    clim = clim.rolling(31, center=True, min_periods=1).mean()
    anom = g[var] - clim
    med = anom.median()
    mad = (anom - med).abs().median() * 1.4826 + 1e-6
    return (anom - med) / mad


def run_qc(obs: pd.DataFrame, stations: pd.DataFrame, cfg: dict | None = None,
           dem: GeoRaster | None = None, outline: list | None = None,
           variables=("precip", "tmax", "tmin", "rh")) -> QCResult:
    cfg = {**DEFAULTS, **(cfg or {})}
    prov = ProvenanceLog("qc")
    audit = AuditLog()
    flags: list[pd.DataFrame] = []
    variables = [v for v in variables if v in obs.columns]
    df, n_exact, n_conf = resolve_duplicates(obs.copy(), audit, variables)
    prov.record("duplicates", "Dropped exact duplicates; conflicting duplicates set NaN", {},
                before=obs, after=df, n_affected=n_exact + n_conf)

    # irregular intervals detected *before* calendar regularisation
    irregular = (df.groupby("station_id").date.diff().dt.days.fillna(1) != 1).sum()
    from ..data.ingest import regularise_daily
    df = regularise_daily(df)

    for v in variables:
        df[f"{v}_flag"] = np.where(df[v].isna(), MISSING, OK).astype("int8")

    if "precip" in df:
        p = df.precip
        _flag(df, flags, audit, p < 0, "precip", ERROR, "negative_precip", "Precipitation < 0 is physically impossible")
        _flag(df, flags, audit, p > cfg["precip_max_mm"], "precip", ERROR, "precip_above_max",
              f"> {cfg['precip_max_mm']} mm/day (likely sentinel/typo)")
        # unit errors: station-years whose total is >> station median annual total
        yr = df.date.dt.year
        tot = df.groupby(["station_id", yr]).precip.transform(lambda s: s.clip(lower=0).sum(min_count=200))
        med = (df.assign(tot=tot, yr=yr).groupby("station_id").apply(
            lambda g: g.drop_duplicates("yr").tot.median(), include_groups=False))
        ratio = tot / df.station_id.map(med)
        _flag(df, flags, audit, ratio > cfg["unit_error_ratio"], "precip", ERROR, "suspected_unit_error",
              f"Annual total > {cfg['unit_error_ratio']}x station median (e.g. 0.1 mm units)")
        # very large but possible values: flagged suspicious, kept
        p99 = df[p > 1].groupby("station_id").precip.quantile(0.999)
        _flag(df, flags, audit, p > 2.5 * df.station_id.map(p99), "precip", SUSPICIOUS, "precip_outlier",
              "> 2.5x station 99.9th percentile of wet days; verify against neighbours")

    for v in ("tmax", "tmin"):
        if v not in df:
            continue
        t = df[v]
        _flag(df, flags, audit, (t < cfg["temp_min_c"]) | (t > cfg["temp_max_c"]), v, ERROR,
              "temperature_out_of_range", f"Outside [{cfg['temp_min_c']}, {cfg['temp_max_c']}] degC")
        jump = df.groupby("station_id")[v].diff().abs()
        _flag(df, flags, audit, jump > cfg["max_daily_temp_jump_c"], v, SUSPICIOUS, "abrupt_jump",
              f"Day-to-day change > {cfg['max_daily_temp_jump_c']} degC")
        z = df.groupby("station_id", group_keys=False)[["date", v]].apply(lambda g: _deseasonalised_robust_z(g, v))
        _flag(df, flags, audit, z.abs() > cfg["robust_z_spike"], v, SUSPICIOUS, "robust_z_spike",
              f"|robust z| of deseasonalised anomaly > {cfg['robust_z_spike']}")
        # stuck sensor: identical values for N consecutive days
        same = df.groupby("station_id")[v].diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        _flag(df, flags, audit, run >= cfg["stuck_run_days"] - 1, v, SUSPICIOUS, "repeated_values",
              f">= {cfg['stuck_run_days']} identical consecutive values")
    if {"tmax", "tmin"} <= set(df.columns):
        bad = df.tmax < df.tmin
        for v in ("tmax", "tmin"):
            _flag(df, flags, audit, bad, v, ERROR, "tmax_lt_tmin", "Tmax < Tmin (possible column swap)")
    if "rh" in df:
        _flag(df, flags, audit, (df.rh < 0) | (df.rh > 100), "rh", ERROR, "rh_out_of_range", "RH outside [0,100]")

    for v in variables:
        df[f"{v}_qc"] = df[v].where(df[f"{v}_flag"] != ERROR)
    prov.record("qc_flags", "Applied QC flags; *_qc columns exclude errors", {"cfg": cfg},
                n_affected=len(audit))

    gaps = pd.concat([find_gaps(df, v) for v in variables], ignore_index=True)
    st_issues = station_consistency(stations, cfg, dem, outline)
    flags_df = (pd.concat(flags, ignore_index=True) if flags else
                pd.DataFrame(columns=["station_id", "date", "variable", "flag", "severity", "reason"]))
    comp = completeness_tables(df, variables)
    summary = {
        "n_records": int(len(df)), "exact_duplicates_dropped": n_exact, "conflicting_duplicates": n_conf,
        "irregular_intervals": int(irregular),
        "flags_by_type": flags_df.groupby("flag").size().to_dict() if len(flags_df) else {},
        "n_errors": int((flags_df.severity == "error").sum()) if len(flags_df) else 0,
        "n_suspicious": int((flags_df.severity == "suspicious").sum()) if len(flags_df) else 0,
        "station_issues": int(len(st_issues)),
        "missing_pct": {v: round(float(df[f"{v}_qc"].isna().mean() * 100), 2) for v in variables},
        "station_years_below_completeness": int(
            (comp["annual"][[v for v in ("precip", "tmax", "tmin") if v in comp["annual"]]]
             .min(axis=1) < cfg["min_annual_completeness"]).sum()),
    }
    log.info("QC: %s errors, %s suspicious, %s station issues", summary["n_errors"],
             summary["n_suspicious"], summary["station_issues"])
    return QCResult(df, flags_df, audit, comp, gaps, st_issues, prov, summary)
