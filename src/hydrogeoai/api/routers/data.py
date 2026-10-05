"""Dataset, station, climate, extremes, whiplash, QC and map endpoints."""
from __future__ import annotations

import io
import json
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ...events import EVENT_COLUMNS, EVENT_GROUPS
from ..state import get_state

router = APIRouter(tags=["data"])
VARIABLES = {"precip": "precip_qc", "tmax": "tmax_qc", "tmin": "tmin_qc", "rh": "rh_qc", "spi30": "spi30", "spi90": "spi90"}


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _station(sid: str) -> pd.Series:
    st = get_state().data.stations
    m = st[st.station_id == sid]
    if m.empty:
        raise HTTPException(404, f"station {sid} not found")
    return m.iloc[0]


def _select_stations(station_id: str | None, basin: str | None, province: str | None) -> list[str]:
    st = get_state().data.stations
    if station_id:
        _station(station_id)
        return [station_id]
    if basin:
        st = st[st.basin == basin]
    if province:
        st = st[st.province == province]
    if st.empty:
        raise HTTPException(404, "no stations match the selection")
    return st.station_id.tolist()


_OVERVIEW_CACHE: dict[str, dict] = {}


@router.get("/overview")
def overview():
    """One small, cached summary for the landing page (computed once per dataset + model version)."""
    st = get_state()
    s = st.data
    dep = st.registry.deployed_model()
    key = f"{s.version}|{dep['id'] if dep else None}"
    if key in _OVERVIEW_CACHE:
        return _OVERVIEW_CACHE[key]
    obs, stations = s.obs, s.stations
    last = obs.date.max()
    year = obs[obs.date > last - pd.Timedelta(days=365)]
    meta = s.metadata
    out = {
        "dataset_version": s.version,
        "synthetic": "synthetic" in str(meta.get("source", "")).lower(),
        "period": [str(obs.date.min().date()), str(last.date())],
        "n_stations": int(stations.station_id.nunique()),
        "n_basins": int(stations.basin.nunique()),
        "elevation_range_m": [int(stations.elevation_m.min()), int(stations.elevation_m.max())],
        "last_12_months": {
            "extreme_wet_days": int(np.nansum(year["extreme_wet_day"])),
            "hot_days": int(np.nansum(year["hot_day"])),
            "dry_spell_days": int(np.nansum(year["dry_spell"])),
            "whiplash_events": int((pd.to_datetime(s.whiplash.transition_date) > last - pd.Timedelta(days=365)).sum()),
        },
        "data_quality": {"errors_removed": int(meta.get("qc_summary", {}).get("n_errors", 0)),
                         "missing_rain_pct": float(meta.get("missing_pct", {}).get("precip", 0.0))},
        "model": None if dep is None else {
            "id": dep["id"], "trained_on": (dep.get("metrics") or {}).get("split_mode"),
            "auprc": (dep.get("metrics") or {}).get("auprc"),
        },
    }
    _OVERVIEW_CACHE.clear()
    _OVERVIEW_CACHE[key] = out
    return out


@router.get("/stations")
def stations(basin: str | None = None, province: str | None = None):
    st = get_state().data.stations
    if basin:
        st = st[st.basin == basin]
    if province:
        st = st[st.province == province]
    cols = ["station_id", "name", "lat", "lon", "elevation_m", "basin", "province", "instrument", "slope_deg",
            "dist_river_km", "climate_zone", "landcover", "ndvi_mean", "snow_cover_fraction", "ref_completeness"]
    return records(st[[c for c in cols if c in st]])


@router.get("/stations/{station_id}")
def station(station_id: str):
    s = get_state().data
    row = _station(station_id)
    obs = s.obs[s.obs.station_id == station_id]
    issues = s.table("station_issues")
    homog = s.table("homogeneity")
    trends = s.table("trends")
    comp = {v: float(obs[f"{v}_qc"].notna().mean()) for v in ("precip", "tmax", "tmin", "rh") if f"{v}_qc" in obs}
    first = obs.dropna(subset=["precip_qc"]).date.min()
    return {
        "station": json.loads(row.to_json(date_format="iso")),
        "record": {"first_valid": None if pd.isna(first) else str(first.date()), "last": str(obs.date.max().date()),
                   "completeness": comp},
        "issues": records(issues[issues.station_id == station_id]) if issues is not None else [],
        "homogeneity": records(homog[homog.station_id == station_id]) if homog is not None else [],
        "trends": records(trends[trends.station_id == station_id]) if trends is not None else [],
        "event_rates_pct": {c: round(float(obs[c].mean() * 100), 3) for c in EVENT_COLUMNS if c in obs},
    }


@router.get("/datasets")
def datasets():
    return {"datasets": get_state().catalog.list(), "active_version": get_state().data.version}


@router.get("/datasets/{key}")
def dataset(key: str):
    cat = get_state().catalog
    try:
        entry = cat.get(key if "@" in key else f"hydrogeoai-nepal@{key}")
    except KeyError:
        raise HTTPException(404, f"dataset {key} not found")
    from ...config import resolve
    d = resolve(entry["files"]["dir"])
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
    prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").exists() else {}
    return {**entry, "metadata": meta, "provenance": prov}


@router.get("/climate")
def climate(variable: str = "precip", station_id: str | None = None, basin: str | None = None,
            province: str | None = None, start: str = "1990-01-01", end: str = "2100-01-01",
            agg: Literal["daily", "monthly", "annual", "climatology"] = "monthly"):
    if variable not in VARIABLES:
        raise HTTPException(400, f"variable must be one of {list(VARIABLES)}")
    s = get_state().data
    ids = _select_stations(station_id, basin, province)
    col = VARIABLES[variable]
    o = s.obs[s.obs.station_id.isin(ids) & (s.obs.date >= start) & (s.obs.date <= end)][["station_id", "date", col]]
    total = variable == "precip"
    if agg == "daily":
        if len(ids) > 1:
            o = o.groupby("date")[col].mean().reset_index()
        out = o.rename(columns={col: "value"})
    elif agg == "climatology":
        g = o.assign(month=o.date.dt.month, year=o.date.dt.year)
        if total:
            m = g.groupby(["station_id", "year", "month"])[col].sum(min_count=20).groupby("month").mean()
        else:
            m = g.groupby("month")[col].mean()
        out = m.rename("value").reset_index()
    else:
        freq = "MS" if agg == "monthly" else "YS"
        per = o.groupby(["station_id", pd.Grouper(key="date", freq=freq)])[col]
        thresh = 20 if agg == "monthly" else 292
        v = per.sum(min_count=thresh) if total else per.mean()
        cnt = per.count()
        v[cnt < thresh] = np.nan
        out = v.groupby("date").mean().rename("value").reset_index()
    units = {"precip": "mm" + ("/day" if agg == "daily" else ""), "tmax": "degC", "tmin": "degC", "rh": "%",
             "spi30": "-", "spi90": "-"}[variable]
    return {"variable": variable, "aggregation": agg, "units": units, "stations": ids,
            "dataset_version": s.version, "series": records(out)}


@router.get("/extremes")
def extremes(event: str = "extreme_wet_day", station_id: str | None = None, basin: str | None = None,
             province: str | None = None, start: str = "1990-01-01", end: str = "2100-01-01"):
    if event not in EVENT_COLUMNS:
        raise HTTPException(400, f"event must be one of {EVENT_COLUMNS}")
    s = get_state().data
    ids = _select_stations(station_id, basin, province)
    o = s.obs[s.obs.station_id.isin(ids) & (s.obs.date >= start) & (s.obs.date <= end)]
    g = o.assign(year=o.date.dt.year).groupby(["station_id", "year"])[event]
    days = g.sum(min_count=1)
    valid = g.count()
    ok = valid >= 292
    annual = (days[ok] / valid[ok] * 365.25).groupby("year").mean().rename("days_per_year").reset_index()
    by_month = o.assign(month=o.date.dt.month).groupby("month")[event].mean().mul(100).rename("pct_days").reset_index()
    recent = o[o[event] == 1].sort_values("date", ascending=False).head(200)[
        ["station_id", "date", "precip_qc", "tmax_qc", "tmin_qc", "spi30"]]
    return {"event": event, "stations": ids, "annual_frequency": records(annual), "seasonality": records(by_month),
            "recent_events": records(recent), "dataset_version": s.version,
            "note": "Annual frequency normalised to 365 valid days; station-years with <80% valid days excluded."}


@router.get("/extremes/taxonomy")
def taxonomy():
    from ...config import load_config
    return {"groups": EVENT_GROUPS, "definitions": load_config("configs/events/taxonomy.yaml")}


@router.get("/whiplash")
def whiplash_events(station_id: str | None = None, basin: str | None = None, direction: str | None = None,
                    limit: int = 500):
    s = get_state().data
    ev = s.whiplash
    ids = _select_stations(station_id, basin, None) if (station_id or basin) else None
    if ids:
        ev = ev[ev.station_id.isin(ids)]
    if direction:
        ev = ev[ev.direction == direction]
    annual = s.table("whiplash_annual")
    from ...config import load_config
    from ...events import whiplash as wl
    cfg = load_config("configs/events/taxonomy.yaml")
    labeled = s.obs[["station_id", "date", "spi30"]]
    if ids:
        labeled = labeled[labeled.station_id.isin(ids)]
    dec = wl.decadal_comparison(ev, labeled, cfg["whiplash"]["decades"], n_perm=500)
    return {"events": records(ev.sort_values("transition_date", ascending=False).head(limit)),
            "n_events": int(len(ev)), "annual": records(annual) if annual is not None and not ids else None,
            "decadal": records(dec["table"]), "tests": {k: v for k, v in dec.items() if k != "table"},
            "definition": cfg["whiplash"]}


@router.get("/qc/summary")
def qc_summary():
    s = get_state().data
    return {"summary": json.loads((s.dir / "qc_summary.json").read_text()),
            "station_issues": records(s.table("station_issues")),
            "homogeneity": records(s.table("homogeneity")), "dataset_version": s.version}


@router.get("/qc/flags")
def qc_flags(station_id: str | None = None, limit: int = 1000):
    f = pd.read_parquet(get_state().data.dir / "qc_flags.parquet")
    if station_id:
        f = f[f.station_id == station_id]
    return records(f.head(limit))


# ------------------------------------------------------------------------------------------------ maps
@router.get("/maps")
def map_layers():
    return {"layers": [
        {"id": "stations", "type": "point", "endpoint": "/maps/stations"},
        {"id": "basins", "type": "polygon", "endpoint": "/maps/basins"},
        {"id": "rivers", "type": "line", "endpoint": "/maps/rivers"},
        {"id": "outline", "type": "polygon", "endpoint": "/maps/outline"},
        {"id": "elevation", "type": "raster", "endpoint": "/maps/raster/dem/image.png", "meta": "/maps/raster/dem/meta"},
        {"id": "slope", "type": "raster", "endpoint": "/maps/raster/slope/image.png", "meta": "/maps/raster/slope/meta"},
        {"id": "climate", "type": "point", "endpoint": "/maps/stations?metric=precip_annual"},
        {"id": "extremes", "type": "point", "endpoint": "/maps/stations?metric=extreme_wet_day"},
        {"id": "predictions", "type": "point", "endpoint": "POST /predict (returns GeoJSON)"},
    ], "crs": "EPSG:4326"}


@router.get("/maps/stations")
def map_stations(metric: str | None = None, start: str = "1990-01-01", end: str = "2100-01-01"):
    s = get_state().data
    st = s.stations
    vals = None
    if metric:
        o = s.obs[(s.obs.date >= start) & (s.obs.date <= end)]
        if metric in EVENT_COLUMNS:
            vals = o.groupby("station_id")[metric].mean() * 365.25
        elif metric == "precip_annual":
            vals = o.groupby("station_id").precip_qc.mean() * 365.25
        elif metric in ("tmax", "tmin"):
            vals = o.groupby("station_id")[f"{metric}_qc"].mean()
        elif metric == "whiplash":
            ev = s.whiplash
            yrs = o.groupby("station_id").date.agg(lambda d: d.dt.year.nunique())
            vals = ev.groupby("station_id").size() / yrs
        else:
            raise HTTPException(400, "unknown metric")
    feats = []
    for r in st.itertuples():
        props = {"station_id": r.station_id, "name": r.name, "elevation_m": r.elevation_m, "basin": r.basin,
                 "province": r.province}
        if vals is not None:
            v = vals.get(r.station_id, np.nan)
            props["value"] = None if pd.isna(v) else round(float(v), 3)
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats, "metric": metric,
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}


@router.get("/maps/{layer}")
def map_vector(layer: Literal["basins", "rivers", "outline"]):
    gj = get_state().data.geojson(layer)
    if gj is None:
        raise HTTPException(404, layer)
    return gj


@router.get("/maps/raster/{name}/meta")
def raster_meta(name: Literal["dem", "slope"]):
    r = _raster(name)
    xmin, ymin, xmax, ymax = r.extent
    return {**r.metadata(), "coordinates": [[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]],
            "min": float(np.nanmin(r.data)), "max": float(np.nanmax(r.data))}


@router.get("/maps/raster/{name}/image.png")
def raster_png(name: Literal["dem", "slope"]):
    import matplotlib

    matplotlib.use("Agg")
    r = _raster(name)
    d = r.data.astype(float)
    norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-9)
    rgba = (matplotlib.colormaps["terrain" if name == "dem" else "magma"](norm) * 255).astype("uint8")
    rgba[..., 3] = 170
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


def _raster(name):
    from ...gis.raster import GeoRaster
    from ...gis.terrain import slope_aspect
    dem = GeoRaster.load(get_state().data.dir / "dem")
    return dem if name == "dem" else slope_aspect(dem)[0]
