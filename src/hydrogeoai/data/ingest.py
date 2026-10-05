"""Multi-format ingestion into the canonical schema.

Supported: CSV, Parquet, NetCDF/HDF5 (xarray), GeoTIFF (rasterio), GeoJSON, Shapefile (geopandas),
and JSON API responses. Each reader returns data plus a `ProvenanceLog` entry; nothing is modified
silently (renames and unit conversions are explicit parameters and are recorded).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..gis.raster import GeoRaster
from ..utils import file_hash, get_logger
from .provenance import ProvenanceLog
from .schema import validate_observations

log = get_logger(__name__)

UNIT_CONVERSIONS = {
    ("precip", "m"): lambda x: x * 1000.0,          # ERA5 total precipitation (m) -> mm
    ("precip", "kg m-2 s-1"): lambda x: x * 86400.0,
    ("tmax", "K"): lambda x: x - 273.15,
    ("tmin", "K"): lambda x: x - 273.15,
    ("tmax", "degF"): lambda x: (x - 32) * 5 / 9,
    ("tmin", "degF"): lambda x: (x - 32) * 5 / 9,
}


def _standardise(df: pd.DataFrame, rename: dict | None, units: dict | None, prov: ProvenanceLog,
                 source: str) -> pd.DataFrame:
    before = df.copy()
    if rename:
        df = df.rename(columns=rename)
        prov.record("rename_columns", f"Renamed columns from {source}", {"mapping": rename})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    for var, unit in (units or {}).items():
        fn = UNIT_CONVERSIONS.get((var, unit))
        if fn is None:
            raise ValueError(f"No registered conversion for {var} in {unit}")
        df[var] = fn(df[var].astype(float))
        prov.record("unit_conversion", f"{var}: {unit} -> canonical", {"variable": var, "from": unit},
                    n_affected=int(df[var].notna().sum()))
    prov.record("ingest", f"Ingested {source}", {"source": source}, before=before, after=df,
                n_affected=len(df))
    problems = validate_observations(df) if "date" in df.columns else []
    if problems:
        log.warning("Schema problems in %s: %s", source, problems)
    return df


def read_tabular(path: str | Path, rename: dict | None = None, units: dict | None = None,
                 prov: ProvenanceLog | None = None, **kw) -> pd.DataFrame:
    path = Path(path)
    prov = prov or ProvenanceLog(path.stem)
    prov.record("source_file", f"Read {path.name}", {"path": str(path), "sha256": file_hash(path)})
    if path.suffix == ".csv":
        df = pd.read_csv(path, **kw)
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, **kw)
    else:
        raise ValueError(f"Unsupported tabular format {path.suffix}")
    return _standardise(df, rename, units, prov, path.name)


def read_gridded(path: str | Path, variables: dict[str, str], stations: pd.DataFrame,
                 units: dict | None = None, prov: ProvenanceLog | None = None,
                 lat_name: str = "latitude", lon_name: str = "longitude", time_name: str = "time") -> pd.DataFrame:
    """Extract station time series from NetCDF/HDF5/Zarr grids (nearest grid cell).

    `variables` maps file variable -> canonical name, e.g. {"tp": "precip"} for ERA5-Land.
    Nearest-cell extraction is recorded; grid-cell vs point scale mismatch must be discussed when
    such products are compared with gauges.
    """
    import xarray as xr

    path = Path(path)
    prov = prov or ProvenanceLog(path.stem)
    ds = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    prov.record("source_file", f"Opened gridded {path.name}", {"path": str(path), "vars": variables})
    lats = xr.DataArray(stations.lat.values, dims="station")
    lons = xr.DataArray(stations.lon.values, dims="station")
    sub = ds[list(variables)].sel({lat_name: lats, lon_name: lons}, method="nearest")
    df = sub.to_dataframe().reset_index()
    df["station_id"] = stations.station_id.values[df["station"].values]
    df = df.rename(columns={time_name: "date", **variables})
    keep = ["station_id", "date", *variables.values()]
    prov.record("grid_to_point", "Nearest-cell extraction at station coordinates",
                {"method": "nearest", "n_stations": len(stations)})
    return _standardise(df[keep], None, units, prov, path.name)


def read_raster(path: str | Path) -> GeoRaster:
    path = Path(path)
    if path.suffix in {".tif", ".tiff"}:
        return GeoRaster.load(path)
    if path.suffix in {".npz", ".json"}:
        return GeoRaster.load(path.with_suffix(""))
    raise ValueError(f"Unsupported raster {path}")


def read_vector(path: str | Path) -> dict:
    """GeoJSON (native) or Shapefile/GPKG (geopandas). Returns GeoJSON dict with explicit CRS."""
    path = Path(path)
    if path.suffix in {".geojson", ".json"}:
        gj = json.loads(path.read_text())
        gj.setdefault("crs", {"type": "name", "properties": {"name": "EPSG:4326"}})
        return gj
    import geopandas as gpd  # optional dependency

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; refusing to guess")
    if gdf.crs.to_epsg() != 4326:
        log.info("Reprojecting %s from %s to EPSG:4326 (recorded)", path.name, gdf.crs)
        gdf = gdf.to_crs(4326)
    gj = json.loads(gdf.to_json())
    gj["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    return gj


def read_api_json(payload: dict | list, records_key: str | None = None, rename: dict | None = None,
                  units: dict | None = None, source: str = "api") -> pd.DataFrame:
    """Normalise a JSON API response (list of records) into the canonical schema."""
    recs: Any = payload[records_key] if records_key else payload
    df = pd.json_normalize(recs)
    return _standardise(df, rename, units, ProvenanceLog(source), source)


def wide_to_long(df: pd.DataFrame, variable: str, date_col: str = "date") -> pd.DataFrame:
    """Convert a station-per-column table (common in DHM exports) to long format."""
    long = df.melt(id_vars=[date_col], var_name="station_id", value_name=variable)
    long[variable] = pd.to_numeric(long[variable], errors="coerce")
    return long.rename(columns={date_col: "date"})


def merge_variables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["station_id", "date"], how="outer")
    return out.sort_values(["station_id", "date"]).reset_index(drop=True)


def regularise_daily(obs: pd.DataFrame) -> pd.DataFrame:
    """Reindex each station to a complete daily calendar (missing days become explicit NaN rows).

    Duplicates must be resolved by QC *before* this step; this function refuses duplicated keys.
    """
    if obs.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate (station_id, date) keys present; run QC duplicate resolution first")
    parts = []
    for sid, g in obs.groupby("station_id", sort=True):
        idx = pd.date_range(g.date.min(), g.date.max(), freq="D")
        gg = g.set_index("date").reindex(idx)
        gg["station_id"] = sid
        parts.append(gg.rename_axis("date").reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out[["station_id", "date"] + [c for c in obs.columns if c not in ("station_id", "date")]]


def to_xarray(obs: pd.DataFrame, stations: pd.DataFrame):
    """Station x time xarray Dataset (for NetCDF/Zarr export) with CF-style attributes."""
    from .schema import VARIABLES

    ds = obs.set_index(["date", "station_id"]).to_xarray()
    st = stations.set_index("station_id").reindex(ds.station_id.values)
    for c in ("lat", "lon", "elevation_m"):
        ds = ds.assign_coords({c: ("station_id", st[c].values.astype(float))})
    for v, meta in VARIABLES.items():
        if v in ds:
            ds[v].attrs.update(units=meta["units"], long_name=meta["long_name"])
    ds.attrs.update(Conventions="CF-1.8", crs="EPSG:4326", title="HydroGeoAI-Nepal station dataset")
    return ds


def summarise_missing(obs: pd.DataFrame, variables=("precip", "tmax", "tmin")) -> pd.DataFrame:
    return (obs.groupby("station_id")[[v for v in variables if v in obs]]
            .apply(lambda g: g.isna().mean() * 100).round(2))


__all__ = ["read_tabular", "read_gridded", "read_raster", "read_vector", "read_api_json", "wide_to_long",
           "merge_variables", "regularise_daily", "to_xarray", "summarise_missing", "np"]
