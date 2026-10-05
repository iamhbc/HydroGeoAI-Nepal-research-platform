"""Static geospatial + remote-sensing + metadata features per station.

Feature groups (used by the ablation study):
    gis  : elevation, slope, northness, eastness, local relief, distance to river, lat, lon,
           elevation-band climate zone (ordinal), land-cover one-hot
    rs   : NDVI mean / seasonal amplitude, snow-cover fraction, LST
    meta : instrument type, reference-period completeness
Basin identity is deliberately NOT a model feature: it would not transfer to held-out basins and is
used only for grouping/splitting and aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..gis.raster import GeoRaster
from ..gis.terrain import aspect_to_components, slope_aspect

LANDCOVER = ["urban", "cropland", "cropland_forest_mosaic", "forest", "shrub_grassland", "bare_snow"]
GIS_FEATURES = ["lat", "lon", "elevation_m", "slope_deg", "northness", "eastness", "relief_m",
                "dist_river_km", "climate_zone"] + [f"lc_{c}" for c in LANDCOVER]
RS_FEATURES = ["ndvi_mean", "ndvi_seasonal_amplitude", "snow_cover_fraction", "lst_day_mean_c"]
META_FEATURES = ["instrument_automatic", "ref_completeness"]
FEATURE_GROUPS = {"gis": GIS_FEATURES, "rs": RS_FEATURES, "meta": META_FEATURES}


def _dist_point_polyline_km(lat, lon, coords) -> float:
    c = np.asarray(coords, float)
    kx = 111.32 * np.cos(np.deg2rad(lat))
    P = np.array([lon * kx, lat * 110.57])
    A = np.c_[c[:-1, 0] * kx, c[:-1, 1] * 110.57]
    B = np.c_[c[1:, 0] * kx, c[1:, 1] * 110.57]
    AB = B - A
    t = np.clip(np.einsum("ij,ij->i", P - A, AB) / np.einsum("ij,ij->i", AB, AB), 0, 1)
    proj = A + t[:, None] * AB
    return float(np.min(np.linalg.norm(proj - P, axis=1)))


def climate_zone(elev_m: np.ndarray) -> np.ndarray:
    """Ordinal elevation-band zone: 0 tropical/subtropical (<1000 m), 1 warm temperate (<2000),
    2 cool temperate (<3000), 3 subalpine (<4000), 4 alpine/nival."""
    return np.digitize(elev_m, [1000, 2000, 3000, 4000])


def station_static_features(stations: pd.DataFrame, dem: GeoRaster, rivers_geojson: dict,
                            remote_sensing: pd.DataFrame | None = None,
                            obs: pd.DataFrame | None = None, ref: tuple[str, str] | None = None) -> pd.DataFrame:
    st = stations.copy()
    slope, aspect = slope_aspect(dem)
    x, y = st.lon.values, st.lat.values
    st["dem_elevation_m"] = dem.sample(x, y, "EPSG:4326")
    st["slope_deg"] = slope.sample(x, y, "EPSG:4326")
    n, e = aspect_to_components(aspect.sample(x, y, "EPSG:4326"))
    st["northness"], st["eastness"] = n, e
    # local relief: std of DEM in a 5x5 window (~25 km) -> exposure/complexity proxy
    r, c = dem.xy_to_rowcol(x, y)
    rel = []
    for rr, cc in zip(r, c):
        win = dem.data[max(rr - 2, 0): rr + 3, max(cc - 2, 0): cc + 3]
        rel.append(float(np.nanstd(win)))
    st["relief_m"] = rel
    lines = [f["geometry"]["coordinates"] for f in rivers_geojson["features"]]
    st["dist_river_km"] = [min(_dist_point_polyline_km(la, lo, ln) for ln in lines) for la, lo in zip(y, x)]
    st["climate_zone"] = climate_zone(st.elevation_m.values)
    if remote_sensing is not None:
        st = st.merge(remote_sensing.drop(columns=["source"], errors="ignore"), on="station_id", how="left")
    if "landcover" in st:
        for lc in LANDCOVER:
            st[f"lc_{lc}"] = (st.landcover == lc).astype(float)
    st["instrument_automatic"] = (st.get("instrument", "manual") == "automatic").astype(float)
    if obs is not None and ref is not None:
        col = "precip_qc" if "precip_qc" in obs else "precip"
        r_ = obs[(obs.date >= ref[0]) & (obs.date <= ref[1])]
        comp = r_.groupby("station_id")[col].apply(lambda s: s.notna().mean())
        st["ref_completeness"] = st.station_id.map(comp).fillna(0.0)
    else:
        st["ref_completeness"] = 1.0
    for c in GIS_FEATURES + RS_FEATURES + META_FEATURES:
        if c not in st:
            st[c] = np.nan
    return st
