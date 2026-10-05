"""Vector/point spatial operations with explicit CRS handling.

Pure-numpy implementations so the core pipeline runs without GDAL; GeoPandas/Shapely are used
automatically when installed for polygon-heavy work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .raster import GeoRaster

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_stations(stations: pd.DataFrame, lat: float, lon: float, k: int = 5,
                     exclude: list[str] | None = None) -> pd.DataFrame:
    """k nearest stations by great-circle distance (stations in EPSG:4326)."""
    s = stations if not exclude else stations[~stations.station_id.isin(exclude)]
    d = haversine_km(lat, lon, s.lat.values, s.lon.values)
    out = s.assign(distance_km=d).sort_values("distance_km").head(k)
    return out.reset_index(drop=True)


def idw_interpolate(xs, ys, values, xq, yq, power: float = 2.0, k: int = 8,
                    max_dist_km: float | None = 150.0) -> np.ndarray:
    """Inverse-distance weighting on lon/lat (great-circle distance).

    Scientific caveat: IDW ignores orography. In the Himalaya precipitation and temperature depend
    strongly on elevation and windward/leeward exposure, so IDW is only appropriate for exploratory
    maps or for anomalies (not absolute values). Cells beyond `max_dist_km` from any station are NaN.
    """
    xs, ys, values = map(np.asarray, (xs, ys, values))
    ok = np.isfinite(values)
    xs, ys, values = xs[ok], ys[ok], values[ok]
    xq, yq = np.atleast_1d(xq).ravel(), np.atleast_1d(yq).ravel()
    out = np.full(xq.shape, np.nan)
    if len(values) == 0:
        return out
    for i, (qx, qy) in enumerate(zip(xq, yq)):
        d = haversine_km(qy, qx, ys, xs)
        idx = np.argsort(d)[:k]
        dk = d[idx]
        if max_dist_km is not None and dk[0] > max_dist_km:
            continue
        if dk[0] < 1e-6:
            out[i] = values[idx[0]]
            continue
        w = 1.0 / dk ** power
        out[i] = np.sum(w * values[idx]) / np.sum(w)
    return out


def point_in_polygon(x: np.ndarray, y: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd ray casting for a single (exterior) ring."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    poly = np.asarray(polygon, float)
    inside = np.zeros(x.shape, bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cond
        xj, yj = xi, yi
    return inside


def _rings(geom: dict) -> list[list[tuple[float, float]]]:
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    raise ValueError(f"Unsupported geometry {geom['type']}")


def spatial_join_points_polygons(points: pd.DataFrame, polygons_geojson: dict, attr: str,
                                 points_crs: str = "EPSG:4326") -> pd.Series:
    """Assign each point (lon/lat columns) the `attr` of the containing polygon (GeoJSON dict)."""
    poly_crs = polygons_geojson.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if "4326" not in poly_crs and "CRS84" not in poly_crs:
        raise ValueError(f"Polygon CRS {poly_crs} differs from point CRS {points_crs}; reproject first")
    out = pd.Series([None] * len(points), index=points.index, dtype=object)
    for feat in polygons_geojson["features"]:
        mask = np.zeros(len(points), bool)
        for ring in _rings(feat["geometry"]):
            mask |= point_in_polygon(points.lon.values, points.lat.values, ring)
        out[mask & out.isna().values] = feat["properties"][attr]
    return out


def zonal_statistics(raster: GeoRaster, polygons_geojson: dict, attr: str,
                     stats=("mean", "min", "max", "std")) -> pd.DataFrame:
    """Zonal statistics of a raster over GeoJSON polygons (cell-centre inclusion)."""
    raster.check_crs("EPSG:4326")
    xx, yy = raster.cell_centers()
    rows = []
    for feat in polygons_geojson["features"]:
        mask = np.zeros(raster.shape, bool)
        for ring in _rings(feat["geometry"]):
            mask |= point_in_polygon(xx, yy, ring)
        vals = raster.data[mask]
        vals = vals[np.isfinite(vals)]
        row = {attr: feat["properties"][attr], "n_cells": int(vals.size)}
        for s in stats:
            row[s] = float(getattr(np, s)(vals)) if vals.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
