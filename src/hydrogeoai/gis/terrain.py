"""Terrain derivatives from a DEM (Horn 1981 finite differences).

For geographic CRSs (degrees) cell sizes are converted to metres using the latitude of each row,
so slope is physically meaningful without silently reprojecting the DEM.
"""
from __future__ import annotations

import numpy as np

from .raster import GeoRaster

_M_PER_DEG_LAT = 111_320.0


def _cell_size_m(dem: GeoRaster) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = dem.resolution
    rows, cols = dem.shape
    if dem.crs.upper() in {"EPSG:4326", "EPSG:4269", "OGC:CRS84"}:
        _, ys = dem.cell_centers()
        dx_m = dx * _M_PER_DEG_LAT * np.cos(np.deg2rad(ys))
        dy_m = np.full((rows, cols), dy * _M_PER_DEG_LAT)
        return dx_m, dy_m
    return np.full((rows, cols), dx), np.full((rows, cols), dy)


def slope_aspect(dem: GeoRaster) -> tuple[GeoRaster, GeoRaster]:
    """Return slope (degrees) and aspect (degrees clockwise from north, -1 = flat)."""
    z = np.pad(dem.data.astype(float), 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dx_m, dy_m = _cell_size_m(dem)
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy_m)  # z_south - z_north
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # aspect = compass direction the slope faces (downhill vector = (east=-dzdx, north=dzdy))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    aspect[slope < 0.5] = -1.0
    hist = dem.history + [f"slope/aspect (Horn 1981) from {dem.name}"]
    return (
        GeoRaster(slope, dem.crs, dem.transform, dem.source, name="slope", units="degrees", history=hist),
        GeoRaster(aspect, dem.crs, dem.transform, dem.source, name="aspect", units="degrees", history=hist),
    )


def aspect_to_components(aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circular encoding (northness, eastness); flat cells -> (0, 0)."""
    a = np.asarray(aspect_deg, dtype=float)
    rad = np.deg2rad(a)
    north, east = np.cos(rad), np.sin(rad)
    flat = a < 0
    north[flat] = 0.0
    east[flat] = 0.0
    return north, east
