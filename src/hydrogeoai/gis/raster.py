"""CRS-aware raster container.

Every raster carries CRS, resolution, extent, affine transform and source. Operations that combine
rasters or points refuse to proceed when coordinate systems differ (never silently mix CRSs).
Reprojection uses pyproj/rasterio when installed; otherwise it raises a clear error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class CRSMismatchError(ValueError):
    pass


@dataclass
class GeoRaster:
    data: np.ndarray                 # (rows, cols); row 0 = north edge
    crs: str                         # e.g. "EPSG:4326"
    transform: tuple[float, float, float, float, float, float]  # GDAL order: x0, dx, 0, y0, 0, dy(<0)
    source: str = "unknown"
    nodata: float | None = np.nan
    name: str = "raster"
    units: str = ""
    history: list[str] = field(default_factory=list)

    # ---- geometry -------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def resolution(self) -> tuple[float, float]:
        return (self.transform[1], abs(self.transform[5]))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)"""
        x0, dx, _, y0, _, dy = self.transform
        rows, cols = self.shape
        return (x0, y0 + dy * rows, x0 + dx * cols, y0)

    def xy_to_rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, dx, _, y0, _, dy = self.transform
        col = np.floor((np.asarray(x) - x0) / dx).astype(int)
        row = np.floor((np.asarray(y) - y0) / dy).astype(int)
        return row, col

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x0, dx, _, y0, _, dy = self.transform
        rows, cols = self.shape
        xs = x0 + dx * (np.arange(cols) + 0.5)
        ys = y0 + dy * (np.arange(rows) + 0.5)
        return np.meshgrid(xs, ys)

    def check_crs(self, other_crs: str) -> None:
        if _norm(other_crs) != _norm(self.crs):
            raise CRSMismatchError(
                f"CRS mismatch: raster '{self.name}' is {self.crs} but input is {other_crs}. "
                "Reproject explicitly before combining."
            )

    # ---- operations -----------------------------------------------------------------------
    def sample(self, x, y, crs: str) -> np.ndarray:
        """Extract raster values at points (nearest cell). NaN outside the raster."""
        self.check_crs(crs)
        row, col = self.xy_to_rowcol(x, y)
        rows, cols = self.shape
        ok = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        out = np.full(len(np.atleast_1d(x)), np.nan)
        out[ok] = self.data[row[ok], col[ok]]
        return out

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "crs": self.crs,
            "resolution": self.resolution,
            "extent": self.extent,
            "transform": self.transform,
            "shape": self.shape,
            "source": self.source,
            "units": self.units,
            "history": self.history,
        }

    def save(self, path: str | Path) -> None:
        """Save as .npz + sidecar JSON (GeoTIFF via rasterio when available)."""
        path = Path(path)
        if path.suffix in {".tif", ".tiff"}:
            import rasterio  # optional
            from rasterio.transform import Affine

            x0, dx, _, y0, _, dy = self.transform
            with rasterio.open(
                path, "w", driver="GTiff", height=self.shape[0], width=self.shape[1], count=1,
                dtype=self.data.dtype, crs=self.crs, transform=Affine(dx, 0, x0, 0, dy, y0),
                nodata=self.nodata,
            ) as dst:
                dst.write(self.data, 1)
            return
        np.savez_compressed(path.with_suffix(".npz"), data=self.data)
        path.with_suffix(".json").write_text(json.dumps(self.metadata(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "GeoRaster":
        path = Path(path)
        if path.suffix in {".tif", ".tiff"}:
            import rasterio

            with rasterio.open(path) as src:
                t = src.transform
                return cls(src.read(1).astype(float), str(src.crs), (t.c, t.a, t.b, t.f, t.d, t.e),
                           source=str(path), nodata=src.nodata, name=path.stem)
        meta = json.loads(path.with_suffix(".json").read_text())
        data = np.load(path.with_suffix(".npz"))["data"]
        return cls(data, meta["crs"], tuple(meta["transform"]), source=meta["source"], name=meta["name"],
                   units=meta.get("units", ""), history=meta.get("history", []))


def _norm(crs: str) -> str:
    return str(crs).upper().replace(" ", "")


def reproject_points(x, y, src_crs: str, dst_crs: str):
    """Explicit point reprojection (requires pyproj)."""
    if _norm(src_crs) == _norm(dst_crs):
        return np.asarray(x), np.asarray(y)
    try:
        from pyproj import Transformer
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyproj is required for reprojection: pip install 'hydrogeoai[geo]'") from e
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tr.transform(x, y)
