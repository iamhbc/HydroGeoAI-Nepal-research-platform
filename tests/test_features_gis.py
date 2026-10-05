import numpy as np
import pandas as pd
import pytest

from hydrogeoai.events.taxonomy import add_target, run_length
from hydrogeoai.events.whiplash import detect_station
from hydrogeoai.features.indices import spi
from hydrogeoai.gis import CRSMismatchError, GeoRaster, haversine_km, idw_interpolate, point_in_polygon, slope_aspect


def test_slope_aspect_plane():
    # elevation rises to the south by 100 m per cell of 1000 m -> slope ~5.7 deg, facing north (0 deg)
    rows, cols = 10, 10
    z = np.repeat((np.arange(rows) * 100.0)[:, None], cols, 1)
    dem = GeoRaster(z, "EPSG:32645", (0, 1000, 0, 10000, 0, -1000))
    s, a = slope_aspect(dem)
    assert np.allclose(s.data[2:-2, 2:-2], np.degrees(np.arctan(0.1)), atol=0.05)
    assert np.allclose(a.data[2:-2, 2:-2], 0.0, atol=0.5)
    zx = np.repeat((np.arange(cols) * 100.0)[None, :], rows, 0)   # rises to the east -> faces west
    _, a2 = slope_aspect(GeoRaster(zx, "EPSG:32645", (0, 1000, 0, 10000, 0, -1000)))
    assert np.allclose(a2.data[2:-2, 2:-2], 270.0, atol=0.5)


def test_crs_mismatch_refused():
    r = GeoRaster(np.zeros((3, 3)), "EPSG:4326", (80, 1, 0, 30, 0, -1))
    with pytest.raises(CRSMismatchError):
        r.sample(np.array([81.0]), np.array([29.0]), "EPSG:32645")
    assert r.sample(np.array([81.5]), np.array([29.5]), "EPSG:4326")[0] == 0


def test_spatial_ops():
    assert abs(haversine_km(27.7, 85.3, 28.2, 83.98) - 141) < 5   # Kathmandu-Pokhara ~141 km
    sq = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert point_in_polygon(np.array([0.5, 2]), np.array([0.5, 0.5]), sq).tolist() == [True, False]
    v = idw_interpolate([85.0, 86.0], [27.0, 27.0], [10.0, 20.0], [85.0, 85.5], [27.0, 27.0])
    assert v[0] == 10.0 and 14 < v[1] < 16


def test_spi_standardised():
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.date_range("1990-01-01", "2019-12-31"))
    p = pd.Series(np.where(rng.random(len(dates)) < 0.4, rng.gamma(0.8, 8, len(dates)), 0.0))
    s = spi(p, dates, 30, ("1990-01-01", "2019-12-31"))
    s = s[np.isfinite(s)]
    assert abs(s.mean()) < 0.1 and 0.85 < s.std() < 1.15


def test_run_length_and_target_alignment():
    rl = run_length(np.array([1, 1, 0, 1, 1, 1], bool))
    assert rl.tolist() == [1, 2, 0, 1, 2, 3]
    df = pd.DataFrame({"station_id": ["a"] * 5, "ev": [0, 1, 0, np.nan, 1]})
    out = add_target(df, "ev", lead=1)
    assert np.isnan(out.target.iloc[2]) and out.target.iloc[3] == 1
    assert out.target.iloc[0] == 1 and out.target.iloc[1] == 0 and np.isnan(out.target.iloc[4])


def test_whiplash_detection():
    s = np.r_[np.zeros(10), np.full(20, -1.5), np.zeros(5), np.full(20, 1.6), np.zeros(10)]
    ev = detect_station(pd.Series(pd.date_range("2000-01-01", periods=len(s))), s)
    assert len(ev) == 1 and ev.direction.iloc[0] == "dry_to_wet"
    assert ev.transition_days.iloc[0] == 6 and abs(ev.intensity_spi_swing.iloc[0] - 3.1) < 1e-6
