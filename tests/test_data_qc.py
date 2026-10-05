import numpy as np
import pandas as pd

from hydrogeoai.data.provenance import ProvenanceLog
from hydrogeoai.data.schema import validate_observations
from hydrogeoai.qc import ERROR, run_qc
from hydrogeoai.qc.homogeneity import benjamini_hochberg, buishand, mann_kendall, pettitt, sens_slope, snht


def test_synthetic_structure(small_synthetic):
    obs, st = small_synthetic["observations"], small_synthetic["stations"]
    assert validate_observations(obs) == []
    assert set(obs.station_id.unique()) == set(st.station_id)
    assert {"lat", "lon", "elevation_m", "basin"} <= set(st.columns)
    assert small_synthetic["dem"].crs == "EPSG:4326"
    assert "SYNTHETIC" in small_synthetic["ground_truth"]["warning"]


def test_qc_flags_injected_errors_without_overwriting(small_synthetic):
    obs, st = small_synthetic["observations"], small_synthetic["stations"]
    res = run_qc(obs, st, dem=small_synthetic["dem"])
    df = res.observations
    # originals preserved, errors excluded only from *_qc
    assert (df.precip < 0).sum() >= 3
    assert (df.precip_qc < 0).sum() == 0
    assert (df.loc[df.precip < 0, "precip_flag"] == ERROR).all()
    assert (df.tmax_qc > 48).sum() == 0
    assert not df.duplicated(["station_id", "date"]).any()
    assert res.summary["conflicting_duplicates"] >= 1
    assert len(res.audit) >= res.summary["n_errors"]
    # metadata elevation typo detected against the DEM
    bad = small_synthetic["ground_truth"]["metadata_errors"][0]["station_id"]
    assert ((res.station_issues.station_id == bad) & (res.station_issues.issue == "elevation_metadata_vs_dem")).any()


def test_change_point_tests_detect_step():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 20), rng.normal(2.0, 1, 20)]
    for t in (pettitt(x), snht(x), buishand(x)):
        assert t.significant, t
        assert 15 <= t.change_index <= 25
    y = rng.normal(0, 1, 40)
    assert sum(t.significant for t in (pettitt(y), snht(y), buishand(y))) <= 1


def test_trend_tests():
    t = np.arange(30)
    x = 0.1 * t + np.random.default_rng(1).normal(0, 0.3, 30)
    mk = mann_kendall(x)
    assert mk["significant"] and mk["z"] > 0
    b, _ = sens_slope(x)
    assert abs(b - 0.1) < 0.03
    sig = benjamini_hochberg(np.array([0.001, 0.01, 0.04, 0.5, 0.9]), 0.05)
    assert sig.tolist() == [True, True, False, False, False]


def test_provenance_records_hashes():
    p = ProvenanceLog("t")
    a = pd.DataFrame({"x": [1, 2]})
    p.record("step", "desc", {"k": 1}, before=a, after=a.assign(x=[1, 3]), n_affected=1)
    r = p.to_records()[0]
    assert r["input_hash"] != r["output_hash"] and r["n_affected"] == 1
