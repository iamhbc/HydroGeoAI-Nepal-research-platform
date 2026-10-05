"""End-to-end: small dataset -> tiny experiment -> registry -> API prediction (a few minutes on CPU)."""
import json

import pandas as pd


def test_dataset_build(workspace):
    d = workspace["dataset_dir"]
    for f in ("observations.parquet", "stations.parquet", "metadata.json", "provenance.json", "homogeneity.csv",
              "whiplash_events.parquet", "qc_summary.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["qc_status"] == "checked" and meta["processing_history"]
    obs = pd.read_parquet(d / "observations.parquet")
    assert {"precip_qc", "precip_flag", "extreme_wet_day", "spi30"} <= set(obs.columns)
    st = pd.read_parquet(d / "stations.parquet")
    assert st[["slope_deg", "dist_river_km", "ndvi_mean"]].notna().all().all()


def test_experiment_and_api(workspace):
    from fastapi.testclient import TestClient

    from hydrogeoai.experiments.runner import run

    res = run(str(workspace["exp_config"]))
    from pathlib import Path
    out = Path(res["results_dir"])
    for f in ("SUMMARY.md", "tables/ablation_A_to_F.csv", "tables/hypotheses_H1_H6.csv",
              "metrics/uncertainty.json", "models/hydrogeoai-nepal-model/config.json"):
        assert (out / f).exists(), f
    abl = pd.read_csv(out / "tables/ablation_A_to_F.csv")
    assert set(abl.variant) == {"A_met_only", "B_met_temporal", "C_met_gis", "D_met_gis_rs",
                                "E_ssl_representation", "F_full_multimodal"}

    from hydrogeoai.api import state
    state.STATE = None
    from hydrogeoai.api.main import app
    c = TestClient(app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["deployed_model"] == res["model_id"]
    r = c.post("/predict", json={"station_ids": ["NP0001", "NP0021"], "start": "2011-06-01", "end": "2011-07-31"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model_version"] == res["model_id"] and j["dataset_version"] == workspace["dataset_dir"].name
    p = j["predictions"][0]
    assert {"probability", "confidence", "epistemic", "uncertainty", "conformal_set"} <= set(p)
    assert c.get("/predictions/log").json()[0]["model_version"] == res["model_id"]
    assert c.get(f"/experiments/{res['experiment_id']}").json()["status"] == "completed"
    assert c.get("/admin/status").status_code == 401
