import numpy as np
import pandas as pd
import pytest
import torch

from hydrogeoai.evaluation import best_f1_threshold, classification_metrics, expected_calibration_error
from hydrogeoai.experiments.splits import Split, check_leakage, make_split
from hydrogeoai.models import EncoderConfig, HydroclimaticEncoder, HydroGeoAIModel, Pretrainer, build_deep
from hydrogeoai.models.encoder import span_mask
from hydrogeoai.uncertainty import ClassConditionalConformal, decompose


def test_splits_have_no_leakage():
    st = pd.DataFrame({"station_id": [f"s{i}" for i in range(12)],
                       "basin": ["A"] * 4 + ["B"] * 4 + ["C"] * 4, "lon": np.arange(12.0)})
    cfg = {"temporal": {"train": ["2000-01-01", "2005-12-31"], "val": ["2006-01-01", "2007-12-31"],
                        "test": ["2008-01-01", "2010-12-31"], "gap_days": 30},
           "spatial": {"strategy": "basin", "test_groups": ["C"], "val_fraction_of_train_stations": 0.25}}
    for mode in ("temporal", "spatial", "spatiotemporal"):
        sp = make_split(mode, st, cfg)
        check_leakage(sp)
    sp = make_split("spatiotemporal", st, cfg)
    assert not set(sp.stations["test"]) & set(sp.stations["train"])
    assert pd.Timestamp(sp.periods["val"][0]) - pd.Timestamp("2006-01-01") == pd.Timedelta(days=30)
    bad = Split("spatial", {"train": ["a"], "val": ["b"], "test": ["a"]}, sp.periods)
    with pytest.raises(AssertionError):
        check_leakage(bad)


def test_metrics_and_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.1, 5000), 0, 1)
    m = classification_metrics(y, p, best_f1_threshold(y, p))
    assert m["auprc"] > 0.5 and m["auroc"] > 0.9 and m["auprc_lift"] > 5
    assert expected_calibration_error(y, np.full(5000, y.mean())) < 0.02


def test_conformal_class_conditional_coverage():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.15, 20000), 0.001, 0.999)
    c = ClassConditionalConformal(0.1).fit(y[:10000], p[:10000])
    ev = c.evaluate(y[10000:], p[10000:])
    assert ev["coverage_events"] >= 0.87 and ev["coverage_non_events"] >= 0.87


def test_uncertainty_decomposition():
    agree = decompose(np.array([[0.9, 0.5], [0.9, 0.5]]))
    disagree = decompose(np.array([[0.99, 0.5], [0.01, 0.5]]))
    assert agree["epistemic"][0] < 1e-6 and disagree["epistemic"][0] > 0.5
    assert agree["confidence"][0] > agree["confidence"][1]


def test_models_forward_shapes(tmp_path):
    x = torch.randn(4, 30, 9)
    st = {"gis": torch.randn(4, 15), "rs": torch.randn(4, 4)}
    for name in ("lstm", "gru", "tcn", "transformer"):
        assert build_deep(name, 9, {"gis": 15, "rs": 4})(x, st).shape == (4,)
    m = HydroGeoAIModel(9, {"gis": 15, "rs": 4}, d_model=32, n_layers=1, n_heads=2)
    o = m(x, st)
    assert o["logit"].shape == (4,) and o["regime_logits"].shape == (4, 3)
    assert torch.allclose(o["modality_weights"].sum(1), torch.ones(4))
    m.eval()
    o2 = m(x, st, drop_modalities=["gis"])
    assert torch.all(o2["modality_weights"][:, 1] < 1e-6)


def test_encoder_pretraining_step_and_roundtrip(tmp_path):
    enc = HydroclimaticEncoder(EncoderConfig(n_in=9, d_model=32, n_layers=1, n_heads=2))
    pt = Pretrainer(enc, [0, 1, 2], [3, 4, 5])
    x = torch.randn(8, 30, 9)
    x[..., 3:6] = 1.0
    out = pt(x, np.random.default_rng(0))
    out["loss"].backward()
    assert {"recon", "contrastive", "multiscale", "loss"} <= set(out)
    m = span_mask(4, 60, 0.3, 5, np.random.default_rng(0))
    assert 0.28 <= m.mean() <= 0.4
    enc.save_pretrained(tmp_path / "enc")
    enc2 = HydroclimaticEncoder.from_pretrained(tmp_path / "enc")
    enc.eval(); enc2.eval()
    assert torch.allclose(enc(x)[1], enc2(x)[1])
