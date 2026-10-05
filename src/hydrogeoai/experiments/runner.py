"""Main experiment pipeline: `make reproduce-main-results`.

For each split mode (temporal / spatial / spatiotemporal):
  1. statistical + classical-ML baselines (and an HGB ablation over feature groups)
  2. deep sequence baselines (LSTM, GRU, TCN, Transformer)
  3. self-supervised pretraining of the Hydroclimatic Encoder (training windows only)
  4. ablation A-F with the HydroGeoAI multimodal architecture
  5. uncertainty: deep ensemble, MC dropout, temperature scaling, class-conditional conformal
  6. failure analysis, explainability, missing-data robustness, temporal degradation
Temporal mode additionally runs the label-scarcity experiment (H2).
Finally H1-H6 are assessed with station-block bootstrap evidence and everything is registered.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import ATTRIBUTION_DISCLAIMER, explain, reporting
from ..config import load_config, resolve, results_dir
from ..data.pipeline import latest_dataset_dir
from ..evaluation import (
    best_f1_threshold,
    bootstrap_ci,
    build_frame,
    classification_metrics,
    error_clustering,
    paired_bootstrap_diff,
    quadrant_table,
    reliability_curve,
)
from ..models import (
    ClimatologyBaseline,
    HydroGeoAIModel,
    PersistenceBaseline,
    build_deep,
    build_model_data,
    predict,
    pretrain_encoder,
    tabular_classifier,
    train_classifier,
)
from ..models.data import ModelData, SplitData
from ..qc.homogeneity import mann_kendall
from ..uncertainty import ClassConditionalConformal, apply_temperature, categorise, decompose, temperature_scale
from ..utils import code_version, get_device, get_logger, set_seed, sklearn_threads
from .registry import Registry
from .splits import make_split

log = get_logger(__name__)
from sklearn.metrics import average_precision_score  # noqa: E402

ABLATION_ORDER = ["A_met_only", "B_met_temporal", "C_met_gis", "D_met_gis_rs", "E_ssl_representation",
                  "F_full_multimodal"]


class Ctx:
    def __init__(self, cfg, md: ModelData, split, device, out: Path, reg: Registry, eid: str, stations):
        self.cfg, self.md, self.split, self.device, self.out, self.reg, self.eid = cfg, md, split, device, out, reg, eid
        self.stations = stations
        self.preds: dict[str, dict[str, np.ndarray]] = {}
        self.rows: list[dict] = []
        self.thresholds: dict[str, float] = {}

    def evaluate(self, name: str, family: str, p_val, p_test, extra: dict | None = None, hp: dict | None = None):
        md = self.md
        yv, yt = md.splits["val"].y, md.splits["test"].y
        thr = best_f1_threshold(yv, p_val)
        base_train = float(md.splits["train"].y.mean())
        m = classification_metrics(yt, p_test, thr, base_rate_ref=base_train)
        groups = md.splits["test"].stations
        n_boot = self.cfg["bootstrap"]["n"]
        m["auprc_ci_low"], m["auprc_ci_high"] = bootstrap_ci(yt, p_test, groups, average_precision_score, n_boot)
        m["val_auprc"] = float(average_precision_score(yv, p_val)) if yv.sum() else np.nan
        row = {"mode": self.split.mode, "model": name, "family": family, **m, **(extra or {})}
        self.rows.append(row)
        self.preds[name] = {"val": p_val, "test": p_test}
        self.thresholds[name] = thr
        self.reg.log_run(self.eid, f"{self.split.mode}/{name}", name, self.split, hp or {}, m)
        log.info("[%s] %-28s AUPRC %.4f [%.4f, %.4f]  BSS %.3f  F1 %.3f", self.split.mode, name, m["auprc"],
                 m["auprc_ci_low"], m["auprc_ci_high"], m["brier_skill_vs_climatology"], m["f1"])
        return m


# ------------------------------------------------------------------------------------------------
def _channels(md: ModelData, kind: str):
    return md.dyn_channels(kind)


def _static_dims(md: ModelData, groups):
    return {g: md.static[g].shape[1] for g in groups}


def run_baselines(ctx: Ctx):
    with sklearn_threads():
        _run_baselines(ctx)


def _run_baselines(ctx: Ctx):
    md, cfg = ctx.md, ctx.cfg
    for B in (ClimatologyBaseline(), PersistenceBaseline()):
        B.fit(md)
        ctx.evaluate(B.name, "statistical", B.predict_proba(md, "val"), B.predict_proba(md, "test"))
    feats = {"full": (["met", "temporal"], ["gis", "rs"])}
    X = {s: md.tabular_matrix(s, *feats["full"])[0] for s in ("train", "val", "test")}
    for name in cfg["baselines"]:
        if name in ("climatology", "persistence"):
            continue
        clf = tabular_classifier(name, cfg["seed"])
        if clf is None:
            log.info("Skipping %s (not installed)", name)
            continue
        t0 = time.time()
        clf.fit(X["train"], md.splits["train"].y)
        ctx.evaluate(name, "statistical" if name == "logistic" else "classical_ml",
                     clf.predict_proba(X["val"])[:, 1], clf.predict_proba(X["test"])[:, 1],
                     {"train_sec": time.time() - t0}, {"features": "met+temporal+gis+rs"})
    # tabular ablation with HistGradientBoosting: does the same information help a second model family?
    for tag, (dyn_g, st_g) in {"A_met_only": (["met"], []), "B_met_temporal": (["met", "temporal"], []),
                               "C_met_gis": (["met", "temporal"], ["gis"]),
                               "D_met_gis_rs": (["met", "temporal"], ["gis", "rs"])}.items():
        Xa = {s: md.tabular_matrix(s, dyn_g, st_g)[0] for s in ("train", "val", "test")}
        clf = tabular_classifier("hist_gbm", cfg["seed"])
        clf.fit(Xa["train"], md.splits["train"].y)
        ctx.evaluate(f"hgb_{tag}", "tabular_ablation", clf.predict_proba(Xa["val"])[:, 1],
                     clf.predict_proba(Xa["test"])[:, 1], {"variant": tag})


def run_deep_baselines(ctx: Ctx):
    md, cfg = ctx.md, ctx.cfg
    ch = _channels(md, "met_temporal")
    groups = ["gis", "rs"]
    for name in cfg["deep_models"]:
        set_seed(cfg["seed"])
        model = build_deep(name, len(ch), _static_dims(md, groups), cfg["deep"]["hidden"], cfg["deep"]["dropout"])
        t0 = time.time()
        model, info = train_classifier(model, md, ch, groups, cfg["deep"], cfg["seed"], ctx.device)
        ctx.evaluate(name, "deep", predict(model, md, "val", ch, groups, device=ctx.device),
                     predict(model, md, "test", ch, groups, device=ctx.device),
                     {"train_sec": time.time() - t0, "epochs_run": len(info["history"])}, cfg["deep"])


def _make_variant(md, spec, cfg, encoder, seed, device):
    ch = _channels(md, spec["dynamic"])
    groups = list(spec["static"])
    enc = copy.deepcopy(encoder) if spec.get("ssl") else None
    set_seed(seed)
    model = HydroGeoAIModel(len(ch), _static_dims(md, groups), d_model=cfg["pretrain"]["d_model"],
                            n_layers=cfg["pretrain"]["n_layers"], n_heads=cfg["pretrain"]["n_heads"],
                            dropout=cfg["deep"]["dropout"], encoder=enc)
    model, info = train_classifier(model, md, ch, groups, cfg["deep"], seed, device,
                                   encoder_lr_scale=0.3 if spec.get("ssl") else 1.0)
    return model, ch, groups, info


def run_ablation(ctx: Ctx, encoder):
    md, cfg = ctx.md, ctx.cfg
    models = {}
    for tag in ABLATION_ORDER:
        spec = cfg["ablation"][tag]
        t0 = time.time()
        model, ch, groups, info = _make_variant(md, spec, cfg, encoder, cfg["seed"], ctx.device)
        ctx.evaluate(tag, "hydrogeoai_ablation", predict(model, md, "val", ch, groups, device=ctx.device),
                     predict(model, md, "test", ch, groups, device=ctx.device),
                     {"variant": tag, "train_sec": time.time() - t0, "epochs_run": len(info["history"])}, spec)
        models[tag] = (model, ch, groups)
    return models


def run_uncertainty(ctx: Ctx, encoder, f_model):
    md, cfg = ctx.md, ctx.cfg
    spec = cfg["ablation"]["F_full_multimodal"]
    model0, ch, groups = f_model
    members = [model0]
    for k in range(1, cfg["uncertainty"]["ensemble_members"]):
        m, _, _, _ = _make_variant(md, spec, cfg, encoder, cfg["seed"] + 101 * k, ctx.device)
        members.append(m)
    pv = np.stack([predict(m, md, "val", ch, groups, device=ctx.device) for m in members])
    pt = np.stack([predict(m, md, "test", ch, groups, device=ctx.device) for m in members])
    dv, dt = decompose(pv), decompose(pt)
    ctx.evaluate("F_ensemble", "hydrogeoai", dv["probability"], dt["probability"],
                 {"variant": "F_full_multimodal", "members": len(members)})
    S = cfg["uncertainty"]["mc_dropout_samples"]
    mcv = np.stack([predict(model0, md, "val", ch, groups, device=ctx.device, mc_dropout=True) for _ in range(S)])
    mct = np.stack([predict(model0, md, "test", ch, groups, device=ctx.device, mc_dropout=True) for _ in range(S)])
    mdv, mdt = decompose(mcv), decompose(mct)
    ctx.evaluate("F_mc_dropout", "hydrogeoai", mdv["probability"], mdt["probability"], {"mc_samples": S})
    T = temperature_scale(md.splits["val"].y, dv["probability"])
    ctx.evaluate("F_ensemble_tempscaled", "hydrogeoai", apply_temperature(dv["probability"], T),
                 apply_temperature(dt["probability"], T), {"temperature": T})

    yv, yt = md.splits["val"].y, md.splits["test"].y
    conf = ClassConditionalConformal(cfg["uncertainty"]["conformal_alpha"]).fit(yv, dv["probability"])
    cov = conf.evaluate(yt, dt["probability"])
    sets_t = conf.predict_sets(dt["probability"])
    cat_t = categorise(dt["epistemic"], sets_t, dv["epistemic"])
    thr = ctx.thresholds["F_ensemble"]
    err = ((dt["probability"] >= thr).astype(int) != yt).astype(int)
    from sklearn.metrics import roc_auc_score
    unc = {
        "conformal": cov, "temperature": T,
        "methods": {k: {"ece_test": r["ece"], "brier_test": r["brier"], "auprc_test": r["auprc"]}
                    for k, r in ((r["model"], r) for r in ctx.rows if r["model"].startswith("F_"))},
        "error_rate_by_category": pd.Series(err).groupby(cat_t).mean().to_dict(),
        "count_by_category": pd.Series(cat_t).value_counts().to_dict(),
        "auroc_epistemic_detects_errors": float(roc_auc_score(err, dt["epistemic"])) if 0 < err.sum() < len(err) else np.nan,
        "auroc_total_entropy_detects_errors": float(roc_auc_score(err, dt["total_entropy"])) if 0 < err.sum() < len(err) else np.nan,
        "mean_confidence": float(dt["confidence"].mean()),
    }
    curves = {"F single": reliability_curve(yt, ctx.preds["F_full_multimodal"]["test"]),
              "F ensemble": reliability_curve(yt, dt["probability"]),
              "F ens. + temp.": reliability_curve(yt, apply_temperature(dt["probability"], T))}
    if "hist_gbm" in ctx.preds:
        curves["HistGBM"] = reliability_curve(yt, ctx.preds["hist_gbm"]["test"])
    reporting.reliability_plot(curves, ctx.out / "figures" / f"reliability_{ctx.split.mode}.png")
    return members, {"dv": dv, "dt": dt, "conformal": conf, "cat_t": cat_t, "T": T}, unc


def run_failure(ctx: Ctx, unc_state):
    md = ctx.md
    sd = md.splits["test"]
    obs_mask = md.dyn[..., md.channel_names.index("precip_obs")].numpy()
    wf = np.array([obs_mask[s, max(0, t - md.window + 1): t + 1].mean() for s, t in sd.idx])
    frame = build_frame(sd.y, unc_state["dt"]["probability"], ctx.thresholds["F_ensemble"], unc_state["cat_t"],
                        sd.dates, sd.stations, ctx.stations, ctx.split.stations["train"], wf)
    clus = error_clustering(frame)
    quad = quadrant_table(frame)
    reporting.write_table(clus, ctx.out / "tables" / f"failure_clustering_{ctx.split.mode}")
    reporting.write_table(quad, ctx.out / "tables" / f"failure_quadrants_{ctx.split.mode}")
    per_station = frame.groupby("station_id").apply(
        lambda g: average_precision_score(g.y, g.p) if 0 < g.y.sum() < len(g) else np.nan, include_groups=False)
    return frame, clus, quad, per_station


def run_explain(ctx: Ctx, f_model):
    md = ctx.md
    model, ch, groups = f_model
    rng = np.random.default_rng(ctx.cfg["seed"])
    yt = md.splits["test"].y
    pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
    sel = np.r_[rng.choice(pos, min(len(pos), 128), replace=False), rng.choice(neg, min(len(neg), 128), replace=False)]
    ig = explain.integrated_gradients(model, md, "test", sel, ch, groups, steps=24, device=ctx.device)
    perm = explain.channel_permutation_importance(model, md, "test", ch, groups, ctx.cfg["seed"], ctx.device)
    abl = explain.modality_ablation(model, md, "test", ch, groups, ctx.device)
    _, extra = predict(model, md, "test", ch, groups, device=ctx.device, return_extra=True)
    w = extra["modality_weights"]
    gate = dict(zip(["met"] + groups, w.mean(0).round(4).tolist())) if w is not None else {}
    cf = explain.counterfactual_precip(model, md, "test", sel[:64], ch, groups, device=ctx.device)
    res = {"integrated_gradients": ig, "permutation": perm, "modality_ablation": abl,
           "mean_modality_gate_weights": gate, "counterfactual_antecedent_precip": cf,
           "disclaimer": ATTRIBUTION_DISCLAIMER}
    reporting.bar(pd.Series(ig["channel_importance"]), ctx.out / "figures" / f"ig_channels_{ctx.split.mode}.png",
                  "Integrated gradients: dynamic channels", "mean |attribution|", ATTRIBUTION_DISCLAIMER)
    reporting.bar(pd.Series(ig["static_importance"]).sort_values().tail(15),
                  ctx.out / "figures" / f"ig_static_{ctx.split.mode}.png",
                  "Integrated gradients: static features (top 15)", "mean |attribution|", ATTRIBUTION_DISCLAIMER)
    return res


def _mask_dyn(md: ModelData, rate: float, seed: int) -> torch.Tensor:
    """Randomly remove a fraction of observed station-days (all variables) from the inputs."""
    g = torch.Generator().manual_seed(seed)
    dyn = md.dyn.clone()
    drop = torch.rand(dyn.shape[:2], generator=g) < rate
    for v, m in (("precip_log1p", "precip_obs"), ("tmax", "tmax_obs"), ("tmin", "tmin_obs")):
        vi, mi = md.channel_names.index(v), md.channel_names.index(m)
        dyn[..., vi][drop] = 0.0
        dyn[..., mi][drop] = 0.0
    return dyn


def run_missing(ctx: Ctx, models: dict):
    rows = []
    md = ctx.md
    yt = md.splits["test"].y
    for rate in ctx.cfg["robustness"]["missing_rates"]:
        dyn = _mask_dyn(md, rate, ctx.cfg["seed"]) if rate > 0 else None
        for tag in ("B_met_temporal", "E_ssl_representation", "F_full_multimodal"):
            model, ch, groups = models[tag]
            p = predict(model, md, "test", ch, groups, device=ctx.device, dyn=dyn)
            rows.append({"mode": ctx.split.mode, "variant": tag, "missing_rate": rate,
                         "auprc": average_precision_score(yt, p)})
    return pd.DataFrame(rows)


def run_temporal_degradation(ctx: Ctx):
    md = ctx.md
    sd = md.splits["test"]
    years = pd.DatetimeIndex(sd.dates).year
    rows = []
    for name in ("F_ensemble", "hist_gbm", "climatology", "B_met_temporal"):
        if name not in ctx.preds:
            continue
        p = ctx.preds[name]["test"]
        for y in np.unique(years):
            m = years == y
            if 0 < sd.y[m].sum() < m.sum():
                rows.append({"mode": ctx.split.mode, "model": name, "year": int(y),
                             "auprc": average_precision_score(sd.y[m], p[m]), "base_rate": sd.y[m].mean(),
                             "brier": float(np.mean((p[m] - sd.y[m]) ** 2))})
    return pd.DataFrame(rows)


def subset_train(md: ModelData, frac: float, seed: int) -> ModelData:
    rng = np.random.default_rng(seed)
    sd = md.splits["train"]
    keep = rng.random(len(sd.y)) < frac
    new = copy.copy(md)
    new.splits = dict(md.splits)
    new.splits["train"] = SplitData(sd.idx[keep], sd.y[keep], sd.regime[keep], sd.dates[keep], sd.stations[keep])
    return new


def run_label_scarcity(ctx: Ctx, encoder):
    rows = []
    cfg = ctx.cfg
    for frac in cfg.get("label_fractions", [0.05, 0.2, 1.0]):
        md_f = subset_train(ctx.md, frac, cfg["seed"])
        for tag in ("B_met_temporal", "E_ssl_representation"):
            model, ch, groups, _ = _make_variant(md_f, cfg["ablation"][tag], cfg, encoder, cfg["seed"], ctx.device)
            p = predict(model, ctx.md, "test", ch, groups, device=ctx.device)
            y = ctx.md.splits["test"].y
            lo, hi = bootstrap_ci(y, p, ctx.md.splits["test"].stations, average_precision_score, cfg["bootstrap"]["n"])
            rows.append({"label_fraction": frac, "variant": tag, "n_train": int(len(md_f.splits["train"].y)),
                         "n_train_events": int(md_f.splits["train"].y.sum()),
                         "auprc": average_precision_score(y, p), "auprc_ci_low": lo, "auprc_ci_high": hi,
                         "p_test": p})
    return rows


# ------------------------------------------------------------------------------------------------
def _verdict(ci_low, ci_high, favourable_positive=True):
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "inconclusive"
    if favourable_positive:
        return "supported" if ci_low > 0 else ("contradicted" if ci_high < 0 else "inconclusive")
    return "supported" if ci_high < 0 else ("contradicted" if ci_low > 0 else "inconclusive")


def assess_hypotheses(mode_ctx: dict[str, Ctx], label_rows, degradation: pd.DataFrame, unc: dict) -> list[dict]:
    H = []
    for mode, ctx in mode_ctx.items():
        y, g = ctx.md.splits["test"].y, ctx.md.splits["test"].stations
        n = ctx.cfg["bootstrap"]["n"]
        d = paired_bootstrap_diff(y, ctx.preds["F_full_multimodal"]["test"], ctx.preds["A_met_only"]["test"], g, n=n)
        H.append({"hypothesis": "H1", "mode": mode, "comparison": "F (full multimodal) - A (meteorology only), AUPRC",
                  **d, "verdict": _verdict(d["ci_low"], d["ci_high"])})
        d = paired_bootstrap_diff(y, ctx.preds["hgb_D_met_gis_rs"]["test"], ctx.preds["hgb_A_met_only"]["test"], g, n=n)
        H.append({"hypothesis": "H1", "mode": mode, "comparison": "HGB D (met+gis+rs) - HGB A (met only), AUPRC",
                  **d, "verdict": _verdict(d["ci_low"], d["ci_high"])})
        if mode in ("spatial", "spatiotemporal"):
            d = paired_bootstrap_diff(y, ctx.preds["C_met_gis"]["test"], ctx.preds["B_met_temporal"]["test"], g, n=n)
            H.append({"hypothesis": "H3", "mode": mode, "comparison": "C (+GIS) - B (no GIS) on unseen stations, AUPRC",
                      **d, "verdict": _verdict(d["ci_low"], d["ci_high"])})
    if label_rows:
        temporal = mode_ctx.get("temporal") or next(iter(mode_ctx.values()))
        y, g = temporal.md.splits["test"].y, temporal.md.splits["test"].stations
        lr = pd.DataFrame(label_rows)
        for frac, grp in lr.groupby("label_fraction"):
            pe = grp[grp.variant == "E_ssl_representation"].p_test.iloc[0]
            pb = grp[grp.variant == "B_met_temporal"].p_test.iloc[0]
            d = paired_bootstrap_diff(y, pe, pb, g, n=temporal.cfg["bootstrap"]["n"])
            H.append({"hypothesis": "H2", "mode": "temporal", "comparison": f"E (SSL) - B (scratch) at {frac:.0%} labels, AUPRC",
                      **d, "verdict": _verdict(d["ci_low"], d["ci_high"])})
    for mode, ctx in mode_ctx.items():
        r = {x["model"]: x for x in ctx.rows}
        if "F_ensemble" in r:
            drop = r["F_ensemble"]["auprc"] - r["F_ensemble"]["val_auprc"]
            dg = degradation[(degradation["mode"] == mode) & (degradation.model == "F_ensemble")]
            mk = mann_kendall(dg.auprc.to_numpy(), prewhiten=False) if len(dg) >= 5 else None
            H.append({"hypothesis": "H4", "mode": mode, "comparison": "test - validation AUPRC (F ensemble); MK trend over test years",
                      "diff": drop, "ci_low": np.nan, "ci_high": np.nan,
                      "mk_z_yearly_auprc": mk["z"] if mk else np.nan, "mk_p": mk["p_value"] if mk else np.nan,
                      "verdict": ("supported" if (mk and mk["p_value"] < 0.05 and mk["z"] < 0) or drop < -0.05 else
                                  "inconclusive" if mode != "spatial" else "not applicable (same period)")})
    for mode, u in unc.items():
        e = u["error_rate_by_category"]
        H.append({"hypothesis": "H5", "mode": mode,
                  "comparison": "error rate high-uncertainty vs low-uncertainty; AUROC(epistemic -> error)",
                  "diff": (e.get("high", np.nan) - e.get("low", np.nan)), "ci_low": np.nan, "ci_high": np.nan,
                  "auroc_epistemic": u["auroc_epistemic_detects_errors"],
                  "conformal_coverage_events": u["conformal"]["coverage_events"],
                  "verdict": "supported" if (e.get("high", 0) > e.get("low", 1) and
                                             (u["auroc_epistemic_detects_errors"] or 0) > 0.6) else "inconclusive"})
    H.append({"hypothesis": "H6", "mode": "all", "comparison": "regime-transition discovery",
              "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
              "verdict": "exploratory: see whiplash analysis; requires real observations and expert review"})
    return H


# ------------------------------------------------------------------------------------------------
def save_hf_model(members, ch, groups, md: ModelData, unc_state, cfg, out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    m0 = members[0]
    m0.encoder.save_pretrained(out_dir / "encoder")
    for i, m in enumerate(members):
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, out_dir / f"member_{i}.pt")
    config = {
        "model_type": "hydrogeoai-multimodal", "version": meta["version"], "target": md.target,
        "lead_days": meta["lead"], "window_days": md.window, "channels": [md.channel_names[c] for c in ch],
        "channel_indices": ch, "static_groups": groups, "static_features": {g: md.static_names[g] for g in groups},
        "static_dims": {g: md.static[g].shape[1] for g in groups}, "d_model": cfg["pretrain"]["d_model"],
        "n_layers": cfg["pretrain"]["n_layers"], "n_heads": cfg["pretrain"]["n_heads"],
        "dropout": cfg["deep"]["dropout"], "n_members": len(members), "normalization": md.norm,
        "decision_threshold": meta["threshold"], "temperature": unc_state["T"],
        "conformal": {"alpha": unc_state["conformal"].alpha, "q": unc_state["conformal"].q_},
        "val_epistemic": np.quantile(unc_state["dv"]["epistemic"], [0.5, 0.9]).tolist(),
        "dataset_version": meta["dataset_version"], "experiment_id": meta["experiment_id"],
        "split_mode": meta["split_mode"], "code_version": meta["code_version"],
    }
    from .registry import _j
    (out_dir / "config.json").write_text(json.dumps(_j(config), indent=2))
    return out_dir


def run(config_path: str = "configs/experiments/main.yaml", modes: list[str] | None = None) -> dict:
    cfg = load_config(config_path)
    ev_cfg = load_config(cfg["events_config"])
    set_seed(cfg["seed"])
    tgt = ev_cfg["prediction_target"]
    ds_dir = latest_dataset_dir()
    obs = pd.read_parquet(ds_dir / "observations.parquet")
    stations = pd.read_parquet(ds_dir / "stations.parquet")
    basins = json.loads((ds_dir / "basins.geojson").read_text()) if (ds_dir / "basins.geojson").exists() else None
    reg = Registry()
    cv = code_version(resolve("."))
    eid = reg.create_experiment(cfg["name"], cfg, ds_dir.name, cv, cfg["seed"],
                                notes=f"target={tgt['name']} lead={tgt['lead_days']}")
    out = results_dir() / eid
    for sub in ("tables", "figures", "metrics", "models", "supplementary"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log.info("Experiment %s | dataset %s | code %s -> %s", eid, ds_dir.name, cv, out)
    device = get_device(cfg["deep"].get("device", "auto"))
    t_start = time.time()
    mode_ctx, unc_all, missing_all, degr_all, label_rows, split_info = {}, {}, [], [], [], {}
    explain_all, per_station_all, ssl_info = {}, {}, {}
    best_model_bundle = None
    try:
        for mode in (modes or cfg["splits"]["modes"]):
            split = make_split(mode, stations, cfg["splits"], cfg["seed"], tgt["lead_days"])
            split_info[mode] = split.to_dict()
            log.info("=== split %s: %d/%d/%d stations; periods %s", mode, *(len(split.stations[k]) for k in ("train", "val", "test")),
                     split.periods)
            md = build_model_data(obs, stations, split, tgt["name"], cfg["window_days"], tgt["lead_days"],
                                  tgt.get("horizon_days", 1), cfg["feature_lags"], cfg["rolling_windows"],
                                  {"train": cfg["max_train_samples"], "val": cfg["max_eval_samples"],
                                   "test": cfg["max_eval_samples"]}, cfg["seed"])
            ctx = Ctx(cfg, md, split, device, out, reg, eid, stations)
            mode_ctx[mode] = ctx
            run_baselines(ctx)
            run_deep_baselines(ctx)
            encoder, info = pretrain_encoder(md, _channels(md, "met_temporal"), cfg["pretrain"], cfg["seed"], device)
            ssl_info[mode] = info
            encoder.save_pretrained(out / "models" / f"hydroclimatic_encoder_{mode}",
                                    {"pretrain_split": mode, "dataset_version": ds_dir.name})
            models = run_ablation(ctx, encoder)
            members, unc_state, unc = run_uncertainty(ctx, encoder, models["F_full_multimodal"])
            unc_all[mode] = unc
            frame, _, _, per_station = run_failure(ctx, unc_state)
            per_station_all[mode] = per_station
            explain_all[mode] = run_explain(ctx, models["F_full_multimodal"])
            missing_all.append(run_missing(ctx, models))
            degr_all.append(run_temporal_degradation(ctx))
            if mode == "temporal":
                label_rows = run_label_scarcity(ctx, encoder)
            if mode == "spatiotemporal" or best_model_bundle is None:
                best_model_bundle = (members, models["F_full_multimodal"][1], models["F_full_multimodal"][2], md,
                                     unc_state, mode)
            reporting.station_map(stations, per_station, basins, out / "figures" / f"map_station_auprc_{mode}.png",
                                  f"Per-station test AUPRC, F ensemble ({mode})", "viridis", "AUPRC")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---------------- tables, figures, hypotheses --------------------------------------------------
        bench = pd.DataFrame([r for c in mode_ctx.values() for r in c.rows])
        reporting.write_table(bench, out / "tables" / "benchmark_all")
        abl = bench[bench.family == "hydrogeoai_ablation"].copy()
        reporting.write_table(abl[["mode", "variant", "auprc", "auprc_ci_low", "auprc_ci_high", "auroc", "brier",
                                   "brier_skill_vs_climatology", "ece", "f1", "base_rate"]],
                              out / "tables" / "ablation_A_to_F")
        reporting.ablation_bars(abl, out / "figures" / "ablation_auprc.png")
        tab_abl = bench[bench.family == "tabular_ablation"]
        if len(tab_abl):
            reporting.ablation_bars(tab_abl.assign(variant=tab_abl.variant), out / "figures" / "ablation_hgb_auprc.png")
        main_cols = ["mode", "model", "family", "auprc", "auprc_ci_low", "auprc_ci_high", "auprc_lift", "auroc",
                     "brier", "brier_skill_vs_climatology", "ece", "precision", "recall", "f1", "base_rate", "n", "n_events"]
        reporting.write_table(bench[main_cols], out / "tables" / "benchmark_main")
        spatial = bench.pivot_table(index="model", columns="mode", values="auprc").reset_index()
        reporting.write_table(spatial, out / "tables" / "transfer_auprc_by_split")
        missing = pd.concat(missing_all, ignore_index=True)
        reporting.write_table(missing, out / "tables" / "missing_data_robustness")
        for mode, g in missing.groupby("mode"):
            reporting.line_plot(g, "missing_rate", "auprc", "variant", out / "figures" / f"missing_data_{mode}.png",
                                f"Missing-data robustness ({mode})", "test AUPRC")
        degradation = pd.concat(degr_all, ignore_index=True)
        reporting.write_table(degradation, out / "tables" / "temporal_degradation_by_year")
        for mode, g in degradation.groupby("mode"):
            reporting.line_plot(g, "year", "auprc", "model", out / "figures" / f"yearly_auprc_{mode}.png",
                                f"Test performance by year ({mode})", "AUPRC")
        if label_rows:
            lr = pd.DataFrame(label_rows).drop(columns=["p_test"])
            reporting.write_table(lr, out / "tables" / "label_scarcity")
            reporting.line_plot(lr, "label_fraction", "auprc", "variant", out / "figures" / "label_scarcity.png",
                                "Label scarcity: SSL vs scratch (temporal)", "test AUPRC")
        hyp = pd.DataFrame(assess_hypotheses(mode_ctx, label_rows, degradation, unc_all))
        reporting.write_table(hyp, out / "tables" / "hypotheses_H1_H6")
        reporting.write_json(unc_all, out / "metrics" / "uncertainty.json")
        reporting.write_json(explain_all, out / "metrics" / "explainability.json")
        reporting.write_json(ssl_info, out / "metrics" / "ssl_pretraining.json")
        reporting.write_json(split_info, out / "supplementary" / "splits.json")
        reporting.write_json({m: c.thresholds for m, c in mode_ctx.items()}, out / "metrics" / "decision_thresholds.json")
        for f in ("whiplash_annual.csv", "homogeneity.csv", "trends.csv", "qc_summary.json", "station_issues.csv"):
            if (ds_dir / f).exists():
                (out / "supplementary" / f).write_bytes((ds_dir / f).read_bytes())
        _whiplash_supplement(ds_dir, ev_cfg, out)

        members, ch, groups, md, unc_state, mmode = best_model_bundle
        f_row = next(r for r in mode_ctx[mmode].rows if r["model"] == "F_ensemble")
        version = f"1.0.0+{eid.split('-')[-1]}"
        model_dir = save_hf_model(members, ch, groups, md, unc_state, cfg, out / "models" / "hydrogeoai-nepal-model",
                                  {"version": version, "lead": tgt["lead_days"], "threshold": mode_ctx[mmode].thresholds["F_ensemble"],
                                   "dataset_version": ds_dir.name, "experiment_id": eid, "split_mode": mmode,
                                   "code_version": cv})
        headline = {m: {r["model"]: {"auprc": r["auprc"], "auprc_ci": [r["auprc_ci_low"], r["auprc_ci_high"]],
                                     "brier_skill": r["brier_skill_vs_climatology"], "ece": r["ece"]}
                        for r in c.rows} for m, c in mode_ctx.items()}
        reporting.write_json(headline, out / "metrics" / "headline.json")
        from ..hf.cards import model_card
        (model_dir / "README.md").write_text(model_card(model_dir, headline, hyp, unc_all, ds_dir))
        mid = reg.register_model("hydrogeoai-nepal-model", version, str(model_dir), eid, ds_dir.name,
                                 {"split_mode": mmode, **{k: f_row[k] for k in ("auprc", "auroc", "brier", "ece", "f1")}})
        # The main profile's model replaces smoke-test models; other profiles only deploy if nothing is deployed.
        if cfg["name"] == "main" or reg.deployed_model("hydrogeoai-nepal-model") is None:
            reg.set_model_status(mid, "deployed")
        artifacts = {"results_dir": str(out), "model_dir": str(model_dir), "model_id": mid}
        summary = {"headline": headline, "hypotheses": hyp.to_dict("records"), "runtime_min": (time.time() - t_start) / 60}
        reg.finish_experiment(eid, summary, artifacts)
        (out / "SUMMARY.md").write_text(_summary_md(eid, ds_dir.name, cv, bench, hyp, unc_all, ssl_info, time.time() - t_start))
        log.info("Done in %.1f min. Results: %s", (time.time() - t_start) / 60, out)
        return {"experiment_id": eid, "results_dir": str(out), "model_id": mid}
    except Exception:
        reg.finish_experiment(eid, {}, {"results_dir": str(out)}, status="failed")
        raise


def _whiplash_supplement(ds_dir: Path, ev_cfg: dict, out: Path):
    from ..events import whiplash
    p = ds_dir / "whiplash_events.parquet"
    if not p.exists():
        return
    ev = pd.read_parquet(p)
    labeled = pd.read_parquet(ds_dir / "observations.parquet", columns=["station_id", "date", "spi30"])
    dec = whiplash.decadal_comparison(ev, labeled, ev_cfg["whiplash"]["decades"])
    reporting.write_table(dec["table"], out / "tables" / "whiplash_decadal")
    annual = pd.read_csv(ds_dir / "whiplash_annual.csv")
    tr = whiplash.trend_tests(annual, ev, labeled)
    reporting.write_json({"decadal_tests": {k: v for k, v in dec.items() if k != "table"},
                          "trends": {k: v for k, v in tr.items() if k != "station_trends"},
                          "station_trends": tr["station_trends"].to_dict("records")},
                         out / "metrics" / "whiplash_tests.json")
    reporting.line_plot(annual.assign(series="network"), "year", "frequency_per_station_year", "series",
                        out / "figures" / "whiplash_frequency.png", "Whiplash events per station-year", "events / station-year")


def _summary_md(eid, dv, cv, bench, hyp, unc, ssl, secs) -> str:
    lines = [f"# Experiment {eid}", "", f"* dataset version: `{dv}`", f"* code version: `{cv}`",
             f"* runtime: {secs / 60:.1f} min", "",
             "> If the dataset is the SYNTHETIC development dataset, these numbers validate the pipeline only "
             "and carry no scientific meaning about Nepal's climate.", "", "## Test AUPRC by model and split", ""]
    piv = bench.pivot_table(index="model", columns="mode", values="auprc").round(4)
    lines += [piv.to_string(), "", "## Hypotheses", ""]
    for r in hyp.to_dict("records"):
        d = r.get("diff")
        lines.append(f"* **{r['hypothesis']}** [{r['mode']}] {r['comparison']}: "
                     f"diff={d if d is None or not np.isfinite(d) else round(d, 4)} "
                     f"CI=[{r.get('ci_low')}, {r.get('ci_high')}] -> **{r['verdict']}**")
    lines += ["", "## Uncertainty", ""]
    for m, u in unc.items():
        lines.append(f"* {m}: conformal coverage={u['conformal']['coverage']:.3f} (events {u['conformal']['coverage_events']:.3f}), "
                     f"error rate by category={ {k: round(v, 4) for k, v in u['error_rate_by_category'].items()} }")
    lines += ["", "## Self-supervised pretraining", ""]
    for m, s in ssl.items():
        lines.append(f"* {m}: held-out masked reconstruction MSE={s['val_recon_mse']:.4f} vs naive {s['val_naive_mse_zero']:.4f}")
    lines += ["", f"_{ATTRIBUTION_DISCLAIMER}_"]
    return "\n".join(lines)
