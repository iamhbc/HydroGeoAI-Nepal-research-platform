"""Model registry, prediction and explanation endpoints."""
from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ... import ATTRIBUTION_DISCLAIMER
from ...utils import utcnow
from ..state import get_state

router = APIRouter(tags=["models"])


class PredictRequest(BaseModel):
    station_ids: list[str] | None = Field(None, description="explicit stations; or use basin/province")
    basin: str | None = None
    province: str | None = None
    start: str = "2023-06-01"
    end: str = "2023-09-30"
    phenomenon: Literal["extreme_wet_day"] = "extreme_wet_day"
    model_id: str | None = Field(None, description="defaults to the deployed model")
    calibrated: bool = True
    max_days: int = Field(800, le=4000)


class ExplainRequest(BaseModel):
    station_id: str
    issued: str
    model_id: str | None = None


@router.get("/models")
def models():
    reg = get_state().registry
    return {"models": reg.list_models(), "deployed": reg.deployed_model()}


@router.get("/models/{model_id}")
def model(model_id: str):
    m = get_state().registry.get_model(model_id)
    if m is None:
        raise HTTPException(404, model_id)
    from pathlib import Path
    p = Path(m["path"])
    m["config"] = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else None
    m["model_card"] = (p / "README.md").read_text() if (p / "README.md").exists() else None
    return m


@router.post("/predict")
def predict(req: PredictRequest):
    st = get_state()
    pred, rec = st.predictor(req.model_id)
    if pred is None:
        raise HTTPException(409, "No deployed model. Run `make reproduce-main-results` or deploy one via /admin.")
    if pred.cfg["target"] != req.phenomenon:
        raise HTTPException(400, f"model predicts {pred.cfg['target']}, not {req.phenomenon}")
    stations = st.data.stations
    if req.station_ids:
        ids = [s for s in req.station_ids if s in set(stations.station_id)]
    else:
        sel = stations
        if req.basin:
            sel = sel[sel.basin == req.basin]
        if req.province:
            sel = sel[sel.province == req.province]
        ids = sel.station_id.tolist()
    if not ids:
        raise HTTPException(404, "no stations selected")
    days = (pd.Timestamp(req.end) - pd.Timestamp(req.start)).days + 1
    if days * len(ids) > req.max_days * 40:
        raise HTTPException(400, "request too large; shorten the period or select fewer stations")
    obs = st.data.obs
    w = pred.cfg["window_days"]
    lo = pd.Timestamp(req.start) - pd.Timedelta(days=w + 5)
    hi = pd.Timestamp(req.end) + pd.Timedelta(days=pred.cfg["lead_days"] + 1)
    sub = obs[obs.station_id.isin(ids) & (obs.date >= lo) & (obs.date <= hi)]
    df = pred.predict(sub, stations, ids, req.start, req.end, req.calibrated)
    if df.empty:
        raise HTTPException(400, "no predictions possible for the requested period (data coverage?)")
    agg = df.groupby("station_id").agg(mean_probability=("probability", "mean"), max_probability=("probability", "max"),
                                       predicted_event_days=("predicted_event", "sum"),
                                       observed_event_days=("observed_event", lambda s: s.dropna().sum()),
                                       frac_high_uncertainty=("uncertainty", lambda s: (s == "high").mean()),
                                       mean_confidence=("confidence", "mean"))
    stx = stations.set_index("station_id")
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(stx.loc[s, "lon"]), float(stx.loc[s, "lat"])]},
         "properties": {"station_id": s, "name": stx.loc[s, "name"], **{k: _f(v) for k, v in r.items()}}}
        for s, r in agg.iterrows()]}
    timeseries = df.groupby("target_date").agg(probability=("probability", "mean"),
                                                 p_low=("probability", lambda s: s.quantile(0.1)),
                                                 p_high=("probability", lambda s: s.quantile(0.9)),
                                                 observed=("observed_event", lambda s: s.dropna().mean() if s.notna().any() else None),
                                                 epistemic=("epistemic", "mean")).reset_index()
    weights = {c.replace("weight_", ""): float(df[c].mean()) for c in df.columns if c.startswith("weight_")}
    eval_ = None
    known = df.dropna(subset=["observed_event"])
    if len(known) and 0 < known.observed_event.sum() < len(known):
        from sklearn.metrics import average_precision_score, roc_auc_score
        eval_ = {"n": int(len(known)), "events": int(known.observed_event.sum()),
                 "auprc": float(average_precision_score(known.observed_event, known.probability)),
                 "auroc": float(roc_auc_score(known.observed_event, known.probability)),
                 "brier": float(np.mean((known.probability - known.observed_event) ** 2)),
                 "note": "Retrospective evaluation; periods inside the training window are optimistic."}
    params = req.model_dump()
    summary = {"n_predictions": int(len(df)), "n_stations": len(ids), "mean_probability": float(df.probability.mean()),
               "frac_high_uncertainty": float((df.uncertainty == "high").mean())}
    pid = st.registry.log_prediction(rec["id"], st.data.version, params, summary)
    return {
        "prediction_id": pid, "timestamp": utcnow(), "model_version": rec["id"], "dataset_version": st.data.version,
        "code_version": pred.cfg.get("code_version"), "parameters": params,
        "training_split": pred.cfg.get("split_mode"), "training_experiment": pred.cfg.get("experiment_id"),
        "decision_threshold": pred.cfg["decision_threshold"], "summary": summary, "evaluation": eval_,
        "station_summary": geo, "timeseries": json.loads(timeseries.to_json(orient="records")),
        "modality_weights": weights,
        "predictions": json.loads(df.head(5000).to_json(orient="records")),
        "notes": ["probability = ensemble mean (temperature-calibrated)", "confidence = 1 - normalised entropy",
                  "uncertainty category from epistemic spread (ensemble disagreement); conformal_set reported separately", ATTRIBUTION_DISCLAIMER],
    }


@router.post("/explain")
def explain(req: ExplainRequest):
    st = get_state()
    pred, rec = st.predictor(req.model_id)
    if pred is None:
        raise HTTPException(409, "no deployed model")
    w = pred.cfg["window_days"]
    obs = st.data.obs
    t = pd.Timestamp(req.issued)
    sub = obs[(obs.station_id == req.station_id) & (obs.date >= t - pd.Timedelta(days=w + 5)) & (obs.date <= t)]
    if len(sub) < w:
        raise HTTPException(400, "insufficient data before the issue date")
    out = pred.explain(sub, st.data.stations, req.station_id, req.issued)
    out["dataset_version"] = st.data.version
    out["model_version"] = rec["id"]
    return out


@router.get("/predictions/log")
def prediction_log(n: int = 50):
    return get_state().registry.recent_predictions(n)


def _f(v):
    try:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return v
