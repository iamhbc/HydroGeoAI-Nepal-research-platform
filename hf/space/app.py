"""HydroGeoAI-Nepal demo Space (Gradio).

Input -> Model -> Prediction -> Explanation -> Uncertainty, with model/dataset versions shown.
Runs locally too:  python hf/space/app.py   (after `make reproduce-quick` and `make hf-export`,
or directly from the repository using the deployed model in the local registry).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch  # noqa: F401  (load torch's OpenMP runtime first; see hydrogeoai/__init__.py)
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # bundled package in an exported Space
sys.path.insert(0, str(HERE.parents[1] / "src"))   # repository checkout

from hydrogeoai import ATTRIBUTION_DISCLAIMER  # noqa: E402
from hydrogeoai.models.inference import Predictor  # noqa: E402


def _locate():
    if (HERE / "model" / "config.json").exists():
        return HERE / "model", HERE / "demo_data"
    from hydrogeoai.data.pipeline import latest_dataset_dir
    from hydrogeoai.experiments.registry import Registry
    rec = Registry().deployed_model("hydrogeoai-nepal-model")
    if rec is None:
        raise SystemExit("No deployed model found. Run `make reproduce-quick` first.")
    return Path(rec["path"]), latest_dataset_dir()


MODEL_DIR, DATA_DIR = _locate()
PRED = Predictor(MODEL_DIR)
OBS = pd.read_parquet(DATA_DIR / "observations.parquet")
STATIONS = pd.read_parquet(DATA_DIR / "stations.parquet")
CHOICES = [f"{r.station_id} · {r.name} ({r.basin}, {int(r.elevation_m)} m)" for r in STATIONS.itertuples()]
SYNTHETIC = "synthetic" in str(STATIONS.get("source", pd.Series(["?"])).iloc[0])


def run(station_label: str, start: str, end: str):
    sid = station_label.split(" · ")[0]
    w = PRED.cfg["window_days"]
    lo = pd.Timestamp(start) - pd.Timedelta(days=w + 5)
    sub = OBS[(OBS.station_id == sid) & (OBS.date >= lo) & (OBS.date <= pd.Timestamp(end) + pd.Timedelta(days=2))]
    df = PRED.predict(sub, STATIONS, [sid], start, end)
    if df.empty:
        raise gr.Error("No data for this period; choose dates with observations.")
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    t = pd.to_datetime(df.target_date)
    ax[0].plot(t, df.probability, color="#2a6f97", label="P(extreme wet day)")
    ax[0].fill_between(t, (df.probability - 2 * df.ensemble_std).clip(0), df.probability + 2 * df.ensemble_std,
                       color="#2a6f97", alpha=0.2, label="ensemble ±2σ")
    ev = df[df.observed_event == 1]
    ax[0].scatter(pd.to_datetime(ev.target_date), ev.probability, color="#b5403a", zorder=3, s=18, label="observed event")
    ax[0].axhline(PRED.cfg["decision_threshold"], ls="--", color="k", lw=0.8, label="decision threshold")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_ylabel("probability")
    p = OBS[(OBS.station_id == sid) & OBS.date.between(t.min(), t.max())]
    ax[1].bar(p.date, p.precip_qc, color="#6b7280")
    ax[1].set_ylabel("precip (mm)")
    fig.tight_layout()
    last = df.iloc[df.probability.idxmax()]
    ex = PRED.explain(sub, STATIONS, sid, last.issued)
    top_ch = sorted(ex["channel_attribution"].items(), key=lambda kv: -abs(kv[1]))[:5]
    top_st = list(ex["top_static_attribution"].items())[:5]
    summary = (f"### Highest-probability day: {last.target_date}\n"
               f"* **Extreme-event probability:** {last.probability:.3f}\n"
               f"* **Prediction confidence:** {last.confidence:.2f} (1 − normalised entropy)\n"
               f"* **Uncertainty:** {last.uncertainty} (epistemic {last.epistemic:.4f}; conformal set {last.conformal_set})\n"
               f"* **Observed:** {'event' if last.observed_event == 1 else 'no event' if last.observed_event == 0 else 'unknown'}\n\n"
               "#### Important factors (model attribution)\n" +
               "\n".join(f"* {k}: {v:+.3f}" for k, v in top_ch + top_st) +
               f"\n\n_{ATTRIBUTION_DISCLAIMER}_\n\n"
               f"Model `{PRED.version}` · dataset `{PRED.cfg['dataset_version']}` · experiment `{PRED.cfg['experiment_id']}`")
    return fig, summary, df


with gr.Blocks(title="HydroGeoAI-Nepal") as demo:
    gr.Markdown("# HydroGeoAI-Nepal\nMultimodal geospatial-temporal model for **extreme wet days (next day)** with "
                "uncertainty and explanations." + ("\n\n> ⚠️ Running on the SYNTHETIC development dataset — "
                                                   "for demonstration of the method only." if SYNTHETIC else ""))
    with gr.Row():
        st = gr.Dropdown(CHOICES, value=CHOICES[0], label="Station")
        s = gr.Textbox("2021-06-01", label="Start (issue date)")
        e = gr.Textbox("2021-09-30", label="End")
    btn = gr.Button("Run analysis", variant="primary")
    plot = gr.Plot()
    md = gr.Markdown()
    table = gr.Dataframe(wrap=True)
    btn.click(run, [st, s, e], [plot, md, table])

if __name__ == "__main__":
    demo.launch(server_port=int(os.environ.get("PORT", 7860)))
