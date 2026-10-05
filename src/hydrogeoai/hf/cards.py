"""Hugging Face dataset card and model card generation (Section 27).

Cards are generated from the actual artifacts (metadata.json, QC summary, experiment metrics) so
that documentation cannot drift from the data and models it describes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import ATTRIBUTION_DISCLAIMER


def _fmt(x, nd=4):
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def dataset_card(ds_dir: Path, namespace: str = "your-hf-username") -> str:
    meta = json.loads((ds_dir / "metadata.json").read_text())
    qc = meta.get("qc_summary", {})
    synthetic = "synthetic" in str(meta.get("source", "")).lower()
    st = pd.read_parquet(ds_dir / "stations.parquet")
    basins = st.basin.value_counts().to_dict()
    homog = pd.read_csv(ds_dir / "homogeneity.csv") if (ds_dir / "homogeneity.csv").exists() else None
    hom_txt = (homog.classification.value_counts().to_string() if homog is not None else "n/a")
    warn = ("\n> **SYNTHETIC DEVELOPMENT DATASET.** Values are simulated by `hydrogeoai.data.synthetic` to "
            "develop and test the pipeline. Station names only indicate approximate locations. Do not use for "
            "scientific conclusions about Nepal's climate.\n" if synthetic else "")
    return f"""---
license: {"cc-by-4.0" if synthetic else "other"}
language: [en]
pretty_name: HydroGeoAI-Nepal Dataset
tags: [climate, hydrology, geospatial, time-series, extreme-events, nepal, himalaya, geoai]
task_categories: [time-series-forecasting, tabular-classification]
size_categories: [100K<n<1M]
configs:
  - config_name: observations
    data_files: observations.parquet
  - config_name: stations
    data_files: stations.parquet
---

# HydroGeoAI-Nepal Dataset (version `{meta['version']}`)
{warn}
Daily station observations with quality-control flags, hydroclimatic event labels, standardised
indices and station-level geospatial / remote-sensing context, prepared for research on
hydroclimatic extremes under sparse, heterogeneous and nonstationary mountain observations.

## Provenance
* Source: `{meta['source']}`; acquisition/processing date: {meta['acquisition_date']}
* Processing history: {len(meta.get('processing_history', []))} recorded steps (`provenance.json`), every
  transformation with parameters and input/output content hashes. Observations are never overwritten:
  QC adds `<var>_flag` (0 ok, 1 suspicious, 2 error, 3 missing) and `<var>_qc` (errors removed).
* Dataset version = content hash of the QC'd core table.

## Coverage
* Temporal: {meta['temporal_coverage'][0]} to {meta['temporal_coverage'][1]} (daily)
* Spatial: lon {meta['spatial_coverage']['lon_min']:.2f}–{meta['spatial_coverage']['lon_max']:.2f}, lat {meta['spatial_coverage']['lat_min']:.2f}–{meta['spatial_coverage']['lat_max']:.2f}; CRS {meta['crs']}
* Stations: {meta['n_stations']} ({', '.join(f'{k}: {v}' for k, v in basins.items())})
* Reference period for thresholds / SPI fitting: {meta.get('reference_period')}

## Variables
| variable | units | description | missing % (after QC) |
|---|---|---|---|
""" + "\n".join(f"| {v} | {d['units']} | {d['long_name']} | {meta['missing_pct'].get(v, 'n/a')} |"
                for v, d in meta["variables"].items()) + f"""

Event labels (station-specific thresholds, reference period only): extreme_wet_day (>P95 wet days),
very_extreme_wet_day (>P99), extreme_rainfall_event (3-day >P99), consecutive_wet_days, rainfall_persistence,
dry_spell, precipitation_deficit (SPI-30<=-1), meteorological_drought (SPI-90<=-1), hot_day (TX90p),
warm_spell, cold_night (TN10p), hot_dry, wet_hot, rain_after_dryness. Labels are NaN where data are missing.

Static features: elevation, slope, aspect (northness/eastness), relief, distance to river, elevation-band
climate zone, land cover, NDVI, snow-cover fraction, LST, instrument type.

## Quality control summary
```
{json.dumps(qc, indent=2, default=str)}
```
Homogeneity classification (Pettitt/SNHT/Buishand on raw and neighbour-relative annual series):
```
{hom_txt}
```
Breaks are **flagged, not adjusted**; analyses can include/exclude flagged series.

## Preprocessing
Duplicate resolution, calendar regularisation, physical-range and consistency checks, unit-error
detection, robust spike detection, station-metadata checks against the DEM, SPI (gamma, Thom estimator)
and ETCCDI indices, whiplash event extraction.

## Licensing
{"CC-BY-4.0 for the synthetic data." if synthetic else "Licences follow the original providers (e.g. DHM Nepal data policy, CHIRPS public domain, ERA5-Land Copernicus licence). Redistribution of raw station data may be restricted; check before publishing."}

## Limitations
* Station density is very low at high elevation; results are least reliable above ~3000 m.
* Gauge undercatch (snow, wind) is not corrected.
* Grid products (when added) differ in scale from point gauges.
* Homogeneity tests detect but do not prove artifacts; metadata are often incomplete.

## Ethical considerations
Data describe physical climate, not people. Predictions are research outputs, not warnings; operational
early warning in Nepal is the mandate of DHM. Avoid presenting model outputs as official forecasts.

## Citation
```
@misc{{hydrogeoai_nepal_dataset,
  title  = {{HydroGeoAI-Nepal dataset}},
  note   = {{version {meta['version']}}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{namespace}/hydrogeoai-nepal-dataset}}}}
}}
```
"""


def model_card(model_dir: Path, headline: dict, hyp: pd.DataFrame, unc: dict, ds_dir: Path,
               namespace: str = "your-hf-username") -> str:
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    meta = json.loads((ds_dir / "metadata.json").read_text())
    synthetic = "synthetic" in str(meta.get("source", "")).lower()
    rows = []
    for mode, models in headline.items():
        for name in ("climatology", "hist_gbm", "lstm", "A_met_only", "F_full_multimodal", "F_ensemble"):
            if name in models:
                r = models[name]
                rows.append(f"| {mode} | {name} | {_fmt(r['auprc'])} [{_fmt(r['auprc_ci'][0])}, {_fmt(r['auprc_ci'][1])}] | "
                            f"{_fmt(r['brier_skill'], 3)} | {_fmt(r['ece'])} |")
    hyp_rows = "\n".join(f"| {r['hypothesis']} | {r['mode']} | {r['comparison']} | {r['verdict']} |"
                         for r in hyp.to_dict("records"))
    unc_rows = "\n".join(f"| {m} | {_fmt(u['conformal']['coverage'], 3)} | {_fmt(u['conformal']['coverage_events'], 3)} | "
                         f"{_fmt(u['conformal']['frac_ambiguous'], 3)} | {_fmt(u['auroc_epistemic_detects_errors'], 3)} |"
                         for m, u in unc.items())
    warn = ("\n> **Trained on the SYNTHETIC development dataset.** The metrics below validate the software "
            "pipeline only; they are not evidence about real hydroclimatic predictability in Nepal.\n" if synthetic else "")
    return f"""---
license: gpl-3.0
library_name: pytorch
tags: [climate, hydrology, geoai, time-series, extreme-events, uncertainty, self-supervised, nepal]
datasets: [{namespace}/hydrogeoai-nepal-dataset]
pipeline_tag: tabular-classification
---

# HydroGeoAI-Nepal multimodal model (v{cfg['version']})
{warn}
## Task
Probability that `{cfg['target']}` occurs at a station on day t+{cfg['lead_days']}, given the previous
{cfg['window_days']} days of precipitation/Tmax/Tmin (with missing-data masks) and static geospatial,
remote-sensing and metadata context. Thresholds are station-specific percentiles (reference period).

## Architecture
* Hydroclimatic Encoder: Transformer (d={cfg['d_model']}, layers={cfg['n_layers']}, heads={cfg['n_heads']}),
  self-supervised pretraining (masked-span reconstruction + temporal contrastive + multi-scale targets);
  encoder weights in `encoder/` (Hugging Face layout).
* Modality encoders (MLP) for: {', '.join(cfg['static_groups'])}; cross-modal attention + gated fusion;
  modality dropout during training.
* Heads: extreme-event logit; auxiliary SPI-30 regime (dry/normal/wet).
* Deep ensemble of {cfg['n_members']} members (`member_*.pt`) + temperature scaling (T={_fmt(cfg['temperature'], 3)})
  + class-conditional conformal prediction (alpha={cfg['conformal']['alpha']}).

## Training data
Dataset `{cfg['dataset_version']}` ({meta['source']}); split mode `{cfg['split_mode']}`; experiment `{cfg['experiment_id']}`;
code `{cfg['code_version']}`. Normalisation statistics from training stations/period only (`config.json`).

## Evaluation (test AUPRC with 95% station-block bootstrap CI)
| split | model | AUPRC [CI] | Brier skill vs climatology | ECE |
|---|---|---|---|---|
{chr(10).join(rows)}

AUPRC is the primary metric because events are rare (~2% of days); compare it with the base rate.

## Uncertainty
| split | conformal coverage | coverage on events | ambiguous sets | AUROC(epistemic → error) |
|---|---|---|---|---|
{unc_rows}

Output per prediction: probability, confidence (1 − normalised entropy), epistemic uncertainty,
conformal set and a low/moderate/high category.

## Hypotheses (automatically assessed; see results tables)
| H | split | comparison | verdict |
|---|---|---|---|
{hyp_rows}

## Intended use
Research on predictability, transferability and uncertainty of hydroclimatic extremes. Educational demos.

## Out-of-scope use
Operational warnings, insurance/financial decisions, or any use as an official forecast. Locations far from
training stations or above ~3000 m are poorly constrained.

## Limitations and known failure cases
* Sparse high-elevation stations; orographic extremes and snowfall are under-represented.
* Nonstationarity: skill can degrade outside the training period (see temporal-degradation table).
* Missing data and unseen basins increase epistemic uncertainty (see failure-analysis tables).

## Explainability
Integrated gradients, permutation importance and modality ablation are provided in the experiment
outputs. {ATTRIBUTION_DISCLAIMER}
"""
