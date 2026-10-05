# HydroGeoAI-Nepal

**A Multimodal Geospatial-Temporal AI Framework for Hydroclimatic Extremes under Sparse and Nonstationary Mountain Observations**

*Learning transferable representations of hydroclimatic extremes across heterogeneous mountainous environments
using meteorological time series, geospatial information, remote sensing, and self-supervised learning.*

> **Central question.** Can multimodal geospatial-temporal representation learning improve the detection and
> characterization of hydroclimatic extremes across spatially heterogeneous and data-sparse mountainous regions?

This repository is a reproducible scientific research system, not a dashboard. The research contribution lies in the
hypotheses (H1–H6), the leakage-safe experimental design and the honest evaluation. The web platform and the
Hugging Face artifacts exist to make that work inspectable and reusable.

> ⚠️ **Data status.** The default dataset is **synthetic**. A generator simulates Nepal-like stations with realistic
> geography, monsoon, orography, trends, gaps, errors and inhomogeneities, all with known ground truth. Every module
> can therefore be developed and tested end to end before real observations are ingested (see [docs/02_data.md](docs/02_data.md)).
> Results computed on synthetic data validate the *software and methodology only*.

---

## What is implemented

| Layer | Implementation | Spec § |
|---|---|---|
| Ingestion | CSV, Parquet, NetCDF/HDF5/Zarr (xarray), GeoTIFF, GeoJSON, Shapefile, JSON API → canonical schema; unit conversions recorded | 6 |
| Provenance | Every step logged with parameters and input/output content hashes; value-level QC audit log; data never overwritten | 6–7 |
| Quality control | completeness (station/annual/seasonal/monthly), physical plausibility, duplicates, gaps, irregular intervals, jumps, robust spikes, stuck sensors, unit errors, Tmax<Tmin, station metadata vs DEM | 7 |
| Homogeneity | Pettitt, SNHT, Buishand, Mann-Kendall (TFPW), Sen's slope, BH-FDR; **raw vs neighbour-relative series + metadata** separates regional climate signal from observational artifacts (iterative reference selection) | 8 |
| Features & indices | lags/rolling/spells/calendar; SPI (Thom gamma), SPEI (Hargreaves, optional), ETCCDI-style annual indices; terrain (Horn slope/aspect), relief, river distance, land cover, RS summaries | 9 |
| Event taxonomy | 14 events: precipitation, dry, thermal, compound extremes | 4 |
| Whiplash engine | anomaly-based dry↔wet transitions: frequency, duration, intensity, speed, spatial extent; hypothesis-neutral decadal tests | 18 |
| GIS engine | CRS-aware `GeoRaster` (CRS/resolution/extent/transform/source), raster extraction, zonal stats, spatial joins, nearest station, IDW (with caveats); refuses to mix CRSs | 19 |
| Baselines | climatology, probabilistic persistence, logistic GLM, Tweedie GLM, Random Forest, HistGBM, XGBoost/LightGBM (optional), LSTM, GRU, TCN, Transformer | 10 |
| Hydroclimatic Encoder | Transformer encoder with self-supervised **masked-span reconstruction + temporal contrastive + multi-scale** objectives, HF-style `save_pretrained` | 11 |
| Multimodal model | met encoder + GIS/RS/metadata encoders → cross-modal attention + gated fusion, modality dropout, event + regime heads | 12 |
| Ablation A–F | identical architecture with information added step by step, plus an HGB ablation as a second model family | 13 |
| Spatial / temporal transfer | held-out basins, chronological splits with embargo, spatio-temporal (unseen stations *and* future) | 14–15 |
| Uncertainty | deep ensembles, MC dropout, temperature scaling, **class-conditional conformal**, epistemic/aleatoric decomposition | 16 |
| Explainability | integrated gradients, permutation importance, modality ablation, gate weights, counterfactuals; attribution ≠ causation everywhere | 17 |
| Evaluation | AUPRC (+ base rate), AUROC, Brier/BSS, ECE, reliability, F1 at validation threshold, CRPS, station-block bootstrap CIs, paired tests | 29–30 |
| Failure analysis | confident/uncertain × correct/wrong quadrants; error clustering by elevation, season, basin, station density, missingness (χ²) | 31 |
| Robustness | missing-data masking curves; label scarcity (SSL vs scratch); per-year degradation | 6, 13 |
| Experiment registry | SQLite/PostgreSQL via SQLAlchemy: experiments, runs, models (deploy/rollback), prediction log | 22–23 |
| Research outputs | `tables/ figures/ metrics/ models/ supplementary/` + `SUMMARY.md` + automated H1–H6 verdicts | 32 |
| API | FastAPI: stations, datasets, climate, extremes, whiplash, QC, maps, predict, explain, models, experiments, admin | 25 |
| Web platform | Next.js + TypeScript + MapLibre: dashboard, climate, extremes, map, run model, compare, experiments, dataset/QC, docs, admin | 20–24 |
| Hugging Face | dataset card, model card, Gradio Space; export staged locally, **publishing only by explicit command** | 27 |
| Reproducibility | `pyproject.toml`, `environment.yml`, Dockerfile, docker-compose (PostGIS), configs, seeds, `make reproduce-main-results`, tests, CI workflow (inactive until pushed) | 28 |

## Quickstart (local, macOS/Linux)

```bash
make setup                 # .venv reusing your installed torch/numpy + API deps
make data                  # synthetic raw data -> QC -> homogeneity -> events -> features -> versioned dataset
make reproduce-quick       # fast end-to-end experiment run (all splits; ~1 h on a laptop CPU)
make api                   # http://localhost:8000/docs
make web-install && make web   # http://localhost:3000
make test                  # unit + integration tests
```

The full study:

```bash
make reproduce-main-results    # all splits, ablation, uncertainty, XAI, robustness, H1-H6 (~2-4 h CPU)
```

Outputs land in `results/<experiment_id>/`. Open `SUMMARY.md` there first.

Hugging Face: `make hf-export` → `make hf-publish-dry` → `make hf-login` → `make hf-publish` (see [docs/10](docs/10_huggingface.md)).

Optional extras: `make setup-full` (GeoPandas/Rasterio/XGBoost/LightGBM/SHAP/HF/NetCDF/PostGIS drivers),
`make db-up && make db-load` (PostGIS), `make space` (Gradio demo, needs `pip install gradio`).

## Repository layout

```
configs/            data, event taxonomy, experiment (main/quick) and model configs
src/hydrogeoai/
  data/             ingest, schema, provenance, catalog (versioning), synthetic generator, pipeline
  qc/               quality control checks, homogeneity / change-point / trend tests
  features/         temporal features, spatial/RS/meta features, SPI/SPEI/ETCCDI
  events/           event taxonomy labelling, whiplash engine
  gis/              GeoRaster (CRS-aware), terrain, spatial operations
  models/           data windows, baselines, deep models, Hydroclimatic Encoder, multimodal model, training, inference
  uncertainty/      ensembles/MC decomposition, conformal, temperature scaling, categories
  explain/          integrated gradients, permutation, modality ablation, counterfactuals
  evaluation/       metrics + bootstrap, failure analysis
  experiments/      splits (leakage checks), registry, main runner (H1-H6)
  reporting/        tables and figures
  hf/               dataset/model cards, export + manual publish
  api/              FastAPI app and routers
  cli.py            `hydrogeoai` command
web/                Next.js research platform
hf/space/           Gradio Space (Input → Model → Prediction → Explanation → Uncertainty)
db/init.sql         PostGIS schema
scripts/            reproduce_main_results.py, load_postgis.py
tests/              unit + end-to-end tests
docs/               research design, methods, evaluation, reproducibility, roadmap, ethics, API, HF, paper outline
```

## Documentation

1. [Research design: questions, hypotheses, operationalisation](docs/00_research_design.md)
2. [Architecture](docs/01_architecture.md)
3. [Data: schema, sources, synthetic generator, provenance, versioning](docs/02_data.md)
4. [Event taxonomy and whiplash](docs/03_event_taxonomy.md)
5. [Methodology: QC, homogeneity, features, models, SSL, fusion, uncertainty, XAI](docs/04_methodology.md)
6. [Evaluation: splits, metrics, benchmark, validation, failure analysis](docs/05_evaluation.md)
7. [Reproducibility](docs/06_reproducibility.md)
8. [Phased roadmap and status](docs/07_roadmap.md)
9. [Ethics, limitations, attribution vs causation](docs/08_ethics_limitations.md)
10. [API reference](docs/09_api.md)
11. [Hugging Face release](docs/10_huggingface.md)
12. [Paper and poster outline](docs/11_paper_outline.md)
13. [Writing style for the user interface (ASD-STE100)](docs/12_writing_style.md)

## Git / GitHub / Hugging Face

The project is kept **local**: no git remote is configured and nothing is pushed or uploaded automatically.
When you are ready, see [docs/06_reproducibility.md](docs/06_reproducibility.md#publishing-later) for the steps
(`git init`, first commit, create the GitHub repo, `hydrogeoai hf export` and then `hydrogeoai hf publish --push`).

## Licence

Code: GPL-3.0 (see `LICENSE`). Synthetic data: CC-BY-4.0. Real data: the licences of the original providers apply.
