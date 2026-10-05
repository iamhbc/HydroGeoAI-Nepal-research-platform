# 7. Phased roadmap and status

| Phase | Scope | Status in this repository | Next steps with real data |
|---|---|---|---|
| 0 Research design (2–3 wk) | RQs, hypotheses, data sources, events, evaluation | ✅ `docs/00`, `03`, `05`, configs | refine hypotheses with supervisor; pre-register analysis plan |
| 1 Data foundation (4–8 wk) | ingestion, QC, metadata, PostGIS, NetCDF/Zarr | ✅ pipeline + catalog + PostGIS schema/loader + NetCDF export | obtain DHM data, adapt readers, tune QC thresholds, document licences → **dataset v1** |
| 2 Scientific baseline (3–5 wk) | climatology, statistical, RF, XGBoost, LSTM, Transformer | ✅ all implemented and benchmarked | run with real data; hyper-parameter search per baseline |
| 3 Self-supervised learning (4–8 wk) | Hydroclimatic Encoder v1 | ✅ masked/contrastive/multi-scale; label-scarcity test | longer pretraining; add unlabeled gridded products (ERA5-Land) in training regions |
| 4 GeoAI (4–8 wk) | DEM, slope, aspect, basins, spatial relations | ✅ terrain + static encoders; spatial splits | real DEM (Copernicus GLO-30), HydroBASINS; neighbour-graph encoder (GNN) as extension |
| 5 Multimodal (6–10 wk) | met + geo + RS fusion | ✅ HydroGeoAI v1 with gated cross-modal fusion | real MODIS time series as a dynamic RS modality |
| 6 Scientific experiments (6–10 wk) | spatial/temporal transfer, missing data, uncertainty, ablation, failure analysis | ✅ automated in the runner + H1–H6 verdicts | run `main` on real data; extend to other targets (3-day extremes, whiplash onset) |
| 7 Research platform | frontend, backend, API, GIS dashboard, inference, registry, admin | ✅ Next.js + FastAPI + MapLibre | authentication for private experiments; task queue (Celery/Arq) |
| 8 Hugging Face release | dataset, model, Space, cards, docs | ✅ export + cards + Space; manual publish | publish after licence check |
| 9 Paper | manuscript, supplementary, benchmark, figures, reproducibility package | 🟡 outputs auto-generated; outline in `docs/11` | write after real-data results |

## Suggested extensions (after v1)
* Graph neural network over the station network (neighbour information without test leakage).
* Gridded downscaling to ungauged locations with conformal intervals.
* Lead times > 1 day and multi-day event horizons; sub-seasonal drivers (ENSO, IOD, MJO indices).
* Physics-informed constraints (lapse rate, water balance) and comparison with process models.
