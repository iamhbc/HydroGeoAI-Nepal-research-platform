# 11. Paper and poster outline

**Working title:** Multimodal self-supervised representations for hydroclimatic extremes under sparse and
nonstationary mountain observations: evidence from Nepal

| Section | Content | Generated inputs |
|---|---|---|
| 1 Introduction | Hydroclimatic extremes in the Himalaya; sparse, heterogeneous, nonstationary observations; gap: transferability and uncertainty of ML | – |
| 2 Related work | climate ML; GeoAI; hydrological forecasting with DL; foundation/self-supervised models for time series; extreme-event detection; conformal prediction | – |
| 3 Data | stations, QC, homogeneity, terrain, RS; event taxonomy | `supplementary/qc_summary.json`, `homogeneity.csv`, dataset card |
| 4 Methodology | Hydroclimatic Encoder (SSL objectives) | `metrics/ssl_pretraining.json` |
| 5 Multimodal architecture | fusion, modality dropout, heads | model card |
| 6 Experimental design | spatial/temporal/spatio-temporal splits, leakage controls, metrics rationale | `supplementary/splits.json`, `docs/05` |
| 7 Results | benchmark; ablation A–F; transfer; label scarcity; missing data; calibration and uncertainty; failure analysis | `tables/*`, `figures/ablation_auprc.png`, `reliability_*.png`, `missing_data_*.png`, `label_scarcity.png`, `map_station_auprc_*.png` |
| 8 Hydroclimatic interpretation | seasonality, elevation dependence, attributions (with caveats), whiplash characteristics and period comparison | `metrics/explainability.json`, `tables/whiplash_decadal`, `metrics/whiplash_tests.json` |
| 9 Limitations | see `docs/08` | – |
| 10 Conclusions | H1–H6 verdicts with effect sizes and CIs | `tables/hypotheses_H1_H6` |

**Supplementary**: environment, configs, station lists, per-station metrics, QC audit summary, extra figures.

**Poster (one page):** question → data map → architecture diagram → ablation bar chart → transfer comparison →
reliability + uncertainty quadrant → whiplash panel → three take-home messages.

**CV framing:** *Independent Research Project — HydroGeoAI-Nepal: Multimodal Geospatial-Temporal Learning for
Hydroclimatic Extremes under Sparse and Nonstationary Mountain Observations.* Designed and built an open-source
geospatial-temporal AI framework integrating long-term meteorological observations, terrain attributes, spatial
information and remote-sensing features to characterise hydroclimatic extremes. Developed self-supervised temporal
representations and multimodal GeoAI models, and evaluated spatial/temporal transfer, uncertainty, missing-data
robustness and failure modes. Released reproducible datasets, model artifacts, benchmark experiments and an
interactive Hugging Face research application.
