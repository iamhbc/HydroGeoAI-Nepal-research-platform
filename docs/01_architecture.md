# 1. Architecture

```
                    ┌───────────────────────┐
                    │  USERS: researchers / │
                    │  public / reviewers   │
                    └───────────┬───────────┘
              Next.js web (web/)  ·  Gradio Space (hf/space)  ·  CLI (hydrogeoai)
                                │
                     FastAPI backend (src/hydrogeoai/api)
        ┌──────────────┬────────┴──────────┬───────────────────┐
   data router     models router     experiments router    admin router
   (stations,      (predict,         (registry, tables,    (dataset lifecycle,
   climate, maps,  explain, models)  figures, compare)     model deploy/rollback,
   extremes, QC)                                           jobs, logs)
                                │
                  ML / GeoAI layer (models, uncertainty, explain, evaluation, experiments)
        ┌──────────────┬────────┴──────────┬───────────────────┐
   baselines &     Hydroclimatic       HydroGeoAI multimodal   uncertainty &
   deep models     Encoder (SSL)       fusion model            explainability
                                │
                  Data engineering (data, qc, features, events, gis)
        ┌──────────────┬────────┴──────────┬───────────────────┐
   meteorological   GIS (DEM, basins,   remote sensing       provenance, QC,
   station series   rivers, terrain)    summaries            homogeneity, events
                                │
   Storage: Parquet / NetCDF / GeoJSON / GeoRaster files + dataset catalog  (default, zero setup)
            PostgreSQL + PostGIS (docker compose; db/init.sql; scripts/load_postgis.py)
            SQLite or PostgreSQL experiment/model registry (SQLAlchemy)
            results/<experiment_id>/ artifacts;  hf/export/ staged Hugging Face repos
```

## Design decisions
* **Files first, database optional.** The scientific pipeline reads and writes versioned Parquet/NetCDF/GeoJSON
  so it runs anywhere. PostGIS is provided for spatial SQL and multi-user deployments.
* **Pure-numpy GIS core with optional GDAL stack.** `GeoRaster`, terrain derivatives and spatial joins work
  without GDAL. GeoPandas, Rasterio and PyProj are used automatically when installed (Shapefile, GeoTIFF, reprojection).
  CRS mismatches raise errors instead of being silently "fixed".
* **One architecture for the ablation.** A–F differ only in their inputs (and SSL initialisation), so differences
  are attributable to information, not capacity.
* **Hugging Face is one component.** It hosts released artifacts (dataset, model, Space). The research system
  itself (QC, experiments, registry, API, web) runs locally.
* **Transformers library not forced.** The encoder is a compact PyTorch Transformer with HF-compatible
  `config.json` + weights. A `transformers.PreTrainedModel` wrapper can be added for Hub integration without
  changing the science.

## Data flow
`raw/` → `ingest` (+provenance) → `run_qc` (flags, `*_qc`, audit) → `homogeneity_report` / `trend_report` →
`fit_thresholds` (reference period) → `label_events` (+SPI) → `whiplash.detect` → `station_static_features` →
`data/processed/<version>/` (+catalog) → `build_model_data` (split-aware normalisation, windows) → models →
`results/<experiment_id>/` + registry → API / web / Space.

## Module map
See the package docstring in `src/hydrogeoai/__init__.py` and the README layout section.
