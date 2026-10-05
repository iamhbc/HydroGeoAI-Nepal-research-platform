# 9. API reference

Start with `make api`. Interactive documentation is at `http://localhost:8000/docs` (OpenAPI).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, dataset version, deployed model |
| GET | `/stations?basin=&province=` | stations + static features |
| GET | `/stations/{id}` | metadata, record completeness, issues, homogeneity, trends, event rates |
| GET | `/datasets` · `/datasets/{version}` | catalog; metadata + provenance |
| GET | `/climate?variable=&station_id\|basin\|province=&start=&end=&agg=daily\|monthly\|annual\|climatology` | series |
| GET | `/extremes?event=&...` · `/extremes/taxonomy` | annual frequency, seasonality, recent events; definitions |
| GET | `/whiplash?station_id\|basin=&direction=` | events, decadal comparison + tests |
| GET | `/qc/summary` · `/qc/flags?station_id=` | QC summary, station issues, homogeneity; flags |
| GET | `/maps` · `/maps/stations?metric=` · `/maps/{basins,rivers,outline}` · `/maps/raster/{dem,slope}/meta` · `/image.png` | GeoJSON / raster layers |
| POST | `/predict` | `{station_ids \| basin \| province, start, end, phenomenon, model_id?, calibrated}` → per-day probability, confidence, epistemic, conformal set, uncertainty category, GeoJSON summary, time series, modality weights, retrospective evaluation, **model/dataset/code versions** (logged) |
| POST | `/explain` | `{station_id, issued}` → channel/lag/static attributions, modality ablation, disclaimer |
| GET | `/predictions/log` | recent logged predictions |
| GET | `/models` · `/models/{id}` | registry, config, model card |
| GET/POST | `/experiments` · `/experiments/{id}` · `/experiments/{id}/tables/{name}` · `/figures/{name}` · `/compare?ids=` | registry; launch `{profile: quick\|main, modes}` |
| * | `/admin/...` (header `X-API-Key`) | `status`, `logs`, `jobs`, `datasets/build`, `datasets/upload`, `datasets/{key}/{qc,approve,archive}`, `models/{id}/{deploy,archive}`, `models/{name}/rollback`, `models/register`, `reload` |

Set `HYDROGEOAI_ADMIN_KEY` (default `change-me`, dev only) and `HYDROGEOAI_CORS` before exposing the API.
