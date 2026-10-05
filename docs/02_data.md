# 2. Data

## Canonical schema
**Observations** (long format, one row per station-day): `station_id, date, precip [mm/day], tmax [°C], tmin [°C]`
plus optional `rh, wind, srad, pressure`. After QC, each variable `v` gains `v_flag` (0 ok, 1 suspicious,
2 error, 3 missing) and `v_qc` (value used downstream, errors removed). Original values are never modified.

**Stations**: `station_id, name, lat, lon (EPSG:4326), elevation_m, basin, province, instrument, relocation_events (JSON)`
plus derived static features (DEM elevation, slope, northness/eastness, relief, river distance, climate zone,
land cover, NDVI, snow cover, LST, reference-period completeness).

**Dataset metadata** (`metadata.json`): source, acquisition date, temporal and spatial coverage, variables and units,
CRS, missing-data percentage, QC status, processing history, licence, notes, reference period.

## Real data sources (to replace the synthetic generator)
| Product | Use | Access / notes |
|---|---|---|
| DHM Nepal station records (precip, Tmax, Tmin) | primary observations | Department of Hydrology and Meteorology; data request/licence; often wide CSV → `ingest.wide_to_long` |
| APHRODITE (Monsoon Asia, 0.25°) | gridded precipitation baseline / gap context | NetCDF → `ingest.read_gridded` |
| CHIRPS v2 (0.05°) | satellite-gauge precipitation | public domain; NetCDF/GeoTIFF |
| ERA5-Land (0.1°) | reanalysis predictors (T, precip, soil moisture) | Copernicus licence; `UNIT_CONVERSIONS` handles m→mm, K→°C |
| SRTM / Copernicus DEM GLO-30 | elevation, slope, aspect | GeoTIFF → `GeoRaster.load` (rasterio) |
| HydroSHEDS / HydroBASINS / HydroRIVERS | basins, rivers | Shapefile → `ingest.read_vector` |
| ESA WorldCover / MODIS MCD12Q1 | land cover | GeoTIFF |
| MODIS MOD13Q1 / MOD10A1 / MOD11A2 | NDVI, snow cover, LST | station-buffer summaries over the training period |

### How to plug in real data
1. Put files under `data/raw/<source>/` with `observations.(parquet|csv)`, `stations.csv`, `dem.tif`,
   `basins.geojson`, `rivers.geojson`, optional `remote_sensing.csv`, `outline.geojson`.
2. Set `source: <source>` in `configs/data/default.yaml`; adjust QC thresholds and the reference period.
3. `make data` builds a new, content-hashed dataset version; `make reproduce-main-results` re-runs everything.
4. Document licences in the dataset card before publishing. DHM station data may not be redistributable.

## Synthetic development dataset
`hydrogeoai.data.synthetic` creates 30 stations at approximate real locations (Terai to trans-Himalaya,
72–3450 m) for 1990–2024 with:
* spatially correlated, persistent weather (Gaussian copula over national/basin/local AR(1) fields);
* orographic precipitation climatology (rain shadows such as Jomsom, very wet Lumle/Pokhara), monsoon JJAS
  dominance, western-disturbance winter precipitation, pre-monsoon thunderstorms in the east;
* interannual monsoon variability; trends (+0.6%/yr wet-day intensity, −0.6%/yr dry-season occurrence,
  elevation-dependent warming);
* lapse-rate temperatures, rain-induced cooling, dry-spell heating, RH for ~70% of stations;
* heterogeneous record start, random missing blocks;
* **ground truth** in `ground_truth.json`: injected errors (negative precip, sentinels, spikes, Tmax/Tmin swaps,
  conflicting duplicates, a ×10 unit error year, an elevation metadata typo) and 4 inhomogeneities (2 documented,
  2 undocumented). The QC and homogeneity modules recover these (verified in tests and in the data build log).
* a physiographic DEM consistent with station elevations, basins and rivers (simplified geometry), and
  MODIS/WorldCover-like remote-sensing summaries.

## Provenance and versioning
* `provenance.json`: ordered steps with parameters, input/output content hashes, affected counts, timestamps.
* `qc_audit.parquet`: every flagged value with original, correction status and reason.
* Dataset version = SHA-256 prefix of the QC'd core table; the catalog (`data/catalog.json`) tracks the lifecycle
  `registered → checked → approved → archived`.
* `observations.nc`: CF-style NetCDF export (station × time) for interoperability.

## CRS policy
Station and vector data are stored in EPSG:4326; terrain derivatives handle geographic cell sizes explicitly.
Projected work uses EPSG:32645 (UTM 45N). Mixing CRSs raises `CRSMismatchError`; reprojection must be explicit
(`gis.raster.reproject_points`, GeoPandas `to_crs`), and it is recorded.
