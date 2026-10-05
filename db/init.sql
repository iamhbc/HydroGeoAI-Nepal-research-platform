-- HydroGeoAI-Nepal PostGIS schema (loaded automatically by docker compose on first start)
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS datasets (
    key              TEXT PRIMARY KEY,           -- name@version
    name             TEXT NOT NULL,
    version          TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'registered' CHECK (status IN ('registered','checked','approved','archived')),
    source           TEXT,
    temporal_start   DATE,
    temporal_end     DATE,
    crs              TEXT NOT NULL DEFAULT 'EPSG:4326',
    metadata         JSONB,
    provenance       JSONB,
    created          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stations (
    station_id       TEXT PRIMARY KEY,
    name             TEXT,
    elevation_m      DOUBLE PRECISION,
    basin            TEXT,
    province         TEXT,
    instrument       TEXT,
    relocation_events JSONB,
    static_features  JSONB,                       -- slope, aspect, relief, river distance, land cover, RS
    geom             GEOMETRY(Point, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS stations_geom_idx ON stations USING GIST (geom);

CREATE TABLE IF NOT EXISTS observations (
    dataset_key      TEXT REFERENCES datasets(key),
    station_id       TEXT REFERENCES stations(station_id),
    date             DATE NOT NULL,
    precip           REAL, tmax REAL, tmin REAL, rh REAL,           -- original values (never modified)
    precip_qc        REAL, tmax_qc REAL, tmin_qc REAL, rh_qc REAL,  -- values used downstream
    precip_flag      SMALLINT, tmax_flag SMALLINT, tmin_flag SMALLINT, rh_flag SMALLINT,
    spi30            REAL, spi90 REAL,
    events           JSONB,                                         -- event taxonomy labels
    PRIMARY KEY (dataset_key, station_id, date)
);
CREATE INDEX IF NOT EXISTS observations_station_date_idx ON observations (station_id, date);

CREATE TABLE IF NOT EXISTS qc_audit (
    id               BIGSERIAL PRIMARY KEY,
    dataset_key      TEXT REFERENCES datasets(key),
    station_id       TEXT, date DATE, variable TEXT,
    original         REAL, corrected TEXT, flag TEXT, reason TEXT,
    created          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS basins (
    basin            TEXT PRIMARY KEY,
    province         TEXT,
    geom             GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS basins_geom_idx ON basins USING GIST (geom);

CREATE TABLE IF NOT EXISTS rivers (
    river            TEXT PRIMARY KEY,
    geom             GEOMETRY(LineString, 4326)
);

CREATE TABLE IF NOT EXISTS whiplash_events (
    id               BIGSERIAL PRIMARY KEY,
    dataset_key      TEXT REFERENCES datasets(key),
    station_id       TEXT REFERENCES stations(station_id),
    direction        TEXT, start_date DATE, transition_date DATE, end_date DATE,
    duration_days    INT, transition_days INT, intensity_spi_swing REAL, speed_spi_per_day REAL,
    spatial_extent   REAL
);

-- Experiment / model registry tables are created by SQLAlchemy (hydrogeoai.experiments.registry)
-- when HYDROGEOAI_REGISTRY_URL points at this database.

-- Example spatial queries
-- nearest stations:   SELECT station_id, ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(85.3, 27.7),4326)::geography)/1000 km
--                     FROM stations ORDER BY geom <-> ST_SetSRID(ST_MakePoint(85.3, 27.7),4326) LIMIT 5;
-- basin aggregation:  SELECT b.basin, avg(o.precip_qc) FROM observations o JOIN stations s USING (station_id)
--                     JOIN basins b ON ST_Contains(b.geom, s.geom) GROUP BY b.basin;
