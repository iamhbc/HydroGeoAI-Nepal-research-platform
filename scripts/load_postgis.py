"""Load the latest processed dataset into PostGIS (docker compose up -d db).

Requires: pip install "psycopg[binary]" geoalchemy2   (or `make setup-full`)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydrogeoai  # noqa: E402,F401
import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from hydrogeoai.data.pipeline import latest_dataset_dir  # noqa: E402
from hydrogeoai.events import EVENT_COLUMNS  # noqa: E402


def main():
    url = os.environ.get("HYDROGEOAI_POSTGIS_URL", "postgresql+psycopg://hydrogeoai:hydrogeoai@localhost:5432/hydrogeoai")
    eng = create_engine(url)
    ds = latest_dataset_dir()
    meta = json.loads((ds / "metadata.json").read_text())
    key = f"{meta['name']}@{meta['version']}"
    st = pd.read_parquet(ds / "stations.parquet")
    obs = pd.read_parquet(ds / "observations.parquet")
    with eng.begin() as c:
        c.execute(text("INSERT INTO datasets(key,name,version,status,source,temporal_start,temporal_end,metadata,provenance) "
                       "VALUES (:k,:n,:v,'checked',:s,:a,:b,:m,:p) ON CONFLICT (key) DO NOTHING"),
                  dict(k=key, n=meta["name"], v=meta["version"], s=meta["source"], a=meta["temporal_coverage"][0],
                       b=meta["temporal_coverage"][1], m=json.dumps(meta, default=str),
                       p=(ds / "provenance.json").read_text()))
        static_cols = [c for c in st.columns if c not in ("station_id", "name", "lat", "lon", "elevation_m", "basin",
                                                          "province", "instrument", "relocation_events")]
        for r in st.itertuples():
            c.execute(text("INSERT INTO stations VALUES (:id,:name,:e,:b,:p,:i,:rel,:sf, ST_SetSRID(ST_MakePoint(:lon,:lat),4326)) "
                           "ON CONFLICT (station_id) DO NOTHING"),
                      dict(id=r.station_id, name=r.name, e=r.elevation_m, b=r.basin, p=r.province, i=r.instrument,
                           rel=r.relocation_events, sf=json.dumps({k: getattr(r, k) for k in static_cols}, default=str),
                           lon=r.lon, lat=r.lat))
        for f in json.loads((ds / "basins.geojson").read_text())["features"]:
            c.execute(text("INSERT INTO basins VALUES (:b,:p, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:g),4326))) "
                           "ON CONFLICT (basin) DO NOTHING"),
                      dict(b=f["properties"]["basin"], p=f["properties"]["province"], g=json.dumps(f["geometry"])))
        for f in json.loads((ds / "rivers.geojson").read_text())["features"]:
            c.execute(text("INSERT INTO rivers VALUES (:r, ST_SetSRID(ST_GeomFromGeoJSON(:g),4326)) ON CONFLICT DO NOTHING"),
                      dict(r=f["properties"]["river"], g=json.dumps(f["geometry"])))
    ev_cols = [c for c in EVENT_COLUMNS if c in obs]
    out = obs[["station_id", "date", "precip", "tmax", "tmin", "rh", "precip_qc", "tmax_qc", "tmin_qc", "rh_qc",
               "precip_flag", "tmax_flag", "tmin_flag", "rh_flag", "spi30", "spi90"]].copy()
    out["events"] = obs[ev_cols].apply(lambda r: json.dumps({k: (None if pd.isna(v) else int(v)) for k, v in r.items()}), axis=1)
    out.insert(0, "dataset_key", key)
    out.to_sql("observations", eng, if_exists="append", index=False, chunksize=20000, method="multi")
    print(f"Loaded {len(out):,} observations for {key}")


if __name__ == "__main__":
    main()
