"""Data-foundation pipeline (Phase 1): raw -> QC -> homogeneity -> events -> static features -> dataset vN.

Outputs in data/processed/<dataset_version>/:
    observations.parquet     QC'd daily data with *_flag / *_qc columns + event labels + SPI
    stations.parquet         station metadata + static GIS/RS/meta features
    thresholds.parquet       reference-period event thresholds
    qc_flags.parquet, qc_audit.parquet, qc_gaps.parquet, station_issues.csv, qc_summary.json
    homogeneity.csv, trends.csv, etccdi_annual.parquet
    whiplash_events.parquet, whiplash_annual.csv
    observations.nc          (optional NetCDF export when netCDF4/h5netcdf available)
    metadata.json, provenance.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import data_dir, load_config, resolve
from ..events import fit_thresholds, label_events, whiplash
from ..features.indices import etccdi_annual
from ..features.spatial import station_static_features
from ..gis.raster import GeoRaster
from ..qc import homogeneity_report, run_qc, trend_report
from ..utils import dataframe_hash, get_logger
from . import synthetic
from .catalog import DatasetCatalog
from .ingest import read_tabular, read_vector
from .provenance import ProvenanceLog
from .schema import describe_observations

log = get_logger(__name__)


def load_raw(cfg: dict) -> dict:
    raw = data_dir() / "raw" / cfg.get("source", "synthetic")
    if cfg.get("source", "synthetic") == "synthetic":
        if not (raw / "observations.parquet").exists():
            log.info("Generating synthetic dataset -> %s", raw)
            synthetic.write(raw, cfg.get("synthetic", {}))
    prov = ProvenanceLog(cfg.get("dataset_name", "hydrogeoai-nepal"))
    obs_path = raw / "observations.parquet"
    if not obs_path.exists():
        obs_path = raw / "observations.csv"
    obs = read_tabular(obs_path, prov=prov)
    stations = pd.read_csv(raw / "stations.csv", dtype={"station_id": str})
    dem_path = raw / "dem.tif" if (raw / "dem.tif").exists() else raw / "dem"
    dem = GeoRaster.load(dem_path)
    rivers = read_vector(raw / "rivers.geojson")
    basins = read_vector(raw / "basins.geojson")
    outline = read_vector(raw / "outline.geojson") if (raw / "outline.geojson").exists() else None
    rs = pd.read_csv(raw / "remote_sensing.csv", dtype={"station_id": str}) \
        if (raw / "remote_sensing.csv").exists() else None
    return dict(obs=obs, stations=stations, dem=dem, rivers=rivers, basins=basins, outline=outline,
                rs=rs, prov=prov, raw_dir=raw)


def build_dataset(data_cfg_path="configs/data/default.yaml", events_cfg_path="configs/events/taxonomy.yaml",
                  force: bool = False) -> Path:
    cfg = load_config(data_cfg_path)
    ev_cfg = load_config(events_cfg_path)
    ref = tuple(cfg["reference_period"])
    r = load_raw(cfg)
    prov: ProvenanceLog = r["prov"]
    outline_ring = r["outline"]["features"][0]["geometry"]["coordinates"][0] if r["outline"] else None

    qc = run_qc(r["obs"], r["stations"], cfg.get("qc"), r["dem"], outline_ring)
    prov.steps += qc.provenance.steps
    obs = qc.observations

    log.info("Homogeneity & trend analysis")
    homog = homogeneity_report(obs, r["stations"])
    trends = trend_report(obs)
    prov.record("homogeneity", "Pettitt/SNHT/Buishand on raw and reference-relative annual series; "
                "data NOT adjusted (flags only)", {"alpha": 0.05})

    log.info("Event labelling (thresholds from reference period %s..%s)", *ref)
    thr = fit_thresholds(obs, ref, ev_cfg)
    labeled = label_events(obs, thr, ref, ev_cfg)
    prov.record("event_labels", "Event taxonomy labels + SPI30/SPI90", {"reference_period": ref},
                n_affected=len(labeled))
    etccdi = etccdi_annual(labeled, thr)

    log.info("Whiplash detection")
    wl = whiplash.detect(labeled, ev_cfg)
    wl_annual = whiplash.annual_metrics(wl, labeled)

    log.info("Static GIS / remote-sensing features")
    static = station_static_features(r["stations"], r["dem"], r["rivers"], r["rs"], obs, ref)
    prov.record("static_features", "Terrain (Horn slope/aspect), river distance, land cover, RS summaries",
                {"dem": r["dem"].metadata()})

    version = cfg.get("dataset_version") or dataframe_hash(labeled[["station_id", "date", "precip_qc", "tmax_qc", "tmin_qc"]])[:12]
    out = data_dir() / "processed" / version
    if out.exists() and force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out / "observations.parquet", index=False)
    static.to_parquet(out / "stations.parquet", index=False)
    thr.to_parquet(out / "thresholds.parquet", index=False)
    qc.flags.to_parquet(out / "qc_flags.parquet", index=False)
    qc.audit.to_frame().to_parquet(out / "qc_audit.parquet", index=False)
    qc.gaps.to_parquet(out / "qc_gaps.parquet", index=False)
    qc.station_issues.to_csv(out / "station_issues.csv", index=False)
    qc.completeness["annual"].to_parquet(out / "completeness_annual.parquet", index=False)
    (out / "qc_summary.json").write_text(json.dumps(qc.summary, indent=2, default=str))
    homog.to_csv(out / "homogeneity.csv", index=False)
    trends.to_csv(out / "trends.csv", index=False)
    etccdi.to_parquet(out / "etccdi_annual.parquet", index=False)
    wl.to_parquet(out / "whiplash_events.parquet", index=False)
    wl_annual.to_csv(out / "whiplash_annual.csv", index=False)
    for k in ("basins", "rivers", "outline"):
        if r[k] is not None:
            (out / f"{k}.geojson").write_text(json.dumps(r[k]))
    r["dem"].save(out / "dem")
    try:
        from .ingest import to_xarray
        cols = ["station_id", "date", "precip_qc", "tmax_qc", "tmin_qc"]
        to_xarray(labeled[cols].rename(columns=lambda c: c.replace("_qc", "")), static).to_netcdf(out / "observations.nc")
    except Exception as e:  # netCDF backend optional
        log.info("NetCDF export skipped (%s)", type(e).__name__)

    meta = describe_observations(
        labeled[["station_id", "date", "precip_qc", "tmax_qc", "tmin_qc", "rh_qc"]].rename(
            columns=lambda c: c.replace("_qc", "")), r["stations"],
        name=cfg.get("dataset_name", "hydrogeoai-nepal"), version=version, source=cfg.get("source", "synthetic"),
        qc_status="checked", processing_history=prov.to_records(),
        license="CC-BY-4.0 (synthetic)" if cfg.get("source") == "synthetic" else "see source licences",
        notes="SYNTHETIC development dataset" if cfg.get("source") == "synthetic" else "",
    ).to_dict()
    meta["reference_period"] = ref
    meta["qc_summary"] = qc.summary
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    prov.save(out / "provenance.json")
    gt = r["raw_dir"] / "ground_truth.json"
    if gt.exists():
        shutil.copy(gt, out / "ground_truth.json")

    cat = DatasetCatalog()
    try:
        rel = str(out.relative_to(resolve(".")))       # portable path inside the project
    except ValueError:
        rel = str(out)                                 # data directory outside the project
    key = cat.register(meta["name"], version, {"dir": rel},
                       {k: meta[k] for k in ("temporal_coverage", "n_stations", "n_records", "missing_pct", "source")})
    cat.set_status(key, "checked", "automated QC completed")
    (data_dir() / "processed" / "LATEST").write_text(version)
    log.info("Dataset %s written to %s", key, out)
    return out


def latest_dataset_dir() -> Path:
    p = data_dir() / "processed" / "LATEST"
    if not p.exists():
        raise FileNotFoundError("No processed dataset. Run `make data` (hydrogeoai data build).")
    return data_dir() / "processed" / p.read_text().strip()
