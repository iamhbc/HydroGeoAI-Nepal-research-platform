"""Admin / researcher endpoints (require X-API-Key == HYDROGEOAI_ADMIN_KEY)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from ...config import data_dir, resolve
from ...data.ingest import read_tabular
from ...data.provenance import ProvenanceLog
from ...data.schema import validate_observations
from ..state import LOG_BUFFER, get_state

router = APIRouter(prefix="/admin", tags=["admin"])


def require_key(x_api_key: str | None = Header(None)):
    if x_api_key != get_state().admin_key:
        raise HTTPException(401, "invalid or missing X-API-Key")


@router.get("/status", dependencies=[Depends(require_key)])
def status():
    st = get_state()

    def size(p: Path):
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0

    jobs = st.jobs.list()
    return {
        "dataset_version": st.data.version, "datasets": st.catalog.list(),
        "models": st.registry.list_models(), "deployed_model": st.registry.deployed_model(),
        "jobs": jobs, "failed_jobs": [j for j in jobs if j["status"] == "failed"],
        "storage_mb": {"data": round(size(data_dir()), 1), "results": round(size(resolve("results")), 1),
                       "registry": round(size(resolve("experiments_registry")), 1)},
        "experiments": [{k: e[k] for k in ("id", "status", "created")} for e in st.registry.list_experiments()[:20]],
    }


@router.get("/logs", dependencies=[Depends(require_key)])
def logs(n: int = 200):
    return list(LOG_BUFFER.buf)[-n:]


@router.get("/jobs", dependencies=[Depends(require_key)])
def jobs():
    return get_state().jobs.list()


@router.post("/datasets/build", dependencies=[Depends(require_key)])
def build_dataset(force: bool = False):
    from ...data.pipeline import build_dataset as build

    def job():
        out = build(force=force)
        get_state().data.reload()
        return str(out)

    return {"job_id": get_state().jobs.submit("dataset:build", job)}


@router.post("/datasets/upload", dependencies=[Depends(require_key)])
async def upload(file: UploadFile = File(...)):
    """Upload a CSV/Parquet observation file into data/raw/uploads and validate its schema.
    The file is NOT merged automatically: a researcher must inspect, run QC and approve."""
    dest = data_dir() / "raw" / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / Path(file.filename).name
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    prov = ProvenanceLog(path.stem)
    try:
        df = read_tabular(path, prov=prov)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not parse: {e}")
    return {"stored_as": str(path), "rows": len(df), "columns": list(df.columns),
            "schema_problems": validate_observations(df), "provenance": prov.to_records(),
            "preview": df.head(5).astype(str).to_dict("records")}


@router.post("/datasets/{key}/qc", dependencies=[Depends(require_key)])
def run_qc_on(key: str):
    """Re-run QC summary for a registered dataset directory (results are stored next to the data)."""
    st = get_state()
    e = st.catalog.get(key)
    d = resolve(e["files"]["dir"])
    from ...qc import run_qc
    obs = pd.read_parquet(d / "observations.parquet", columns=["station_id", "date", "precip", "tmax", "tmin"])
    stations = pd.read_parquet(d / "stations.parquet")
    res = run_qc(obs, stations)
    st.catalog.set_status(key, "checked", "QC re-run via admin API")
    return res.summary


@router.post("/datasets/{key}/{action}", dependencies=[Depends(require_key)])
def dataset_status(key: str, action: str):
    mapping = {"approve": "approved", "archive": "archived", "check": "checked"}
    if action not in mapping:
        raise HTTPException(400, f"action must be one of {list(mapping)}")
    try:
        return get_state().catalog.set_status(key, mapping[action])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@router.post("/models/{model_id}/deploy", dependencies=[Depends(require_key)])
def deploy(model_id: str):
    try:
        return get_state().registry.set_model_status(model_id, "deployed")
    except KeyError:
        raise HTTPException(404, model_id)


@router.post("/models/{model_id}/archive", dependencies=[Depends(require_key)])
def archive(model_id: str):
    return get_state().registry.set_model_status(model_id, "archived")


@router.post("/models/{name}/rollback", dependencies=[Depends(require_key)])
def rollback(name: str):
    r = get_state().registry.rollback_model(name)
    if r is None:
        raise HTTPException(409, "no earlier version available to roll back to")
    return r


@router.post("/models/register", dependencies=[Depends(require_key)])
def register(path: str, name: str = "hydrogeoai-nepal-model", version: str | None = None):
    """Register an existing model directory (must contain config.json with member weights)."""
    import json
    p = resolve(path)
    if not (p / "config.json").exists():
        raise HTTPException(400, "config.json not found in model directory")
    cfg = json.loads((p / "config.json").read_text())
    return {"model_id": get_state().registry.register_model(name, version or cfg.get("version", "manual"), str(p),
                                                            cfg.get("experiment_id", "manual"),
                                                            cfg.get("dataset_version", "?"), {}, status="registered")}


@router.post("/reload", dependencies=[Depends(require_key)])
def reload():
    get_state().data.reload()
    get_state()._predictors.clear()
    return {"dataset_version": get_state().data.version}
