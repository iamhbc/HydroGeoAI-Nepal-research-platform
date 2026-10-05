"""Experiment registry endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..state import get_state

router = APIRouter(tags=["experiments"])


class ExperimentRequest(BaseModel):
    profile: Literal["quick", "main"] = "quick"
    modes: list[Literal["temporal", "spatial", "spatiotemporal"]] | None = None


@router.post("/experiments")
def launch(req: ExperimentRequest):
    from ...experiments.runner import run
    jid = get_state().jobs.submit(f"experiment:{req.profile}", run, f"configs/experiments/{req.profile}.yaml", req.modes)
    return {"job_id": jid, "status": "running",
            "note": "Track with GET /admin/jobs; the experiment appears in GET /experiments once started."}


@router.get("/experiments")
def experiments():
    return get_state().registry.list_experiments()


def _exp(eid: str) -> dict:
    e = get_state().registry.get_experiment(eid)
    if e is None:
        raise HTTPException(404, eid)
    return e


@router.get("/experiments/{eid}")
def experiment(eid: str):
    e = _exp(eid)
    d = Path((e.get("artifacts") or {}).get("results_dir", ""))
    e["tables"] = sorted(p.stem for p in (d / "tables").glob("*.csv")) if d.exists() else []
    e["figures"] = sorted(p.name for p in (d / "figures").glob("*.png")) if d.exists() else []
    e["summary_md"] = (d / "SUMMARY.md").read_text() if (d / "SUMMARY.md").exists() else None
    return e


@router.get("/experiments/{eid}/tables/{name}")
def table(eid: str, name: str):
    d = Path(_exp(eid)["artifacts"]["results_dir"]) / "tables" / f"{Path(name).stem}.csv"
    if not d.exists():
        raise HTTPException(404, name)
    df = pd.read_csv(d)
    return {"columns": list(df.columns), "rows": df.astype(object).where(df.notna(), None).to_dict("records")}


@router.get("/experiments/{eid}/figures/{name}")
def figure(eid: str, name: str):
    d = Path(_exp(eid)["artifacts"]["results_dir"]) / "figures" / Path(name).name
    if not d.exists():
        raise HTTPException(404, name)
    return FileResponse(d, media_type="image/png")


@router.get("/compare")
def compare(ids: str, metric: str = "auprc"):
    """Compare experiments: metric per (model, split) for each experiment id (comma-separated)."""
    out = []
    for eid in ids.split(","):
        e = _exp(eid.strip())
        for mode, models in (e.get("metrics") or {}).get("headline", {}).items():
            for model, m in models.items():
                out.append({"experiment": eid, "mode": mode, "model": model, metric: m.get(metric)})
    return out
