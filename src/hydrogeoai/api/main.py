"""HydroGeoAI-Nepal API (FastAPI).

Run:  make api   (uvicorn hydrogeoai.api.main:app --reload --port 8000)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .routers import admin, data, experiments, models
from .state import get_state

app = FastAPI(
    title="HydroGeoAI-Nepal API",
    version=__version__,
    description="Research API for multimodal geospatial-temporal analysis of hydroclimatic extremes. "
                "Every prediction is logged with model version, dataset version and parameters.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("HYDROGEOAI_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"], allow_headers=["*"],
)
for r in (data.router, models.router, experiments.router, admin.router):
    app.include_router(r)


@app.on_event("startup")
def _warm_cache() -> None:
    """Load the dataset and the landing-page summary in the background so the first visitor is fast."""
    import threading

    def warm():
        try:
            data.overview()
        except Exception:  # noqa: BLE001 - no dataset yet; /health reports it
            pass

    threading.Thread(target=warm, daemon=True).start()


@app.get("/health", tags=["system"])
def health():
    st = get_state()
    try:
        dv = st.data.version
        ok_data = True
    except FileNotFoundError:
        dv, ok_data = None, False
    dep = st.registry.deployed_model()
    return {"status": "ok" if ok_data else "degraded", "version": __version__, "dataset_version": dv,
            "deployed_model": dep["id"] if dep else None,
            "admin_key_is_default": st.admin_key == "change-me"}
