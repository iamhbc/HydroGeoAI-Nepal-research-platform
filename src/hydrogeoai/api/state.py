"""Shared API state: lazily loaded dataset, predictor cache, background jobs, log buffer."""
from __future__ import annotations

import json
import logging
import os
import threading
import traceback
import uuid
from collections import deque
from functools import cached_property
from pathlib import Path

import pandas as pd

from ..data.catalog import DatasetCatalog
from ..data.pipeline import latest_dataset_dir
from ..experiments.registry import Registry
from ..utils import utcnow


class _RingHandler(logging.Handler):
    def __init__(self, n=500):
        super().__init__()
        self.buf: deque[str] = deque(maxlen=n)
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

    def emit(self, record):
        self.buf.append(self.format(record))


LOG_BUFFER = _RingHandler()
logging.getLogger("hydrogeoai").addHandler(LOG_BUFFER)


class DataStore:
    def __init__(self):
        self.reload()

    def reload(self):
        for k in ("dir", "obs", "stations", "thresholds", "whiplash", "basins", "rivers", "outline", "metadata"):
            self.__dict__.pop(k, None)

    @cached_property
    def dir(self) -> Path:
        return latest_dataset_dir()

    @property
    def version(self) -> str:
        return self.dir.name

    @cached_property
    def obs(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "observations.parquet")

    @cached_property
    def stations(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "stations.parquet")

    @cached_property
    def whiplash(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "whiplash_events.parquet")

    @cached_property
    def metadata(self) -> dict:
        return json.loads((self.dir / "metadata.json").read_text())

    def geojson(self, name: str) -> dict | None:
        p = self.dir / f"{name}.geojson"
        return json.loads(p.read_text()) if p.exists() else None

    def table(self, name: str) -> pd.DataFrame | None:
        for ext, fn in ((".csv", pd.read_csv), (".parquet", pd.read_parquet)):
            p = self.dir / f"{name}{ext}"
            if p.exists():
                return fn(p)
        return None


class Jobs:
    """Minimal background job manager (threads). For production use a task queue (Celery/RQ/Arq)."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit(self, kind: str, fn, *args, **kw) -> str:
        jid = uuid.uuid4().hex[:10]
        self.jobs[jid] = {"id": jid, "kind": kind, "status": "running", "started": utcnow(), "result": None,
                          "error": None}

        def target():
            try:
                res = fn(*args, **kw)
                with self.lock:
                    self.jobs[jid].update(status="completed", result=_safe(res), finished=utcnow())
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.jobs[jid].update(status="failed", error=f"{type(e).__name__}: {e}",
                                          trace=traceback.format_exc()[-2000:], finished=utcnow())

        threading.Thread(target=target, daemon=True).start()
        return jid

    def list(self):
        return sorted(self.jobs.values(), key=lambda j: j["started"], reverse=True)


def _safe(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


class State:
    def __init__(self):
        self.data = DataStore()
        self.registry = Registry()
        self.catalog = DatasetCatalog()
        self.jobs = Jobs()
        self._predictors: dict[str, object] = {}
        self.admin_key = os.environ.get("HYDROGEOAI_ADMIN_KEY", "change-me")

    def predictor(self, model_id: str | None = None):
        from ..models.inference import Predictor

        rec = self.registry.get_model(model_id) if model_id else self.registry.deployed_model("hydrogeoai-nepal-model")
        if rec is None:
            return None, None
        if rec["id"] not in self._predictors:
            self._predictors[rec["id"]] = Predictor(rec["path"])
        return self._predictors[rec["id"]], rec


STATE: State | None = None


def get_state() -> State:
    global STATE
    if STATE is None:
        STATE = State()
    return STATE
