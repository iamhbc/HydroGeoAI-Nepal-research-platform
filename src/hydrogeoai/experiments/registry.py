"""Research experiment registry + model registry + prediction log (Sections 22, 23, 25).

Default backend: SQLite file (experiments_registry/registry.db). Any SQLAlchemy URL works, e.g. the
PostGIS database from docker-compose. Every experiment stores dataset version, code version (git
commit or source hash), configuration, hyper-parameters, periods, geographic split, seed, metrics and
artifact paths. Every API prediction is logged with model + dataset version and parameters.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Session

from ..config import registry_url
from ..utils import utcnow


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    name = Column(String, index=True)
    status = Column(String, default="created")        # created | running | completed | failed
    created = Column(String, default=utcnow)
    finished = Column(String, nullable=True)
    dataset_version = Column(String)
    code_version = Column(String)
    seed = Column(Float)
    config = Column(JSON)
    split = Column(JSON)
    metrics = Column(JSON, default=dict)
    artifacts = Column(JSON, default=dict)
    notes = Column(Text, default="")


class Run(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, index=True)
    name = Column(String)
    model = Column(String)
    split_mode = Column(String)
    hyperparameters = Column(JSON)
    train_period = Column(JSON)
    val_period = Column(JSON)
    test_period = Column(JSON)
    stations = Column(JSON)
    metrics = Column(JSON)
    created = Column(String, default=utcnow)


class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True)                 # name@version
    name = Column(String, index=True)
    version = Column(String)
    path = Column(String)
    status = Column(String, default="registered")         # registered | evaluated | deployed | archived
    experiment_id = Column(String)
    dataset_version = Column(String)
    metrics = Column(JSON)
    card = Column(JSON)
    created = Column(String, default=utcnow)


class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now())
    model_version = Column(String)
    dataset_version = Column(String)
    parameters = Column(JSON)
    result_summary = Column(JSON)


class Registry:
    def __init__(self, url: str | None = None):
        self.url = url or registry_url()
        if self.url.startswith("sqlite:///"):
            from pathlib import Path
            Path(self.url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(self.url, future=True)
        Base.metadata.create_all(self.engine)

    # ---- experiments --------------------------------------------------------------------------
    def create_experiment(self, name: str, config: dict, dataset_version: str, code_version: str,
                          seed: int, split: dict | None = None, notes: str = "") -> str:
        eid = f"{name}-{utcnow()[:19].replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:6]}"
        with Session(self.engine) as s:
            s.add(Experiment(id=eid, name=name, status="running", dataset_version=dataset_version,
                             code_version=code_version, seed=seed, config=_j(config), split=_j(split or {}),
                             metrics={}, artifacts={}, notes=notes))
            s.commit()
        return eid

    def finish_experiment(self, eid: str, metrics: dict, artifacts: dict, status: str = "completed") -> None:
        with Session(self.engine) as s:
            e = s.get(Experiment, eid)
            e.metrics, e.artifacts, e.status, e.finished = _j(metrics), _j(artifacts), status, utcnow()
            s.commit()

    def log_run(self, eid: str, name: str, model: str, split, hyperparameters: dict, metrics: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with Session(self.engine) as s:
            s.add(Run(id=rid, experiment_id=eid, name=name, model=model, split_mode=split.mode,
                      hyperparameters=_j(hyperparameters), train_period=split.periods["train"],
                      val_period=split.periods["val"], test_period=split.periods["test"],
                      stations=_j(split.stations), metrics=_j(metrics)))
            s.commit()
        return rid

    def list_experiments(self) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Experiment).order_by(Experiment.created.desc())).all()
            return [_row(r, exclude=("config",)) for r in rows]

    def get_experiment(self, eid: str) -> dict | None:
        with Session(self.engine) as s:
            e = s.get(Experiment, eid)
            if e is None:
                return None
            d = _row(e)
            d["runs"] = [_row(r) for r in s.scalars(select(Run).where(Run.experiment_id == eid)).all()]
            return d

    # ---- models ------------------------------------------------------------------------------------
    def register_model(self, name: str, version: str, path: str, experiment_id: str, dataset_version: str,
                       metrics: dict, card: dict | None = None, status: str = "evaluated") -> str:
        mid = f"{name}@{version}"
        with Session(self.engine) as s:
            s.merge(ModelRecord(id=mid, name=name, version=version, path=path, status=status,
                                experiment_id=experiment_id, dataset_version=dataset_version,
                                metrics=_j(metrics), card=_j(card or {})))
            s.commit()
        return mid

    def set_model_status(self, mid: str, status: str) -> dict:
        with Session(self.engine) as s:
            m = s.get(ModelRecord, mid)
            if m is None:
                raise KeyError(mid)
            if status == "deployed":   # one deployed version per model name
                for o in s.scalars(select(ModelRecord).where(ModelRecord.name == m.name,
                                                             ModelRecord.status == "deployed")).all():
                    o.status = "evaluated"
            m.status = status
            s.commit()
            return _row(m)

    def rollback_model(self, name: str) -> dict | None:
        """Deploy the most recent evaluated version older than the currently deployed one."""
        with Session(self.engine) as s:
            ms = s.scalars(select(ModelRecord).where(ModelRecord.name == name).order_by(ModelRecord.created.desc())).all()
            dep = [m for m in ms if m.status == "deployed"]
            if not dep:
                return None
            older = [m for m in ms if m.created < dep[0].created and m.status != "archived"]
            if not older:
                return None
            target = older[0].id
        return self.set_model_status(target, "deployed")

    def list_models(self) -> list[dict]:
        with Session(self.engine) as s:
            return [_row(m, exclude=("card",)) for m in s.scalars(select(ModelRecord).order_by(ModelRecord.created.desc())).all()]

    def get_model(self, mid: str) -> dict | None:
        with Session(self.engine) as s:
            m = s.get(ModelRecord, mid)
            return _row(m) if m else None

    def deployed_model(self, name: str | None = None) -> dict | None:
        with Session(self.engine) as s:
            q = select(ModelRecord).where(ModelRecord.status == "deployed")
            if name:
                q = q.where(ModelRecord.name == name)
            m = s.scalars(q.order_by(ModelRecord.created.desc())).first()
            return _row(m) if m else None

    # ---- prediction log ---------------------------------------------------------------------------
    def log_prediction(self, model_version: str, dataset_version: str, parameters: dict, summary: dict) -> str:
        pid = uuid.uuid4().hex
        with Session(self.engine) as s:
            s.add(PredictionLog(id=pid, model_version=model_version, dataset_version=dataset_version,
                                parameters=_j(parameters), result_summary=_j(summary)))
            s.commit()
        return pid

    def recent_predictions(self, n: int = 50) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(n)).all()
            return [_row(r) for r in rows]


def _j(x: Any):
    """JSON-safe copy: numpy/pandas -> python, NaN/inf -> None."""
    return _clean(json.loads(json.dumps(x, default=_default)))


def _clean(x):
    import math
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _default(o):
    import numpy as np
    import pandas as pd
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _row(r, exclude=()) -> dict:
    return {c.name: getattr(r, c.name) for c in r.__table__.columns if c.name not in exclude}
