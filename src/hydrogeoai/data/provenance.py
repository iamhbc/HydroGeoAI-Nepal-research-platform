"""Provenance and audit logging.

Principle: scientific data are never silently modified. Every transformation is a recorded step
with parameters, input/output content hashes and the number of affected values. Corrections are
written to *new* columns (e.g. `precip_qc`) and the original observation is preserved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import dataframe_hash, utcnow


@dataclass
class ProvenanceStep:
    step: str
    description: str
    params: dict[str, Any]
    input_hash: str | None
    output_hash: str | None
    n_affected: int | None
    timestamp: str = field(default_factory=utcnow)


@dataclass
class ProvenanceLog:
    dataset: str
    steps: list[ProvenanceStep] = field(default_factory=list)

    def record(self, step: str, description: str, params: dict | None = None,
               before: pd.DataFrame | None = None, after: pd.DataFrame | None = None,
               n_affected: int | None = None) -> None:
        self.steps.append(ProvenanceStep(
            step=step, description=description, params=params or {},
            input_hash=dataframe_hash(before) if before is not None else None,
            output_hash=dataframe_hash(after) if after is not None else None,
            n_affected=n_affected,
        ))

    def to_records(self) -> list[dict]:
        return [asdict(s) for s in self.steps]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"dataset": self.dataset, "steps": self.to_records()},
                                         indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "ProvenanceLog":
        d = json.loads(Path(path).read_text())
        return cls(d["dataset"], [ProvenanceStep(**s) for s in d["steps"]])


@dataclass
class AuditEntry:
    station_id: str
    date: str
    variable: str
    original: float | None
    corrected: float | None
    flag: str
    reason: str
    timestamp: str = field(default_factory=utcnow)


class AuditLog:
    """Value-level audit trail for every QC flag / correction."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def add(self, **kw) -> None:
        self.entries.append(AuditEntry(**kw))

    def extend_from_flags(self, df: pd.DataFrame, mask: pd.Series, variable: str, flag: str,
                          reason: str, corrected=None) -> None:
        sub = df.loc[mask, ["station_id", "date", variable]]
        for sid, date, val in sub.itertuples(index=False):
            self.entries.append(AuditEntry(str(sid), str(pd.Timestamp(date).date()), variable,
                                           None if pd.isna(val) else float(val), corrected, flag, reason))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.entries])

    def __len__(self) -> int:
        return len(self.entries)
