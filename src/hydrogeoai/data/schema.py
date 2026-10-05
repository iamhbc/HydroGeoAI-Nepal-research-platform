"""Canonical data schema and dataset metadata (Section 6 of the specification)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils import utcnow

# Canonical variables, units and physical meaning.
VARIABLES: dict[str, dict[str, str]] = {
    "precip":   {"units": "mm/day", "long_name": "Daily accumulated precipitation"},
    "tmax":     {"units": "degC",   "long_name": "Daily maximum 2 m air temperature"},
    "tmin":     {"units": "degC",   "long_name": "Daily minimum 2 m air temperature"},
    "rh":       {"units": "%",      "long_name": "Daily mean relative humidity"},
    "wind":     {"units": "m/s",    "long_name": "Daily mean wind speed"},
    "srad":     {"units": "MJ/m2/day", "long_name": "Daily incoming solar radiation"},
    "pressure": {"units": "hPa",    "long_name": "Daily mean surface pressure"},
}

STATION_COLUMNS = ["station_id", "name", "lat", "lon", "elevation_m", "basin", "province"]
OBS_KEY = ["station_id", "date"]


@dataclass
class DatasetMetadata:
    name: str
    version: str
    source: str
    acquisition_date: str
    temporal_coverage: tuple[str, str]
    spatial_coverage: dict[str, float]      # bbox
    variables: dict[str, dict[str, str]]
    crs: str
    n_stations: int
    n_records: int
    missing_pct: dict[str, float]
    qc_status: str = "unchecked"            # unchecked | checked | approved | archived
    processing_history: list[dict[str, Any]] = field(default_factory=list)
    license: str = "unspecified"
    citation: str = ""
    notes: str = ""
    created: str = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return asdict(self)


def describe_observations(obs: pd.DataFrame, stations: pd.DataFrame, *, name: str, version: str,
                          source: str, crs: str = "EPSG:4326", **extra) -> DatasetMetadata:
    variables = {v: VARIABLES[v] for v in VARIABLES if v in obs.columns}
    missing = {v: float(np.round(obs[v].isna().mean() * 100, 3)) for v in variables}
    return DatasetMetadata(
        name=name, version=version, source=source, acquisition_date=utcnow(),
        temporal_coverage=(str(obs.date.min().date()), str(obs.date.max().date())),
        spatial_coverage={"lon_min": float(stations.lon.min()), "lat_min": float(stations.lat.min()),
                          "lon_max": float(stations.lon.max()), "lat_max": float(stations.lat.max())},
        variables=variables, crs=crs, n_stations=int(stations.station_id.nunique()),
        n_records=int(len(obs)), missing_pct=missing, **extra,
    )


def validate_observations(obs: pd.DataFrame) -> list[str]:
    """Schema validation (structure only; physical QC lives in hydrogeoai.qc)."""
    problems = []
    for c in OBS_KEY:
        if c not in obs.columns:
            problems.append(f"missing key column '{c}'")
    if "date" in obs.columns and not np.issubdtype(obs["date"].dtype, np.datetime64):
        problems.append("'date' is not datetime64")
    if not any(v in obs.columns for v in ("precip", "tmax", "tmin")):
        problems.append("no primary variable (precip/tmax/tmin) present")
    return problems
