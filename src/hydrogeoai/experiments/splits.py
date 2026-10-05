"""Leakage-aware spatial, temporal and spatio-temporal splits (Sections 14-15).

Random splits of individual days are NOT offered: neighbouring days and nearby stations are strongly
dependent, so random splits inflate skill. Splits are by station group and/or by contiguous period,
with an embargo (`gap_days`) at period boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Split:
    mode: str
    stations: dict[str, list[str]]                      # train/val/test -> station ids
    periods: dict[str, tuple[str, str]]                 # train/val/test -> (start, end) of sample dates
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "stations": self.stations, "periods": self.periods, "notes": self.notes}


def _embargo(period: tuple[str, str], gap_days: int, lead: int, is_first: bool) -> tuple[str, str]:
    s, e = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if not is_first:
        s = s + pd.Timedelta(days=gap_days)
    e = e - pd.Timedelta(days=lead)        # target date must stay inside the period
    return str(s.date()), str(e.date())


def station_groups(stations: pd.DataFrame, cfg: dict, seed: int) -> dict[str, list[str]]:
    sp = cfg["spatial"]
    rng = np.random.default_rng(seed)
    strat = sp.get("strategy", "basin")
    if strat == "basin":
        test = stations[stations.basin.isin(sp["test_groups"])].station_id.tolist()
    elif strat == "longitude_block":
        cut = np.quantile(stations.lon, 1 - sp.get("test_fraction", 0.3))
        test = stations[stations.lon >= cut].station_id.tolist()
    elif strat == "random_station":
        ids = stations.station_id.to_numpy(dtype=object)
        test = list(rng.choice(ids, size=int(len(ids) * sp.get("test_fraction", 0.3)), replace=False))
    else:
        raise ValueError(strat)
    rest = [s for s in stations.station_id if s not in set(test)]
    # validation stations: stratified by basin among the training basins
    rest_df = stations[stations.station_id.isin(rest)]
    val = []
    for _, g in rest_df.groupby("basin"):
        k = max(1, int(round(len(g) * sp.get("val_fraction_of_train_stations", 0.2))))
        if len(g) > 1:
            val += list(rng.choice(g.station_id.to_numpy(dtype=object), size=min(k, len(g) - 1), replace=False))
    train = [s for s in rest if s not in set(val)]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def make_split(mode: str, stations: pd.DataFrame, cfg: dict, seed: int = 0, lead: int = 1) -> Split:
    tp = cfg["temporal"]
    gap = tp.get("gap_days", 60)
    per = {k: _embargo(tuple(tp[k]), gap, lead, k == "train") for k in ("train", "val", "test")}
    all_ids = sorted(stations.station_id)
    if mode == "temporal":
        st = {"train": all_ids, "val": all_ids, "test": all_ids}
        notes = ["All stations; chronological train/val/test with embargo."]
    elif mode == "spatial":
        st = station_groups(stations, cfg, seed)
        per = {"train": per["train"], "val": per["train"], "test": per["train"]}
        notes = ["Unseen stations/basins, same period as training: isolates spatial transfer."]
    elif mode == "spatiotemporal":
        st = station_groups(stations, cfg, seed)
        notes = ["Unseen stations AND future period: most realistic and hardest setting."]
    else:
        raise ValueError(f"Unknown split mode {mode}")
    sp = Split(mode, st, per, notes)
    check_leakage(sp)
    return sp


def check_leakage(sp: Split) -> None:
    S = {k: set(v) for k, v in sp.stations.items()}
    P = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in sp.periods.items()}
    if sp.mode in ("spatial", "spatiotemporal"):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if S[a] & S[b]:
                raise AssertionError(f"Station leakage between {a} and {b}: {S[a] & S[b]}")
    if sp.mode in ("temporal", "spatiotemporal"):
        for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
            if P[a][1] >= P[b][0]:
                raise AssertionError(f"Temporal overlap between {a} {P[a]} and {b} {P[b]}")
