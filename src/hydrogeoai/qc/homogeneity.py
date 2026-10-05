"""Climate homogeneity, change-point and trend analysis (Section 8 of the specification).

Key scientific idea: a break in a station's raw series can be *climate* (shared by the region) or an
*observational artifact* (relocation, instrument change). We therefore test both
  (a) the raw candidate series, and
  (b) the candidate-minus-reference series (ratio for precipitation), where the reference is a
      correlation-weighted composite of neighbouring stations' anomalies.
A break in (b) is not shared regionally -> possible artifact; a break only in (a) -> regional signal.
Station metadata (documented relocations/instrument changes) are used to support interpretation.
None of these tests "prove" a cause; they prioritise series for expert review.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from ..gis.spatial import haversine_km


# ------------------------------------------------------------------------------------------------
# Change-point tests
# ------------------------------------------------------------------------------------------------
@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    change_index: int | None
    significant: bool

    def to_dict(self):
        return self.__dict__.copy()


def pettitt(x: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Pettitt (1979) non-parametric change-point test."""
    x = np.asarray(x, float)
    n = len(x)
    sgn = np.sign(x[None, :] - x[:, None])            # sgn[i, j] = sign(x_j - x_i)
    U = np.array([sgn[: t + 1, t + 1:].sum() for t in range(n - 1)])
    K = np.abs(U).max()
    k = int(np.abs(U).argmax())
    p = min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    return TestResult("pettitt", float(K), float(p), k + 1, p < alpha)


def _snht_stat(z: np.ndarray) -> tuple[float, int]:
    n = z.shape[-1]
    k = np.arange(1, n)
    c = np.cumsum(z, axis=-1)[..., :-1]
    z1 = c / k
    z2 = (z.sum(-1, keepdims=True) - c) / (n - k)
    T = k * z1 ** 2 + (n - k) * z2 ** 2
    return T.max(-1), T.argmax(-1) + 1


def _buishand_stat(z: np.ndarray) -> tuple[float, int]:
    S = np.cumsum(z, axis=-1)
    n = z.shape[-1]
    R = (np.maximum(S.max(-1), 0) - np.minimum(S.min(-1), 0)) / np.sqrt(n)   # S_0 = 0 included
    return R, np.abs(S).argmax(-1) + 1


@lru_cache(maxsize=64)
def _null(n: int, which: str, reps: int = 4000, seed: int = 0) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal((reps, n))
    z = (z - z.mean(1, keepdims=True)) / z.std(1, ddof=1, keepdims=True)
    return (_snht_stat(z) if which == "snht" else _buishand_stat(z))[0]


def snht(x: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Standard Normal Homogeneity Test (Alexandersson 1986); p-value by Monte Carlo."""
    x = np.asarray(x, float)
    z = (x - x.mean()) / x.std(ddof=1)
    T0, k = _snht_stat(z[None, :])
    p = float((_null(len(x), "snht") >= T0[0]).mean())
    return TestResult("snht", float(T0[0]), p, int(k[0]), p < alpha)


def buishand(x: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Buishand (1982) range test; p-value by Monte Carlo."""
    x = np.asarray(x, float)
    z = (x - x.mean()) / x.std(ddof=1)
    R, k = _buishand_stat(z[None, :])
    p = float((_null(len(x), "buishand") >= R[0]).mean())
    return TestResult("buishand", float(R[0]), p, int(k[0]), p < alpha)


# ------------------------------------------------------------------------------------------------
# Trend tests
# ------------------------------------------------------------------------------------------------
def sens_slope(x: np.ndarray, t: np.ndarray | None = None) -> tuple[float, float]:
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float) if t is None else np.asarray(t, float)
    i, j = np.triu_indices(len(x), 1)
    slopes = (x[j] - x[i]) / (t[j] - t[i])
    b = float(np.median(slopes))
    return b, float(np.median(x - b * t))


def mann_kendall(x: np.ndarray, alpha: float = 0.05, prewhiten: bool = True) -> dict:
    """Mann-Kendall trend test with optional trend-free pre-whitening (Yue et al. 2002)."""
    x = np.asarray(x, float)
    n = len(x)
    if prewhiten and n > 10:
        b, a = sens_slope(x)
        t = np.arange(n)
        d = x - b * t
        r1 = np.corrcoef(d[:-1], d[1:])[0, 1]
        if np.isfinite(r1) and abs(r1) > 1.96 / np.sqrt(n):
            d = d[1:] - r1 * d[:-1]
            x = d + b * t[1:]
            n = len(x)
    s = np.sign(x[None, :] - x[:, None])
    S = float(np.triu(s, 1).sum())
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    z = (S - np.sign(S)) / np.sqrt(var) if S != 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    slope, _ = sens_slope(np.asarray(x))
    return {"S": S, "z": float(z), "p_value": p, "significant": p < alpha, "sens_slope": slope,
            "tau": S / (0.5 * n * (n - 1))}


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """FDR control across many stations (field significance)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


# ------------------------------------------------------------------------------------------------
# Station-level homogeneity workflow
# ------------------------------------------------------------------------------------------------
def annual_series(obs: pd.DataFrame, var: str, min_completeness: float = 0.8) -> pd.DataFrame:
    """station x year table: annual total (precip) or mean (temperature); incomplete years -> NaN."""
    col = f"{var}_qc" if f"{var}_qc" in obs else var
    g = obs.assign(year=obs.date.dt.year).groupby(["station_id", "year"])[col]
    agg = g.sum(min_count=1) if var == "precip" else g.mean()
    comp = g.count() / g.size()
    agg[comp < min_completeness] = np.nan
    if var == "precip":
        agg = agg / comp  # scale partial years conservatively (only years >= min_completeness kept)
    return agg.unstack("station_id")


def reference_series(table: pd.DataFrame, stations: pd.DataFrame, sid: str, var: str,
                     k: int = 5, max_km: float = 250.0, min_overlap: int = 10,
                     exclude: set[str] | frozenset = frozenset()) -> tuple[pd.Series, list[str]]:
    st = stations.set_index("station_id")
    d = haversine_km(st.loc[sid, "lat"], st.loc[sid, "lon"], st.lat.values, st.lon.values)
    cands = [s for s, dd in sorted(zip(st.index, d), key=lambda t: t[1])
             if s != sid and dd <= max_km and s not in exclude and s in table.columns]
    cand = table[sid]
    ratio = var == "precip"
    anoms, weights, used = [], [], []
    for s in cands:
        ok = cand.notna() & table[s].notna()
        if ok.sum() < min_overlap:
            continue
        r = np.corrcoef(np.diff(cand[ok]), np.diff(table[s][ok]))[0, 1]  # first-difference correlation
        if not np.isfinite(r) or r <= 0.2:
            continue
        ref = table[s] / table[s].mean() if ratio else table[s] - table[s].mean()
        anoms.append(ref)
        weights.append(r ** 2)
        used.append(s)
        if len(used) >= k:
            break
    if not used:
        return pd.Series(np.nan, index=table.index), []
    A = pd.concat(anoms, axis=1)
    W = np.array(weights)
    ref = (A * W).sum(axis=1) / (A.notna() * W).sum(axis=1)
    return ref, used


def _vote(x: np.ndarray, alpha: float) -> dict:
    res = [pettitt(x, alpha), snht(x, alpha), buishand(x, alpha)]
    n_sig = sum(r.significant for r in res)
    idx = [r.change_index for r in res if r.significant]
    return {"tests": [r.to_dict() for r in res], "n_significant": n_sig, "break": n_sig >= 2,
            "change_index": int(np.median(idx)) if idx else None}


def classify_station(obs: pd.DataFrame, stations: pd.DataFrame, sid: str, var: str,
                     table: pd.DataFrame | None = None, alpha: float = 0.05, min_years: int = 15,
                     exclude_refs: set[str] | frozenset = frozenset()) -> dict:
    table = annual_series(obs, var) if table is None else table
    cand = table[sid].dropna()
    out = {"station_id": sid, "variable": var, "n_years": int(len(cand))}
    if len(cand) < min_years:
        return {**out, "classification": "insufficient_data"}
    raw = _vote(cand.values, alpha)
    ref, used = reference_series(table, stations, sid, var, exclude=exclude_refs)
    ref = ref.reindex(cand.index)
    out.update(raw_break=raw["break"], raw_break_year=int(cand.index[raw["change_index"]])
               if raw["break"] else None, reference_stations=used, raw_tests=raw["tests"])
    if used and ref.notna().sum() >= min_years:
        diff = (cand / cand.mean()) / ref if var == "precip" else (cand - cand.mean()) - ref
        diff = diff.dropna()
        rel = _vote(diff.values, alpha)
        out.update(rel_break=rel["break"], rel_tests=rel["tests"],
                   rel_break_year=int(diff.index[rel["change_index"]]) if rel["break"] else None)
        if rel["break"]:
            y = out["rel_break_year"]
            before, after = diff[diff.index < y], diff[diff.index >= y]
            out["shift_estimate"] = float(after.mean() - before.mean()) if var != "precip" else \
                float(after.mean() / before.mean())
    else:
        out.update(rel_break=None, rel_break_year=None)

    documented = []
    ev = stations.set_index("station_id").get("relocation_events")
    if ev is not None and isinstance(ev.get(sid), str):
        documented = [int(e["date"][:4]) for e in json.loads(ev[sid] or "[]")]
    out["documented_events"] = documented
    by = out.get("rel_break_year") or out.get("raw_break_year")
    near_doc = bool(by and any(abs(by - d) <= 2 for d in documented))
    if out["rel_break"] is None:
        cls = "raw_break_no_reference" if raw["break"] else "homogeneous_no_reference"
    elif out["rel_break"] and not _practically_relevant(var, out.get("shift_estimate")):
        cls = "minor_break"
    elif out["rel_break"]:
        cls = "artifact_supported_by_metadata" if near_doc else "possible_artifact_undocumented"
    elif raw["break"]:
        cls = "regional_signal"
    else:
        cls = "homogeneous"
    out["classification"] = cls
    out["interpretation"] = {
        "artifact_supported_by_metadata": "Break not shared by neighbours and coincides with documented station change: likely observational artifact.",
        "possible_artifact_undocumented": "Break not shared by neighbours, no metadata: candidate artifact requiring expert review.",
        "regional_signal": "Shift present in candidate and neighbours: consistent with regional climate variability/change.",
        "homogeneous": "No break detected in raw or relative series.",
        "minor_break": "Statistically detected relative shift below practical relevance threshold; monitor only.",
        "raw_break_no_reference": "Raw break but no adequate reference stations; cannot separate climate from artifact.",
        "homogeneous_no_reference": "No raw break; no adequate reference for relative testing.",
    }[cls]
    return out


ARTIFACT_CLASSES = ("artifact_supported_by_metadata", "possible_artifact_undocumented")


MIN_SHIFT = {"tmax": 0.3, "tmin": 0.3, "precip": 0.08}   # degC; precip as |ratio - 1|


def _practically_relevant(var: str, shift: float | None) -> bool:
    if shift is None:
        return True
    mag = abs(shift - 1.0) if var == "precip" else abs(shift)
    return mag >= MIN_SHIFT.get(var, 0.0)


def homogeneity_report(obs: pd.DataFrame, stations: pd.DataFrame,
                       variables=("precip", "tmax", "tmin"), alpha: float = 0.05,
                       iterations: int = 3) -> pd.DataFrame:
    """Iterative relative homogeneity testing.

    Pass 1 uses all neighbours as references. Stations flagged as possible artifacts are then removed
    from the reference pool and every station is re-tested, so one inhomogeneous neighbour does not
    propagate spurious breaks into the difference series of others (standard practice, cf. MASH/ACMANT).
    """
    rows = []
    for var in variables:
        table = annual_series(obs, var)
        excluded: set[str] = set()
        for it in range(iterations):
            res = [classify_station(obs, stations, sid, var, table, alpha, exclude_refs=excluded)
                   for sid in table.columns]
            flagged = {r["station_id"] for r in res if r["classification"] in ARTIFACT_CLASSES}
            if flagged <= excluded:
                break
            # exclude only the strongest suspects first: those whose raw series also breaks
            strong = {r["station_id"] for r in res if r["classification"] in ARTIFACT_CLASSES
                      and r.get("raw_break") and r.get("raw_break_year") == r.get("rel_break_year")}
            excluded |= strong or flagged
        for r in res:
            r["iterations"] = it + 1
            r["excluded_from_references"] = r["station_id"] in excluded
            rows.append({k: v for k, v in r.items() if k not in ("raw_tests", "rel_tests")})
    return pd.DataFrame(rows)


def trend_report(obs: pd.DataFrame, variables=("precip", "tmax", "tmin"), alpha: float = 0.05,
                 min_years: int = 15) -> pd.DataFrame:
    rows = []
    for var in variables:
        table = annual_series(obs, var)
        for sid in table.columns:
            s = table[sid].dropna()
            if len(s) < min_years:
                continue
            mk = mann_kendall(s.values, alpha)
            slope, _ = sens_slope(s.values, s.index.values)
            rows.append({"station_id": sid, "variable": var, "n_years": len(s), "sens_slope_per_year": slope,
                         "mk_z": mk["z"], "p_value": mk["p_value"], "tau": mk["tau"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["significant_fdr"] = False
        for var, g in df.groupby("variable"):
            df.loc[g.index, "significant_fdr"] = benjamini_hochberg(g.p_value.values, alpha)
    return df
