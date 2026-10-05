"""Metrics, chosen for the task rather than by habit (Section 29).

Extreme events are rare (~2% of days), so:
* AUPRC (with the base rate reported alongside) is the primary ranking metric; AUROC is reported but
  is optimistic under heavy class imbalance.
* Brier score and Brier Skill Score vs climatology measure probabilistic accuracy; ECE and reliability
  curves measure calibration (essential when probabilities feed decisions).
* Precision/recall/F1 require a threshold; it is chosen on the VALIDATION set (max F1), never on test.
* Regression (precipitation amounts): MAE, RMSE, R2; CRPS for ensembles/probabilistic forecasts.
* Uncertainty in every metric: station-block bootstrap CIs (stations are the independent units).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def expected_calibration_error(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile" else np.linspace(0, 1, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins: int = 10) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    out = {"mean_predicted": [], "observed_frequency": [], "count": []}
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out["mean_predicted"].append(float(p[m].mean()))
            out["observed_frequency"].append(float(y[m].mean()))
            out["count"].append(int(m.sum()))
    return out


def classification_metrics(y, p, threshold: float | None = None, base_rate_ref: float | None = None) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    thr = 0.5 if threshold is None else threshold
    yhat = (p >= thr).astype(int)
    base = float(y.mean())
    ref = base if base_rate_ref is None else base_rate_ref
    bs = brier_score_loss(y, p)
    bs_ref = np.mean((ref - y) ** 2)
    has_both = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)), "n_events": int(y.sum()), "base_rate": base,
        "auprc": float(average_precision_score(y, p)) if has_both else np.nan,
        "auprc_lift": float(average_precision_score(y, p) / base) if has_both and base > 0 else np.nan,
        "auroc": float(roc_auc_score(y, p)) if has_both else np.nan,
        "brier": float(bs), "brier_skill_vs_climatology": float(1 - bs / bs_ref) if bs_ref > 0 else np.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "threshold": float(thr),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }


def regression_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"mae": float(mean_absolute_error(y, yhat)), "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "r2": float(r2_score(y, yhat)), "bias": float(np.mean(yhat - y))}


def crps_ensemble(y: np.ndarray, ens: np.ndarray) -> float:
    """Empirical CRPS for ensemble forecasts; ens shape (n, m)."""
    y = np.asarray(y)[:, None]
    ens = np.asarray(ens)
    t1 = np.abs(ens - y).mean(1)
    t2 = np.abs(ens[:, :, None] - ens[:, None, :]).mean((1, 2)) / 2
    return float((t1 - t2).mean())


def crps_bernoulli(y, p) -> float:
    """For binary outcomes CRPS equals the Brier score."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bootstrap_ci(y, p, groups, metric_fn, n: int = 200, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Station-block bootstrap: resample stations with replacement."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            vals.append(metric_fn(yy, p[idx]))
    if not vals:
        return (np.nan, np.nan)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y, p_a, p_b, groups, metric_fn=average_precision_score, n: int = 500, seed: int = 0) -> dict:
    """Is model A better than model B? Station-block bootstrap of metric(A) - metric(B)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        if 0 < y[idx].sum() < len(idx):
            d.append(metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx]))
    d = np.asarray(d)
    return {"diff": float(metric_fn(y, p_a) - metric_fn(y, p_b)),
            "ci_low": float(np.quantile(d, 0.025)) if len(d) else np.nan,
            "ci_high": float(np.quantile(d, 0.975)) if len(d) else np.nan,
            "p_a_not_better": float((d <= 0).mean()) if len(d) else np.nan}
