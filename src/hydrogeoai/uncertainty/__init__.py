"""Uncertainty quantification (Section 16).

The final method is chosen experimentally: the runner compares deep ensembles, MC dropout and
(class-conditional) split conformal prediction on calibration (ECE), sharpness and coverage.

Output contract for every prediction:
    probability  : mean predicted probability of the event
    confidence   : 1 - normalised predictive entropy (0 = coin flip, 1 = certain)
    epistemic    : mutual information between prediction and model parameters (ensemble/MC spread)
    conformal_set: {0}, {1} or {0,1} at level 1 - alpha (ambiguous when {0,1})
    uncertainty  : "low" | "moderate" | "high" category from epistemic spread (rules in `categorise`)
"""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def entropy_bits(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def decompose(member_probs: np.ndarray) -> dict:
    """member_probs (M, N) from ensemble members or MC-dropout samples."""
    mp = np.asarray(member_probs)
    p = mp.mean(0)
    total = entropy_bits(p)
    aleatoric = entropy_bits(mp).mean(0)
    return {"probability": p, "std": mp.std(0), "total_entropy": total, "aleatoric": aleatoric,
            "epistemic": np.clip(total - aleatoric, 0, None), "confidence": 1 - total}


class ClassConditionalConformal:
    """Mondrian (class-conditional) split conformal classification.

    Class-conditional calibration guarantees coverage separately for events and non-events, which
    matters under strong imbalance (marginal coverage could ignore the rare class entirely).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray) -> "ClassConditionalConformal":
        y_cal, p_cal = np.asarray(y_cal).astype(int), np.asarray(p_cal)
        self.q_ = {}
        for c in (0, 1):
            s = 1 - (p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c])
            n = len(s)
            k = min(int(np.ceil((n + 1) * (1 - self.alpha))), n) if n else 0
            self.q_[c] = float(np.sort(s)[k - 1]) if n else 1.0
        return self

    def predict_sets(self, p: np.ndarray) -> np.ndarray:
        """Returns (N, 2) boolean membership for classes 0 and 1."""
        p = np.asarray(p)
        return np.stack([(1 - (1 - p)) <= self.q_[0], (1 - p) <= self.q_[1]], 1)

    def evaluate(self, y: np.ndarray, p: np.ndarray) -> dict:
        sets = self.predict_sets(p)
        y = np.asarray(y).astype(int)
        cover = sets[np.arange(len(y)), y]
        size = sets.sum(1)
        return {"alpha": self.alpha, "coverage": float(cover.mean()),
                "coverage_events": float(cover[y == 1].mean()) if (y == 1).any() else np.nan,
                "coverage_non_events": float(cover[y == 0].mean()) if (y == 0).any() else np.nan,
                "mean_set_size": float(size.mean()), "frac_ambiguous": float((size == 2).mean()),
                "frac_empty": float((size == 0).mean()), "q": self.q_}


def temperature_scale(y_val, p_val) -> float:
    """Fit a single temperature on validation logits (post-hoc calibration)."""
    from scipy.optimize import minimize_scalar

    z = np.log(np.clip(p_val, EPS, 1 - EPS) / (1 - np.clip(p_val, EPS, 1 - EPS)))

    def nll(T):
        q = 1 / (1 + np.exp(-z / T))
        return -np.mean(y_val * np.log(q + EPS) + (1 - y_val) * np.log(1 - q + EPS))

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p, T: float):
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    return 1 / (1 + np.exp(-z / T))


def categorise(epistemic: np.ndarray, conformal_sets: np.ndarray | None, ref_epistemic: np.ndarray) -> np.ndarray:
    """low / moderate / high uncertainty from epistemic spread (validation quantiles).

    high     : epistemic above the validation 90th percentile, or an empty conformal set
    moderate : epistemic above the validation median
    low      : otherwise
    Conformal sets are reported separately rather than folded into the category: for rare events,
    class-conditional sets are ambiguous ({0,1}) on most days by construction (to cover 1-alpha of the
    events), which would make the category uninformative.
    """
    q50, q90 = np.quantile(ref_epistemic, [0.5, 0.9])
    cat = np.where(epistemic > q50, "moderate", "low").astype(object)
    high = epistemic > q90
    if conformal_sets is not None:
        high |= conformal_sets.sum(1) == 0
    cat[high] = "high"
    return cat
