"""Scientifically defensible baselines (Section 10).

Statistical: climatology (day-of-year), probabilistic persistence, logistic regression (a GLM),
             Tweedie GLM for precipitation amounts (zero-inflated, right-skewed -> power 1.5).
Classical ML: Random Forest, HistGradientBoosting, XGBoost, LightGBM (the last two if installed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ModelData


class ClimatologyBaseline:
    """P(event | day-of-year +/- 15 d) pooled over training stations (station-agnostic, so it is
    applicable to unseen stations)."""

    name = "climatology"

    def fit(self, md: ModelData, split="train"):
        sd = md.splits[split]
        doy = pd.DatetimeIndex(sd.dates).dayofyear.to_numpy()
        rate = np.zeros(367)
        for d in range(1, 367):
            dist = np.minimum(np.abs(doy - d), 366 - np.abs(doy - d))
            m = dist <= 15
            rate[d] = (sd.y[m].sum() + 0.5) / (m.sum() + 1.0)
        self.rate_ = rate
        return self

    def predict_proba(self, md: ModelData, split: str) -> np.ndarray:
        doy = pd.DatetimeIndex(md.splits[split].dates).dayofyear.to_numpy()
        return self.rate_[doy]


class PersistenceBaseline:
    """Probabilistic persistence: P(event_{t+1} | event_t) estimated on training data."""

    name = "persistence"

    def _today(self, md: ModelData, split: str) -> np.ndarray:
        """Event on the forecast day t equals target(t - lead) for lead = 1."""
        sd = md.splits[split]
        return md.y_full[sd.idx[:, 0], np.maximum(sd.idx[:, 1] - 1, 0)]

    def fit(self, md: ModelData, split="train"):
        today = self._today(md, split)
        y = md.splits[split].y
        ok = np.isfinite(today)
        self.p1_ = (y[ok & (today == 1)].sum() + 0.5) / ((ok & (today == 1)).sum() + 1)
        self.p0_ = (y[ok & (today == 0)].sum() + 0.5) / ((ok & (today == 0)).sum() + 1)
        self.base_ = y.mean()
        return self

    def predict_proba(self, md: ModelData, split: str) -> np.ndarray:
        today = self._today(md, split)
        return np.where(np.isnan(today), self.base_, np.where(today == 1, self.p1_, self.p0_))


def _optional(name):
    try:
        if name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, verbosity=0)
        if name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1)
    except ImportError:
        return None


def tabular_classifier(name: str, seed: int = 0):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                    n_jobs=-1, random_state=seed))
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.15, random_state=seed)
    if name in ("xgboost", "lightgbm"):
        return _optional(name)
    raise ValueError(name)


def tabular_regressor(name: str, seed: int = 0):
    if name == "linear":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    if name == "tweedie_glm":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             TweedieRegressor(power=1.5, alpha=0.01, link="log", max_iter=1000))
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, loss="poisson", random_state=seed)
    raise ValueError(name)
