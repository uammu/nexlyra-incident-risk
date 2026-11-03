import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .boosters import make_booster
from .config import ROOT
from .features import build_preprocessor


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """What an on-call engineer would do: score = how close metrics are to their train p95."""

    def __init__(self, cols=("error_rate_pct", "p95_latency_ms", "queue_depth", "active_alerts")):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y):
        self.classes_ = np.array([0, 1])
        self.p95_ = {c: max(float(X[c].quantile(0.95)), 1e-9) for c in self.cols}
        return self

    def predict_proba(self, X: pd.DataFrame):
        s = np.mean([np.clip(X[c].fillna(0) / self.p95_[c], 0, 1) for c in self.cols], axis=0)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(cfg: dict, pos_weight: float) -> dict:
    f, seed = cfg["features"], cfg["seed"]

    def wrap(clf):
        return Pipeline(
            [("prep", build_preprocessor(f["numeric"], f["categorical"])), ("clf", clf)]
        )

    return {
        "majority_class": wrap(DummyClassifier(strategy="prior")),
        "rule_baseline": RuleBaseline(),
        "logistic_regression": wrap(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "random_forest": wrap(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "lightgbm": wrap(
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=pos_weight,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
            )
        ),
    }


def tuned_models(cfg: dict, pos_ratio: float) -> dict:
    path = ROOT / "configs/best_params.json"
    if not path.exists():
        return {}
    f, out = cfg["features"], {}
    for name, b in json.loads(path.read_text()).items():
        params = {**b["params"], "n_estimators": b["n_estimators"]}
        out[f"{name}_tuned"] = Pipeline(
            [
                ("prep", build_preprocessor(f["numeric"], f["categorical"])),
                ("clf", make_booster(name, params, cfg["seed"], pos_ratio)),
            ]
        )
    return out
