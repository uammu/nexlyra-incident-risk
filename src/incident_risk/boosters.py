import warnings

import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")


def _spw(mode: str, ratio: float) -> float:
    """Class weighting: none / sqrt (gentle) / full (negatives per positive)."""
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "full": float(ratio)}[mode]


def make_booster(
    name: str, params: dict, seed: int, pos_ratio: float, early_stopping_rounds: int | None = None
):
    p = dict(params)
    spw = _spw(p.pop("class_weight_mode", "none"), pos_ratio)
    n_est = p.pop("n_estimators", 2000)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=n_est,
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=early_stopping_rounds,
            **p,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_est, scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1, **p
        )
    raise ValueError(name)


def fit_with_early_stopping(name, model, X_fit, y_fit, X_es, y_es, rounds: int) -> int:
    """Stops on a mock-test set (X_es). Returns the best number of trees."""
    if name == "xgboost":
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        return int(model.best_iteration) + 1
    import lightgbm as lgb

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_es, y_es)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(rounds, verbose=False)],
    )
    return int(model.best_iteration_)
