import argparse
import json
import time

import mlflow
import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from .boosters import fit_with_early_stopping, make_booster
from .config import ROOT, load_config
from .cv import assert_chronological, gap_rows, time_folds
from .data import load_raw
from .features import build_preprocessor
from .splits import chronological_split


def suggest(trial: optuna.Trial, name: str) -> dict:
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "class_weight_mode": trial.suggest_categorical(
            "class_weight_mode", ["none", "sqrt", "full"]
        ),
    }
    if name == "xgboost":
        p["min_child_weight"] = trial.suggest_float("min_child_weight", 1, 30, log=True)
    else:
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
        p["num_leaves"] = trial.suggest_int("num_leaves", 4, 64)
    if p["subsample"] < 1.0 and name == "lightgbm":
        p["subsample_freq"] = 1  # LightGBM ignores subsample unless this is set
    return p


def make_objective(name, train, cfg, folds, feats):
    f, t, seed = cfg["features"], cfg["data"]["target"], cfg["seed"]
    rounds, es_frac = cfg["cv"]["early_stopping_rounds"], cfg["cv"]["es_fraction"]
    g = gap_rows(train, cfg)

    def objective(trial):
        params, scores, gaps, iters = suggest(trial, name), [], [], []
        t0 = time.time()
        for tr_idx, va_idx in folds:
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            cut = int(len(tr) * (1 - es_frac))
            fit_part, es_part = tr.iloc[:cut], tr.iloc[cut + g :]  # gap before the ES tail
            prep = build_preprocessor(f["numeric"], f["categorical"]).fit(fit_part[feats])
            ratio = (fit_part[t] == 0).sum() / max((fit_part[t] == 1).sum(), 1)
            model = make_booster(name, params, seed, ratio, rounds)
            it = fit_with_early_stopping(
                name,
                model,
                prep.transform(fit_part[feats]),
                fit_part[t],
                prep.transform(es_part[feats]),
                es_part[t],
                rounds,
            )
            p_va = model.predict_proba(prep.transform(va[feats]))[:, 1]
            p_fit = model.predict_proba(prep.transform(fit_part[feats]))[:, 1]
            s = average_precision_score(va[t], p_va)
            scores.append(s)
            gaps.append(average_precision_score(fit_part[t], p_fit) - s)  # overfit gap
            iters.append(it)
        secs = time.time() - t0
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        with mlflow.start_run(run_name=f"{name}_trial_{trial.number}", nested=True):
            mlflow.log_params({**trial.params, "model": name})
            mlflow.log_metrics(
                {
                    "cv_pr_auc": float(np.mean(scores)),
                    "cv_pr_auc_std": float(np.std(scores)),
                    "overfit_gap": float(np.mean(gaps)),
                    "train_seconds": secs,
                    "best_iteration": float(np.mean(iters)),
                }
            )
        return float(np.mean(scores))

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    f, seed = cfg["features"], cfg["seed"]
    trials = args.trials or cfg["cv"]["optuna_trials"]

    # Tuning sees TRAIN only. Validation and test stay untouched.
    train, _, _ = chronological_split(load_raw(cfg), cfg)
    train = train.reset_index(drop=True)
    folds = time_folds(train, cfg)
    assert_chronological(train, folds, cfg)
    feats = f["numeric"] + f["categorical"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"] + "-tuning")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best = {}
    for name in ["xgboost", "lightgbm"]:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        with mlflow.start_run(run_name=f"optuna_{name}"):
            study.optimize(make_objective(name, train, cfg, folds, feats), n_trials=trials)
            mlflow.log_metric("best_cv_pr_auc", study.best_value)
        bt = study.best_trial
        best[name] = {
            "params": bt.params,
            "n_estimators": bt.user_attrs["n_estimators"],
            "cv_pr_auc": study.best_value,
        }
        if bt.params.get("subsample", 1) < 1 and name == "lightgbm":
            best[name]["params"]["subsample_freq"] = 1
        print(name, "best CV PR-AUC:", round(study.best_value, 4))

    (ROOT / "configs/best_params.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
