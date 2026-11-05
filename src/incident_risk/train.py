import json
import time

import joblib
import mlflow
import numpy as np

from .calibration import CalibratedModel
from .config import ROOT, load_config
from .data import load_raw, sha256_of, validate_schema
from .metrics import evaluate, threshold_for_budget
from .models import build_models, tuned_models
from .splits import chronological_split, save_boundaries


def main():
    cfg = load_config()
    seed, d = cfg["seed"], cfg["data"]
    np.random.seed(seed)

    df = load_raw(cfg)
    problems = validate_schema(df, cfg)
    if problems:
        raise SystemExit(f"Data validation failed: {problems}")

    train, val, test = chronological_split(df, cfg)
    bounds = save_boundaries(train, val, test, cfg)  # test used for boundaries only
    print(json.dumps(bounds, indent=2))

    feat_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X_tr, y_tr = train[feat_cols], train[d["target"]]
    X_va, y_va = val[feat_cols], val[d["target"]]

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    models = build_models(cfg, pos_weight)
    models.update(tuned_models(cfg, pos_weight))

    half = len(val) // 2
    X_cal, y_cal = X_va.iloc[:half], y_va.iloc[:half]
    X_thr = X_va.iloc[half:]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    results, fitted = {}, {}
    budget = cfg["alert"]["budget_per_1000"]
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_s = time.time() - t0
            p_val = model.predict_proba(X_va)[:, 1]
            m = evaluate(y_va, p_val, threshold_for_budget(p_val, budget))
            m["train_seconds"] = train_s
            t0 = time.time()
            model.predict_proba(X_va.iloc[:1000])
            m["infer_ms_per_1000"] = (time.time() - t0) * 1000
            mlflow.log_params({"model": name, "seed": seed})
            mlflow.log_metrics({k: v for k, v in m.items() if np.isfinite(v)})
            results[name], fitted[name] = m, model
            print(
                f"{name:22s} val PR-AUC={m['pr_auc']:.4f} recall@budget={m['recall_at_budget']:.3f}"
            )

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print("Selected:", best)

    final = CalibratedModel(fitted[best], cfg["calibration"]["method"]).fit_calibrator(X_cal, y_cal)
    thr = threshold_for_budget(final.predict_proba(X_thr)[:, 1], budget)

    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(exist_ok=True, parents=True)
    (rep / "validation_comparison.json").write_text(json.dumps(results, indent=2))

    meta = {
        "selected_model": best,
        "threshold": thr,
        "alert_budget_per_1000": budget,
        "features": feat_cols,
        "data_sha256": sha256_of(ROOT / d["path"]),
        "split": bounds,
        "seed": seed,
        "disclaimer": "Trained on SYNTHETIC data. Not validated for production use.",
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "threshold": thr, "meta": meta}, ROOT / cfg["paths"]["model"])
    (ROOT / cfg["paths"]["metadata"]).write_text(json.dumps(meta, indent=2))
    print("Threshold frozen:", round(thr, 4), "(test set not scored here)")


if __name__ == "__main__":
    main()
