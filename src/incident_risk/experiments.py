import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from .boosters import make_booster
from .config import ROOT, load_config
from .data import load_raw
from .features import build_preprocessor
from .metrics import evaluate, threshold_for_budget
from .splits import chronological_split

GROUPS = {
    "traffic_latency": ["requests_per_minute", "p95_latency_ms", "error_rate_pct"],
    "resources": [
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "db_connections_pct",
        "queue_depth",
    ],
    "change_release": ["release_age_hours", "change_failure_rate_30d", "deployment_type"],
    "history_alerts": ["active_alerts", "incident_history_30d", "dependency_failures_15m"],
    "context": ["service_name", "environment", "region", "customer_tier"],
}


def fit_eval(cfg, name, params, train, val, cols: set, frac: float = 1.0) -> dict:
    f, t = cfg["features"], cfg["data"]["target"]
    num = [c for c in f["numeric"] if c in cols]
    cat = [c for c in f["categorical"] if c in cols]
    tr = train.iloc[: int(len(train) * frac)]
    ratio = (tr[t] == 0).sum() / max((tr[t] == 1).sum(), 1)
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(num, cat)),
            ("clf", make_booster(name, params, cfg["seed"], ratio)),
        ]
    )
    pipe.fit(tr[num + cat], tr[t])
    p_tr = pipe.predict_proba(tr[num + cat])[:, 1]
    p_va = pipe.predict_proba(val[num + cat])[:, 1]
    m = evaluate(val[t], p_va, threshold_for_budget(p_va, cfg["alert"]["budget_per_1000"]))
    return {
        "train_pr_auc": average_precision_score(tr[t], p_tr),
        "val_pr_auc": m["pr_auc"],
        "recall_at_budget": m["recall_at_budget"],
        "overfit_gap": average_precision_score(tr[t], p_tr) - m["pr_auc"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    name = ap.parse_args().model
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    best = json.loads((ROOT / "configs/best_params.json").read_text())[name]
    params = {**best["params"], "n_estimators": best["n_estimators"]}
    train, val, _ = chronological_split(load_raw(cfg), cfg)
    allc = set(cfg["features"]["numeric"] + cfg["features"]["categorical"])
    assert set(sum(GROUPS.values(), [])) == allc, "GROUPS must cover exactly the config features"

    rows = [{"run": "full", **fit_eval(cfg, name, params, train, val, allc)}]
    for g, cols in GROUPS.items():
        rows.append(
            {"run": f"drop_{g}", **fit_eval(cfg, name, params, train, val, allc - set(cols))}
        )
        rows.append({"run": f"only_{g}", **fit_eval(cfg, name, params, train, val, set(cols))})
    pd.DataFrame(rows).to_csv(rep / "feature_ablation.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    lc = [
        {"fraction": fr, **fit_eval(cfg, name, params, train, val, allc, fr)}
        for fr in [0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    lc = pd.DataFrame(lc)
    lc.to_csv(rep / "learning_curve.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(lc["fraction"], lc["train_pr_auc"], marker="o", label="train")
    plt.plot(lc["fraction"], lc["val_pr_auc"], marker="o", label="validation")
    plt.xlabel("Fraction of training data")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.savefig(rep / "learning_curve.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
