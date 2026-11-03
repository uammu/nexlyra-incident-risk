import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .champion import features, load_champion, validation_holdout
from .config import ROOT, load_config
from .data import load_raw
from .metrics import bootstrap_ci
from .models import build_models, tuned_models
from .splits import chronological_split


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    cfg = load_config()
    t, feats, out = cfg["data"]["target"], features(cfg), ROOT / cfg["paths"]["reports"]
    b = load_champion(cfg)
    hold = validation_holdout(cfg, b)
    res = {
        "bootstrap_ci_validation": bootstrap_ci(
            hold[t], hold["risk"], cfg["alert"]["budget_per_1000"]
        )
    }

    # 1) How much does the score move if only the random seed changes?
    train, val, _ = chronological_split(load_raw(cfg), cfg)
    name, ratio = b["meta"]["selected_model"], (train[t] == 0).sum() / max((train[t] == 1).sum(), 1)
    prs = []
    for s in range(5):
        c = {**cfg, "seed": cfg["seed"] + s}
        m = {**build_models(c, ratio), **tuned_models(c, ratio)}[name]
        m.fit(train[feats], train[t])
        prs.append(average_precision_score(val[t], m.predict_proba(val[feats])[:, 1]))
    res["seed_variance_val_pr_auc"] = {
        "values": [round(v, 4) for v in prs],
        "mean": float(np.mean(prs)),
        "sd": float(np.std(prs)),
    }

    # 2) "The model says 30%": what range of real incident rates does that cover?
    hold["bin"] = pd.qcut(hold["risk"], 10, duplicates="drop")
    rows = []
    for _, g in hold.groupby("bin", observed=True):
        k, n = int(g[t].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append(
            {
                "mean_predicted": g["risk"].mean(),
                "observed_rate": k / n,
                "n": n,
                "observed_lo95": lo,
                "observed_hi95": hi,
            }
        )
    pd.DataFrame(rows).to_csv(out / "reliability_bins.csv", index=False)
    (out / "uncertainty.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
