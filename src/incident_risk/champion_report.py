import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ROOT, load_config
from .data import load_raw
from .metrics import bootstrap_ci, evaluate
from .splits import chronological_split

SEGMENTS = ["service_name", "environment", "customer_tier"]


def segment_table(hold: pd.DataFrame, t: str) -> pd.DataFrame:
    rows = []
    for col in SEGMENTS:
        for level, g in hold.groupby(col):
            n, pos, hit = len(g), int(g[t].sum()), g[g["alert"]]
            rows.append(
                {
                    "segment": col,
                    "level": level,
                    "n": n,
                    "incidents": pos,
                    "pr_auc": average_precision_score(g[t], g["risk"]) if 0 < pos < n else np.nan,
                    "alerts_per_1000": g["alert"].mean() * 1000,
                    "recall": hit[t].sum() / pos if pos else np.nan,
                    "precision": hit[t].mean() if len(hit) else np.nan,
                    "low_sample_warning": n < 200,
                }
            )
    return pd.DataFrame(rows)


def sensitivity(model, hold, feats, col, n=15) -> pd.DataFrame:
    """Set every row's `col` to a value, average the risk. Shows how the model reacts to it."""
    grid = np.quantile(hold[col], np.linspace(0.05, 0.95, n))
    rows = []
    for v in grid:
        X = hold[feats].copy()
        X[col] = v
        rows.append({"value": v, "mean_risk": model.predict_proba(X)[:, 1].mean()})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    t, rep = cfg["data"]["target"], ROOT / cfg["paths"]["reports"]
    b = joblib.load(ROOT / cfg["paths"]["model"])
    model, thr, feats = b["model"], b["threshold"], b["meta"]["features"]
    _, val, _ = chronological_split(load_raw(cfg), cfg)
    hold = val.iloc[len(val) // 2 :].copy()  # the half used for threshold choice, NOT the test set
    hold["risk"] = model.predict_proba(hold[feats])[:, 1]
    hold["alert"] = hold["risk"] >= thr

    summary = evaluate(hold[t], hold["risk"], thr)
    summary["bootstrap_ci"] = bootstrap_ci(hold[t], hold["risk"], cfg["alert"]["budget_per_1000"])
    (rep / "validation_champion_summary.json").write_text(json.dumps(summary, indent=2))

    cols = [cfg["data"]["id_col"], cfg["data"]["time_col"], "risk", t] + feats
    hold[(hold["alert"]) & (hold[t] == 0)].nlargest(25, "risk")[cols].to_csv(
        rep / "false_positive_samples.csv", index=False
    )
    hold[(~hold["alert"]) & (hold[t] == 1)].nlargest(25, "risk")[cols].to_csv(
        rep / "false_negative_samples.csv", index=False
    )

    segment_table(hold, t).to_csv(rep / "segment_validation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["error_rate_pct", "p95_latency_ms"]):
        s = sensitivity(model, hold, feats, col)
        ax.plot(s["value"], s["mean_risk"], marker="o")
        ax.set_xlabel(col)
        ax.set_ylabel("mean predicted risk")
        s.to_csv(rep / f"sensitivity_{col}.csv", index=False)
    plt.tight_layout()
    plt.savefig(rep / "sensitivity.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
