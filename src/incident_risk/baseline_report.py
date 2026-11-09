import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import ROOT, load_config
from .data import load_raw
from .metrics import threshold_for_budget
from .models import build_models, tuned_models
from .splits import chronological_split


def main():
    cfg = load_config()
    t = cfg["data"]["target"]
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    budget = cfg["alert"]["budget_per_1000"]
    train, val, _ = chronological_split(load_raw(cfg), cfg)
    ratio = float((train[t] == 0).sum() / max((train[t] == 1).sum(), 1))
    models = {**build_models(cfg, ratio), **tuned_models(cfg, ratio)}
    out = ROOT / cfg["paths"]["reports"] / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    rows = []
    for name, m in models.items():
        m.fit(train[feats], train[t])
        p = m.predict_proba(val[feats])[:, 1]
        a = p >= threshold_for_budget(p, budget)
        tn, fp, fn, tp = confusion_matrix(val[t], a).ravel()
        rows.append(
            {
                "model": name,
                "pr_auc": average_precision_score(val[t], p),
                "roc_auc": roc_auc_score(val[t], p),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "f1": f1_score(val[t], a),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        pr, rc, _ = precision_recall_curve(val[t], p)
        fpr, tpr, _ = roc_curve(val[t], p)
        ax_pr.plot(rc, pr, label=name)
        ax_roc.plot(fpr, tpr, label=name)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.legend(fontsize=6)
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.legend(fontsize=6)
    fig_pr.savefig(out / "pr_curves.png", dpi=150, bbox_inches="tight")
    fig_roc.savefig(out / "roc_curves.png", dpi=150, bbox_inches="tight")
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    df.to_csv(out / "baseline_table.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
