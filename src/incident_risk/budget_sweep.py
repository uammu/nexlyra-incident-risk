import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .champion import load_champion, validation_holdout
from .config import ROOT, load_config


def main():
    cfg = load_config()
    t, a = cfg["data"]["target"], cfg["alert"]
    b = load_champion(cfg)
    hold = validation_holdout(cfg, b)  # second half of VALIDATION; test stays closed
    y, p, n = hold[t].values, hold["risk"].values, len(hold)
    order = np.argsort(-p)
    rows = []
    for budget in [10, 25, 50, 75, 100, 125, 150, 200]:
        k = max(1, round(n * budget / 1000))
        hit = int(y[order[:k]].sum())
        fp, fn = k - hit, int(y.sum()) - hit
        rows.append(
            {
                "budget_per_1000": budget,
                "threshold": float(p[order[k - 1]]),
                "recall": hit / y.sum(),
                "precision": hit / k,
                "false_alerts_per_1000": fp / n * 1000,
                "missed_per_1000": fn / n * 1000,
                "cost_per_1000": (fp * a["cost_false_alert"] + fn * a["cost_missed_incident"])
                / n
                * 1000,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / cfg["paths"]["reports"] / "budget_sweep.csv", index=False)
    print(df.round(3).to_string(index=False))
    best = df.loc[df.cost_per_1000.idxmin()]
    print(
        f"\nLowest expected cost on validation: budget {int(best.budget_per_1000)} per 1,000 "
        f"(recall {best.recall:.2f}, precision {best.precision:.2f})."
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.budget_per_1000, df.recall, marker="o", label="recall")
    ax.plot(df.budget_per_1000, df.precision, marker="o", label="precision")
    ax.axvline(a["budget_per_1000"], color="k", ls=":", label="current budget")
    ax.set_xlabel("alerts per 1,000 events")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / cfg["paths"]["reports"] / "budget_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
