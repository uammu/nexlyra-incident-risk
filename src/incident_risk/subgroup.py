import numpy as np
import pandas as pd

from .champion import load_champion, validation_holdout
from .config import ROOT, load_config
from .metrics import quick_metrics

SEG = ["service_name", "environment", "region", "customer_tier"]


def main():
    cfg = load_config()
    t, out = cfg["data"]["target"], ROOT / cfg["paths"]["reports"]
    b = load_champion(cfg)
    thr = b["threshold"]
    # Whole validation period for more samples: sigmoid calibration keeps ranking, so PR-AUC is unaffected.
    hold = validation_holdout(cfg, b, full=True)

    overall = quick_metrics(hold[t], hold["risk"], thr)
    overall_lift = overall["pr_auc"] / overall["incident_rate"]
    rows = []
    for col in SEG:
        for lvl, g in hold.groupby(col):
            m = quick_metrics(g[t], g["risk"], thr)
            m.update(segment=col, level=lvl, incidents=int(g[t].sum()))
            m["lift"] = m["pr_auc"] / m["incident_rate"] if m["incident_rate"] > 0 else np.nan
            m["low_sample"] = m["n"] < 200 or m["incidents"] < 20
            m["weak"] = (not m["low_sample"]) and bool(
                m["lift"] < 0.7 * overall_lift or m["recall"] < overall["recall"] - 0.15
            )
            rows.append(m)
    df = pd.DataFrame(rows)[
        [
            "segment",
            "level",
            "n",
            "incidents",
            "incident_rate",
            "pr_auc",
            "lift",
            "recall",
            "precision",
            "alerts_per_1000",
            "low_sample",
            "weak",
        ]
    ]
    df.to_csv(
        out / "segment_performance.csv", index=False
    )  # supersedes Day 3's segment_validation.csv
    df[df.weak].to_csv(out / "weak_segments.csv", index=False)
    print("overall:", {k: round(v, 4) for k, v in overall.items()})
    print(df.round(3).to_string(index=False))
    print(f"\nweak: {int(df.weak.sum())}   low-sample (ignore): {int(df.low_sample.sum())}")


if __name__ == "__main__":
    main()
