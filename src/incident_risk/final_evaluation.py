import json
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score

from .champion import features, load_champion, validation_holdout
from .config import ROOT, load_config
from .data import load_raw
from .metrics import bootstrap_ci, evaluate, quick_metrics
from .service_hash import file_sha256
from .splits import chronological_split

SEGMENTS = ["service_name", "environment", "region", "customer_tier"]


def main():
    cfg = load_config()
    t, tc, feats = cfg["data"]["target"], cfg["data"]["time_col"], features(cfg)
    rep = ROOT / cfg["paths"]["reports"] / "final"
    lock = rep / "FINAL_TEST_LOCK.json"
    if lock.exists():
        sys.exit(
            f"Final test already evaluated ({lock}). Not running again. Read the saved results."
        )
    rep.mkdir(parents=True, exist_ok=True)

    # tasks 3, 4: frozen artifact must match its manifest
    model_path = ROOT / cfg["paths"]["model"]
    man = json.loads((ROOT / "models/artifact_manifest.json").read_text())
    if man["model_sha256"] != file_sha256(model_path):
        sys.exit("model.joblib does not match artifact_manifest.json. Run register first.")
    b = load_champion(cfg)
    thr = b["threshold"]

    _, _, test = chronological_split(load_raw(cfg), cfg)
    test = test.copy().reset_index(drop=True)
    test["risk"] = b["model"].predict_proba(test[feats])[:, 1]
    test["alert"] = test["risk"] >= thr
    y, p = test[t].values, test["risk"].values

    m = evaluate(y, p, thr)  # frozen threshold, not re-tuned
    tn, fp, fn, tp = confusion_matrix(y, test["alert"]).ravel()
    m.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), f1=float(f1_score(y, test["alert"])))
    m["bootstrap_ci"] = bootstrap_ci(y, p, cfg["alert"]["budget_per_1000"])

    # task 21: test vs validation (second half of validation = threshold half)
    v = validation_holdout(cfg, b)
    vm = quick_metrics(v[t], v["risk"], thr)
    tm = quick_metrics(y, p, thr)
    gap = pd.DataFrame({"validation": vm, "test": tm}).T
    gap["pr_auc_gap_test_minus_val"] = gap["pr_auc"] - gap.loc["validation", "pr_auc"]
    gap.to_csv(rep / "test_vs_validation.csv")

    # tasks 17 to 19: segments (small ones are flagged, not interpreted)
    rows = []
    for col in SEGMENTS:
        for lvl, g in test.groupby(col):
            q = quick_metrics(g[t], g["risk"], thr)
            rows.append(
                {
                    "segment": col,
                    "level": lvl,
                    "incidents": int(g[t].sum()),
                    "low_sample": len(g) < 200 or g[t].sum() < 20,
                    **q,
                }
            )
    pd.DataFrame(rows).to_csv(rep / "test_by_segment.csv", index=False)

    # task 22: generalization check by time window inside test
    wins = [
        {
            "window": i + 1,
            "start": w[tc].min(),
            "end": w[tc].max(),
            **quick_metrics(w[t], w["risk"], thr),
        }
        for i, idx in enumerate(np.array_split(np.arange(len(test)), 5))
        for w in [test.iloc[idx]]
    ]
    pd.DataFrame(wins).to_csv(rep / "test_time_windows.csv", index=False)

    # task 14: calibration plot
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean, frac, marker="o", label="test")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed incident rate")
    plt.legend()
    plt.savefig(rep / "test_calibration.png", dpi=150, bbox_inches="tight")

    test[["event_id", tc, "risk", "alert", t]].to_csv(rep / "test_predictions.csv", index=False)
    (rep / "final_test_metrics.json").write_text(json.dumps(m, indent=2))
    lock.write_text(
        json.dumps(
            {
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "artifact_version": man["version"],
                "model_sha256": man["model_sha256"],
                "threshold": thr,
            },
            indent=2,
        )
    )
    print(json.dumps(m, indent=2))
    print(gap.round(4).to_string())


if __name__ == "__main__":
    main()
