import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

from .calibration import CalibratedModel
from .config import ROOT, load_config
from .data import load_raw
from .models import build_models, tuned_models
from .splits import chronological_split


def ece(y, p, bins=10) -> float:
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(frac - mean)))


def main():
    cfg = load_config()
    t, rep = cfg["data"]["target"], ROOT / cfg["paths"]["reports"]
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    train, val, _ = chronological_split(load_raw(cfg), cfg)

    # Champion = best validation PR-AUC from train.py's comparison
    comp = json.loads((rep / "validation_comparison.json").read_text())
    name = max(comp, key=lambda k: comp[k]["pr_auc"])
    ratio = (train[t] == 0).sum() / max((train[t] == 1).sum(), 1)
    base = {**build_models(cfg, ratio), **tuned_models(cfg, ratio)}[name]
    base.fit(train[feats], train[t])

    half = len(val) // 2  # first half fits calibrator, second half judges it
    cal, ev = val.iloc[:half], val.iloc[half:]
    results, curves = {}, {}
    for method in ["none", "sigmoid", "isotonic"]:
        if method == "none":
            p = base.predict_proba(ev[feats])[:, 1]
        else:
            p = (
                CalibratedModel(base, method)
                .fit_calibrator(cal[feats], cal[t])
                .predict_proba(ev[feats])[:, 1]
            )
        results[method] = {
            "brier": float(brier_score_loss(ev[t], p)),
            "ece": ece(ev[t], p),
            "pr_auc": float(average_precision_score(ev[t], p)),
        }
        curves[method] = calibration_curve(ev[t], p, n_bins=10, strategy="quantile")

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    for m, (frac, mean) in curves.items():
        plt.plot(mean, frac, marker="o", label=m)
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed incident rate")
    plt.legend()
    plt.title(f"Calibration: {name}")
    plt.savefig(rep / "calibration_curves.png", dpi=150, bbox_inches="tight")

    best = min(["sigmoid", "isotonic"], key=lambda m: results[m]["brier"])
    results["recommended"] = best
    (rep / "calibration_comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSet calibration.method: {best} in configs/config.yaml, then run `make train`.")


if __name__ == "__main__":
    main()
