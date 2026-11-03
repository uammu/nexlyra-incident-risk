import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def threshold_for_budget(scores: np.ndarray, budget_per_1000: float) -> float:
    """Score of the k-th highest event, so about budget/1000 of events trigger alerts."""
    k = max(1, int(round(len(scores) * budget_per_1000 / 1000)))
    return float(np.sort(scores)[::-1][k - 1])


def best_cost_threshold(y, p, cost_fp: float, cost_fn: float) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
    costs = [
        cost_fp * ((p >= t) & (y == 0)).sum() + cost_fn * ((p < t) & (y == 1)).sum() for t in grid
    ]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    alert = p >= threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    pos = int(y.sum())
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < pos < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(np.mean(np.abs(frac_pos - mean_pred))),  # simple calibration error
        "threshold": float(threshold),
        "alerts_per_1000": float(alert.mean() * 1000),
        "recall_at_budget": tp / pos if pos else float("nan"),
        "precision_at_budget": tp / max(tp + fp, 1),
        "false_alerts_per_1000": fp / len(y) * 1000,
        "missed_incidents": pos - tp,
    }


def bootstrap_ci(
    y, p, budget_per_1000: float, n_boot: int = 1000, block: int = 48, seed: int = 42
) -> dict:
    """Block bootstrap: resamples whole days (48 x 30-min rows) so time-correlation is respected."""
    y, p = np.asarray(y), np.asarray(p)
    n, rng = len(y), np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = {"pr_auc": [], "recall_at_budget": [], "precision_at_budget": []}
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0:
            continue
        k = max(1, int(round(len(yb) * budget_per_1000 / 1000)))
        tp = yb[np.argsort(-pb)[:k]].sum()
        out["pr_auc"].append(average_precision_score(yb, pb))
        out["recall_at_budget"].append(tp / yb.sum())
        out["precision_at_budget"].append(tp / k)
    return {
        k: {
            "mean": float(np.mean(v)),
            "lo95": float(np.percentile(v, 2.5)),
            "hi95": float(np.percentile(v, 97.5)),
        }
        for k, v in out.items()
    }


def quick_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    a, pos = p >= thr, int(np.sum(y))
    tp = int((a & (y == 1)).sum())
    return {
        "n": len(y),
        "incident_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)) if pos else float("nan"),
        "recall": tp / pos if pos else float("nan"),
        "precision": tp / a.sum() if a.sum() else float("nan"),
        "alerts_per_1000": float(a.mean() * 1000),
    }
