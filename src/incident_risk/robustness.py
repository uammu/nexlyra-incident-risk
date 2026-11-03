import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from .champion import features, load_champion, validation_holdout
from .config import ROOT, load_config
from .metrics import quick_metrics

SCENARIOS = {  # stress edits (other correlated metrics are NOT adjusted; this is a crash test)
    "latency_spike_x3": lambda X: X.assign(p95_latency_ms=X["p95_latency_ms"] * 3),
    "cpu_saturation_98": lambda X: X.assign(cpu_utilization_pct=98.0),
    "queue_backlog_x5": lambda X: X.assign(queue_depth=X["queue_depth"] * 5),
    "dependency_burst_+5": lambda X: X.assign(
        dependency_failures_15m=X["dependency_failures_15m"] + 5
    ),
    "error_rate_+5pts": lambda X: X.assign(
        error_rate_pct=(X["error_rate_pct"] + 5).clip(upper=100)
    ),
}
EXPECT_UP = [
    "error_rate_pct",
    "p95_latency_ms",
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "db_connections_pct",
    "queue_depth",
    "dependency_failures_15m",
    "active_alerts",
    "incident_history_30d",
    "change_failure_rate_30d",
]


def score(model, X):
    return model.predict_proba(X)[:, 1]


def upper(c, f):
    return 100 if c in f["percent_cols"] else 1 if c in f.get("fraction_cols", []) else None


def boundary_test(model, hold, cfg):  # task 17
    f, base, rows = cfg["features"], hold[features(cfg)], []
    for c in f["numeric"]:
        for edge, v in [("min", 0.0), ("max", upper(c, f) or 1e6)]:
            X = base.copy()
            X[c] = v
            p = score(model, X)
            rows.append(
                {
                    "feature": c,
                    "edge": edge,
                    "value": v,
                    "finite": bool(np.isfinite(p).all()),
                    "in_0_1": bool(((p >= 0) & (p <= 1)).all()),
                    "mean_risk": float(p.mean()),
                }
            )
    return pd.DataFrame(rows)


def unseen_test(model, hold, cfg):  # task 18
    base, rows = hold[features(cfg)], []
    for c in cfg["features"]["categorical"]:
        X = base.copy()
        X[c] = "__UNSEEN__"
        p = score(model, X)
        d = np.abs(p - hold["risk"].values)
        rows.append(
            {
                "feature": c,
                "finite": bool(np.isfinite(p).all()),
                "mean_abs_change": float(d.mean()),
                "max_abs_change": float(d.max()),
            }
        )
    return pd.DataFrame(rows)


def scenario_test(model, hold, cfg, thr):  # tasks 21 to 25
    base, p0, rows = hold[features(cfg)], hold["risk"].values, []
    for name, fn in SCENARIOS.items():
        p = score(model, fn(base.copy()))
        rows.append(
            {
                "scenario": name,
                "mean_risk_before": p0.mean(),
                "mean_risk_after": p.mean(),
                "alerts_per_1000_before": (p0 >= thr).mean() * 1000,
                "alerts_per_1000_after": (p >= thr).mean() * 1000,
                "newly_alerting_share": ((p >= thr) & (p0 < thr)).sum() / max((p0 < thr).sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def monotonicity(model, hold, cfg, seed):  # tasks 26, 27
    base = hold[features(cfg)]
    sample, rows = base.sample(min(200, len(base)), random_state=seed), []
    for c in EXPECT_UP:
        grid = np.unique(np.quantile(base[c].dropna(), np.linspace(0.05, 0.95, 15)))
        if len(grid) < 3:
            continue
        curves = np.empty((len(sample), len(grid)))
        for j, v in enumerate(grid):
            X = sample.copy()
            X[c] = v
            curves[:, j] = score(model, X)
        steps, mean_curve = np.diff(curves, axis=1), curves.mean(0)
        rows.append(
            {
                "feature": c,
                "mean_curve_spearman": float(spearmanr(grid, mean_curve)[0]),
                "share_rows_any_decrease": float((steps < -1e-9).any(axis=1).mean()),
                "share_steps_decreasing": float((steps < -1e-9).mean()),
                "net_change": float(mean_curve[-1] - mean_curve[0]),
            }
        )
    df = pd.DataFrame(rows)
    df["non_monotonic"] = (df.mean_curve_spearman < 0.8) | (df.share_steps_decreasing > 0.35)
    return df


def noise_test(model, hold, cfg, thr, rng):  # task 28
    f, feats, p0, rows = cfg["features"], features(cfg), hold["risk"].values, []
    for s in [0.01, 0.05, 0.10, 0.20]:
        X = hold[feats].copy()
        for c in f["numeric"]:
            X[c] = (X[c] * (1 + rng.normal(0, s, len(X)))).clip(lower=0, upper=upper(c, f))
        p = score(model, X)
        rows.append(
            {
                "noise_sd": s,
                "pr_auc": average_precision_score(hold[cfg["data"]["target"]], p),
                "rank_corr_with_clean": spearmanr(p0, p)[0],
                "alert_flip_rate": float(((p0 >= thr) != (p >= thr)).mean()),
                "alerts_per_1000": float((p >= thr).mean() * 1000),
            }
        )
    return pd.DataFrame(rows)


def missingness_test(model, hold, cfg, thr, rng):  # task 29
    f, feats, rows = cfg["features"], features(cfg), []
    for frac in [0.0, 0.05, 0.10, 0.20, 0.40]:
        X = hold[feats].copy()
        X[f["numeric"]] = X[f["numeric"]].mask(rng.random((len(X), len(f["numeric"]))) < frac)
        rows.append(
            {
                "missing_fraction": frac,
                **quick_metrics(hold[cfg["data"]["target"]], score(model, X), thr),
            }
        )
    return pd.DataFrame(rows)


def category_shift_test(model, hold, cfg, thr, rng):  # task 30
    t, rows = cfg["data"]["target"], []
    for col in ["service_name", "environment", "region", "customer_tier"]:
        for lvl in hold[col].unique():
            w = np.where(hold[col] == lvl, 5.0, 1.0)
            w /= w.sum()
            s = hold.iloc[rng.choice(len(hold), size=len(hold), replace=True, p=w)]
            rows.append(
                {
                    "column": col,
                    "overweighted_level": lvl,
                    "level_share_after": float((s[col] == lvl).mean()),
                    **quick_metrics(s[t], s["risk"], thr),
                }
            )
    return pd.DataFrame(rows)


def time_window_test(hold, cfg, thr):  # task 31
    t, tc, rows = cfg["data"]["target"], cfg["data"]["time_col"], []
    for i, idx in enumerate(np.array_split(np.arange(len(hold)), 5)):
        w = hold.iloc[idx]
        rows.append(
            {
                "window": i + 1,
                "start": w[tc].min(),
                "end": w[tc].max(),
                **quick_metrics(w[t], w["risk"], thr),
            }
        )
    return pd.DataFrame(rows)


def main():
    cfg, seed = load_config(), load_config()["seed"]
    out = ROOT / cfg["paths"]["reports"] / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    b = load_champion(cfg)
    model, thr = b["model"], b["threshold"]
    hold, rng = validation_holdout(cfg, b), np.random.default_rng(seed)

    tables = {
        "boundary": boundary_test(model, hold, cfg),
        "unseen_category": unseen_test(model, hold, cfg),
        "scenarios": scenario_test(model, hold, cfg, thr),
        "monotonicity": monotonicity(model, hold, cfg, seed),
        "noise": noise_test(model, hold, cfg, thr, rng),
        "missingness": missingness_test(model, hold, cfg, thr, rng),
        "category_shift": category_shift_test(model, hold, cfg, thr, rng),
        "time_windows": time_window_test(hold, cfg, thr),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{df.round(4).to_string(index=False)}")

    bt, un = tables["boundary"], tables["unseen_category"]
    summary = {
        "boundary_all_ok": bool(bt.finite.all() and bt.in_0_1.all()),
        "unseen_all_ok": bool(un.finite.all()),
        "non_monotonic_features": tables["monotonicity"].query("non_monotonic").feature.tolist(),
    }
    (out / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
