import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .champion import features, load_champion
from .config import ROOT, load_config
from .data import load_raw
from .splits import chronological_split

warnings.filterwarnings("ignore")
WATCH, ACT = 0.10, 0.25


def psi_num(ref, cur, bins=10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    a = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_cat(ref, cur) -> float:
    lv = sorted(set(ref.unique()) | set(cur.unique()))
    e = np.clip(ref.value_counts(normalize=True).reindex(lv).fillna(0).values, 1e-6, None)
    a = np.clip(cur.value_counts(normalize=True).reindex(lv).fillna(0).values, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(x: float) -> str:
    return "stable" if x < WATCH else "watch" if x < ACT else "investigate"


def drift_table(ref, cur, cfg) -> pd.DataFrame:
    f, rows = cfg["features"], []
    for c in f["numeric"]:
        v = psi_num(ref[c], cur[c])
        rows.append(
            {
                "feature": c,
                "type": "numeric",
                "psi": v,
                "ks": ks_2samp(ref[c], cur[c]).statistic,
                "status": status(v),
            }
        )
    for c in f["categorical"]:
        v = psi_cat(ref[c], cur[c])
        rows.append(
            {"feature": c, "type": "categorical", "psi": v, "ks": np.nan, "status": status(v)}
        )
    return pd.DataFrame(rows)


SCENARIOS = {
    "latency_x1.5": lambda d: d.assign(p95_latency_ms=d["p95_latency_ms"] * 1.5),
    "traffic_x2_errors_x2": lambda d: d.assign(
        requests_per_minute=d["requests_per_minute"] * 2,
        error_rate_pct=(d["error_rate_pct"] * 2).clip(upper=100),
    ),
    "cpu_mem_+15pts": lambda d: d.assign(
        cpu_utilization_pct=(d["cpu_utilization_pct"] + 15).clip(upper=100),
        memory_utilization_pct=(d["memory_utilization_pct"] + 15).clip(upper=100),
    ),
}


def main():
    cfg = load_config()
    f, feats = cfg["features"], features(cfg)
    out = ROOT / cfg["paths"]["reports"] / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    b = load_champion(cfg)
    train, val, _ = chronological_split(load_raw(cfg), cfg)  # test is never used here
    score = lambda d: b["model"].predict_proba(d[feats])[:, 1]
    thr = b["threshold"]
    ref_s = score(train)

    # task 23: reference profile
    prof = {
        "built_from": "train",
        "rows": len(train),
        "threshold": thr,
        "numeric": {
            c: {
                "mean": float(train[c].mean()),
                "std": float(train[c].std()),
                "p01": float(train[c].quantile(0.01)),
                "p50": float(train[c].median()),
                "p99": float(train[c].quantile(0.99)),
                "missing": float(train[c].isna().mean()),
            }
            for c in f["numeric"]
        },
        "categorical": {
            c: train[c].value_counts(normalize=True).round(4).to_dict() for c in f["categorical"]
        },
        "score": {
            "p50": float(np.median(ref_s)),
            "p95": float(np.quantile(ref_s, 0.95)),
            "alerts_per_1000_at_threshold": float((ref_s >= thr).mean() * 1000),
        },
    }
    (out / "reference_profile.json").write_text(json.dumps(prof, indent=2))

    cases = {"normal_validation": val, **{k: fn(val.copy()) for k, fn in SCENARIOS.items()}}
    rows = []
    for name, cur in cases.items():
        dt = drift_table(train, cur, cfg)
        dt.assign(scenario=name).to_csv(out / f"drift_{name}.csv", index=False)
        cs = score(cur)
        ps = psi_num(pd.Series(ref_s), pd.Series(cs))  # tasks 25, 29: prediction drift
        rows.append(
            {
                "scenario": name,
                "features_watch": int((dt.status == "watch").sum()),
                "features_investigate": int((dt.status == "investigate").sum()),
                "max_feature_psi": dt.psi.max(),
                "worst_feature": dt.loc[dt.psi.idxmax(), "feature"],
                "score_psi": ps,
                "score_status": status(ps),
                "mean_risk": cs.mean(),
                "alerts_per_1000": (cs >= thr).mean() * 1000,
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "drift_scenarios_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    try:  # task 27: Evidently (API of 0.7.x)
        from evidently import Report
        from evidently.presets import DataDriftPreset

        cols = f["numeric"] + f["categorical"]
        snap = Report([DataDriftPreset()]).run(cases["latency_x1.5"][cols], train[cols])
        snap.save_html(str(out / "evidently_drift_report.html"))
        print("Evidently report written")
    except Exception as e:
        print("Evidently skipped:", type(e).__name__, str(e)[:150])


if __name__ == "__main__":
    main()
