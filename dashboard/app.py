import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCORES, META = ROOT / "reports/dashboard_scores.csv", ROOT / "reports/dashboard_meta.json"
MODEL, MANIFEST = ROOT / "models/model.joblib", ROOT / "models/artifact_manifest.json"
SHAP_IMG, TEST_METRICS = (
    ROOT / "reports/explain/shap_summary.png",
    ROOT / "reports/test_metrics.json",
)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Incident Risk Dashboard", layout="wide")

if not SCORES.exists() or not META.exists():
    st.error("No scores yet. Run `make dashboard-data` first.")
    st.stop()


@st.cache_data
def load_scores(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["event_timestamp"])


df = load_scores(str(SCORES), SCORES.stat().st_mtime)
meta = json.loads(META.read_text())
man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
thr = float(meta["threshold"])

# ---------- header: model version + timestamps (task 46) ----------
st.title("Nexlyra Incident Risk")
st.warning("SYNTHETIC data. Decision support only: a human decides every action.")
h = st.columns(4)
h[0].metric("Model", meta["model"])
h[1].metric("Artifact version", man.get("version", "unversioned"))
h[2].metric("Frozen threshold", f"{thr:.4f}")
h[3].metric(
    "Model file updated",
    datetime.fromtimestamp(MODEL.stat().st_mtime).strftime("%d %b %Y %H:%M")
    if MODEL.exists()
    else "n/a",
)
st.caption(
    f"Scores generated {meta['generated_at']} for the validation period "
    f"{meta['period_start'][:10]} to {meta['period_end'][:10]} ({meta['n']:,} events)."
)

# ---------- filters ----------
st.sidebar.header("Filters")
f = df
for col, label in [
    ("service_name", "Service"),
    ("environment", "Environment"),
    ("region", "Region"),
    ("customer_tier", "Customer tier"),
]:
    opts = sorted(df[col].dropna().unique())
    f = f[f[col].isin(st.sidebar.multiselect(label, opts, default=opts))]
dmin, dmax = df.event_timestamp.min().date(), df.event_timestamp.max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(rng) == 2:
    f = f[(f.event_timestamp.dt.date >= rng[0]) & (f.event_timestamp.dt.date <= rng[1])]
if f.empty:
    st.warning("No events match these filters.")
    st.stop()

# ---------- threshold simulation (task 45) ----------
st.sidebar.header("Alert simulation (what-if)")
thr_sim = st.sidebar.slider("Alert threshold", 0.0, 1.0, round(thr, 3), 0.001)


def alert_stats(d: pd.DataFrame, t: float) -> dict:
    a, pos = d["risk"] >= t, int(d["actual"].sum())
    tp = int((a & (d["actual"] == 1)).sum())
    return {
        "alerts": a.mean() * 1000,
        "recall": tp / pos if pos else np.nan,
        "precision": tp / a.sum() if a.sum() else np.nan,
        "false": (a & (d["actual"] == 0)).sum() / len(d) * 1000,
    }


sim, frozen = alert_stats(f, thr_sim), alert_stats(f, thr)
pct = lambda x: "n/a" if np.isnan(x) else f"{x:.1%}"
k = st.columns(4)
k[0].metric(
    "Alerts per 1,000",
    f"{sim['alerts']:.1f}",
    f"{sim['alerts'] - frozen['alerts']:+.1f} vs frozen",
    delta_color="off",
)
k[1].metric("Recall", pct(sim["recall"]))
k[2].metric("Precision", pct(sim["precision"]))
k[3].metric(
    "False alerts per 1,000",
    f"{sim['false']:.1f}",
    f"{sim['false'] - frozen['false']:+.1f} vs frozen",
    delta_color="off",
)
if abs(thr_sim - thr) > 0.0005:
    st.info(
        f"Simulation only. The real threshold stays frozen at {thr:.4f}. Picking a new one from this "
        "page would re-tune on validation data, so do not report it as a result."
    )

# ---------- service overview (task 41) ----------
st.subheader("Service risk overview")
g = f.groupby("service_name").agg(
    events=("risk", "size"), mean_risk=("risk", "mean"), incidents=("actual", "sum")
)
g["alerts"] = (
    f[f.risk >= thr_sim].groupby("service_name").size().reindex(g.index).fillna(0).astype(int)
)
g["alerts_per_1000"] = g.alerts / g.events * 1000
g["incident_rate"] = g.incidents / g.events
left, right = st.columns([3, 2])
left.dataframe(g.sort_values("alerts_per_1000", ascending=False).round(3))
right.bar_chart(g["alerts_per_1000"])

# ---------- risk distribution (task 42) ----------
st.subheader("Risk score distribution")
fig, ax = plt.subplots(figsize=(8, 3))
bins = np.linspace(0, 1, 41)
ax.hist(f.loc[f.actual == 0, "risk"], bins, alpha=0.6, label="no incident")
ax.hist(f.loc[f.actual == 1, "risk"], bins, alpha=0.6, label="incident within 60 min")
ax.axvline(thr, color="k", ls=":", label="frozen threshold")
ax.axvline(thr_sim, color="r", ls="--", label="simulated threshold")
ax.set_yscale("log")
ax.set_xlabel("predicted risk")
ax.set_ylabel("events (log)")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ---------- filtered table (task 43) ----------
st.subheader("Filtered incident risk table")
only = st.checkbox("Show only events at or above the simulated threshold", value=True)
view = (f[f.risk >= thr_sim] if only else f).sort_values("risk", ascending=False)
cols = [
    c
    for c in [
        "event_id",
        "event_timestamp",
        "service_name",
        "environment",
        "region",
        "customer_tier",
        "risk",
        "actual",
        "error_rate_pct",
        "p95_latency_ms",
        "cpu_utilization_pct",
        "queue_depth",
        "active_alerts",
    ]
    if c in view
]
st.caption(f"{len(view):,} events")
st.dataframe(
    view[cols].head(500),
    hide_index=True,
    column_config={
        "risk": st.column_config.ProgressColumn("risk", min_value=0.0, max_value=1.0, format="%.3f")
    },
)

# ---------- explanation panel (task 44) ----------
st.subheader("Why did the model score it this way?")
e1, e2 = st.columns(2)
if SHAP_IMG.exists():
    e1.image(str(SHAP_IMG), caption="Global SHAP summary (validation sample)")
else:
    e1.info("Run `make explain` to create the global SHAP plot.")
top = view.head(50)
if top.empty:
    e2.info("No events to explain with the current filters.")
else:
    risk_of = dict(zip(top.event_id, top.risk))
    pick = e2.selectbox(
        "Pick one of the 50 highest-risk events",
        list(top.event_id),
        format_func=lambda e: f"{e}  (risk {risk_of[e]:.3f})",
    )
    row = top[top.event_id == pick].iloc[0]
    e2.write(
        f"**Risk {row.risk:.3f}**, actual label: **{'incident' if row.actual == 1 else 'no incident'}**"
    )
    e2.write("Top drivers (positive pushes toward an alert):")
    for d in str(row.top_drivers).split("; "):
        e2.markdown(f"- {d}")
    e2.caption(
        "SHAP units are log-odds of the model before calibration. On synthetic data they describe "
        "the simulator's rules, not real incident causes."
    )

# ---------- frozen test metrics + live API ----------
with st.expander("Final test metrics (frozen, evaluated once)"):
    st.json(json.loads(TEST_METRICS.read_text())) if TEST_METRICS.exists() else st.write(
        "Not available yet."
    )
with st.expander("Live API check"):
    if st.button("Check /ready and /model-info"):
        try:
            st.json(
                {
                    "ready": httpx.get(f"{API_URL}/ready", timeout=3).json(),
                    "model_info": httpx.get(f"{API_URL}/model-info", timeout=3).json(),
                }
            )
        except Exception as e:
            st.error(f"API not reachable at {API_URL}: {type(e).__name__}")
