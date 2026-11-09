from pathlib import Path

FILES = {}

FILES["scripts/log_environment.py"] = """import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version

from incident_risk.config import ROOT

pkgs = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "optuna", "shap", "mlflow",
        "fastapi", "pydantic", "streamlit", "scipy", "statsmodels", "pytest", "ruff"]
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
except (OSError, subprocess.CalledProcessError):
    commit = "nogit"
env = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
       "git_commit": commit, "packages": {p: version(p) for p in pkgs}}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "environment.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))
"""

FILES["scripts/make_notebook.py"] = """import json

from incident_risk.config import ROOT


def cell(kind, text):
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        base.update(outputs=[], execution_count=None)
    return base


cells = [
    cell("markdown", "# EDA (train period only)\\nSYNTHETIC data. Run top to bottom."),
    cell("code", "from incident_risk import eda\\neda.main()"),
    cell("code", "from IPython.display import Image, display\\nimport glob\\n"
                 "for f in sorted(glob.glob('../reports/eda/*.png')):\\n    display(Image(f))"),
    cell("code", "import pandas as pd\\npd.read_csv('../reports/eda/univariate_auc.csv', index_col=0)"),
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks").mkdir(exist_ok=True)
(ROOT / "notebooks" / "01_eda.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/01_eda.ipynb")
"""

FILES["scripts/benchmark_batch.py"] = """import json
import time

import httpx
import numpy as np

from incident_risk.config import ROOT, load_config
from incident_risk.data import load_raw
from incident_risk.splits import chronological_split

cfg = load_config()
feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
train, _, _ = chronological_split(load_raw(cfg), cfg)
rows = json.loads(train[feats].dropna().sample(3000, random_state=cfg["seed"]).to_json(orient="records"))
out = {}
with httpx.Client(base_url="http://localhost:8000", timeout=60) as c:
    c.get("/ready").raise_for_status()
    for size in (10, 100, 500):
        lat = []
        for k in range(5):
            chunk = rows[k * size : (k + 1) * size]
            t0 = time.perf_counter()
            r = c.post("/predict/batch?suppress=false", json={"events": chunk})
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        out[f"batch_{size}"] = {"p50_ms": round(float(np.percentile(lat, 50)), 2),
                                "max_ms": round(float(max(lat)), 2),
                                "ms_per_event": round(float(np.mean(lat)) / size, 3)}
(ROOT / "reports" / "benchmark_batch.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
"""

FILES["src/incident_risk/baseline_report.py"] = """import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

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
        rows.append({"model": name, "pr_auc": average_precision_score(val[t], p),
                     "roc_auc": roc_auc_score(val[t], p), "precision": tp / max(tp + fp, 1),
                     "recall": tp / max(tp + fn, 1), "f1": f1_score(val[t], a),
                     "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        pr, rc, _ = precision_recall_curve(val[t], p)
        fpr, tpr, _ = roc_curve(val[t], p)
        ax_pr.plot(rc, pr, label=name)
        ax_roc.plot(fpr, tpr, label=name)
    ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision"); ax_pr.legend(fontsize=6)
    ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate"); ax_roc.legend(fontsize=6)
    fig_pr.savefig(out / "pr_curves.png", dpi=150, bbox_inches="tight")
    fig_roc.savefig(out / "roc_curves.png", dpi=150, bbox_inches="tight")
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    df.to_csv(out / "baseline_table.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
"""

for path, text in FILES.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print("created", path)

# A4: tighten API schema limits
s = Path("api/schemas.py")
t = s.read_text(encoding="utf-8")
if "Hours = " not in t:
    t = t.replace(
        "Pct = Annotated",
        "Hours = Annotated[float, Field(ge=0, le=10_000, allow_inf_nan=False)]\n"
        "Rpm = Annotated[float, Field(ge=0, le=1_000_000, allow_inf_nan=False)]\n"
        "LatMs = Annotated[float, Field(ge=0, le=60_000, allow_inf_nan=False)]\n"
        "Count = Annotated[float, Field(ge=0, le=10_000, allow_inf_nan=False)]\n"
        "Pct = Annotated",
        1,
    )
    for field, typ in [
        ("release_age_hours", "Hours"),
        ("requests_per_minute", "Rpm"),
        ("p95_latency_ms", "LatMs"),
        ("queue_depth", "Count"),
        ("dependency_failures_15m", "Count"),
        ("active_alerts", "Count"),
        ("incident_history_30d", "Count"),
    ]:
        t = t.replace(f"    {field}: NonNeg", f"    {field}: {typ}")
    s.write_text(t, encoding="utf-8")
    print("tightened api/schemas.py")

ta = Path("tests/test_api.py")
body = ta.read_text(encoding="utf-8")
if "test_absurd_value_rejected" not in body:
    ta.write_text(
        body
        + """

def test_absurd_value_rejected(api_client, good_payload):
    r = api_client.post("/predict", json={**good_payload, "queue_depth": 1_000_000})
    assert r.status_code == 422
""",
        encoding="utf-8",
    )
    print("added test")
