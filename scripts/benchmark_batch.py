import json
import time

import httpx
import numpy as np

from incident_risk.config import ROOT, load_config
from incident_risk.data import load_raw
from incident_risk.splits import chronological_split

cfg = load_config()
feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
train, _, _ = chronological_split(load_raw(cfg), cfg)
rows = json.loads(
    train[feats].dropna().sample(3000, random_state=cfg["seed"]).to_json(orient="records")
)
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
        out[f"batch_{size}"] = {
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "max_ms": round(float(max(lat)), 2),
            "ms_per_event": round(float(np.mean(lat)) / size, 3),
        }
(ROOT / "reports" / "benchmark_batch.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
