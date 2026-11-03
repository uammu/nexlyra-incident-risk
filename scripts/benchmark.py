import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from incident_risk.config import ROOT, load_config
from incident_risk.data import load_raw
from incident_risk.splits import chronological_split

def stats(ms):
    return {"p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "p99_ms": round(float(np.percentile(ms, 99)), 2), "mean_ms": round(float(np.mean(ms)), 2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    a = ap.parse_args()
    cfg = load_config()
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    train, _, _ = chronological_split(load_raw(cfg), cfg)          # train rows only: timing, not evaluation
    rows = json.loads(train[feats].dropna().sample(1000, random_state=cfg["seed"]).to_json(orient="records"))
    res = {"url": a.url, "cpu_count": os.cpu_count(), "platform": platform.platform()}

    with httpx.Client(base_url=a.url, timeout=30) as c:
        c.get("/ready").raise_for_status()
        for r in rows[:30]:                                         # warm-up, not measured
            c.post("/predict", json=r)
        lat = []
        for r in rows[: a.n]:
            t0 = time.perf_counter()
            c.post("/predict", json=r).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        res["single_sequential"] = {"n": a.n, **stats(lat)}

        blat = []
        for k in range(30):
            chunk = rows[k * a.batch:(k + 1) * a.batch]
            t0 = time.perf_counter()
            c.post("/predict/batch?suppress=false", json={"events": chunk}).raise_for_status()
            blat.append((time.perf_counter() - t0) * 1000)
        res["batch"] = {"batch_size": a.batch, **stats(blat),
                        "ms_per_event": round(float(np.mean(blat)) / a.batch, 3)}

    def worker(chunk):
        out = []
        with httpx.Client(base_url=a.url, timeout=30) as c:
            for r in chunk:
                t0 = time.perf_counter()
                code = c.post("/predict", json=r).status_code
                out.append((code, (time.perf_counter() - t0) * 1000))
        return out

    res["concurrency"] = []
    for k in a.concurrency:
        chunks = [rows[: a.n][i::k] for i in range(k)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(k) as ex:
            done = [x for part in ex.map(worker, chunks) for x in part]
        wall = time.perf_counter() - t0
        res["concurrency"].append({"clients": k, "requests_per_sec": round(len(done) / wall, 1),
                                   "errors": sum(code != 200 for code, _ in done),
                                   **stats([ms for _, ms in done])})
    (ROOT / cfg["paths"]["reports"] / "benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()