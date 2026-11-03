import hashlib
import json
import logging
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from api.schemas import EventIn, RiskOut
from incident_risk.alert_policy import LEVELS, Suppressor, alert_level
from incident_risk.config import ROOT, load_config

cfg = load_config()
MODEL_PATH = ROOT / cfg["paths"]["model"]
MANIFEST = ROOT / "models/artifact_manifest.json"
FEATURES = cfg["features"]["numeric"] + cfg["features"]["categorical"]
POLICY = cfg["policy"]
SUPPRESSOR = Suppressor(POLICY["suppression_cooldown_minutes"])

app = FastAPI(title="Nexlyra Incident Risk API", version="0.2.0")
app.mount("/metrics", make_asgi_app())
REQS = Counter("risk_requests_total", "Predictions served")
ALERTS = Counter("risk_alerts_total", "Alerts raised", ["level"])
SUPPRESSED = Counter("risk_alerts_suppressed_total", "Alerts muted by cooldown")
SCORES = Histogram(
    "risk_score", "Predicted risk", buckets=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
)

audit = logging.getLogger("audit")  # append-only JSON lines, no raw inputs
(ROOT / "logs").mkdir(exist_ok=True)
_h = logging.FileHandler(ROOT / "logs/audit.jsonl")
_h.setFormatter(logging.Formatter("%(message)s"))
audit.addHandler(_h)
audit.setLevel(logging.INFO)
audit.propagate = False


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _log(**kw):
    audit.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}))


_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run `make train`.")
        try:
            version = "unversioned"
            if MANIFEST.exists():
                m = json.loads(MANIFEST.read_text())
                if m["model_sha256"] != _sha(MODEL_PATH):
                    raise ValueError("model checksum differs from manifest (re-run make register)")
                version = m["version"]
            b = joblib.load(MODEL_PATH)
            if list(b["meta"]["features"]) != FEATURES:
                raise ValueError("feature order mismatch")
            b["version"] = version
        except Exception as e:
            _log(event="model_load_failed", error=str(e)[:200])
            raise HTTPException(503, "Model artifact is corrupted or incompatible.") from e
        _bundle = b
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=RiskOut)
def predict(event: EventIn):
    b, rid, row = bundle(), uuid.uuid4().hex, event.model_dump()
    try:
        p = float(b["model"].predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    except Exception as e:
        _log(event="scoring_failed", request_id=rid, error=str(e)[:200])
        raise HTTPException(500, "Scoring failed.") from e
    thr = b["threshold"]
    level = alert_level(p, thr, row["environment"], POLICY)
    suppressed = SUPPRESSOR.check((row["service_name"], row["region"], row["environment"]), level)
    REQS.inc()
    SCORES.observe(p)
    if level:
        ALERTS.labels(LEVELS[level]).inc()
    if suppressed:
        SUPPRESSED.inc()
    _log(
        event="prediction",
        request_id=rid,
        model=b["meta"]["selected_model"],
        artifact_version=b["version"],
        input_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        service=row["service_name"],
        environment=row["environment"],
        risk=round(p, 5),
        threshold=round(thr, 5),
        level=level,
        suppressed=suppressed,
    )
    return RiskOut(
        request_id=rid,
        risk_probability=p,
        alert=level > 0,
        level=level,
        level_name=LEVELS[level],
        suppressed=suppressed,
        notify=level > 0 and not suppressed,
        threshold=thr,
        model=b["meta"]["selected_model"],
        artifact_version=b["version"],
    )
