import hashlib
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from incident_risk.calibration import CalibratedModel
from incident_risk.features import build_preprocessor

CATS = {"service_name": ["inventory", "search", "checkout"], "environment": ["production", "staging"],
        "region": ["ap-south-1", "eu-west-1"], "deployment_type": ["none", "blue_green", "canary"],
        "customer_tier": ["standard", "business", "enterprise"]}

def build_tiny_bundle(tmp_path, cfg):
    f, t = cfg["features"], cfg["data"]["target"]
    feats = f["numeric"] + f["categorical"]
    rng, n = np.random.default_rng(0), 2000
    df = pd.DataFrame({c: rng.uniform(0, 1 if c in f.get("fraction_cols", []) else 100, n)
                       for c in f["numeric"]})
    for c in f["categorical"]:
        df[c] = rng.choice(CATS.get(c, ["a", "b"]), n)
    df[t] = (rng.random(n) < 0.03 + 0.5 * (df["error_rate_pct"] > 85)).astype(int)
    base = Pipeline([("prep", build_preprocessor(f["numeric"], f["categorical"])),
                     ("clf", LogisticRegression(max_iter=500))]).fit(df[feats], df[t])
    model = CalibratedModel(base, "sigmoid").fit_calibrator(df[feats], df[t])
    thr = float(np.quantile(model.predict_proba(df[feats])[:, 1], 0.95))
    path, manifest = tmp_path / "model.joblib", tmp_path / "artifact_manifest.json"
    joblib.dump({"model": model, "threshold": thr,
                 "meta": {"selected_model": "logreg_test", "features": feats,
                          "alert_budget_per_1000": 50}}, path)
    manifest.write_text(json.dumps({"version": "test-0",
                                    "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path, manifest

@pytest.fixture
def svc():
    import api.main as m
    orig = (m.service.model_path, m.service.manifest_path)
    yield m.service
    m.service.configure(*orig)

@pytest.fixture
def api_client(svc, tmp_path, cfg):
    from fastapi.testclient import TestClient
    import api.main as m
    svc.configure(*build_tiny_bundle(tmp_path, cfg))
    with TestClient(m.app) as c:
        yield c

@pytest.fixture
def no_model_client(svc, tmp_path):
    from fastapi.testclient import TestClient
    import api.main as m
    svc.configure(tmp_path / "missing.joblib", tmp_path / "missing.json")
    with TestClient(m.app) as c:
        yield c