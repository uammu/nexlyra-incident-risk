import pytest

from incident_risk.alert_policy import Suppressor, alert_level
from incident_risk.champion import load_champion, verify_feature_order
from incident_risk.config import ROOT, load_config

POL = {"l2_multiplier": 2.0, "l3_multiplier": 3.0, "page_environments": ["production"]}
needs_model = pytest.mark.skipif(
    not (ROOT / load_config()["paths"]["model"]).exists(), reason="run make train first"
)


def test_empty_input_rejected():
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict").status_code == 422


def test_corrupted_artifact_returns_503(tmp_path, monkeypatch, good_payload):
    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr(m, "MODEL_PATH", bad)
    monkeypatch.setattr(m, "_bundle", None)
    assert client.post("/predict", json=good_payload).status_code == 503


@needs_model
def test_unseen_category_still_scores(good_payload):
    r = client.post("/predict", json={**good_payload, "service_name": "brand_new_service"})
    assert r.status_code == 200 and 0 <= r.json()["risk_probability"] <= 1


@needs_model
def test_boundary_values_score(good_payload):
    r = client.post("/predict", json={**good_payload, "cpu_utilization_pct": 100, "queue_depth": 0})
    assert r.status_code == 200


@needs_model
def test_feature_order_verified(cfg):
    assert verify_feature_order(load_champion(cfg), cfg)


def test_alert_levels():
    assert alert_level(0.05, 0.1, "production", POL) == 0
    assert alert_level(0.12, 0.1, "production", POL) == 1
    assert alert_level(0.25, 0.1, "production", POL) == 2
    assert alert_level(0.35, 0.1, "production", POL) == 3
    assert alert_level(0.35, 0.1, "staging", POL) == 2  # staging never pages


def test_suppression_and_escalation():
    s, k = Suppressor(60), ("svc", "r", "production")
    assert s.check(k, 1, now=0) is False
    assert s.check(k, 1, now=600) is True  # repeat inside cooldown is muted
    assert s.check(k, 2, now=900) is False  # escalation always passes
    assert s.check(k, 1, now=5000) is False  # cooldown expired
