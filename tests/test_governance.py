import pytest

from incident_risk.alert_policy import Suppressor, alert_level
from incident_risk.champion import load_champion, verify_feature_order
from incident_risk.config import ROOT, load_config

POL = {"l2_multiplier": 2.0, "l3_multiplier": 3.0, "page_environments": ["production"]}
needs_model = pytest.mark.skipif(
    not (ROOT / load_config()["paths"]["model"]).exists(), reason="run train first"
)


@needs_model
def test_feature_order_verified(cfg):
    assert verify_feature_order(load_champion(cfg), cfg)


def test_alert_levels():
    assert alert_level(0.05, 0.1, "production", POL) == 0
    assert alert_level(0.12, 0.1, "production", POL) == 1
    assert alert_level(0.25, 0.1, "production", POL) == 2
    assert alert_level(0.35, 0.1, "production", POL) == 3
    assert alert_level(0.35, 0.1, "staging", POL) == 2


def test_suppression_and_escalation():
    s, k = Suppressor(60), ("svc", "r", "production")
    assert s.check(k, 1, now=0) is False
    assert s.check(k, 1, now=600) is True
    assert s.check(k, 2, now=900) is False
    assert s.check(k, 1, now=5000) is False
