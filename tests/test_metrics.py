import numpy as np

from incident_risk.metrics import evaluate, threshold_for_budget


def test_budget_threshold():
    p = np.linspace(0, 1, 1000)
    thr = threshold_for_budget(p, 50)
    assert abs((p >= thr).sum() - 50) <= 1


def test_perfect_model():
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0.1] * 90 + [0.9] * 10)
    m = evaluate(y, p, 0.5)
    assert m["recall_at_budget"] == 1.0 and m["false_alerts_per_1000"] == 0
