from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

GOOD = dict(
    service_name="inventory",
    environment="production",
    region="ap-south-1",
    deployment_type="none",
    customer_tier="standard",
    release_age_hours=6.3,
    requests_per_minute=3911,
    p95_latency_ms=55,
    error_rate_pct=0.48,
    cpu_utilization_pct=75.2,
    memory_utilization_pct=64.21,
    db_connections_pct=38.18,
    queue_depth=24,
    dependency_failures_15m=0,
    change_failure_rate_30d=0.177,
    active_alerts=6,
    incident_history_30d=0,
)


def test_health():
    assert client.get("/health").status_code == 200


def test_invalid_percentage_rejected():
    assert client.post("/predict", json={**GOOD, "cpu_utilization_pct": 150}).status_code == 422


def test_negative_latency_rejected():
    assert client.post("/predict", json={**GOOD, "p95_latency_ms": -5}).status_code == 422


def test_fraction_above_one_rejected():
    assert client.post("/predict", json={**GOOD, "change_failure_rate_30d": 1.5}).status_code == 422
