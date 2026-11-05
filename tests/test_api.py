import hashlib
import json

import pytest
from fastapi.testclient import TestClient

import api.main as m


# ---- startup, health, readiness ----
def test_liveness_without_model(no_model_client):
    r = no_model_client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is False


def test_readiness_503_without_model(no_model_client):
    r = no_model_client.get("/ready")
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_predict_503_without_model(no_model_client, good_payload):
    assert no_model_client.post("/predict", json=good_payload).status_code == 503


def test_readiness_ok(api_client):
    r = api_client.get("/ready")
    assert r.status_code == 200 and r.json()["artifact_version"] == "test-0"


def test_model_info(api_client):
    body = api_client.get("/model-info").json()
    assert "inventory" in body["known_categories"]["service_name"] and body["threshold"] > 0


# ---- valid request ----
def test_valid_request(api_client, good_payload):
    r = api_client.post("/predict", json={**good_payload, "event_id": "NXE-1"})
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["risk_probability"] <= 1 and b["model"] == "logreg_test"
    assert b["artifact_version"] == "test-0" and b["event_id"] == "NXE-1"
    assert b["level_name"] in {"none", "review", "acknowledge", "page"}
    assert b["request_id"] == r.headers["x-request-id"] and b["unknown_categories"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_utilization_pct", 0),
        ("cpu_utilization_pct", 100),
        ("queue_depth", 0),
        ("change_failure_rate_30d", 1.0),
    ],
)
def test_boundary_values_score(api_client, good_payload, field, value):
    assert api_client.post("/predict", json={**good_payload, field: value}).status_code == 200


# ---- invalid input ----
@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_utilization_pct", 150),
        ("cpu_utilization_pct", -1),
        ("p95_latency_ms", -5),
        ("change_failure_rate_30d", 1.5),
        ("queue_depth", "abc"),
        ("error_rate_pct", None),
    ],
)
def test_invalid_numbers_rejected(api_client, good_payload, field, value):
    r = api_client.post("/predict", json={**good_payload, field: value})
    err = r.json()["error"]
    assert r.status_code == 422 and err["code"] == "validation_error"
    assert any(d["field"] == field for d in err["details"])


def test_missing_required_field(api_client, good_payload):
    bad = {k: v for k, v in good_payload.items() if k != "queue_depth"}
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "queue_depth"


def test_extra_field_rejected(api_client, good_payload):
    assert api_client.post("/predict", json={**good_payload, "surprise": 1}).status_code == 422


def test_malformed_json(api_client):
    r = api_client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "malformed_json"


def test_empty_body(api_client):
    assert api_client.post("/predict").status_code == 422
    assert api_client.post("/predict", json={}).status_code == 422


def test_bad_category_format_rejected_and_not_echoed(api_client, good_payload):
    r = api_client.post("/predict", json={**good_payload, "service_name": "secret-value-123!"})
    assert r.status_code == 422 and "secret-value" not in r.text


def test_unknown_route_is_structured(api_client):
    r = api_client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


# ---- unknown categories ----
def test_unknown_category_scored_and_flagged(api_client, good_payload):
    r = api_client.post("/predict", json={**good_payload, "service_name": "brand_new_service"})
    assert r.status_code == 200 and r.json()["unknown_categories"] == ["service_name"]


def test_unknown_category_strict_mode(api_client, good_payload, monkeypatch):
    monkeypatch.setattr(m.service, "strict", True)
    r = api_client.post("/predict", json={**good_payload, "region": "mars-1"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_category"


# ---- request id ----
def test_request_id_echo_and_sanitised(api_client, good_payload):
    r = api_client.post("/predict", json=good_payload, headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.json()["request_id"] == "abc-123"
    r = api_client.post(
        "/predict", json=good_payload, headers={"X-Request-ID": "bad id with spaces"}
    )
    assert r.headers["x-request-id"] != "bad id with spaces"


# ---- batch ----
def _batch(good, n):
    return [{**good, "event_id": f"e{i}", "error_rate_pct": (i * 0.97) % 100} for i in range(n)]


def test_batch_100_records(api_client, good_payload):
    r = api_client.post("/predict/batch?suppress=false", json={"events": _batch(good_payload, 100)})
    b = r.json()
    assert r.status_code == 200 and b["count"] == 100 and len(b["results"]) == 100
    assert [x["event_id"] for x in b["results"]] == [f"e{i}" for i in range(100)]


def test_batch_matches_single(api_client, good_payload):
    ev = _batch(good_payload, 3)
    batch = api_client.post("/predict/batch?suppress=false", json={"events": ev}).json()["results"]
    single = api_client.post("/predict", json=ev[1]).json()["risk_probability"]
    assert batch[1]["risk_probability"] == pytest.approx(single, abs=1e-9)


def test_batch_oversized_413(api_client, good_payload, monkeypatch):
    monkeypatch.setattr(m, "MAX_BATCH", 5)
    r = api_client.post("/predict/batch", json={"events": _batch(good_payload, 6)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "batch_too_large"


def test_batch_empty_rejected(api_client):
    assert api_client.post("/predict/batch", json={"events": []}).status_code == 422


def test_batch_error_points_to_item(api_client, good_payload):
    ev = _batch(good_payload, 5)
    ev[3]["cpu_utilization_pct"] = 999
    r = api_client.post("/predict/batch", json={"events": ev})
    assert r.status_code == 422
    assert any(d["field"].startswith("events.3.") for d in r.json()["error"]["details"])


# ---- corrupted artifacts ----
@pytest.mark.parametrize("case", ["checksum_mismatch", "unreadable_pickle", "missing_manifest"])
def test_corrupted_artifact_503(case, svc, tmp_path, good_payload):
    model, manifest = tmp_path / "model.joblib", tmp_path / "manifest.json"
    model.write_bytes(b"not a model")
    if case == "checksum_mismatch":
        manifest.write_text(json.dumps({"version": "x", "model_sha256": "0" * 64}))
    elif case == "unreadable_pickle":
        manifest.write_text(
            json.dumps({"version": "x", "model_sha256": hashlib.sha256(b"not a model").hexdigest()})
        )
    svc.configure(model, manifest)
    with TestClient(m.app) as c:
        r = c.post("/predict", json=good_payload)
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


# ---- metrics ----
def test_metrics_exposed(api_client, good_payload):
    api_client.post("/predict", json=good_payload)
    text = api_client.get("/metrics/").text
    for name in [
        "risk_predictions_total",
        "risk_http_requests_total",
        "risk_inference_seconds_bucket",
        "risk_score_bucket",
        "risk_model_load_failures_total",
        "risk_model_loaded 1.0",
    ]:
        assert name in text
