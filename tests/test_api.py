"""Smoke tests for the FastAPI service — uses TestClient, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weld_ml_service.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------------

def test_root_describes_the_service(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "weld-ml-service"
    assert body["health"] == "/health"


def test_health_reports_ok_and_version(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "0.1.0"
    assert 0.0 <= body["model_cv_accuracy"] <= 1.0
    assert body["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

VALID_FEATURES = {
    "peak_power_W": 2100,
    "pulse_duration_ms": 10.0,
    "spot_diameter_mm": 0.7,
    "travel_speed_mm_s": 25.0,
    "assist_gas_flow_lpm": 15.0,
}


def test_predict_returns_label_and_probabilities(client: TestClient):
    r = client.post("/predict", json=VALID_FEATURES)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"good", "spatter", "porosity", "undercut", "cold_weld"}
    assert len(body["probabilities"]) == 5
    total = sum(p["probability"] for p in body["probabilities"])
    assert 0.99 < total < 1.01   # probs should sum to 1
    assert body["model_version"] == "0.1.0"


def test_predict_rejects_out_of_range_power(client: TestClient):
    bad = {**VALID_FEATURES, "peak_power_W": 50_000}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field(client: TestClient):
    bad = {k: v for k, v in VALID_FEATURES.items() if k != "assist_gas_flow_lpm"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

def test_predict_batch_returns_one_prediction_per_item(client: TestClient):
    r = client.post("/predict/batch", json={"items": [VALID_FEATURES, VALID_FEATURES, VALID_FEATURES]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert p["label"] in {"good", "spatter", "porosity", "undercut", "cold_weld"}


def test_predict_batch_rejects_oversized_batch(client: TestClient):
    r = client.post("/predict/batch", json={"items": [VALID_FEATURES] * 2000})
    assert r.status_code == 422


def test_predict_batch_rejects_empty_items(client: TestClient):
    # Schema sets min_length=1 — empty batches should be a 422, not a server-side crash
    r = client.post("/predict/batch", json={"items": []})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_prometheus_text(client: TestClient):
    # Hit predict first so the counter has a value
    client.post("/predict", json=VALID_FEATURES)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "weld_ml_predictions_total" in body
    assert "weld_ml_predict_latency_seconds" in body


def test_openapi_schema_is_served(client: TestClient):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "Weld ML Service"
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]


# ---------------------------------------------------------------------------
# Structured logging + request IDs
# ---------------------------------------------------------------------------

def test_response_includes_request_id_header(client: TestClient):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    # Looks like a UUID
    rid = r.headers["x-request-id"]
    assert len(rid) >= 8  # some non-empty id


def test_client_can_supply_its_own_request_id(client: TestClient):
    custom = "abc-123-trace-id"
    r = client.get("/health", headers={"x-request-id": custom})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == custom


def test_access_log_is_structured_json(client: TestClient, caplog):
    import json
    import logging as pylog
    from weld_ml_service.logging_config import JsonFormatter

    # Attach a capture handler with the service's JSON formatter so we see
    # exactly the lines the real service would emit.
    captured = []

    class _Capture(pylog.Handler):
        def emit(self, record):
            captured.append(JsonFormatter().format(record))

    handler = _Capture()
    access_logger = pylog.getLogger("weld_ml_service.access")
    access_logger.addHandler(handler)
    try:
        client.get("/health")
    finally:
        access_logger.removeHandler(handler)

    assert captured, "no access log line was emitted"
    payload = json.loads(captured[-1])
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert payload["status"] == 200
    assert "duration_ms" in payload
    assert payload["request_id"] != "-"
