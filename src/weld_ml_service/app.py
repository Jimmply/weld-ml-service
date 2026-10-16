"""FastAPI service: /predict, /predict/batch, /health, /metrics."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, generate_latest

from . import __version__
from .logging_config import configure_logging, request_id_var
from .model import MODEL_PATH, ModelBundle, load_bundle, save_bundle, train_bundle
from .schemas import (
    BatchPredictionResponse,
    ClassProbability,
    HealthResponse,
    PredictionResponse,
    WeldFeatures,
    WeldFeaturesBatch,
)

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("weld_ml_service.access")

# Prometheus metrics — small set, deliberately
PREDICT_COUNTER = Counter(
    "weld_ml_predictions_total",
    "Total predictions served",
    ["label"],
)
PREDICT_LATENCY = Histogram(
    "weld_ml_predict_latency_seconds",
    "Latency of /predict in seconds",
)

# Simple module-level state (good enough for a single-process service)
_bundle: ModelBundle | None = None
_startup_time: float = 0.0


def _ensure_bundle() -> ModelBundle:
    """Load the bundled model, training it on-the-fly the first time if absent."""
    global _bundle
    if _bundle is not None:
        return _bundle
    path = Path(MODEL_PATH)
    if not path.exists():
        logger.warning("no model at %s — training a fresh baseline (synthetic data)", path)
        bundle = train_bundle()
        save_bundle(bundle)
    else:
        bundle = load_bundle(path)
    _bundle = bundle
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    _ensure_bundle()
    logger.info("weld-ml-service started")
    yield
    logger.info("weld-ml-service shutting down")


app = FastAPI(
    title="Weld ML Service",
    description="Laser-weld quality classifier served via FastAPI. Self-contained demo for the MLOps portfolio piece.",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_and_access_log(request: Request, call_next):
    """Attach a request_id to every log line + emit a single access log at the end."""
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status = 500   # pessimistic default if the handler raises
    response: Response | None = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        access_logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(duration_ms, 2),
            },
        )
        if response is not None:
            response.headers["x-request-id"] = rid
        request_id_var.reset(token)


def _predict_one(features: WeldFeatures, bundle: ModelBundle) -> PredictionResponse:
    x = np.array(features.to_vector()).reshape(1, -1)
    probs = bundle.estimator.predict_proba(x)[0]
    label_idx = int(np.argmax(probs))
    label = bundle.classes[label_idx]
    PREDICT_COUNTER.labels(label=label).inc()
    return PredictionResponse(
        label=label,
        probabilities=[
            ClassProbability(label=bundle.classes[i], probability=float(p))
            for i, p in enumerate(probs)
        ],
        model_version=bundle.version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WeldFeatures) -> PredictionResponse:
    bundle = _ensure_bundle()
    with PREDICT_LATENCY.time():
        return _predict_one(features, bundle)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: WeldFeaturesBatch) -> BatchPredictionResponse:
    bundle = _ensure_bundle()
    if not batch.items:
        raise HTTPException(status_code=400, detail="empty batch")
    with PREDICT_LATENCY.time():
        return BatchPredictionResponse(
            predictions=[_predict_one(f, bundle) for f in batch.items]
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _ensure_bundle()
    return HealthResponse(
        status="ok",
        model_version=bundle.version,
        model_cv_accuracy=round(bundle.accuracy_cv, 4),
        uptime_seconds=round(time.time() - _startup_time, 2),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/")
def root() -> dict:
    return {
        "service": "weld-ml-service",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
