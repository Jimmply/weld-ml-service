# 🚀 Weld ML Service

![CI](https://github.com/Jimmply/weld-ml-service/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009485?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Self-contained FastAPI prediction service for laser-weld quality classification.** 5-class RandomForest (good / spatter / porosity / undercut / cold weld) wrapped in a production-shaped service with `/predict`, `/predict/batch`, `/health`, `/metrics`, Pydantic validation, Prometheus-compatible metrics, Dockerfile, docker-compose, and CI that builds the image and smoke-tests it before merging.

Built as the MLOps companion to my manufacturing-ML portfolio — demonstrates I can ship a trained model, not only train one.

## Why this exists

Everyone has a `train.py` repo. Far fewer candidates have a `serve.py + Dockerfile + CI smoke test on the container` repo. This closes that gap.

## 60-second local run

```bash
git clone https://github.com/Jimmply/weld-ml-service.git
cd weld-ml-service
make setup          # venv + deps + trains bundled model
make run            # uvicorn on :8000

# then, in another shell:
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{
    "peak_power_W": 2100, "pulse_duration_ms": 10,
    "spot_diameter_mm": 0.7, "travel_speed_mm_s": 25,
    "assist_gas_flow_lpm": 15
  }'
```

```json
{
  "label": "good",
  "probabilities": [
    {"label": "good", "probability": 0.78},
    {"label": "spatter", "probability": 0.06},
    {"label": "porosity", "probability": 0.08},
    {"label": "undercut", "probability": 0.04},
    {"label": "cold_weld", "probability": 0.04}
  ],
  "model_version": "0.1.0"
}
```

OpenAPI docs auto-generated at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Docker

```bash
make docker-build   # builds a multi-stage image, runs as non-root 'app' user
make docker-run     # docker-compose up (healthcheck included)
curl -sf http://127.0.0.1:8000/health
make docker-stop
```

CI runs the full container smoke test on every push — see the "docker-build" job in `.github/workflows/ci.yml`.

## Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/predict` | POST | Single-weld classification |
| `/predict/batch` | POST | Batch classification (max 1024 items) |
| `/health` | GET | Status + model version + CV accuracy + uptime — Kubernetes/Docker-friendly |
| `/metrics` | GET | Prometheus-format counters + latency histogram |
| `/docs` | GET | Auto-generated OpenAPI (Swagger UI) |
| `/openapi.json` | GET | OpenAPI schema |

## Why the model is a RandomForest

Deliberately. This repo is about the **serving** story, not the modeling. A ~50 KB sklearn RF trained on physics-inspired synthetic weld features gives the service real behavior out-of-the-box without external data dependencies. In a real deployment you'd replace `models/weld_rf.joblib` with your real trained model (XGBoost, PyTorch, whatever) — the API surface stays identical because the swap happens behind `model.load_bundle()`.

See [weld-quality-classifier](https://github.com/Jimmply/weld-quality-classifier) for the "actually train the model" project. See [weld-defect-vision](https://github.com/Jimmply/weld-defect-vision) for the CV equivalent.

## Deployment targets

The repo is deployment-platform-neutral; here's how you'd ship it to the common ones:

| Platform | Steps |
|---|---|
| **Fly.io** | `fly launch` → `fly deploy`. Dockerfile is picked up automatically. $5/mo on the smallest machine. |
| **Railway** | Point at the GitHub repo, Dockerfile is auto-detected. Free tier handles light demo traffic. |
| **AWS Lambda** | Wrap the FastAPI app with Mangum (`pip install mangum`), add handler in `app.py`. Build container image, push to ECR, create Lambda function from image. |
| **GCP Cloud Run** | `gcloud run deploy --source .` — autobuild the Docker image, autoscale 0→N. |
| **k8s** | Image is already non-root + has a HEALTHCHECK; add a Deployment+Service manifest. |

Deliberately **not deployed to any of these yet** — adding "live demo URL" is a follow-up commit, not day-one scope.

## Project layout

```
weld-ml-service/
├── .github/workflows/ci.yml       # pytest + docker-build + container smoke
├── src/weld_ml_service/
│   ├── app.py                     # FastAPI endpoints + Prometheus metrics
│   ├── schemas.py                 # Pydantic validation
│   └── model.py                   # sklearn RF + synthetic data generator
├── scripts/train_baseline.py      # trains + saves models/weld_rf.joblib
├── models/weld_rf.joblib          # committed (<1 MB — repo is self-contained)
├── tests/test_api.py              # 9 smoke tests via FastAPI TestClient
├── Dockerfile                     # multi-stage, non-root, healthcheck
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Roadmap

- [ ] Live demo URL on Fly.io or Railway (next commit)
- [ ] Add structured JSON logging + request IDs
- [ ] Add rate limiting (slowapi or an API gateway)
- [ ] Postgres predictions log for monitoring drift over time
- [ ] Grafana dashboard config pulling from the `/metrics` endpoint
- [ ] Swap the baseline RF for the actual trained XGBoost from `weld-quality-classifier`

## Author

**Dmitry Shurkhai** — Manufacturing Data Scientist · [GitHub](https://github.com/Jimmply) · [LinkedIn](https://linkedin.com/in/etozhejimmy)
