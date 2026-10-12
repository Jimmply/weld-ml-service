.PHONY: setup train run docker-build docker-run docker-stop test lint clean help

PYTHON := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest

help:
	@echo "targets:"
	@echo "  make setup         create venv + install deps + train baseline"
	@echo "  make train         retrain the bundled sklearn baseline"
	@echo "  make run           uvicorn locally on port 8000"
	@echo "  make test          pytest (TestClient — no running server needed)"
	@echo "  make docker-build  build the production image"
	@echo "  make docker-run    docker-compose up"
	@echo "  make docker-stop   docker-compose down"
	@echo "  make clean         remove venv + caches"

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt pytest httpx
	$(PYTHON) scripts/train_baseline.py

train:
	$(PYTHON) scripts/train_baseline.py

run:
	PYTHONPATH=src $(UVICORN) weld_ml_service.app:app --reload --port 8000

test:
	PYTHONPATH=src $(PYTEST) tests/ -v

docker-build:
	docker build -t weld-ml-service:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	rm -rf .venv .pytest_cache __pycache__ src/weld_ml_service/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
