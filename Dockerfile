# Lightweight image — python + uvicorn only, no build tools in the final layer.
FROM python:3.11-slim AS build

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Non-root runtime user
RUN useradd --create-home --shell /bin/bash app
USER app
ENV PATH=/home/app/.local/bin:$PATH

COPY --from=build --chown=app /install /usr/local
COPY --chown=app src ./src
COPY --chown=app scripts ./scripts
COPY --chown=app models ./models

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "weld_ml_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
