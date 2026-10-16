"""Structured JSON logging + per-request IDs.

Every log line is one JSON object with ``ts / level / logger / message /
request_id`` plus any extra fields passed via ``logger.<level>("…", extra={})``.
Request-scoped fields (``method / path / status / duration_ms``) are picked
up automatically from the ``extra`` dict.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


_RESERVED_LOGRECORD_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """One compact JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Lift any custom extras passed via logger.info("…", extra={"k": v}).
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter as the sole handler on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Make uvicorn + fastapi obey the same formatter
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        logger = logging.getLogger(noisy)
        logger.handlers = []
        logger.propagate = True
