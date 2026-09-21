"""Structured JSON logging.

Every record is emitted as one JSON object per line so Railway / Docker log
collectors can index it. Any `extra={...}` passed to a logger call is merged
into the object, which is how `request_id`, `latency_ms`, etc. get attached.

Rules enforced by convention across the codebase:
    - always include `request_id` on per-request records
    - never log API keys
    - raw user text only at DEBUG level
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes that exist on every LogRecord; anything else came from `extra`.
_STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    # Third-party chatter is noisy at INFO; keep it to warnings.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
