"""
monitoring.py — Structured logging, Prometheus metrics, and prediction persistence.

Every prediction request is logged in structured JSON and stored to a lightweight
SQLite database so actual prices can be joined later for accuracy evaluation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

log = logging.getLogger("prism.monitoring")

# ── Prometheus metrics ────────────────────────────────────────────────────────
# Labels kept low-cardinality: market (US/IN) + endpoint only.
# Symbol is tracked separately in the DB.

PREDICTION_REQUESTS = Counter(
    "prism_prediction_requests_total",
    "Total prediction requests",
    ["market", "status"],
)
PREDICTION_ERRORS = Counter(
    "prism_prediction_errors_total",
    "Total prediction errors",
    ["market"],
)
REQUEST_LATENCY = Histogram(
    "prism_request_latency_seconds",
    "Request processing latency",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
MODEL_INFERENCE_LATENCY = Histogram(
    "prism_model_inference_latency_seconds",
    "Model inference latency (feature engineering + forward pass)",
    ["market"],
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)


def get_metrics_output() -> tuple[bytes, str]:
    """Return (body, content_type) for GET /metrics."""
    return generate_latest(), CONTENT_TYPE_LATEST


# ── Structured event logger ───────────────────────────────────────────────────

_struct_log = logging.getLogger("prism.events")


def log_prediction_event(
    symbol: str,
    market: str,
    model_version: str,
    latency_ms: float,
    status: int,
    request_id: str | None = None,
) -> None:
    """Log a structured JSON event for each prediction request."""
    event = {
        "event":         "prediction",
        "request_id":    request_id or str(uuid.uuid4()),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "symbol":        symbol,
        "market":        market,
        "model_version": model_version,
        "latency_ms":    round(latency_ms, 2),
        "status":        status,
    }
    _struct_log.info(json.dumps(event))


# ── Prediction persistence (SQLite) ──────────────────────────────────────────

_DB_PATH = Path("monitoring") / "predictions.db"


def _get_connection() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create the predictions table if it doesn't exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id  TEXT PRIMARY KEY,
                timestamp      TEXT NOT NULL,
                symbol         TEXT NOT NULL,
                market         TEXT NOT NULL,
                model_version  TEXT NOT NULL,
                horizon        INTEGER NOT NULL,
                predicted_value REAL NOT NULL,
                actual_value   REAL,
                error          REAL,
                absolute_error REAL
            )
        """)
        conn.commit()


def store_prediction(
    symbol: str,
    market: str,
    model_version: str,
    horizon: int,
    predicted_value: float,
) -> str:
    """Persist a prediction record. Returns the generated prediction_id."""
    try:
        _init_db()
        prediction_id = str(uuid.uuid4())
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                    (prediction_id, timestamp, symbol, market, model_version,
                     horizon, predicted_value, actual_value, error, absolute_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    prediction_id,
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    market,
                    model_version,
                    horizon,
                    predicted_value,
                ),
            )
            conn.commit()
        return prediction_id
    except Exception as exc:
        log.warning("Failed to persist prediction: %s", exc)
        return ""


# ── Timer context helper ──────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer for measuring latency."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
