"""
PRISM — Root entry-point shim.
Re-exports the FastAPI application and universe symbols from app.main
for backwards compatibility with existing deployments, tests, and CLI runners.

Supports:
    uvicorn main:app
    uvicorn app.main:app
"""

import os

import uvicorn

from app.main import (  # noqa: F401
    CURATED_STOCK_METRICS,
    IN_AGGREGATE_MODEL_METRICS,
    IN_SYMBOL_TO_IDX,
    IN_SYMBOL_TO_META,
    INDIA_CURATED_STOCK_METRICS,
    INDIA_STOCKS,
    US_AGGREGATE_MODEL_METRICS,
    US_STOCKS,
    US_SYMBOL_TO_IDX,
    US_SYMBOL_TO_META,
    _fetch_yf,
    app,
    get_metrics,
    get_prediction,
    health,
    health_detailed,
    healthz,
    model_aggregate_metrics,
    model_info,
    model_metrics,
    monitoring_backfill,
    monitoring_drift,
    monitoring_performance,
    predict,
    ready,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
