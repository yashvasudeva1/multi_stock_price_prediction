"""
Pydantic response schemas for PRISM API.

All public-facing response models are defined here so FastAPI can generate
accurate OpenAPI docs and validate outbound data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ── Ops ───────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    models_loaded: bool


class ReadyResponse(BaseModel):
    status: str
    models_loaded: bool
    us_model: bool
    in_model: bool


class ErrorResponse(BaseModel):
    error: str
    message: str


# ── Stock / watchlist ─────────────────────────────────────────────────────────

class StockEntry(BaseModel):
    symbol: str
    name: str
    sector: str


# ── Model info ────────────────────────────────────────────────────────────────

class ModelArchitecture(BaseModel):
    hidden: int
    n_layers: int
    embed_dim: int
    dropout: float
    n_feat: int


class ModelInfoResponse(BaseModel):
    architecture: ModelArchitecture
    seq_len: int
    n_stocks: int
    model_file: str
    device: str
    market: str
    model_name: str
    model_version: str
    framework: str
    loaded_at: str | None = None


# ── Metrics ──────────────────────────────────────────────────────────────────

class MetricBlock(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    r2: float | None = None
    dir_acc: float | None = None
    max_err: float | None = None
    n: int | None = None


class MetricsChart(BaseModel):
    labels: list[str]
    train: list[float | None]
    test: list[float | None]


class StockMetricsResponse(BaseModel):
    symbol: str
    scope: str
    source: str
    total_samples: int | None = None
    train_size: int | None = None
    test_size: int | None = None
    train: MetricBlock
    test: MetricBlock
    metrics_chart: MetricsChart


class AggregateMetricRow(BaseModel):
    metric: str
    train: float
    test: float
    difference: float


class AggregateMetricsResponse(BaseModel):
    scope: str
    source: str
    market: str
    rows: list[AggregateMetricRow]


# ── Prediction ────────────────────────────────────────────────────────────────

class ForecastStep(BaseModel):
    price: float
    return_: float
    change_pct: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    symbol: str
    company_name: str
    sector: str
    latest_close: float
    predicted_next: float
    change_pct: float
    market_cap: int
    avg_volume: int
    dates: list[str]
    actual_prices: list[float]
    predicted_prices: list[float]
    five_day_forecast: list[float]
    five_day_details: list[dict[str, Any]]
    market: str
    currency: str


# ── History ──────────────────────────────────────────────────────────────────

class HistoryResponse(BaseModel):
    symbol: str
    period: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[int]
