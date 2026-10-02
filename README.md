# PRISM — Multi-Stock Price Prediction Platform

A production-grade Deep Learning platform for simultaneous multi-stock forecasting using PyTorch LSTM models, served through a FastAPI backend with full MLOps infrastructure.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## Overview

PRISM uses multi-stock LSTM neural networks with per-stock learned embeddings to predict next-day price movements for **30 US equities** and **29 Indian NSE stocks** simultaneously.

Instead of training separate models per stock, a shared representation captures market-wide dynamics, sector influences, and cross-asset relationships.

---

## Project Architecture

```
prism/
├── main.py                      # FastAPI entry-point (thin — delegates to app/)
├── app/
│   ├── model_registry.py        # Model loading, caching, metadata
│   ├── predictor.py             # Feature engineering + inference + forecasting
│   ├── monitoring.py            # Prometheus metrics, structured logging, SQLite persistence
│   └── schemas.py               # Pydantic response schemas
├── models/
│   ├── us_stock_lstm.pth        # US LSTM model checkpoint
│   └── multi_stock_lstm_v2.pth  # India LSTM model checkpoint
├── tests/
│   ├── conftest.py              # Shared fixtures (mocked yfinance, async client)
│   ├── test_health.py           # /health and /ready endpoint tests
│   ├── test_prediction.py       # /api/predict tests
│   ├── test_metrics.py          # /api/model/metrics tests
│   └── test_model_loading.py    # ModelRegistry unit tests
├── monitoring/
│   ├── collect_predictions.py   # Backfill actual prices into predictions DB
│   ├── evaluate_predictions.py  # Compute MAE/RMSE/MAPE from stored predictions
│   └── drift_report.py          # PSI + KS drift detection on prediction distributions
├── scripts/
│   └── evaluate_model.py        # Offline evaluation from results/ CSV files
├── results/
│   ├── us/                      # US model evaluation CSVs
│   └── india/                   # India model evaluation CSVs
├── .github/workflows/
│   ├── ci.yml                   # CI: lint → test → Docker build → smoke test
│   ├── cd.yml                   # CD: trigger Render deployment → smoke test
│   └── monitoring.yml           # Daily: performance + drift monitoring reports
├── Dockerfile                   # Production image (Python 3.12-slim)
├── .dockerignore
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Dev/test dependencies
├── pytest.ini                  # Test configuration
├── ruff.toml                   # Linter configuration
├── conftest.py                 # Root pytest config (Windows DLL fix)
└── .env.example                # Environment variable template
```

### Architecture Flow

```
HTTP Request
     ↓
FastAPI (main.py)         ← thin: routing, validation, error handling
     ↓
app/predictor.py          ← feature engineering, model inference
     ↓
app/model_registry.py     ← model loading + caching (singleton)
     ↓
PyTorch LSTM models       ← us_stock_lstm.pth / multi_stock_lstm_v2.pth
     ↓
app/monitoring.py         ← Prometheus counters, structured JSON log, SQLite storage
```

---

## Model Architecture

### US Model (`us_stock_lstm.pth`)
| Parameter | Value |
|---|---|
| Input features | 26 (OHLCV + technical indicators) |
| Sequence length | 30 trading days |
| Stock embedding | 30 stocks → dim 12 |
| LSTM | 2 layers, hidden=128, dropout=0.35 |
| Output | Scalar log-return (next day) |

### India Model (`multi_stock_lstm_v2.pth`)
| Parameter | Value |
|---|---|
| Input features | 33 (OHLCV + technical + market benchmarks) |
| Sequence length | 30 trading days |
| Stock embedding | 29 stocks → dim 12 |
| Architecture | Embedding → Project (Linear+LN) → LSTM → LN → Head (Linear→GELU→Drop→Linear) |
| Output | Scalar log-return (next day) |

---

## Local Development

### 1. Clone

```bash
git clone https://github.com/yashvasudeva1/multi_stock_price_prediction.git
cd multi_stock_price_prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt    # dev + test dependencies
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env as needed — defaults work out of the box
```

### 5. Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

---

## API Endpoints

### Ops

| Method | Path | Description |
|---|---|---|
| GET | `/health` | **Liveness probe** — Is the process alive? Always lightweight. |
| GET | `/ready` | **Readiness probe** — Are models loaded and ready? |
| GET | `/metrics` | Prometheus metrics |
| GET | `/` | Service info |
| GET | `/docs` | Swagger UI |

#### `/health` — Liveness Probe

```bash
curl https://YOUR-DOMAIN/health
```

```json
{
  "status": "ok",
  "service": "multi-stock-price-prediction",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:00:00Z",
  "models_loaded": true
}
```

> **Important:** `/health` is intentionally **lightweight and side-effect-free**. It performs no model inference, no data downloads, and no file writes. It is safe to call frequently from uptime monitors and cron jobs.

#### `/ready` — Readiness Probe

```bash
curl https://YOUR-DOMAIN/ready
```

Returns `200` when models are loaded, `503` when not ready:

```json
{
  "status": "ready",
  "models_loaded": true,
  "us_model": true,
  "in_model": true
}
```

### Data

| Method | Path | Parameters | Description |
|---|---|---|---|
| GET | `/api/stocks` | `?market=US\|IN` | Full watchlist |
| GET | `/api/history` | `?symbol=AAPL&period=3mo&market=US` | OHLCV history |

### Model

| Method | Path | Parameters | Description |
|---|---|---|---|
| GET | `/api/model/info` | `?market=US\|IN` | Architecture + metadata |
| GET | `/api/predict` | `?symbol=AAPL&market=US` | LSTM prediction + 5-day forecast |
| GET | `/api/model/metrics` | `?symbol=AAPL&market=US` | Per-stock train/test metrics |
| GET | `/api/model/aggregate-metrics` | `?market=US\|IN` | Aggregate model performance |

#### Example Prediction

```bash
curl "https://YOUR-DOMAIN/api/predict?symbol=AAPL&market=US"
```

```json
{
  "symbol": "AAPL",
  "company_name": "Apple Inc.",
  "latest_close": 189.34,
  "predicted_next": 190.12,
  "change_pct": 0.4121,
  "five_day_forecast": [190.12, 190.85, 191.23, 191.67, 192.10],
  "currency": "$",
  ...
}
```

#### Error Responses

All errors return structured JSON — no raw stack traces:

```json
{
  "detail": {
    "error": "SYMBOL_NOT_FOUND",
    "message": "Symbol 'XYZ' not in US model universe"
  }
}
```

---

## Running Tests

```bash
pytest                          # run all tests
pytest -v                       # verbose
pytest tests/test_health.py     # single file
pytest --cov=app --cov=main     # with coverage
```

Tests are deterministic — yfinance calls are mocked with synthetic OHLCV data.

**Test coverage:**
- `/health` and `/ready` endpoint contracts
- `/api/predict` response schema and numeric validity
- `/api/model/metrics` and `/api/model/aggregate-metrics` structure
- `ModelRegistry` loading, singleton, graceful degradation when files missing
- Invalid input → controlled 4xx (not 500)

---

## Docker

### Build

```bash
docker build -t prism:latest .
```

### Run

```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  -e TRUSTED_MODEL_CHECKPOINT=true \
  prism:latest
```

### Test the container

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl "http://localhost:8000/api/predict?symbol=AAPL&market=US"
```

---

## CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

Runs on every `push` to `main` and every pull request:

```
Checkout
  ↓
Python 3.12 + pip cache
  ↓
Install production deps
  ↓
Install dev deps
  ↓
Ruff (lint)
  ↓
pytest (31 tests)
  ↓
Docker build
  ↓
Smoke test /health in container
  ↓
PASS / FAIL
```

### CD Pipeline (`.github/workflows/cd.yml`)

Runs on merge to `main`. Optionally triggers Render via deploy hook and smoke-tests `/health` on production.

### Daily Monitoring (`.github/workflows/monitoring.yml`)

Runs at 02:00 UTC daily:
- Evaluate stored predictions (MAE/RMSE/MAPE)
- Generate drift report (PSI, KS statistic)
- Upload reports as GitHub Actions artifacts
- Monitoring failures do NOT block deployment

---

## Deployment on Render

### Automatic Git Deployment (Recommended)

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repository
3. Configure:

| Setting | Value |
|---|---|
| **Environment** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |

4. The app binds to `0.0.0.0:$PORT` automatically — Render injects `PORT`.

### Environment Variables on Render

Set these in Render → Environment:

| Variable | Required | Description |
|---|---|---|
| `PORT` | Auto-injected | Do not set — Render sets this |
| `TRUSTED_MODEL_CHECKPOINT` | Optional | `true` (default) to load `.pth` files |
| `US_MODEL_PATH` | Optional | Path to US model (default: `us_stock_lstm.pth`) |
| `INDIA_MODEL_PATH` | Optional | Path to India model (default: `multi_stock_lstm_v2.pth`) |
| `APP_VERSION` | Optional | Version string shown in `/health` |
| `GIT_COMMIT` | Optional | Inject via CI: `${{ github.sha }}` |

### GitHub Secrets for CD

Add these in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `RENDER_DEPLOY_HOOK_URL` | From Render → Service → Settings → Deploy Hook |
| `RENDER_APP_URL` | e.g. `https://my-app.onrender.com` |

---

## Uptime Monitoring / Cron Health Check

`/health` is designed to be safe for frequent polling. **Example cron jobs:**

**Linux / macOS** — check every 10 minutes:
```bash
*/10 * * * * curl -fsS https://YOUR-APP.onrender.com/health > /dev/null || echo "API health check failed"
```

**Windows** — use Task Scheduler, or an external uptime service (e.g. UptimeRobot, Better Uptime) pointing to:
```
GET https://YOUR-APP.onrender.com/health
Expected status: 200
```

> **Note:** Do not create this cron job automatically — add it manually to your monitoring setup.

---

## Prometheus Metrics

```bash
curl https://YOUR-DOMAIN/metrics
```

Available metrics:
| Metric | Type | Labels | Description |
|---|---|---|---|
| `prism_prediction_requests_total` | Counter | `market`, `status` | Total prediction requests |
| `prism_prediction_errors_total` | Counter | `market` | Total prediction errors |
| `prism_request_latency_seconds` | Histogram | `endpoint` | Request latency |
| `prism_model_inference_latency_seconds` | Histogram | `market` | Model inference latency |

---

---

## Streamlit Model Drift Radar Dashboard

PRISM includes an interactive, dark-themed Streamlit monitoring dashboard matching the web UI's visual identity (deep onyx background, gold/cyan accents, and glassmorphic cards).

### Launch the Dashboard

```bash
streamlit run monitoring/dashboard.py
```

### Dashboard Capabilities

- **Real-Time Drift Diagnostics**: Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics between a historical baseline (e.g. 30 days) and recent production predictions (e.g. 7 days).
- **Interactive Visualizations**:
  - Overlaid distribution histograms + KDE curves comparing baseline vs. recent prediction spreads.
  - Empirical Cumulative Distribution Functions (ECDF) highlighting maximum KS divergence ($D$).
  - Quantile migration bar charts (P10, P25, P50, P75, P90).
- **Simulation Sandbox**: Allows you to simulate different market regime shifts ("Bull Market Shift +25%", "High Volatility Regime", "Bear Market Plunge -30%") to visualize how drift alerts trigger.
- **Prediction Ledger**: Filterable table of all predictions stored in SQLite with real-time yfinance price backfill trigger.
- **Retraining Control Center**: Live prerequisite verification and manual pipeline trigger.

---

## Fully Automated MLOps Lifecycle

The monitoring and retraining lifecycle is automated across daily and monthly pipelines:

```
                    ┌────────────────────────────┐
                    │      Live Traffic          │
                    │  GET /api/predict (FastAPI)│
                    └─────────────┬──────────────┘
                                  │ Persist predictions
                                  ▼
                    ┌────────────────────────────┐
                    │ monitoring/predictions.db  │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │ Daily at 02:00 UTC                              │ Monthly / Drift Alert
         ▼                                                 ▼
┌─────────────────────────────────┐               ┌──────────────────────────────────┐
│ .github/workflows/              │               │ .github/workflows/               │
│   monitoring.yml                │               │   retrain.yml                    │
├─────────────────────────────────┤               ├──────────────────────────────────┤
│ 1. Collect actual prices (yf)   │               │ 1. Load latest 5-year OHLCV data │
│ 2. Evaluate rolling errors      │               │ 2. Compute 26/33 technical feats │
│ 3. Compute PSI & KS drift       │  PSI > 0.20   │ 3. Train MultiStockLSTM models   │
│ 4. Alert if PSI > 0.20          ├──────────────►│ 4. Validate on out-of-sample set │
└─────────────────────────────────┘  DirAcc < 50% │ 5. Update model_metadata.json    │
                                                  │ 6. Package model artifacts       │
                                                  └──────────────────────────────────┘
```

### 1. Automated Daily Monitoring (`.github/workflows/monitoring.yml`)
- **Trigger**: Runs automatically every day at 02:00 UTC (after US & Indian markets close) or via manual dispatch.
- **Execution**:
  1. Runs `monitoring/collect_predictions.py` to backfill real market closing prices.
  2. Runs `monitoring/evaluate_predictions.py` to compute 30-day rolling MAE, RMSE, and Directional Accuracy.
  3. Runs `monitoring/drift_report.py` to check PSI and KS divergence.
  4. If PSI exceeds the alert threshold (`PSI > 0.20`), raises an alert and triggers automated retraining.

### 2. Automated Retraining Pipeline (`.github/workflows/retrain.yml`)
- **Trigger**:
  - **Scheduled**: Runs automatically on the 1st of every month at midnight UTC.
  - **Drift-Triggered**: Triggered automatically when `monitoring.yml` detects significant distribution shift.
  - **Manual**: Triggered on-demand with custom market (`both`, `US`, `IN`) and epoch settings.
- **Validation Gates**: New candidate models are only promoted if they beat champion Test MAE and Directional Accuracy exceeds 50.0%.

### 3. Monitoring REST Endpoints

The API also exposes live monitoring endpoints:
- `GET /api/monitoring/drift?reference_days=30&recent_days=7&market=US` — live PSI, KS test, and quantiles.
- `GET /api/monitoring/performance?window=30` — rolling MAE, RMSE, MAPE, and directional accuracy.
- `POST /api/monitoring/backfill` — triggers actual market close price backfill in the background.

---

## MLflow Integration (Optional)

The project is prepared for MLflow model tracking without requiring it for the API to function.

To enable:
```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

The API works identically whether or not MLflow is configured.

---

## Environment Variables

See [`.env.example`](.env.example) for all configurable values:

```bash
cp .env.example .env
# Fill in any overrides — all have sensible defaults
```

---

## How the Model Works

1. **Data Ingestion** — Fetches OHLCV via yfinance for the last 2 years
2. **Feature Engineering** — 26 (US) or 33 (India) technical features per day
3. **Scaling** — `RobustScaler` per feature sequence (India: saved scalers from training)
4. **Inference** — Sliding 30-day window → LSTM → scalar log-return prediction
5. **Forecasting** — Autoregressive 5-day forecast: each prediction feeds back as synthetic OHLCV
6. **Output** — Denormalized price predictions + KPIs

---

## Disclaimer

This project is for **educational and research purposes only**. Stock price prediction is inherently uncertain. Do not use these predictions for actual trading decisions without rigorous backtesting and professional financial advice.

---

## License

Open-source. Feel free to use, modify, and distribute. Attribution is appreciated.
