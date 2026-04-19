# PRISM — Multi-Stock Price Prediction Platform

A Deep Learning Project for Simultaneous Multi-Stock Forecasting using LSTM

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Data%20Source-4CAF50?style=for-the-badge)

---

## Overview

PRISM is a sophisticated stock intelligence platform that uses a multi-stock LSTM (Long Short-Term Memory) neural network to analyze historical price data and predict future price movements of multiple stocks simultaneously.

Instead of training separate models for each stock, PRISM learns temporal patterns and inter-stock correlations across a universe of 30 major U.S. equities. This shared representation allows the model to capture market-wide dynamics, sector influences, and cross-asset relationships, leading to more robust and context-aware forecasts.

The project combines:

- A powerful PyTorch LSTM model with embedding layers for stock identity
- A FastAPI backend exposing clean RESTful endpoints
- Real-time data fetching via yfinance
- A modern, responsive HTML frontend (single-file `index.html`) for interactive exploration

---

## Key Features

- **Multi-Stock LSTM Architecture** — One model predicts for 30 stocks by learning shared temporal features and per-stock embeddings
- **30-Day Historical Sequence Input** — Uses the last 30 trading days of engineered features
- **26 Input Features** per stock per day (OHLCV + technical indicators + volume ratios + returns)
- **5-Day Ahead Forecasting** — Predicts the next 5 trading days
- **Comprehensive Evaluation Metrics** — MAE, RMSE, MAPE, R², and Directional Accuracy
- **FastAPI Backend** with CORS support and health checks
- **Responsive Web UI** with charts, technical signals (RSI, MACD, Bollinger), and model transparency
- **Pre-trained Model Included** — `aapl_multi_stock_lstm.pth` (focused on AAPL but supports all 30 stocks)

---

## Model Architecture

```
Input:           (batch, seq_len=30, n_features=26)
Stock Embedding: 30 stocks → embedding_dim=12
LSTM:            2 layers, hidden_size=128, dropout=0.35
Output Head:     Fully Connected → 5-day price predictions
Device:          CUDA (if available) or CPU
```

The model concatenates time-series features with a learned stock-specific embedding to enable multi-stock learning.

---

## Stock Universe (30 Tickers)

| Sector | Tickers |
|---|---|
| Technology (12) | AAPL, MSFT, GOOGL, NVDA, META, AVGO, CSCO, ADBE, CRM, NFLX, INTC, AMD |
| Finance (5) | BRK-B, JPM, V, MA, BAC |
| Healthcare (4) | JNJ, MRK, ABBV, PFE |
| Consumer (5) | AMZN, WMT, PG, KO, PEP |
| Retail / Energy / Automotive (4) | HD, COST, CVX, TSLA |

---

## Project Structure

```
multi_stock_price_prediction/
├── main.py                   # FastAPI backend + model inference logic
├── aapl_multi_stock_lstm.pth # Pre-trained PyTorch model
├── index.html                # Interactive frontend (single-file web UI)
├── requirements (2).txt      # Python dependencies
├── results/                  # Folder for saving outputs/plots (created at runtime)
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yashvasudeva1/multi_stock_price_prediction.git
cd multi_stock_price_prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r "requirements (2).txt"
```

Dependencies:

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
torch>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
yfinance>=0.2.40
pydantic>=2.0.0
httpx>=0.27.0
```

---

## Running the Application

### Start the FastAPI Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://127.0.0.1:8000`.

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/stocks` | Return full watchlist with metadata |
| GET | `/api/model/info` | Model architecture and hyperparameters |
| GET | `/api/predict` | LSTM prediction + 5-day forecast + KPIs |
| GET | `/api/history` | OHLCV historical data |
| GET | `/api/model/metrics` | Per-stock train/test metrics |
| GET | `/api/model/aggregate-metrics` | Overall model performance |
| GET | `/health` | Health check |

Example prediction request:

```
GET /api/predict?symbol=AAPL&days=5
```

### Open the Web Interface

Open `index.html` in any modern browser. Basic viewing requires no server; full predictions require the FastAPI backend running.

---

## Model Performance

Per-stock evaluation metrics are available for several tickers. Example for AAPL:

| Split | MAE | RMSE | MAPE | R² | Directional Accuracy |
|---|---|---|---|---|---|
| Train | 2.1128 | 2.8409 | 1.2451% | 0.9911 | 53.00% |
| Test | 2.4469 | 3.4620 | 0.9260% | 0.8649 | 51.30% |

Detailed metrics for MSFT, NVDA, AMZN, GOOGL, META, and others are available in `main.py`.

---

## Configuration

Key hyperparameters (editable in `main.py`):

```python
SEQ_LEN    = 30
N_FEATURES = 26
HIDDEN_SIZE = 128
N_LAYERS   = 2
EMBED_DIM  = 12
DROPOUT    = 0.35
```

The model path can be overridden via environment variable:

```bash
export MODEL_PATH="path/to/your/model.pth"
```

---

## How the Model Works

1. **Data Ingestion** — Fetches OHLCV and volume data for all 30 stocks using yfinance
2. **Feature Engineering** — Creates 26 features including returns, volatility, technical indicators, and normalized volume
3. **Multi-Stock Input** — Sequences are shaped as `(stocks, seq_len, features)` with stock embeddings
4. **Training / Inference** — LSTM processes sequences and predicts next 5 days of closing prices
5. **Scaling** — Uses RobustScaler for robustness to outliers
6. **Output** — Denormalized price predictions and confidence-style KPIs

---

## Frontend Features

- Stock selector dropdown
- Current price and predicted price display
- 5-day forecast table
- Interactive price chart (1M / 3M / 6M / 1Y / MAX)
- Technical indicators panel (RSI, MACD, Bollinger Bands, Volume Ratio)
- Model transparency section
- Train/Test metrics and directional accuracy
- Responsive design

---

## Future Enhancements

- Add training script (`train.py`) for end-to-end retraining
- Incorporate sentiment and macroeconomic data
- Deploy to cloud (Vercel / Render / AWS)
- Add user authentication and watchlist persistence
- Implement attention mechanisms or Transformer backbone
- Real-time WebSocket updates

---

## Disclaimer

This project is for educational and research purposes only. Stock price prediction is inherently uncertain. Past performance does not guarantee future results. Do not use these predictions for actual trading decisions without rigorous backtesting and professional financial advice.

---

## License

This project is open-source. Feel free to use, modify, and distribute it. Attribution is appreciated.

---

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, and submit a pull request with a detailed description.
