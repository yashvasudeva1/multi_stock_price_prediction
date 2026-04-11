"""
PRISM — Stock Intelligence Platform
FastAPI Backend
============================================================
Endpoints
---------
GET  /api/stocks           → watchlist (symbol, name, sector)
GET  /api/model/info       → model architecture metadata
GET  /api/predict          → LSTM prediction + KPIs + 5-day forecast
GET  /api/history          → OHLCV history for a given period
GET  /api/model/metrics    → train/test split evaluation metrics
GET  /api/model/aggregate-metrics → overall aggregated model metrics
GET  /health               → liveness probe
============================================================
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.preprocessing import RobustScaler

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

MODEL_PATH   = Path(os.getenv("MODEL_PATH", "aapl_multi_stock_lstm.pth"))
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN      = 30
N_FEATURES   = 26
HIDDEN_SIZE  = 128
N_LAYERS     = 2
EMBED_DIM    = 12
DROPOUT      = 0.35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("prism")

# ──────────────────────────────────────────────────────────────
# Stock Universe  (30 tickers used during training)
# ──────────────────────────────────────────────────────────────

STOCKS: list[dict[str, str]] = [
    {"symbol": "AAPL",  "name": "Apple Inc.",                "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft Corporation",     "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",              "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",            "sector": "Consumer"},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation",         "sector": "Technology"},
    {"symbol": "META",  "name": "Meta Platforms Inc.",        "sector": "Technology"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",                 "sector": "Automotive"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway",         "sector": "Finance"},
    {"symbol": "JPM",   "name": "JPMorgan Chase & Co.",       "sector": "Finance"},
    {"symbol": "V",     "name": "Visa Inc.",                  "sector": "Finance"},
    {"symbol": "JNJ",   "name": "Johnson & Johnson",          "sector": "Healthcare"},
    {"symbol": "WMT",   "name": "Walmart Inc.",               "sector": "Consumer"},
    {"symbol": "PG",    "name": "Procter & Gamble Co.",       "sector": "Consumer"},
    {"symbol": "MA",    "name": "Mastercard Inc.",            "sector": "Finance"},
    {"symbol": "HD",    "name": "Home Depot Inc.",            "sector": "Retail"},
    {"symbol": "CVX",   "name": "Chevron Corporation",        "sector": "Energy"},
    {"symbol": "MRK",   "name": "Merck & Co. Inc.",           "sector": "Healthcare"},
    {"symbol": "ABBV",  "name": "AbbVie Inc.",                "sector": "Healthcare"},
    {"symbol": "PFE",   "name": "Pfizer Inc.",                "sector": "Healthcare"},
    {"symbol": "BAC",   "name": "Bank of America Corp.",      "sector": "Finance"},
    {"symbol": "KO",    "name": "Coca-Cola Company",          "sector": "Consumer"},
    {"symbol": "PEP",   "name": "PepsiCo Inc.",               "sector": "Consumer"},
    {"symbol": "COST",  "name": "Costco Wholesale Corp.",     "sector": "Retail"},
    {"symbol": "AVGO",  "name": "Broadcom Inc.",              "sector": "Technology"},
    {"symbol": "CSCO",  "name": "Cisco Systems Inc.",         "sector": "Technology"},
    {"symbol": "ADBE",  "name": "Adobe Inc.",                 "sector": "Technology"},
    {"symbol": "CRM",   "name": "Salesforce Inc.",            "sector": "Technology"},
    {"symbol": "NFLX",  "name": "Netflix Inc.",               "sector": "Media"},
    {"symbol": "INTC",  "name": "Intel Corporation",          "sector": "Technology"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices",     "sector": "Technology"},
]

# Build quick lookup maps
SYMBOL_TO_IDX  = {s["symbol"]: i for i, s in enumerate(STOCKS)}
SYMBOL_TO_META = {s["symbol"]: s for s in STOCKS}

# Curated per-stock train/test metrics provided by user.
CURATED_STOCK_METRICS: dict[str, dict[str, dict[str, float] | None]] = {
    "AAPL":  {"train": {"mae": 2.1128, "rmse": 2.8409, "mape": 1.2451, "r2": 0.9911, "dir_acc": 53.0023}, "test": {"mae": 2.4469, "rmse": 3.4620, "mape": 0.9260, "r2": 0.8649, "dir_acc": 51.3043}},
    "MSFT":  {"train": {"mae": 3.9405, "rmse": 5.2465, "mape": 1.2611, "r2": 0.9941, "dir_acc": 51.5012}, "test": {"mae": 5.4825, "rmse": 7.9643, "mape": 1.2388, "r2": 0.9741, "dir_acc": 43.4783}},
    "NVDA":  {"train": {"mae": 1.3436, "rmse": 2.3409, "mape": 2.5903, "r2": 0.9969, "dir_acc": 53.2333}, "test": {"mae": 3.3625, "rmse": 4.2357, "mape": 1.8234, "r2": 0.6723, "dir_acc": 52.1739}},
    "AMZN":  {"train": {"mae": 2.3596, "rmse": 3.2212, "mape": 1.6664, "r2": 0.9920, "dir_acc": 51.1547}, "test": {"mae": 3.5010, "rmse": 4.7469, "mape": 1.5560, "r2": 0.8820, "dir_acc": 55.6522}},
    "GOOGL": {"train": {"mae": 1.8988, "rmse": 2.5897, "mape": 1.4611, "r2": 0.9910, "dir_acc": 53.1178}, "test": {"mae": 4.2076, "rmse": 5.4138, "mape": 1.3898, "r2": 0.9140, "dir_acc": 50.4348}},
    "META":  {"train": {"mae": 5.4428, "rmse": 8.3642, "mape": 1.8867, "r2": 0.9969, "dir_acc": 52.1940}, "test": {"mae": 10.8139, "rmse": 16.2508, "mape": 1.6939, "r2": 0.8268, "dir_acc": 49.5652}},
    "TSLA":  {"train": {"mae": 6.9907, "rmse": 9.8656, "mape": 2.8038, "r2": 0.9789, "dir_acc": 52.3095}, "test": {"mae": 8.7707, "rmse": 10.9868, "mape": 2.0715, "r2": 0.8908, "dir_acc": 49.5652}},
    "BRK-B": {"train": {"mae": 2.7492, "rmse": 3.6508, "mape": 0.8026, "r2": 0.9964, "dir_acc": 53.0023}, "test": {"mae": 3.6128, "rmse": 4.9757, "mape": 0.7349, "r2": 0.7745, "dir_acc": 46.9565}},
    "JPM":   {"train": {"mae": 1.5948, "rmse": 2.3122, "mape": 1.0915, "r2": 0.9964, "dir_acc": 53.1178}, "test": {"mae": 3.5019, "rmse": 4.7498, "mape": 1.1476, "r2": 0.8324, "dir_acc": 53.9130}},
    "V":     {"train": {"mae": 2.2903, "rmse": 3.1665, "mape": 1.0124, "r2": 0.9919, "dir_acc": 54.2725}, "test": {"mae": 3.1392, "rmse": 4.5378, "mape": 0.9629, "r2": 0.9169, "dir_acc": 46.0870}},
    "UNH":   {"train": {"mae": 5.1661, "rmse": 7.3746, "mape": 1.0804, "r2": 0.9717, "dir_acc": 52.7714}, "test": {"mae": 5.3272, "rmse": 8.8183, "mape": 1.7415, "r2": 0.8940, "dir_acc": 46.9565}},
    "XOM":   {"train": {"mae": 1.1383, "rmse": 1.4909, "mape": 1.2896, "r2": 0.9936, "dir_acc": 52.8868}, "test": {"mae": 1.7511, "rmse": 2.3212, "mape": 1.2762, "r2": 0.9841, "dir_acc": 56.5217}},
    "LLY":   {"train": {"mae": 6.4487, "rmse": 10.2638, "mape": 1.2880, "r2": 0.9980, "dir_acc": 53.5797}, "test": {"mae": 16.6246, "rmse": 22.9732, "mape": 1.6568, "r2": 0.9011, "dir_acc": 51.3043}},
    "JNJ":   {"train": {"mae": 1.1352, "rmse": 1.5419, "mape": 0.7603, "r2": 0.9473, "dir_acc": 49.4226}, "test": {"mae": 1.7300, "rmse": 2.2414, "mape": 0.7923, "r2": 0.9881, "dir_acc": 53.9130}},
    "MA":    {"train": {"mae": 4.0502, "rmse": 5.5452, "mape": 1.0854, "r2": 0.9924, "dir_acc": 53.2333}, "test": {"mae": 5.7397, "rmse": 7.7904, "mape": 1.0815, "r2": 0.9096, "dir_acc": 46.0870}},
    "PG":    {"train": {"mae": 1.1194, "rmse": 1.5426, "mape": 0.7918, "r2": 0.9856, "dir_acc": 53.5797}, "test": {"mae": 1.4801, "rmse": 1.8566, "mape": 0.9933, "r2": 0.9254, "dir_acc": 46.9565}},
    "HD":    {"train": {"mae": 3.4956, "rmse": 4.6836, "mape": 1.1373, "r2": 0.9873, "dir_acc": 52.8868}, "test": {"mae": 4.4666, "rmse": 5.7911, "mape": 1.2606, "r2": 0.9106, "dir_acc": 49.5652}},
    "MRK":   {"train": {"mae": 0.8713, "rmse": 1.2598, "mape": 0.9416, "r2": 0.9943, "dir_acc": 52.0785}, "test": {"mae": 1.2845, "rmse": 1.7052, "mape": 1.2100, "r2": 0.9793, "dir_acc": 54.7826}},
    "AVGO":  {"train": {"mae": 1.7886, "rmse": 3.5031, "mape": 1.7839, "r2": 0.9950, "dir_acc": 52.4249}, "test": {"mae": 7.3340, "rmse": 10.1847, "mape": 2.1291, "r2": 0.8050, "dir_acc": 52.1739}},
    "CVX":   {"train": {"mae": 1.5797, "rmse": 2.1689, "mape": 1.1711, "r2": 0.9861, "dir_acc": 54.8499}, "test": {"mae": 1.8731, "rmse": 2.5664, "mape": 1.0985, "r2": 0.9838, "dir_acc": 55.6522}},
    "ABBV":  {"train": {"mae": 1.3574, "rmse": 2.0486, "mape": 0.9688, "r2": 0.9923, "dir_acc": 54.3880}, "test": {"mae": 2.7281, "rmse": 3.6880, "mape": 1.2321, "r2": 0.7522, "dir_acc": 48.6957}},
    "COST":  {"train": {"mae": 6.0795, "rmse": 8.5494, "mape": 1.0453, "r2": 0.9973, "dir_acc": 54.7344}, "test": {"mae": 8.8522, "rmse": 11.4343, "mape": 0.9360, "r2": 0.9485, "dir_acc": 47.8261}},
    "PEP":   {"train": {"mae": 1.2012, "rmse": 1.6421, "mape": 0.7801, "r2": 0.9696, "dir_acc": 51.6166}, "test": {"mae": 1.5009, "rmse": 1.9367, "mape": 0.9883, "r2": 0.9517, "dir_acc": 48.6957}},
    "KO":    {"train": {"mae": 0.4013, "rmse": 0.5510, "mape": 0.7195, "r2": 0.9857, "dir_acc": 53.3487}, "test": {"mae": 0.5865, "rmse": 0.7514, "mape": 0.8003, "r2": 0.9632, "dir_acc": 52.1739}},
    "BAC":   {"train": {"mae": 0.4234, "rmse": 0.5764, "mape": 1.2578, "r2": 0.9900, "dir_acc": 48.9607}, "test": {"mae": 0.5651, "rmse": 0.7444, "mape": 1.0888, "r2": 0.9126, "dir_acc": 57.3913}},
    "WMT":   {"train": {"mae": 0.4602, "rmse": 0.6757, "mape": 0.8634, "r2": 0.9976, "dir_acc": 55.5427}, "test": {"mae": 1.4022, "rmse": 1.8685, "mape": 1.1858, "r2": 0.9562, "dir_acc": 52.1739}},
    "ADBE":  {"train": {"mae": 7.7277, "rmse": 11.3746, "mape": 1.6549, "r2": 0.9870, "dir_acc": 49.8845}, "test": {"mae": 4.5228, "rmse": 6.4123, "mape": 1.5445, "r2": 0.9746, "dir_acc": 46.9565}},
    "CRM":   {"train": {"mae": 3.5600, "rmse": 5.2903, "mape": 1.6433, "r2": 0.9907, "dir_acc": 51.2702}, "test": {"mae": 3.8145, "rmse": 5.2145, "mape": 1.7746, "r2": 0.9700, "dir_acc": 46.0870}},
    "MCD":   {"train": {"mae": 1.9795, "rmse": 2.6592, "mape": 0.7949, "r2": 0.9862, "dir_acc": 51.9630}, "test": {"mae": 2.6221, "rmse": 3.3595, "mape": 0.8427, "r2": 0.9029, "dir_acc": 47.8261}},
    "NFLX":  {"train": {"mae": 0.7955, "rmse": 1.2220, "mape": 1.9114, "r2": 0.9960, "dir_acc": 50.5774}, "test": None},
}

AGGREGATE_MODEL_METRICS: dict[str, Any] = {
    "scope": "aggregate_all_stocks",
    "source": "provided_aggregate_summary",
    "rows": [
        {"metric": "MAE", "train": 2.716726, "test": 4.148920, "difference": 1.432194},
        {"metric": "RMSE", "train": 3.918751, "test": 5.701946, "difference": 1.783196},
        {"metric": "MAPE", "train": 1.292997, "test": 1.289340, "difference": -0.003657},
        {"metric": "R2", "train": 0.988981, "test": 0.904052, "difference": -0.084929},
        {"metric": "DirAcc", "train": 52.563510, "test": 50.289855, "difference": -2.273655},
    ],
}


def _metric_block(raw: dict[str, float] | None) -> dict[str, float | None]:
    raw = raw or {}
    return {
        "mae": raw.get("mae"),
        "rmse": raw.get("rmse"),
        "mape": raw.get("mape"),
        "r2": raw.get("r2"),
        "dir_acc": raw.get("dir_acc"),
        "max_err": raw.get("max_err"),
        "n": raw.get("n"),
    }

# ──────────────────────────────────────────────────────────────
# LSTM Model Architecture  (must match training exactly)
# ──────────────────────────────────────────────────────────────

class MultiStockLSTM(nn.Module):
    """
    Multi-stock LSTM with per-stock learned embeddings.

    Input per timestep: [n_features]  →  concat with embed  →  LSTM  →  FC
    Output: scalar next-day log-return
    """

    def __init__(
        self,
        n_stocks: int,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed      = nn.Embedding(n_stocks, embed_dim)
        self.lstm       = nn.LSTM(
            input_size   = n_features + embed_dim,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,          # (batch, seq, n_features)
        stock_ids: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        emb = self.embed(stock_ids)                        # (batch, embed_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq, embed_dim)
        x   = torch.cat([x, emb], dim=-1)                 # (batch, seq, n_features+embed_dim)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])                  # last timestep
        return self.fc(out).squeeze(-1)                    # (batch,)


# ──────────────────────────────────────────────────────────────
# Feature Engineering  (26 features, same order as training)
# ──────────────────────────────────────────────────────────────

def build_features(df) -> np.ndarray:
    """
    Computes 26 technical features from a raw OHLCV DataFrame.
    Returns an array of shape (T, 26).  Rows with NaN are forward-filled.
    """
    c  = df["Close"].values.astype(np.float64)
    h  = df["High"].values.astype(np.float64)
    lo = df["Low"].values.astype(np.float64)
    v  = df["Volume"].values.astype(np.float64)
    o  = df["Open"].values.astype(np.float64)

    def ema(arr: np.ndarray, span: int) -> np.ndarray:
        k, result = 2 / (span + 1), arr.copy()
        for i in range(1, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = arr[i - w + 1 : i + 1].mean()
        return out

    def rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = arr[i - w + 1 : i + 1].std(ddof=0)
        return out

    # Returns & momentum
    ret1  = np.diff(np.log(c + 1e-9), prepend=np.nan)  # log return 1d
    ret5  = np.concatenate([[np.nan] * 5,  np.log(c[5:]  / (c[:-5]  + 1e-9))])
    ret10 = np.concatenate([[np.nan] * 10, np.log(c[10:] / (c[:-10] + 1e-9))])
    ret20 = np.concatenate([[np.nan] * 20, np.log(c[20:] / (c[:-20] + 1e-9))])

    # EMAs
    ema12, ema26 = ema(c, 12), ema(c, 26)
    macd_line  = ema12 - ema26
    macd_sig   = ema(macd_line, 9)
    macd_hist  = macd_line - macd_sig

    # Bollinger Bands (20, 2σ)
    bb_mid = rolling_mean(c, 20)
    bb_std = rolling_std(c, 20)
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn + 1e-9)

    # RSI (14)
    delta   = np.diff(c, prepend=c[0])
    gain    = np.where(delta > 0, delta, 0.0)
    loss    = np.where(delta < 0, -delta, 0.0)
    avg_g   = ema(gain, 14)
    avg_l   = ema(loss, 14)
    rs      = avg_g / (avg_l + 1e-9)
    rsi     = 100 - 100 / (1 + rs)

    # ATR (14)
    tr = np.maximum(h - lo,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(lo - np.roll(c, 1))))
    atr = rolling_mean(tr, 14)

    # OBV
    direction = np.sign(np.diff(c, prepend=c[0]))
    obv       = np.cumsum(v * direction)
    obv_norm  = obv / (np.abs(obv).max() + 1e-9)  # normalize

    # Volume features
    vol_ma20  = rolling_mean(v, 20)
    vol_ratio = v / (vol_ma20 + 1e-9)
    log_vol   = np.log1p(v)

    # Price vs moving averages
    sma20  = rolling_mean(c, 20)
    sma50  = rolling_mean(c, 50)
    dist20 = (c - sma20) / (sma20 + 1e-9)
    dist50 = (c - sma50) / (sma50 + 1e-9)

    # HL spread & open-close spread
    hl_spread = (h - lo) / (c + 1e-9)
    oc_spread = (c - o) / (o + 1e-9)

    # Stochastic %K (14)
    lo14  = np.array([lo[max(0, i-13):i+1].min() for i in range(len(lo))])
    hi14  = np.array([h[max(0,  i-13):i+1].max() for i in range(len(h))])
    stoch = (c - lo14) / (hi14 - lo14 + 1e-9) * 100

    # Assemble — 26 columns
    features = np.column_stack([
        ret1,        # 0  log-return 1d
        ret5,        # 1  log-return 5d
        ret10,       # 2  log-return 10d
        ret20,       # 3  log-return 20d
        macd_line,   # 4
        macd_sig,    # 5
        macd_hist,   # 6
        bb_pct,      # 7  bollinger %B
        rsi,         # 8
        atr,         # 9
        obv_norm,    # 10
        vol_ratio,   # 11
        log_vol,     # 12
        dist20,      # 13 price/SMA20 - 1
        dist50,      # 14 price/SMA50 - 1
        hl_spread,   # 15
        oc_spread,   # 16
        stoch,       # 17 stochastic %K
        c,           # 18 raw close (scaled later)
        h,           # 19
        lo,          # 20
        o,           # 21
        v,           # 22
        ema12,       # 23
        ema26,       # 24
        bb_mid,      # 25
    ])

    # Forward-fill NaN, then fill remaining with 0
    for col in range(features.shape[1]):
        mask = np.isnan(features[:, col])
        idx  = np.where(~mask)[0]
        if len(idx):
            features[:, col] = np.interp(
                np.arange(len(features[:, col])),
                idx,
                features[idx, col],
            )
    features = np.nan_to_num(features, nan=0.0)
    return features.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model loader (singleton)
# ──────────────────────────────────────────────────────────────

class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self.model      = self._load_model()
        self.loaded_at  = datetime.utcnow().isoformat()
        self.meta: dict = {
            "architecture": {
                "hidden":   HIDDEN_SIZE,
                "n_layers": N_LAYERS,
                "embed_dim": EMBED_DIM,
                "dropout":  DROPOUT,
                "n_feat":   N_FEATURES,
            },
            "seq_len":  SEQ_LEN,
            "n_stocks": len(STOCKS),
            "model_file": MODEL_PATH.name,
            "device": str(DEVICE),
        }

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> MultiStockLSTM:
        net = MultiStockLSTM(
            n_stocks    = len(STOCKS),
            n_features  = N_FEATURES,
            hidden_size = HIDDEN_SIZE,
            n_layers    = N_LAYERS,
            embed_dim   = EMBED_DIM,
            dropout     = DROPOUT,
        ).to(DEVICE)

        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
                # unwrap common checkpoint wrappers
                if isinstance(state, dict):
                    state = state.get("model_state_dict", state.get("state_dict", state))
                net.load_state_dict(state, strict=False)
                log.info("✓ Model loaded from %s on %s", MODEL_PATH, DEVICE)
            except Exception as exc:
                log.warning("Could not load weights (%s). Running with random weights.", exc)
        else:
            log.warning("Model file %s not found. Running with random weights.", MODEL_PATH)

        net.eval()
        return net

    @torch.no_grad()
    def predict_sequence(
        self,
        features: np.ndarray,   # (T, N_FEATURES)
        stock_idx: int,
    ) -> np.ndarray:
        """Return per-timestep next-day return predictions (length = T - SEQ_LEN)."""
        scaler = RobustScaler()
        scaled = scaler.fit_transform(features)

        preds  = []
        for start in range(len(scaled) - SEQ_LEN):
            window = scaled[start : start + SEQ_LEN]                  # (SEQ_LEN, N_FEATURES)
            x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            sid    = torch.tensor([stock_idx], dtype=torch.long).to(DEVICE)
            ret    = self.model(x_t, sid).item()
            preds.append(ret)

        return np.array(preds, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _fetch_yf(symbol: str, period: str) -> dict:
    """Cached yfinance fetch. Returns raw dict for JSON serialisation."""
    log.info("yf.download(%s, period=%s)", symbol, period)
    tk     = yf.Ticker(symbol)
    hist   = tk.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data for {symbol}")
    info   = {}
    try:
        info = tk.info or {}
    except Exception:
        pass
    hist.index = hist.index.tz_localize(None)
    return {"hist": hist, "info": info}


def get_history(symbol: str, period: str) -> dict:
    data = _fetch_yf(symbol, period)
    hist = data["hist"]
    return {
        "symbol":  symbol,
        "period":  period,
        "dates":   [d.strftime("%Y-%m-%d") for d in hist.index],
        "open":    [round(float(v), 4) for v in hist["Open"]],
        "high":    [round(float(v), 4) for v in hist["High"]],
        "low":     [round(float(v), 4) for v in hist["Low"]],
        "close":   [round(float(v), 4) for v in hist["Close"]],
        "volume":  [int(v) for v in hist["Volume"]],
    }


def _safe(v: Any, decimals: int = 2) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), decimals)


def get_prediction(symbol: str) -> dict:
    if symbol not in SYMBOL_TO_IDX:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in model universe")

    stock_idx = SYMBOL_TO_IDX[symbol]
    meta      = SYMBOL_TO_META[symbol]

    # Fetch ~2 years so we have enough history for indicators + sequences
    data = _fetch_yf(symbol, "2y")
    hist = data["hist"]
    info = data["info"]

    if len(hist) < SEQ_LEN + 30:
        raise HTTPException(status_code=422, detail="Not enough historical data")

    features = build_features(hist)          # (T, 26)
    reg      = ModelRegistry.get()
    pred_ret = reg.predict_sequence(features, stock_idx)  # (T - SEQ_LEN,)

    # Align dates / actual close
    close_arr = hist["Close"].values.astype(np.float64)
    dates_arr = [d.strftime("%Y-%m-%d") for d in hist.index]

    # Align model outputs to true target day:
    # each pred_ret[t] predicts return from day (t+SEQ_LEN-1) -> (t+SEQ_LEN).
    aligned_close    = close_arr[SEQ_LEN:]            # actual target close at day t+1
    aligned_dates    = dates_arr[SEQ_LEN:]            # date of actual target close
    prev_close_arr   = close_arr[SEQ_LEN - 1 : -1]    # base close at day t
    predicted_prices = prev_close_arr * np.exp(pred_ret)

    latest_close    = float(close_arr[-1])
    next_day_return = float(pred_ret[-1])
    predicted_next  = round(latest_close * math.exp(next_day_return), 4)
    change_pct      = round((predicted_next - latest_close) / latest_close * 100, 4)

    # 5-day forecast (iterative, simple version)
    last_close = latest_close
    five_day: list[float] = []
    for _ in range(5):
        last_close = round(last_close * math.exp(next_day_return * 0.85), 4)  # slight mean-reversion
        five_day.append(last_close)

    return {
        "symbol":           symbol,
        "company_name":     info.get("longName", meta["name"]),
        "sector":           meta["sector"],
        "latest_close":     round(latest_close, 4),
        "predicted_next":   predicted_next,
        "change_pct":       change_pct,
        "52w_high":         _safe(info.get("fiftyTwoWeekHigh")),
        "52w_low":          _safe(info.get("fiftyTwoWeekLow")),
        "pe_ratio":         _safe(info.get("trailingPE"), 1),
        "market_cap":       info.get("marketCap"),
        "avg_volume":       info.get("averageVolume"),
        "dates":            aligned_dates,
        "actual_prices":    [round(float(v), 4) for v in aligned_close],
        "predicted_prices": [round(float(v), 4) for v in predicted_prices],
        "five_day_forecast": five_day,
    }


# ──────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Regression + directional accuracy metrics between two price arrays."""
    actual    = np.array(actual,    dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)

    mae      = float(np.mean(np.abs(actual - predicted)))
    rmse     = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape     = float(np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100)
    ss_res   = float(np.sum((actual - predicted) ** 2))
    ss_tot   = float(np.sum((actual - actual.mean()) ** 2))
    r2       = 1.0 - ss_res / (ss_tot + 1e-9)
    act_dir  = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    dir_acc  = float(np.mean(act_dir == pred_dir) * 100)
    max_err  = float(np.max(np.abs(actual - predicted)))

    return {
        "mae":     round(mae,     4),
        "rmse":    round(rmse,    4),
        "mape":    round(mape,    4),
        "r2":      round(r2,      4),
        "dir_acc": round(dir_acc, 2),
        "max_err": round(max_err, 4),
        "n":       int(len(actual)),
    }


def get_metrics(symbol: str) -> dict:
    """Compute train/test metrics for the selected stock using a 2y 80/20 split."""
    if symbol not in SYMBOL_TO_IDX:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in model universe")

    curated = CURATED_STOCK_METRICS.get(symbol)
    if curated is not None:
        train_metrics = _metric_block(curated.get("train"))
        test_metrics = _metric_block(curated.get("test"))
        return {
            "symbol": symbol,
            "scope": "stock_specific_curated_metrics",
            "source": "provided_per_stock_table",
            "total_samples": None,
            "train_size": None,
            "test_size": None,
            "train": train_metrics,
            "test": test_metrics,
            "metrics_chart": {
                "labels": ["MAE", "RMSE", "MAPE", "R2", "DirAcc", "MaxErr"],
                "train": [
                    train_metrics["mae"],
                    train_metrics["rmse"],
                    train_metrics["mape"],
                    train_metrics["r2"],
                    train_metrics["dir_acc"],
                    train_metrics["max_err"],
                ],
                "test": [
                    test_metrics["mae"],
                    test_metrics["rmse"],
                    test_metrics["mape"],
                    test_metrics["r2"],
                    test_metrics["dir_acc"],
                    test_metrics["max_err"],
                ],
            },
        }

    stock_idx = SYMBOL_TO_IDX[symbol]
    data = _fetch_yf(symbol, "2y")
    hist = data["hist"]

    if len(hist) < SEQ_LEN + 50:
        raise HTTPException(status_code=422, detail="Not enough data to compute metrics")

    features = build_features(hist)
    reg = ModelRegistry.get()
    pred_ret = reg.predict_sequence(features, stock_idx)

    close_arr = hist["Close"].values.astype(np.float64)

    # For each predicted return, predict day t+1 close from day t close.
    actual_target = close_arr[SEQ_LEN:]
    prev_close = close_arr[SEQ_LEN - 1 : -1]
    pred_prices = prev_close * np.exp(pred_ret)

    n_samples = len(actual_target)
    split = int(n_samples * 0.80)
    split = min(max(split, 1), n_samples - 1)

    train_metrics = compute_metrics(actual_target[:split], pred_prices[:split])
    test_metrics = compute_metrics(actual_target[split:], pred_prices[split:])

    return {
        "symbol": symbol,
        "scope": "stock_level_selected_symbol",
        "source": "computed_from_current_model",
        "total_samples": n_samples,
        "train_size": split,
        "test_size": int(n_samples - split),
        "train": train_metrics,
        "test": test_metrics,
        "metrics_chart": {
            "labels": ["MAE", "RMSE", "MAPE", "R2", "DirAcc", "MaxErr"],
            "train": [
                train_metrics["mae"],
                train_metrics["rmse"],
                train_metrics["mape"],
                train_metrics["r2"],
                train_metrics["dir_acc"],
                train_metrics["max_err"],
            ],
            "test": [
                test_metrics["mae"],
                test_metrics["rmse"],
                test_metrics["mape"],
                test_metrics["r2"],
                test_metrics["dir_acc"],
                test_metrics["max_err"],
            ],
        },
    }


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PRISM Stock Intelligence API",
    description = "Multi-Stock LSTM prediction backend for the PRISM frontend",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Startup ───────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    log.info("Warming up model registry…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ModelRegistry.get)
    log.info("PRISM backend ready on port 5050")


# ── Routes ────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe."""
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/api/stocks", tags=["data"])
def list_stocks():
    """Return the full watchlist used during model training."""
    return STOCKS


@app.get("/api/model/info", tags=["model"])
def model_info():
    """Return model architecture and runtime metadata."""
    return ModelRegistry.get().meta


@app.get("/api/predict", tags=["model"])
def predict(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"),
):
    """
    Run the LSTM model and return:
    - latest_close, predicted_next, change_pct
    - 52w high/low, PE, market cap, avg volume
    - dates[], actual_prices[], predicted_prices[]  (full history)
    - five_day_forecast[]
    """
    symbol = symbol.upper().strip()
    try:
        return get_prediction(symbol)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction error for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history", tags=["data"])
def history(
    symbol: str = Query(..., description="Ticker symbol"),
    period: str = Query("3mo", description="yfinance period: 1mo | 3mo | 6mo | 1y | 2y | max"),
):
    """
    Return raw OHLCV history for chart rendering and signal computation.
    """
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "max"}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"period must be one of {valid_periods}")

    symbol = symbol.upper().strip()
    try:
        return get_history(symbol, period)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("History error for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model/metrics", tags=["model"])
def model_metrics(
    symbol: str = Query(..., description="Ticker symbol to evaluate, e.g. AAPL"),
):
    """Return stock-level train/test model metrics for the selected symbol."""
    symbol = symbol.upper().strip()
    try:
        return get_metrics(symbol)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Metrics error for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model/aggregate-metrics", tags=["model"])
def model_aggregate_metrics():
    """Return overall aggregated model metrics across all stocks."""
    return AGGREGATE_MODEL_METRICS


# ──────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 5050,
        reload  = True,
        workers = 1,
    )
