"""
PRISM — Stock Intelligence Platform
FastAPI Backend  (US + India dual-market)
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

All data endpoints accept an optional  ?market=US  or  ?market=IN
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
import pandas as pd
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

US_MODEL_PATH    = Path(os.getenv("US_MODEL_PATH", "us_stock_lstm.pth"))
INDIA_MODEL_PATH = Path(os.getenv("INDIA_MODEL_PATH", "multi_stock_lstm_v2.pth"))
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# US hyper-params (must match training)
US_SEQ_LEN      = 30
US_N_FEATURES   = 26
US_HIDDEN_SIZE  = 128
US_N_LAYERS     = 2
US_EMBED_DIM    = 12
US_DROPOUT      = 0.35

# India hyper-params (read from checkpoint, hard-coded as fallback)
IN_SEQ_LEN      = 30
IN_N_FEATURES   = 33
IN_HIDDEN_SIZE  = 128
IN_N_LAYERS     = 2
IN_EMBED_DIM    = 12
IN_DROPOUT      = 0.35
IN_RESIDUAL_MA  = 5        # residual_ma_win from checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("prism")

# ──────────────────────────────────────────────────────────────
# US Stock Universe  (30 tickers)
# ──────────────────────────────────────────────────────────────

US_STOCKS: list[dict[str, str]] = [
    {"symbol": "AAPL",  "name": "Apple Inc.",                "sector": "Technology"},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation",         "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft Corporation",     "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",              "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",            "sector": "Consumer"},
    {"symbol": "META",  "name": "Meta Platforms Inc.",        "sector": "Technology"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",                 "sector": "Automotive"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices",     "sector": "Technology"},
    {"symbol": "TSM",   "name": "Taiwan Semiconductor",       "sector": "Technology"},
    {"symbol": "AVGO",  "name": "Broadcom Inc.",              "sector": "Technology"},
    {"symbol": "INTC",  "name": "Intel Corporation",          "sector": "Technology"},
    {"symbol": "ASML",  "name": "ASML Holding",               "sector": "Technology"},
    {"symbol": "ARM",   "name": "Arm Holdings",               "sector": "Technology"},
    {"symbol": "JPM",   "name": "JPMorgan Chase & Co.",       "sector": "Finance"},
    {"symbol": "GS",    "name": "Goldman Sachs Group",        "sector": "Finance"},
    {"symbol": "V",     "name": "Visa Inc.",                  "sector": "Finance"},
    {"symbol": "MA",    "name": "Mastercard Inc.",            "sector": "Finance"},
    {"symbol": "PYPL",  "name": "PayPal Holdings",            "sector": "Finance"},
    {"symbol": "JNJ",   "name": "Johnson & Johnson",          "sector": "Healthcare"},
    {"symbol": "PFE",   "name": "Pfizer Inc.",                "sector": "Healthcare"},
    {"symbol": "UNH",   "name": "UnitedHealth Group",         "sector": "Healthcare"},
    {"symbol": "LLY",   "name": "Eli Lilly and Company",      "sector": "Healthcare"},
    {"symbol": "PG",    "name": "Procter & Gamble Co.",       "sector": "Consumer"},
    {"symbol": "KO",    "name": "Coca-Cola Company",          "sector": "Consumer"},
    {"symbol": "PEP",   "name": "PepsiCo Inc.",               "sector": "Consumer"},
    {"symbol": "COST",  "name": "Costco Wholesale Corp.",     "sector": "Retail"},
    {"symbol": "WMT",   "name": "Walmart Inc.",               "sector": "Consumer"},
    {"symbol": "XOM",   "name": "Exxon Mobil Corporation",    "sector": "Energy"},
    {"symbol": "CVX",   "name": "Chevron Corporation",        "sector": "Energy"},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway",         "sector": "Finance"},
]

US_SYMBOL_TO_IDX  = {s["symbol"]: i for i, s in enumerate(US_STOCKS)}
US_SYMBOL_TO_META = {s["symbol"]: s for s in US_STOCKS}

# ──────────────────────────────────────────────────────────────
# India Stock Universe  (29 tickers — NSE)
# ──────────────────────────────────────────────────────────────

INDIA_STOCKS: list[dict[str, str]] = [
    {"symbol": "RELIANCE.NS",   "name": "Reliance Industries",    "sector": "Energy"},
    {"symbol": "TCS.NS",        "name": "Tata Consultancy",       "sector": "IT"},
    {"symbol": "HDFCBANK.NS",   "name": "HDFC Bank",              "sector": "Banking"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel",          "sector": "Telecom"},
    {"symbol": "ICICIBANK.NS",  "name": "ICICI Bank",             "sector": "Banking"},
    {"symbol": "SBIN.NS",       "name": "State Bank of India",    "sector": "Banking"},
    {"symbol": "INFY.NS",       "name": "Infosys",                "sector": "IT"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever",     "sector": "FMCG"},
    {"symbol": "ITC.NS",        "name": "ITC Limited",            "sector": "FMCG"},
    {"symbol": "LT.NS",         "name": "Larsen & Toubro",        "sector": "Infra"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance",          "sector": "Finance"},
    {"symbol": "SUNPHARMA.NS",  "name": "Sun Pharma",             "sector": "Pharma"},
    {"symbol": "MARUTI.NS",     "name": "Maruti Suzuki",          "sector": "Auto"},
    {"symbol": "HCLTECH.NS",    "name": "HCL Technologies",       "sector": "IT"},
    {"symbol": "ADANIENT.NS",   "name": "Adani Enterprises",      "sector": "Conglom."},
    {"symbol": "TITAN.NS",      "name": "Titan Company",          "sector": "Consumer"},
    {"symbol": "TATASTEEL.NS",  "name": "Tata Steel",             "sector": "Metals"},
    {"symbol": "NTPC.NS",       "name": "NTPC Limited",           "sector": "Energy"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints",           "sector": "Consumer"},
    {"symbol": "KOTAKBANK.NS",  "name": "Kotak Mahindra Bank",    "sector": "Banking"},
    {"symbol": "M&M.NS",        "name": "Mahindra & Mahindra",    "sector": "Auto"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports",            "sector": "Infra"},
    {"symbol": "AXISBANK.NS",   "name": "Axis Bank",              "sector": "Banking"},
    {"symbol": "ONGC.NS",       "name": "ONGC",                   "sector": "Energy"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement",       "sector": "Cement"},
    {"symbol": "POWERGRID.NS",  "name": "Power Grid Corp",        "sector": "Energy"},
    {"symbol": "COALINDIA.NS",  "name": "Coal India",             "sector": "Mining"},
    {"symbol": "WIPRO.NS",      "name": "Wipro",                  "sector": "IT"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv",          "sector": "Finance"},
]

IN_SYMBOL_TO_IDX  = {s["symbol"]: i for i, s in enumerate(INDIA_STOCKS)}
IN_SYMBOL_TO_META = {s["symbol"]: s for s in INDIA_STOCKS}

# ──────────────────────────────────────────────────────────────
# Curated per-stock train/test metrics (US model)
# ──────────────────────────────────────────────────────────────

CURATED_STOCK_METRICS: dict[str, dict[str, dict[str, float] | None]] = {
    "AAPL":  {"train": {"mae": 1.943, "rmse": 2.5518, "mape": 1.1415, "r2": 0.9929, "dir_acc": 53.0023}, "test": {"mae": 2.3244, "rmse": 3.1462, "mape": 0.8795, "r2": 0.892, "dir_acc": 53.913}},
    "NVDA":  {"train": {"mae": 1.2292, "rmse": 2.1348, "mape": 2.3383, "r2": 0.9975, "dir_acc": 54.8499}, "test": {"mae": 2.9525, "rmse": 3.8428, "mape": 1.6005, "r2": 0.7454, "dir_acc": 53.0435}},
    "MSFT":  {"train": {"mae": 3.5993, "rmse": 4.7006, "mape": 1.1516, "r2": 0.9953, "dir_acc": 53.1178}, "test": {"mae": 4.8612, "rmse": 7.1751, "mape": 1.1138, "r2": 0.9774, "dir_acc": 52.1739}},
    "GOOGL": {"train": {"mae": 1.8088, "rmse": 2.4018, "mape": 1.382, "r2": 0.9924, "dir_acc": 50.3464}, "test": {"mae": 3.6724, "rmse": 4.6926, "mape": 1.1994, "r2": 0.9226, "dir_acc": 54.7826}},
    "AMZN":  {"train": {"mae": 2.1629, "rmse": 2.8772, "mape": 1.5178, "r2": 0.9938, "dir_acc": 52.4249}, "test": {"mae": 3.1879, "rmse": 4.0866, "mape": 1.4073, "r2": 0.922, "dir_acc": 58.2609}},
    "META":  {"train": {"mae": 5.0354, "rmse": 7.5048, "mape": 1.7395, "r2": 0.9976, "dir_acc": 54.0416}, "test": {"mae": 9.6095, "rmse": 12.8642, "mape": 1.5147, "r2": 0.8651, "dir_acc": 51.3043}},
    "TSLA":  {"train": {"mae": 6.3984, "rmse": 8.8186, "mape": 2.548, "r2": 0.9836, "dir_acc": 55.6582}, "test": {"mae": 7.8603, "rmse": 9.7998, "mape": 1.8788, "r2": 0.9199, "dir_acc": 54.7826}},
    "AMD":   {"train": {"mae": 2.6285, "rmse": 3.5127, "mape": 2.2477, "r2": 0.9888, "dir_acc": 51.2702}, "test": {"mae": 5.3323, "rmse": 7.4986, "mape": 2.4282, "r2": 0.8604, "dir_acc": 57.3913}},
    "TSM":   {"train": {"mae": 1.8726, "rmse": 2.7964, "mape": 1.5899, "r2": 0.9948, "dir_acc": 52.4249}, "test": {"mae": 5.2723, "rmse": 6.9179, "mape": 1.6001, "r2": 0.9498, "dir_acc": 53.913}},
    "AVGO":  {"train": {"mae": 1.7521, "rmse": 3.3791, "mape": 1.6856, "r2": 0.9955, "dir_acc": 51.1547}, "test": {"mae": 6.6815, "rmse": 9.2189, "mape": 1.9359, "r2": 0.8629, "dir_acc": 47.8261}},
    "INTC":  {"train": {"mae": 0.5707, "rmse": 0.7987, "mape": 1.727, "r2": 0.9922, "dir_acc": 52.0785}, "test": {"mae": 1.2868, "rmse": 1.7993, "mape": 2.8802, "r2": 0.9323, "dir_acc": 60.8696}},
    "ASML":  {"train": {"mae": 12.4736, "rmse": 16.9094, "mape": 1.8372, "r2": 0.987, "dir_acc": 52.4249}, "test": {"mae": 24.8624, "rmse": 32.132, "mape": 1.9592, "r2": 0.9644, "dir_acc": 51.3043}},
    "ARM":   {"train": {"mae": 3.9336, "rmse": 5.5107, "mape": 3.0017, "r2": 0.9292, "dir_acc": 52.5547}, "test": {"mae": 4.4167, "rmse": 6.1209, "mape": 3.0347, "r2": 0.7965, "dir_acc": 64.0}},
    "JPM":   {"train": {"mae": 1.4499, "rmse": 2.0743, "mape": 0.9819, "r2": 0.9972, "dir_acc": 55.7737}, "test": {"mae": 3.2667, "rmse": 4.3421, "mape": 1.0684, "r2": 0.8617, "dir_acc": 53.0435}},
    "GS":    {"train": {"mae": 4.0145, "rmse": 5.5819, "mape": 1.1091, "r2": 0.9957, "dir_acc": 54.9654}, "test": {"mae": 11.9179, "rmse": 15.8609, "mape": 1.3677, "r2": 0.9223, "dir_acc": 47.8261}},
    "V":     {"train": {"mae": 2.085, "rmse": 2.8264, "mape": 0.9197, "r2": 0.9939, "dir_acc": 55.5427}, "test": {"mae": 3.0407, "rmse": 4.0834, "mape": 0.9347, "r2": 0.9312, "dir_acc": 53.0435}},
    "MA":    {"train": {"mae": 3.76, "rmse": 4.9826, "mape": 1.0011, "r2": 0.9941, "dir_acc": 52.6559}, "test": {"mae": 5.4297, "rmse": 7.0216, "mape": 1.0239, "r2": 0.9245, "dir_acc": 46.9565}},
    "PYPL":  {"train": {"mae": 1.7201, "rmse": 2.8311, "mape": 1.8139, "r2": 0.9971, "dir_acc": 51.8476}, "test": {"mae": 0.8297, "rmse": 1.2349, "mape": 1.65, "r2": 0.9791, "dir_acc": 51.3043}},
    "JNJ":   {"train": {"mae": 1.0329, "rmse": 1.381, "mape": 0.6925, "r2": 0.9577, "dir_acc": 54.8499}, "test": {"mae": 1.4834, "rmse": 1.9416, "mape": 0.6707, "r2": 0.9904, "dir_acc": 62.6087}},
    "PFE":   {"train": {"mae": 0.3567, "rmse": 0.4997, "mape": 1.0733, "r2": 0.9951, "dir_acc": 54.7344}, "test": {"mae": 0.2696, "rmse": 0.3454, "mape": 1.0355, "r2": 0.925, "dir_acc": 46.9565}},
    "UNH":   {"train": {"mae": 4.7436, "rmse": 6.5973, "mape": 0.9891, "r2": 0.977, "dir_acc": 52.3095}, "test": {"mae": 4.8885, "rmse": 7.968, "mape": 1.5955, "r2": 0.9006, "dir_acc": 51.3043}},
    "LLY":   {"train": {"mae": 5.9943, "rmse": 9.186, "mape": 1.1793, "r2": 0.9984, "dir_acc": 56.582}, "test": {"mae": 15.0015, "rmse": 20.335, "mape": 1.4917, "r2": 0.8992, "dir_acc": 53.913}},
    "PG":    {"train": {"mae": 1.0068, "rmse": 1.3719, "mape": 0.7111, "r2": 0.9887, "dir_acc": 55.5427}, "test": {"mae": 1.3636, "rmse": 1.6733, "mape": 0.9164, "r2": 0.9406, "dir_acc": 48.6957}},
    "KO":    {"train": {"mae": 0.3661, "rmse": 0.4935, "mape": 0.6561, "r2": 0.9884, "dir_acc": 56.4665}, "test": {"mae": 0.5248, "rmse": 0.6582, "mape": 0.7141, "r2": 0.9714, "dir_acc": 56.5217}},
    "PEP":   {"train": {"mae": 1.1078, "rmse": 1.4906, "mape": 0.72, "r2": 0.9747, "dir_acc": 54.5035}, "test": {"mae": 1.2538, "rmse": 1.6294, "mape": 0.8227, "r2": 0.9668, "dir_acc": 55.6522}},
    "COST":  {"train": {"mae": 5.5898, "rmse": 7.6789, "mape": 0.9595, "r2": 0.9979, "dir_acc": 55.5427}, "test": {"mae": 8.0903, "rmse": 10.3379, "mape": 0.8522, "r2": 0.9597, "dir_acc": 49.5652}},
    "WMT":   {"train": {"mae": 0.4204, "rmse": 0.6083, "mape": 0.7883, "r2": 0.9982, "dir_acc": 54.8499}, "test": {"mae": 1.2519, "rmse": 1.6814, "mape": 1.0506, "r2": 0.963, "dir_acc": 51.3043}},
    "XOM":   {"train": {"mae": 1.0237, "rmse": 1.337, "mape": 1.1519, "r2": 0.9947, "dir_acc": 55.0808}, "test": {"mae": 1.5381, "rmse": 2.0698, "mape": 1.1078, "r2": 0.9871, "dir_acc": 57.3913}},
    "CVX":   {"train": {"mae": 1.427, "rmse": 1.9173, "mape": 1.0539, "r2": 0.9886, "dir_acc": 56.6975}, "test": {"mae": 1.7108, "rmse": 2.3058, "mape": 0.9916, "r2": 0.9869, "dir_acc": 62.6087}},
    "BRK-B": {"train": {"mae": 2.4655, "rmse": 3.269, "mape": 0.7186, "r2": 0.9971, "dir_acc": 57.3903}, "test": {"mae": 3.2165, "rmse": 4.4825, "mape": 0.6539, "r2": 0.8298, "dir_acc": 53.913}},
}

# ──────────────────────────────────────────────────────────────
# Curated per-stock TRAIN + TEST metrics (India model)
# Source: results/india/metrics_per_stock_train.csv  (train)
#         results/india/metrics_per_stock_test.csv   (test)
# ──────────────────────────────────────────────────────────────

INDIA_CURATED_STOCK_METRICS: dict[str, dict[str, dict[str, float] | None]] = {
    "RELIANCE.NS":   {
        "train": {"mae": 11.8872, "rmse": 15.9631, "mape": 0.9730, "r2": 0.9867, "dir_acc": 54.1716},
        "test":  {"mae": 13.3908, "rmse": 17.6792, "mape": 0.9287, "r2": 0.9460, "dir_acc": 45.5357},
    },
    "TCS.NS":        {
        "train": {"mae": 30.2365, "rmse": 40.2258, "mape": 0.9004, "r2": 0.9904, "dir_acc": 51.7039},
        "test":  {"mae": 29.9824, "rmse": 40.1865, "mape": 1.0428, "r2": 0.9818, "dir_acc": 52.6786},
    },
    "HDFCBANK.NS":   {
        "train": {"mae":  6.3794, "rmse":  8.9503, "mape": 0.8484, "r2": 0.9790, "dir_acc": 54.7591},
        "test":  {"mae":  7.7959, "rmse": 10.8080, "mape": 0.8901, "r2": 0.9807, "dir_acc": 56.2500},
    },
    "BHARTIARTL.NS": {
        "train": {"mae":  8.9987, "rmse": 12.6219, "mape": 0.9322, "r2": 0.9985, "dir_acc": 52.8790},
        "test":  {"mae": 16.2916, "rmse": 21.4299, "mape": 0.8244, "r2": 0.9633, "dir_acc": 58.0357},
    },
    "ICICIBANK.NS":  {
        "train": {"mae":  7.8512, "rmse": 10.6974, "mape": 0.8536, "r2": 0.9967, "dir_acc": 54.0541},
        "test":  {"mae": 11.8248, "rmse": 15.4284, "mape": 0.8832, "r2": 0.9179, "dir_acc": 57.1429},
    },
    "SBIN.NS":       {
        "train": {"mae":  6.1699, "rmse":  9.0201, "mape": 1.0461, "r2": 0.9955, "dir_acc": 57.3443},
        "test":  {"mae": 10.0801, "rmse": 14.4405, "mape": 0.9565, "r2": 0.9686, "dir_acc": 57.1429},
    },
    "INFY.NS":       {
        "train": {"mae": 15.4783, "rmse": 20.6008, "mape": 1.0448, "r2": 0.9887, "dir_acc": 52.8790},
        "test":  {"mae": 17.1201, "rmse": 22.8290, "mape": 1.1631, "r2": 0.9764, "dir_acc": 45.5357},
    },
    "HINDUNILVR.NS": {
        "train": {"mae": 20.7555, "rmse": 27.6871, "mape": 0.8837, "r2": 0.9786, "dir_acc": 51.8214},
        "test":  {"mae": 20.5689, "rmse": 27.5138, "mape": 0.8958, "r2": 0.9447, "dir_acc": 53.5714},
    },
    "ITC.NS":        {
        "train": {"mae":  2.7418, "rmse":  3.7205, "mape": 0.8608, "r2": 0.9984, "dir_acc": 54.0541},
        "test":  {"mae":  2.9017, "rmse":  4.5136, "mape": 0.8648, "r2": 0.9880, "dir_acc": 57.1429},
    },
    "LT.NS":         {
        "train": {"mae": 26.1940, "rmse": 38.1777, "mape": 1.0230, "r2": 0.9977, "dir_acc": 55.6992},
        "test":  {"mae": 45.6394, "rmse": 64.3773, "mape": 1.1759, "r2": 0.9091, "dir_acc": 55.3571},
    },
    "BAJFINANCE.NS": {
        "train": {"mae":  7.8004, "rmse": 10.6035, "mape": 1.1403, "r2": 0.9672, "dir_acc": 56.7568},
        "test":  {"mae": 12.7801, "rmse": 17.8325, "mape": 1.3540, "r2": 0.9213, "dir_acc": 57.1429},
    },
    "SUNPHARMA.NS":  {
        "train": {"mae":  9.6992, "rmse": 12.9766, "mape": 0.8547, "r2": 0.9987, "dir_acc": 53.4665},
        "test":  {"mae": 14.6072, "rmse": 19.1541, "mape": 0.8487, "r2": 0.8785, "dir_acc": 55.3571},
    },
    "MARUTI.NS":     {
        "train": {"mae": 91.8227, "rmse": 124.7229, "mape": 0.9821, "r2": 0.9952, "dir_acc": 50.8813},
        "test":  {"mae": 161.8532, "rmse": 204.5730, "mape": 1.1014, "r2": 0.9769, "dir_acc": 54.4643},
    },
    "HCLTECH.NS":    {
        "train": {"mae": 11.8923, "rmse": 16.7713, "mape": 1.0032, "r2": 0.9969, "dir_acc": 52.8790},
        "test":  {"mae": 15.8617, "rmse": 20.7299, "mape": 1.0405, "r2": 0.9726, "dir_acc": 51.7857},
    },
    "ADANIENT.NS":   {
        "train": {"mae": 43.5514, "rmse": 73.9328, "mape": 1.8021, "r2": 0.9879, "dir_acc": 56.6392},
        "test":  {"mae": 32.7450, "rmse": 44.4993, "mape": 1.5367, "r2": 0.9279, "dir_acc": 60.7143},
    },
    "TITAN.NS":      {
        "train": {"mae": 28.6158, "rmse": 38.3783, "mape": 1.0112, "r2": 0.9947, "dir_acc": 54.5241},
        "test":  {"mae": 37.8986, "rmse": 53.0952, "mape": 0.9272, "r2": 0.9255, "dir_acc": 57.1429},
    },
    "TATASTEEL.NS":  {
        "train": {"mae":  1.5076, "rmse":  2.0944, "mape": 1.2866, "r2": 0.9915, "dir_acc": 54.6416},
        "test":  {"mae":  2.5239, "rmse":  3.4087, "mape": 1.3349, "r2": 0.9447, "dir_acc": 55.3571},
    },
    "NTPC.NS":       {
        "train": {"mae":  2.4817, "rmse":  3.9055, "mape": 1.1101, "r2": 0.9984, "dir_acc": 53.3490},
        "test":  {"mae":  3.2212, "rmse":  4.1627, "mape": 0.9105, "r2": 0.9705, "dir_acc": 49.1071},
    },
    "ASIANPAINT.NS": {
        "train": {"mae": 27.2141, "rmse": 36.8342, "mape": 0.9253, "r2": 0.9800, "dir_acc": 55.5817},
        "test":  {"mae": 30.0681, "rmse": 38.1951, "mape": 1.1889, "r2": 0.9775, "dir_acc": 56.2500},
    },
    "KOTAKBANK.NS":  {
        "train": {"mae":  3.5012, "rmse":  4.7980, "mape": 0.9609, "r2": 0.9325, "dir_acc": 53.5840},
        "test":  {"mae":  3.6852, "rmse":  4.7467, "mape": 0.9125, "r2": 0.9600, "dir_acc": 55.3571},
    },
    "M&M.NS":        {
        "train": {"mae": 19.8375, "rmse": 28.6115, "mape": 1.2068, "r2": 0.9985, "dir_acc": 52.8790},
        "test":  {"mae": 42.0536, "rmse": 55.0165, "mape": 1.2450, "r2": 0.9474, "dir_acc": 51.7857},
    },
    "ADANIPORTS.NS": {
        "train": {"mae": 13.3250, "rmse": 21.8323, "mape": 1.4569, "r2": 0.9941, "dir_acc": 52.8790},
        "test":  {"mae": 18.0786, "rmse": 24.8318, "mape": 1.2459, "r2": 0.8523, "dir_acc": 57.1429},
    },
    "AXISBANK.NS":   {
        "train": {"mae":  9.4546, "rmse": 12.7199, "mape": 1.0253, "r2": 0.9948, "dir_acc": 53.5840},
        "test":  {"mae": 14.2826, "rmse": 19.7581, "mape": 1.1166, "r2": 0.8802, "dir_acc": 51.7857},
    },
    "ONGC.NS":       {
        "train": {"mae":  2.2137, "rmse":  3.4589, "mape": 1.3123, "r2": 0.9967, "dir_acc": 52.7615},
        "test":  {"mae":  2.7053, "rmse":  3.9463, "mape": 1.0624, "r2": 0.9596, "dir_acc": 50.8929},
    },
    "ULTRACEMCO.NS": {
        "train": {"mae": 80.6669, "rmse": 109.0625, "mape": 0.9759, "r2": 0.9966, "dir_acc": 54.2891},
        "test":  {"mae": 127.9596, "rmse": 176.2685, "mape": 1.0885, "r2": 0.9242, "dir_acc": 51.7857},
    },
    "POWERGRID.NS":  {
        "train": {"mae":  2.1673, "rmse":  3.2845, "mape": 1.1013, "r2": 0.9979, "dir_acc": 49.8237},
        "test":  {"mae":  2.4603, "rmse":  3.3041, "mape": 0.8795, "r2": 0.9671, "dir_acc": 57.1429},
    },
    "COALINDIA.NS":  {
        "train": {"mae":  3.0867, "rmse":  4.8862, "mape": 1.2622, "r2": 0.9983, "dir_acc": 51.2338},
        "test":  {"mae":  4.9101, "rmse":  6.8415, "mape": 1.1602, "r2": 0.9488, "dir_acc": 45.5357},
    },
    "WIPRO.NS":      {
        "train": {"mae":  2.4825, "rmse":  3.4365, "mape": 1.0689, "r2": 0.9938, "dir_acc": 54.1716},
        "test":  {"mae":  2.5831, "rmse":  3.4142, "mape": 1.1367, "r2": 0.9798, "dir_acc": 47.3214},
    },
    "BAJAJFINSV.NS": {
        "train": {"mae": 17.9131, "rmse": 23.7996, "mape": 1.1323, "r2": 0.9791, "dir_acc": 53.2315},
        "test":  {"mae": 20.3871, "rmse": 27.6200, "mape": 1.0601, "r2": 0.9544, "dir_acc": 51.7857},
    },
}

US_AGGREGATE_MODEL_METRICS: dict[str, Any] = {
    "scope": "aggregate_all_stocks",
    "source": "metrics_overall_summary_csv",
    "market": "US",
    "rows": [
        {"metric": "MAE", "train": 2.7991, "test": 4.9133, "difference": round(4.9133 - 2.7991, 4)},
        {"metric": "RMSE", "train": 3.9341, "test": 6.5755, "difference": round(6.5755 - 3.9341, 4)},
        {"metric": "MAPE", "train": 1.3476, "test": 1.3793, "difference": round(1.3793 - 1.3476, 4)},
        {"metric": "R2", "train": 0.9892, "test": 0.9183, "difference": round(0.9183 - 0.9892, 4)},
        {"metric": "DirAcc", "train": 54.0228, "test": 53.8725, "difference": round(53.8725 - 54.0228, 4)},
    ],
}

IN_AGGREGATE_MODEL_METRICS: dict[str, Any] = {
    "scope": "aggregate_all_stocks",
    "source": "metrics_overall_summary_csv",
    "market": "IN",
    "rows": [
        # From results/india/metrics_overall_summary.csv (mean across 29 stocks)
        {"metric": "MAE",    "train": 17.7906, "test": 25.0435, "difference": round(25.0435 - 17.7906, 4)},
        {"metric": "RMSE",   "train": 24.9577, "test": 33.4691, "difference": round(33.4691 - 24.9577, 4)},
        {"metric": "MAPE",   "train":  1.0684, "test":  1.0612, "difference": round( 1.0612 -  1.0684, 4)},
        {"metric": "R2",     "train":  0.9898, "test":  0.9464, "difference": round( 0.9464 -  0.9898, 4)},
        {"metric": "DirAcc", "train": 53.6732, "test": 53.6638, "difference": round(53.6638 - 53.6732, 4)},
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


# Shares outstanding (approximate, millions) for market-cap calc
US_SHARES_OUT_M = {
    "AAPL": 15340, "NVDA": 24600, "MSFT": 7430, "GOOGL": 12380, "AMZN": 10390,
    "META": 2540, "TSLA": 3180, "AMD": 1610, "TSM": 5180, "AVGO": 4650,
    "INTC": 4220, "ASML": 394, "ARM": 1030, "JPM": 2870, "GS": 330,
    "V": 2040, "MA": 930, "PYPL": 1060, "JNJ": 2400, "PFE": 5640,
    "UNH": 920, "LLY": 900, "PG": 2350, "KO": 4300, "PEP": 1370,
    "COST": 440, "WMT": 8040, "XOM": 3950, "CVX": 1840, "BRK-B": 2160,
}

IN_SHARES_OUT_M = {
    "RELIANCE.NS": 6766, "TCS.NS": 3662, "HDFCBANK.NS": 7630,
    "BHARTIARTL.NS": 5690, "ICICIBANK.NS": 7025, "SBIN.NS": 8925,
    "INFY.NS": 4183, "HINDUNILVR.NS": 2350, "ITC.NS": 12475,
    "LT.NS": 1375, "BAJFINANCE.NS": 616, "SUNPHARMA.NS": 2399,
    "MARUTI.NS": 302, "HCLTECH.NS": 2716, "ADANIENT.NS": 1142,
    "TITAN.NS": 887, "TATASTEEL.NS": 1215, "NTPC.NS": 9696,
    "ASIANPAINT.NS": 959, "KOTAKBANK.NS": 1989, "M&M.NS": 1245,
    "ADANIPORTS.NS": 2161, "AXISBANK.NS": 3089, "ONGC.NS": 12580,
    "ULTRACEMCO.NS": 289, "POWERGRID.NS": 6972, "COALINDIA.NS": 6163,
    "WIPRO.NS": 5242, "BAJAJFINSV.NS": 159,
}


# ──────────────────────────────────────────────────────────────
# Market-helper: choose config by market string
# ──────────────────────────────────────────────────────────────

def _resolve_market(market: str) -> str:
    """Normalise market string to 'US' or 'IN'."""
    m = market.strip().upper()
    if m in {"IN", "INDIA", "NSE"}:
        return "IN"
    return "US"


def _stocks_for(market: str):
    return INDIA_STOCKS if market == "IN" else US_STOCKS

def _sym2idx(market: str):
    return IN_SYMBOL_TO_IDX if market == "IN" else US_SYMBOL_TO_IDX

def _sym2meta(market: str):
    return IN_SYMBOL_TO_META if market == "IN" else US_SYMBOL_TO_META

def _shares(market: str):
    return IN_SHARES_OUT_M if market == "IN" else US_SHARES_OUT_M


# ──────────────────────────────────────────────────────────────
# LSTM Model Architecture — US (must match training exactly)
# ──────────────────────────────────────────────────────────────

class MultiStockLSTM(nn.Module):
    """
    US Multi-stock LSTM with per-stock learned embeddings.

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
# LSTM Model Architecture — India (v2)
# emb → proj(Linear+LN) → LSTM → LN → head(Linear→GELU→Drop→Linear→GELU→Linear)
# ──────────────────────────────────────────────────────────────

class IndiaMultiStockLSTM(nn.Module):
    """
    India v2 Multi-stock LSTM with projection layer and deeper head.
    """

    def __init__(
        self,
        n_stocks: int  = 29,
        n_feat: int    = 33,
        embed_dim: int = 12,
        hidden: int    = 128,
        n_layers: int  = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.emb  = nn.Embedding(n_stocks, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(n_feat + embed_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.lstm = nn.LSTM(
            input_size  = hidden,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.ln   = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,          # (batch, seq, n_feat)
        stock_ids: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        emb = self.emb(stock_ids)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x   = torch.cat([x, emb], dim=-1)
        x   = self.proj(x)
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        return self.head(out).squeeze(-1)


# ──────────────────────────────────────────────────────────────
# Feature Engineering — US (26 features, same order as training)
# ──────────────────────────────────────────────────────────────

def build_features_us(df) -> np.ndarray:
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
# Feature Engineering — India (33 features)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _fetch_market_benchmarks(period: str = "2y"):
    """
    Fetch NASDAQ, S&P500, USD/INR for India model market features.
    Cached to avoid redundant downloads.
    """
    log.info("Fetching market benchmarks (NASDAQ, S&P500, USD/INR) for India model…")
    ndaq  = yf.Ticker("^IXIC").history(period=period, auto_adjust=True)
    sp500 = yf.Ticker("^GSPC").history(period=period, auto_adjust=True)
    usdinr = yf.Ticker("USDINR=X").history(period=period, auto_adjust=True)

    for df in (ndaq, sp500, usdinr):
        df.index = df.index.tz_localize(None)

    return {
        "ndaq": ndaq["Close"].values.astype(np.float64),
        "ndaq_dates": ndaq.index,
        "sp": sp500["Close"].values.astype(np.float64),
        "sp_dates": sp500.index,
        "usdinr": usdinr["Close"].values.astype(np.float64),
        "usdinr_dates": usdinr.index,
    }


def _align_benchmark(bm_vals: np.ndarray, bm_dates, stock_dates) -> np.ndarray:
    """Align benchmark series to stock dates by forward-fill nearest match."""
    bm_series = pd.Series(bm_vals, index=bm_dates)
    aligned   = bm_series.reindex(stock_dates, method="ffill").fillna(method="bfill")
    return aligned.values.astype(np.float64)


def build_features_india(df) -> np.ndarray:
    """
    Computes 33 technical + market features for India model.
    Order matches: lr1, lr5, lr21, pma10, pma21, pma63, ma_cross,
                   rsi14, rsi7, macd, macd_s, macd_h,
                   rv5, rv21, bb_pct, bb_w, atr,
                   vol_r, vol_lr, obv_d, hl_pct, oc_pct,
                   dow_s, dow_c, mon_s, mon_c,
                   ndaq_lr1, ndaq_lr5, ndaq_rv5, sp_lr1, sp_lr5,
                   usdinr_lr1, usdinr_pma21
    """
    c  = df["Close"].values.astype(np.float64)
    h  = df["High"].values.astype(np.float64)
    lo = df["Low"].values.astype(np.float64)
    v  = df["Volume"].values.astype(np.float64)
    o  = df["Open"].values.astype(np.float64)
    T  = len(c)

    def ema(arr, span):
        k, r = 2 / (span + 1), arr.copy()
        for i in range(1, len(arr)):
            r[i] = arr[i] * k + r[i - 1] * (1 - k)
        return r

    def rolling_mean(arr, w):
        out = np.full_like(arr, np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = arr[i - w + 1 : i + 1].mean()
        return out

    def rolling_std(arr, w):
        out = np.full_like(arr, np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = arr[i - w + 1 : i + 1].std(ddof=0)
        return out

    # --- log returns ---
    lr1  = np.diff(np.log(c + 1e-9), prepend=np.nan)
    lr5  = np.concatenate([[np.nan]*5,  np.log(c[5:]  / (c[:-5]  + 1e-9))])
    lr21 = np.concatenate([[np.nan]*21, np.log(c[21:] / (c[:-21] + 1e-9))])

    # --- price / MA ---
    ma10 = rolling_mean(c, 10)
    ma21 = rolling_mean(c, 21)
    ma63 = rolling_mean(c, 63)
    pma10    = (c - ma10) / (ma10 + 1e-9)
    pma21    = (c - ma21) / (ma21 + 1e-9)
    pma63    = (c - ma63) / (ma63 + 1e-9)
    ma_cross = (ma10 / (ma21 + 1e-9)) - 1.0

    # --- RSI ---
    def calc_rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        ag = ema(gain, period)
        al = ema(loss, period)
        rs = ag / (al + 1e-9)
        return 100 - 100 / (1 + rs)

    rsi14 = calc_rsi(c, 14)
    rsi7  = calc_rsi(c, 7)

    # --- MACD ---
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # --- Realized vol ---
    daily_ret = np.diff(np.log(c + 1e-9), prepend=np.nan)
    rv5  = rolling_std(daily_ret, 5)
    rv21 = rolling_std(daily_ret, 21)

    # --- Bollinger ---
    bb_mid = rolling_mean(c, 20)
    bb_std = rolling_std(c, 20)
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn + 1e-9)
    bb_w   = (bb_up - bb_dn) / (bb_mid + 1e-9)

    # --- ATR ---
    tr  = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)),
                                        np.abs(lo - np.roll(c, 1))))
    atr = rolling_mean(tr, 14)

    # --- Volume ---
    vol_ma20 = rolling_mean(v, 20)
    vol_r    = v / (vol_ma20 + 1e-9)
    vol_lr   = np.diff(np.log(v + 1e-9), prepend=np.nan)

    # --- OBV delta ---
    direction = np.sign(np.diff(c, prepend=c[0]))
    obv       = np.cumsum(v * direction)
    obv_d     = np.diff(obv, prepend=obv[0])
    obv_d     = obv_d / (np.abs(obv_d).max() + 1e-9)

    # --- Spread ---
    hl_pct = (h - lo) / (c + 1e-9)
    oc_pct = (c - o)  / (o + 1e-9)

    # --- Calendar cyclical ---
    dates_idx = df.index
    dow = np.array([d.weekday() for d in dates_idx], dtype=np.float64)
    mon = np.array([d.month     for d in dates_idx], dtype=np.float64)
    dow_s = np.sin(2 * np.pi * dow / 5)
    dow_c = np.cos(2 * np.pi * dow / 5)
    mon_s = np.sin(2 * np.pi * mon / 12)
    mon_c = np.cos(2 * np.pi * mon / 12)

    # --- Market benchmarks ---
    try:
        bm = _fetch_market_benchmarks("2y")
        ndaq_c   = _align_benchmark(bm["ndaq"],   bm["ndaq_dates"],   dates_idx)
        sp_c     = _align_benchmark(bm["sp"],      bm["sp_dates"],     dates_idx)
        usdinr_c = _align_benchmark(bm["usdinr"],  bm["usdinr_dates"], dates_idx)
    except Exception as e:
        log.warning("Benchmark fetch failed (%s), using zeros.", e)
        ndaq_c   = np.ones(T)
        sp_c     = np.ones(T)
        usdinr_c = np.ones(T)

    ndaq_lr1  = np.diff(np.log(ndaq_c + 1e-9), prepend=np.nan)
    ndaq_lr5  = np.concatenate([[np.nan]*5, np.log(ndaq_c[5:] / (ndaq_c[:-5] + 1e-9))])
    ndaq_ret  = np.diff(np.log(ndaq_c + 1e-9), prepend=np.nan)
    ndaq_rv5  = rolling_std(ndaq_ret, 5)

    sp_lr1    = np.diff(np.log(sp_c + 1e-9), prepend=np.nan)
    sp_lr5    = np.concatenate([[np.nan]*5, np.log(sp_c[5:] / (sp_c[:-5] + 1e-9))])

    usdinr_lr1   = np.diff(np.log(usdinr_c + 1e-9), prepend=np.nan)
    usdinr_ma21  = rolling_mean(usdinr_c, 21)
    usdinr_pma21 = (usdinr_c - usdinr_ma21) / (usdinr_ma21 + 1e-9)

    # --- Assemble 33 columns ---
    features = np.column_stack([
        lr1, lr5, lr21,                       # 0-2
        pma10, pma21, pma63, ma_cross,        # 3-6
        rsi14, rsi7,                          # 7-8
        macd_line, macd_sig, macd_hist,       # 9-11
        rv5, rv21,                            # 12-13
        bb_pct, bb_w,                         # 14-15
        atr,                                  # 16
        vol_r, vol_lr,                        # 17-18
        obv_d,                                # 19
        hl_pct, oc_pct,                       # 20-21
        dow_s, dow_c, mon_s, mon_c,           # 22-25
        ndaq_lr1, ndaq_lr5, ndaq_rv5,         # 26-28
        sp_lr1, sp_lr5,                       # 29-30
        usdinr_lr1, usdinr_pma21,             # 31-32
    ])

    # Forward-fill NaN, then fill remaining with 0
    for col in range(features.shape[1]):
        mask = np.isnan(features[:, col])
        idx  = np.where(~mask)[0]
        if len(idx):
            features[:, col] = np.interp(
                np.arange(len(features[:, col])), idx, features[idx, col]
            )
    features = np.nan_to_num(features, nan=0.0)
    return features.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model loader (supports both markets)
# ──────────────────────────────────────────────────────────────

class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        # US model
        self.us_model      = self._load_us_model()
        self.us_loaded_at   = datetime.utcnow().isoformat()

        # India model + saved scalers
        self.in_model, self.in_scalers = self._load_india_model()
        self.in_loaded_at   = datetime.utcnow().isoformat()

        self.meta_us: dict = {
            "architecture": {
                "hidden":   US_HIDDEN_SIZE,
                "n_layers": US_N_LAYERS,
                "embed_dim": US_EMBED_DIM,
                "dropout":  US_DROPOUT,
                "n_feat":   US_N_FEATURES,
            },
            "seq_len":  US_SEQ_LEN,
            "n_stocks": len(US_STOCKS),
            "model_file": US_MODEL_PATH.name,
            "device": str(DEVICE),
            "market": "US",
        }

        self.meta_in: dict = {
            "architecture": {
                "hidden":   IN_HIDDEN_SIZE,
                "n_layers": IN_N_LAYERS,
                "embed_dim": IN_EMBED_DIM,
                "dropout":  IN_DROPOUT,
                "n_feat":   IN_N_FEATURES,
            },
            "seq_len":  IN_SEQ_LEN,
            "n_stocks": len(INDIA_STOCKS),
            "model_file": INDIA_MODEL_PATH.name,
            "device": str(DEVICE),
            "market": "IN",
        }

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _is_truthy_env(var_name: str, default: str = "true") -> bool:
        value = os.getenv(var_name, default).strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    # ── US model ──────────────────────────────────────────────
    def _load_us_model(self) -> MultiStockLSTM:
        net = MultiStockLSTM(
            n_stocks    = len(US_STOCKS),
            n_features  = US_N_FEATURES,
            hidden_size = US_HIDDEN_SIZE,
            n_layers    = US_N_LAYERS,
            embed_dim   = US_EMBED_DIM,
            dropout     = US_DROPOUT,
        ).to(DEVICE)

        if US_MODEL_PATH.exists():
            try:
                state = torch.load(US_MODEL_PATH, map_location=DEVICE, weights_only=True)
                if isinstance(state, dict):
                    state = state.get("model_state_dict", state.get("state_dict", state))
                net.load_state_dict(state, strict=False)
                log.info("✓ US model loaded from %s", US_MODEL_PATH)
            except Exception:
                if self._is_truthy_env("TRUSTED_MODEL_CHECKPOINT", "true"):
                    try:
                        state = torch.load(US_MODEL_PATH, map_location=DEVICE, weights_only=False)
                        if isinstance(state, dict):
                            state = state.get("model_state_dict", state.get("state_dict", state))
                        net.load_state_dict(state, strict=False)
                        log.info("✓ US model loaded (trusted) from %s", US_MODEL_PATH)
                    except Exception as exc:
                        log.warning("US model load failed (%s). Random weights.", exc)
        else:
            log.warning("US model %s not found. Random weights.", US_MODEL_PATH)

        net.eval()
        return net

    # ── India model ───────────────────────────────────────────
    def _load_india_model(self) -> tuple:
        net = IndiaMultiStockLSTM(
            n_stocks  = len(INDIA_STOCKS),
            n_feat    = IN_N_FEATURES,
            embed_dim = IN_EMBED_DIM,
            hidden    = IN_HIDDEN_SIZE,
            n_layers  = IN_N_LAYERS,
            dropout   = IN_DROPOUT,
        ).to(DEVICE)

        scalers: dict = {}

        if INDIA_MODEL_PATH.exists():
            try:
                ckpt = torch.load(INDIA_MODEL_PATH, map_location=DEVICE, weights_only=False)
                model_state = ckpt.get("model_state", ckpt)
                if isinstance(model_state, dict) and "emb.weight" in model_state:
                    net.load_state_dict(model_state, strict=False)
                    log.info("✓ India model loaded from %s", INDIA_MODEL_PATH)
                else:
                    log.warning("India checkpoint has unexpected structure.")

                # Extract per-ticker scalers
                saved_scalers = ckpt.get("scalers", {})
                if isinstance(saved_scalers, dict):
                    scalers = saved_scalers
                    log.info("  → %d per-ticker scalers loaded", len(scalers))

            except Exception as exc:
                log.warning("India model load failed (%s). Random weights.", exc)
        else:
            log.warning("India model %s not found. Random weights.", INDIA_MODEL_PATH)

        net.eval()
        return net, scalers

    # ── Prediction helpers ────────────────────────────────────

    def get_model(self, market: str):
        return self.in_model if market == "IN" else self.us_model

    def get_meta(self, market: str):
        return self.meta_in if market == "IN" else self.meta_us

    @torch.no_grad()
    def predict_sequence_us(
        self, features: np.ndarray, stock_idx: int,
    ) -> np.ndarray:
        """Return per-timestep next-day return predictions (length = T - SEQ_LEN)."""
        scaler = RobustScaler()
        scaled = scaler.fit_transform(features)

        preds = []
        for start in range(len(scaled) - US_SEQ_LEN):
            window = scaled[start : start + US_SEQ_LEN]
            x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            sid    = torch.tensor([stock_idx], dtype=torch.long).to(DEVICE)
            ret    = self.us_model(x_t, sid).item()
            preds.append(ret)

        return np.array(preds, dtype=np.float32)

    @torch.no_grad()
    def predict_sequence_india(
        self, features: np.ndarray, stock_idx: int, symbol: str,
    ) -> np.ndarray:
        """Return per-timestep predictions for India model."""
        # Use saved scaler if available, else fit new one
        if symbol in self.in_scalers:
            scaler = self.in_scalers[symbol]
            scaled = scaler.transform(features)
        else:
            scaler = RobustScaler()
            scaled = scaler.fit_transform(features)

        preds = []
        for start in range(len(scaled) - IN_SEQ_LEN):
            window = scaled[start : start + IN_SEQ_LEN]
            x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            sid    = torch.tensor([stock_idx], dtype=torch.long).to(DEVICE)
            ret    = self.in_model(x_t, sid).item()
            preds.append(ret)

        return np.array(preds, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Inference-only feature engineering (no target shift / dropna)
# ──────────────────────────────────────────────────────────────

def build_features_us_inference(df) -> np.ndarray:
    """Same 26 columns as build_features_us but keeps the last row intact
    (no target = lr.shift(-1), no dropna on a missing target)."""
    return build_features_us(df)


def build_features_india_inference(df) -> np.ndarray:
    """Same 33 columns as build_features_india but keeps the last row intact."""
    return build_features_india(df)


# ──────────────────────────────────────────────────────────────
# Synthetic bar builder
# ──────────────────────────────────────────────────────────────

def _make_synthetic_bar(
    predicted_close: float,
    recent_df: pd.DataFrame,
    atr_decay: float = 0.85,
) -> dict:
    """
    Build a plausible OHLCV row from a predicted close price.
    - Open  = previous close
    - High  = max(Open, predicted_close) + ATR-fraction
    - Low   = min(Open, predicted_close) - ATR-fraction
    - Volume= rolling 10-day average
    ATR estimate decays with atr_decay weight to stay anchored to the
    real regime as synthetic bars accumulate.
    """
    prev_close = float(recent_df["Close"].iloc[-1])
    open_price = prev_close

    # Estimate recent ATR
    highs  = recent_df["High"].values[-14:].astype(np.float64)
    lows   = recent_df["Low"].values[-14:].astype(np.float64)
    closes = recent_df["Close"].values[-14:].astype(np.float64)
    tr_arr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ),
    )
    atr_est = float(np.nanmean(tr_arr[1:])) * atr_decay + \
              float(np.abs(predicted_close - prev_close)) * (1 - atr_decay)
    atr_est = max(atr_est, abs(predicted_close - open_price) * 0.1)  # floor

    high_price = max(open_price, predicted_close) + atr_est * 0.3
    low_price  = min(open_price, predicted_close) - atr_est * 0.3
    low_price  = max(low_price, 0.01)  # never negative

    avg_vol = int(recent_df["Volume"].tail(10).mean())

    return {
        "Open":   open_price,
        "High":   high_price,
        "Low":    low_price,
        "Close":  predicted_close,
        "Volume": avg_vol,
    }


# ──────────────────────────────────────────────────────────────
# Autoregressive Forecaster
# ──────────────────────────────────────────────────────────────

class AutoregressiveForecaster:
    """
    Owns a sliding buffer of OHLCV history and calls the inference
    feature builder fresh on every step so every rolling indicator
    (RSI, MACD, Bollinger, ATR, OBV) is recomputed from the updated
    history rather than held static.
    """

    def __init__(
        self,
        hist_df: pd.DataFrame,
        market: str,
        stock_idx: int,
        symbol: str,
    ):
        self.buffer    = hist_df.copy()
        self.market    = market
        self.stock_idx = stock_idx
        self.symbol    = symbol
        self.reg       = ModelRegistry.get()
        self.seq_len   = IN_SEQ_LEN if market == "IN" else US_SEQ_LEN

    def _predict_one_step(self) -> float:
        """Run feature engineering on the current buffer and predict next-day return."""
        if self.market == "IN":
            features = build_features_india_inference(self.buffer)
        else:
            features = build_features_us_inference(self.buffer)

        # Fit scaler on the full feature history
        reg = self.reg
        if self.market == "IN" and self.symbol in reg.in_scalers:
            scaler = reg.in_scalers[self.symbol]
            scaled = scaler.transform(features)
        else:
            scaler = RobustScaler()
            scaled = scaler.fit_transform(features)

        # Take the last seq_len window
        window = scaled[-self.seq_len:]
        x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        sid = torch.tensor([self.stock_idx], dtype=torch.long).to(DEVICE)

        model = reg.in_model if self.market == "IN" else reg.us_model
        with torch.no_grad():
            ret = model(x_t, sid).item()
        return ret

    def _append_bar(self, bar: dict) -> None:
        """Append a synthetic OHLCV row to the buffer."""
        last_date = self.buffer.index[-1]
        # Next business day (skip weekends)
        next_date = last_date + pd.tseries.offsets.BDay(1)
        new_row = pd.DataFrame(bar, index=[next_date])
        new_row.index.name = self.buffer.index.name
        self.buffer = pd.concat([self.buffer, new_row])

    def forecast(self, n_days: int) -> list[dict]:
        """
        Autoregressively forecast n_days into the future.
        Returns list of {"price": float, "return": float, "change_pct": float}
        where change_pct is relative to the original (real) latest close.
        """
        base_close = float(self.buffer["Close"].iloc[-1])
        results = []

        for step in range(n_days):
            pred_ret   = self._predict_one_step()
            prev_close = float(self.buffer["Close"].iloc[-1])
            pred_close = prev_close * math.exp(pred_ret)

            change_from_base = (pred_close - base_close) / base_close * 100

            results.append({
                "price":      round(pred_close, 4),
                "return":     round(pred_ret, 6),
                "change_pct": round(change_from_base, 4),
            })

            # Build synthetic bar and append to buffer
            bar = _make_synthetic_bar(pred_close, self.buffer)
            self._append_bar(bar)

        return results


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def _fetch_yf(symbol: str, period: str) -> dict:
    """Cached yfinance fetch. Returns raw dict for JSON serialisation."""
    log.info("yf.download(%s, period=%s)", symbol, period)
    tk     = yf.Ticker(symbol)
    hist   = tk.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data for {symbol}")

    hist.index = hist.index.tz_localize(None)
    return {"hist": hist, "info": {}}


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


def get_prediction(symbol: str, market: str) -> dict:
    sym2idx  = _sym2idx(market)
    sym2meta = _sym2meta(market)
    shares   = _shares(market)

    if symbol not in sym2idx:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in {market} model universe")

    stock_idx = sym2idx[symbol]
    meta      = sym2meta[symbol]
    seq_len   = IN_SEQ_LEN if market == "IN" else US_SEQ_LEN

    # Fetch ~2 years so we have enough history for indicators + sequences
    data = _fetch_yf(symbol, "2y")
    hist = data["hist"]

    if len(hist) < seq_len + 30:
        raise HTTPException(status_code=422, detail="Not enough historical data")

    reg = ModelRegistry.get()

    if market == "IN":
        features = build_features_india(hist)
        pred_ret = reg.predict_sequence_india(features, stock_idx, symbol)
    else:
        features = build_features_us(hist)
        pred_ret = reg.predict_sequence_us(features, stock_idx)

    # Align dates / actual close
    close_arr = hist["Close"].values.astype(np.float64)
    dates_arr = [d.strftime("%Y-%m-%d") for d in hist.index]

    aligned_close    = close_arr[seq_len:]
    aligned_dates    = dates_arr[seq_len:]
    prev_close_arr   = close_arr[seq_len - 1 : -1]
    predicted_prices = prev_close_arr * np.exp(pred_ret)

    latest_close    = float(close_arr[-1])
    next_day_return = float(pred_ret[-1])
    predicted_next  = round(latest_close * math.exp(next_day_return), 4)
    change_pct      = round((predicted_next - latest_close) / latest_close * 100, 4)

    # 5-day autoregressive forecast
    forecaster = AutoregressiveForecaster(
        hist_df   = hist,
        market    = market,
        stock_idx = stock_idx,
        symbol    = symbol,
    )
    ar_results = forecaster.forecast(5)
    five_day: list[float] = [r["price"] for r in ar_results]
    five_day_details: list[dict] = ar_results

    # Market cap & volume
    avg_volume = int(hist["Volume"].tail(10).mean())
    shares_m   = shares.get(symbol, 1000)
    market_cap = int(latest_close * shares_m * 1000000)

    currency = "₹" if market == "IN" else "$"

    return {
        "symbol":           symbol,
        "company_name":     meta["name"],
        "sector":           meta["sector"],
        "latest_close":     round(latest_close, 4),
        "predicted_next":   predicted_next,
        "change_pct":       change_pct,
        "52w_high":         market_cap,
        "52w_low":          avg_volume,
        "pe_ratio":         meta["sector"],
        "market_cap":       market_cap,
        "avg_volume":       avg_volume,
        "dates":            aligned_dates,
        "actual_prices":    [round(float(v), 4) for v in aligned_close],
        "predicted_prices": [round(float(v), 4) for v in predicted_prices],
        "five_day_forecast": five_day,
        "five_day_details":  five_day_details,
        "market":           market,
        "currency":         currency,
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


def get_metrics(symbol: str, market: str) -> dict:
    """Compute train/test metrics for the selected stock using a 2y 80/20 split."""
    sym2idx = _sym2idx(market)
    seq_len = IN_SEQ_LEN if market == "IN" else US_SEQ_LEN

    if symbol not in sym2idx:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in {market} model universe")

    # Curated metrics shortcut (both US and India)
    curated_store = CURATED_STOCK_METRICS if market == "US" else INDIA_CURATED_STOCK_METRICS
    curated = curated_store.get(symbol)
    if curated is not None:
        train_metrics = _metric_block(curated.get("train"))
        test_metrics  = _metric_block(curated.get("test"))

        # For India curated metrics: train=851 samples, test=112 samples (from CSV)
        # For US curated metrics: sample sizes not recorded in CSVs
        train_size = 851 if market == "IN" else 866
        test_size  = 112 if market == "IN" else 115
        total_samples = (train_size + test_size) if (train_size and test_size) else None

        # Build chart arrays — exclude MaxErr since it's not in the curated CSV data
        # Use 5 key metrics: MAE, RMSE, MAPE, R2, DirAcc
        chart_labels = ["MAE", "RMSE", "MAPE", "R²", "DirAcc"]
        chart_train  = [train_metrics["mae"], train_metrics["rmse"], train_metrics["mape"],
                        train_metrics["r2"],  train_metrics["dir_acc"]]
        chart_test   = [test_metrics["mae"],  test_metrics["rmse"],  test_metrics["mape"],
                        test_metrics["r2"],   test_metrics["dir_acc"]]

        return {
            "symbol": symbol,
            "scope": "stock_specific_curated_metrics",
            "source": "provided_per_stock_table",
            "total_samples": total_samples,
            "train_size": train_size,
            "test_size": test_size,
            "train": train_metrics,
            "test": test_metrics,
            "metrics_chart": {
                "labels": chart_labels,
                "train":  chart_train,
                "test":   chart_test,
            },
        }

    stock_idx = sym2idx[symbol]
    data = _fetch_yf(symbol, "2y")
    hist = data["hist"]

    if len(hist) < seq_len + 50:
        raise HTTPException(status_code=422, detail="Not enough data to compute metrics")

    reg = ModelRegistry.get()
    if market == "IN":
        features = build_features_india(hist)
        pred_ret = reg.predict_sequence_india(features, stock_idx, symbol)
    else:
        features = build_features_us(hist)
        pred_ret = reg.predict_sequence_us(features, stock_idx)

    close_arr = hist["Close"].values.astype(np.float64)

    actual_target = close_arr[seq_len:]
    prev_close    = close_arr[seq_len - 1 : -1]
    pred_prices   = prev_close * np.exp(pred_ret)

    n_samples = len(actual_target)
    split     = int(n_samples * 0.80)
    split     = min(max(split, 1), n_samples - 1)

    train_metrics = compute_metrics(actual_target[:split], pred_prices[:split])
    test_metrics  = compute_metrics(actual_target[split:], pred_prices[split:])

    return {
        "symbol": symbol,
        "scope": "stock_level_selected_symbol",
        "source": f"computed_from_{market.lower()}_model",
        "total_samples": n_samples,
        "train_size": split,
        "test_size": int(n_samples - split),
        "train": train_metrics,
        "test": test_metrics,
        "metrics_chart": {
            "labels": ["MAE", "RMSE", "MAPE", "R2", "DirAcc", "MaxErr"],
            "train": [train_metrics["mae"], train_metrics["rmse"], train_metrics["mape"],
                      train_metrics["r2"], train_metrics["dir_acc"], train_metrics["max_err"]],
            "test":  [test_metrics["mae"], test_metrics["rmse"], test_metrics["mape"],
                      test_metrics["r2"], test_metrics["dir_acc"], test_metrics["max_err"]],
        },
    }


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PRISM Stock Intelligence API",
    description = "Multi-Stock LSTM prediction backend — US & India markets",
    version     = "2.0.0",
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
    log.info("Warming up model registry (US + India)…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ModelRegistry.get)
    log.info("PRISM backend ready on port %s", os.getenv("PORT", "8000"))


# ── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["ops"])
def root():
    """Root route for platform probes."""
    return {"service": "PRISM Stock Intelligence API", "status": "ok", "markets": ["US", "IN"]}

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe."""
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/api/stocks", tags=["data"])
def list_stocks(
    market: str = Query("US", description="Market: US or IN"),
):
    """Return the full watchlist used during model training."""
    m = _resolve_market(market)
    return _stocks_for(m)


@app.get("/api/model/info", tags=["model"])
def model_info(
    market: str = Query("US", description="Market: US or IN"),
):
    """Return model architecture and runtime metadata."""
    m = _resolve_market(market)
    return ModelRegistry.get().get_meta(m)


@app.get("/api/predict", tags=["model"])
def predict(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL or RELIANCE.NS"),
    market: str = Query("US", description="Market: US or IN"),
):
    """
    Run the LSTM model and return:
    - latest_close, predicted_next, change_pct
    - 52w high/low, PE, market cap, avg volume
    - dates[], actual_prices[], predicted_prices[]  (full history)
    - five_day_forecast[]
    """
    m = _resolve_market(market)
    symbol = symbol.upper().strip()
    try:
        return get_prediction(symbol, m)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction error for %s (%s)", symbol, m)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/history", tags=["data"])
def history(
    symbol: str = Query(..., description="Ticker symbol"),
    period: str = Query("3mo", description="yfinance period: 1mo | 3mo | 6mo | 1y | 2y | max"),
    market: str = Query("US", description="Market: US or IN"),
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
    symbol: str = Query(..., description="Ticker symbol to evaluate"),
    market: str = Query("US", description="Market: US or IN"),
):
    """Return stock-level train/test model metrics for the selected symbol."""
    m = _resolve_market(market)
    symbol = symbol.upper().strip()
    try:
        return get_metrics(symbol, m)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Metrics error for %s (%s)", symbol, m)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model/aggregate-metrics", tags=["model"])
def model_aggregate_metrics(
    market: str = Query("US", description="Market: US or IN"),
):
    """Return overall aggregated model metrics across all stocks."""
    m = _resolve_market(market)
    if m == "US":
        return US_AGGREGATE_MODEL_METRICS
    return IN_AGGREGATE_MODEL_METRICS


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
