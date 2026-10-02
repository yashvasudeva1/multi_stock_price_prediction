"""
PRISM — Stock Intelligence Platform
FastAPI entry-point  (US + India dual-market LSTM)
=============================================================================
Endpoints
---------
GET  /health                      → liveness probe
GET  /ready                       → readiness probe (models loaded?)
GET  /metrics                     → Prometheus metrics
GET  /api/stocks                  → watchlist (symbol, name, sector)
GET  /api/model/info              → model architecture + metadata
GET  /api/predict                 → LSTM prediction + KPIs + 5-day forecast
GET  /api/history                 → OHLCV history
GET  /api/model/metrics           → per-stock train/test evaluation metrics
GET  /api/model/aggregate-metrics → overall aggregated metrics

All data endpoints accept ?market=US (default) or ?market=IN
=============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model_registry import VERSION, ModelRegistry
from app.monitoring import (
    MODEL_INFERENCE_LATENCY,
    PREDICTION_ERRORS,
    PREDICTION_REQUESTS,
    get_metrics_output,
    log_prediction_event,
    store_prediction,
)
from app.predictor import (
    AutoregressiveForecaster,
    _fetch_yf,
    build_features_india,
    build_features_us,
    get_history,
    predict_sequence_india,
    predict_sequence_us,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("prism")

# ── Stock universes ───────────────────────────────────────────────────────────

US_STOCKS: list[dict[str, str]] = [
    {"symbol": "AAPL",  "name": "Apple Inc.",                "sector": "Technology"},
    {"symbol": "NVDA",  "name": "NVIDIA Corporation",         "sector": "Technology"},
    {"symbol": "MSFT",  "name": "Microsoft Corporation",     "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.",             "sector": "Technology"},
    {"symbol": "AMZN",  "name": "Amazon.com Inc.",            "sector": "Consumer"},
    {"symbol": "META",  "name": "Meta Platforms Inc.",        "sector": "Technology"},
    {"symbol": "TSLA",  "name": "Tesla Inc.",                 "sector": "Automotive"},
    {"symbol": "AMD",   "name": "Advanced Micro Devices",     "sector": "Technology"},
    {"symbol": "TSM",   "name": "Taiwan Semiconductor",       "sector": "Technology"},
    {"symbol": "AVGO",  "name": "Broadcom Inc.",              "sector": "Technology"},
    {"symbol": "INTC",  "name": "Intel Corporation",          "sector": "Technology"},
    {"symbol": "ASML",  "name": "ASML Holding",               "sector": "Technology"},
    {"symbol": "ARM",   "name": "Arm Holdings",               "sector": "Technology"},
    {"symbol": "JPM",   "name": "JPMorgan Chase & Co.",      "sector": "Finance"},
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

US_SYMBOL_TO_IDX  = {s["symbol"]: i for i, s in enumerate(US_STOCKS)}
US_SYMBOL_TO_META = {s["symbol"]: s for s in US_STOCKS}
IN_SYMBOL_TO_IDX  = {s["symbol"]: i for i, s in enumerate(INDIA_STOCKS)}
IN_SYMBOL_TO_META = {s["symbol"]: s for s in INDIA_STOCKS}

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

# ── Curated per-stock metrics (from results/ CSVs) ────────────────────────────

CURATED_STOCK_METRICS: dict[str, dict] = {
    "AAPL":  {"train": {"mae": 1.943,  "rmse": 2.5518, "mape": 1.1415, "r2": 0.9929, "dir_acc": 53.0023}, "test": {"mae": 2.3244, "rmse": 3.1462, "mape": 0.8795, "r2": 0.892,  "dir_acc": 53.913}},
    "NVDA":  {"train": {"mae": 1.2292, "rmse": 2.1348, "mape": 2.3383, "r2": 0.9975, "dir_acc": 54.8499}, "test": {"mae": 2.9525, "rmse": 3.8428, "mape": 1.6005, "r2": 0.7454, "dir_acc": 53.0435}},
    "MSFT":  {"train": {"mae": 3.5993, "rmse": 4.7006, "mape": 1.1516, "r2": 0.9953, "dir_acc": 53.1178}, "test": {"mae": 4.8612, "rmse": 7.1751, "mape": 1.1138, "r2": 0.9774, "dir_acc": 52.1739}},
    "GOOGL": {"train": {"mae": 1.8088, "rmse": 2.4018, "mape": 1.382,  "r2": 0.9924, "dir_acc": 50.3464}, "test": {"mae": 3.6724, "rmse": 4.6926, "mape": 1.1994, "r2": 0.9226, "dir_acc": 54.7826}},
    "AMZN":  {"train": {"mae": 2.1629, "rmse": 2.8772, "mape": 1.5178, "r2": 0.9938, "dir_acc": 52.4249}, "test": {"mae": 3.1879, "rmse": 4.0866, "mape": 1.4073, "r2": 0.922,  "dir_acc": 58.2609}},
    "META":  {"train": {"mae": 5.0354, "rmse": 7.5048, "mape": 1.7395, "r2": 0.9976, "dir_acc": 54.0416}, "test": {"mae": 9.6095, "rmse": 12.8642,"mape": 1.5147, "r2": 0.8651, "dir_acc": 51.3043}},
    "TSLA":  {"train": {"mae": 6.3984, "rmse": 8.8186, "mape": 2.548,  "r2": 0.9836, "dir_acc": 55.6582}, "test": {"mae": 7.8603, "rmse": 9.7998, "mape": 1.8788, "r2": 0.9199, "dir_acc": 54.7826}},
    "AMD":   {"train": {"mae": 2.6285, "rmse": 3.5127, "mape": 2.2477, "r2": 0.9888, "dir_acc": 51.2702}, "test": {"mae": 5.3323, "rmse": 7.4986, "mape": 2.4282, "r2": 0.8604, "dir_acc": 57.3913}},
    "TSM":   {"train": {"mae": 1.8726, "rmse": 2.7964, "mape": 1.5899, "r2": 0.9948, "dir_acc": 52.4249}, "test": {"mae": 5.2723, "rmse": 6.9179, "mape": 1.6001, "r2": 0.9498, "dir_acc": 53.913}},
    "AVGO":  {"train": {"mae": 1.7521, "rmse": 3.3791, "mape": 1.6856, "r2": 0.9955, "dir_acc": 51.1547}, "test": {"mae": 6.6815, "rmse": 9.2189, "mape": 1.9359, "r2": 0.8629, "dir_acc": 47.8261}},
    "INTC":  {"train": {"mae": 0.5707, "rmse": 0.7987, "mape": 1.727,  "r2": 0.9922, "dir_acc": 52.0785}, "test": {"mae": 1.2868, "rmse": 1.7993, "mape": 2.8802, "r2": 0.9323, "dir_acc": 60.8696}},
    "ASML":  {"train": {"mae": 12.4736,"rmse": 16.9094,"mape": 1.8372, "r2": 0.987,  "dir_acc": 52.4249}, "test": {"mae": 24.8624,"rmse": 32.132, "mape": 1.9592, "r2": 0.9644, "dir_acc": 51.3043}},
    "ARM":   {"train": {"mae": 3.9336, "rmse": 5.5107, "mape": 3.0017, "r2": 0.9292, "dir_acc": 52.5547}, "test": {"mae": 4.4167, "rmse": 6.1209, "mape": 3.0347, "r2": 0.7965, "dir_acc": 64.0}},
    "JPM":   {"train": {"mae": 1.4499, "rmse": 2.0743, "mape": 0.9819, "r2": 0.9972, "dir_acc": 55.7737}, "test": {"mae": 3.2667, "rmse": 4.3421, "mape": 1.0684, "r2": 0.8617, "dir_acc": 53.0435}},
    "GS":    {"train": {"mae": 4.0145, "rmse": 5.5819, "mape": 1.1091, "r2": 0.9957, "dir_acc": 54.9654}, "test": {"mae": 11.9179,"rmse": 15.8609,"mape": 1.3677, "r2": 0.9223, "dir_acc": 47.8261}},
    "V":     {"train": {"mae": 2.085,  "rmse": 2.8264, "mape": 0.9197, "r2": 0.9939, "dir_acc": 55.5427}, "test": {"mae": 3.0407, "rmse": 4.0834, "mape": 0.9347, "r2": 0.9312, "dir_acc": 53.0435}},
    "MA":    {"train": {"mae": 3.76,   "rmse": 4.9826, "mape": 1.0011, "r2": 0.9941, "dir_acc": 52.6559}, "test": {"mae": 5.4297, "rmse": 7.0216, "mape": 1.0239, "r2": 0.9245, "dir_acc": 46.9565}},
    "PYPL":  {"train": {"mae": 1.7201, "rmse": 2.8311, "mape": 1.8139, "r2": 0.9971, "dir_acc": 51.8476}, "test": {"mae": 0.8297, "rmse": 1.2349, "mape": 1.65,   "r2": 0.9791, "dir_acc": 51.3043}},
    "JNJ":   {"train": {"mae": 1.0329, "rmse": 1.381,  "mape": 0.6925, "r2": 0.9577, "dir_acc": 54.8499}, "test": {"mae": 1.4834, "rmse": 1.9416, "mape": 0.6707, "r2": 0.9904, "dir_acc": 62.6087}},
    "PFE":   {"train": {"mae": 0.3567, "rmse": 0.4997, "mape": 1.0733, "r2": 0.9951, "dir_acc": 54.7344}, "test": {"mae": 0.2696, "rmse": 0.3454, "mape": 1.0355, "r2": 0.925,  "dir_acc": 46.9565}},
    "UNH":   {"train": {"mae": 4.7436, "rmse": 6.5973, "mape": 0.9891, "r2": 0.977,  "dir_acc": 52.3095}, "test": {"mae": 4.8885, "rmse": 7.968,  "mape": 1.5955, "r2": 0.9006, "dir_acc": 51.3043}},
    "LLY":   {"train": {"mae": 5.9943, "rmse": 9.186,  "mape": 1.1793, "r2": 0.9984, "dir_acc": 56.582},  "test": {"mae": 15.0015,"rmse": 20.335, "mape": 1.4917, "r2": 0.8992, "dir_acc": 53.913}},
    "PG":    {"train": {"mae": 1.0068, "rmse": 1.3719, "mape": 0.7111, "r2": 0.9887, "dir_acc": 55.5427}, "test": {"mae": 1.3636, "rmse": 1.6733, "mape": 0.9164, "r2": 0.9406, "dir_acc": 48.6957}},
    "KO":    {"train": {"mae": 0.3661, "rmse": 0.4935, "mape": 0.6561, "r2": 0.9884, "dir_acc": 56.4665}, "test": {"mae": 0.5248, "rmse": 0.6582, "mape": 0.7141, "r2": 0.9714, "dir_acc": 56.5217}},
    "PEP":   {"train": {"mae": 1.1078, "rmse": 1.4906, "mape": 0.72,   "r2": 0.9747, "dir_acc": 54.5035}, "test": {"mae": 1.2538, "rmse": 1.6294, "mape": 0.8227, "r2": 0.9668, "dir_acc": 55.6522}},
    "COST":  {"train": {"mae": 5.5898, "rmse": 7.6789, "mape": 0.9595, "r2": 0.9979, "dir_acc": 55.5427}, "test": {"mae": 8.0903, "rmse": 10.3379,"mape": 0.8522, "r2": 0.9597, "dir_acc": 49.5652}},
    "WMT":   {"train": {"mae": 0.4204, "rmse": 0.6083, "mape": 0.7883, "r2": 0.9982, "dir_acc": 54.8499}, "test": {"mae": 1.2519, "rmse": 1.6814, "mape": 1.0506, "r2": 0.963,  "dir_acc": 51.3043}},
    "XOM":   {"train": {"mae": 1.0237, "rmse": 1.337,  "mape": 1.1519, "r2": 0.9947, "dir_acc": 55.0808}, "test": {"mae": 1.5381, "rmse": 2.0698, "mape": 1.1078, "r2": 0.9871, "dir_acc": 57.3913}},
    "CVX":   {"train": {"mae": 1.427,  "rmse": 1.9173, "mape": 1.0539, "r2": 0.9886, "dir_acc": 56.6975}, "test": {"mae": 1.7108, "rmse": 2.3058, "mape": 0.9916, "r2": 0.9869, "dir_acc": 62.6087}},
    "BRK-B": {"train": {"mae": 2.4655, "rmse": 3.269,  "mape": 0.7186, "r2": 0.9971, "dir_acc": 57.3903}, "test": {"mae": 3.2165, "rmse": 4.4825, "mape": 0.6539, "r2": 0.8298, "dir_acc": 53.913}},
}

INDIA_CURATED_STOCK_METRICS: dict[str, dict] = {
    "RELIANCE.NS":   {"train": {"mae": 11.8872, "rmse": 15.9631, "mape": 0.9730, "r2": 0.9867, "dir_acc": 54.1716}, "test": {"mae": 13.3908, "rmse": 17.6792, "mape": 0.9287, "r2": 0.9460, "dir_acc": 45.5357}},
    "TCS.NS":        {"train": {"mae": 30.2365, "rmse": 40.2258, "mape": 0.9004, "r2": 0.9904, "dir_acc": 51.7039}, "test": {"mae": 29.9824, "rmse": 40.1865, "mape": 1.0428, "r2": 0.9818, "dir_acc": 52.6786}},
    "HDFCBANK.NS":   {"train": {"mae":  6.3794, "rmse":  8.9503, "mape": 0.8484, "r2": 0.9790, "dir_acc": 54.7591}, "test": {"mae":  7.7959, "rmse": 10.8080, "mape": 0.8901, "r2": 0.9807, "dir_acc": 56.2500}},
    "BHARTIARTL.NS": {"train": {"mae":  8.9987, "rmse": 12.6219, "mape": 0.9322, "r2": 0.9985, "dir_acc": 52.8790}, "test": {"mae": 16.2916, "rmse": 21.4299, "mape": 0.8244, "r2": 0.9633, "dir_acc": 58.0357}},
    "ICICIBANK.NS":  {"train": {"mae":  7.8512, "rmse": 10.6974, "mape": 0.8536, "r2": 0.9967, "dir_acc": 54.0541}, "test": {"mae": 11.8248, "rmse": 15.4284, "mape": 0.8832, "r2": 0.9179, "dir_acc": 57.1429}},
    "SBIN.NS":       {"train": {"mae":  6.1699, "rmse":  9.0201, "mape": 1.0461, "r2": 0.9955, "dir_acc": 57.3443}, "test": {"mae": 10.0801, "rmse": 14.4405, "mape": 0.9565, "r2": 0.9686, "dir_acc": 57.1429}},
    "INFY.NS":       {"train": {"mae": 15.4783, "rmse": 20.6008, "mape": 1.0448, "r2": 0.9887, "dir_acc": 52.8790}, "test": {"mae": 17.1201, "rmse": 22.8290, "mape": 1.1631, "r2": 0.9764, "dir_acc": 45.5357}},
    "HINDUNILVR.NS": {"train": {"mae": 20.7555, "rmse": 27.6871, "mape": 0.8837, "r2": 0.9786, "dir_acc": 51.8214}, "test": {"mae": 20.5689, "rmse": 27.5138, "mape": 0.8958, "r2": 0.9447, "dir_acc": 53.5714}},
    "ITC.NS":        {"train": {"mae":  2.7418, "rmse":  3.7205, "mape": 0.8608, "r2": 0.9984, "dir_acc": 54.0541}, "test": {"mae":  2.9017, "rmse":  4.5136, "mape": 0.8648, "r2": 0.9880, "dir_acc": 57.1429}},
    "LT.NS":         {"train": {"mae": 26.1940, "rmse": 38.1777, "mape": 1.0230, "r2": 0.9977, "dir_acc": 55.6992}, "test": {"mae": 45.6394, "rmse": 64.3773, "mape": 1.1759, "r2": 0.9091, "dir_acc": 55.3571}},
    "BAJFINANCE.NS": {"train": {"mae":  7.8004, "rmse": 10.6035, "mape": 1.1403, "r2": 0.9672, "dir_acc": 56.7568}, "test": {"mae": 12.7801, "rmse": 17.8325, "mape": 1.3540, "r2": 0.9213, "dir_acc": 57.1429}},
    "SUNPHARMA.NS":  {"train": {"mae":  9.6992, "rmse": 12.9766, "mape": 0.8547, "r2": 0.9987, "dir_acc": 53.4665}, "test": {"mae": 14.6072, "rmse": 19.1541, "mape": 0.8487, "r2": 0.8785, "dir_acc": 55.3571}},
    "MARUTI.NS":     {"train": {"mae": 91.8227, "rmse": 124.7229,"mape": 0.9821, "r2": 0.9952, "dir_acc": 50.8813}, "test": {"mae": 161.8532,"rmse": 204.5730,"mape": 1.1014, "r2": 0.9769, "dir_acc": 54.4643}},
    "HCLTECH.NS":    {"train": {"mae": 11.8923, "rmse": 16.7713, "mape": 1.0032, "r2": 0.9969, "dir_acc": 52.8790}, "test": {"mae": 15.8617, "rmse": 20.7299, "mape": 1.0405, "r2": 0.9726, "dir_acc": 51.7857}},
    "ADANIENT.NS":   {"train": {"mae": 43.5514, "rmse": 73.9328, "mape": 1.8021, "r2": 0.9879, "dir_acc": 56.6392}, "test": {"mae": 32.7450, "rmse": 44.4993, "mape": 1.5367, "r2": 0.9279, "dir_acc": 60.7143}},
    "TITAN.NS":      {"train": {"mae": 28.6158, "rmse": 38.3783, "mape": 1.0112, "r2": 0.9947, "dir_acc": 54.5241}, "test": {"mae": 37.8986, "rmse": 53.0952, "mape": 0.9272, "r2": 0.9255, "dir_acc": 57.1429}},
    "TATASTEEL.NS":  {"train": {"mae":  1.5076, "rmse":  2.0944, "mape": 1.2866, "r2": 0.9915, "dir_acc": 54.6416}, "test": {"mae":  2.5239, "rmse":  3.4087, "mape": 1.3349, "r2": 0.9447, "dir_acc": 55.3571}},
    "NTPC.NS":       {"train": {"mae":  2.4817, "rmse":  3.9055, "mape": 1.1101, "r2": 0.9984, "dir_acc": 53.3490}, "test": {"mae":  3.2212, "rmse":  4.1627, "mape": 0.9105, "r2": 0.9705, "dir_acc": 49.1071}},
    "ASIANPAINT.NS": {"train": {"mae": 27.2141, "rmse": 36.8342, "mape": 0.9253, "r2": 0.9800, "dir_acc": 55.5817}, "test": {"mae": 30.0681, "rmse": 38.1951, "mape": 1.1889, "r2": 0.9775, "dir_acc": 56.2500}},
    "KOTAKBANK.NS":  {"train": {"mae":  3.5012, "rmse":  4.7980, "mape": 0.9609, "r2": 0.9325, "dir_acc": 53.5840}, "test": {"mae":  3.6852, "rmse":  4.7467, "mape": 0.9125, "r2": 0.9600, "dir_acc": 55.3571}},
    "M&M.NS":        {"train": {"mae": 19.8375, "rmse": 28.6115, "mape": 1.2068, "r2": 0.9985, "dir_acc": 52.8790}, "test": {"mae": 42.0536, "rmse": 55.0165, "mape": 1.2450, "r2": 0.9474, "dir_acc": 51.7857}},
    "ADANIPORTS.NS": {"train": {"mae": 13.3250, "rmse": 21.8323, "mape": 1.4569, "r2": 0.9941, "dir_acc": 52.8790}, "test": {"mae": 18.0786, "rmse": 24.8318, "mape": 1.2459, "r2": 0.8523, "dir_acc": 57.1429}},
    "AXISBANK.NS":   {"train": {"mae":  9.4546, "rmse": 12.7199, "mape": 1.0253, "r2": 0.9948, "dir_acc": 53.5840}, "test": {"mae": 14.2826, "rmse": 19.7581, "mape": 1.1166, "r2": 0.8802, "dir_acc": 51.7857}},
    "ONGC.NS":       {"train": {"mae":  2.2137, "rmse":  3.4589, "mape": 1.3123, "r2": 0.9967, "dir_acc": 52.7615}, "test": {"mae":  2.7053, "rmse":  3.9463, "mape": 1.0624, "r2": 0.9596, "dir_acc": 50.8929}},
    "ULTRACEMCO.NS": {"train": {"mae": 80.6669, "rmse": 109.0625,"mape": 0.9759, "r2": 0.9966, "dir_acc": 54.2891}, "test": {"mae": 127.9596,"rmse": 176.2685,"mape": 1.0885, "r2": 0.9242, "dir_acc": 51.7857}},
    "POWERGRID.NS":  {"train": {"mae":  2.1673, "rmse":  3.2845, "mape": 1.1013, "r2": 0.9979, "dir_acc": 49.8237}, "test": {"mae":  2.4603, "rmse":  3.3041, "mape": 0.8795, "r2": 0.9671, "dir_acc": 57.1429}},
    "COALINDIA.NS":  {"train": {"mae":  3.0867, "rmse":  4.8862, "mape": 1.2622, "r2": 0.9983, "dir_acc": 51.2338}, "test": {"mae":  4.9101, "rmse":  6.8415, "mape": 1.1602, "r2": 0.9488, "dir_acc": 45.5357}},
    "WIPRO.NS":      {"train": {"mae":  2.4825, "rmse":  3.4365, "mape": 1.0689, "r2": 0.9938, "dir_acc": 54.1716}, "test": {"mae":  2.5831, "rmse":  3.4142, "mape": 1.1367, "r2": 0.9798, "dir_acc": 47.3214}},
    "BAJAJFINSV.NS": {"train": {"mae": 17.9131, "rmse": 23.7996, "mape": 1.1323, "r2": 0.9791, "dir_acc": 53.2315}, "test": {"mae": 20.3871, "rmse": 27.6200, "mape": 1.0601, "r2": 0.9544, "dir_acc": 51.7857}},
}

US_AGGREGATE_MODEL_METRICS: dict[str, Any] = {
    "scope": "aggregate_all_stocks", "source": "metrics_overall_summary_csv", "market": "US",
    "rows": [
        {"metric": "MAE",    "train": 2.7991, "test": 4.9133, "difference": round(4.9133 - 2.7991, 4)},
        {"metric": "RMSE",   "train": 3.9341, "test": 6.5755, "difference": round(6.5755 - 3.9341, 4)},
        {"metric": "MAPE",   "train": 1.3476, "test": 1.3793, "difference": round(1.3793 - 1.3476, 4)},
        {"metric": "R2",     "train": 0.9892, "test": 0.9183, "difference": round(0.9183 - 0.9892, 4)},
        {"metric": "DirAcc", "train": 54.0228,"test": 53.8725,"difference": round(53.8725 - 54.0228, 4)},
    ],
}

IN_AGGREGATE_MODEL_METRICS: dict[str, Any] = {
    "scope": "aggregate_all_stocks", "source": "metrics_overall_summary_csv", "market": "IN",
    "rows": [
        {"metric": "MAE",    "train": 17.7906, "test": 25.0435, "difference": round(25.0435 - 17.7906, 4)},
        {"metric": "RMSE",   "train": 24.9577, "test": 33.4691, "difference": round(33.4691 - 24.9577, 4)},
        {"metric": "MAPE",   "train":  1.0684, "test":  1.0612, "difference": round( 1.0612 -  1.0684, 4)},
        {"metric": "R2",     "train":  0.9898, "test":  0.9464, "difference": round( 0.9464 -  0.9898, 4)},
        {"metric": "DirAcc", "train": 53.6732, "test": 53.6638, "difference": round(53.6638 - 53.6732, 4)},
    ],
}


# ── Market helpers ────────────────────────────────────────────────────────────

def _resolve_market(market: str) -> str:
    m = market.strip().upper()
    return "IN" if m in {"IN", "INDIA", "NSE"} else "US"

def _stocks_for(market: str):    return INDIA_STOCKS if market == "IN" else US_STOCKS
def _sym2idx(market: str):       return IN_SYMBOL_TO_IDX if market == "IN" else US_SYMBOL_TO_IDX
def _sym2meta(market: str):      return IN_SYMBOL_TO_META if market == "IN" else US_SYMBOL_TO_META
def _shares(market: str):        return IN_SHARES_OUT_M if market == "IN" else US_SHARES_OUT_M

def _metric_block(raw: dict | None) -> dict:
    raw = raw or {}
    return {k: raw.get(k) for k in ("mae", "rmse", "mape", "r2", "dir_acc", "max_err", "n")}


# ── Business logic ────────────────────────────────────────────────────────────

def get_prediction(symbol: str, market: str) -> dict:
    seq_len  = 30
    sym2idx  = _sym2idx(market)
    sym2meta = _sym2meta(market)

    if symbol not in sym2idx:
        raise HTTPException(404, detail={"error": "SYMBOL_NOT_FOUND", "message": f"Symbol {symbol!r} not in {market} model universe"})

    stock_idx = sym2idx[symbol]
    meta      = sym2meta[symbol]
    data      = _fetch_yf(symbol, "2y")
    hist      = data["hist"]

    if len(hist) < seq_len + 30:
        raise HTTPException(422, detail={"error": "INSUFFICIENT_DATA", "message": "Not enough historical data"})

    reg = ModelRegistry.get(len(US_STOCKS), len(INDIA_STOCKS))

    with MODEL_INFERENCE_LATENCY.labels(market=market).time():
        if market == "IN":
            features = build_features_india(hist)
            pred_ret = predict_sequence_india(features, stock_idx, symbol, reg)
        else:
            features = build_features_us(hist)
            pred_ret = predict_sequence_us(features, stock_idx, reg)

    close_arr = hist["Close"].values.astype(float)
    dates_arr = [d.strftime("%Y-%m-%d") for d in hist.index]

    aligned_close    = close_arr[seq_len:]
    aligned_dates    = dates_arr[seq_len:]
    prev_close_arr   = close_arr[seq_len - 1 : -1]
    predicted_prices = prev_close_arr * np.exp(pred_ret)

    latest_close    = float(close_arr[-1])
    next_day_return = float(pred_ret[-1])
    predicted_next  = round(latest_close * math.exp(next_day_return), 4)
    change_pct      = round((predicted_next - latest_close) / latest_close * 100, 4)

    # Autoregressive 5-day forecast
    forecaster = AutoregressiveForecaster(hist, market, stock_idx, symbol)
    five_day_details = forecaster.forecast(5)

    # Simple iterative fallback (kept for backwards-compat with response field)
    last_close = latest_close
    five_day: list[float] = []
    for _ in range(5):
        last_close = round(last_close * math.exp(next_day_return * 0.85), 4)
        five_day.append(last_close)

    avg_volume = int(hist["Volume"].tail(10).mean())
    shares_m   = _shares(market).get(symbol, 1000)
    market_cap = int(latest_close * shares_m * 1_000_000)
    currency   = "₹" if market == "IN" else "$"

    return {
        "symbol":            symbol,
        "company_name":      meta["name"],
        "sector":            meta["sector"],
        "latest_close":      round(latest_close, 4),
        "predicted_next":    predicted_next,
        "change_pct":        change_pct,
        "52w_high":          market_cap,
        "52w_low":           avg_volume,
        "pe_ratio":          meta["sector"],
        "market_cap":        market_cap,
        "avg_volume":        avg_volume,
        "dates":             aligned_dates,
        "actual_prices":     [round(float(v), 4) for v in aligned_close],
        "predicted_prices":  [round(float(v), 4) for v in predicted_prices],
        "five_day_forecast": five_day,
        "five_day_details":  five_day_details,
        "market":            market,
        "currency":          currency,
    }


def get_metrics(symbol: str, market: str) -> dict:
    seq_len  = 30
    sym2idx  = _sym2idx(market)
    curated  = (CURATED_STOCK_METRICS if market == "US" else INDIA_CURATED_STOCK_METRICS).get(symbol)

    if symbol not in sym2idx:
        raise HTTPException(404, detail={"error": "SYMBOL_NOT_FOUND", "message": f"Symbol {symbol!r} not in {market} model universe"})

    if curated is not None:
        train_m = _metric_block(curated.get("train"))
        test_m  = _metric_block(curated.get("test"))
        train_size, test_size = (851, 112) if market == "IN" else (866, 115)
        return {
            "symbol": symbol, "scope": "stock_specific_curated_metrics",
            "source": "provided_per_stock_table",
            "total_samples": train_size + test_size,
            "train_size": train_size, "test_size": test_size,
            "train": train_m, "test": test_m,
            "metrics_chart": {
                "labels": ["MAE", "RMSE", "MAPE", "R²", "DirAcc"],
                "train":  [train_m["mae"], train_m["rmse"], train_m["mape"], train_m["r2"], train_m["dir_acc"]],
                "test":   [test_m["mae"],  test_m["rmse"],  test_m["mape"],  test_m["r2"],  test_m["dir_acc"]],
            },
        }

    # Compute on-the-fly for uncurated symbols
    stock_idx = sym2idx[symbol]
    data = _fetch_yf(symbol, "2y")
    hist = data["hist"]
    if len(hist) < seq_len + 50:
        raise HTTPException(422, detail={"error": "INSUFFICIENT_DATA", "message": "Not enough data to compute metrics"})

    reg = ModelRegistry.get(len(US_STOCKS), len(INDIA_STOCKS))
    if market == "IN":
        features = build_features_india(hist)
        pred_ret = predict_sequence_india(features, stock_idx, symbol, reg)
    else:
        features = build_features_us(hist)
        pred_ret = predict_sequence_us(features, stock_idx, reg)

    close_arr     = hist["Close"].values.astype(float)
    actual_target = close_arr[seq_len:]
    prev_close    = close_arr[seq_len - 1 : -1]
    pred_prices   = prev_close * np.exp(pred_ret)
    n_samples     = len(actual_target)
    split         = max(1, min(int(n_samples * 0.80), n_samples - 1))

    def _compute(actual, predicted):
        a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
        mae  = float(np.mean(np.abs(a - p)))
        rmse = float(np.sqrt(np.mean((a - p) ** 2)))
        mape = float(np.mean(np.abs((a - p) / (a + 1e-9))) * 100)
        ss_res, ss_tot = float(np.sum((a - p) ** 2)), float(np.sum((a - a.mean()) ** 2))
        r2   = 1.0 - ss_res / (ss_tot + 1e-9)
        dir_acc = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100)
        max_err = float(np.max(np.abs(a - p)))
        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4),
                "r2": round(r2, 4), "dir_acc": round(dir_acc, 2), "max_err": round(max_err, 4), "n": int(len(a))}

    train_m = _compute(actual_target[:split], pred_prices[:split])
    test_m  = _compute(actual_target[split:], pred_prices[split:])
    return {
        "symbol": symbol, "scope": "stock_level_selected_symbol",
        "source": f"computed_from_{market.lower()}_model",
        "total_samples": n_samples, "train_size": split, "test_size": int(n_samples - split),
        "train": train_m, "test": test_m,
        "metrics_chart": {
            "labels": ["MAE", "RMSE", "MAPE", "R2", "DirAcc", "MaxErr"],
            "train": [train_m["mae"], train_m["rmse"], train_m["mape"], train_m["r2"], train_m["dir_acc"], train_m["max_err"]],
            "test":  [test_m["mae"],  test_m["rmse"],  test_m["mape"],  test_m["r2"],  test_m["dir_acc"],  test_m["max_err"]],
        },
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

STARTUP_TIME = time.time()

app = FastAPI(
    title       = "PRISM Stock Intelligence API",
    description = "Multi-Stock LSTM prediction backend — US & India markets",
    version     = VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.on_event("startup")
async def startup_event():
    log.info("Warming up model registry (US + India)…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: ModelRegistry.get(len(US_STOCKS), len(INDIA_STOCKS)))
    log.info("PRISM backend ready on port %s", os.getenv("PORT", "8000"))


# ── Ops endpoints ─────────────────────────────────────────────────────────────

@app.get("/", tags=["ops"])
def root():
    return {"service": "PRISM Stock Intelligence API", "status": "ok", "markets": ["US", "IN"]}


@app.get("/health", tags=["ops"])
@app.get("/api/health", tags=["ops"])
def health():
    """
    Liveness probe.

    Returns HTTP 200 whenever the API process is alive.
    This endpoint is intentionally lightweight:
    - no model inference
    - no data downloads
    - no external API calls
    Safe to call frequently (e.g. every 10 minutes from a cron job).
    """
    return {
        "status":        "ok",
        "service":       "multi-stock-price-prediction",
        "version":       VERSION,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "models_loaded": ModelRegistry._instance is not None and ModelRegistry._instance.all_ok,
    }


@app.get("/healthz", tags=["ops"])
@app.get("/livez", tags=["ops"])
def healthz():
    """Lightweight Kubernetes / cloud ping probe."""
    return {"status": "ok"}


@app.get("/health/detailed", tags=["ops"])
@app.get("/api/health/detailed", tags=["ops"])
def health_detailed():
    """
    Detailed system and diagnostic health check.

    Exposes service uptime, memory usage, database status, and model readiness.
    """
    import sqlite3
    import sys
    from pathlib import Path

    import psutil

    now_utc = datetime.now(timezone.utc)
    uptime_sec = int(time.time() - STARTUP_TIME)

    # Process RAM usage
    memory_mb = 0.0
    try:
        proc = psutil.Process(os.getpid())
        memory_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    # Model readiness
    reg = ModelRegistry._instance
    us_ok = reg is not None and reg.us_ok
    in_ok = reg is not None and reg.in_ok

    # Database connectivity & count
    db_ok = False
    db_records = 0
    db_file = Path("monitoring/predictions.db")
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file))
            row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
            db_records = row[0] if row else 0
            db_ok = True
            conn.close()
        except Exception:
            db_ok = False

    return {
        "status": "ok" if (us_ok and in_ok) else "degraded",
        "service": "multi-stock-price-prediction",
        "version": VERSION,
        "timestamp": now_utc.isoformat(),
        "uptime": {
            "seconds": uptime_sec,
            "human": f"{uptime_sec // 3600}h {(uptime_sec % 3600) // 60}m {uptime_sec % 60}s",
        },
        "system": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "memory_usage_mb": memory_mb,
        },
        "models": {
            "ready": us_ok and in_ok,
            "us_model": "loaded" if us_ok else "unloaded",
            "india_model": "loaded" if in_ok else "unloaded",
        },
        "database": {
            "connected": db_ok,
            "total_predictions": db_records,
        },
    }


@app.get("/ready", tags=["ops"])
def ready():
    """
    Readiness probe.

    Returns HTTP 200 only when both models are loaded and the service
    can serve predictions. Returns 503 otherwise.
    """
    reg    = ModelRegistry._instance
    us_ok  = reg is not None and reg.us_ok
    in_ok  = reg is not None and reg.in_ok
    status = "ready" if (us_ok and in_ok) else "not_ready"
    body   = {"status": status, "models_loaded": us_ok and in_ok, "us_model": us_ok, "in_model": in_ok}
    if not (us_ok and in_ok):
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/metrics", tags=["ops"])
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    output, content_type = get_metrics_output()
    return Response(content=output, media_type=content_type)


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.get("/api/stocks", tags=["data"])
def list_stocks(market: str = Query("US", description="Market: US or IN")):
    """Return the full watchlist used during model training."""
    return _stocks_for(_resolve_market(market))


@app.get("/api/history", tags=["data"])
def history(
    symbol: str = Query(..., description="Ticker symbol"),
    period: str = Query("3mo", description="yfinance period: 1mo | 3mo | 6mo | 1y | 2y | max"),
    market: str = Query("US", description="Market: US or IN"),
):
    """Return raw OHLCV history for chart rendering and signal computation."""
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "max"}
    if period not in valid_periods:
        raise HTTPException(400, detail={"error": "INVALID_PERIOD", "message": f"period must be one of {valid_periods}"})
    symbol = symbol.upper().strip()
    try:
        return get_history(symbol, period)
    except HTTPException:
        raise
    except Exception:
        log.exception("History error for %s", symbol)
        raise HTTPException(500, detail={"error": "HISTORY_ERROR", "message": "Failed to fetch historical data"})


# ── Model endpoints ───────────────────────────────────────────────────────────

@app.get("/api/model/info", tags=["model"])
def model_info(market: str = Query("US", description="Market: US or IN")):
    """Return model architecture, metadata, and version information."""
    m   = _resolve_market(market)
    reg = ModelRegistry.get(len(US_STOCKS), len(INDIA_STOCKS))
    return reg.get_meta(m)


@app.get("/api/predict", tags=["model"])
def predict(
    request: Request,
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL or RELIANCE.NS"),
    market: str = Query("US", description="Market: US or IN"),
):
    """
    Run the LSTM model and return predictions, KPIs, and 5-day forecast.
    """
    m      = _resolve_market(market)
    symbol = symbol.upper().strip()

    _start = time.perf_counter()

    try:
        result = get_prediction(symbol, m)
        _latency_ms = (time.perf_counter() - _start) * 1000
        PREDICTION_REQUESTS.labels(market=m, status="200").inc()
        reg = ModelRegistry._instance
        model_version = reg.get_meta(m).get("model_version", "v1") if reg else "v1"
        log_prediction_event(symbol, m, model_version, _latency_ms, 200)
        store_prediction(symbol, m, model_version, 1, result["predicted_next"])
        return result
    except HTTPException as exc:
        _latency_ms = (time.perf_counter() - _start) * 1000
        PREDICTION_REQUESTS.labels(market=m, status=str(exc.status_code)).inc()
        PREDICTION_ERRORS.labels(market=m).inc()
        log_prediction_event(symbol, m, "v1", _latency_ms, exc.status_code)
        raise
    except Exception:
        PREDICTION_REQUESTS.labels(market=m, status="500").inc()
        PREDICTION_ERRORS.labels(market=m).inc()
        log.exception("Prediction error for %s (%s)", symbol, m)
        raise HTTPException(500, detail={"error": "PREDICTION_ERROR", "message": "Internal prediction error"})


@app.get("/api/model/metrics", tags=["model"])
def model_metrics(
    symbol: str = Query(..., description="Ticker symbol to evaluate"),
    market: str = Query("US", description="Market: US or IN"),
):
    """Return stock-level train/test model metrics for the selected symbol."""
    m      = _resolve_market(market)
    symbol = symbol.upper().strip()
    try:
        return get_metrics(symbol, m)
    except HTTPException:
        raise
    except Exception:
        log.exception("Metrics error for %s (%s)", symbol, m)
        raise HTTPException(500, detail={"error": "METRICS_ERROR", "message": "Failed to compute metrics"})


@app.get("/api/model/aggregate-metrics", tags=["model"])
def model_aggregate_metrics(market: str = Query("US", description="Market: US or IN")):
    """Return overall aggregated model metrics across all stocks."""
    m = _resolve_market(market)
    return US_AGGREGATE_MODEL_METRICS if m == "US" else IN_AGGREGATE_MODEL_METRICS


# ── Monitoring endpoints ──────────────────────────────────────────────────────

@app.get("/api/monitoring/drift", tags=["monitoring"])
def monitoring_drift(
    reference_days: int = Query(30, description="Reference baseline window in days"),
    recent_days: int = Query(7, description="Recent evaluation window in days"),
    market: str | None = Query(None, description="Optional market filter (US or IN)"),
):
    """Return model prediction drift analysis (PSI, KS statistic, quantiles)."""
    from monitoring.drift_report import generate_report

    m = _resolve_market(market) if market else None
    return generate_report(reference_days=reference_days, recent_days=recent_days, market=m)


@app.get("/api/monitoring/performance", tags=["monitoring"])
def monitoring_performance(
    window: int | None = Query(None, description="Rolling window in days (e.g. 7 or 30)"),
):
    """Return rolling accuracy metrics (MAE, RMSE, MAPE, Directional Accuracy) from stored predictions."""
    from monitoring.evaluate_predictions import compute_metrics, load_predictions

    rows = load_predictions(window_days=window)
    res = compute_metrics(rows)
    if window:
        res["window_days"] = window
    return res


@app.post("/api/monitoring/backfill", tags=["monitoring"])
def monitoring_backfill(
    market: str | None = Query(None, description="Optional market filter (US or IN)"),
):
    """Backfill actual closing market prices for pending stored predictions via Yahoo Finance."""
    from monitoring.collect_predictions import backfill

    m = _resolve_market(market) if market else None
    return backfill(market=m, dry_run=False)


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
