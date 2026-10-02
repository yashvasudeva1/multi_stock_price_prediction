"""
predictor.py — Feature engineering, inference, and forecasting.

This module owns all ML logic so FastAPI route handlers remain thin.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from app.model_registry import (
    DEVICE,
    IN_SEQ_LEN,
    US_SEQ_LEN,
    ModelRegistry,
)

log = logging.getLogger("prism.predictor")


# ── Feature engineering — US (26 features) ───────────────────────────────────

def build_features_us(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 26 technical features from an OHLCV DataFrame.
    Returns float32 array of shape (T, 26).  NaNs are interpolated/zero-filled.
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

    ret1  = np.diff(np.log(c + 1e-9), prepend=np.nan)
    ret5  = np.concatenate([[np.nan] * 5,  np.log(c[5:]  / (c[:-5]  + 1e-9))])
    ret10 = np.concatenate([[np.nan] * 10, np.log(c[10:] / (c[:-10] + 1e-9))])
    ret20 = np.concatenate([[np.nan] * 20, np.log(c[20:] / (c[:-20] + 1e-9))])

    ema12, ema26 = ema(c, 12), ema(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig

    bb_mid = rolling_mean(c, 20)
    bb_std = rolling_std(c, 20)
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn + 1e-9)

    delta = np.diff(c, prepend=c[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    rsi   = 100 - 100 / (1 + ema(gain, 14) / (ema(loss, 14) + 1e-9))

    tr  = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr = rolling_mean(tr, 14)

    direction = np.sign(np.diff(c, prepend=c[0]))
    obv       = np.cumsum(v * direction)
    obv_norm  = obv / (np.abs(obv).max() + 1e-9)

    vol_ma20  = rolling_mean(v, 20)
    vol_ratio = v / (vol_ma20 + 1e-9)
    log_vol   = np.log1p(v)

    sma20  = rolling_mean(c, 20)
    sma50  = rolling_mean(c, 50)
    dist20 = (c - sma20) / (sma20 + 1e-9)
    dist50 = (c - sma50) / (sma50 + 1e-9)

    hl_spread = (h - lo) / (c + 1e-9)
    oc_spread = (c - o) / (o + 1e-9)

    lo14  = np.array([lo[max(0, i - 13) : i + 1].min() for i in range(len(lo))])
    hi14  = np.array([h[max(0, i - 13) : i + 1].max() for i in range(len(h))])
    stoch = (c - lo14) / (hi14 - lo14 + 1e-9) * 100

    features = np.column_stack([
        ret1, ret5, ret10, ret20,
        macd_line, macd_sig, macd_hist,
        bb_pct, rsi, atr, obv_norm,
        vol_ratio, log_vol, dist20, dist50,
        hl_spread, oc_spread, stoch,
        c, h, lo, o, v, ema12, ema26, bb_mid,
    ])

    return _fill_nans(features).astype(np.float32)


# ── Feature engineering — India (33 features) ────────────────────────────────

@lru_cache(maxsize=4)
def _fetch_market_benchmarks(period: str = "2y") -> dict:
    """Fetch NASDAQ, S&P500, and USD/INR benchmarks. Cached to avoid re-downloads."""
    log.info("Fetching market benchmarks for India model…")
    ndaq   = yf.Ticker("^IXIC").history(period=period, auto_adjust=True)
    sp500  = yf.Ticker("^GSPC").history(period=period, auto_adjust=True)
    usdinr = yf.Ticker("USDINR=X").history(period=period, auto_adjust=True)

    for df in (ndaq, sp500, usdinr):
        df.index = df.index.tz_localize(None)

    return {
        "ndaq":        ndaq["Close"].values.astype(np.float64),
        "ndaq_dates":  ndaq.index,
        "sp":          sp500["Close"].values.astype(np.float64),
        "sp_dates":    sp500.index,
        "usdinr":      usdinr["Close"].values.astype(np.float64),
        "usdinr_dates": usdinr.index,
    }


def _align_benchmark(bm_vals: np.ndarray, bm_dates, stock_dates) -> np.ndarray:
    bm_series = pd.Series(bm_vals, index=bm_dates)
    aligned   = bm_series.reindex(stock_dates, method="ffill").ffill().bfill()
    return aligned.values.astype(np.float64)


def build_features_india(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 33 technical + market features for the India LSTM model.
    Returns float32 array of shape (T, 33).
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

    lr1  = np.diff(np.log(c + 1e-9), prepend=np.nan)
    lr5  = np.concatenate([[np.nan] * 5,  np.log(c[5:]  / (c[:-5]  + 1e-9))])
    lr21 = np.concatenate([[np.nan] * 21, np.log(c[21:] / (c[:-21] + 1e-9))])

    ma10 = rolling_mean(c, 10)
    ma21 = rolling_mean(c, 21)
    ma63 = rolling_mean(c, 63)
    pma10    = (c - ma10) / (ma10 + 1e-9)
    pma21    = (c - ma21) / (ma21 + 1e-9)
    pma63    = (c - ma63) / (ma63 + 1e-9)
    ma_cross = (ma10 / (ma21 + 1e-9)) - 1.0

    def calc_rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        return 100 - 100 / (1 + ema(gain, period) / (ema(loss, period) + 1e-9))

    rsi14     = calc_rsi(c, 14)
    rsi7      = calc_rsi(c, 7)
    ema12     = ema(c, 12)
    ema26     = ema(c, 26)
    macd_line = ema12 - ema26
    macd_sig  = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig

    daily_ret = np.diff(np.log(c + 1e-9), prepend=np.nan)
    rv5  = rolling_std(daily_ret, 5)
    rv21 = rolling_std(daily_ret, 21)

    bb_mid = rolling_mean(c, 20)
    bb_std = rolling_std(c, 20)
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pct = (c - bb_dn) / (bb_up - bb_dn + 1e-9)
    bb_w   = (bb_up - bb_dn) / (bb_mid + 1e-9)

    tr  = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr = rolling_mean(tr, 14)

    vol_ma20 = rolling_mean(v, 20)
    vol_r    = v / (vol_ma20 + 1e-9)
    vol_lr   = np.diff(np.log(v + 1e-9), prepend=np.nan)

    direction = np.sign(np.diff(c, prepend=c[0]))
    obv       = np.cumsum(v * direction)
    obv_d     = np.diff(obv, prepend=obv[0])
    obv_d     = obv_d / (np.abs(obv_d).max() + 1e-9)

    hl_pct = (h - lo) / (c + 1e-9)
    oc_pct = (c - o)  / (o + 1e-9)

    dates_idx = df.index
    dow = np.array([d.weekday() for d in dates_idx], dtype=np.float64)
    mon = np.array([d.month     for d in dates_idx], dtype=np.float64)
    dow_s = np.sin(2 * np.pi * dow / 5)
    dow_c = np.cos(2 * np.pi * dow / 5)
    mon_s = np.sin(2 * np.pi * mon / 12)
    mon_c = np.cos(2 * np.pi * mon / 12)

    try:
        bm        = _fetch_market_benchmarks("2y")
        ndaq_c    = _align_benchmark(bm["ndaq"],   bm["ndaq_dates"],   dates_idx)
        sp_c      = _align_benchmark(bm["sp"],      bm["sp_dates"],     dates_idx)
        usdinr_c  = _align_benchmark(bm["usdinr"],  bm["usdinr_dates"], dates_idx)
    except Exception as e:
        log.warning("Benchmark fetch failed (%s). Using ones.", e)
        ndaq_c = sp_c = usdinr_c = np.ones(T)

    ndaq_lr1  = np.diff(np.log(ndaq_c + 1e-9), prepend=np.nan)
    ndaq_lr5  = np.concatenate([[np.nan] * 5, np.log(ndaq_c[5:] / (ndaq_c[:-5] + 1e-9))])
    ndaq_ret  = np.diff(np.log(ndaq_c + 1e-9), prepend=np.nan)
    ndaq_rv5  = rolling_std(ndaq_ret, 5)
    sp_lr1    = np.diff(np.log(sp_c + 1e-9), prepend=np.nan)
    sp_lr5    = np.concatenate([[np.nan] * 5, np.log(sp_c[5:] / (sp_c[:-5] + 1e-9))])
    usdinr_lr1   = np.diff(np.log(usdinr_c + 1e-9), prepend=np.nan)
    usdinr_ma21  = rolling_mean(usdinr_c, 21)
    usdinr_pma21 = (usdinr_c - usdinr_ma21) / (usdinr_ma21 + 1e-9)

    features = np.column_stack([
        lr1, lr5, lr21,
        pma10, pma21, pma63, ma_cross,
        rsi14, rsi7,
        macd_line, macd_sig, macd_hist,
        rv5, rv21,
        bb_pct, bb_w, atr,
        vol_r, vol_lr, obv_d,
        hl_pct, oc_pct,
        dow_s, dow_c, mon_s, mon_c,
        ndaq_lr1, ndaq_lr5, ndaq_rv5,
        sp_lr1, sp_lr5,
        usdinr_lr1, usdinr_pma21,
    ])

    return _fill_nans(features).astype(np.float32)


def _fill_nans(features: np.ndarray) -> np.ndarray:
    """Interpolate NaN columns, then zero-fill any remaining."""
    for col in range(features.shape[1]):
        mask = np.isnan(features[:, col])
        idx  = np.where(~mask)[0]
        if len(idx):
            features[:, col] = np.interp(np.arange(len(features[:, col])), idx, features[idx, col])
    return np.nan_to_num(features, nan=0.0)


# ── Sequence inference ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_sequence_us(
    features: np.ndarray,
    stock_idx: int,
    registry: ModelRegistry,
) -> np.ndarray:
    """Slide a 30-day window over the feature array and return next-day log-return predictions."""
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)

    model = registry.us_model
    preds = []
    for start in range(len(scaled) - US_SEQ_LEN):
        window = scaled[start : start + US_SEQ_LEN]
        x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        sid    = torch.tensor([stock_idx], dtype=torch.long).to(DEVICE)
        preds.append(model(x_t, sid).item())

    return np.array(preds, dtype=np.float32)


@torch.no_grad()
def predict_sequence_india(
    features: np.ndarray,
    stock_idx: int,
    symbol: str,
    registry: ModelRegistry,
) -> np.ndarray:
    """Slide a 30-day window over features and return next-day log-return predictions."""
    scaler = registry.get_scaler(symbol)
    if scaler is not None:
        scaled = scaler.transform(features)
    else:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(features)

    model = registry.in_model
    preds = []
    for start in range(len(scaled) - IN_SEQ_LEN):
        window = scaled[start : start + IN_SEQ_LEN]
        x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        sid    = torch.tensor([stock_idx], dtype=torch.long).to(DEVICE)
        preds.append(model(x_t, sid).item())

    return np.array(preds, dtype=np.float32)


# ── Autoregressive 5-day forecaster ──────────────────────────────────────────

def _make_synthetic_bar(predicted_close: float, recent_df: pd.DataFrame, atr_decay: float = 0.85) -> dict:
    prev_close = float(recent_df["Close"].iloc[-1])
    open_price = prev_close

    highs  = recent_df["High"].values[-14:].astype(np.float64)
    lows   = recent_df["Low"].values[-14:].astype(np.float64)
    closes = recent_df["Close"].values[-14:].astype(np.float64)
    tr_arr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))),
    )
    atr_est = float(np.nanmean(tr_arr[1:])) * atr_decay + \
              float(np.abs(predicted_close - prev_close)) * (1 - atr_decay)
    atr_est = max(atr_est, abs(predicted_close - open_price) * 0.1)

    high_price = max(open_price, predicted_close) + atr_est * 0.3
    low_price  = max(min(open_price, predicted_close) - atr_est * 0.3, 0.01)
    avg_vol    = int(recent_df["Volume"].tail(10).mean())

    return {"Open": open_price, "High": high_price, "Low": low_price, "Close": predicted_close, "Volume": avg_vol}


class AutoregressiveForecaster:
    """
    Iteratively predicts N future days, feeding each predicted price back
    as synthetic OHLCV history so all technical indicators update correctly.
    """

    def __init__(self, hist_df: pd.DataFrame, market: str, stock_idx: int, symbol: str) -> None:
        self.buffer    = hist_df.copy()
        self.market    = market
        self.stock_idx = stock_idx
        self.symbol    = symbol
        self.reg       = ModelRegistry.get()
        self.seq_len   = IN_SEQ_LEN if market == "IN" else US_SEQ_LEN

    def _predict_one_step(self) -> float:
        if self.market == "IN":
            features = build_features_india(self.buffer)
            scaler   = self.reg.get_scaler(self.symbol)
            if scaler is not None:
                scaled = scaler.transform(features)
            else:
                scaled = RobustScaler().fit_transform(features)
            model = self.reg.in_model
        else:
            features = build_features_us(self.buffer)
            scaled   = RobustScaler().fit_transform(features)
            model    = self.reg.us_model

        window = scaled[-self.seq_len:]
        x_t    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        sid    = torch.tensor([self.stock_idx], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            return model(x_t, sid).item()

    def _append_bar(self, bar: dict) -> None:
        last_date = self.buffer.index[-1]
        next_date = last_date + pd.tseries.offsets.BDay(1)
        new_row   = pd.DataFrame(bar, index=[next_date])
        new_row.index.name = self.buffer.index.name
        self.buffer = pd.concat([self.buffer, new_row])

    def forecast(self, n_days: int) -> list[dict]:
        """Return list of {price, return, change_pct} for each future day."""
        base_close = float(self.buffer["Close"].iloc[-1])
        results    = []

        for _ in range(n_days):
            pred_ret   = self._predict_one_step()
            prev_close = float(self.buffer["Close"].iloc[-1])
            pred_close = prev_close * math.exp(pred_ret)
            change_pct = (pred_close - base_close) / base_close * 100

            results.append({
                "price":      round(pred_close, 4),
                "return":     round(pred_ret, 6),
                "change_pct": round(change_pct, 4),
            })

            self._append_bar(_make_synthetic_bar(pred_close, self.buffer))

        return results


# ── yfinance data fetch ───────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def _fetch_yf(symbol: str, period: str) -> dict:
    """Cached yfinance download. Returns {"hist": DataFrame}."""
    log.info("yf.download(%s, period=%s)", symbol, period)
    tk   = yf.Ticker(symbol)
    hist = tk.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data returned for {symbol!r}")
    hist.index = hist.index.tz_localize(None)
    return {"hist": hist}


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
