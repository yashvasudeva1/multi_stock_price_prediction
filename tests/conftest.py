"""
Shared pytest fixtures for PRISM tests.

Uses httpx.AsyncClient with the FastAPI test transport so tests run
without a live server or network connection.  yfinance calls are mocked
throughout to keep the CI pipeline deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.model_registry import ModelRegistry
from main import INDIA_STOCKS, US_STOCKS, app

# ── Shared mock data ──────────────────────────────────────────────────────────

def _make_fake_ohlcv(n: int = 600) -> pd.DataFrame:
    """Return a realistic-looking OHLCV DataFrame with n rows."""
    rng   = np.random.default_rng(42)
    close = np.cumprod(1 + rng.normal(0.0002, 0.01, n)) * 150
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(10_000_000, 50_000_000, n)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


FAKE_HIST = _make_fake_ohlcv()


@pytest.fixture(scope="session", autouse=True)
def reset_model_registry():
    """Ensure ModelRegistry singleton is cleared between test sessions."""
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = None


@pytest.fixture(scope="session")
def fake_yf_patch():
    """Patch _fetch_yf so no real network calls are made."""
    fake_result = {"hist": FAKE_HIST}
    with patch("main._fetch_yf", return_value=fake_result) as m1, \
         patch("app.main._fetch_yf", return_value=fake_result) as m2, \
         patch("app.predictor._fetch_yf", return_value=fake_result) as m3:
        yield m1, m2, m3


@pytest_asyncio.fixture(scope="session")
async def client(fake_yf_patch):
    """Shared async test client with mocked yfinance."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c
