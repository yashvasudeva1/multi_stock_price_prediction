"""
Tests for GET /api/predict
"""

import pytest


@pytest.mark.asyncio
async def test_predict_aapl_200(client):
    """Valid US prediction request should succeed."""
    response = await client.get("/api/predict?symbol=AAPL&market=US")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_response_required_fields(client):
    data = (await client.get("/api/predict?symbol=AAPL&market=US")).json()
    required = {
        "symbol", "company_name", "sector", "latest_close", "predicted_next",
        "change_pct", "market_cap", "avg_volume", "dates",
        "actual_prices", "predicted_prices", "five_day_forecast", "market", "currency",
    }
    for field in required:
        assert field in data, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_predict_numeric_values(client):
    data = (await client.get("/api/predict?symbol=AAPL&market=US")).json()
    assert isinstance(data["latest_close"], (int, float))
    assert isinstance(data["predicted_next"], (int, float))
    assert isinstance(data["change_pct"], (int, float))
    assert len(data["five_day_forecast"]) == 5
    for v in data["five_day_forecast"]:
        assert isinstance(v, (int, float))


@pytest.mark.asyncio
async def test_predict_arrays_same_length(client):
    data = (await client.get("/api/predict?symbol=AAPL&market=US")).json()
    assert len(data["dates"]) == len(data["actual_prices"])
    assert len(data["dates"]) == len(data["predicted_prices"])


@pytest.mark.asyncio
async def test_predict_invalid_symbol_404(client):
    """Unknown ticker should return 404, not 500."""
    response = await client.get("/api/predict?symbol=ZZZZZNOTREAL&market=US")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_predict_missing_symbol_422(client):
    """Omitting the required `symbol` parameter should return 422 (validation error)."""
    response = await client.get("/api/predict?market=US")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_invalid_market_defaults_to_us(client):
    """Unknown market value should fall back to US without crashing."""
    response = await client.get("/api/predict?symbol=AAPL&market=MOON")
    # Should treat MOON as US
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_currency_us(client):
    data = (await client.get("/api/predict?symbol=AAPL&market=US")).json()
    assert data["currency"] == "$"
