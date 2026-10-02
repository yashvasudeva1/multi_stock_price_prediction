"""
Tests for metric endpoints:
  GET /api/model/metrics
  GET /api/model/aggregate-metrics
"""

import pytest


@pytest.mark.asyncio
async def test_per_stock_metrics_200(client):
    response = await client.get("/api/model/metrics?symbol=AAPL&market=US")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_per_stock_metrics_fields(client):
    data = (await client.get("/api/model/metrics?symbol=AAPL&market=US")).json()
    assert "symbol" in data
    assert "train" in data
    assert "test" in data
    assert "metrics_chart" in data


@pytest.mark.asyncio
async def test_per_stock_metrics_numeric(client):
    data = (await client.get("/api/model/metrics?symbol=AAPL&market=US")).json()
    for split in ("train", "test"):
        m = data[split]
        for key in ("mae", "rmse", "mape", "r2", "dir_acc"):
            assert m[key] is not None, f"{split}.{key} is None"
            assert isinstance(m[key], (int, float))


@pytest.mark.asyncio
async def test_per_stock_metrics_invalid_symbol(client):
    response = await client.get("/api/model/metrics?symbol=FAKESTOCK&market=US")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_aggregate_metrics_us_200(client):
    response = await client.get("/api/model/aggregate-metrics?market=US")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_aggregate_metrics_us_structure(client):
    data = (await client.get("/api/model/aggregate-metrics?market=US")).json()
    assert data["market"] == "US"
    assert isinstance(data["rows"], list)
    assert len(data["rows"]) > 0


@pytest.mark.asyncio
async def test_aggregate_metrics_in_200(client):
    response = await client.get("/api/model/aggregate-metrics?market=IN")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_aggregate_metrics_in_structure(client):
    data = (await client.get("/api/model/aggregate-metrics?market=IN")).json()
    assert data["market"] == "IN"
    assert isinstance(data["rows"], list)


@pytest.mark.asyncio
async def test_metrics_chart_labels_match_values(client):
    data = (await client.get("/api/model/metrics?symbol=AAPL&market=US")).json()
    chart = data["metrics_chart"]
    assert len(chart["labels"]) == len(chart["train"])
    assert len(chart["labels"]) == len(chart["test"])
