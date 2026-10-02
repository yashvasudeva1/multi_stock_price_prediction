"""
Tests for GET /health and GET /ready
"""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_schema(client):
    data = (await client.get("/health")).json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data
    assert "timestamp" in data
    assert isinstance(data["models_loaded"], bool)


@pytest.mark.asyncio
async def test_health_no_inference_side_effects(client):
    """Health endpoint must be safe to call frequently — no model inference."""
    from app.model_registry import ModelRegistry
    initial_instance = ModelRegistry._instance

    await client.get("/health")
    await client.get("/health")
    await client.get("/health")

    # Instance should not change (no new model loads triggered)
    assert ModelRegistry._instance is initial_instance


@pytest.mark.asyncio
async def test_ready_returns_json(client):
    response = await client.get("/ready")
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "us_model" in data
    assert "in_model" in data


@pytest.mark.asyncio
async def test_ready_status_200_or_503(client):
    """Ready must return 200 (models OK) or 503 (not ready) — never 500."""
    response = await client.get("/ready")
    assert response.status_code in (200, 503)
