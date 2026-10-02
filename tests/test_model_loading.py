"""
Tests for ModelRegistry — model loading, properties, and metadata.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.model_registry import INDIA_MODEL_PATH, US_MODEL_PATH, ModelRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset singleton before each test in this module."""
    original = ModelRegistry._instance
    ModelRegistry._instance = None
    yield
    ModelRegistry._instance = original


def test_registry_instantiates():
    reg = ModelRegistry.get(30, 29)
    assert reg is not None


def test_registry_singleton():
    reg1 = ModelRegistry.get(30, 29)
    reg2 = ModelRegistry.get(30, 29)
    assert reg1 is reg2


def test_us_model_is_not_none():
    reg = ModelRegistry.get(30, 29)
    assert reg.us_model is not None


def test_in_model_is_not_none():
    reg = ModelRegistry.get(30, 29)
    assert reg.in_model is not None


def test_us_meta_has_required_fields():
    reg  = ModelRegistry.get(30, 29)
    meta = reg.get_meta("US")
    for key in ("model_name", "model_version", "framework", "architecture", "market"):
        assert key in meta, f"Missing key: {key}"


def test_in_meta_has_required_fields():
    reg  = ModelRegistry.get(30, 29)
    meta = reg.get_meta("IN")
    for key in ("model_name", "model_version", "framework", "architecture", "market"):
        assert key in meta, f"Missing key: {key}"


def test_registry_when_model_file_missing():
    """Registry must not crash when model files are absent — uses random weights."""
    with patch.object(Path, "exists", return_value=False):
        ModelRegistry._instance = None
        reg = ModelRegistry.get(30, 29)
        # us_ok and in_ok will be False but models still exist
        assert reg.us_model is not None
        assert reg.in_model is not None
        assert reg.us_ok is False
        assert reg.in_ok is False


def test_get_model_returns_correct_model():
    reg    = ModelRegistry.get(30, 29)
    us_net = reg.get_model("US")
    in_net = reg.get_model("IN")
    assert us_net is reg.us_model
    assert in_net is reg.in_model


def test_all_ok_property():
    reg = ModelRegistry.get(30, 29)
    # The test environment may or may not have model files
    assert isinstance(reg.all_ok, bool)
    assert reg.all_ok == (reg.us_ok and reg.in_ok)
