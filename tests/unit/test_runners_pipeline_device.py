from __future__ import annotations

import pytest

from src.core.exceptions import ConfigurationError
from src.runners.pipeline import _resolve_device


def test_resolve_device_rejects_auto() -> None:
    with pytest.raises(ConfigurationError, match="device must be explicit"):
        _resolve_device("auto")


def test_resolve_device_rejects_empty_value() -> None:
    with pytest.raises(ConfigurationError, match="device must be explicit"):
        _resolve_device("   ")


def test_resolve_device_accepts_explicit_strings() -> None:
    assert _resolve_device(" cpu ") == "cpu"
    assert _resolve_device("CUDA:0") == "cuda:0"


def test_resolve_device_rejects_invalid_value() -> None:
    with pytest.raises(ConfigurationError, match="Unsupported device"):
        _resolve_device("not-a-device")
