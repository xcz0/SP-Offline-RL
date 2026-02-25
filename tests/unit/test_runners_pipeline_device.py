from __future__ import annotations

import pytest

from src.core.exceptions import ConfigurationError
from src.runners.pipeline import _resolve_device


@pytest.mark.parametrize(
    ("raw_device", "expected"),
    [
        pytest.param(" cpu ", "cpu", id="cpu-with-whitespace"),
        pytest.param("CUDA:0", "cuda:0", id="cuda-normalized"),
    ],
)
def test_resolve_device_valid_cases(raw_device: str, expected: str) -> None:
    assert _resolve_device(raw_device) == expected


@pytest.mark.parametrize(
    ("raw_device", "error_match"),
    [
        pytest.param("auto", "device must be explicit", id="auto-rejected"),
        pytest.param("not-a-device", "Unsupported device", id="invalid-rejected"),
    ],
)
def test_resolve_device_invalid_cases(raw_device: str, error_match: str) -> None:
    with pytest.raises(ConfigurationError, match=error_match):
        _resolve_device(raw_device)
