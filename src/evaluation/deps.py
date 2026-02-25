"""Dependency checks for spaced-repetition simulator evaluation."""

from __future__ import annotations

from typing import Final

from src.core.exceptions import ConfigurationError

AUTO_DEVICE_SENTINELS: Final[frozenset[str]] = frozenset({"", "auto"})


def require_sprwkv():
    """Import sprwkv symbols or raise a clear configuration error."""

    try:
        from sprwkv import RWKVSrsPredictor, RWKVSrsRlEnv  # type: ignore
        from sprwkv.core.simulator import RWKVSrsSimulator  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "sprwkv is required for simulator-based evaluation but is not available. "
            "Please install project dependencies including 'sprwkv'."
        ) from exc

    return RWKVSrsPredictor, RWKVSrsRlEnv, RWKVSrsSimulator


def resolve_predictor_device(device: str) -> str:
    """Normalize predictor device string to a concrete torch device."""

    raw = str(device).strip()
    normalized = raw.lower()
    if normalized in AUTO_DEVICE_SENTINELS:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        return "cuda:0"
    return raw
