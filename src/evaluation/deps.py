"""Dependency checks for spaced-repetition simulator evaluation."""

from __future__ import annotations

from src.core.exceptions import ConfigurationError


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

