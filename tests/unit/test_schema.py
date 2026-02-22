from __future__ import annotations

import numpy as np
import pytest

from src.core.exceptions import DataValidationError
from src.data.schema import validate_and_standardize_dataset


def _sample_data(n: int = 8) -> dict[str, np.ndarray]:
    obs = np.random.randn(n, 3).astype(np.float32)
    act = np.random.randn(n, 1).astype(np.float32)
    rew = np.random.randn(n).astype(np.float32)
    done = np.zeros(n, dtype=np.bool_)
    obs_next = np.random.randn(n, 3).astype(np.float32)
    return {
        "obs": obs,
        "act": act,
        "rew": rew,
        "done": done,
        "obs_next": obs_next,
        "terminated": done.copy(),
        "truncated": np.zeros(n, dtype=np.bool_),
    }


def test_schema_valid() -> None:
    data = _sample_data()
    out = validate_and_standardize_dataset(data)
    assert out["obs"].dtype == np.float32
    assert out["act"].dtype == np.float32
    assert out["rew"].dtype == np.float32
    assert out["done"].dtype == np.bool_
    assert out["terminated"].dtype == np.bool_
    assert out["truncated"].dtype == np.bool_
    assert out["obs"].shape[0] == out["rew"].shape[0]


def test_schema_fail_on_length_mismatch() -> None:
    data = _sample_data()
    data["rew"] = data["rew"][:-1]
    with pytest.raises(DataValidationError):
        validate_and_standardize_dataset(data)


def test_schema_fail_on_missing_required() -> None:
    data = _sample_data()
    data.pop("obs_next")
    with pytest.raises(DataValidationError):
        validate_and_standardize_dataset(data)


def test_schema_fail_on_missing_terminated_truncated() -> None:
    data = _sample_data()
    data.pop("terminated")
    data.pop("truncated")
    with pytest.raises(DataValidationError):
        validate_and_standardize_dataset(data)
