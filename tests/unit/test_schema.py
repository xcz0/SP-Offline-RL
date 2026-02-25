from __future__ import annotations

import numpy as np
import pytest

from src.core.exceptions import DataValidationError
from src.data.schema import validate_and_standardize_dataset, validate_obs_act_dataset
from tests.factories.dataset_factory import build_obs_act_dataset, build_offline_dataset


def _sample_data(n: int = 8) -> dict[str, np.ndarray]:
    return build_offline_dataset(n=n, seed=41)


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


@pytest.mark.parametrize(
    "missing_fields",
    [
        pytest.param(("obs_next",), id="missing-obs-next"),
        pytest.param(("terminated", "truncated"), id="missing-terminal-flags"),
    ],
)
def test_schema_fail_on_missing_required_fields(
    missing_fields: tuple[str, ...],
) -> None:
    data = _sample_data()
    for field in missing_fields:
        data.pop(field)
    with pytest.raises(DataValidationError):
        validate_and_standardize_dataset(data)


def test_obs_act_schema_valid_without_rewards() -> None:
    n = 10
    data = build_obs_act_dataset(n=n, seed=42)
    data["act"] = data["act"].reshape(n)
    out = validate_obs_act_dataset(data)

    assert set(out.keys()) == {"obs", "act"}
    assert out["obs"].dtype == np.float32
    assert out["act"].dtype == np.float32
    assert out["act"].shape == (n, 1)


def test_obs_act_schema_fail_on_missing_required() -> None:
    data = {"obs": build_obs_act_dataset(n=4, seed=43)["obs"]}
    with pytest.raises(DataValidationError):
        validate_obs_act_dataset(data)
