"""Schema validation and standardization for offline datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from src.core.exceptions import DataValidationError
from src.core.types import DatasetDict

REQUIRED_FIELDS = (
    "obs",
    "act",
    "rew",
    "done",
    "obs_next",
    "terminated",
    "truncated",
)

BC_REQUIRED_FIELDS = (
    "obs",
    "act",
)


def _as_np(name: str, value: Any, dtype: np.dtype) -> np.ndarray:
    try:
        arr = np.asarray(value)
        if dtype == np.bool_:
            arr = arr.astype(np.bool_)
        else:
            arr = arr.astype(dtype)
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(
            f"Field '{name}' cannot be converted to {dtype}."
        ) from exc
    return arr


def _ensure_vector(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        raise DataValidationError(f"Field '{name}' must have at least 1 dimension.")
    if array.ndim > 1:
        if array.shape[-1] == 1:
            array = array.reshape(array.shape[0])
        else:
            raise DataValidationError(
                f"Field '{name}' must be 1D, got shape {array.shape}."
            )
    return array


def validate_and_standardize_dataset(data: DatasetDict) -> DatasetDict:
    """Validate required fields and return standardized dtypes/shapes."""

    standardized: DatasetDict = dict(data)

    missing = set(REQUIRED_FIELDS) - set(standardized)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise DataValidationError(
            f"Missing required dataset fields: {missing_fields}."
        )

    standardized["obs"] = _as_np("obs", standardized["obs"], np.float32)
    standardized["act"] = _as_np("act", standardized["act"], np.float32)
    standardized["rew"] = _ensure_vector(
        "rew", _as_np("rew", standardized["rew"], np.float32)
    )
    standardized["done"] = _ensure_vector(
        "done", _as_np("done", standardized["done"], np.bool_)
    )
    standardized["obs_next"] = _as_np("obs_next", standardized["obs_next"], np.float32)
    standardized["terminated"] = _ensure_vector(
        "terminated", _as_np("terminated", standardized["terminated"], np.bool_)
    )
    standardized["truncated"] = _ensure_vector(
        "truncated", _as_np("truncated", standardized["truncated"], np.bool_)
    )

    if standardized["obs"].shape != standardized["obs_next"].shape:
        raise DataValidationError(
            "'obs' and 'obs_next' must have exactly the same shape, "
            f"got {standardized['obs'].shape} vs {standardized['obs_next'].shape}."
        )

    n = standardized["obs"].shape[0]
    for field in REQUIRED_FIELDS:
        arr = standardized[field]
        if arr.shape[0] != n:
            raise DataValidationError(
                f"Field '{field}' first dimension mismatch: expected {n}, got {arr.shape[0]}."
            )

    if standardized["act"].ndim == 1:
        standardized["act"] = standardized["act"].reshape(-1, 1)

    return standardized


def validate_obs_act_dataset(data: DatasetDict) -> DatasetDict:
    """Validate behavior-cloning datasets with only observation/action fields."""

    standardized: DatasetDict = dict(data)

    missing = set(BC_REQUIRED_FIELDS) - set(standardized)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise DataValidationError(
            f"Missing required dataset fields: {missing_fields}."
        )

    standardized["obs"] = _as_np("obs", standardized["obs"], np.float32)
    standardized["act"] = _as_np("act", standardized["act"], np.float32)

    n = standardized["obs"].shape[0]
    if standardized["act"].shape[0] != n:
        raise DataValidationError(
            f"Field 'act' first dimension mismatch: expected {n}, got {standardized['act'].shape[0]}."
        )

    if standardized["act"].ndim == 1:
        standardized["act"] = standardized["act"].reshape(-1, 1)

    return {
        "obs": standardized["obs"],
        "act": standardized["act"],
    }


def validate_against_env(data: DatasetDict, env: Any) -> None:
    """Validate observation/action dimensions against env spaces."""

    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)

    if isinstance(obs_space, Box):
        expected_obs = tuple(obs_space.shape)
        got_obs = tuple(data["obs"].shape[1:])
        if got_obs != expected_obs:
            raise DataValidationError(
                f"Observation shape mismatch: dataset {got_obs} vs env {expected_obs}."
            )

    if isinstance(act_space, Box):
        expected_act = tuple(act_space.shape)
        got_act = tuple(data["act"].shape[1:])
        if got_act != expected_act:
            raise DataValidationError(
                f"Action shape mismatch: dataset {got_act} vs env {expected_act}."
            )
