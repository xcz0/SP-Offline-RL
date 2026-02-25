"""Schema validation and standardization for offline datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from src.core.exceptions import DataValidationError
from src.core.types import DatasetDict
from src.data.pipeline.contracts import BC_DATA_FIELDS, RL_DATA_FIELDS, validate_dataset_fields

REQUIRED_FIELDS = RL_DATA_FIELDS
BC_REQUIRED_FIELDS = BC_DATA_FIELDS

_FIELD_DTYPES: dict[str, np.dtype] = {
    "obs": np.float32,
    "act": np.float32,
    "rew": np.float32,
    "done": np.bool_,
    "obs_next": np.float32,
    "terminated": np.bool_,
    "truncated": np.bool_,
}
_VECTOR_FIELDS: frozenset[str] = frozenset({"rew", "done", "terminated", "truncated"})


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


def _standardize_field(name: str, value: Any) -> np.ndarray:
    dtype = _FIELD_DTYPES.get(name)
    if dtype is None:
        raise DataValidationError(
            f"Field '{name}' is not supported. "
            f"Supported fields: {', '.join(sorted(_FIELD_DTYPES))}."
        )
    out = _as_np(name, value, dtype)
    if name in _VECTOR_FIELDS:
        out = _ensure_vector(name, out)
    if name == "act" and out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def validate_dataset_for_fields(
    data: DatasetDict,
    fields: tuple[str, ...] | list[str],
) -> DatasetDict:
    """Validate and standardize a dataset for an arbitrary canonical field subset."""

    requested = validate_dataset_fields(fields)
    standardized: DatasetDict = {}

    missing = set(requested) - set(data)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise DataValidationError(
            f"Missing required dataset fields: {missing_fields}."
        )

    expected_size: int | None = None
    for field in requested:
        arr = _standardize_field(field, data[field])
        if expected_size is None:
            expected_size = int(arr.shape[0])
        elif int(arr.shape[0]) != expected_size:
            raise DataValidationError(
                f"Field '{field}' first dimension mismatch: expected {expected_size}, "
                f"got {arr.shape[0]}."
            )
        standardized[field] = arr

    if "obs" in standardized and "obs_next" in standardized:
        if standardized["obs"].shape != standardized["obs_next"].shape:
            raise DataValidationError(
                "'obs' and 'obs_next' must have exactly the same shape, "
                f"got {standardized['obs'].shape} vs {standardized['obs_next'].shape}."
            )

    return standardized


def validate_and_standardize_dataset(data: DatasetDict) -> DatasetDict:
    """Validate required fields and return standardized dtypes/shapes."""
    return validate_dataset_for_fields(data, REQUIRED_FIELDS)


def validate_obs_act_dataset(data: DatasetDict) -> DatasetDict:
    """Validate behavior-cloning datasets with only observation/action fields."""
    standardized = validate_dataset_for_fields(data, BC_REQUIRED_FIELDS)
    return {"obs": standardized["obs"], "act": standardized["act"]}


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
