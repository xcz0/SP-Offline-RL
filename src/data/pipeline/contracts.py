"""Data contracts for offline dataset preparation and loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

RL_DATA_FIELDS: tuple[str, ...] = (
    "obs",
    "act",
    "rew",
    "done",
    "obs_next",
    "terminated",
    "truncated",
)
BC_DATA_FIELDS: tuple[str, ...] = ("obs", "act")
SUPPORTED_DATA_FIELDS: tuple[str, ...] = RL_DATA_FIELDS


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Canonical field requirements for an algorithm data path."""

    name: str
    fields: tuple[str, ...]
    validate_env_shapes: bool = True


def validate_dataset_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Validate selected canonical fields."""

    if not fields:
        raise ValueError("At least one dataset field must be requested.")

    normalized = tuple(str(name) for name in fields)
    unknown = sorted(set(normalized) - set(SUPPORTED_DATA_FIELDS))
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(
            f"Unsupported dataset fields: {unknown_text}. "
            f"Supported fields: {', '.join(SUPPORTED_DATA_FIELDS)}."
        )
    return normalized


def make_data_spec(
    *,
    name: str,
    fields: Sequence[str],
    validate_env_shapes: bool = True,
) -> DataSpec:
    """Build a validated immutable data spec."""

    validated = validate_dataset_fields(fields)
    return DataSpec(
        name=str(name),
        fields=validated,
        validate_env_shapes=bool(validate_env_shapes),
    )

