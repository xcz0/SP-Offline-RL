"""Dataset adapter interface for offline RL data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from src.core.types import DatasetDict

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


class OfflineDatasetAdapter(ABC):
    """Adapter contract: load selected canonical fields as numpy arrays."""

    @abstractmethod
    def load_prepared(self, fields: Sequence[str]) -> DatasetDict:
        """Load selected canonical fields into numpy arrays."""


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


def build_dataset_adapter(cfg: Any) -> OfflineDatasetAdapter:
    """Build dataset adapter based on config group."""

    adapter_type = str(cfg.adapter).lower()
    if adapter_type == "parquet":
        from src.data.parquet_adapter import ParquetOfflineDatasetAdapter

        raw_columns = dict(cfg.columns)
        columns = {
            str(key): (str(value) if value is not None else "")
            for key, value in raw_columns.items()
        }
        return ParquetOfflineDatasetAdapter(
            path=str(cfg.path),
            columns=columns,
        )
    raise ValueError(
        f"Unsupported data adapter type: {cfg.adapter}. "
        "Only 'parquet' is supported."
    )
